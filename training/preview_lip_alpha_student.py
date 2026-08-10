from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from dataset import IMAGENET_MEAN, IMAGENET_STD
from lip_alpha_dataset import LipAlphaDataset
from model import build_model
from train_lip_alpha_student import choose_device


LIPSTICK = np.array([205, 48, 73], dtype=np.float32)


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    image = (tensor.cpu() * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1)
    array = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(array)


def alpha_image(alpha: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(alpha * 255.0, 0, 255).astype(np.uint8)).convert("RGB")


def blend_lipstick(image: Image.Image, alpha: np.ndarray, strength: float = 0.72) -> Image.Image:
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    amount = np.clip(alpha[:, :, None] * strength, 0.0, 1.0)
    output = (base * (1.0 - amount)) + (LIPSTICK[None, None, :] * amount)
    return Image.fromarray(np.clip(output, 0, 255).astype(np.uint8))


def titled(image: Image.Image, title: str) -> Image.Image:
    title_height = 28
    output = Image.new("RGB", (image.width, image.height + title_height), "white")
    output.paste(image.convert("RGB"), (0, title_height))
    draw = ImageDraw.Draw(output)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    draw.text((8, 7), title, fill=(30, 30, 30), font=font)
    return output


def hstack(images: list[Image.Image]) -> Image.Image:
    width = sum(image.width for image in images)
    height = max(image.height for image in images)
    output = Image.new("RGB", (width, height), "white")
    offset = 0
    for image in images:
        output.paste(image, (offset, 0))
        offset += image.width
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="training/outputs/datasets/lip_segface512_plus_lip001_manual_v2_clean2_256x192")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="training/outputs/previews/lip_alpha_student")
    parser.add_argument("--split", default="test")
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    image_width = int(checkpoint.get("input_width", 256))
    image_height = int(checkpoint.get("input_height", 192))
    feather_radius = float(checkpoint.get("alpha_feather_radius", 3.0))
    shrink_pixels = float(checkpoint.get("alpha_shrink_pixels", 0.0))

    model = build_model(
        num_classes=int(checkpoint.get("num_classes", 1)),
        width=int(checkpoint.get("model_width", 32)),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    dataset = LipAlphaDataset(
        args.root,
        args.split,
        image_width=image_width,
        image_height=image_height,
        feather_radius=feather_radius,
        shrink_pixels=shrink_pixels,
        augment=False,
    )
    end = min(args.start + args.count, len(dataset))

    print(f"Device: {device}")
    print(f"Writing previews to {output_dir}")
    with torch.no_grad():
        for output_index, dataset_index in enumerate(range(args.start, end), start=1):
            sample = dataset[dataset_index]
            image_tensor = sample["image"]
            target = sample["alpha"].squeeze(0).cpu().numpy().astype(np.float32)
            logits = model(image_tensor.unsqueeze(0).to(device))
            prediction = torch.sigmoid(logits)[0, 0].cpu().numpy().astype(np.float32)
            image = tensor_to_image(image_tensor)

            grid = hstack(
                [
                    titled(image, "image"),
                    titled(alpha_image(target), "target alpha"),
                    titled(alpha_image(prediction), "pred alpha"),
                    titled(blend_lipstick(image, target), "target lipstick"),
                    titled(blend_lipstick(image, prediction), "pred lipstick"),
                ]
            )
            output_path = output_dir / f"preview_{output_index:02d}_{Path(sample['image_path']).stem}.jpg"
            grid.save(output_path, quality=94)
            print(output_path)


if __name__ == "__main__":
    main()
