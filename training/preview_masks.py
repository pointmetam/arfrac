from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from dataset import IMAGENET_MEAN, IMAGENET_STD, LaPaDataset
from model import build_model


LIP_CLASSES = {5, 6}
INNER_MOUTH_CLASS = 7


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    image = (tensor.cpu() * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1)
    array = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def label_to_mask_image(label: np.ndarray) -> Image.Image:
    rgba = np.zeros((label.shape[0], label.shape[1], 4), dtype=np.uint8)
    lip = np.isin(label, list(LIP_CLASSES))
    mouth = label == INNER_MOUTH_CLASS
    rgba[lip] = [226, 42, 86, 210]
    rgba[mouth] = [24, 24, 28, 230]
    return Image.fromarray(rgba, mode="RGBA")


def overlay_mask(image: Image.Image, label: np.ndarray) -> Image.Image:
    overlay = label_to_mask_image(label)
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def titled(image: Image.Image, title: str) -> Image.Image:
    title_height = 28
    output = Image.new("RGB", (image.width, image.height + title_height), "white")
    output.paste(image, (0, title_height))
    draw = ImageDraw.Draw(output)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    draw.text((8, 7), title, fill=(30, 30, 30), font=font)
    return output


def make_grid(columns: list[Image.Image]) -> Image.Image:
    width = sum(image.width for image in columns)
    height = max(image.height for image in columns)
    output = Image.new("RGB", (width, height), "white")
    offset = 0
    for image in columns:
        output.paste(image, (offset, 0))
        offset += image.width
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/Users/lee/Datasets/LaPa")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="training/outputs/previews")
    parser.add_argument("--split", default="val")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = build_model(num_classes=checkpoint.get("num_classes", 9))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    dataset = LaPaDataset(args.root, args.split, image_size=args.image_size, augment=False)
    end = min(args.start + args.count, len(dataset))
    indices = range(args.start, end)

    print(f"Device: {device}")
    print(f"Writing previews to {output_dir}")

    with torch.no_grad():
        for output_index, dataset_index in enumerate(indices, start=1):
            sample = dataset[dataset_index]
            image_tensor = sample["image"]
            label_tensor = sample["label"]
            logits = model(image_tensor.unsqueeze(0).to(device))
            prediction = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
            label = label_tensor.cpu().numpy().astype(np.uint8)
            image = tensor_to_image(image_tensor)

            columns = [
                titled(image, "image"),
                titled(overlay_mask(image, label), "ground truth"),
                titled(overlay_mask(image, prediction), "prediction"),
                titled(label_to_mask_image(prediction).convert("RGB"), "pred mask"),
            ]
            grid = make_grid(columns)
            output_path = output_dir / f"preview_{output_index:02d}_{Path(sample['image_path']).stem}.jpg"
            grid.save(output_path, quality=94)
            print(output_path)


if __name__ == "__main__":
    main()
