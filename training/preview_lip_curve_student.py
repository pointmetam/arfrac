from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from dataset import IMAGENET_MEAN, IMAGENET_STD
from lip_curve_dataset import (
    CURVE_NAMES,
    LipCurveDataset,
    catmull_rom_open,
    curves_to_mask,
    denormalize_curves,
    mask_iou,
)
from lip_curve_model import build_lip_curve_model
from train_lip_curve_student import choose_device


TARGET_COLORS = [(0, 188, 255), (40, 155, 255), (255, 198, 0), (255, 138, 0)]
PRED_COLORS = [(38, 255, 70), (0, 220, 80), (0, 255, 210), (0, 210, 255)]
FILL_COLOR = np.array([38, 255, 70], dtype=np.float32)


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    image = (tensor.cpu() * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1)
    array = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(array)


def draw_curves(image: Image.Image, curves: np.ndarray, colors: list[tuple[int, int, int]], *, width: int = 2) -> Image.Image:
    output = image.convert("RGB")
    draw = ImageDraw.Draw(output)
    for curve, color in zip(curves, colors):
        smooth = catmull_rom_open(curve, samples_per_segment=10)
        if smooth.shape[0] >= 2:
            draw.line([(float(x), float(y)) for x, y in smooth], fill=color, width=width, joint="curve")
    return output


def blend_fill(image: Image.Image, mask: np.ndarray, strength: float = 0.82) -> Image.Image:
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    amount = np.clip(mask[:, :, None].astype(np.float32) * strength, 0.0, 1.0)
    output = (base * (1.0 - amount)) + (FILL_COLOR[None, None, :] * amount)
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
    parser.add_argument("--root", default="training/outputs/datasets/lip001_video_interp_fixed_v2_clean2_roi_384x256_step2")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="training/outputs/previews/lip_curve_student")
    parser.add_argument("--split", default="test")
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    image_width = int(checkpoint.get("input_width", 384))
    image_height = int(checkpoint.get("input_height", 256))
    points_per_curve = int(checkpoint.get("points_per_curve", 24))
    model_width = int(checkpoint.get("model_width", 32))

    model = build_lip_curve_model(points_per_curve=points_per_curve, width=model_width)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    dataset = LipCurveDataset(args.root, args.split, image_width, image_height, points_per_curve, augment=False)
    end = min(args.start + args.count, len(dataset))

    print(f"Device: {device}")
    print(f"Input: {image_width}x{image_height}")
    print(f"Curves: {', '.join(CURVE_NAMES)}")
    print(f"Writing previews to {output_dir}")

    with torch.no_grad():
        for output_index, dataset_index in enumerate(range(args.start, end), start=1):
            sample = dataset[dataset_index]
            image_tensor = sample["image"]
            image = tensor_to_image(image_tensor)
            target_curves = denormalize_curves(sample["curves"].numpy(), image_width, image_height)
            prediction = model(image_tensor.unsqueeze(0).to(device))[0].cpu().numpy().astype(np.float32)
            pred_curves = denormalize_curves(prediction, image_width, image_height)
            pred_mask = curves_to_mask(pred_curves, image_width, image_height, smooth=True)
            target_mask = sample["lip_mask"].squeeze(0).numpy().astype(np.uint8)
            iou = mask_iou(pred_mask, target_mask)

            target_panel = draw_curves(image, target_curves, TARGET_COLORS, width=2)
            pred_panel = draw_curves(image, pred_curves, PRED_COLORS, width=2)
            both_panel = draw_curves(draw_curves(image, target_curves, TARGET_COLORS, width=2), pred_curves, PRED_COLORS, width=2)
            fill_panel = draw_curves(blend_fill(image, pred_mask), pred_curves, PRED_COLORS, width=2)

            grid = hstack(
                [
                    titled(image, "image"),
                    titled(target_panel, "target 4 curves"),
                    titled(pred_panel, "pred 4 curves"),
                    titled(both_panel, f"target/pred iou={iou:.3f}"),
                    titled(fill_panel, "curve raster fill"),
                ]
            )
            output_path = output_dir / f"preview_{output_index:02d}_{Path(sample['image_path']).stem}.jpg"
            grid.save(output_path, quality=94)
            print(output_path)


if __name__ == "__main__":
    main()
