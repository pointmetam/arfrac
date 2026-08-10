from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from dataset import IMAGENET_MEAN, IMAGENET_STD
from lip_contour_dataset import (
    LipContourDataset,
    catmull_rom_closed,
    denormalize_points,
    mask_iou,
    points_to_mask,
)
from lip_contour_model import build_lip_contour_model
from train_lip_contour_student import choose_device


TARGET_COLOR = (0, 188, 255)
PRED_COLOR = (38, 255, 70)
FILL_COLOR = np.array([38, 255, 70], dtype=np.float32)


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    image = (tensor.cpu() * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1)
    array = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(array)


def draw_curve(
    image: Image.Image,
    points: np.ndarray,
    color: tuple[int, int, int],
    *,
    width: int = 2,
    samples_per_segment: int = 10,
) -> Image.Image:
    output = image.convert("RGB")
    draw = ImageDraw.Draw(output)
    curve = catmull_rom_closed(points, samples_per_segment=samples_per_segment)
    if curve.shape[0] >= 2:
        xy = [(float(x), float(y)) for x, y in curve]
        draw.line(xy + [xy[0]], fill=color, width=width, joint="curve")
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
    parser.add_argument("--root", default="training/outputs/datasets/lip_segface512_plus_lip001_manual_v2_clean2_256x192")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="training/outputs/previews/lip_contour_student")
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
    point_count = int(checkpoint.get("point_count", 64))
    model_width = int(checkpoint.get("model_width", 32))

    model = build_lip_contour_model(point_count=point_count, width=model_width)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    dataset = LipContourDataset(
        args.root,
        args.split,
        image_width=image_width,
        image_height=image_height,
        point_count=point_count,
        augment=False,
    )
    end = min(args.start + args.count, len(dataset))

    print(f"Device: {device}")
    print(f"Input: {image_width}x{image_height}")
    print(f"Point count: {point_count}")
    print(f"Writing previews to {output_dir}")

    with torch.no_grad():
        for output_index, dataset_index in enumerate(range(args.start, end), start=1):
            sample = dataset[dataset_index]
            image_tensor = sample["image"]
            image = tensor_to_image(image_tensor)
            target_points = denormalize_points(sample["points"].numpy(), image_width, image_height)

            prediction = model(image_tensor.unsqueeze(0).to(device))[0].cpu().numpy().astype(np.float32)
            pred_points = denormalize_points(prediction, image_width, image_height)
            pred_mask = points_to_mask(pred_points, image_width, image_height, smooth=True)
            target_mask = sample["lip_mask"].squeeze(0).numpy().astype(np.uint8)
            iou = mask_iou(pred_mask, target_mask)

            target_curve = draw_curve(image, target_points, TARGET_COLOR, width=2)
            pred_curve = draw_curve(image, pred_points, PRED_COLOR, width=2)
            both_curves = draw_curve(draw_curve(image, target_points, TARGET_COLOR, width=2), pred_points, PRED_COLOR, width=2)
            pred_fill = draw_curve(blend_fill(image, pred_mask), pred_points, PRED_COLOR, width=2)

            grid = hstack(
                [
                    titled(image, "image"),
                    titled(target_curve, "target contour"),
                    titled(pred_curve, "pred contour"),
                    titled(both_curves, f"target cyan / pred green iou={iou:.3f}"),
                    titled(pred_fill, "pred smooth fill"),
                ]
            )
            output_path = output_dir / f"preview_{output_index:02d}_{Path(sample['image_path']).stem}.jpg"
            grid.save(output_path, quality=94)
            print(output_path)


if __name__ == "__main__":
    main()
