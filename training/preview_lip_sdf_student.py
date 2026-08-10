from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from dataset import IMAGENET_MEAN, IMAGENET_STD
from lip_sdf_dataset import LipSDFDataset
from lip_sdf_postprocess import alpha_to_smooth_mask, mask_iou
from model import build_model
from train_lip_sdf_student import choose_device


GREEN = np.array([38, 255, 70], dtype=np.float32)


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    image = (tensor.cpu() * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1)
    array = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(array)


def gray_image(values: np.ndarray) -> Image.Image:
    array = np.clip(values * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(array).convert("RGB")


def sdf_image(values: np.ndarray) -> Image.Image:
    # -1 outside is dark, +1 inside is light.
    array = np.clip((values * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(array).convert("RGB")


def blend_fill(image: Image.Image, alpha: np.ndarray, strength: float = 0.86) -> Image.Image:
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    amount = np.clip(alpha[:, :, None].astype(np.float32) * strength, 0.0, 1.0)
    output = (base * (1.0 - amount)) + (GREEN[None, None, :] * amount)
    return Image.fromarray(np.clip(output, 0, 255).astype(np.uint8))


def draw_mask_outline(image: Image.Image, mask: np.ndarray, color: tuple[int, int, int] = (20, 255, 50)) -> Image.Image:
    output = image.convert("RGB")
    draw = ImageDraw.Draw(output)
    mask_image = Image.fromarray((mask.astype(np.uint8) * 255)).convert("L")
    # PIL's contour filter is not available everywhere; use alpha edges via FIND_EDGES fallback.
    try:
        from PIL import ImageFilter

        edge = mask_image.filter(ImageFilter.FIND_EDGES)
        ys, xs = np.where(np.asarray(edge) > 0)
        for x, y in zip(xs, ys):
            draw.point((int(x), int(y)), fill=color)
    except Exception:
        pass
    return output


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
    parser.add_argument("--root", default="training/outputs/datasets/lip_segface512_plus_lip001_manual_v2_clean2_384x256")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="training/outputs/previews/lip_sdf_student")
    parser.add_argument("--split", default="test")
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--smooth-point-count", type=int, default=128)
    parser.add_argument("--close-kernel", type=int, default=5)
    parser.add_argument("--open-kernel", type=int, default=3)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    image_width = int(checkpoint.get("input_width", 384))
    image_height = int(checkpoint.get("input_height", 256))
    threshold = float(args.threshold if args.threshold is not None else checkpoint.get("alpha_threshold", 0.5))

    model = build_model(num_classes=int(checkpoint.get("num_classes", 3)), width=int(checkpoint.get("model_width", 32)))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    dataset = LipSDFDataset(
        args.root,
        args.split,
        image_width=image_width,
        image_height=image_height,
        feather_radius=float(checkpoint.get("feather_radius", 2.5)),
        sdf_radius=float(checkpoint.get("sdf_radius", 16.0)),
        boundary_radius=float(checkpoint.get("boundary_radius", 2.0)),
        shrink_pixels=float(checkpoint.get("shrink_pixels", 0.0)),
        augment=False,
    )
    end = min(args.start + args.count, len(dataset))

    print(f"Device: {device}")
    print(f"Input: {image_width}x{image_height}")
    print(f"Threshold: {threshold}")
    print(f"Writing previews to {output_dir}")

    with torch.no_grad():
        for output_index, dataset_index in enumerate(range(args.start, end), start=1):
            sample = dataset[dataset_index]
            image_tensor = sample["image"]
            image = tensor_to_image(image_tensor)
            target_hard = sample["hard_lip"].squeeze(0).numpy().astype(np.uint8)

            logits = model(image_tensor.unsqueeze(0).to(device))[0].cpu()
            pred_alpha = torch.sigmoid(logits[0]).numpy().astype(np.float32)
            pred_sdf = torch.tanh(logits[1]).numpy().astype(np.float32)
            pred_edge = torch.sigmoid(logits[2]).numpy().astype(np.float32)

            raw_mask = (pred_alpha >= threshold).astype(np.uint8)
            smooth_mask = alpha_to_smooth_mask(
                pred_alpha,
                threshold=threshold,
                point_count=args.smooth_point_count,
                close_kernel=args.close_kernel,
                open_kernel=args.open_kernel,
            )
            raw_iou = mask_iou(raw_mask, target_hard)
            smooth_iou = mask_iou(smooth_mask, target_hard)

            panels = [
                titled(image, "image"),
                titled(gray_image(sample["alpha"].squeeze(0).numpy()), "target alpha"),
                titled(gray_image(pred_alpha), "pred alpha"),
                titled(sdf_image(pred_sdf), "pred sdf"),
                titled(gray_image(pred_edge), "pred edge"),
                titled(draw_mask_outline(blend_fill(image, raw_mask), raw_mask), f"raw mask iou={raw_iou:.3f}"),
                titled(draw_mask_outline(blend_fill(image, smooth_mask), smooth_mask), f"smooth mask iou={smooth_iou:.3f}"),
            ]
            grid = hstack(panels)
            output_path = output_dir / f"preview_{output_index:02d}_{Path(sample['image_path']).stem}.jpg"
            grid.save(output_path, quality=94)
            print(output_path)


if __name__ == "__main__":
    main()
