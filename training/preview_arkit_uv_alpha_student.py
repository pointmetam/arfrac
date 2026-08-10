from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from arkit_uv_alpha_dataset import ARKitUVAlphaDataset
from dataset import IMAGENET_MEAN, IMAGENET_STD
from evaluate_arkit_uv_student import parse_crop_box
from model import build_model


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    image = (tensor.cpu() * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1)
    array = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(array)


def alpha_to_rgba(alpha: np.ndarray, color: tuple[int, int, int] = (236, 36, 82)) -> Image.Image:
    alpha_u8 = np.clip(alpha * 255, 0, 255).astype(np.uint8)
    rgba = np.zeros((alpha_u8.shape[0], alpha_u8.shape[1], 4), dtype=np.uint8)
    rgba[:, :, 0] = color[0]
    rgba[:, :, 1] = color[1]
    rgba[:, :, 2] = color[2]
    rgba[:, :, 3] = alpha_u8
    return Image.fromarray(rgba)


def overlay_alpha(image: Image.Image, alpha: np.ndarray) -> Image.Image:
    return Image.alpha_composite(image.convert("RGBA"), alpha_to_rgba(alpha)).convert("RGB")


def alpha_to_mask_image(alpha: np.ndarray, color: tuple[int, int, int] = (236, 36, 82)) -> Image.Image:
    alpha = np.clip(alpha, 0, 1).astype(np.float32)
    rgb = np.zeros((alpha.shape[0], alpha.shape[1], 3), dtype=np.uint8)
    rgb[:, :, 0] = np.clip(alpha * color[0], 0, 255).astype(np.uint8)
    rgb[:, :, 1] = np.clip(alpha * color[1], 0, 255).astype(np.uint8)
    rgb[:, :, 2] = np.clip(alpha * color[2], 0, 255).astype(np.uint8)
    return Image.fromarray(rgb)


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


def make_grid(columns: list[Image.Image]) -> Image.Image:
    width = sum(image.width for image in columns)
    height = max(image.height for image in columns)
    output = Image.new("RGB", (width, height), "white")
    offset = 0
    for image in columns:
        output.paste(image, (offset, 0))
        offset += image.width
    return output


def resolve_checkpoint_shape(args: argparse.Namespace, checkpoint: dict) -> tuple[int, int, int, tuple[int, int, int, int] | None, float]:
    checkpoint_args = checkpoint.get("args", {})
    uv_size = int(args.uv_size or checkpoint.get("uv_size") or checkpoint_args.get("uv_size") or 256)
    image_width = int(args.image_width or checkpoint.get("input_width") or checkpoint_args.get("image_width") or uv_size)
    image_height = int(args.image_height or checkpoint.get("input_height") or checkpoint_args.get("image_height") or uv_size)
    crop_box = args.crop_box or checkpoint.get("crop_box") or checkpoint_args.get("crop_box")
    if crop_box is not None:
        crop_box = tuple(int(value) for value in crop_box)
    feather_radius = float(
        args.alpha_feather_radius
        or checkpoint.get("alpha_feather_radius")
        or checkpoint_args.get("alpha_feather_radius")
        or 3.0
    )
    return uv_size, image_width, image_height, crop_box, feather_radius


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/Users/lee/Datasets/ARKitLipUVTeacher")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="training/outputs/previews/arkit_uv_alpha_student")
    parser.add_argument("--split", default="val")
    parser.add_argument("--uv-size", type=int, default=None)
    parser.add_argument("--image-width", type=int, default=None)
    parser.add_argument("--image-height", type=int, default=None)
    parser.add_argument("--crop-box", type=parse_crop_box, default=None)
    parser.add_argument("--alpha-feather-radius", type=float, default=None)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--cache-alpha-labels", action="store_true")
    parser.add_argument("--no-cull-backfaces", action="store_true")
    parser.add_argument("--front-face-winding", choices=["negative", "positive"], default="negative")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    uv_size, image_width, image_height, crop_box, feather_radius = resolve_checkpoint_shape(args, checkpoint)
    model = build_model(
        num_classes=int(checkpoint.get("num_classes", 1)),
        width=int(checkpoint.get("model_width", 24)),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    dataset = ARKitUVAlphaDataset(
        args.root,
        args.split,
        uv_size=uv_size,
        image_width=image_width,
        image_height=image_height,
        crop_box=crop_box,
        feather_radius=feather_radius,
        cache_alpha_labels=args.cache_alpha_labels,
        augment=False,
        cache_uv_images=False,
        cull_backfaces=not args.no_cull_backfaces,
        front_face_winding=args.front_face_winding,
    )
    end = min(args.start + args.count, len(dataset))

    print(f"Device: {device}")
    print(f"UV source: {uv_size}x{uv_size}")
    print(f"Input: {image_width}x{image_height}")
    print(f"Crop box: {crop_box}")
    print(f"Alpha feather radius: {feather_radius}")
    print(f"Writing previews to {output_dir}")

    with torch.no_grad():
        for output_index, dataset_index in enumerate(range(args.start, end), start=1):
            sample = dataset[dataset_index]
            image_tensor = sample["image"]
            target = sample["alpha"].squeeze(0).cpu().numpy()
            logits = model(image_tensor.unsqueeze(0).to(device))
            pred = torch.sigmoid(logits)[0, 0].cpu().numpy()
            image = tensor_to_image(image_tensor)

            columns = [
                titled(image, "uv image"),
                titled(overlay_alpha(image, target), "teacher alpha"),
                titled(overlay_alpha(image, pred), "student alpha"),
                titled(alpha_to_mask_image(target), "target"),
                titled(alpha_to_mask_image(pred), "pred"),
            ]
            grid = make_grid(columns)
            output_path = output_dir / f"preview_{output_index:02d}_{sample['stem']}.jpg"
            grid.save(output_path, quality=94)
            print(output_path)


if __name__ == "__main__":
    main()
