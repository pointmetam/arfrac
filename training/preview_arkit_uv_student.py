from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from arkit_uv_student_dataset import ARKitUVTeacherDataset
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


def label_to_mask_image(label: np.ndarray) -> Image.Image:
    rgba = np.zeros((label.shape[0], label.shape[1], 4), dtype=np.uint8)
    rgba[label == 1] = [236, 36, 82, 220]
    rgba[label == 2] = [255, 92, 128, 220]
    rgba[label == 3] = [24, 24, 28, 235]
    return Image.fromarray(rgba)


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


def resolve_checkpoint_shape(args: argparse.Namespace, checkpoint: dict) -> tuple[int, int, int, tuple[int, int, int, int] | None]:
    checkpoint_args = checkpoint.get("args", {})
    uv_size = int(args.uv_size or checkpoint.get("uv_size") or checkpoint_args.get("uv_size") or 256)
    image_width = int(args.image_width or checkpoint.get("input_width") or checkpoint_args.get("image_width") or uv_size)
    image_height = int(args.image_height or checkpoint.get("input_height") or checkpoint_args.get("image_height") or uv_size)
    crop_box = args.crop_box or checkpoint.get("crop_box") or checkpoint_args.get("crop_box")
    if crop_box is not None:
        crop_box = tuple(int(value) for value in crop_box)
    return uv_size, image_width, image_height, crop_box


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/Users/lee/Datasets/ARKitLipUVTeacher")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="training/outputs/previews/arkit_uv_student")
    parser.add_argument("--split", default="val")
    parser.add_argument("--uv-size", type=int, default=None)
    parser.add_argument("--image-width", type=int, default=None)
    parser.add_argument("--image-height", type=int, default=None)
    parser.add_argument("--crop-box", type=parse_crop_box, default=None)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--no-cull-backfaces", action="store_true")
    parser.add_argument("--front-face-winding", choices=["negative", "positive"], default="negative")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    uv_size, image_width, image_height, crop_box = resolve_checkpoint_shape(args, checkpoint)
    model = build_model(
        num_classes=int(checkpoint.get("num_classes", 4)),
        width=int(checkpoint.get("model_width", 16)),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    dataset = ARKitUVTeacherDataset(
        args.root,
        args.split,
        uv_size=uv_size,
        image_width=image_width,
        image_height=image_height,
        crop_box=crop_box,
        augment=False,
        cache_uv_images=False,
        cull_backfaces=not args.no_cull_backfaces,
        front_face_winding=args.front_face_winding,
    )
    end = min(args.start + args.count, len(dataset))
    indices = range(args.start, end)

    print(f"Device: {device}")
    print(f"UV source: {uv_size}x{uv_size}")
    print(f"Input: {image_width}x{image_height}")
    print(f"Crop box: {crop_box}")
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
                titled(image, "uv image"),
                titled(overlay_mask(image, label), "teacher label"),
                titled(overlay_mask(image, prediction), "student pred"),
                titled(label_to_mask_image(prediction).convert("RGB"), "pred mask"),
            ]
            grid = make_grid(columns)
            output_path = output_dir / f"preview_{output_index:02d}_{sample['stem']}.jpg"
            grid.save(output_path, quality=94)
            print(output_path)


if __name__ == "__main__":
    main()
