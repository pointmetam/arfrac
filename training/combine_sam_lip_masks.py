from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


CLASS_FOLDERS = ("upper_lip", "lower_lip", "inner_mouth")
CLASS_IDS = {"upper_lip": 1, "lower_lip": 2, "inner_mouth": 3}
CLASS_COLORS = {
    1: (236, 36, 82, 220),
    2: (255, 92, 128, 220),
    3: (24, 24, 28, 235),
}


def binary_mask(path: Path, threshold: int = 1) -> np.ndarray:
    array = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    return array >= threshold


def mask_stems(input_dir: Path) -> list[str]:
    stems: set[str] = set()
    for folder in CLASS_FOLDERS:
        class_dir = input_dir / folder
        if class_dir.exists():
            stems.update(path.stem for path in class_dir.glob("*.png"))
    return sorted(stems)


def load_class_masks(input_dir: Path, stem: str, threshold: int = 1) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    shape: tuple[int, int] | None = None
    for folder in CLASS_FOLDERS:
        path = input_dir / folder / f"{stem}.png"
        if not path.exists():
            continue
        mask = binary_mask(path, threshold=threshold)
        if shape is None:
            shape = mask.shape
        elif mask.shape != shape:
            raise ValueError(f"Mask shape mismatch for {stem}: {path} has {mask.shape}, expected {shape}")
        masks[folder] = mask.astype(np.uint8) * 255
    return masks


def combine_class_masks(masks: dict[str, np.ndarray], threshold: int = 1) -> np.ndarray:
    if not masks:
        raise ValueError("No masks to combine")
    first = next(iter(masks.values()))
    label = np.zeros(first.shape, dtype=np.uint8)
    for folder in ("upper_lip", "lower_lip", "inner_mouth"):
        mask = np.asarray(masks.get(folder, np.zeros_like(first)), dtype=np.uint8) >= threshold
        label[mask] = CLASS_IDS[folder]
    return label


def label_to_rgba(label: np.ndarray) -> Image.Image:
    rgba = np.zeros((label.shape[0], label.shape[1], 4), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        rgba[label == class_id] = color
    return Image.fromarray(rgba)


def overlay_label(image: Image.Image, label: np.ndarray) -> Image.Image:
    if image.size != (label.shape[1], label.shape[0]):
        image = image.resize((label.shape[1], label.shape[0]), Image.Resampling.BILINEAR)
    return Image.alpha_composite(image.convert("RGBA"), label_to_rgba(label)).convert("RGB")


def titled(image: Image.Image, title: str) -> Image.Image:
    image = image.convert("RGB")
    output = Image.new("RGB", (image.width, image.height + 24), "white")
    output.paste(image, (0, 24))
    draw = ImageDraw.Draw(output)
    try:
        font = ImageFont.truetype("Arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    draw.text((6, 5), title, fill=(20, 20, 20), font=font)
    return output


def hstack(images: list[Image.Image]) -> Image.Image:
    output = Image.new("RGB", (sum(image.width for image in images), max(image.height for image in images)), "white")
    x = 0
    for image in images:
        output.paste(image, (x, 0))
        x += image.width
    return output


def write_preview(image_path: Path | None, label: np.ndarray, output_path: Path, stem: str) -> None:
    if image_path is not None and image_path.exists():
        image = Image.open(image_path).convert("RGB")
    else:
        image = Image.new("RGB", (label.shape[1], label.shape[0]), (120, 90, 72))
    mask_image = label_to_rgba(label).convert("RGB")
    if mask_image.size != image.size:
        mask_image = mask_image.resize(image.size, Image.Resampling.NEAREST)
    preview = hstack([titled(image, "image"), titled(overlay_label(image, label), "combined label"), titled(mask_image, stem)])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preview.save(output_path, quality=94)


def combine_mask_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    image_dir: str | Path | None = None,
    preview_dir: str | Path | None = None,
    threshold: int = 1,
    preview_count: int = 24,
) -> dict[str, Any]:
    input_dir = Path(input_dir).expanduser()
    output_dir = Path(output_dir)
    image_dir = Path(image_dir).expanduser() if image_dir else None
    preview_dir = Path(preview_dir) if preview_dir else None
    output_dir.mkdir(parents=True, exist_ok=True)
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)

    saved: list[str] = []
    skipped: list[str] = []
    for index, stem in enumerate(mask_stems(input_dir), start=1):
        masks = load_class_masks(input_dir, stem, threshold=threshold)
        if not masks:
            skipped.append(stem)
            continue
        label = combine_class_masks(masks, threshold=threshold)
        output_path = output_dir / f"{stem}.png"
        Image.fromarray(label).save(output_path)
        saved.append(str(output_path))
        if preview_dir is not None and len(saved) <= preview_count:
            image_path = image_dir / f"{stem}.jpg" if image_dir else None
            write_preview(image_path, label, preview_dir / f"preview_{len(saved):03d}_{stem}.jpg", stem)

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "count_saved": len(saved),
        "skipped": skipped,
        "labels": {
            "0": "background",
            "1": "upper_lip",
            "2": "lower_lip",
            "3": "inner_mouth",
        },
        "saved": saved,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine SAM2/CVAT class mask folders into 4-class lip label PNGs.")
    parser.add_argument("--input-dir", required=True, help="Folder containing upper_lip/lower_lip/inner_mouth subfolders")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-dir", help="Optional portrait image folder for previews")
    parser.add_argument("--preview-dir")
    parser.add_argument("--preview-count", type=int, default=24)
    parser.add_argument("--threshold", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = combine_mask_directory(
        args.input_dir,
        args.output_dir,
        image_dir=args.image_dir,
        preview_dir=args.preview_dir,
        threshold=args.threshold,
        preview_count=args.preview_count,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
