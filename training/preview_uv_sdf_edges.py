from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from arkit_uv_alpha_dataset import distance_to_mask
from select_arkit_uv_hard_frames import expanded_bbox_from_mask


DEFAULT_OUTPUT_DIR = Path("training/outputs/uv_sdf_edge_preview")
LIPSTICK = np.array([205, 48, 73], dtype=np.float32)


def smoothstep(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - (2.0 * value))


def lip_sdf_alpha(
    label: np.ndarray,
    feather_radius: float = 3.0,
    shrink_pixels: float = 0.0,
    suppress_inner_mouth: bool = True,
) -> np.ndarray:
    label = np.asarray(label, dtype=np.uint8)
    lip_mask = (label == 1) | (label == 2)
    if not lip_mask.any():
        return np.zeros(label.shape, dtype=np.float32)

    radius = max(float(feather_radius), 0.25)
    distance_to_lip = distance_to_mask(lip_mask)
    distance_to_background = distance_to_mask(~lip_mask)
    signed_distance = distance_to_background - distance_to_lip - float(shrink_pixels)
    alpha = smoothstep((signed_distance + radius) / (radius * 2.0))
    if suppress_inner_mouth:
        alpha[label == 3] = 0.0
    return alpha.astype(np.float32)


def hard_lip_alpha(label: np.ndarray) -> np.ndarray:
    label = np.asarray(label, dtype=np.uint8)
    return (((label == 1) | (label == 2)).astype(np.float32))


def blend_lipstick(image: Image.Image, alpha: np.ndarray, strength: float = 0.70) -> Image.Image:
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    if alpha.shape[:2] != base.shape[:2]:
        alpha_image = Image.fromarray(np.clip(alpha * 255.0, 0, 255).astype(np.uint8)).resize(
            (base.shape[1], base.shape[0]),
            Image.Resampling.BILINEAR,
        )
        alpha = np.asarray(alpha_image, dtype=np.float32) / 255.0
    amount = np.clip(alpha[:, :, None] * strength, 0.0, 1.0)
    # Color blend keeps some original texture; hard mask still shows hard geometry.
    colored = (base * (1.0 - amount)) + (LIPSTICK[None, None, :] * amount)
    return Image.fromarray(np.clip(colored, 0, 255).astype(np.uint8))


def alpha_preview(alpha: np.ndarray, size: tuple[int, int]) -> Image.Image:
    gray = Image.fromarray(np.clip(alpha * 255.0, 0, 255).astype(np.uint8)).convert("RGB")
    if gray.size != size:
        gray = gray.resize(size, Image.Resampling.BILINEAR)
    return gray


def titled(image: Image.Image, title: str) -> Image.Image:
    image = image.convert("RGB")
    title_height = 24
    output = Image.new("RGB", (image.width, image.height + title_height), "white")
    output.paste(image, (0, title_height))
    draw = ImageDraw.Draw(output)
    try:
        font = ImageFont.truetype("Arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    draw.text((6, 5), title, fill=(20, 20, 20), font=font)
    return output


def hstack(images: list[Image.Image]) -> Image.Image:
    width = sum(image.width for image in images)
    height = max(image.height for image in images)
    output = Image.new("RGB", (width, height), "white")
    x = 0
    for image in images:
        output.paste(image, (x, 0))
        x += image.width
    return output


def resize_to_height(image: Image.Image, target_height: int, resample: Image.Resampling) -> Image.Image:
    scale = target_height / float(image.height)
    return image.resize((max(1, int(image.width * scale)), target_height), resample)


def compose_sdf_edge_preview(
    uv_image: Image.Image,
    label: np.ndarray,
    output_path: str | Path | None = None,
    feather_radius: float = 3.0,
    shrink_pixels: float = 1.5,
    crop_padding: int = 24,
    target_height: int = 360,
) -> Image.Image:
    label = np.asarray(label, dtype=np.uint8)
    if uv_image.size != (label.shape[1], label.shape[0]):
        uv_image = uv_image.resize((label.shape[1], label.shape[0]), Image.Resampling.BILINEAR)

    bbox = expanded_bbox_from_mask(label > 0, padding=crop_padding, min_size=(96, 72))
    hard_alpha = hard_lip_alpha(label)
    soft_alpha = lip_sdf_alpha(label, feather_radius=feather_radius, shrink_pixels=0.0)
    shrink_alpha = lip_sdf_alpha(label, feather_radius=feather_radius, shrink_pixels=shrink_pixels)

    panels = [
        titled(
            resize_to_height(blend_lipstick(uv_image, hard_alpha).crop(bbox), target_height, Image.Resampling.NEAREST),
            "hard mask",
        ),
        titled(
            resize_to_height(blend_lipstick(uv_image, soft_alpha).crop(bbox), target_height, Image.Resampling.BILINEAR),
            f"SDF feather {feather_radius:g}px",
        ),
        titled(
            resize_to_height(blend_lipstick(uv_image, shrink_alpha).crop(bbox), target_height, Image.Resampling.BILINEAR),
            f"shrink {shrink_pixels:g}px + SDF",
        ),
        titled(
            resize_to_height(alpha_preview(shrink_alpha, uv_image.size).crop(bbox), target_height, Image.Resampling.BILINEAR),
            "shrink alpha",
        ),
    ]
    preview = hstack(panels)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        preview.save(output_path, quality=95)
    return preview


def uv_image_for_candidate(root: Path, split: str, stem: str, label: np.ndarray, uv_image_dir: str) -> Image.Image:
    uv_image_path = root / split / uv_image_dir / f"{stem}.jpg"
    if uv_image_path.exists():
        image = Image.open(uv_image_path).convert("RGB")
    else:
        image = Image.new("RGB", (label.shape[1], label.shape[0]), (130, 94, 76))
    if image.size != (label.shape[1], label.shape[0]):
        image = image.resize((label.shape[1], label.shape[0]), Image.Resampling.BILINEAR)
    return image


def load_manifest_rows(path: Path, count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
        if len(rows) >= count:
            break
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview hard UV lip masks against SDF soft/shrink lipstick edges.")
    parser.add_argument("--label", help="Single UV label PNG")
    parser.add_argument("--uv-image", help="Optional UV image for --label")
    parser.add_argument("--manifest", help="Candidate JSONL from select_arkit_uv_hard_frames.py")
    parser.add_argument("--root", default="/Users/lee/Datasets/ARKitLipUVTeacherRefined")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--feather-radius", type=float, default=3.0)
    parser.add_argument("--shrink-pixels", type=float, default=1.5)
    parser.add_argument("--crop-padding", type=int, default=24)
    parser.add_argument("--uv-image-dir", default="uv_images_128_culled")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    if args.label:
        label_path = Path(args.label).expanduser()
        label = np.asarray(Image.open(label_path).convert("L"), dtype=np.uint8)
        if args.uv_image:
            uv_image = Image.open(Path(args.uv_image).expanduser()).convert("RGB")
        else:
            uv_image = Image.new("RGB", (label.shape[1], label.shape[0]), (130, 94, 76))
        output_path = output_dir / f"{label_path.stem}_sdf_edges.jpg"
        compose_sdf_edge_preview(
            uv_image,
            label,
            output_path=output_path,
            feather_radius=args.feather_radius,
            shrink_pixels=args.shrink_pixels,
            crop_padding=args.crop_padding,
        )
        print(output_path)
        return

    if not args.manifest:
        raise SystemExit("Provide either --label or --manifest")

    root = Path(args.root).expanduser()
    rows = load_manifest_rows(Path(args.manifest), args.count)
    written: list[str] = []
    for index, row in enumerate(rows, start=1):
        split = str(row["split"])
        stem = str(row["stem"])
        label_path = Path(row.get("uv_label") or (root / split / "uv_labels" / f"{stem}.png")).expanduser()
        label = np.asarray(Image.open(label_path).convert("L"), dtype=np.uint8)
        uv_image = uv_image_for_candidate(root, split, stem, label, args.uv_image_dir)
        output_path = output_dir / f"sdf_{index:03d}_{split}_{stem}.jpg"
        compose_sdf_edge_preview(
            uv_image,
            label,
            output_path=output_path,
            feather_radius=args.feather_radius,
            shrink_pixels=args.shrink_pixels,
            crop_padding=args.crop_padding,
        )
        written.append(str(output_path))

    summary = {
        "count": len(written),
        "output_dir": str(output_dir),
        "feather_radius": args.feather_radius,
        "shrink_pixels": args.shrink_pixels,
        "previews": written,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
