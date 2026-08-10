from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
LAPA_UPPER_LIP = 7
LAPA_INNER_MOUTH = 8
LAPA_LOWER_LIP = 9
LIP_CLASSES = {LAPA_UPPER_LIP, LAPA_INNER_MOUTH, LAPA_LOWER_LIP}


def parse_splits(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def image_by_stem(image_dir: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for extension in IMAGE_EXTENSIONS:
        for path in image_dir.glob(f"*{extension}"):
            result[path.stem] = path
    return result


def lip_label_from_lapa(label: np.ndarray) -> np.ndarray:
    output = np.zeros(label.shape, dtype=np.uint8)
    output[label == LAPA_UPPER_LIP] = 1
    output[label == LAPA_LOWER_LIP] = 2
    output[label == LAPA_INNER_MOUTH] = 3
    return output


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def expanded_aspect_bbox(
    bbox: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    target_width: int,
    target_height: int,
    padding_x: float,
    padding_y: float,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = bbox
    width = max(right - left, 1)
    height = max(bottom - top, 1)
    center_x = (left + right) * 0.5
    center_y = (top + bottom) * 0.5

    crop_width = width * (1.0 + padding_x * 2.0)
    crop_height = height * (1.0 + padding_y * 2.0)
    target_aspect = target_width / target_height

    if crop_width / crop_height < target_aspect:
        crop_width = crop_height * target_aspect
    else:
        crop_height = crop_width / target_aspect

    crop_width = min(crop_width, image_width)
    crop_height = min(crop_height, image_height)

    left = center_x - crop_width * 0.5
    top = center_y - crop_height * 0.5
    right = left + crop_width
    bottom = top + crop_height

    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > image_width:
        left -= right - image_width
        right = image_width
    if bottom > image_height:
        top -= bottom - image_height
        bottom = image_height

    left = max(0, int(round(left)))
    top = max(0, int(round(top)))
    right = min(image_width, int(round(right)))
    bottom = min(image_height, int(round(bottom)))
    return left, top, right, bottom


def prepare_split(
    source_root: Path,
    output_root: Path,
    split: str,
    width: int,
    height: int,
    padding_x: float,
    padding_y: float,
    limit: int | None,
) -> dict[str, int]:
    split_root = source_root / split
    image_dir = split_root / "images"
    label_dir = split_root / "labels"
    if not image_dir.exists() or not label_dir.exists():
        raise FileNotFoundError(f"Missing LaPa folders under {split_root}")

    output_image_dir = output_root / split / "images"
    output_label_dir = output_root / split / "labels"
    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    images = image_by_stem(image_dir)
    label_paths = sorted(label_dir.glob("*.png"))
    if limit is not None:
        label_paths = label_paths[:limit]

    kept = 0
    skipped = 0
    metadata_path = output_root / split / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        for label_path in tqdm(label_paths, desc=split):
            image_path = images.get(label_path.stem)
            if image_path is None:
                skipped += 1
                continue

            image = Image.open(image_path).convert("RGB")
            label = Image.open(label_path).convert("L")
            label_array = np.asarray(label, dtype=np.uint8)
            lip_region = np.isin(label_array, list(LIP_CLASSES))
            bbox = bbox_from_mask(lip_region)
            if bbox is None:
                skipped += 1
                continue

            crop_box = expanded_aspect_bbox(
                bbox,
                image.width,
                image.height,
                width,
                height,
                padding_x,
                padding_y,
            )

            image_crop = image.crop(crop_box).resize((width, height), Image.BILINEAR)
            lip_label = lip_label_from_lapa(label_array)
            label_crop = Image.fromarray(lip_label).crop(crop_box).resize((width, height), Image.NEAREST)

            output_stem = label_path.stem
            image_crop.save(output_image_dir / f"{output_stem}.jpg", quality=96)
            label_crop.save(output_label_dir / f"{output_stem}.png")
            metadata_file.write(
                json.dumps(
                    {
                        "stem": output_stem,
                        "source_image": str(image_path),
                        "source_label": str(label_path),
                        "crop_box": crop_box,
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )
            kept += 1

    return {"kept": kept, "skipped": skipped}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/Users/lee/Datasets/LaPa")
    parser.add_argument("--output-root", default="/Users/lee/Datasets/LaPaLip")
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--padding-x", type=float, default=0.82)
    parser.add_argument("--padding-y", type=float, default=1.25)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    source_root = Path(args.root).expanduser()
    output_root = Path(args.output_root).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    labels_source = Path(__file__).with_name("lip_labels.json")
    labels_target = output_root / "lip_labels.json"
    labels_target.write_text(labels_source.read_text(encoding="utf-8"), encoding="utf-8")

    summary: dict[str, dict[str, int]] = {}
    for split in parse_splits(args.splits):
        summary[split] = prepare_split(
            source_root=source_root,
            output_root=output_root,
            split=split,
            width=args.width,
            height=args.height,
            padding_x=args.padding_x,
            padding_y=args.padding_y,
            limit=args.limit,
        )

    print(f"Lip ROI dataset: {output_root}")
    for split, stats in summary.items():
        print(f"{split}: kept={stats['kept']} skipped={stats['skipped']}")


if __name__ == "__main__":
    main()
