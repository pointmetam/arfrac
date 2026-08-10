from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from prepare_lip_dataset import bbox_from_mask, expanded_aspect_bbox


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".webp")
IBUG_UPPER_LIP = 7
IBUG_INNER_MOUTH = 8
IBUG_LOWER_LIP = 9
IBUG_LIP_CLASSES = {IBUG_UPPER_LIP, IBUG_INNER_MOUTH, IBUG_LOWER_LIP}


def image_by_stem(split_dir: Path) -> dict[str, Path]:
    images: dict[str, Path] = {}
    for extension in IMAGE_EXTENSIONS:
        for path in split_dir.glob(f"*{extension}"):
            images[path.stem] = path
    return images


def lip_label_from_ibugmask(label: np.ndarray) -> np.ndarray:
    output = np.zeros(label.shape, dtype=np.uint8)
    output[label == IBUG_UPPER_LIP] = 1
    output[label == IBUG_LOWER_LIP] = 2
    output[label == IBUG_INNER_MOUTH] = 3
    return output


def target_split_for_train_index(index: int, total: int, val_ratio: float) -> str:
    train_count = max(1, int(round(total * (1.0 - val_ratio))))
    if index < train_count:
        return "train"
    return "val"


def prepare_source_split(
    *,
    source_split_dir: Path,
    output_root: Path,
    source_split: str,
    target_split: str | None,
    width: int,
    height: int,
    padding_x: float,
    padding_y: float,
    val_ratio: float,
    limit: int | None,
    summary: dict[str, dict[str, int]],
) -> None:
    if not source_split_dir.exists():
        raise FileNotFoundError(source_split_dir)

    images = image_by_stem(source_split_dir)
    label_paths = [path for path in sorted(source_split_dir.glob("*.png")) if path.stem in images]
    if limit is not None:
        label_paths = label_paths[:limit]

    metadata_files: dict[str, object] = {}
    try:
        for label_index, label_path in enumerate(tqdm(label_paths, desc=source_split)):
            split = target_split
            if split is None:
                split = target_split_for_train_index(label_index, len(label_paths), val_ratio)

            output_image_dir = output_root / split / "images"
            output_label_dir = output_root / split / "labels"
            output_image_dir.mkdir(parents=True, exist_ok=True)
            output_label_dir.mkdir(parents=True, exist_ok=True)

            if split not in metadata_files:
                metadata_files[split] = (output_root / split / "metadata.jsonl").open("w", encoding="utf-8")
            metadata_file = metadata_files[split]

            image_path = images[label_path.stem]
            image = Image.open(image_path).convert("RGB")
            source_label = np.asarray(Image.open(label_path).convert("L"), dtype=np.uint8)
            lip_region = np.isin(source_label, list(IBUG_LIP_CLASSES))
            bbox = bbox_from_mask(lip_region)
            if bbox is None:
                summary[split]["skipped"] += 1
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
            image_crop = image.crop(crop_box).resize((width, height), Image.Resampling.BILINEAR)
            lip_label = lip_label_from_ibugmask(source_label)
            label_crop = Image.fromarray(lip_label).crop(crop_box).resize((width, height), Image.Resampling.NEAREST)

            output_stem = label_path.stem
            image_crop.save(output_image_dir / f"{output_stem}.jpg", quality=96)
            label_crop.save(output_label_dir / f"{output_stem}.png")
            metadata_file.write(
                json.dumps(
                    {
                        "stem": output_stem,
                        "source_dataset": "ibugmask",
                        "source_split": source_split,
                        "source_stem": label_path.stem,
                        "source_image": str(image_path),
                        "source_label": str(label_path),
                        "crop_box": crop_box,
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )
            summary[split]["kept"] += 1
    finally:
        for metadata_file in metadata_files.values():
            metadata_file.close()


def prepare_ibugmask_lip_dataset(
    *,
    root: str | Path,
    output_root: str | Path,
    width: int = 192,
    height: int = 128,
    padding_x: float = 0.82,
    padding_y: float = 1.25,
    val_ratio: float = 0.05,
    limit: int | None = None,
    overwrite: bool = False,
) -> dict[str, dict[str, int]]:
    source_root = Path(root).expanduser()
    output = Path(output_root).expanduser()
    if overwrite and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    labels_source = Path(__file__).with_name("lip_labels.json")
    (output / "lip_labels.json").write_text(labels_source.read_text(encoding="utf-8"), encoding="utf-8")

    summary = {
        "train": {"kept": 0, "skipped": 0},
        "val": {"kept": 0, "skipped": 0},
        "test": {"kept": 0, "skipped": 0},
    }
    prepare_source_split(
        source_split_dir=source_root / "train",
        output_root=output,
        source_split="train",
        target_split=None,
        width=width,
        height=height,
        padding_x=padding_x,
        padding_y=padding_y,
        val_ratio=val_ratio,
        limit=limit,
        summary=summary,
    )
    prepare_source_split(
        source_split_dir=source_root / "test",
        output_root=output,
        source_split="test",
        target_split="test",
        width=width,
        height=height,
        padding_x=padding_x,
        padding_y=padding_y,
        val_ratio=val_ratio,
        limit=limit,
        summary=summary,
    )

    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/Users/lee/Datasets/ibugmask_release")
    parser.add_argument("--output-root", default="/Users/lee/Datasets/iBugMaskLip")
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--padding-x", type=float, default=0.82)
    parser.add_argument("--padding-y", type=float, default=1.25)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    summary = prepare_ibugmask_lip_dataset(
        root=args.root,
        output_root=args.output_root,
        width=args.width,
        height=args.height,
        padding_x=args.padding_x,
        padding_y=args.padding_y,
        val_ratio=args.val_ratio,
        limit=args.limit,
        overwrite=args.overwrite,
    )
    print(f"Lip ROI dataset: {Path(args.output_root).expanduser()}")
    for split, stats in summary.items():
        print(f"{split}: kept={stats['kept']} skipped={stats['skipped']}")


if __name__ == "__main__":
    main()
