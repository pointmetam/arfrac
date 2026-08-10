from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from prepare_lip_dataset import bbox_from_mask, expanded_aspect_bbox


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
MASK_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
DEFAULT_UPPER_ALIASES = ("u_lip", "upper_lip", "upperlip", "up_lip")
DEFAULT_LOWER_ALIASES = ("l_lip", "lower_lip", "lowerlip", "lo_lip")
DEFAULT_INNER_ALIASES = ("inner_mouth", "mouth", "mouth_opening", "oral")


def parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip().lower() for item in value.split(",") if item.strip())


def numeric_key(stem: str) -> str | None:
    return str(int(stem)) if stem.isdigit() else None


def image_by_key(image_dir: Path) -> dict[str, Path]:
    images: dict[str, Path] = {}
    for extension in IMAGE_EXTENSIONS:
        for path in image_dir.rglob(f"*{extension}"):
            keys = {path.stem}
            numeric = numeric_key(path.stem)
            if numeric is not None:
                keys.add(numeric)
                keys.add(numeric.zfill(5))
            for key in keys:
                images.setdefault(key, path)
    return images


def parse_mask_name(
    path: Path,
    class_aliases: dict[int, tuple[str, ...]],
) -> tuple[str, int] | None:
    stem = path.stem
    lowered = stem.lower()
    aliases: list[tuple[int, str]] = []
    for class_id, names in class_aliases.items():
        aliases.extend((class_id, name) for name in names)
    aliases.sort(key=lambda item: len(item[1]), reverse=True)

    for class_id, alias in aliases:
        for separator in ("_", "-", "."):
            suffix = f"{separator}{alias}"
            if lowered.endswith(suffix):
                return stem[: -len(suffix)], class_id
    return None


def class_masks_by_sample(
    mask_dir: Path,
    class_aliases: dict[int, tuple[str, ...]],
) -> dict[str, dict[int, Path]]:
    samples: dict[str, dict[int, Path]] = {}
    for extension in MASK_EXTENSIONS:
        for path in mask_dir.rglob(f"*{extension}"):
            parsed = parse_mask_name(path, class_aliases)
            if parsed is None:
                continue
            stem, class_id = parsed
            samples.setdefault(stem, {})[class_id] = path
    return samples


def lip_label_from_class_masks(mask_paths: dict[int, Path]) -> np.ndarray:
    opened: dict[int, np.ndarray] = {}
    shape: tuple[int, int] | None = None
    for class_id, path in mask_paths.items():
        mask = np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 0
        opened[class_id] = mask
        shape = mask.shape

    if shape is None:
        raise ValueError("No class masks supplied")

    label = np.zeros(shape, dtype=np.uint8)
    for class_id in (1, 2, 3):
        mask = opened.get(class_id)
        if mask is not None:
            label[mask] = class_id
    return label


def split_for_index(index: int, total: int, split_ratios: tuple[float, float, float]) -> str:
    train_ratio, val_ratio, _ = split_ratios
    train_end = int(round(total * train_ratio))
    val_end = train_end + int(round(total * val_ratio))
    if index < train_end:
        return "train"
    if index < val_end:
        return "val"
    return "test"


def prepare_public_lip_dataset(
    *,
    root: str | Path,
    output_root: str | Path,
    image_dir_name: str = "CelebA-HQ-img",
    mask_dir_name: str = "CelebAMask-HQ-mask-anno",
    source_dataset: str = "public_class_masks",
    width: int = 192,
    height: int = 128,
    padding_x: float = 0.82,
    padding_y: float = 1.25,
    split_ratios: tuple[float, float, float] = (0.9, 0.05, 0.05),
    upper_aliases: tuple[str, ...] = DEFAULT_UPPER_ALIASES,
    lower_aliases: tuple[str, ...] = DEFAULT_LOWER_ALIASES,
    inner_aliases: tuple[str, ...] = DEFAULT_INNER_ALIASES,
    limit: int | None = None,
    overwrite: bool = False,
) -> dict[str, dict[str, int]]:
    source_root = Path(root).expanduser()
    output = Path(output_root).expanduser()
    image_dir = source_root / image_dir_name
    mask_dir = source_root / mask_dir_name
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"Missing mask directory: {mask_dir}")

    if overwrite and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    labels_source = Path(__file__).with_name("lip_labels.json")
    (output / "lip_labels.json").write_text(labels_source.read_text(encoding="utf-8"), encoding="utf-8")

    class_aliases = {
        1: tuple(alias.lower() for alias in upper_aliases),
        2: tuple(alias.lower() for alias in lower_aliases),
        3: tuple(alias.lower() for alias in inner_aliases),
    }
    images = image_by_key(image_dir)
    masks_by_stem = class_masks_by_sample(mask_dir, class_aliases)
    sample_stems = sorted(
        stem
        for stem, masks in masks_by_stem.items()
        if 1 in masks or 2 in masks
    )
    if limit is not None:
        sample_stems = sample_stems[:limit]

    summary = {
        "train": {"kept": 0, "skipped": 0},
        "val": {"kept": 0, "skipped": 0},
        "test": {"kept": 0, "skipped": 0},
    }
    metadata_files = {}
    try:
        for split in summary:
            (output / split / "images").mkdir(parents=True, exist_ok=True)
            (output / split / "labels").mkdir(parents=True, exist_ok=True)
            metadata_files[split] = (output / split / "metadata.jsonl").open("w", encoding="utf-8")

        for index, stem in enumerate(tqdm(sample_stems, desc=source_dataset)):
            split = split_for_index(index, len(sample_stems), split_ratios)
            lookup_keys = {stem}
            numeric = numeric_key(stem)
            if numeric is not None:
                lookup_keys.add(numeric)
                lookup_keys.add(numeric.zfill(5))
            image_path = next((images[key] for key in lookup_keys if key in images), None)
            if image_path is None:
                summary[split]["skipped"] += 1
                continue

            image = Image.open(image_path).convert("RGB")
            lip_label = lip_label_from_class_masks(masks_by_stem[stem])
            bbox = bbox_from_mask(lip_label > 0)
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
            image_crop = image.crop(crop_box).resize((width, height), Image.BILINEAR)
            label_crop = Image.fromarray(lip_label).crop(crop_box).resize((width, height), Image.NEAREST)

            output_stem = stem
            image_crop.save(output / split / "images" / f"{output_stem}.jpg", quality=96)
            label_crop.save(output / split / "labels" / f"{output_stem}.png")
            metadata_files[split].write(
                json.dumps(
                    {
                        "stem": output_stem,
                        "source_dataset": source_dataset,
                        "source_stem": stem,
                        "source_image": str(image_path),
                        "source_masks": {str(class_id): str(path) for class_id, path in masks_by_stem[stem].items()},
                        "crop_box": crop_box,
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )
            summary[split]["kept"] += 1
    finally:
        for file in metadata_files.values():
            file.close()

    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/Users/lee/Datasets/CelebAMask-HQ")
    parser.add_argument("--output-root", default="/Users/lee/Datasets/CelebAMaskHQLip")
    parser.add_argument("--image-dir", default="CelebA-HQ-img")
    parser.add_argument("--mask-dir", default="CelebAMask-HQ-mask-anno")
    parser.add_argument("--source-dataset", default="celebamask_hq")
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--padding-x", type=float, default=0.82)
    parser.add_argument("--padding-y", type=float, default=1.25)
    parser.add_argument("--split-ratios", default="0.9,0.05,0.05")
    parser.add_argument("--upper-aliases", default=",".join(DEFAULT_UPPER_ALIASES))
    parser.add_argument("--lower-aliases", default=",".join(DEFAULT_LOWER_ALIASES))
    parser.add_argument("--inner-aliases", default=",".join(DEFAULT_INNER_ALIASES))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    ratios = tuple(float(item.strip()) for item in args.split_ratios.split(",") if item.strip())
    if len(ratios) != 3:
        raise ValueError("--split-ratios must contain train,val,test, e.g. 0.9,0.05,0.05")

    summary = prepare_public_lip_dataset(
        root=args.root,
        output_root=args.output_root,
        image_dir_name=args.image_dir,
        mask_dir_name=args.mask_dir,
        source_dataset=args.source_dataset,
        width=args.width,
        height=args.height,
        padding_x=args.padding_x,
        padding_y=args.padding_y,
        split_ratios=ratios,  # type: ignore[arg-type]
        upper_aliases=parse_csv(args.upper_aliases),
        lower_aliases=parse_csv(args.lower_aliases),
        inner_aliases=parse_csv(args.inner_aliases),
        limit=args.limit,
        overwrite=args.overwrite,
    )
    print(f"Lip ROI dataset: {Path(args.output_root).expanduser()}")
    for split, stats in summary.items():
        print(f"{split}: kept={stats['kept']} skipped={stats['skipped']}")


if __name__ == "__main__":
    main()
