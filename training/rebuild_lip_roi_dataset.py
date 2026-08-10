from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image
from tqdm import tqdm


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


def parse_splits(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def image_by_stem(image_dir: Path) -> dict[str, Path]:
    images: dict[str, Path] = {}
    for extension in IMAGE_EXTENSIONS:
        for path in image_dir.glob(f"*{extension}"):
            images[path.stem] = path
    return images


def load_metadata(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}

    records: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            stem = record.get("stem")
            if stem:
                records[stem] = record
    return records


def resolve_path(path_value: str | None, root: Path) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return (root / path).resolve()


def rebuild_split(
    *,
    source_root: Path,
    output_root: Path,
    split: str,
    width: int,
    height: int,
    workspace_root: Path,
) -> dict[str, int]:
    image_dir = source_root / split / "images"
    label_dir = source_root / split / "labels"
    metadata = load_metadata(source_root / split / "metadata.jsonl")
    if not image_dir.exists() or not label_dir.exists():
        return {"rebuilt": 0, "resized": 0, "skipped": 1}

    output_image_dir = output_root / split / "images"
    output_label_dir = output_root / split / "labels"
    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    images = image_by_stem(image_dir)
    label_paths = sorted(label_dir.glob("*.png"))
    rebuilt = 0
    resized = 0
    skipped = 0

    metadata_path = output_root / split / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        for label_path in tqdm(label_paths, desc=split):
            image_path = images.get(label_path.stem)
            if image_path is None:
                skipped += 1
                continue

            record = metadata.get(label_path.stem, {})
            crop_box = record.get("crop_box")
            source_image_path = resolve_path(record.get("source_image"), workspace_root)
            source_label_path = resolve_path(record.get("source_label"), workspace_root)

            output_image_path = output_image_dir / f"{label_path.stem}.jpg"
            output_label_path = output_label_dir / f"{label_path.stem}.png"

            used_original_crop = (
                isinstance(crop_box, list)
                and len(crop_box) == 4
                and source_image_path is not None
                and source_label_path is not None
                and source_image_path.exists()
                and source_label_path.exists()
            )

            if used_original_crop:
                crop_tuple = tuple(int(value) for value in crop_box)
                image = Image.open(source_image_path).convert("RGB")
                label = Image.open(source_label_path).convert("L")
                image.crop(crop_tuple).resize((width, height), Image.Resampling.BILINEAR).save(
                    output_image_path,
                    quality=96,
                )
                label.crop(crop_tuple).resize((width, height), Image.Resampling.NEAREST).save(output_label_path)
                rebuilt += 1
            else:
                image = Image.open(image_path).convert("RGB")
                label = Image.open(label_path).convert("L")
                image.resize((width, height), Image.Resampling.BILINEAR).save(output_image_path, quality=96)
                label.resize((width, height), Image.Resampling.NEAREST).save(output_label_path)
                resized += 1

            output_record = {
                **record,
                "stem": label_path.stem,
                "source_dataset": str(source_root),
                "resized_from": str(source_root),
                "output_width": width,
                "output_height": height,
                "used_original_crop": used_original_crop,
            }
            metadata_file.write(json.dumps(output_record, ensure_ascii=True) + "\n")

    return {"rebuilt": rebuilt, "resized": resized, "skipped": skipped}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_root = Path(args.root).expanduser()
    output_root = Path(args.output_root).expanduser()
    if args.overwrite and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    labels_source = source_root / "lip_labels.json"
    if labels_source.exists():
        shutil.copy2(labels_source, output_root / "lip_labels.json")
    else:
        fallback_labels = Path(__file__).with_name("lip_labels.json")
        shutil.copy2(fallback_labels, output_root / "lip_labels.json")

    workspace_root = Path.cwd()
    summary = {}
    for split in parse_splits(args.splits):
        summary[split] = rebuild_split(
            source_root=source_root,
            output_root=output_root,
            split=split,
            width=args.width,
            height=args.height,
            workspace_root=workspace_root,
        )

    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Rebuilt lip ROI dataset: {output_root}")
    for split, stats in summary.items():
        print(
            f"{split}: rebuilt={stats['rebuilt']} "
            f"resized={stats['resized']} skipped={stats['skipped']}"
        )


if __name__ == "__main__":
    main()
