from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


def image_by_stem(image_dir: Path) -> dict[str, Path]:
    images: dict[str, Path] = {}
    for extension in IMAGE_EXTENSIONS:
        for path in image_dir.glob(f"*{extension}"):
            images[path.stem] = path
    return images


def parse_sources(value: str) -> dict[str, Path]:
    sources: dict[str, Path] = {}
    for item in value.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError("Sources must be name=/path pairs separated by commas")
        name, path = item.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError("Source name cannot be empty")
        sources[name] = Path(path.strip()).expanduser()
    if not sources:
        raise ValueError("At least one source is required")
    return sources


def combine_lip_datasets(
    *,
    sources: dict[str, str | Path],
    output_root: str | Path,
    splits: list[str],
    overwrite: bool = False,
) -> dict[str, dict[str, int]]:
    output = Path(output_root).expanduser()
    if overwrite and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    labels_source = Path(__file__).with_name("lip_labels.json")
    (output / "lip_labels.json").write_text(labels_source.read_text(encoding="utf-8"), encoding="utf-8")

    summary = {split: {"kept": 0, "skipped": 0} for split in splits}
    for split in splits:
        output_image_dir = output / split / "images"
        output_label_dir = output / split / "labels"
        output_image_dir.mkdir(parents=True, exist_ok=True)
        output_label_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = output / split / "metadata.jsonl"
        with metadata_path.open("w", encoding="utf-8") as metadata_file:
            for source_name, source_root_value in sources.items():
                source_root = Path(source_root_value).expanduser()
                image_dir = source_root / split / "images"
                label_dir = source_root / split / "labels"
                if not image_dir.exists() or not label_dir.exists():
                    summary[split]["skipped"] += 1
                    continue

                images = image_by_stem(image_dir)
                for label_path in sorted(label_dir.glob("*.png")):
                    image_path = images.get(label_path.stem)
                    if image_path is None:
                        summary[split]["skipped"] += 1
                        continue

                    output_stem = f"{source_name}_{label_path.stem}"
                    shutil.copy2(image_path, output_image_dir / f"{output_stem}{image_path.suffix.lower()}")
                    shutil.copy2(label_path, output_label_dir / f"{output_stem}.png")
                    metadata_file.write(
                        json.dumps(
                            {
                                "stem": output_stem,
                                "source_name": source_name,
                                "source_image": str(image_path),
                                "source_label": str(label_path),
                            },
                            ensure_ascii=True,
                        )
                        + "\n"
                    )
                    summary[split]["kept"] += 1

    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sources",
        required=True,
        help="Comma-separated name=/path pairs, e.g. lapa=/Users/lee/Datasets/LaPaLip,celeb=/Users/lee/Datasets/CelebAMaskHQLip",
    )
    parser.add_argument("--output-root", default="/Users/lee/Datasets/LipMixedPublic")
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    summary = combine_lip_datasets(
        sources=parse_sources(args.sources),
        output_root=args.output_root,
        splits=[item.strip() for item in args.splits.split(",") if item.strip()],
        overwrite=args.overwrite,
    )
    print(f"Combined lip dataset: {Path(args.output_root).expanduser()}")
    for split, stats in summary.items():
        print(f"{split}: kept={stats['kept']} skipped={stats['skipped']}")


if __name__ == "__main__":
    main()
