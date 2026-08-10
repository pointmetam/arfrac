from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable


DEFAULT_SOURCE_ROOT = Path("/Users/lee/Datasets/ARKitLipUVTeacher")
DEFAULT_OUTPUT_ROOT = Path("/Users/lee/Datasets/ARKitLipUVTeacherRefined")
DEFAULT_SPLITS = ("train", "val", "test")


def count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def count_files(path: Path, pattern: str) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.glob(pattern))


def ensure_split_exists(source_root: Path, split: str) -> None:
    split_root = source_root / split
    required = ("images", "uv_labels", "frames", "metadata.jsonl")
    missing = [name for name in required if not (split_root / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing {split} entries under {split_root}: {', '.join(missing)}")


def prepare_refined_dataset(
    source_root: str | Path = DEFAULT_SOURCE_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    splits: Iterable[str] = DEFAULT_SPLITS,
) -> dict:
    source_root = Path(source_root).expanduser()
    output_root = Path(output_root).expanduser()
    split_names = tuple(str(split) for split in splits)

    if not source_root.exists():
        raise FileNotFoundError(f"Source dataset does not exist: {source_root}")
    if output_root.exists():
        raise FileExistsError(f"Refined dataset already exists: {output_root}")

    for split in split_names:
        ensure_split_exists(source_root, split)

    output_root.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir()

    split_summaries: dict[str, dict[str, int]] = {}
    try:
        for split in split_names:
            source_split = source_root / split
            output_split = output_root / split
            shutil.copytree(source_split, output_split)
            split_summaries[split] = {
                "metadata_rows": count_jsonl_rows(output_split / "metadata.jsonl"),
                "images": count_files(output_split / "images", "*.jpg"),
                "uv_labels": count_files(output_split / "uv_labels", "*.png"),
                "frames": count_files(output_split / "frames", "*.json"),
            }

        source_summary = source_root / "summary.json"
        if source_summary.exists():
            shutil.copy2(source_summary, output_root / "source_summary.json")

        summary = {
            "dataset": "ARKitLipUVTeacherRefined",
            "source_root": str(source_root),
            "output_root": str(output_root),
            "splits": split_summaries,
            "labels": {
                "0": "background_or_non_lip_uv",
                "1": "upper_lip",
                "2": "lower_lip",
                "3": "inner_mouth",
            },
            "notes": [
                "This dataset is a copy intended for manual or semi-automatic UV label refinement.",
                "Original teacher labels are preserved in the source dataset.",
            ],
        }
        (output_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        return summary
    except Exception:
        if output_root.exists():
            shutil.rmtree(output_root)
        raise


def parse_splits(value: str) -> tuple[str, ...]:
    splits = tuple(part.strip() for part in value.split(",") if part.strip())
    if not splits:
        raise argparse.ArgumentTypeError("--splits must contain at least one split")
    return splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy ARKit UV teacher data into a safe refined-label dataset.")
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--splits", type=parse_splits, default=DEFAULT_SPLITS, help="Comma-separated split names")
    args = parser.parse_args()

    summary = prepare_refined_dataset(args.source_root, args.output_root, args.splits)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
