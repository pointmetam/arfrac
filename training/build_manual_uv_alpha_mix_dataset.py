from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def parse_stems(value: str | None) -> set[str]:
    if not value:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def resolve_manual_split(root: Path, split: str) -> Path:
    if (root / "metadata.jsonl").exists():
        return root
    candidate = root / split
    if (candidate / "metadata.jsonl").exists():
        return candidate
    raise FileNotFoundError(f"Cannot find manual metadata.jsonl under {root} or {candidate}")


def row_matches_stem(row: dict[str, Any], stems: set[str]) -> bool:
    stem = str(row["stem"])
    sample_id = str(row.get("sample_id", ""))
    return stem in stems or sample_id in stems or any(stem.endswith(item) for item in stems)


def link_file(source: Path, target: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        existing = target.resolve(strict=False)
        if existing == source.resolve(strict=True):
            return
        raise FileExistsError(f"{target} already exists and points to {existing}")
    os.symlink(source.resolve(strict=True), target)


def link_split_files(source_split: Path, output_split: Path, stem: str) -> None:
    for folder, suffix in (("images", ".jpg"), ("uv_labels", ".png"), ("frames", ".json")):
        link_file(source_split / folder / f"{stem}{suffix}", output_split / folder / f"{stem}{suffix}")


def copy_base_split(base_root: Path, output_root: Path, split: str) -> list[dict[str, Any]]:
    source_split = base_root / split
    output_split = output_root / split
    rows = read_jsonl(source_split / "metadata.jsonl")
    output_rows: list[dict[str, Any]] = []
    for row in rows:
        stem = str(row["stem"])
        link_split_files(source_split, output_split, stem)
        payload = dict(row)
        payload["split"] = split
        payload["manual_pixel_label"] = False
        output_rows.append(payload)
    write_jsonl(output_split / "metadata.jsonl", output_rows)
    return output_rows


def add_manual_rows(
    manual_split: Path,
    output_root: Path,
    split: str,
    rows: list[dict[str, Any]],
    repeat: int,
) -> list[dict[str, Any]]:
    output_split = output_root / split
    output_rows: list[dict[str, Any]] = []
    for row in rows:
        stem = str(row["stem"])
        link_split_files(manual_split, output_split, stem)
        payload = dict(row)
        payload["split"] = split
        payload["manual_pixel_label"] = True
        payload["source_manual_split"] = str(manual_split)
        for _ in range(max(1, repeat)):
            output_rows.append(dict(payload))
    return output_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-root", required=True)
    parser.add_argument("--manual-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--manual-split", default="test")
    parser.add_argument("--manual-repeat", type=int, default=20)
    parser.add_argument("--holdout-stems", default="")
    parser.add_argument("--train-stems", default="")
    parser.add_argument("--test-stems", default="")
    args = parser.parse_args()

    base_root = Path(args.base_root).expanduser().resolve()
    manual_split = resolve_manual_split(Path(args.manual_root).expanduser().resolve(), args.manual_split)
    output_root = Path(args.output_root).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output root is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    train_rows = copy_base_split(base_root, output_root, "train")
    val_rows = copy_base_split(base_root, output_root, "val")

    manual_rows = read_jsonl(manual_split / "metadata.jsonl")
    holdout_stems = parse_stems(args.holdout_stems)
    explicit_train_stems = parse_stems(args.train_stems)
    explicit_test_stems = parse_stems(args.test_stems)

    if explicit_train_stems:
        manual_train = [row for row in manual_rows if row_matches_stem(row, explicit_train_stems)]
    else:
        manual_train = [row for row in manual_rows if not row_matches_stem(row, holdout_stems)]

    if explicit_test_stems:
        manual_test = [row for row in manual_rows if row_matches_stem(row, explicit_test_stems)]
    else:
        manual_test = [row for row in manual_rows if row_matches_stem(row, holdout_stems)]

    if not manual_train:
        raise RuntimeError("No manual train rows selected")
    if not manual_test:
        raise RuntimeError("No manual test rows selected")

    manual_train_rows = add_manual_rows(manual_split, output_root, "train", manual_train, args.manual_repeat)
    train_rows.extend(manual_train_rows)
    write_jsonl(output_root / "train" / "metadata.jsonl", train_rows)

    test_rows = add_manual_rows(manual_split, output_root, "test", manual_test, 1)
    write_jsonl(output_root / "test" / "metadata.jsonl", test_rows)

    summary = {
        "base_root": str(base_root),
        "manual_split": str(manual_split),
        "output_root": str(output_root),
        "manual_repeat": args.manual_repeat,
        "holdout_stems": sorted(holdout_stems),
        "counts": {
            "train_base_rows": len(train_rows) - len(manual_train_rows),
            "train_manual_unique": len(manual_train),
            "train_manual_weighted_rows": len(manual_train_rows),
            "train_total_rows": len(train_rows),
            "val_rows": len(val_rows),
            "test_manual_rows": len(test_rows),
        },
        "train_manual_stems": [str(row["stem"]) for row in manual_train],
        "test_manual_stems": [str(row["stem"]) for row in manual_test],
    }
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
