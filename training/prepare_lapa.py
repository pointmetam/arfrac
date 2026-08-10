from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def count_files(path: Path, suffixes: tuple[str, ...]) -> int:
    return sum(1 for item in path.iterdir() if item.is_file() and item.suffix.lower() in suffixes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/Users/lee/Datasets/LaPa")
    parser.add_argument("--sample-labels", type=int, default=20)
    args = parser.parse_args()

    root = Path(args.root).expanduser()
    if not root.exists():
        raise FileNotFoundError(root)

    print(f"LaPa root: {root}")
    for split in ("train", "val", "test"):
        split_root = root / split
        image_dir = split_root / "images"
        label_dir = split_root / "labels"
        landmark_dir = split_root / "landmarks"
        print(
            f"{split}: "
            f"images={count_files(image_dir, ('.jpg', '.jpeg', '.png'))} "
            f"labels={count_files(label_dir, ('.png',))} "
            f"landmarks={count_files(landmark_dir, ('.txt',))}"
        )

    values: set[int] = set()
    for label_path in sorted((root / "train" / "labels").glob("*.png"))[: args.sample_labels]:
        label = np.asarray(Image.open(label_path).convert("L"))
        values.update(int(value) for value in np.unique(label))

    print(f"Observed train label ids in first {args.sample_labels} labels: {sorted(values)}")
    print("Expected LaPa ids: 0..10")


if __name__ == "__main__":
    main()
