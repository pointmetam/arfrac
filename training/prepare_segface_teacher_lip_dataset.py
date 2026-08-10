from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from prepare_lip_dataset import (
    bbox_from_mask,
    expanded_aspect_bbox,
    image_by_stem,
    lip_label_from_lapa,
    parse_splits,
)
from segface_teacher_preview import (
    choose_device,
    load_segface_model,
    read_lapa_landmarks,
    segface_lip_label,
)


PredictLipLabel = Callable[[Image.Image, np.ndarray], np.ndarray]
TEACHER_MAPPING = {"7": 1, "9": 2, "8": 3}


def prepare_segface_teacher_split(
    *,
    source_root: str | Path,
    output_root: str | Path,
    split: str,
    width: int,
    height: int,
    padding_x: float,
    padding_y: float,
    limit: int | None,
    predict_lip_label: PredictLipLabel,
    teacher_name: str,
    teacher_checkpoint: str,
    teacher_resolution: int,
) -> dict[str, int]:
    source = Path(source_root).expanduser()
    output = Path(output_root).expanduser()
    split_root = source / split
    image_dir = split_root / "images"
    label_dir = split_root / "labels"
    landmark_dir = split_root / "landmarks"
    if not image_dir.exists() or not label_dir.exists() or not landmark_dir.exists():
        raise FileNotFoundError(f"Missing LaPa images/labels/landmarks under {split_root}")

    output_image_dir = output / split / "images"
    output_label_dir = output / split / "labels"
    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    images = image_by_stem(image_dir)
    label_paths = sorted(label_dir.glob("*.png"))
    if limit is not None:
        label_paths = label_paths[:limit]

    kept = 0
    skipped = 0
    metadata_path = output / split / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        for label_path in tqdm(label_paths, desc=split):
            image_path = images.get(label_path.stem)
            landmark_path = landmark_dir / f"{label_path.stem}.txt"
            if image_path is None or not landmark_path.exists():
                skipped += 1
                continue

            image = Image.open(image_path).convert("RGB")
            lapa_label = np.asarray(Image.open(label_path).convert("L"), dtype=np.uint8)
            lapa_lip_label = lip_label_from_lapa(lapa_label)
            bbox = bbox_from_mask(lapa_lip_label > 0)
            if bbox is None:
                skipped += 1
                continue

            landmarks = read_lapa_landmarks(landmark_path)
            teacher_lip_label = predict_lip_label(image, landmarks)
            if teacher_lip_label.shape != lapa_label.shape:
                raise ValueError(
                    f"Teacher label shape {teacher_lip_label.shape} does not match "
                    f"LaPa label shape {lapa_label.shape} for {label_path.stem}"
                )

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
            label_crop = (
                Image.fromarray(teacher_lip_label.astype(np.uint8))
                .crop(crop_box)
                .resize((width, height), Image.Resampling.NEAREST)
            )

            output_stem = label_path.stem
            image_crop.save(output_image_dir / f"{output_stem}.jpg", quality=96)
            label_crop.save(output_label_dir / f"{output_stem}.png")
            metadata_file.write(
                json.dumps(
                    {
                        "stem": output_stem,
                        "source_stem": label_path.stem,
                        "source_image": str(image_path),
                        "source_lapa_label": str(label_path),
                        "source_landmarks": str(landmark_path),
                        "crop_box": crop_box,
                        "teacher": teacher_name,
                        "teacher_checkpoint": teacher_checkpoint,
                        "teacher_resolution": teacher_resolution,
                        "teacher_mapping": TEACHER_MAPPING,
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )
            kept += 1

    return {"kept": kept, "skipped": skipped}


def prepare_segface_teacher_dataset(
    *,
    source_root: str | Path,
    output_root: str | Path,
    splits: list[str],
    width: int,
    height: int,
    padding_x: float,
    padding_y: float,
    limit: int | None,
    segface_root: str | Path,
    checkpoint: str | Path,
    resolution: int,
    device_name: str,
    overwrite: bool = False,
) -> dict[str, dict[str, int]]:
    output = Path(output_root).expanduser()
    if overwrite and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    labels_source = Path(__file__).with_name("lip_labels.json")
    (output / "lip_labels.json").write_text(labels_source.read_text(encoding="utf-8"), encoding="utf-8")

    device = choose_device(device_name)
    checkpoint_path = Path(checkpoint).expanduser()
    print(f"Device: {device}")
    print(f"Loading SegFace LaPa {resolution}: {checkpoint_path}")
    model = load_segface_model(Path(segface_root), checkpoint_path, resolution, device)
    teacher_name = f"segface/swinb_lapa_{resolution}"

    def predict_lip_label(image: Image.Image, landmarks: np.ndarray) -> np.ndarray:
        return segface_lip_label(model, image, landmarks, resolution, device)

    summary: dict[str, dict[str, int]] = {}
    for split in splits:
        summary[split] = prepare_segface_teacher_split(
            source_root=source_root,
            output_root=output,
            split=split,
            width=width,
            height=height,
            padding_x=padding_x,
            padding_y=padding_y,
            limit=limit,
            predict_lip_label=predict_lip_label,
            teacher_name=teacher_name,
            teacher_checkpoint=str(checkpoint_path),
            teacher_resolution=resolution,
        )

    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "teacher": teacher_name,
                "teacher_checkpoint": str(checkpoint_path),
                "teacher_resolution": resolution,
                "splits": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/Users/lee/Datasets/LaPa")
    parser.add_argument("--output-root", default="/Users/lee/Datasets/LaPaLipTeacherSegFace")
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--padding-x", type=float, default=0.82)
    parser.add_argument("--padding-y", type=float, default=1.25)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--segface-root", default="third_party/SegFace")
    parser.add_argument("--checkpoint", default="/Users/lee/Models/SegFace/swinb_lapa_512/model_299.pt")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    summary = prepare_segface_teacher_dataset(
        source_root=args.root,
        output_root=args.output_root,
        splits=parse_splits(args.splits),
        width=args.width,
        height=args.height,
        padding_x=args.padding_x,
        padding_y=args.padding_y,
        limit=args.limit,
        segface_root=args.segface_root,
        checkpoint=args.checkpoint,
        resolution=args.resolution,
        device_name=args.device,
        overwrite=args.overwrite,
    )
    output_root = Path(args.output_root).expanduser()
    print(f"SegFace teacher lip ROI dataset: {output_root}")
    for split, stats in summary.items():
        print(f"{split}: kept={stats['kept']} skipped={stats['skipped']}")


if __name__ == "__main__":
    main()
