from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from prepare_lip_dataset import (
    bbox_from_mask,
    expanded_aspect_bbox,
    image_by_stem,
    lip_label_from_lapa,
)
from segface_teacher_preview import (
    crop_array,
    difference_image,
    label_to_rgba,
    load_segface_model,
    make_grid,
    overlay_label,
    read_lapa_landmarks,
    segface_lip_label,
    titled,
    update_iou,
)


def choose_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        print("MPS requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(name)


def centroid(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def padded_box(mask: np.ndarray, image_size: tuple[int, int], padding: float) -> np.ndarray:
    bbox = bbox_from_mask(mask)
    if bbox is None:
        raise ValueError("Cannot build a prompt box from an empty mask")
    width, height = image_size
    left, top, right, bottom = bbox
    box_width = right - left
    box_height = bottom - top
    pad_x = box_width * padding
    pad_y = box_height * padding
    return np.asarray(
        [
            max(0.0, left - pad_x),
            max(0.0, top - pad_y),
            min(float(width - 1), right + pad_x),
            min(float(height - 1), bottom + pad_y),
        ],
        dtype=np.float32,
    )


def prompt_points_from_label(label: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points: list[tuple[float, float]] = []
    point_labels: list[int] = []

    for class_id in (1, 2):
        point = centroid(label == class_id)
        if point is not None:
            points.append(point)
            point_labels.append(1)

    combined = np.isin(label, [1, 2])
    point = centroid(combined)
    if point is not None:
        points.append(point)
        point_labels.append(1)

    inner = centroid(label == 3)
    if inner is not None:
        points.append(inner)
        point_labels.append(0)

    bbox = bbox_from_mask(combined)
    if bbox is not None:
        left, top, right, bottom = bbox
        width = right - left
        height = bottom - top
        negatives = [
            (left - width * 0.18, (top + bottom) * 0.5),
            (right + width * 0.18, (top + bottom) * 0.5),
            ((left + right) * 0.5, top - height * 0.30),
            ((left + right) * 0.5, bottom + height * 0.30),
        ]
        image_height, image_width = label.shape
        for x, y in negatives:
            points.append((float(np.clip(x, 0, image_width - 1)), float(np.clip(y, 0, image_height - 1))))
            point_labels.append(0)

    return np.asarray(points, dtype=np.float32), np.asarray(point_labels, dtype=np.int32)


def assign_refined_lip_classes(original_label: np.ndarray, refined_lip_surface: np.ndarray) -> np.ndarray:
    output = np.zeros(original_label.shape, dtype=np.uint8)
    lip_seed = np.isin(original_label, [1, 2])
    if not lip_seed.any():
        output[original_label == 3] = 3
        return output

    try:
        from scipy import ndimage  # type: ignore

        _, nearest_indices = ndimage.distance_transform_edt(~lip_seed, return_indices=True)
        nearest_classes = original_label[nearest_indices[0], nearest_indices[1]]
    except ModuleNotFoundError:
        seed_yx = np.argwhere(lip_seed)
        seed_classes = original_label[lip_seed]
        nearest_classes = np.zeros(original_label.shape, dtype=np.uint8)
        target_yx = np.argwhere(refined_lip_surface)
        for start in range(0, target_yx.shape[0], 2048):
            target_chunk = target_yx[start : start + 2048]
            distances = ((target_chunk[:, None, :] - seed_yx[None, :, :]) ** 2).sum(axis=2)
            nearest_classes[target_chunk[:, 0], target_chunk[:, 1]] = seed_classes[distances.argmin(axis=1)]

    output[refined_lip_surface] = nearest_classes[refined_lip_surface]
    output[(output == 0) & refined_lip_surface] = original_label[(output == 0) & refined_lip_surface]
    output[original_label == 3] = 3
    return output


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    mask = mask.astype(bool)
    if radius <= 0 or not mask.any():
        return mask.copy()

    height, width = mask.shape
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    output = np.zeros_like(mask, dtype=bool)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            y0 = radius + dy
            x0 = radius + dx
            output |= padded[y0 : y0 + height, x0 : x0 + width]
    return output


def best_sam2_mask(
    masks: np.ndarray,
    scores: np.ndarray,
    initial_lip_surface: np.ndarray,
    *,
    min_area_ratio: float,
    max_area_ratio: float,
    max_expand_px: int,
) -> np.ndarray:
    if masks.ndim == 4:
        masks = masks[:, 0]
    initial = initial_lip_surface.astype(bool)
    initial_area = max(float(initial.sum()), 1.0)
    allowed = dilate_mask(initial, max_expand_px)
    best_mask: np.ndarray | None = None
    best_score = -1.0
    for index, mask in enumerate(masks):
        candidate = mask.astype(bool) & allowed
        area_ratio = float(candidate.sum()) / initial_area
        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            continue
        union = np.logical_or(candidate, initial).sum()
        overlap = 0.0 if union == 0 else np.logical_and(candidate, initial).sum() / union
        area_penalty = abs(np.log(max(area_ratio, 1e-6)))
        score = overlap + 0.08 * float(scores[index]) - 0.20 * area_penalty
        if score > best_score:
            best_score = score
            best_mask = candidate
    if best_mask is None:
        return initial.copy()
    return best_mask.astype(bool)


def load_sam2_predictor(
    *,
    sam2_root: Path | None,
    model_cfg: str,
    checkpoint: Path,
    device: torch.device,
) -> Any:
    if sam2_root is not None:
        sys.path.insert(0, str(sam2_root))
    try:
        from sam2.build_sam import build_sam2  # type: ignore
        from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing SAM2 dependency. Install facebookresearch/sam2 into the active environment, "
            "then rerun this script."
        ) from exc

    model = build_sam2(model_cfg, str(checkpoint), device=device)
    return SAM2ImagePredictor(model)


def refine_label_with_sam2(
    predictor: Any,
    image: Image.Image,
    segface_label: np.ndarray,
    box_padding: float,
    min_area_ratio: float,
    max_area_ratio: float,
    max_expand_px: int,
) -> np.ndarray:
    initial_lip_surface = np.isin(segface_label, [1, 2])
    if not initial_lip_surface.any():
        return segface_label

    predictor.set_image(np.asarray(image.convert("RGB")))
    points, point_labels = prompt_points_from_label(segface_label)
    box = padded_box(initial_lip_surface, image.size, box_padding)
    masks, scores, _ = predictor.predict(
        point_coords=points,
        point_labels=point_labels,
        box=box,
        multimask_output=True,
    )
    refined_surface = best_sam2_mask(
        np.asarray(masks),
        np.asarray(scores),
        initial_lip_surface,
        min_area_ratio=min_area_ratio,
        max_area_ratio=max_area_ratio,
        max_expand_px=max_expand_px,
    )
    return assign_refined_lip_classes(segface_label, refined_surface)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/Users/lee/Datasets/LaPa")
    parser.add_argument("--split", default="val")
    parser.add_argument("--output-dir", default="training/outputs/teacher/segface_sam2_lip_refine_preview")
    parser.add_argument("--segface-root", default="third_party/SegFace")
    parser.add_argument("--segface-checkpoint", default="/Users/lee/Models/SegFace/swinb_lapa_512/model_299.pt")
    parser.add_argument("--segface-resolution", type=int, default=512)
    parser.add_argument("--sam2-root", default=None)
    parser.add_argument("--sam2-checkpoint", required=True)
    parser.add_argument("--sam2-model-cfg", default="configs/sam2.1/sam2.1_hiera_s.yaml")
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--crop-width", type=int, default=384)
    parser.add_argument("--crop-height", type=int, default=256)
    parser.add_argument("--box-padding", type=float, default=0.02)
    parser.add_argument("--min-area-ratio", type=float, default=0.75)
    parser.add_argument("--max-area-ratio", type=float, default=1.25)
    parser.add_argument("--max-expand-px", type=int, default=2)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda")
    args = parser.parse_args()

    root = Path(args.root).expanduser()
    split_root = root / args.split
    image_dir = split_root / "images"
    label_dir = split_root / "labels"
    landmark_dir = split_root / "landmarks"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    print(f"Device: {device}")
    segface_model = load_segface_model(
        Path(args.segface_root),
        Path(args.segface_checkpoint).expanduser(),
        args.segface_resolution,
        device,
    )
    predictor = load_sam2_predictor(
        sam2_root=None if args.sam2_root is None else Path(args.sam2_root).expanduser(),
        model_cfg=args.sam2_model_cfg,
        checkpoint=Path(args.sam2_checkpoint).expanduser(),
        device=device,
    )

    images = image_by_stem(image_dir)
    labels = sorted(label_dir.glob("*.png"))[args.start : args.start + args.count]
    intersections = np.zeros(4, dtype=np.float64)
    unions = np.zeros(4, dtype=np.float64)
    saved: list[str] = []
    skipped: list[str] = []

    with torch.inference_mode():
        for output_index, label_path in enumerate(tqdm(labels), start=1):
            image_path = images.get(label_path.stem)
            landmark_path = landmark_dir / f"{label_path.stem}.txt"
            if image_path is None or not landmark_path.exists():
                skipped.append(label_path.stem)
                continue

            image = Image.open(image_path).convert("RGB")
            lapa_label = np.asarray(Image.open(label_path).convert("L"), dtype=np.uint8)
            gt_lip = lip_label_from_lapa(lapa_label)
            bbox = bbox_from_mask(gt_lip > 0)
            if bbox is None:
                skipped.append(label_path.stem)
                continue

            landmarks = read_lapa_landmarks(landmark_path)
            segface_label = segface_lip_label(segface_model, image, landmarks, args.segface_resolution, device)
            refined_label = refine_label_with_sam2(
                predictor,
                image,
                segface_label,
                args.box_padding,
                args.min_area_ratio,
                args.max_area_ratio,
                args.max_expand_px,
            )

            crop_box = expanded_aspect_bbox(
                bbox,
                image.width,
                image.height,
                args.crop_width,
                args.crop_height,
                padding_x=0.82,
                padding_y=1.25,
            )
            preview_size = (args.crop_width, args.crop_height)
            image_crop = image.crop(crop_box).resize(preview_size, Image.Resampling.BILINEAR)
            segface_crop = crop_array(segface_label, crop_box, preview_size)
            refined_crop = crop_array(refined_label, crop_box, preview_size)
            update_iou(intersections, unions, segface_crop, refined_crop)
            diff = Image.alpha_composite(
                image_crop.convert("RGBA"),
                difference_image(segface_crop, refined_crop),
            ).convert("RGB")
            grid = make_grid(
                [
                    titled(image_crop, "image"),
                    titled(overlay_label(image_crop, segface_crop), "SegFace"),
                    titled(overlay_label(image_crop, refined_crop), "SegFace + SAM2"),
                    titled(diff, "diff"),
                    titled(label_to_rgba(refined_crop).convert("RGB"), "refined mask"),
                ]
            )
            output_path = output_dir / f"preview_{output_index:02d}_{label_path.stem}.jpg"
            grid.save(output_path, quality=94)
            saved.append(str(output_path))
            print(output_path)

    class_names = ["background_skin", "upper_lip", "lower_lip", "inner_mouth"]
    per_class_iou = {}
    valid_ious = []
    for class_id, class_name in enumerate(class_names):
        if unions[class_id] > 0:
            value = float(intersections[class_id] / unions[class_id])
            per_class_iou[class_name] = value
            valid_ious.append(value)

    summary = {
        "teacher": "segface+sam2",
        "device": str(device),
        "split": args.split,
        "count_requested": args.count,
        "count_saved": len(saved),
        "skipped": skipped,
        "segface_checkpoint": args.segface_checkpoint,
        "sam2_checkpoint": args.sam2_checkpoint,
        "sam2_model_cfg": args.sam2_model_cfg,
        "box_padding": args.box_padding,
        "min_area_ratio": args.min_area_ratio,
        "max_area_ratio": args.max_area_ratio,
        "max_expand_px": args.max_expand_px,
        "mean_iou_vs_segface": float(sum(valid_ious) / max(len(valid_ious), 1)),
        "class_iou_vs_segface": per_class_iou,
        "previews": saved,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
