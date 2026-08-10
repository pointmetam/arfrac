from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from tqdm import tqdm

from teacher_2d_to_arkit_uv import label_to_rgba, overlay_label, render_uv_label_to_portrait


SPLITS = ("train", "val", "test")
LIP_CLASSES = (1, 2)
LABEL_NAMES = {
    "0": "background_or_non_lip_uv",
    "1": "upper_lip",
    "2": "lower_lip",
    "3": "inner_mouth",
}


@dataclass
class Sample:
    split: str
    payload: dict[str, Any]
    image_path: Path
    label_path: Path
    frame_path: Path
    index: int

    @property
    def stem(self) -> str:
        return str(self.payload["stem"])

    @property
    def session(self) -> str:
        return str(self.payload.get("session", "unknown"))

    @property
    def sample_id(self) -> str:
        return str(self.payload.get("sample_id", self.stem))

    @property
    def yaw_deg(self) -> float:
        return float(self.payload.get("yaw_deg") or 0.0)

    @property
    def jaw_open(self) -> float:
        return float(self.payload.get("jaw_open") or 0.0)


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_bounds(value: str) -> tuple[int, int, int, int]:
    parts = value.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("bounds must be x0,y0,x1,y1")
    try:
        x0, y0, x1, y1 = (int(part) for part in parts)
    except ValueError as error:
        raise argparse.ArgumentTypeError("bounds values must be integers") from error
    if x0 >= x1 or y0 >= y1:
        raise argparse.ArgumentTypeError("bounds must satisfy x0<x1 and y0<y1")
    return x0, y0, x1, y1


def frame_index_from_stem(stem: str) -> int:
    digits = stem.rsplit("_", 1)[-1]
    try:
        return int(digits)
    except ValueError:
        return 0


def load_label(path: Path) -> np.ndarray:
    label = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    if int(label.max(initial=0)) > 3:
        label = (label > 0).astype(np.uint8)
    return label


def lip_mask(label: np.ndarray) -> np.ndarray:
    return np.isin(label, LIP_CLASSES)


def lip_area(label: np.ndarray) -> int:
    return int(lip_mask(label).sum())


def mask_bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def mask_centroid(mask: np.ndarray) -> tuple[float | None, float | None]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None, None
    return float(xs.mean()), float(ys.mean())


def overlaps_bounds(bbox: list[int], bounds: tuple[int, int, int, int]) -> bool:
    x0, y0, x1, y1 = bbox
    bx0, by0, bx1, by1 = bounds
    return x0 >= bx0 and y0 >= by0 and x1 <= bx1 and y1 <= by1


def bbox_shape_ok(
    bbox: list[int],
    min_width: int,
    max_width: int,
    min_height: int,
    max_height: int,
) -> bool:
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return min_width <= width <= max_width and min_height <= height <= max_height


def neighbor_indices(index: int, count: int, radius: int) -> range:
    return range(max(0, index - radius), min(count, index + radius + 1))


def temporal_majority(
    labels: list[np.ndarray],
    min_votes: int,
) -> np.ndarray:
    if not labels:
        raise ValueError("temporal_majority requires at least one label")
    if len(labels) == 1:
        return labels[0].copy()

    stack = np.stack(labels, axis=0)
    effective_votes = min(max(min_votes, 1), len(labels))
    class_counts = np.stack([(stack == class_id).sum(axis=0) for class_id in (1, 2, 3)], axis=0)
    winner = class_counts.argmax(axis=0) + 1
    votes = class_counts.max(axis=0)
    output = np.zeros(labels[0].shape, dtype=np.uint8)
    output[votes >= effective_votes] = winner[votes >= effective_votes].astype(np.uint8)
    return output


def remove_small_components(label: np.ndarray, min_pixels: int) -> np.ndarray:
    if min_pixels <= 1:
        return label

    output = label.copy()
    height, width = label.shape
    for class_id in (1, 2, 3):
        mask = label == class_id
        visited = np.zeros(mask.shape, dtype=bool)
        for start_y, start_x in zip(*np.nonzero(mask & ~visited)):
            if visited[start_y, start_x]:
                continue
            stack = [(int(start_y), int(start_x))]
            visited[start_y, start_x] = True
            component: list[tuple[int, int]] = []

            while stack:
                y, x = stack.pop()
                component.append((y, x))
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if ny < 0 or ny >= height or nx < 0 or nx >= width:
                        continue
                    if visited[ny, nx] or not mask[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    stack.append((ny, nx))

            if len(component) < min_pixels:
                for y, x in component:
                    output[y, x] = 0

    return output


def erode_binary(mask: np.ndarray, iterations: int) -> np.ndarray:
    result = mask.copy()
    for _ in range(max(iterations, 0)):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        result = (
            padded[1:-1, 1:-1]
            & padded[:-2, 1:-1]
            & padded[2:, 1:-1]
            & padded[1:-1, :-2]
            & padded[1:-1, 2:]
        )
    return result


def dilate_binary(mask: np.ndarray, iterations: int) -> np.ndarray:
    result = mask.copy()
    for _ in range(max(iterations, 0)):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        result = (
            padded[1:-1, 1:-1]
            | padded[:-2, 1:-1]
            | padded[2:, 1:-1]
            | padded[1:-1, :-2]
            | padded[1:-1, 2:]
        )
    return result


def trim_class_outer_bands(
    label: np.ndarray,
    class_id: int,
    top_pixels: int = 0,
    bottom_pixels: int = 0,
    left_pixels: int = 0,
    right_pixels: int = 0,
) -> np.ndarray:
    output = label.copy()
    class_mask = output == class_id
    if not class_mask.any():
        return output

    if top_pixels > 0 or bottom_pixels > 0:
        height, width = class_mask.shape
        for x in range(width):
            ys = np.flatnonzero(class_mask[:, x])
            if ys.size == 0:
                continue
            if top_pixels > 0:
                y1 = min(int(ys.min()) + top_pixels, height)
                output[int(ys.min()) : y1, x] = np.where(
                    output[int(ys.min()) : y1, x] == class_id,
                    0,
                    output[int(ys.min()) : y1, x],
                )
            if bottom_pixels > 0:
                y0 = max(int(ys.max()) + 1 - bottom_pixels, 0)
                output[y0 : int(ys.max()) + 1, x] = np.where(
                    output[y0 : int(ys.max()) + 1, x] == class_id,
                    0,
                    output[y0 : int(ys.max()) + 1, x],
                )

    class_mask = output == class_id
    if left_pixels > 0 or right_pixels > 0:
        height, width = class_mask.shape
        for y in range(height):
            xs = np.flatnonzero(class_mask[y, :])
            if xs.size == 0:
                continue
            if left_pixels > 0:
                x1 = min(int(xs.min()) + left_pixels, width)
                output[y, int(xs.min()) : x1] = np.where(
                    output[y, int(xs.min()) : x1] == class_id,
                    0,
                    output[y, int(xs.min()) : x1],
                )
            if right_pixels > 0:
                x0 = max(int(xs.max()) + 1 - right_pixels, 0)
                output[y, x0 : int(xs.max()) + 1] = np.where(
                    output[y, x0 : int(xs.max()) + 1] == class_id,
                    0,
                    output[y, x0 : int(xs.max()) + 1],
                )

    return output


def trim_lower_lip_center_bottom(
    label: np.ndarray,
    pixels: int,
    width_ratio: float,
) -> np.ndarray:
    if pixels <= 0:
        return label

    output = label.copy()
    lower_mask = output == 2
    bbox = mask_bbox(lower_mask)
    if bbox is None:
        return output

    x0, _, x1, _ = bbox
    width = max(1, x1 - x0)
    half_width = max(1.0, width * max(0.05, min(width_ratio, 1.0)) * 0.5)
    center_x = (x0 + x1 - 1) * 0.5

    for x in range(x0, x1):
        distance = abs(x - center_x) / half_width
        if distance >= 1.0:
            continue
        trim_pixels = int(math.ceil(pixels * (1.0 - distance * distance)))
        if trim_pixels <= 0:
            continue
        ys = np.flatnonzero(output[:, x] == 2)
        if ys.size == 0:
            continue
        y0 = max(int(ys.max()) + 1 - trim_pixels, 0)
        output[y0 : int(ys.max()) + 1, x] = np.where(
            output[y0 : int(ys.max()) + 1, x] == 2,
            0,
            output[y0 : int(ys.max()) + 1, x],
        )

    return output


def gaussian_score(mask: np.ndarray, radius: float) -> np.ndarray:
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    blurred = image.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.asarray(blurred, dtype=np.float32)


def top_area_mask(score: np.ndarray, target_area: int) -> np.ndarray:
    target_area = max(0, min(int(target_area), score.size))
    if target_area == 0:
        return np.zeros(score.shape, dtype=bool)
    if target_area == score.size:
        return np.ones(score.shape, dtype=bool)

    flat = score.reshape(-1)
    threshold = np.partition(flat, flat.size - target_area)[flat.size - target_area]
    mask = score >= threshold
    if int(mask.sum()) <= target_area:
        return mask

    # Gaussian scores can tie on flat regions. Trim tied pixels deterministically
    # so area does not grow just because of a threshold tie.
    selected = np.flatnonzero(mask.reshape(-1))
    tied = selected[flat[selected] == threshold]
    excess = int(mask.sum()) - target_area
    if excess > 0 and tied.size >= excess:
        mask_flat = mask.reshape(-1)
        mask_flat[tied[-excess:]] = False
    return mask


def smooth_label_edges(
    label: np.ndarray,
    radius: float,
    area_scale: float,
) -> np.ndarray:
    if radius <= 0:
        return label

    active = label > 0
    area = int(active.sum())
    if area == 0:
        return label

    target_area = int(round(area * max(0.01, area_scale)))
    smooth_active = top_area_mask(gaussian_score(active, radius), target_area)

    class_scores = [gaussian_score(label == class_id, radius) for class_id in (1, 2, 3)]
    score_stack = np.stack(class_scores, axis=0)
    winners = np.argmax(score_stack, axis=0).astype(np.uint8) + 1

    output = np.zeros_like(label)
    output[smooth_active] = winners[smooth_active]
    return output


def postprocess_label(
    label: np.ndarray,
    min_component_pixels: int,
    lip_erode: int,
    lip_dilate: int,
    lower_lip_bottom_trim: int,
    lower_lip_side_trim: int,
    lower_lip_center_bottom_trim: int,
    lower_lip_center_width_ratio: float,
    smooth_edge_radius: float,
    smooth_edge_area_scale: float,
) -> np.ndarray:
    output = remove_small_components(label, min_component_pixels)

    lip = lip_mask(output)
    if lip_erode > 0:
        eroded = erode_binary(lip, lip_erode)
        output[np.isin(output, LIP_CLASSES) & ~eroded] = 0
        lip = eroded

    if lip_dilate > 0:
        dilated = dilate_binary(lip, lip_dilate)
        added = dilated & ~lip
        upper_count = int((output == 1).sum())
        lower_count = int((output == 2).sum())
        fill_class = 1 if upper_count >= lower_count else 2
        output[added] = fill_class

    if lower_lip_bottom_trim > 0 or lower_lip_side_trim > 0:
        output = trim_class_outer_bands(
            output,
            class_id=2,
            bottom_pixels=lower_lip_bottom_trim,
            left_pixels=lower_lip_side_trim,
            right_pixels=lower_lip_side_trim,
        )

    if lower_lip_center_bottom_trim > 0:
        output = trim_lower_lip_center_bottom(
            output,
            pixels=lower_lip_center_bottom_trim,
            width_ratio=lower_lip_center_width_ratio,
        )

    if smooth_edge_radius > 0:
        output = smooth_label_edges(
            output,
            radius=smooth_edge_radius,
            area_scale=smooth_edge_area_scale,
        )

    output = remove_small_components(output, min_component_pixels)
    return output


def gate_sample(
    sample: Sample,
    label: np.ndarray,
    session_median_area: float,
    args: argparse.Namespace,
) -> tuple[bool, str, dict[str, Any]]:
    area = lip_area(label)
    bbox = mask_bbox(lip_mask(label))
    cx, cy = mask_centroid(lip_mask(label))
    metric = {
        "lip_area_px": area,
        "bbox_uv_px": bbox,
        "centroid_uv_px": None if cx is None or cy is None else [cx, cy],
    }

    if area < args.min_area:
        return False, "area_low", metric
    if area > args.max_area:
        return False, "area_high", metric
    if session_median_area > 0:
        ratio = area / session_median_area
        metric["session_area_ratio"] = ratio
        if ratio < args.min_session_area_ratio:
            return False, "session_area_low", metric
        if ratio > args.max_session_area_ratio:
            return False, "session_area_high", metric
    if abs(sample.yaw_deg) > args.max_abs_yaw:
        return False, "yaw_high", metric
    prev_iou = sample.payload.get("prev_binary_iou")
    if prev_iou is not None and float(prev_iou) < args.min_prev_iou:
        return False, "prev_iou_low", metric
    centroid_delta = sample.payload.get("centroid_delta_uv_px")
    if centroid_delta is not None and float(centroid_delta) > args.max_centroid_delta:
        return False, "centroid_jump", metric
    if bbox is None:
        return False, "empty_bbox", metric
    if not overlaps_bounds(bbox, args.uv_lip_bounds):
        return False, "bbox_outside_lip_bounds", metric
    if not bbox_shape_ok(bbox, args.min_bbox_width, args.max_bbox_width, args.min_bbox_height, args.max_bbox_height):
        return False, "bbox_shape", metric

    return True, "kept", metric


def read_metadata(root: Path, split: str, limit: int | None) -> list[Sample]:
    metadata_path = root / split / "metadata.jsonl"
    if not metadata_path.exists():
        return []

    samples: list[Sample] = []
    for line in metadata_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        stem = str(payload["stem"])
        image_path = root / payload.get("image", f"{split}/images/{stem}.jpg")
        label_path = root / payload.get("uv_label", f"{split}/uv_labels/{stem}.png")
        frame_path = root / payload.get("frame", f"{split}/frames/{stem}.json")
        if image_path.exists() and label_path.exists() and frame_path.exists():
            samples.append(
                Sample(
                    split=split,
                    payload=payload,
                    image_path=image_path,
                    label_path=label_path,
                    frame_path=frame_path,
                    index=frame_index_from_stem(stem),
                )
            )
    samples.sort(key=lambda sample: (sample.session, sample.index, sample.stem))
    return samples[:limit] if limit is not None else samples


def prepare_output_dirs(output_root: Path) -> None:
    for split in SPLITS:
        for folder in ("images", "uv_labels", "frames"):
            (output_root / split / folder).mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def select_neighbor_labels(
    samples: list[Sample],
    labels: dict[str, np.ndarray],
    position: int,
    args: argparse.Namespace,
) -> list[np.ndarray]:
    center = samples[position]
    selected: list[np.ndarray] = []
    for neighbor_position in neighbor_indices(position, len(samples), args.temporal_radius):
        neighbor = samples[neighbor_position]
        if neighbor.session != center.session:
            continue
        if abs(neighbor.yaw_deg - center.yaw_deg) > args.max_neighbor_yaw_delta:
            continue
        if abs(neighbor.jaw_open - center.jaw_open) > args.max_neighbor_jaw_delta:
            continue
        label = labels.get(neighbor.stem)
        if label is not None:
            selected.append(label)
    return selected


def resize_for_preview(image: Image.Image, width: int) -> Image.Image:
    if image.width == width:
        return image
    height = max(1, int(round(image.height * width / image.width)))
    return image.resize((width, height), Image.Resampling.BILINEAR)


def title(image: Image.Image, text: str) -> Image.Image:
    header = 32
    output = Image.new("RGB", (image.width, image.height + header), "white")
    output.paste(image.convert("RGB"), (0, header))
    draw = ImageDraw.Draw(output)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    draw.text((8, 9), text, fill=(28, 28, 28), font=font)
    return output


def uv_label_rgb(label: np.ndarray, size: int) -> Image.Image:
    image = label_to_rgba(label).convert("RGB")
    return image.resize((size, size), Image.Resampling.NEAREST)


def diff_rgb(raw: np.ndarray, refined: np.ndarray, size: int) -> Image.Image:
    rgba = np.zeros((raw.shape[0], raw.shape[1], 4), dtype=np.uint8)
    removed = (raw > 0) & (refined == 0)
    added = (raw == 0) & (refined > 0)
    changed = (raw > 0) & (refined > 0) & (raw != refined)
    rgba[removed] = [255, 44, 44, 235]
    rgba[added] = [46, 214, 90, 235]
    rgba[changed] = [255, 196, 0, 235]
    return Image.fromarray(rgba).convert("RGB").resize((size, size), Image.Resampling.NEAREST)


def make_preview_panel(
    sample: Sample,
    raw_label: np.ndarray,
    refined_label: np.ndarray,
    metric: dict[str, Any],
    output_width: int,
) -> Image.Image:
    portrait = Image.open(sample.image_path).convert("RGB")
    frame = json.loads(sample.frame_path.read_text(encoding="utf-8"))
    portrait_size = portrait.size
    raw_portrait = render_uv_label_to_portrait(frame, raw_label, portrait_size)
    refined_portrait = render_uv_label_to_portrait(frame, refined_label, portrait_size)
    preview_width = max(240, output_width // 5)

    portrait_small = resize_for_preview(portrait, preview_width)
    raw_overlay = resize_for_preview(overlay_label(portrait, raw_portrait), preview_width)
    refined_overlay = resize_for_preview(overlay_label(portrait, refined_portrait), preview_width)
    uv_size = portrait_small.height
    raw_uv = uv_label_rgb(raw_label, uv_size)
    refined_uv = uv_label_rgb(refined_label, uv_size)
    diff_uv = diff_rgb(raw_label, refined_label, uv_size)

    columns = [
        title(portrait_small, "image"),
        title(raw_overlay, "raw teacher"),
        title(refined_overlay, "refined"),
        title(raw_uv, "raw UV"),
        title(refined_uv, "refined UV"),
        title(diff_uv, "diff R-/G+"),
    ]
    height = max(column.height for column in columns)
    width = sum(column.width for column in columns)
    grid = Image.new("RGB", (width, height), "white")
    x = 0
    for column in columns:
        grid.paste(column, (x, 0))
        x += column.width

    footer = 56
    output = Image.new("RGB", (grid.width, grid.height + footer), "white")
    output.paste(grid, (0, 0))
    draw = ImageDraw.Draw(output)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    raw_area = lip_area(raw_label)
    refined_area = lip_area(refined_label)
    draw.text(
        (8, grid.height + 8),
        (
            f"{sample.stem}  split={sample.split} yaw={sample.yaw_deg:+.1f} jaw={sample.jaw_open:.2f}  "
            f"area {raw_area}->{refined_area}  reason={metric.get('reason', 'kept')}"
        ),
        fill=(28, 28, 28),
        font=font,
    )
    draw.text(
        (8, grid.height + 30),
        "diff: red removed, green added, yellow class changed",
        fill=(72, 72, 72),
        font=font,
    )
    return resize_for_preview(output, output_width)


def make_contact_sheet(panels: list[Image.Image], columns: int = 2) -> Image.Image:
    if not panels:
        return Image.new("RGB", (1, 1), "white")
    cell_width = max(panel.width for panel in panels)
    cell_height = max(panel.height for panel in panels)
    rows = int(math.ceil(len(panels) / columns))
    output = Image.new("RGB", (cell_width * columns, cell_height * rows), "white")
    for index, panel in enumerate(panels):
        x = (index % columns) * cell_width
        y = (index // columns) * cell_height
        output.paste(panel, (x, y))
    return output


def copy_sample(sample: Sample, refined_label: np.ndarray, output_root: Path, metadata: dict[str, Any]) -> None:
    split_root = output_root / sample.split
    image_out = split_root / "images" / f"{sample.stem}.jpg"
    label_out = split_root / "uv_labels" / f"{sample.stem}.png"
    frame_out = split_root / "frames" / f"{sample.stem}.json"
    shutil.copy2(sample.image_path, image_out)
    shutil.copy2(sample.frame_path, frame_out)
    Image.fromarray(refined_label).save(label_out)
    metadata.update(
        {
            "image": str(image_out.relative_to(output_root)),
            "uv_label": str(label_out.relative_to(output_root)),
            "frame": str(frame_out.relative_to(output_root)),
            "refined_from": str(sample.label_path),
        }
    )


def process_split(
    root: Path,
    output_root: Path,
    split: str,
    args: argparse.Namespace,
) -> tuple[dict[str, int], list[tuple[Sample, np.ndarray, np.ndarray, dict[str, Any]]]]:
    samples = read_metadata(root, split, args.limit_per_split)
    by_session: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        by_session[sample.session].append(sample)

    counts: Counter[str] = Counter()
    kept_rows: list[dict[str, Any]] = []
    preview_items: list[tuple[Sample, np.ndarray, np.ndarray, dict[str, Any]]] = []

    for session, session_samples in by_session.items():
        session_samples.sort(key=lambda sample: sample.index)
        raw_labels = {sample.stem: load_label(sample.label_path) for sample in session_samples}
        raw_areas = [lip_area(label) for label in raw_labels.values() if lip_area(label) > 0]
        session_median_area = float(np.median(raw_areas)) if raw_areas else 0.0

        for position, sample in enumerate(tqdm(session_samples, desc=f"{split}:{session}", leave=False)):
            raw_label = raw_labels[sample.stem]
            ok, reason, raw_metric = gate_sample(sample, raw_label, session_median_area, args)
            if not ok:
                counts[reason] += 1
                continue

            neighbor_labels = select_neighbor_labels(session_samples, raw_labels, position, args)
            refined_label = temporal_majority(neighbor_labels, args.temporal_min_votes)
            refined_label = postprocess_label(
                refined_label,
                min_component_pixels=args.min_component_pixels,
                lip_erode=args.lip_erode,
                lip_dilate=args.lip_dilate,
                lower_lip_bottom_trim=args.lower_lip_bottom_trim,
                lower_lip_side_trim=args.lower_lip_side_trim,
                lower_lip_center_bottom_trim=args.lower_lip_center_bottom_trim,
                lower_lip_center_width_ratio=args.lower_lip_center_width_ratio,
                smooth_edge_radius=args.smooth_edge_radius,
                smooth_edge_area_scale=args.smooth_edge_area_scale,
            )
            ok, reason, refined_metric = gate_sample(sample, refined_label, session_median_area, args)
            if not ok:
                counts[f"refined_{reason}"] += 1
                continue

            metadata = dict(sample.payload)
            metadata.update(
                {
                    "lip_area_px": refined_metric["lip_area_px"],
                    "bbox_uv_px": refined_metric["bbox_uv_px"],
                    "centroid_uv_px": refined_metric["centroid_uv_px"],
                    "raw_lip_area_px": raw_metric["lip_area_px"],
                    "raw_bbox_uv_px": raw_metric["bbox_uv_px"],
                    "raw_centroid_uv_px": raw_metric["centroid_uv_px"],
                    "refined_lip_area_px": refined_metric["lip_area_px"],
                    "refined_bbox_uv_px": refined_metric["bbox_uv_px"],
                    "refined_centroid_uv_px": refined_metric["centroid_uv_px"],
                    "refine_temporal_radius": args.temporal_radius,
                    "refine_temporal_votes": min(args.temporal_min_votes, len(neighbor_labels)),
                    "refine_neighbor_count": len(neighbor_labels),
                    "refine_lip_erode": args.lip_erode,
                    "refine_lip_dilate": args.lip_dilate,
                    "refine_lower_lip_bottom_trim": args.lower_lip_bottom_trim,
                    "refine_lower_lip_side_trim": args.lower_lip_side_trim,
                    "refine_lower_lip_center_bottom_trim": args.lower_lip_center_bottom_trim,
                    "refine_lower_lip_center_width_ratio": args.lower_lip_center_width_ratio,
                    "refine_smooth_edge_radius": args.smooth_edge_radius,
                    "refine_smooth_edge_area_scale": args.smooth_edge_area_scale,
                }
            )
            copy_sample(sample, refined_label, output_root, metadata)
            kept_rows.append(metadata)
            counts["kept"] += 1
            preview_items.append((sample, raw_label, refined_label, {**refined_metric, "reason": "kept"}))

    write_jsonl(output_root / split / "metadata.jsonl", kept_rows)
    return dict(counts), preview_items


def select_evenly(items: list[Any], count: int) -> list[Any]:
    if count <= 0 or len(items) <= count:
        return items
    indices = np.linspace(0, len(items) - 1, count).round().astype(int)
    return [items[int(index)] for index in indices]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/Users/lee/Datasets/ARKitLipUVTeacherV2")
    parser.add_argument("--output-root", default="/Users/lee/Datasets/ARKitLipUVTeacherV2Refined")
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit-per-split", type=int, default=None)
    parser.add_argument("--temporal-radius", type=int, default=2)
    parser.add_argument("--temporal-min-votes", type=int, default=2)
    parser.add_argument("--max-neighbor-yaw-delta", type=float, default=18.0)
    parser.add_argument("--max-neighbor-jaw-delta", type=float, default=0.20)
    parser.add_argument("--min-area", type=int, default=700)
    parser.add_argument("--max-area", type=int, default=9500)
    parser.add_argument("--min-session-area-ratio", type=float, default=0.45)
    parser.add_argument("--max-session-area-ratio", type=float, default=1.75)
    parser.add_argument("--max-centroid-delta", type=float, default=45.0)
    parser.add_argument("--min-prev-iou", type=float, default=0.25)
    parser.add_argument("--max-abs-yaw", type=float, default=70.0)
    parser.add_argument("--uv-lip-bounds", type=parse_bounds, default=parse_bounds("135,230,385,450"))
    parser.add_argument("--min-bbox-width", type=int, default=24)
    parser.add_argument("--max-bbox-width", type=int, default=220)
    parser.add_argument("--min-bbox-height", type=int, default=12)
    parser.add_argument("--max-bbox-height", type=int, default=170)
    parser.add_argument("--min-component-pixels", type=int, default=40)
    parser.add_argument("--lip-erode", type=int, default=0)
    parser.add_argument("--lip-dilate", type=int, default=0)
    parser.add_argument("--lower-lip-bottom-trim", type=int, default=0)
    parser.add_argument("--lower-lip-side-trim", type=int, default=0)
    parser.add_argument("--lower-lip-center-bottom-trim", type=int, default=0)
    parser.add_argument("--lower-lip-center-width-ratio", type=float, default=0.55)
    parser.add_argument("--smooth-edge-radius", type=float, default=0.0)
    parser.add_argument("--smooth-edge-area-scale", type=float, default=1.0)
    parser.add_argument("--preview-count", type=int, default=48)
    parser.add_argument("--preview-dir", default="training/outputs/arkit_uv_teacher_refine_preview")
    parser.add_argument("--preview-width", type=int, default=1500)
    args = parser.parse_args()

    root = Path(args.root).expanduser()
    output_root = Path(args.output_root).expanduser()
    preview_dir = Path(args.preview_dir).expanduser()
    splits = [split for split in parse_csv(args.splits) if split in SPLITS]
    if not splits:
        raise SystemExit("No valid splits selected")
    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    prepare_output_dirs(output_root)

    summary: dict[str, Any] = {
        "root": str(root),
        "output_root": str(output_root),
        "splits": splits,
        "labels": LABEL_NAMES,
        "params": {
            "temporal_radius": args.temporal_radius,
            "temporal_min_votes": args.temporal_min_votes,
            "max_neighbor_yaw_delta": args.max_neighbor_yaw_delta,
            "max_neighbor_jaw_delta": args.max_neighbor_jaw_delta,
            "min_area": args.min_area,
            "max_area": args.max_area,
            "min_session_area_ratio": args.min_session_area_ratio,
            "max_session_area_ratio": args.max_session_area_ratio,
            "max_centroid_delta": args.max_centroid_delta,
            "min_prev_iou": args.min_prev_iou,
            "max_abs_yaw": args.max_abs_yaw,
            "uv_lip_bounds": list(args.uv_lip_bounds),
            "min_component_pixels": args.min_component_pixels,
            "lip_erode": args.lip_erode,
            "lip_dilate": args.lip_dilate,
            "lower_lip_bottom_trim": args.lower_lip_bottom_trim,
            "lower_lip_side_trim": args.lower_lip_side_trim,
            "lower_lip_center_bottom_trim": args.lower_lip_center_bottom_trim,
            "lower_lip_center_width_ratio": args.lower_lip_center_width_ratio,
            "smooth_edge_radius": args.smooth_edge_radius,
            "smooth_edge_area_scale": args.smooth_edge_area_scale,
        },
        "counts": {},
        "previews": [],
    }

    all_preview_items: list[tuple[Sample, np.ndarray, np.ndarray, dict[str, Any]]] = []
    for split in splits:
        counts, preview_items = process_split(root, output_root, split, args)
        summary["counts"][split] = counts
        all_preview_items.extend(preview_items)

    selected = select_evenly(all_preview_items, args.preview_count)
    panels: list[Image.Image] = []
    for index, (sample, raw_label, refined_label, metric) in enumerate(selected, start=1):
        panel = make_preview_panel(sample, raw_label, refined_label, metric, args.preview_width)
        output_path = preview_dir / f"preview_{index:02d}_{sample.stem}.jpg"
        panel.save(output_path, quality=94)
        panels.append(panel)
        summary["previews"].append(str(output_path))

    contact_sheet = make_contact_sheet(panels, columns=2)
    contact_sheet_path = preview_dir / "contact_sheet.jpg"
    contact_sheet.save(contact_sheet_path, quality=94)
    summary["contact_sheet"] = str(contact_sheet_path)

    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (preview_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
