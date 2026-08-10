from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from prepare_lip_dataset import bbox_from_mask, expanded_aspect_bbox


CLASS_IDS = {
    "upper_lip": 1,
    "lower_lip": 2,
    "inner_mouth": 3,
}

CLASS_COLORS = {
    1: (236, 36, 82, 185),
    2: (255, 92, 128, 185),
    3: (24, 24, 28, 230),
}


def annotations_xml_path(path: Path) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    if path.is_dir():
        candidate = path / "annotations.xml"
        if candidate.exists():
            return candidate, None
    if path.suffix.lower() == ".xml":
        return path, None
    if path.suffix.lower() != ".zip":
        raise ValueError(f"Expected CVAT annotations .zip/.xml, got {path}")

    temp_dir = tempfile.TemporaryDirectory()
    with zipfile.ZipFile(path) as zf:
        candidates = [name for name in zf.namelist() if Path(name).name == "annotations.xml"]
        if not candidates:
            raise FileNotFoundError("annotations.xml not found in CVAT zip")
        zf.extract(candidates[0], temp_dir.name)
        return Path(temp_dir.name) / candidates[0], temp_dir


def parse_points(value: str) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for item in value.split(";"):
        if not item.strip():
            continue
        x, y = item.split(",", 1)
        points.append((float(x), float(y)))
    return points


def segment_lengths(points: list[tuple[float, float]]) -> tuple[list[float], float]:
    lengths: list[float] = []
    total = 0.0
    for index, point in enumerate(points):
        next_point = points[(index + 1) % len(points)]
        length = math.hypot(next_point[0] - point[0], next_point[1] - point[1])
        lengths.append(length)
        total += length
    return lengths, total


def resample_closed(points: list[tuple[float, float]], count: int) -> list[tuple[float, float]]:
    if len(points) == count:
        return points
    if len(points) < 3:
        return points

    lengths, total = segment_lengths(points)
    if total <= 1e-6:
        return points

    output: list[tuple[float, float]] = []
    step = total / count
    segment_index = 0
    segment_start_distance = 0.0
    for point_index in range(count):
        target = point_index * step
        while segment_index < len(points) - 1 and segment_start_distance + lengths[segment_index] < target:
            segment_start_distance += lengths[segment_index]
            segment_index += 1

        start = points[segment_index]
        end = points[(segment_index + 1) % len(points)]
        length = lengths[segment_index]
        t = 0.0 if length <= 1e-6 else (target - segment_start_distance) / length
        output.append((start[0] + (end[0] - start[0]) * t, start[1] + (end[1] - start[1]) * t))
    return output


def interpolate_points(
    before: list[tuple[float, float]],
    after: list[tuple[float, float]],
    t: float,
    count: int,
) -> list[tuple[float, float]]:
    before = resample_closed(before, count)
    after = resample_closed(after, count)
    if len(before) != len(after):
        raise ValueError(f"Cannot interpolate polygons with point counts {len(before)} and {len(after)}")
    return [(ax + (bx - ax) * t, ay + (by - ay) * t) for (ax, ay), (bx, by) in zip(before, after)]


def read_tracks(
    annotations_path: Path,
    *,
    points_per_polygon: int,
) -> tuple[dict[str, list[dict[str, Any]]], int | None, tuple[int, int] | None]:
    xml_path, temp_dir = annotations_xml_path(annotations_path)
    try:
        root = ET.parse(xml_path).getroot()
        task_size: int | None = None
        original_size: tuple[int, int] | None = None
        task_node = root.find("./meta/task")
        if task_node is not None:
            size_node = task_node.find("size")
            if size_node is not None and size_node.text:
                task_size = int(size_node.text)
            width_node = task_node.find("./original_size/width")
            height_node = task_node.find("./original_size/height")
            if width_node is not None and height_node is not None and width_node.text and height_node.text:
                original_size = (int(width_node.text), int(height_node.text))

        tracks: dict[str, list[dict[str, Any]]] = {name: [] for name in CLASS_IDS}
        for track in root.findall("track"):
            label = track.attrib.get("label", "")
            if label not in CLASS_IDS:
                continue
            for polygon in track.findall("polygon"):
                outside = polygon.attrib.get("outside", "0") in {"1", "true", "True"}
                if outside:
                    continue
                points = parse_points(polygon.attrib.get("points", ""))
                if len(points) < 3:
                    continue
                tracks[label].append(
                    {
                        "frame": int(float(polygon.attrib["frame"])),
                        "keyframe": polygon.attrib.get("keyframe", "0") in {"1", "true", "True"},
                        "points": resample_closed(points, points_per_polygon),
                        "z_order": int(float(polygon.attrib.get("z_order", "0"))),
                    }
                )
        for items in tracks.values():
            items.sort(key=lambda item: int(item["frame"]))
        return tracks, task_size, original_size
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


def interpolate_track(items: list[dict[str, Any]], frame_index: int, points_per_polygon: int) -> dict[str, Any] | None:
    if not items:
        return None
    if frame_index < int(items[0]["frame"]) or frame_index > int(items[-1]["frame"]):
        return None

    previous = items[0]
    for current in items:
        current_frame = int(current["frame"])
        if current_frame == frame_index:
            return current
        if current_frame > frame_index:
            previous_frame = int(previous["frame"])
            if current_frame == previous_frame:
                return previous
            t = (frame_index - previous_frame) / (current_frame - previous_frame)
            return {
                "frame": frame_index,
                "points": interpolate_points(previous["points"], current["points"], t, points_per_polygon),
                "z_order": int(current.get("z_order", previous.get("z_order", 0))),
            }
        previous = current
    return previous


def rasterize_frame(
    size: tuple[int, int],
    tracks: dict[str, list[dict[str, Any]]],
    frame_index: int,
    points_per_polygon: int,
) -> np.ndarray:
    width, height = size
    label = np.zeros((height, width), dtype=np.uint8)
    frame_polygons: list[tuple[int, int, list[tuple[float, float]]]] = []
    for class_name, items in tracks.items():
        polygon = interpolate_track(items, frame_index, points_per_polygon)
        if polygon is None:
            continue
        frame_polygons.append((int(polygon.get("z_order", 0)), CLASS_IDS[class_name], polygon["points"]))

    for _z_order, class_id, points in sorted(frame_polygons, key=lambda item: item[0]):
        mask = Image.new("L", size, 0)
        ImageDraw.Draw(mask).polygon([(float(x), float(y)) for x, y in points], fill=255)
        label[np.asarray(mask, dtype=np.uint8) > 0] = class_id

    if (label == CLASS_IDS["inner_mouth"]).any():
        label[label == CLASS_IDS["inner_mouth"]] = CLASS_IDS["inner_mouth"]
    return label


def label_to_overlay(image: Image.Image, label: np.ndarray) -> Image.Image:
    rgba = np.zeros((label.shape[0], label.shape[1], 4), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        rgba[label == class_id] = color
    overlay = Image.fromarray(rgba, "RGBA")
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def titled(image: Image.Image, text: str) -> Image.Image:
    output = Image.new("RGB", (image.width, image.height + 28), "white")
    output.paste(image.convert("RGB"), (0, 28))
    draw = ImageDraw.Draw(output)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    draw.text((6, 6), text, fill=(20, 20, 20), font=font)
    return output


def make_grid(images: list[Image.Image], columns: int) -> Image.Image:
    if not images:
        return Image.new("RGB", (1, 1), "white")
    cell_width = max(image.width for image in images)
    cell_height = max(image.height for image in images)
    rows = int(math.ceil(len(images) / columns))
    output = Image.new("RGB", (cell_width * columns, cell_height * rows), "white")
    for index, image in enumerate(images):
        output.paste(image, ((index % columns) * cell_width, (index // columns) * cell_height))
    return output


def crop_label_and_image(
    image: Image.Image,
    label: np.ndarray,
    *,
    width: int,
    height: int,
    padding_x: float,
    padding_y: float,
) -> tuple[Image.Image, Image.Image, tuple[int, int, int, int] | None]:
    bbox = bbox_from_mask(label > 0)
    if bbox is None:
        return image, Image.fromarray(label), None

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
    label_crop = Image.fromarray(label).crop(crop_box).resize((width, height), Image.Resampling.NEAREST)
    return image_crop, label_crop, crop_box


def split_for_sample(sample_index: int) -> str:
    bucket = sample_index % 10
    if bucket == 0:
        return "test"
    if bucket == 5:
        return "val"
    return "train"


def parse_ranges(value: str | None) -> list[tuple[int, int]]:
    if value is None or not value.strip():
        return []
    ranges: list[tuple[int, int]] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start = int(start_text)
            end = int(end_text)
        else:
            start = end = int(item)
        if end < start:
            start, end = end, start
        ranges.append((start, end))
    return ranges


def in_ranges(frame_index: int, ranges: list[tuple[int, int]]) -> bool:
    return any(start <= frame_index <= end for start, end in ranges)


def prepare_dataset(
    *,
    video: Path,
    annotations: Path,
    output_root: Path,
    image_width: int,
    image_height: int,
    frame_step: int,
    points_per_polygon: int,
    padding_x: float,
    padding_y: float,
    preview_count: int,
    include_ranges: list[tuple[int, int]],
    exclude_ranges: list[tuple[int, int]],
    overwrite: bool,
) -> dict[str, Any]:
    if output_root.exists() and overwrite:
        shutil.rmtree(output_root)
    if output_root.exists():
        raise FileExistsError(output_root)
    for split in ("train", "val", "test"):
        (output_root / split / "images").mkdir(parents=True, exist_ok=True)
        (output_root / split / "labels").mkdir(parents=True, exist_ok=True)
    labels_source = Path(__file__).with_name("lip_labels.json")
    if labels_source.exists():
        shutil.copy2(labels_source, output_root / "lip_labels.json")

    tracks, task_size, original_size = read_tracks(annotations, points_per_polygon=points_per_polygon)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open video {video}")
    video_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    video_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = original_size or (video_width, video_height)
    frame_count = min(value for value in (task_size, video_frame_count) if value)

    selected_frames = set(range(0, frame_count, frame_step))
    for items in tracks.values():
        selected_frames.update(
            int(item["frame"])
            for item in items
            if item.get("keyframe", False) and int(item["frame"]) < frame_count
        )
    if include_ranges:
        selected_frames = {frame for frame in selected_frames if in_ranges(frame, include_ranges)}
    if exclude_ranges:
        selected_frames = {frame for frame in selected_frames if not in_ranges(frame, exclude_ranges)}
    selected = sorted(selected_frames)

    metadata_files = {
        split: (output_root / split / "metadata.jsonl").open("w", encoding="utf-8")
        for split in ("train", "val", "test")
    }
    counts = {split: 0 for split in ("train", "val", "test")}
    previews: list[Image.Image] = []
    skipped = 0
    sample_index = 0

    try:
        selected_iter = iter(selected)
        next_selected = next(selected_iter, None)
        frame_index = 0
        while next_selected is not None and frame_index < frame_count:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            if frame_index != next_selected:
                frame_index += 1
                continue

            image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            label = rasterize_frame(size, tracks, frame_index, points_per_polygon)
            if image.size != size:
                label_image = Image.fromarray(label).resize(image.size, Image.Resampling.NEAREST)
                label = np.asarray(label_image, dtype=np.uint8)

            if not (label > 0).any():
                skipped += 1
                next_selected = next(selected_iter, None)
                frame_index += 1
                continue

            image_crop, label_crop, crop_box = crop_label_and_image(
                image,
                label,
                width=image_width,
                height=image_height,
                padding_x=padding_x,
                padding_y=padding_y,
            )
            if crop_box is None:
                skipped += 1
                next_selected = next(selected_iter, None)
                frame_index += 1
                continue

            split = split_for_sample(sample_index)
            stem = f"lip001_f{frame_index:06d}"
            image_path = output_root / split / "images" / f"{stem}.jpg"
            label_path = output_root / split / "labels" / f"{stem}.png"
            image_crop.save(image_path, quality=96)
            label_crop.save(label_path)
            metadata_files[split].write(
                json.dumps(
                    {
                        "stem": stem,
                        "source_video": str(video),
                        "source_annotations": str(annotations),
                        "frame": frame_index,
                        "crop_box": list(crop_box),
                        "points_per_polygon": points_per_polygon,
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )
            counts[split] += 1

            if len(previews) < preview_count:
                overlay = label_to_overlay(image_crop, np.asarray(label_crop, dtype=np.uint8))
                panel = Image.new("RGB", (image_width * 2, image_height + 28), "white")
                panel.paste(titled(image_crop, f"frame {frame_index}").crop((0, 0, image_width, image_height + 28)), (0, 0))
                panel.paste(titled(overlay, "label").crop((0, 0, image_width, image_height + 28)), (image_width, 0))
                previews.append(panel)

            sample_index += 1
            next_selected = next(selected_iter, None)
            frame_index += 1
    finally:
        capture.release()
        for handle in metadata_files.values():
            handle.close()

    preview_path = output_root / "preview_sheet.jpg"
    make_grid(previews, columns=3).save(preview_path, quality=94)

    summary = {
        "video": str(video),
        "annotations": str(annotations),
        "output_root": str(output_root),
        "video_frame_count": video_frame_count,
        "task_size": task_size,
        "selected_frames": len(selected),
        "saved": counts,
        "skipped": skipped,
        "image_width": image_width,
        "image_height": image_height,
        "frame_step": frame_step,
        "points_per_polygon": points_per_polygon,
        "include_ranges": include_ranges,
        "exclude_ranges": exclude_ranges,
        "preview_sheet": str(preview_path),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert CVAT video lip polygon tracks into a lip ROI dataset.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--annotations", required=True, help="CVAT 1.1 exported zip/xml with polygon tracks.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--image-width", type=int, default=192)
    parser.add_argument("--image-height", type=int, default=128)
    parser.add_argument("--frame-step", type=int, default=2)
    parser.add_argument("--points-per-polygon", type=int, default=24)
    parser.add_argument("--padding-x", type=float, default=0.42)
    parser.add_argument("--padding-y", type=float, default=0.50)
    parser.add_argument("--include-ranges", default=None, help="Optional comma-separated frame ranges, e.g. 0-320,540-760")
    parser.add_argument("--exclude-ranges", default=None, help="Optional comma-separated frame ranges to drop.")
    parser.add_argument("--preview-count", type=int, default=36)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    summary = prepare_dataset(
        video=Path(args.video).expanduser(),
        annotations=Path(args.annotations).expanduser(),
        output_root=Path(args.output_root).expanduser(),
        image_width=args.image_width,
        image_height=args.image_height,
        frame_step=max(1, args.frame_step),
        points_per_polygon=max(8, args.points_per_polygon),
        padding_x=args.padding_x,
        padding_y=args.padding_y,
        include_ranges=parse_ranges(args.include_ranges),
        exclude_ranges=parse_ranges(args.exclude_ranges),
        preview_count=args.preview_count,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
