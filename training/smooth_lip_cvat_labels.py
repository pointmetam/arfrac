from __future__ import annotations

import argparse
import json
import math
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw, ImageFont


CLASS_NAMES = {
    1: "upper_lip",
    2: "lower_lip",
    3: "inner_mouth",
}

CLASS_COLORS = {
    1: (236, 36, 82, 185),
    2: (255, 92, 128, 185),
    3: (24, 24, 28, 230),
}

KEYPOINT_NAMES = (
    "left_corner",
    "right_corner",
    "upper_left_peak",
    "upper_cupid_bow",
    "upper_right_peak",
    "upper_outer_mid",
    "lower_left_arc",
    "lower_right_arc",
    "lower_outer_mid",
    "inner_left_corner",
    "inner_right_corner",
    "inner_upper_mid",
    "inner_lower_mid",
)


def load_label(path: Path, size: tuple[int, int]) -> np.ndarray:
    label = Image.open(path).convert("L")
    if label.size != size:
        label = label.resize(size, Image.Resampling.NEAREST)
    return np.asarray(label, dtype=np.uint8)


def smooth_1d(values: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    if window <= 1 or values.size < 3:
        return values.astype(np.float64)
    if window % 2 == 0:
        window += 1
    window = min(window, values.size if values.size % 2 == 1 else values.size - 1)
    if window <= 1:
        return values.astype(np.float64)
    pad = window // 2
    padded = np.pad(values.astype(np.float64), (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return mask

    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    cropped = mask[y0:y1, x0:x1].astype(bool)
    visited = np.zeros(cropped.shape, dtype=bool)
    best_pixels: list[tuple[int, int]] = []
    height, width = cropped.shape

    for start_y, start_x in zip(*np.nonzero(cropped & ~visited)):
        if visited[start_y, start_x]:
            continue
        stack = [(int(start_y), int(start_x))]
        visited[start_y, start_x] = True
        pixels: list[tuple[int, int]] = []
        while stack:
            y, x = stack.pop()
            pixels.append((y, x))
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < height and 0 <= nx < width and cropped[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        if len(pixels) > len(best_pixels):
            best_pixels = pixels

    output = np.zeros_like(mask, dtype=bool)
    if best_pixels:
        yy = np.array([point[0] + y0 for point in best_pixels], dtype=np.int64)
        xx = np.array([point[1] + x0 for point in best_pixels], dtype=np.int64)
        output[yy, xx] = True
    return output


def mask_to_smooth_polygon(
    mask: np.ndarray,
    points_per_edge: int,
    smooth_window: int,
    min_points: int = 8,
) -> list[tuple[float, float]]:
    mask = keep_largest_component(mask.astype(bool))
    if int(mask.sum()) < min_points:
        return []

    ys, xs = np.nonzero(mask)
    unique_x = np.unique(xs)
    if unique_x.size < 4:
        return []

    top: list[float] = []
    bottom: list[float] = []
    columns: list[int] = []
    for x in unique_x:
        col_ys = ys[xs == x]
        if col_ys.size == 0:
            continue
        columns.append(int(x))
        top.append(float(col_ys.min()))
        bottom.append(float(col_ys.max()))

    x_values = np.asarray(columns, dtype=np.float64)
    top_values = smooth_1d(np.asarray(top, dtype=np.float64), smooth_window)
    bottom_values = smooth_1d(np.asarray(bottom, dtype=np.float64), smooth_window)
    count = int(min(max(points_per_edge, 6), x_values.size))
    sample_indices = np.linspace(0, x_values.size - 1, count).round().astype(np.int64)
    sample_indices = np.unique(sample_indices)

    top_points = [(float(x_values[i]), float(top_values[i])) for i in sample_indices]
    bottom_points = [(float(x_values[i]), float(bottom_values[i])) for i in sample_indices[::-1]]
    polygon = top_points + bottom_points
    if len(polygon) < min_points:
        return []
    return polygon


def polygon_to_mask(points: list[tuple[float, float]], size: tuple[int, int]) -> np.ndarray:
    image = Image.new("L", size, 0)
    if points:
        ImageDraw.Draw(image).polygon([(float(x), float(y)) for x, y in points], fill=255)
    return np.asarray(image, dtype=np.uint8) > 0


def _mask_columns(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    order = np.argsort(xs)
    return xs[order].astype(np.int64), ys[order].astype(np.int64)


def _column_point(mask: np.ndarray, x: float, mode: str) -> tuple[float, float] | None:
    xs, ys = _mask_columns(mask)
    if xs.size == 0:
        return None
    column_values = np.unique(xs)
    nearest_x = int(column_values[np.argmin(np.abs(column_values.astype(np.float64) - float(x)))])
    column_ys = ys[xs == nearest_x]
    if column_ys.size == 0:
        return None
    if mode == "top":
        y = float(column_ys.min())
    elif mode == "bottom":
        y = float(column_ys.max())
    else:
        y = float((column_ys.min() + column_ys.max()) / 2.0)
    return (float(nearest_x), y)


def _extreme_point(mask: np.ndarray, side: str) -> tuple[float, float] | None:
    xs, _ = _mask_columns(mask)
    if xs.size == 0:
        return None
    x = float(xs.min() if side == "left" else xs.max())
    return _column_point(mask, x, "middle")


def _top_point_in_range(mask: np.ndarray, start_fraction: float, end_fraction: float) -> tuple[float, float] | None:
    xs, ys = _mask_columns(mask)
    if xs.size == 0:
        return None
    x_min = float(xs.min())
    x_max = float(xs.max())
    lo = x_min + (x_max - x_min) * start_fraction
    hi = x_min + (x_max - x_min) * end_fraction
    candidates: list[tuple[float, float]] = []
    for x in np.unique(xs):
        if lo <= float(x) <= hi:
            column_ys = ys[xs == x]
            candidates.append((float(x), float(column_ys.min())))
    if not candidates:
        return _column_point(mask, (lo + hi) / 2.0, "top")
    return min(candidates, key=lambda point: (point[1], abs(point[0] - ((lo + hi) / 2.0))))


def _fractional_point(mask: np.ndarray, fraction: float, mode: str) -> tuple[float, float] | None:
    xs, _ = _mask_columns(mask)
    if xs.size == 0:
        return None
    x = float(xs.min()) + (float(xs.max()) - float(xs.min())) * fraction
    return _column_point(mask, x, mode)


def extract_lip_keypoints(label: np.ndarray) -> dict[str, tuple[float, float]]:
    label = np.asarray(label, dtype=np.uint8)
    upper = label == 1
    lower = label == 2
    inner = label == 3
    outer = upper | lower

    points: dict[str, tuple[float, float]] = {}

    left_corner = _extreme_point(outer, "left")
    right_corner = _extreme_point(outer, "right")
    if left_corner is not None:
        points["left_corner"] = left_corner
    if right_corner is not None:
        points["right_corner"] = right_corner

    upper_left_peak = _top_point_in_range(upper, 0.22, 0.48)
    upper_cupid_bow = _fractional_point(upper, 0.50, "top")
    upper_right_peak = _top_point_in_range(upper, 0.52, 0.78)
    upper_outer_mid = _fractional_point(upper, 0.50, "middle")
    for name, point in (
        ("upper_left_peak", upper_left_peak),
        ("upper_cupid_bow", upper_cupid_bow),
        ("upper_right_peak", upper_right_peak),
        ("upper_outer_mid", upper_outer_mid),
    ):
        if point is not None:
            points[name] = point

    lower_left_arc = _fractional_point(lower, 0.30, "bottom")
    lower_right_arc = _fractional_point(lower, 0.70, "bottom")
    lower_outer_mid = _fractional_point(lower, 0.50, "bottom")
    for name, point in (
        ("lower_left_arc", lower_left_arc),
        ("lower_right_arc", lower_right_arc),
        ("lower_outer_mid", lower_outer_mid),
    ):
        if point is not None:
            points[name] = point

    if inner.any():
        inner_points = {
            "inner_left_corner": _extreme_point(inner, "left"),
            "inner_right_corner": _extreme_point(inner, "right"),
            "inner_upper_mid": _fractional_point(inner, 0.50, "top"),
            "inner_lower_mid": _fractional_point(inner, 0.50, "bottom"),
        }
    else:
        center = _fractional_point(outer, 0.50, "middle")
        left = left_corner
        right = right_corner
        inner_points = {
            "inner_left_corner": left,
            "inner_right_corner": right,
            "inner_upper_mid": center,
            "inner_lower_mid": center,
        }
    for name, point in inner_points.items():
        if point is not None:
            points[name] = point

    return points


def label_to_rgba(label: np.ndarray) -> Image.Image:
    rgba = np.zeros((label.shape[0], label.shape[1], 4), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        rgba[label == class_id] = color
    return Image.fromarray(rgba)


def overlay_label(image: Image.Image, label: np.ndarray) -> Image.Image:
    return Image.alpha_composite(image.convert("RGBA"), label_to_rgba(label)).convert("RGB")


def draw_polygons(image: Image.Image, polygons: dict[int, list[tuple[float, float]]]) -> Image.Image:
    output = image.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for class_id, points in polygons.items():
        if len(points) < 3:
            continue
        fill = CLASS_COLORS[class_id]
        line = fill[:3] + (255,)
        draw.polygon(points, fill=fill)
        draw.line(points + [points[0]], fill=line, width=3)
    return Image.alpha_composite(output, overlay).convert("RGB")


def draw_keypoints(image: Image.Image, keypoints: dict[str, tuple[float, float]]) -> Image.Image:
    output = image.convert("RGBA")
    draw = ImageDraw.Draw(output)
    radius = max(4, int(round(min(image.size) * 0.006)))
    for name, (x, y) in keypoints.items():
        color = (255, 230, 48, 255) if name.startswith("inner_") else (70, 220, 255, 255)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=(20, 20, 20, 255), width=1)
    return output.convert("RGB")


def title(image: Image.Image, text: str) -> Image.Image:
    output = Image.new("RGB", (image.width, image.height + 28), "white")
    output.paste(image.convert("RGB"), (0, 28))
    draw = ImageDraw.Draw(output)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    draw.text((6, 6), text, fill=(20, 20, 20), font=font)
    return output


def hstack(images: list[Image.Image]) -> Image.Image:
    output = Image.new("RGB", (sum(image.width for image in images), max(image.height for image in images)), "white")
    x = 0
    for image in images:
        output.paste(image.convert("RGB"), (x, 0))
        x += image.width
    return output


def fit_width(image: Image.Image, width: int) -> Image.Image:
    if image.width == width:
        return image
    height = max(1, int(round(image.height * width / image.width)))
    return image.resize((width, height), Image.Resampling.BILINEAR)


def make_grid(images: list[Image.Image], columns: int) -> Image.Image:
    if not images:
        return Image.new("RGB", (1, 1), "white")
    cell_w = max(image.width for image in images)
    cell_h = max(image.height for image in images)
    rows = int(math.ceil(len(images) / columns))
    output = Image.new("RGB", (cell_w * columns, cell_h * rows), "white")
    for index, image in enumerate(images):
        output.paste(image.convert("RGB"), ((index % columns) * cell_w, (index // columns) * cell_h))
    return output


def polygon_points_attribute(points: list[tuple[float, float]]) -> str:
    return ";".join(f"{x:.2f},{y:.2f}" for x, y in points)


def build_cvat_xml(
    task_name: str,
    images: list[dict[str, Any]],
    labels: dict[str, dict[int, list[tuple[float, float]]]],
    keypoints_by_stem: dict[str, dict[str, tuple[float, float]]] | None = None,
) -> ET.ElementTree:
    annotations = ET.Element("annotations")
    ET.SubElement(annotations, "version").text = "1.1"
    meta = ET.SubElement(annotations, "meta")
    task = ET.SubElement(meta, "task")
    ET.SubElement(task, "id").text = "0"
    ET.SubElement(task, "name").text = task_name
    ET.SubElement(task, "size").text = str(len(images))
    ET.SubElement(task, "mode").text = "annotation"
    ET.SubElement(task, "overlap").text = "0"
    ET.SubElement(task, "bugtracker").text = ""
    ET.SubElement(task, "flipped").text = "False"
    now = datetime.now(timezone.utc).isoformat()
    ET.SubElement(task, "created").text = now
    ET.SubElement(task, "updated").text = now

    labels_node = ET.SubElement(task, "labels")
    for class_name in CLASS_NAMES.values():
        label_node = ET.SubElement(labels_node, "label")
        ET.SubElement(label_node, "name").text = class_name
        ET.SubElement(label_node, "type").text = "polygon"
        ET.SubElement(label_node, "attributes")
    if keypoints_by_stem is not None:
        for keypoint_name in KEYPOINT_NAMES:
            label_node = ET.SubElement(labels_node, "label")
            ET.SubElement(label_node, "name").text = keypoint_name
            ET.SubElement(label_node, "type").text = "points"
            ET.SubElement(label_node, "attributes")

    segments = ET.SubElement(task, "segments")
    segment = ET.SubElement(segments, "segment")
    ET.SubElement(segment, "id").text = "0"
    ET.SubElement(segment, "start").text = "0"
    ET.SubElement(segment, "stop").text = str(max(len(images) - 1, 0))
    ET.SubElement(segment, "url").text = ""

    owner = ET.SubElement(task, "owner")
    ET.SubElement(owner, "username").text = "codex"
    ET.SubElement(owner, "email").text = ""
    ET.SubElement(meta, "dumped").text = now

    for index, item in enumerate(images):
        image_node = ET.SubElement(
            annotations,
            "image",
            {
                "id": str(index),
                "name": item["name"],
                "width": str(item["width"]),
                "height": str(item["height"]),
            },
        )
        for z_order, class_id in enumerate((1, 2, 3), start=1):
            points = labels[item["stem"]].get(class_id, [])
            if len(points) < 3:
                continue
            ET.SubElement(
                image_node,
                "polygon",
                {
                    "label": CLASS_NAMES[class_id],
                    "source": "auto",
                    "points": polygon_points_attribute(points),
                    "occluded": "0",
                    "z_order": str(z_order),
                },
            )
        if keypoints_by_stem is not None:
            keypoints = keypoints_by_stem.get(item["stem"], {})
            for z_order, keypoint_name in enumerate(KEYPOINT_NAMES, start=10):
                point = keypoints.get(keypoint_name)
                if point is None:
                    continue
                ET.SubElement(
                    image_node,
                    "points",
                    {
                        "label": keypoint_name,
                        "source": "auto",
                        "points": f"{point[0]:.2f},{point[1]:.2f}",
                        "occluded": "0",
                        "z_order": str(z_order),
                    },
                )
    return ET.ElementTree(annotations)


def write_xml(tree: ET.ElementTree, path: Path) -> None:
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)


def smooth_labels(
    image_dir: Path,
    source_combined_dir: Path,
    output_dir: Path,
    task_name: str,
    points_per_edge: int,
    smooth_window: int,
    panel_width: int,
    overwrite: bool,
    add_keypoints: bool = False,
) -> dict[str, Any]:
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)

    masks_dir = output_dir / "masks"
    combined_dir = output_dir / "combined_masks"
    previews_dir = output_dir / "previews"
    cvat_dir = output_dir / "cvat"
    for class_name in CLASS_NAMES.values():
        (masks_dir / class_name).mkdir(parents=True, exist_ok=True)
    combined_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)
    cvat_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(image_dir.glob("*.jpg"))
    images_meta: list[dict[str, Any]] = []
    polygons_by_stem: dict[str, dict[int, list[tuple[float, float]]]] = {}
    keypoints_by_stem: dict[str, dict[str, tuple[float, float]]] = {}
    preview_panels: list[Image.Image] = []

    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        source_path = source_combined_dir / f"{image_path.stem}.png"
        if not source_path.exists():
            continue
        source_label = load_label(source_path, image.size)
        combined = np.zeros((image.height, image.width), dtype=np.uint8)
        polygons: dict[int, list[tuple[float, float]]] = {}
        for class_id, class_name in CLASS_NAMES.items():
            class_mask = source_label == class_id
            points = mask_to_smooth_polygon(
                class_mask,
                points_per_edge=points_per_edge,
                smooth_window=smooth_window,
            )
            polygons[class_id] = points
            binary = polygon_to_mask(points, image.size)
            Image.fromarray((binary.astype(np.uint8) * 255)).save(masks_dir / class_name / f"{image_path.stem}.png")
            combined[binary] = class_id

        # Inner mouth should stay visible where it overlaps lips.
        inner = polygon_to_mask(polygons.get(3, []), image.size)
        combined[inner] = 3
        keypoints = extract_lip_keypoints(combined) if add_keypoints else {}
        Image.fromarray(combined).save(combined_dir / f"{image_path.stem}.png")
        polygons_by_stem[image_path.stem] = polygons
        keypoints_by_stem[image_path.stem] = keypoints
        images_meta.append(
            {
                "stem": image_path.stem,
                "name": image_path.name,
                "width": image.width,
                "height": image.height,
                "image": str(image_path),
            }
        )

        preview = hstack(
            [
                title(image, "image"),
                title(overlay_label(image, source_label), "old mask"),
                title(draw_keypoints(draw_polygons(image, polygons), keypoints), "smooth relabel"),
            ]
        )
        preview = fit_width(preview, panel_width)
        preview.save(previews_dir / f"preview_{image_path.stem}.jpg", quality=94)
        preview_panels.append(preview)

    contact_sheet = previews_dir / "preview_sheet.jpg"
    make_grid(preview_panels, columns=2).save(contact_sheet, quality=94)

    annotations_xml = cvat_dir / "annotations.xml"
    tree = build_cvat_xml(
        task_name,
        images_meta,
        polygons_by_stem,
        keypoints_by_stem=keypoints_by_stem if add_keypoints else None,
    )
    write_xml(tree, annotations_xml)

    zip_path = cvat_dir / f"{task_name}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(annotations_xml, "annotations.xml")
        for item in images_meta:
            zf.write(Path(item["image"]), f"images/{Path(item['image']).name}")

    summary = {
        "image_dir": str(image_dir),
        "source_combined_dir": str(source_combined_dir),
        "output_dir": str(output_dir),
        "task_name": task_name,
        "count": len(images_meta),
        "points_per_edge": points_per_edge,
        "smooth_window": smooth_window,
        "add_keypoints": add_keypoints,
        "keypoint_names": list(KEYPOINT_NAMES) if add_keypoints else [],
        "masks_dir": str(masks_dir),
        "combined_dir": str(combined_dir),
        "previews_dir": str(previews_dir),
        "preview_sheet": str(contact_sheet),
        "cvat_xml": str(annotations_xml),
        "cvat_zip": str(zip_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smooth rough lip masks into CVAT polygon labels.")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--source-combined-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-name", default="arface_lip_uv_smooth")
    parser.add_argument("--points-per-edge", type=int, default=28)
    parser.add_argument("--smooth-window", type=int, default=15)
    parser.add_argument("--panel-width", type=int, default=1200)
    parser.add_argument("--add-keypoints", action="store_true", help="Add auto lip point labels to the CVAT XML.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = smooth_labels(
        image_dir=Path(args.image_dir).expanduser(),
        source_combined_dir=Path(args.source_combined_dir).expanduser(),
        output_dir=Path(args.output_dir).expanduser(),
        task_name=args.task_name,
        points_per_edge=args.points_per_edge,
        smooth_window=args.smooth_window,
        panel_width=args.panel_width,
        overwrite=args.overwrite,
        add_keypoints=args.add_keypoints,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
