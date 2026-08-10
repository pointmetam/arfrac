from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from smooth_lip_cvat_labels import smooth_labels
from teacher_2d_to_arkit_uv import portrait_image, render_uv_label_to_portrait


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def source_path(row: dict[str, Any], key: str) -> Path:
    path = Path(str(row[key])).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Missing {key}: {path}")
    return path


def metadata_path(row: dict[str, Any], key: str) -> Path | None:
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        return None
    value = metadata.get(key)
    if value is None:
        return None
    path = Path(str(value)).expanduser()
    return path if path.exists() else None


def candidate_path(row: dict[str, Any], key: str) -> Path:
    path = metadata_path(row, f"source_{key}")
    if path is not None:
        return path
    return source_path(row, key)


def frame_viewport_size(frame: dict[str, Any]) -> tuple[float, float]:
    viewport = frame.get("viewport")
    if not isinstance(viewport, dict):
        raise ValueError("Frame metadata is missing viewport")
    return float(viewport["width"]), float(viewport["height"])


def portrait_scale_for_target_long_edge(
    raw_image_size: tuple[int, int],
    frame: dict[str, Any],
    target_long_edge: int,
) -> float:
    viewport_width, viewport_height = frame_viewport_size(frame)
    if viewport_width <= 0 or viewport_height <= 0:
        raise ValueError(f"Invalid viewport size: {(viewport_width, viewport_height)}")

    target_scale = float(target_long_edge) / max(viewport_width, viewport_height)
    raw_width, raw_height = raw_image_size
    max_scale_without_upscaling = min(raw_width / viewport_width, raw_height / viewport_height)
    return max(1.0, min(target_scale, max_scale_without_upscaling))


def clamp_interval(start: int, end: int, limit: int) -> tuple[int, int]:
    length = end - start
    if length >= limit:
        return 0, limit
    if start < 0:
        end -= start
        start = 0
    if end > limit:
        start -= end - limit
        end = limit
    return max(0, start), min(limit, end)


def expanded_bbox_from_mask(
    mask: np.ndarray,
    padding: int,
    min_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape[:2]
    ys, xs = np.nonzero(mask)
    if xs.size == 0 or ys.size == 0:
        center_x = width // 2
        center_y = int(height * 0.58)
        half_width = min_size[0] // 2
        half_height = min_size[1] // 2
        x0, x1 = clamp_interval(center_x - half_width, center_x + half_width, width)
        y0, y1 = clamp_interval(center_y - half_height, center_y + half_height, height)
        return (x0, y0, x1, y1)

    x0 = int(xs.min()) - padding
    y0 = int(ys.min()) - padding
    x1 = int(xs.max()) + 1 + padding
    y1 = int(ys.max()) + 1 + padding

    min_width, min_height = min_size
    center_x = (x0 + x1) / 2.0
    center_y = (y0 + y1) / 2.0
    box_width = max(x1 - x0, min_width)
    box_height = max(y1 - y0, min_height)
    x0 = int(np.floor(center_x - box_width / 2.0))
    x1 = int(np.ceil(center_x + box_width / 2.0))
    y0 = int(np.floor(center_y - box_height / 2.0))
    y1 = int(np.ceil(center_y + box_height / 2.0))

    x0, x1 = clamp_interval(x0, x1, width)
    y0, y1 = clamp_interval(y0, y1, height)
    return (x0, y0, x1, y1)


def crop_array(array: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    return array[y0:y1, x0:x1]


def prepare_workspace(
    manifest: Path,
    output_dir: Path,
    task_name: str,
    limit: int | None,
    points_per_edge: int,
    smooth_window: int,
    panel_width: int,
    portrait_scale: float,
    portrait_target_long_edge: int | None,
    mouth_crop: bool,
    crop_padding: int,
    min_crop_width: int,
    min_crop_height: int,
    add_keypoints: bool,
    require_source_image: bool,
    overwrite: bool,
) -> dict[str, Any]:
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)

    images_dir = output_dir / "images"
    frames_dir = output_dir / "frames"
    uv_labels_dir = output_dir / "source_uv_labels"
    projected_dir = output_dir / "projected_combined_masks"
    for directory in (images_dir, frames_dir, uv_labels_dir, projected_dir):
        directory.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(manifest)
    if limit is not None:
        rows = rows[:limit]

    exported_rows: list[dict[str, Any]] = []
    used_source_image_count = 0
    missing_source_image_count = 0
    for index, row in enumerate(rows):
        stem = str(row["stem"])
        source_image_path = metadata_path(row, "source_image")
        metadata = row.get("metadata")
        expected_source_image = metadata.get("source_image") if isinstance(metadata, dict) else None
        if source_image_path is None and expected_source_image is not None:
            missing_source_image_count += 1
            if require_source_image:
                raise FileNotFoundError(f"Missing source_image for {stem}: {expected_source_image}")
        image_path = source_image_path if source_image_path is not None else source_path(row, "image")
        source_frame_path = metadata_path(row, "source_frame")
        frame_path = source_frame_path if source_frame_path is not None else source_path(row, "frame")
        uv_label_path = source_path(row, "uv_label")

        frame = json.loads(frame_path.read_text(encoding="utf-8"))
        raw_image = Image.open(image_path).convert("RGB")
        if source_image_path is not None:
            used_source_image_count += 1
            effective_portrait_scale = (
                portrait_scale_for_target_long_edge(raw_image.size, frame, portrait_target_long_edge)
                if portrait_target_long_edge and portrait_target_long_edge > 0
                else portrait_scale
            )
            image = portrait_image(raw_image, frame, scale=effective_portrait_scale)
        elif portrait_scale != 1.0:
            effective_portrait_scale = portrait_scale
            image = raw_image.resize(
                (
                    max(1, int(round(raw_image.width * portrait_scale))),
                    max(1, int(round(raw_image.height * portrait_scale))),
                ),
                Image.Resampling.BILINEAR,
            )
        else:
            effective_portrait_scale = portrait_scale
            image = raw_image
        uv_label = np.asarray(Image.open(uv_label_path).convert("L"), dtype=np.uint8)
        projected = render_uv_label_to_portrait(frame, uv_label, image.size)
        full_portrait_size = [image.width, image.height]
        crop_box: tuple[int, int, int, int] | None = None
        if mouth_crop:
            crop_box = expanded_bbox_from_mask(
                projected > 0,
                padding=crop_padding,
                min_size=(min_crop_width, min_crop_height),
            )
            image = image.crop(crop_box)
            projected = crop_array(projected, crop_box)

        output_image = images_dir / f"{stem}.jpg"
        output_frame = frames_dir / f"{stem}.json"
        output_uv_label = uv_labels_dir / f"{stem}.png"
        output_projected = projected_dir / f"{stem}.png"
        image.save(output_image, quality=95)
        shutil.copy2(frame_path, output_frame)
        shutil.copy2(uv_label_path, output_uv_label)
        Image.fromarray(projected).save(output_projected)

        payload = dict(row)
        payload.update(
            {
                "index": index,
                "image": str(output_image),
                "frame": str(output_frame),
                "source_uv_label_copy": str(output_uv_label),
                "projected_combined_mask": str(output_projected),
                "mask_space": "portrait",
                "source_image_for_annotation": str(image_path),
                "source_frame_for_annotation": str(frame_path),
                "portrait_scale": portrait_scale,
                "effective_portrait_scale": effective_portrait_scale,
                "portrait_target_long_edge": portrait_target_long_edge,
                "full_portrait_size": full_portrait_size,
                "crop_box_xyxy": list(crop_box) if crop_box is not None else None,
                "annotation_image_size": [image.width, image.height],
                "prelabel_source": "arkit_uv_teacher_projection",
            }
        )
        exported_rows.append(payload)

    write_jsonl(output_dir / "manifest.jsonl", exported_rows)

    cvat_output = output_dir / "cvat_prelabeled_polygons"
    smooth_summary = smooth_labels(
        image_dir=images_dir,
        source_combined_dir=projected_dir,
        output_dir=cvat_output,
        task_name=task_name,
        points_per_edge=points_per_edge,
        smooth_window=smooth_window,
        panel_width=panel_width,
        overwrite=False,
        add_keypoints=add_keypoints,
    )

    summary = {
        "manifest": str(manifest),
        "output_dir": str(output_dir),
        "task_name": task_name,
        "count": len(exported_rows),
        "images_dir": str(images_dir),
        "frames_dir": str(frames_dir),
        "source_uv_labels_dir": str(uv_labels_dir),
        "projected_combined_masks_dir": str(projected_dir),
        "portrait_scale": portrait_scale,
        "portrait_target_long_edge": portrait_target_long_edge,
        "mouth_crop": mouth_crop,
        "crop_padding": crop_padding,
        "min_crop_width": min_crop_width,
        "min_crop_height": min_crop_height,
        "add_keypoints": add_keypoints,
        "require_source_image": require_source_image,
        "used_source_image_count": used_source_image_count,
        "missing_source_image_count": missing_source_image_count,
        "cvat_zip": smooth_summary["cvat_zip"],
        "cvat_xml": smooth_summary["cvat_xml"],
        "preview_sheet": smooth_summary["preview_sheet"],
        "stems": [row["stem"] for row in exported_rows],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare CVAT lip annotation package from ARKit UV candidate frames.")
    parser.add_argument("--manifest", required=True, help="candidates.jsonl from select_arkit_uv_hard_frames.py")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-name", default="arface_lip_manual48_new36")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--points-per-edge", type=int, default=30)
    parser.add_argument("--smooth-window", type=int, default=17)
    parser.add_argument("--panel-width", type=int, default=1400)
    parser.add_argument("--portrait-scale", type=float, default=1.0, help="Scale applied when rebuilding portrait images from raw camera frames.")
    parser.add_argument(
        "--portrait-target-long-edge",
        type=int,
        default=None,
        help="When raw source images are available, choose a portrait scale whose long edge approaches this value without upscaling the source.",
    )
    parser.add_argument("--mouth-crop", action="store_true", help="Export a mouth-only crop instead of the full portrait frame.")
    parser.add_argument("--crop-padding", type=int, default=220, help="Padding in scaled portrait pixels around the projected lip mask.")
    parser.add_argument("--min-crop-width", type=int, default=900)
    parser.add_argument("--min-crop-height", type=int, default=620)
    parser.add_argument("--add-keypoints", action="store_true", help="Add auto lip point labels to the CVAT prelabels.")
    parser.add_argument("--require-source-image", action="store_true", help="Fail if candidate metadata references a missing raw recording image.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = prepare_workspace(
        manifest=Path(args.manifest).expanduser(),
        output_dir=Path(args.output_dir).expanduser(),
        task_name=args.task_name,
        limit=args.limit,
        points_per_edge=args.points_per_edge,
        smooth_window=args.smooth_window,
        panel_width=args.panel_width,
        portrait_scale=args.portrait_scale,
        portrait_target_long_edge=args.portrait_target_long_edge,
        mouth_crop=args.mouth_crop,
        crop_padding=args.crop_padding,
        min_crop_width=args.min_crop_width,
        min_crop_height=args.min_crop_height,
        add_keypoints=args.add_keypoints,
        require_source_image=args.require_source_image,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
