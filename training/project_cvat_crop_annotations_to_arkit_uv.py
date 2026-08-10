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

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from teacher_2d_to_arkit_uv import label_to_rgba, overlay_label, portrait_image, rasterize_2d_label_to_uv


CLASS_IDS = {
    "upper_lip": 1,
    "lower_lip": 2,
    "inner_mouth": 3,
}


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


def annotations_xml_path(path: Path) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    path = path.expanduser()
    if path.suffix.lower() == ".xml":
        return path, None
    if path.suffix.lower() != ".zip":
        raise ValueError(f"Expected .zip or .xml annotations, got {path}")
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
        x_text, y_text = item.split(",", maxsplit=1)
        points.append((float(x_text), float(y_text)))
    return points


def parse_cvat_labels(annotations_path: Path) -> dict[str, dict[str, Any]]:
    xml_path, temp_dir = annotations_xml_path(annotations_path)
    try:
        tree = ET.parse(xml_path)
        labels_by_stem: dict[str, dict[str, Any]] = {}
        for image_node in tree.getroot().findall("image"):
            name = str(image_node.attrib["name"])
            stem = Path(name).stem
            width = int(float(image_node.attrib["width"]))
            height = int(float(image_node.attrib["height"]))
            polygons: list[dict[str, Any]] = []
            for polygon_node in image_node.findall("polygon"):
                label = str(polygon_node.attrib.get("label", ""))
                if label not in CLASS_IDS:
                    continue
                polygons.append(
                    {
                        "label": label,
                        "class_id": CLASS_IDS[label],
                        "z_order": int(float(polygon_node.attrib.get("z_order", "0"))),
                        "points": parse_points(str(polygon_node.attrib.get("points", ""))),
                    }
                )
            labels_by_stem[stem] = {"name": name, "width": width, "height": height, "polygons": polygons}
        return labels_by_stem
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


def rasterize_crop_label(width: int, height: int, polygons: list[dict[str, Any]]) -> np.ndarray:
    label = np.zeros((height, width), dtype=np.uint8)
    for polygon in sorted(polygons, key=lambda item: item["z_order"]):
        points = polygon["points"]
        if len(points) < 3:
            continue
        mask = Image.new("L", (width, height), 0)
        ImageDraw.Draw(mask).polygon(points, fill=255)
        label[np.asarray(mask, dtype=np.uint8) > 0] = int(polygon["class_id"])
    return label


def paste_crop_label(
    crop_label: np.ndarray,
    full_size: tuple[int, int],
    crop_box: tuple[int, int, int, int] | None,
) -> np.ndarray:
    full = np.zeros((full_size[1], full_size[0]), dtype=np.uint8)
    if crop_box is None:
        if crop_label.shape != full.shape:
            return np.asarray(Image.fromarray(crop_label).resize(full_size, Image.Resampling.NEAREST), dtype=np.uint8)
        return crop_label

    x0, y0, x1, y1 = crop_box
    target_size = (x1 - x0, y1 - y0)
    if crop_label.shape[:2] != (target_size[1], target_size[0]):
        crop_label = np.asarray(Image.fromarray(crop_label).resize(target_size, Image.Resampling.NEAREST), dtype=np.uint8)
    full[y0:y1, x0:x1] = crop_label
    return full


def mask_bbox(label: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(label > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def resize_to_height(image: Image.Image, target_height: int) -> Image.Image:
    scale = target_height / float(image.height)
    return image.resize((max(1, int(round(image.width * scale))), target_height), Image.Resampling.BILINEAR)


def titled(image: Image.Image, title: str) -> Image.Image:
    output = Image.new("RGB", (image.width, image.height + 28), "white")
    output.paste(image.convert("RGB"), (0, 28))
    draw = ImageDraw.Draw(output)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    draw.text((6, 6), title, fill=(20, 20, 20), font=font)
    return output


def hstack(images: list[Image.Image]) -> Image.Image:
    output = Image.new("RGB", (sum(image.width for image in images), max(image.height for image in images)), "white")
    x = 0
    for image in images:
        output.paste(image.convert("RGB"), (x, 0))
        x += image.width
    return output


def make_grid(images: list[Image.Image], columns: int) -> Image.Image:
    if not images:
        return Image.new("RGB", (1, 1), "white")
    cell_width = max(image.width for image in images)
    cell_height = max(image.height for image in images)
    rows = int(math.ceil(len(images) / columns))
    output = Image.new("RGB", (cell_width * columns, cell_height * rows), "white")
    for index, image in enumerate(images):
        output.paste(image.convert("RGB"), ((index % columns) * cell_width, (index // columns) * cell_height))
    return output


def write_preview(
    portrait: Image.Image,
    crop_image: Image.Image,
    crop_label: np.ndarray,
    full_label: np.ndarray,
    uv_label: np.ndarray,
    output_path: Path,
) -> Image.Image:
    crop_overlay = overlay_label(crop_image, crop_label)
    full_overlay = overlay_label(portrait, full_label)
    uv_preview = label_to_rgba(uv_label).convert("RGB").resize((portrait.height, portrait.height), Image.Resampling.NEAREST)
    panel = hstack(
        [
            titled(resize_to_height(crop_image, 280), "crop image"),
            titled(resize_to_height(crop_overlay, 280), "CVAT crop label"),
            titled(resize_to_height(full_overlay, 280), "full portrait label"),
            titled(resize_to_height(uv_preview, 280), "UV label"),
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(output_path, quality=94)
    return panel


def project_annotations(
    manifest_path: Path,
    annotations_path: Path,
    output_root: Path,
    split: str,
    uv_size: int,
    cull_backfaces: bool,
    front_face_winding: str,
    preview_limit: int,
    overwrite: bool,
) -> dict[str, Any]:
    if output_root.exists() and overwrite:
        shutil.rmtree(output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(output_root)

    rows = read_jsonl(manifest_path)
    labels_by_stem = parse_cvat_labels(annotations_path)

    split_root = output_root / split
    images_dir = split_root / "images"
    frames_dir = split_root / "frames"
    uv_labels_dir = split_root / "uv_labels"
    crop_masks_dir = output_root / "debug" / "crop_masks"
    full_masks_dir = output_root / "debug" / "full_portrait_masks"
    previews_dir = output_root / "previews"
    for directory in (images_dir, frames_dir, uv_labels_dir, crop_masks_dir, full_masks_dir, previews_dir):
        directory.mkdir(parents=True, exist_ok=True)
    for empty_split in ("train", "val"):
        if empty_split != split:
            (output_root / empty_split).mkdir(parents=True, exist_ok=True)
            write_jsonl(output_root / empty_split / "metadata.jsonl", [])

    output_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    preview_panels: list[Image.Image] = []
    for row in rows:
        stem = str(row["stem"])
        annotation = labels_by_stem.get(stem)
        if annotation is None:
            skipped.append({"stem": stem, "reason": "missing_cvat_image"})
            continue
        if not annotation["polygons"]:
            skipped.append({"stem": stem, "reason": "no_supported_polygons"})
            continue

        frame_path = Path(str(row["source_frame_for_annotation"])).expanduser()
        raw_image_path = Path(str(row["source_image_for_annotation"])).expanduser()
        crop_image_path = Path(str(row["image"])).expanduser()
        frame = json.loads(frame_path.read_text(encoding="utf-8"))
        raw_image = Image.open(raw_image_path).convert("RGB")
        crop_image = Image.open(crop_image_path).convert("RGB")

        full_size_values = row.get("full_portrait_size")
        if not isinstance(full_size_values, list) or len(full_size_values) != 2:
            raise ValueError(f"Missing full_portrait_size for {stem}")
        full_size = (int(full_size_values[0]), int(full_size_values[1]))
        crop_box_values = row.get("crop_box_xyxy")
        crop_box = tuple(int(value) for value in crop_box_values) if isinstance(crop_box_values, list) else None

        crop_label = rasterize_crop_label(int(annotation["width"]), int(annotation["height"]), annotation["polygons"])
        full_label = paste_crop_label(crop_label, full_size, crop_box)
        uv_label = rasterize_2d_label_to_uv(
            frame,
            full_label,
            "portrait",
            uv_size,
            cull_backfaces=cull_backfaces,
            front_face_winding=front_face_winding,  # type: ignore[arg-type]
        )
        portrait = portrait_image(raw_image, frame, scale=1.0)

        output_image = images_dir / f"{stem}.jpg"
        output_frame = frames_dir / f"{stem}.json"
        output_uv_label = uv_labels_dir / f"{stem}.png"
        crop_mask_path = crop_masks_dir / f"{stem}.png"
        full_mask_path = full_masks_dir / f"{stem}.png"
        portrait.save(output_image, quality=95)
        shutil.copy2(frame_path, output_frame)
        Image.fromarray(uv_label).save(output_uv_label)
        Image.fromarray(crop_label).save(crop_mask_path)
        Image.fromarray(full_label).save(full_mask_path)

        if len(preview_panels) < preview_limit:
            preview_panels.append(
                write_preview(
                    portrait=portrait,
                    crop_image=crop_image,
                    crop_label=crop_label,
                    full_label=full_label,
                    uv_label=uv_label,
                    output_path=previews_dir / f"preview_{stem}.jpg",
                )
            )

        output_rows.append(
            {
                "stem": stem,
                "sample_id": row.get("sample_id") or row.get("metadata", {}).get("sample_id") or stem,
                "session": row.get("session") or row.get("metadata", {}).get("session"),
                "split": split,
                "image": f"{split}/images/{stem}.jpg",
                "frame": f"{split}/frames/{stem}.json",
                "uv_label": f"{split}/uv_labels/{stem}.png",
                "source_image": str(raw_image_path),
                "source_frame": str(frame_path),
                "source_cvat_annotations": str(annotations_path),
                "source_cvat_crop_image": str(crop_image_path),
                "source_cvat_crop_mask": str(crop_mask_path),
                "source_full_portrait_mask": str(full_mask_path),
                "manual_pixel_label": True,
                "annotation_image_size": [int(annotation["width"]), int(annotation["height"])],
                "full_portrait_size": list(full_size),
                "crop_box_xyxy": list(crop_box) if crop_box is not None else None,
                "uv_size": uv_size,
                "lip_area_px": int((uv_label > 0).sum()),
                "upper_lip_area_px": int((uv_label == 1).sum()),
                "lower_lip_area_px": int((uv_label == 2).sum()),
                "inner_mouth_area_px": int((uv_label == 3).sum()),
                "bbox_uv_px": mask_bbox(uv_label),
                "polygon_count": len(annotation["polygons"]),
            }
        )

    write_jsonl(split_root / "metadata.jsonl", output_rows)
    preview_sheet = previews_dir / "preview_sheet.jpg"
    make_grid(preview_panels, columns=1).save(preview_sheet, quality=94)
    summary = {
        "manifest": str(manifest_path),
        "annotations": str(annotations_path),
        "output_root": str(output_root),
        "split": split,
        "uv_size": uv_size,
        "count": len(output_rows),
        "skipped": skipped,
        "metadata": str(split_root / "metadata.jsonl"),
        "preview_sheet": str(preview_sheet),
        "stems": [row["stem"] for row in output_rows],
    }
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project manual CVAT mouth-crop polygons back to ARKit UV labels.")
    parser.add_argument("--manifest", required=True, help="manifest.jsonl from prepare_cvat_lip_annotation_from_arkit_uv_candidates.py")
    parser.add_argument("--annotations", required=True, help="CVAT 1.1 exported zip or annotations.xml")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--uv-size", type=int, default=512)
    parser.add_argument("--cull-backfaces", action="store_true")
    parser.add_argument("--front-face-winding", choices=["negative", "positive"], default="negative")
    parser.add_argument("--preview-limit", type=int, default=48)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = project_annotations(
        manifest_path=Path(args.manifest).expanduser(),
        annotations_path=Path(args.annotations).expanduser(),
        output_root=Path(args.output_root).expanduser(),
        split=args.split,
        uv_size=args.uv_size,
        cull_backfaces=args.cull_backfaces,
        front_face_winding=args.front_face_winding,
        preview_limit=args.preview_limit,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
