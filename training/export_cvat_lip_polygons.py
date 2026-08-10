from __future__ import annotations

import argparse
import getpass
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont


CLASS_IDS = {
    "upper_lip": 1,
    "lower_lip": 2,
    "inner_mouth": 3,
}

CLASS_COLORS = {
    1: (236, 36, 82, 190),
    2: (255, 92, 128, 190),
    3: (24, 24, 28, 230),
}


def login(base_url: str, username: str, password: str) -> str:
    response = requests.post(
        f"{base_url.rstrip('/')}/api/auth/login",
        json={"username": username, "password": password},
        timeout=20,
    )
    response.raise_for_status()
    return str(response.json()["key"])


def api_get(base_url: str, path: str, token: str, params: dict[str, Any] | None = None) -> Any:
    response = requests.get(
        f"{base_url.rstrip('/')}{path}",
        headers={"Authorization": f"Token {token}"},
        params=params,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def draw_polygon(size: tuple[int, int], points: list[float]) -> np.ndarray:
    image = Image.new("L", size, 0)
    if len(points) >= 6:
        pairs = [(float(points[index]), float(points[index + 1])) for index in range(0, len(points) - 1, 2)]
        ImageDraw.Draw(image).polygon(pairs, fill=255)
    return np.asarray(image, dtype=np.uint8) > 0


def label_to_rgba(label: np.ndarray) -> Image.Image:
    rgba = np.zeros((label.shape[0], label.shape[1], 4), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        rgba[label == class_id] = color
    return Image.fromarray(rgba)


def overlay_label(image: Image.Image, label: np.ndarray) -> Image.Image:
    if image.size != (label.shape[1], label.shape[0]):
        image = image.resize((label.shape[1], label.shape[0]), Image.Resampling.BILINEAR)
    return Image.alpha_composite(image.convert("RGBA"), label_to_rgba(label)).convert("RGB")


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


def fit_width(image: Image.Image, width: int) -> Image.Image:
    if image.width == width:
        return image
    height = max(1, int(round(image.height * width / image.width)))
    return image.resize((width, height), Image.Resampling.BILINEAR)


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


def label_from_shapes(
    frame_index: int,
    frame: dict[str, Any],
    shapes: list[dict[str, Any]],
    label_names: dict[int, str],
) -> tuple[np.ndarray, dict[str, np.ndarray], int]:
    size = (int(frame["width"]), int(frame["height"]))
    combined = np.zeros((size[1], size[0]), dtype=np.uint8)
    class_masks = {name: np.zeros((size[1], size[0]), dtype=bool) for name in CLASS_IDS}
    frame_shapes = [
        shape
        for shape in shapes
        if int(shape.get("frame", -1)) == frame_index and shape.get("type") == "polygon" and not shape.get("outside", False)
    ]
    frame_shapes.sort(key=lambda shape: int(shape.get("z_order", 0)))
    for shape in frame_shapes:
        class_name = label_names.get(int(shape["label_id"]))
        if class_name not in CLASS_IDS:
            continue
        mask = draw_polygon(size, [float(value) for value in shape.get("points", [])])
        class_masks[class_name] |= mask
        combined[mask] = CLASS_IDS[class_name]

    # Make the mouth cutout authoritative when present.
    if class_masks["inner_mouth"].any():
        combined[class_masks["inner_mouth"]] = CLASS_IDS["inner_mouth"]
    return combined, class_masks, len(frame_shapes)


def export_task(
    base_url: str,
    task_id: int,
    output_dir: Path,
    image_dir: Path | None,
    token: str,
    panel_width: int,
    overwrite: bool,
) -> dict[str, Any]:
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)

    masks_dir = output_dir / "masks"
    combined_dir = output_dir / "combined_masks"
    preview_dir = output_dir / "previews"
    raw_dir = output_dir / "raw"
    for class_name in CLASS_IDS:
        (masks_dir / class_name).mkdir(parents=True, exist_ok=True)
    combined_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    task = api_get(base_url, f"/api/tasks/{task_id}", token)
    meta = api_get(base_url, f"/api/tasks/{task_id}/data/meta", token)
    labels_payload = api_get(base_url, "/api/labels", token, params={"task_id": task_id})
    annotations = api_get(base_url, f"/api/tasks/{task_id}/annotations", token)

    (raw_dir / "task.json").write_text(json.dumps(task, indent=2), encoding="utf-8")
    (raw_dir / "data_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (raw_dir / "labels.json").write_text(json.dumps(labels_payload, indent=2), encoding="utf-8")
    (raw_dir / "annotations.json").write_text(json.dumps(annotations, indent=2), encoding="utf-8")

    label_names = {int(label["id"]): str(label["name"]) for label in labels_payload["results"]}
    frames = meta["frames"]
    saved: list[str] = []
    preview_panels: list[Image.Image] = []
    frame_shape_counts: dict[str, int] = {}
    for frame_index, frame in enumerate(frames):
        frame_name = Path(frame["name"]).name
        stem = Path(frame_name).stem
        combined, class_masks, shape_count = label_from_shapes(frame_index, frame, annotations["shapes"], label_names)
        frame_shape_counts[stem] = shape_count
        for class_name, mask in class_masks.items():
            Image.fromarray(mask.astype(np.uint8) * 255).save(masks_dir / class_name / f"{stem}.png")
        combined_path = combined_dir / f"{stem}.png"
        Image.fromarray(combined).save(combined_path)
        saved.append(str(combined_path))

        if image_dir is not None:
            image_path = image_dir / frame_name
            if image_path.exists():
                image = Image.open(image_path).convert("RGB")
            else:
                image = Image.new("RGB", (combined.shape[1], combined.shape[0]), (96, 86, 76))
        else:
            image = Image.new("RGB", (combined.shape[1], combined.shape[0]), (96, 86, 76))
        preview = fit_width(hstack([titled(image, frame_name), titled(overlay_label(image, combined), "CVAT label")]), panel_width)
        preview.save(preview_dir / f"preview_{frame_index:03d}_{stem}.jpg", quality=94)
        preview_panels.append(preview)

    contact_sheet = preview_dir / "preview_sheet.jpg"
    make_grid(preview_panels, columns=2).save(contact_sheet, quality=94)

    summary = {
        "base_url": base_url,
        "task_id": task_id,
        "task_name": task["name"],
        "frame_count": len(frames),
        "shape_count": len(annotations["shapes"]),
        "frame_shape_counts": frame_shape_counts,
        "output_dir": str(output_dir),
        "masks_dir": str(masks_dir),
        "combined_dir": str(combined_dir),
        "preview_sheet": str(contact_sheet),
        "labels": {str(key): value for key, value in label_names.items()},
        "saved": saved,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export CVAT lip polygon annotations to class masks.")
    parser.add_argument("--base-url", default="http://localhost:8080")
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-dir")
    parser.add_argument("--token")
    parser.add_argument("--username")
    parser.add_argument("--password")
    parser.add_argument("--panel-width", type=int, default=1200)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = args.token
    if not token:
        if not args.username:
            raise SystemExit("Pass --token or --username/--password")
        password = args.password if args.password is not None else getpass.getpass("CVAT password: ")
        token = login(args.base_url, args.username, password)

    summary = export_task(
        base_url=args.base_url,
        task_id=args.task_id,
        output_dir=Path(args.output_dir).expanduser(),
        image_dir=Path(args.image_dir).expanduser() if args.image_dir else None,
        token=token,
        panel_width=args.panel_width,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
