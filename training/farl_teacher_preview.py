from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from prepare_lip_dataset import (
    IMAGE_EXTENSIONS,
    bbox_from_mask,
    expanded_aspect_bbox,
    lip_label_from_lapa,
)


DEFAULT_LAPA_LIP_IDS = {
    "upper_lip": 7,
    "inner_mouth": 8,
    "lower_lip": 9,
}

CLASS_COLORS = {
    1: (236, 36, 82, 220),
    2: (255, 92, 128, 220),
    3: (24, 24, 28, 230),
}


def import_facer() -> Any:
    try:
        import facer  # type: ignore
    except ImportError as error:
        raise SystemExit(
            "Missing teacher dependency: facer.\n"
            "Create/activate .venv-teacher, then run:\n"
            "  pip install -r training/requirements-teacher.txt"
        ) from error
    return facer


def choose_device(name: str) -> torch.device:
    if name == "mps" and not torch.backends.mps.is_available():
        print("MPS requested but unavailable in this Python/PyTorch/macOS setup; falling back to CPU.")
        return torch.device("cpu")
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def image_by_stem(image_dir: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for extension in IMAGE_EXTENSIONS:
        for path in image_dir.glob(f"*{extension}"):
            result[path.stem] = path
    return result


def load_image_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def labels_from_parser(face_parser: Any) -> list[str] | None:
    for attr in ("labels", "label_names", "classes", "class_names"):
        value = getattr(face_parser, attr, None)
        if isinstance(value, (list, tuple)) and value:
            return [str(item) for item in value]

    config = getattr(face_parser, "config", None)
    if isinstance(config, dict):
        for key in ("labels", "label_names", "classes", "class_names"):
            value = config.get(key)
            if isinstance(value, (list, tuple)) and value:
                return [str(item) for item in value]
    return None


def lip_id_map(face_parser: Any, num_classes: int) -> dict[int, int]:
    labels = labels_from_parser(face_parser)
    if labels is None:
        if num_classes <= max(DEFAULT_LAPA_LIP_IDS.values()):
            raise RuntimeError(f"Cannot infer FaRL lip classes from {num_classes} output classes.")
        return {
            DEFAULT_LAPA_LIP_IDS["upper_lip"]: 1,
            DEFAULT_LAPA_LIP_IDS["lower_lip"]: 2,
            DEFAULT_LAPA_LIP_IDS["inner_mouth"]: 3,
        }

    normalized = [normalize_name(label) for label in labels]
    candidates = {
        1: {"upper_lip", "u_lip", "ulip", "upperlip"},
        2: {"lower_lip", "l_lip", "llip", "lowerlip"},
        3: {"inner_mouth", "mouth", "i_mouth", "imouth", "oral"},
    }

    result: dict[int, int] = {}
    for output_id, names in candidates.items():
        for class_id, name in enumerate(normalized):
            if name in names:
                result[class_id] = output_id
                break

    if len(result) != 3:
        if num_classes > max(DEFAULT_LAPA_LIP_IDS.values()):
            return {
                DEFAULT_LAPA_LIP_IDS["upper_lip"]: 1,
                DEFAULT_LAPA_LIP_IDS["lower_lip"]: 2,
                DEFAULT_LAPA_LIP_IDS["inner_mouth"]: 3,
            }
        raise RuntimeError(f"Cannot infer lip class ids from parser labels: {labels}")
    return result


def select_face_index(faces: dict[str, Any]) -> int:
    rects = faces.get("rects", faces.get("boxes"))
    if torch.is_tensor(rects) and rects.ndim == 2 and rects.shape[0] > 0 and rects.shape[1] >= 4:
        rects_cpu = rects.detach().cpu()
        widths = (rects_cpu[:, 2] - rects_cpu[:, 0]).clamp_min(0)
        heights = (rects_cpu[:, 3] - rects_cpu[:, 1]).clamp_min(0)
        return int(torch.argmax(widths * heights).item())
    return 0


def resize_label(label: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(label.astype(np.uint8))
    return np.asarray(image.resize(size, Image.NEAREST), dtype=np.uint8)


def teacher_lip_label(
    face_parser: Any,
    faces: dict[str, Any],
    image_size: tuple[int, int],
) -> tuple[np.ndarray, dict[int, int]]:
    seg_logits = faces.get("seg", {}).get("logits")
    if not torch.is_tensor(seg_logits):
        raise RuntimeError("Facer parser did not return faces['seg']['logits'].")

    if seg_logits.ndim != 4:
        raise RuntimeError(f"Unexpected seg logits shape: {tuple(seg_logits.shape)}")

    face_index = min(select_face_index(faces), seg_logits.shape[0] - 1)
    prediction = seg_logits.argmax(dim=1)[face_index].detach().cpu().numpy().astype(np.uint8)
    if prediction.shape != (image_size[1], image_size[0]):
        prediction = resize_label(prediction, image_size)

    mapping = lip_id_map(face_parser, int(seg_logits.shape[1]))
    output = np.zeros(prediction.shape, dtype=np.uint8)
    for source_id, lip_id in mapping.items():
        output[prediction == source_id] = lip_id
    return output, mapping


def label_to_rgba(label: np.ndarray) -> Image.Image:
    rgba = np.zeros((label.shape[0], label.shape[1], 4), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        rgba[label == class_id] = color
    return Image.fromarray(rgba)


def overlay_label(image: Image.Image, label: np.ndarray) -> Image.Image:
    overlay = label_to_rgba(label)
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def difference_image(gt: np.ndarray, teacher: np.ndarray) -> Image.Image:
    rgba = np.zeros((gt.shape[0], gt.shape[1], 4), dtype=np.uint8)
    rgba[(gt > 0) & (teacher == 0)] = [255, 210, 0, 230]
    rgba[(gt == 0) & (teacher > 0)] = [0, 180, 255, 230]
    rgba[(gt > 0) & (teacher > 0) & (gt != teacher)] = [170, 60, 255, 230]
    return Image.fromarray(rgba)


def titled(image: Image.Image, title: str) -> Image.Image:
    title_height = 30
    output = Image.new("RGB", (image.width, image.height + title_height), "white")
    output.paste(image.convert("RGB"), (0, title_height))
    draw = ImageDraw.Draw(output)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    draw.text((8, 8), title, fill=(30, 30, 30), font=font)
    return output


def make_grid(columns: list[Image.Image]) -> Image.Image:
    width = sum(image.width for image in columns)
    height = max(image.height for image in columns)
    output = Image.new("RGB", (width, height), "white")
    offset = 0
    for image in columns:
        output.paste(image, (offset, 0))
        offset += image.width
    return output


def crop_array(label: np.ndarray, crop_box: tuple[int, int, int, int], size: tuple[int, int]) -> np.ndarray:
    cropped = Image.fromarray(label.astype(np.uint8)).crop(crop_box)
    return np.asarray(cropped.resize(size, Image.NEAREST), dtype=np.uint8)


def update_iou(intersections: np.ndarray, unions: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> None:
    for class_id in range(4):
        gt_mask = gt == class_id
        pred_mask = pred == class_id
        intersections[class_id] += np.logical_and(gt_mask, pred_mask).sum()
        unions[class_id] += np.logical_or(gt_mask, pred_mask).sum()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/Users/lee/Datasets/LaPa")
    parser.add_argument("--split", default="val")
    parser.add_argument("--output-dir", default="training/outputs/teacher/farl_lapa448_preview")
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--crop-width", type=int, default=384)
    parser.add_argument("--crop-height", type=int, default=256)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda")
    args = parser.parse_args()

    root = Path(args.root).expanduser()
    split_root = root / args.split
    image_dir = split_root / "images"
    label_dir = split_root / "labels"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    facer = import_facer()
    device = choose_device(args.device)
    print(f"Device: {device}")
    print("Loading FaRL/facer teacher: farl/lapa/448")
    face_detector = facer.face_detector("retinaface/mobilenet", device=device)
    face_parser = facer.face_parser("farl/lapa/448", device=device)

    images = image_by_stem(image_dir)
    labels = sorted(label_dir.glob("*.png"))
    labels = labels[args.start : args.start + args.count]

    intersections = np.zeros(4, dtype=np.float64)
    unions = np.zeros(4, dtype=np.float64)
    saved: list[str] = []
    teacher_mapping: dict[int, int] | None = None

    for output_index, label_path in enumerate(tqdm(labels), start=1):
        image_path = images.get(label_path.stem)
        if image_path is None:
            continue

        image = Image.open(image_path).convert("RGB")
        label = np.asarray(Image.open(label_path).convert("L"), dtype=np.uint8)
        gt_lip = lip_label_from_lapa(label)
        bbox = bbox_from_mask(gt_lip > 0)
        if bbox is None:
            continue

        with torch.no_grad():
            image_tensor = load_image_tensor(image, device)
            faces = face_detector(image_tensor)
            if not isinstance(faces, dict) or int(faces.get("rects", torch.empty(0)).shape[0]) == 0:
                print(f"Skipped no-face image: {image_path}")
                continue
            faces = face_parser(image_tensor, faces)
            teacher_lip, mapping = teacher_lip_label(face_parser, faces, image.size)
            teacher_mapping = mapping

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
        image_crop = image.crop(crop_box).resize(preview_size, Image.BILINEAR)
        gt_crop = crop_array(gt_lip, crop_box, preview_size)
        teacher_crop = crop_array(teacher_lip, crop_box, preview_size)
        update_iou(intersections, unions, gt_crop, teacher_crop)

        diff = Image.alpha_composite(image_crop.convert("RGBA"), difference_image(gt_crop, teacher_crop)).convert("RGB")
        grid = make_grid(
            [
                titled(image_crop, "image"),
                titled(overlay_label(image_crop, gt_crop), "LaPa label"),
                titled(overlay_label(image_crop, teacher_crop), "FaRL teacher"),
                titled(diff, "diff"),
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
        "teacher": "farl/lapa/448",
        "device": str(device),
        "split": args.split,
        "count_requested": args.count,
        "count_saved": len(saved),
        "crop_size": [args.crop_width, args.crop_height],
        "teacher_lip_mapping": teacher_mapping,
        "mean_iou_vs_lapa": float(sum(valid_ious) / max(len(valid_ious), 1)),
        "class_iou_vs_lapa": per_class_iou,
        "previews": saved,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Summary: {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
