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
    IMAGE_EXTENSIONS,
    bbox_from_mask,
    expanded_aspect_bbox,
    lip_label_from_lapa,
)


CLASS_COLORS = {
    1: (236, 36, 82, 220),
    2: (255, 92, 128, 220),
    3: (24, 24, 28, 230),
}

DEFAULT_LAPA_LIP_IDS = {
    "upper_lip": 7,
    "inner_mouth": 8,
    "lower_lip": 9,
}


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


def image_by_stem(image_dir: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for extension in IMAGE_EXTENSIONS:
        for path in image_dir.glob(f"*{extension}"):
            result[path.stem] = path
    return result


def read_lapa_landmarks(path: Path) -> np.ndarray:
    values = [float(value) for value in path.read_text(encoding="utf-8").split()]
    if not values or int(values[0]) != 106 or len(values) != 106 * 2 + 1:
        raise ValueError(f"Unexpected LaPa landmark file: {path}")
    return np.asarray(values[1:], dtype=np.float32).reshape(106, 2)


def five_lapa_landmarks(landmarks: np.ndarray, image_size: tuple[int, int], resolution: int) -> torch.Tensor:
    width, height = image_size
    points = np.array(
        [
            [(landmarks[74][0] + landmarks[104][0]) / 2, (landmarks[74][1] + landmarks[104][1]) / 2],
            [(landmarks[83][0] + landmarks[105][0]) / 2, (landmarks[83][1] + landmarks[105][1]) / 2],
            [landmarks[54][0], landmarks[54][1]],
            [landmarks[84][0], landmarks[84][1]],
            [landmarks[90][0], landmarks[90][1]],
        ],
        dtype=np.float32,
    )
    points[:, 0] *= resolution / float(width)
    points[:, 1] *= resolution / float(height)
    return torch.from_numpy(points).float().unsqueeze(0)


def image_tensor(image: Image.Image, resolution: int) -> torch.Tensor:
    resized = image.convert("RGB").resize((resolution, resolution), Image.Resampling.BICUBIC)
    array = np.asarray(resized, dtype=np.float32).copy() / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
    return ((tensor - mean) / std).unsqueeze(0)


def load_segface_model(repo_dir: Path, checkpoint_path: Path, resolution: int, device: torch.device) -> torch.nn.Module:
    sys.path.insert(0, str(repo_dir))
    from network import get_model  # type: ignore

    model = get_model("segface_lapa", resolution, "swin_base").to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict_backbone"])
    model.eval()
    return model


def segface_lip_label(
    model: torch.nn.Module,
    image: Image.Image,
    landmarks: np.ndarray,
    resolution: int,
    device: torch.device,
) -> np.ndarray:
    image_input = image_tensor(image, resolution).to(device)
    labels = {"lnm_seg": five_lapa_landmarks(landmarks, image.size, resolution).to(device)}
    dataset = torch.ones((1,), dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(image_input, labels, dataset)
        logits = torch.nn.functional.interpolate(
            logits,
            size=(resolution, resolution),
            mode="bilinear",
            align_corners=False,
        )
        prediction = logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)

    if prediction.shape != (image.height, image.width):
        prediction = np.asarray(
            Image.fromarray(prediction).resize(image.size, Image.Resampling.NEAREST),
            dtype=np.uint8,
        )

    output = np.zeros(prediction.shape, dtype=np.uint8)
    output[prediction == DEFAULT_LAPA_LIP_IDS["upper_lip"]] = 1
    output[prediction == DEFAULT_LAPA_LIP_IDS["lower_lip"]] = 2
    output[prediction == DEFAULT_LAPA_LIP_IDS["inner_mouth"]] = 3
    return output


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
    return np.asarray(cropped.resize(size, Image.Resampling.NEAREST), dtype=np.uint8)


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
    parser.add_argument("--output-dir", default="training/outputs/teacher/segface_lapa512_preview")
    parser.add_argument("--segface-root", default="third_party/SegFace")
    parser.add_argument("--checkpoint", default="/Users/lee/Models/SegFace/swinb_lapa_512/model_299.pt")
    parser.add_argument("--resolution", type=int, default=512)
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
    landmark_dir = split_root / "landmarks"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    print(f"Device: {device}")
    print(f"Loading SegFace LaPa {args.resolution}: {args.checkpoint}")
    model = load_segface_model(Path(args.segface_root), Path(args.checkpoint).expanduser(), args.resolution, device)

    images = image_by_stem(image_dir)
    labels = sorted(label_dir.glob("*.png"))
    labels = labels[args.start : args.start + args.count]

    intersections = np.zeros(4, dtype=np.float64)
    unions = np.zeros(4, dtype=np.float64)
    saved: list[str] = []
    skipped: list[str] = []

    for output_index, label_path in enumerate(tqdm(labels), start=1):
        image_path = images.get(label_path.stem)
        landmark_path = landmark_dir / f"{label_path.stem}.txt"
        if image_path is None or not landmark_path.exists():
            skipped.append(label_path.stem)
            continue

        image = Image.open(image_path).convert("RGB")
        label = np.asarray(Image.open(label_path).convert("L"), dtype=np.uint8)
        gt_lip = lip_label_from_lapa(label)
        bbox = bbox_from_mask(gt_lip > 0)
        if bbox is None:
            skipped.append(label_path.stem)
            continue

        landmarks = read_lapa_landmarks(landmark_path)
        teacher_lip = segface_lip_label(model, image, landmarks, args.resolution, device)

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
        gt_crop = crop_array(gt_lip, crop_box, preview_size)
        teacher_crop = crop_array(teacher_lip, crop_box, preview_size)
        update_iou(intersections, unions, gt_crop, teacher_crop)

        diff = Image.alpha_composite(image_crop.convert("RGBA"), difference_image(gt_crop, teacher_crop)).convert("RGB")
        grid = make_grid(
            [
                titled(image_crop, "image"),
                titled(overlay_label(image_crop, gt_crop), "LaPa label"),
                titled(overlay_label(image_crop, teacher_crop), "SegFace teacher"),
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
        "teacher": f"segface/swinb_lapa_{args.resolution}",
        "device": str(device),
        "split": args.split,
        "count_requested": args.count,
        "count_saved": len(saved),
        "skipped": skipped,
        "crop_size": [args.crop_width, args.crop_height],
        "teacher_lip_mapping": {"7": 1, "9": 2, "8": 3},
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
