from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from arkit_uv_student_dataset import ARKitUVTeacherDataset
from metrics import segmentation_stats
from model import build_model


NUM_CLASSES = 4
CLASS_NAMES = ["background_or_non_lip_uv", "upper_lip", "lower_lip", "inner_mouth"]


def parse_crop_box(value: str | None) -> tuple[int, int, int, int] | None:
    if value is None or value.lower() in {"", "none"}:
        return None
    parts = value.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--crop-box must be x0,y0,x1,y1")
    try:
        x0, y0, x1, y1 = (int(part) for part in parts)
    except ValueError as error:
        raise argparse.ArgumentTypeError("--crop-box values must be integers") from error
    return x0, y0, x1, y1


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/Users/lee/Datasets/ARKitLipUVTeacher")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--uv-size", type=int, default=None)
    parser.add_argument("--image-width", type=int, default=None)
    parser.add_argument("--image-height", type=int, default=None)
    parser.add_argument("--crop-box", type=parse_crop_box, default=None, help="Optional UV crop as x0,y0,x1,y1")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--cache-uv-images", action="store_true")
    parser.add_argument("--no-cull-backfaces", action="store_true")
    parser.add_argument("--front-face-winding", choices=["negative", "positive"], default="negative")
    args = parser.parse_args()

    device = choose_device()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    checkpoint_args = checkpoint.get("args", {})
    uv_size = int(args.uv_size or checkpoint.get("uv_size") or checkpoint_args.get("uv_size") or 256)
    image_width = int(args.image_width or checkpoint.get("input_width") or checkpoint_args.get("image_width") or uv_size)
    image_height = int(args.image_height or checkpoint.get("input_height") or checkpoint_args.get("image_height") or uv_size)
    crop_box = args.crop_box or checkpoint.get("crop_box") or checkpoint_args.get("crop_box")
    if crop_box is not None:
        crop_box = tuple(int(value) for value in crop_box)
    num_classes = int(checkpoint.get("num_classes", NUM_CLASSES))

    model = build_model(num_classes=num_classes, width=int(checkpoint.get("model_width", 16)))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    dataset = ARKitUVTeacherDataset(
        args.root,
        args.split,
        uv_size=uv_size,
        image_width=image_width,
        image_height=image_height,
        crop_box=crop_box,
        limit=args.limit,
        augment=False,
        cache_uv_images=args.cache_uv_images,
        cull_backfaces=not args.no_cull_backfaces,
        front_face_winding=args.front_face_winding,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_accuracy = 0.0
    total_iou = 0.0
    batches = 0
    intersections = torch.zeros(num_classes, dtype=torch.float64)
    unions = torch.zeros(num_classes, dtype=torch.float64)
    lip_intersection = torch.tensor(0.0, dtype=torch.float64)
    lip_union = torch.tensor(0.0, dtype=torch.float64)
    lip_predicted = torch.tensor(0.0, dtype=torch.float64)
    lip_labeled = torch.tensor(0.0, dtype=torch.float64)

    with torch.no_grad():
        for batch in tqdm(loader):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            stats = segmentation_stats(logits.cpu(), labels.cpu(), num_classes)
            predictions = logits.argmax(dim=1).cpu()
            cpu_labels = labels.cpu()
            for class_id in range(num_classes):
                pred_mask = predictions == class_id
                label_mask = cpu_labels == class_id
                intersections[class_id] += (pred_mask & label_mask).sum().item()
                unions[class_id] += (pred_mask | label_mask).sum().item()
            pred_lip = (predictions == 1) | (predictions == 2)
            label_lip = (cpu_labels == 1) | (cpu_labels == 2)
            lip_intersection += (pred_lip & label_lip).sum().item()
            lip_union += (pred_lip | label_lip).sum().item()
            lip_predicted += pred_lip.sum().item()
            lip_labeled += label_lip.sum().item()
            total_loss += float(loss.cpu())
            total_accuracy += stats["pixel_accuracy"]
            total_iou += stats["mean_iou"]
            batches += 1

    print(f"Checkpoint: {Path(args.checkpoint)}")
    print(f"Split: {args.split}")
    print(f"UV source: {uv_size}x{uv_size}")
    print(f"Input: {image_width}x{image_height}")
    print(f"Crop box: {crop_box}")
    print(f"Loss: {total_loss / max(batches, 1):.4f}")
    print(f"Pixel accuracy: {total_accuracy / max(batches, 1):.4f}")
    print(f"Mean IoU: {total_iou / max(batches, 1):.4f}")
    if lip_union > 0:
        lip_iou = lip_intersection / lip_union
        lip_precision = lip_intersection / lip_predicted if lip_predicted > 0 else torch.tensor(0.0)
        lip_recall = lip_intersection / lip_labeled if lip_labeled > 0 else torch.tensor(0.0)
        print(f"Lip union IoU: {lip_iou:.4f}")
        print(f"Lip precision: {lip_precision:.4f}")
        print(f"Lip recall: {lip_recall:.4f}")
    print("Class IoU:")
    for class_id in range(num_classes):
        class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"class_{class_id}"
        if unions[class_id] > 0:
            iou = intersections[class_id] / unions[class_id]
            print(f"  {class_id} {class_name}: {iou:.4f}")


if __name__ == "__main__":
    main()
