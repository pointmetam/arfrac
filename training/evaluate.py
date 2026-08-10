from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import LaPaDataset
from metrics import segmentation_stats
from model import build_model


NUM_CLASSES = 9
CLASS_NAMES = [
    "background",
    "skin",
    "eyebrow",
    "eye",
    "nose",
    "upper_lip",
    "lower_lip",
    "inner_mouth",
    "hair",
]


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/Users/lee/Datasets/LaPa")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    device = choose_device()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = build_model(num_classes=checkpoint.get("num_classes", NUM_CLASSES))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    dataset = LaPaDataset(args.root, args.split, image_size=args.image_size, limit=args.limit)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_accuracy = 0.0
    total_iou = 0.0
    batches = 0
    intersections = torch.zeros(NUM_CLASSES, dtype=torch.float64)
    unions = torch.zeros(NUM_CLASSES, dtype=torch.float64)

    with torch.no_grad():
        for batch in tqdm(loader):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            stats = segmentation_stats(logits.cpu(), labels.cpu(), NUM_CLASSES)
            predictions = logits.argmax(dim=1).cpu()
            cpu_labels = labels.cpu()
            for class_id in range(NUM_CLASSES):
                pred_mask = predictions == class_id
                label_mask = cpu_labels == class_id
                intersections[class_id] += (pred_mask & label_mask).sum().item()
                unions[class_id] += (pred_mask | label_mask).sum().item()
            total_loss += float(loss.cpu())
            total_accuracy += stats["pixel_accuracy"]
            total_iou += stats["mean_iou"]
            batches += 1

    print(f"Checkpoint: {Path(args.checkpoint)}")
    print(f"Split: {args.split}")
    print(f"Loss: {total_loss / max(batches, 1):.4f}")
    print(f"Pixel accuracy: {total_accuracy / max(batches, 1):.4f}")
    print(f"Mean IoU: {total_iou / max(batches, 1):.4f}")
    print("Class IoU:")
    for class_id, class_name in enumerate(CLASS_NAMES):
        if unions[class_id] > 0:
            iou = intersections[class_id] / unions[class_id]
            print(f"  {class_id} {class_name}: {iou:.4f}")


if __name__ == "__main__":
    main()
