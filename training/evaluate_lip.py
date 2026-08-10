from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from lip_dataset import LipRoiDataset
from lip_template_dataset import TemplateGuidedLipRoiDataset
from metrics import segmentation_stats
from model import build_model


NUM_CLASSES = 4
CLASS_NAMES = ["background_skin", "upper_lip", "lower_lip", "inner_mouth"]


def build_lip_eval_dataset(
    root: str | Path,
    split: str,
    image_width: int,
    image_height: int,
    limit: int | None,
    checkpoint: dict[str, object],
) -> LipRoiDataset | TemplateGuidedLipRoiDataset:
    uses_template = bool(checkpoint.get("template_prior")) or int(checkpoint.get("input_channels", 3)) == 4
    dataset_cls = TemplateGuidedLipRoiDataset if uses_template else LipRoiDataset
    return dataset_cls(
        root,
        split,
        image_width=image_width,
        image_height=image_height,
        limit=limit,
        augment=False,
    )


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/Users/lee/Datasets/LaPaLip")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--image-width", type=int, default=192)
    parser.add_argument("--image-height", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    device = choose_device()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    num_classes = int(checkpoint.get("num_classes", NUM_CLASSES))
    image_width = int(checkpoint.get("input_width", args.image_width))
    image_height = int(checkpoint.get("input_height", args.image_height))
    input_channels = int(checkpoint.get("input_channels", 3))

    model = build_model(
        num_classes=num_classes,
        width=int(checkpoint.get("model_width", 32)),
        input_channels=input_channels,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    dataset = build_lip_eval_dataset(
        root=args.root,
        split=args.split,
        image_width=image_width,
        image_height=image_height,
        limit=args.limit,
        checkpoint=checkpoint,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_accuracy = 0.0
    total_iou = 0.0
    batches = 0
    intersections = torch.zeros(num_classes, dtype=torch.float64)
    unions = torch.zeros(num_classes, dtype=torch.float64)

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
            total_loss += float(loss.cpu())
            total_accuracy += stats["pixel_accuracy"]
            total_iou += stats["mean_iou"]
            batches += 1

    print(f"Checkpoint: {Path(args.checkpoint)}")
    print(f"Split: {args.split}")
    print(f"Input: {image_width}x{image_height}x{input_channels}")
    print(f"Loss: {total_loss / max(batches, 1):.4f}")
    print(f"Pixel accuracy: {total_accuracy / max(batches, 1):.4f}")
    print(f"Mean IoU: {total_iou / max(batches, 1):.4f}")
    print("Class IoU:")
    for class_id in range(num_classes):
        class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"class_{class_id}"
        if unions[class_id] > 0:
            iou = intersections[class_id] / unions[class_id]
            print(f"  {class_id} {class_name}: {iou:.4f}")


if __name__ == "__main__":
    main()
