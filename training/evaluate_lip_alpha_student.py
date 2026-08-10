from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lip_alpha_dataset import LipAlphaDataset
from model import build_model
from train_arkit_uv_alpha_student import SoftAlphaLoss, alpha_stats
from train_lip_alpha_student import choose_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="training/outputs/datasets/lip_segface512_plus_lip001_manual_v2_clean2_256x192")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--image-height", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    device = choose_device()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    image_width = int(checkpoint.get("input_width", args.image_width))
    image_height = int(checkpoint.get("input_height", args.image_height))
    model_width = int(checkpoint.get("model_width", 32))
    feather_radius = float(checkpoint.get("alpha_feather_radius", 3.0))
    shrink_pixels = float(checkpoint.get("alpha_shrink_pixels", 0.0))
    threshold = float(checkpoint.get("alpha_threshold", 0.5))

    dataset = LipAlphaDataset(
        args.root,
        args.split,
        image_width=image_width,
        image_height=image_height,
        feather_radius=feather_radius,
        shrink_pixels=shrink_pixels,
        limit=args.limit,
        augment=False,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = build_model(num_classes=int(checkpoint.get("num_classes", 1)), width=model_width)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    criterion = SoftAlphaLoss(lip_weight=5.0, edge_weight=8.0, l1_weight=0.35)

    totals = {
        "loss": 0.0,
        "mae": 0.0,
        "rmse": 0.0,
        "iou": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "pred_coverage": 0.0,
        "target_coverage": 0.0,
    }
    batches = 0
    with torch.no_grad():
        for batch in tqdm(loader):
            images = batch["image"].to(device)
            targets = batch["alpha"].to(device)
            logits = model(images)
            loss = criterion(logits, targets)
            stats = alpha_stats(logits.cpu(), targets.cpu(), threshold)
            totals["loss"] += float(loss.cpu())
            for key, value in stats.items():
                totals[key] += value
            batches += 1

    print(f"Checkpoint: {Path(args.checkpoint)}")
    print(f"Split: {args.split}")
    print(f"Input: {image_width}x{image_height}")
    print(f"Alpha feather radius: {feather_radius}")
    print(f"Alpha shrink pixels: {shrink_pixels}")
    print(f"Alpha threshold: {threshold}")
    for key, value in totals.items():
        print(f"{key}: {value / max(batches, 1):.4f}")


if __name__ == "__main__":
    main()
