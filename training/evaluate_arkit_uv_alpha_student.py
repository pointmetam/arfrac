from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from arkit_uv_alpha_dataset import ARKitUVAlphaDataset
from evaluate_arkit_uv_student import parse_crop_box
from model import build_model
from train_arkit_uv_alpha_student import SoftAlphaLoss, alpha_stats


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
    parser.add_argument("--crop-box", type=parse_crop_box, default=None)
    parser.add_argument("--alpha-feather-radius", type=float, default=None)
    parser.add_argument("--alpha-threshold", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--cache-uv-images", action="store_true")
    parser.add_argument("--cache-alpha-labels", action="store_true")
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
    feather_radius = float(
        args.alpha_feather_radius
        or checkpoint.get("alpha_feather_radius")
        or checkpoint_args.get("alpha_feather_radius")
        or 3.0
    )
    threshold = float(args.alpha_threshold or checkpoint.get("alpha_threshold") or checkpoint_args.get("alpha_threshold") or 0.5)

    model = build_model(num_classes=int(checkpoint.get("num_classes", 1)), width=int(checkpoint.get("model_width", 24)))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    dataset = ARKitUVAlphaDataset(
        args.root,
        args.split,
        uv_size=uv_size,
        image_width=image_width,
        image_height=image_height,
        crop_box=crop_box,
        feather_radius=feather_radius,
        cache_alpha_labels=args.cache_alpha_labels,
        limit=args.limit,
        augment=False,
        cache_uv_images=args.cache_uv_images,
        cull_backfaces=not args.no_cull_backfaces,
        front_face_winding=args.front_face_winding,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    criterion: nn.Module = SoftAlphaLoss(lip_weight=5.0, edge_weight=5.0, l1_weight=0.35)

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
    print(f"UV source: {uv_size}x{uv_size}")
    print(f"Input: {image_width}x{image_height}")
    print(f"Crop box: {crop_box}")
    print(f"Alpha feather radius: {feather_radius}")
    print(f"Alpha threshold: {threshold}")
    for key, value in totals.items():
        print(f"{key}: {value / max(batches, 1):.4f}")


if __name__ == "__main__":
    main()
