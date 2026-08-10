from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from lip_contour_dataset import LipContourDataset, denormalize_points, mask_iou, points_to_mask
from lip_contour_model import build_lip_contour_model


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_loader(dataset: LipContourDataset, batch_size: int, shuffle: bool, workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=False,
        drop_last=shuffle,
    )


def contour_smoothness_loss(points: torch.Tensor) -> torch.Tensor:
    previous_points = torch.roll(points, shifts=1, dims=1)
    next_points = torch.roll(points, shifts=-1, dims=1)
    return (previous_points - (2.0 * points) + next_points).abs().mean()


def contour_stats(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    target_masks: torch.Tensor,
    width: int,
    height: int,
) -> dict[str, float]:
    predictions_np = predictions.detach().cpu().numpy().astype(np.float32)
    targets_np = targets.detach().cpu().numpy().astype(np.float32)
    target_masks_np = target_masks.detach().cpu().numpy().astype(np.uint8)

    pixel_scale = np.array([max(width - 1, 1), max(height - 1, 1)], dtype=np.float32)
    point_mae = float(np.abs(predictions_np - targets_np).mean())
    pixel_mae = float((np.abs(predictions_np - targets_np) * pixel_scale[None, None, :]).mean())

    ious: list[float] = []
    pred_coverages: list[float] = []
    target_coverages: list[float] = []
    for prediction, target_mask in zip(predictions_np, target_masks_np[:, 0]):
        pixel_points = denormalize_points(prediction, width, height)
        pred_mask = points_to_mask(pixel_points, width, height, smooth=True)
        ious.append(mask_iou(pred_mask, target_mask))
        pred_coverages.append(float(pred_mask.mean()))
        target_coverages.append(float(target_mask.mean()))

    return {
        "point_mae": point_mae,
        "pixel_mae": pixel_mae,
        "mask_iou": float(np.mean(ious)) if ious else 0.0,
        "pred_coverage": float(np.mean(pred_coverages)) if pred_coverages else 0.0,
        "target_coverage": float(np.mean(target_coverages)) if target_coverages else 0.0,
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    args: argparse.Namespace,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    totals = {
        "loss": 0.0,
        "point_loss": 0.0,
        "smooth_loss": 0.0,
        "point_mae": 0.0,
        "pixel_mae": 0.0,
        "mask_iou": 0.0,
        "pred_coverage": 0.0,
        "target_coverage": 0.0,
    }
    batches = 0

    progress = tqdm(loader, leave=False)
    for batch in progress:
        images = batch["image"].to(device)
        targets = batch["points"].to(device)

        with torch.set_grad_enabled(is_train):
            predictions = model(images)
            point_loss = criterion(predictions, targets)
            smooth_loss = contour_smoothness_loss(predictions)
            loss = point_loss + (args.smooth_weight * smooth_loss)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        stats = contour_stats(
            predictions.detach(),
            targets.detach(),
            batch["lip_mask"],
            args.image_width,
            args.image_height,
        )
        totals["loss"] += float(loss.detach().cpu())
        totals["point_loss"] += float(point_loss.detach().cpu())
        totals["smooth_loss"] += float(smooth_loss.detach().cpu())
        for key, value in stats.items():
            totals[key] += value
        batches += 1
        progress.set_postfix(
            loss=f"{totals['loss'] / batches:.4f}",
            iou=f"{totals['mask_iou'] / batches:.4f}",
            px=f"{totals['pixel_mae'] / batches:.2f}",
        )

    return {key: value / max(batches, 1) for key, value in totals.items()}


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
    metrics: dict[str, float],
    best_score: float,
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "args": vars(args),
            "metrics": metrics,
            "best_metric": args.best_metric,
            "best_score": best_score,
            "input_width": args.image_width,
            "input_height": args.image_height,
            "model_width": args.model_width,
            "point_count": args.point_count,
            "smooth_weight": args.smooth_weight,
            "task": "lip_contour_student",
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="training/outputs/datasets/lip_segface512_plus_lip001_manual_v2_clean2_256x192")
    parser.add_argument("--output-dir", default="training/outputs/lip_contour_student_256x192_p64_w32")
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--image-height", type=int, default=192)
    parser.add_argument("--point-count", type=int, default=64)
    parser.add_argument("--model-width", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--smooth-weight", type=float, default=0.02)
    parser.add_argument("--best-metric", choices=["pixel_mae", "point_mae", "mask_iou", "loss"], default="pixel_mae")
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=5)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device()
    print(f"Device: {device}")
    print(f"Point count: {args.point_count}")
    print(f"Input: {args.image_width}x{args.image_height}")
    print(f"Smooth weight: {args.smooth_weight}")

    train_dataset = LipContourDataset(
        args.root,
        "train",
        image_width=args.image_width,
        image_height=args.image_height,
        point_count=args.point_count,
        limit=args.limit_train,
        augment=True,
    )
    val_dataset = LipContourDataset(
        args.root,
        "val",
        image_width=args.image_width,
        image_height=args.image_height,
        point_count=args.point_count,
        limit=args.limit_val,
        augment=False,
    )
    train_loader = make_loader(train_dataset, args.batch_size, shuffle=True, workers=args.workers)
    val_loader = make_loader(val_dataset, args.batch_size, shuffle=False, workers=args.workers)

    model = build_lip_contour_model(point_count=args.point_count, width=args.model_width).to(device)
    criterion = nn.SmoothL1Loss(beta=0.015)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    lower_is_better = args.best_metric in {"pixel_mae", "point_mae", "loss"}
    best_score = float("inf") if lower_is_better else -1.0
    history: list[dict[str, object]] = []
    started_at = time.time()

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_metrics = run_epoch(model, train_loader, criterion, device, args, optimizer)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, criterion, device, args)

        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(
            f"train loss={train_metrics['loss']:.4f} mask_iou={train_metrics['mask_iou']:.4f} "
            f"px_mae={train_metrics['pixel_mae']:.2f} | val loss={val_metrics['loss']:.4f} "
            f"mask_iou={val_metrics['mask_iou']:.4f} px_mae={val_metrics['pixel_mae']:.2f}"
        )

        score = val_metrics[args.best_metric]
        is_best = score < best_score if lower_is_better else score > best_score
        if is_best:
            best_score = score

        save_checkpoint(output_dir / "last.pt", model, optimizer, epoch, args, val_metrics, best_score)
        if is_best:
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, args, val_metrics, best_score)
        if args.save_every > 0 and (epoch % args.save_every == 0 or epoch == args.epochs):
            save_checkpoint(output_dir / f"epoch_{epoch:03d}.pt", model, optimizer, epoch, args, val_metrics, best_score)

        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    elapsed = time.time() - started_at
    print(f"\nDone in {elapsed / 60:.1f} min. Best val {args.best_metric}: {best_score:.4f}")
    print(f"Best checkpoint: {output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
