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

from lip_curve_dataset import CURVE_COUNT, LipCurveDataset, curves_to_mask, denormalize_curves, mask_iou
from lip_curve_model import build_lip_curve_model


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_loader(dataset: LipCurveDataset, batch_size: int, shuffle: bool, workers: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers, pin_memory=False, drop_last=shuffle)


def curve_smoothness_loss(curves: torch.Tensor) -> torch.Tensor:
    if curves.shape[2] < 3:
        return curves.new_tensor(0.0)
    second = curves[:, :, :-2] - (2.0 * curves[:, :, 1:-1]) + curves[:, :, 2:]
    return second.abs().mean()


def monotonic_x_loss(curves: torch.Tensor) -> torch.Tensor:
    dx = curves[:, :, 1:, 0] - curves[:, :, :-1, 0]
    return torch.relu(-dx).mean()


def vertical_order_loss(curves: torch.Tensor) -> torch.Tensor:
    upper_outer = curves[:, 0, :, 1]
    upper_inner = curves[:, 1, :, 1]
    lower_inner = curves[:, 2, :, 1]
    lower_outer = curves[:, 3, :, 1]
    return (
        torch.relu(upper_outer - upper_inner).mean()
        + torch.relu(upper_inner - lower_inner).mean()
        + torch.relu(lower_inner - lower_outer).mean()
    )


def curve_stats(
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
    pixel_mae = float((np.abs(predictions_np - targets_np) * pixel_scale[None, None, None, :]).mean())

    ious: list[float] = []
    pred_coverages: list[float] = []
    target_coverages: list[float] = []
    for prediction, target_mask in zip(predictions_np, target_masks_np[:, 0]):
        pixel_curves = denormalize_curves(prediction, width, height)
        pred_mask = curves_to_mask(pixel_curves, width, height, smooth=True)
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
        "x_loss": 0.0,
        "order_loss": 0.0,
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
        targets = batch["curves"].to(device)

        with torch.set_grad_enabled(is_train):
            predictions = model(images)
            point_loss = criterion(predictions, targets)
            smooth_loss = curve_smoothness_loss(predictions)
            x_loss = monotonic_x_loss(predictions)
            order_loss = vertical_order_loss(predictions)
            loss = (
                point_loss
                + (args.smooth_weight * smooth_loss)
                + (args.x_weight * x_loss)
                + (args.order_weight * order_loss)
            )

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        stats = curve_stats(predictions.detach(), targets.detach(), batch["lip_mask"], args.image_width, args.image_height)
        totals["loss"] += float(loss.detach().cpu())
        totals["point_loss"] += float(point_loss.detach().cpu())
        totals["smooth_loss"] += float(smooth_loss.detach().cpu())
        totals["x_loss"] += float(x_loss.detach().cpu())
        totals["order_loss"] += float(order_loss.detach().cpu())
        for key, value in stats.items():
            totals[key] += value
        batches += 1
        progress.set_postfix(loss=f"{totals['loss'] / batches:.4f}", iou=f"{totals['mask_iou'] / batches:.4f}", px=f"{totals['pixel_mae'] / batches:.2f}")

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
            "curve_count": CURVE_COUNT,
            "points_per_curve": args.points_per_curve,
            "task": "lip_curve_student",
        },
        path,
    )


def load_history(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return data if isinstance(data, list) else []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="training/outputs/datasets/lip001_video_interp_fixed_v2_clean2_roi_384x256_step2")
    parser.add_argument("--output-dir", default="training/outputs/lip_curve_student_384x256_p24_w32")
    parser.add_argument("--image-width", type=int, default=384)
    parser.add_argument("--image-height", type=int, default=256)
    parser.add_argument("--points-per-curve", type=int, default=24)
    parser.add_argument("--model-width", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--smooth-weight", type=float, default=0.04)
    parser.add_argument("--x-weight", type=float, default=0.10)
    parser.add_argument("--order-weight", type=float, default=0.12)
    parser.add_argument("--init-checkpoint", default=None, help="Optional checkpoint to initialize model weights from.")
    parser.add_argument("--resume-checkpoint", default=None, help="Resume full training state from a checkpoint, including optimizer and epoch.")
    parser.add_argument("--best-metric", choices=["pixel_mae", "point_mae", "mask_iou", "loss"], default="pixel_mae")
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=10)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device()
    print(f"Device: {device}")
    print(f"Input: {args.image_width}x{args.image_height}")
    print(f"Curves: {CURVE_COUNT} x {args.points_per_curve}")

    train_dataset = LipCurveDataset(args.root, "train", args.image_width, args.image_height, args.points_per_curve, args.limit_train, True)
    val_dataset = LipCurveDataset(args.root, "val", args.image_width, args.image_height, args.points_per_curve, args.limit_val, False)
    train_loader = make_loader(train_dataset, args.batch_size, shuffle=True, workers=args.workers)
    val_loader = make_loader(val_dataset, args.batch_size, shuffle=False, workers=args.workers)

    model = build_lip_curve_model(points_per_curve=args.points_per_curve, width=args.model_width).to(device)
    criterion = nn.SmoothL1Loss(beta=0.012)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    lower_is_better = args.best_metric in {"pixel_mae", "point_mae", "loss"}
    best_score = float("inf") if lower_is_better else -1.0
    history_path = output_dir / "history.json"
    history: list[dict[str, object]] = []
    start_epoch = 1
    checkpoint_path = args.resume_checkpoint or args.init_checkpoint
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        checkpoint_width = int(checkpoint.get("model_width", args.model_width))
        checkpoint_curves = int(checkpoint.get("curve_count", CURVE_COUNT))
        checkpoint_points = int(checkpoint.get("points_per_curve", args.points_per_curve))
        if checkpoint_width != args.model_width or checkpoint_curves != CURVE_COUNT or checkpoint_points != args.points_per_curve:
            raise ValueError(
                "Checkpoint architecture does not match: "
                f"checkpoint width/curves/points={checkpoint_width}/{checkpoint_curves}/{checkpoint_points}, "
                f"requested={args.model_width}/{CURVE_COUNT}/{args.points_per_curve}"
            )
        model.load_state_dict(checkpoint["model_state"])
        best_score = float(checkpoint.get("best_score", best_score))
        print(f"Initialized from: {checkpoint_path}")
        if args.resume_checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            start_epoch = int(checkpoint.get("epoch", 0)) + 1
            history = load_history(history_path)
            print(f"Resuming from epoch {start_epoch - 1}")

    started_at = time.time()

    if start_epoch > args.epochs:
        print(f"Checkpoint already reached epoch {start_epoch - 1}, which is >= --epochs {args.epochs}. Nothing to do.")
        return

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_metrics = run_epoch(model, train_loader, criterion, device, args, optimizer)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, criterion, device, args)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(
            f"train loss={train_metrics['loss']:.4f} iou={train_metrics['mask_iou']:.4f} px={train_metrics['pixel_mae']:.2f} | "
            f"val loss={val_metrics['loss']:.4f} iou={val_metrics['mask_iou']:.4f} px={val_metrics['pixel_mae']:.2f}"
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
