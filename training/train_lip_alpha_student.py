from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from lip_alpha_dataset import LipAlphaDataset
from model import build_model
from train_arkit_uv_alpha_student import SoftAlphaLoss, alpha_stats


NUM_OUTPUT_CHANNELS = 1


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_initial_model_state(model: nn.Module, checkpoint_path: str | Path) -> None:
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
    state = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
    model.load_state_dict(state)


def make_loader(dataset: LipAlphaDataset, batch_size: int, shuffle: bool, workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=False,
        drop_last=shuffle,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
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

    progress = tqdm(loader, leave=False)
    for batch in progress:
        images = batch["image"].to(device)
        targets = batch["alpha"].to(device)

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, targets)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        stats = alpha_stats(logits.detach().cpu(), targets.detach().cpu(), threshold)
        totals["loss"] += float(loss.detach().cpu())
        for key, value in stats.items():
            totals[key] += value
        batches += 1
        progress.set_postfix(
            loss=f"{totals['loss'] / batches:.4f}",
            iou=f"{totals['iou'] / batches:.4f}",
            mae=f"{totals['mae'] / batches:.4f}",
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
            "num_classes": NUM_OUTPUT_CHANNELS,
            "input_width": args.image_width,
            "input_height": args.image_height,
            "model_width": args.model_width,
            "alpha_feather_radius": args.alpha_feather_radius,
            "alpha_shrink_pixels": args.alpha_shrink_pixels,
            "alpha_threshold": args.alpha_threshold,
            "task": "lip_roi_alpha_student",
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="training/outputs/datasets/lip_segface512_plus_lip001_manual_v2_clean2_256x192")
    parser.add_argument("--output-dir", default="training/outputs/lip_alpha_student_256x192_w32")
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--image-height", type=int, default=192)
    parser.add_argument("--model-width", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--alpha-feather-radius", type=float, default=3.0)
    parser.add_argument("--alpha-shrink-pixels", type=float, default=0.0)
    parser.add_argument("--alpha-threshold", type=float, default=0.5)
    parser.add_argument("--lip-weight", type=float, default=5.0)
    parser.add_argument("--edge-weight", type=float, default=8.0)
    parser.add_argument("--l1-weight", type=float, default=0.35)
    parser.add_argument("--best-metric", choices=["mae", "rmse", "loss", "iou"], default="mae")
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--init-checkpoint", default=None, help="Checkpoint path to initialize model weights from.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device()
    print(f"Device: {device}")
    print(f"Alpha feather radius: {args.alpha_feather_radius}")
    print(f"Alpha shrink pixels: {args.alpha_shrink_pixels}")
    print(f"Loss weights: lip={args.lip_weight} edge={args.edge_weight} l1={args.l1_weight}")

    train_dataset = LipAlphaDataset(
        args.root,
        "train",
        image_width=args.image_width,
        image_height=args.image_height,
        feather_radius=args.alpha_feather_radius,
        shrink_pixels=args.alpha_shrink_pixels,
        limit=args.limit_train,
        augment=True,
    )
    val_dataset = LipAlphaDataset(
        args.root,
        "val",
        image_width=args.image_width,
        image_height=args.image_height,
        feather_radius=args.alpha_feather_radius,
        shrink_pixels=args.alpha_shrink_pixels,
        limit=args.limit_val,
        augment=False,
    )
    train_loader = make_loader(train_dataset, args.batch_size, shuffle=True, workers=args.workers)
    val_loader = make_loader(val_dataset, args.batch_size, shuffle=False, workers=args.workers)

    model = build_model(num_classes=NUM_OUTPUT_CHANNELS, width=args.model_width).to(device)
    if args.init_checkpoint:
        load_initial_model_state(model, args.init_checkpoint)
        print(f"Initialized model weights from {args.init_checkpoint}")

    criterion = SoftAlphaLoss(args.lip_weight, args.edge_weight, args.l1_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    lower_is_better = args.best_metric in {"mae", "rmse", "loss"}
    best_score = float("inf") if lower_is_better else -1.0
    history: list[dict[str, object]] = []
    started_at = time.time()

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_metrics = run_epoch(model, train_loader, criterion, device, args.alpha_threshold, optimizer)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, criterion, device, args.alpha_threshold)

        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)
        print(
            f"train loss={train_metrics['loss']:.4f} iou={train_metrics['iou']:.4f} "
            f"mae={train_metrics['mae']:.4f} | val loss={val_metrics['loss']:.4f} "
            f"iou={val_metrics['iou']:.4f} mae={val_metrics['mae']:.4f}"
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
