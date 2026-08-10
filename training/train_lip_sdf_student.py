from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from lip_sdf_dataset import LipSDFDataset
from model import build_model
from train_arkit_uv_alpha_student import alpha_stats


NUM_OUTPUT_CHANNELS = 3


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class LipSDFLoss(nn.Module):
    def __init__(self, alpha_weight: float, sdf_weight: float, edge_weight: float, edge_pos_weight: float) -> None:
        super().__init__()
        self.alpha_weight = float(alpha_weight)
        self.sdf_weight = float(sdf_weight)
        self.edge_weight = float(edge_weight)
        self.edge_pos_weight = float(edge_pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        alpha_logits = logits[:, 0:1]
        sdf_pred = torch.tanh(logits[:, 1:2])
        edge_logits = logits[:, 2:3]

        alpha_target = targets[:, 0:1].clamp(0, 1)
        sdf_target = targets[:, 1:2].clamp(-1, 1)
        edge_target = targets[:, 2:3].clamp(0, 1)

        edge_emphasis = 1.0 + (8.0 * edge_target)
        alpha_loss = (F.binary_cross_entropy_with_logits(alpha_logits, alpha_target, reduction="none") * edge_emphasis)
        alpha_loss = alpha_loss.mean()

        sdf_loss = (F.smooth_l1_loss(sdf_pred, sdf_target, reduction="none", beta=0.08) * edge_emphasis).mean()

        edge_bce = F.binary_cross_entropy_with_logits(edge_logits, edge_target, reduction="none")
        edge_weights = 1.0 + (edge_target * self.edge_pos_weight)
        edge_loss = (edge_bce * edge_weights).mean()

        return (self.alpha_weight * alpha_loss) + (self.sdf_weight * sdf_loss) + (self.edge_weight * edge_loss)


def make_loader(dataset: LipSDFDataset, batch_size: int, shuffle: bool, workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=False,
        drop_last=shuffle,
    )


def sdf_stats(logits: torch.Tensor, targets: torch.Tensor, threshold: float) -> dict[str, float]:
    alpha_logits = logits[:, 0:1]
    alpha_target = targets[:, 0:1]
    stats = alpha_stats(alpha_logits.detach().cpu(), alpha_target.detach().cpu(), threshold)
    sdf_pred = torch.tanh(logits[:, 1:2]).detach().cpu()
    sdf_target = targets[:, 1:2].detach().cpu()
    edge_pred = torch.sigmoid(logits[:, 2:3]).detach().cpu()
    edge_target = targets[:, 2:3].detach().cpu()
    stats["sdf_mae"] = (sdf_pred - sdf_target).abs().mean().item()
    stats["edge_mae"] = (edge_pred - edge_target).abs().mean().item()
    return stats


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
        "sdf_mae": 0.0,
        "edge_mae": 0.0,
    }
    batches = 0

    progress = tqdm(loader, leave=False)
    for batch in progress:
        images = batch["image"].to(device)
        targets = batch["target"].to(device)

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, targets)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        stats = sdf_stats(logits, targets, threshold)
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
            "alpha_threshold": args.alpha_threshold,
            "feather_radius": args.feather_radius,
            "sdf_radius": args.sdf_radius,
            "boundary_radius": args.boundary_radius,
            "shrink_pixels": args.shrink_pixels,
            "task": "lip_sdf_student",
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
    parser.add_argument("--root", default="training/outputs/datasets/lip_segface512_plus_lip001_manual_v2_clean2_384x256")
    parser.add_argument("--output-dir", default="training/outputs/lip_sdf_student_384x256_w32")
    parser.add_argument("--image-width", type=int, default=384)
    parser.add_argument("--image-height", type=int, default=256)
    parser.add_argument("--model-width", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--feather-radius", type=float, default=2.5)
    parser.add_argument("--sdf-radius", type=float, default=16.0)
    parser.add_argument("--boundary-radius", type=float, default=2.0)
    parser.add_argument("--shrink-pixels", type=float, default=0.0)
    parser.add_argument("--alpha-threshold", type=float, default=0.5)
    parser.add_argument("--alpha-weight", type=float, default=1.0)
    parser.add_argument("--sdf-weight", type=float, default=0.75)
    parser.add_argument("--edge-weight", type=float, default=0.60)
    parser.add_argument("--edge-pos-weight", type=float, default=8.0)
    parser.add_argument("--init-checkpoint", default=None, help="Optional checkpoint to initialize model weights from.")
    parser.add_argument("--resume-checkpoint", default=None, help="Resume full training state from a checkpoint, including optimizer and epoch.")
    parser.add_argument("--best-metric", choices=["mae", "rmse", "loss", "iou", "sdf_mae", "edge_mae"], default="mae")
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=5)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device()
    print(f"Device: {device}")
    print(f"Input: {args.image_width}x{args.image_height}")
    print(f"Targets: feather={args.feather_radius} sdf_radius={args.sdf_radius} boundary={args.boundary_radius}")

    train_dataset = LipSDFDataset(
        args.root,
        "train",
        image_width=args.image_width,
        image_height=args.image_height,
        feather_radius=args.feather_radius,
        sdf_radius=args.sdf_radius,
        boundary_radius=args.boundary_radius,
        shrink_pixels=args.shrink_pixels,
        limit=args.limit_train,
        augment=True,
    )
    val_dataset = LipSDFDataset(
        args.root,
        "val",
        image_width=args.image_width,
        image_height=args.image_height,
        feather_radius=args.feather_radius,
        sdf_radius=args.sdf_radius,
        boundary_radius=args.boundary_radius,
        shrink_pixels=args.shrink_pixels,
        limit=args.limit_val,
        augment=False,
    )
    train_loader = make_loader(train_dataset, args.batch_size, shuffle=True, workers=args.workers)
    val_loader = make_loader(val_dataset, args.batch_size, shuffle=False, workers=args.workers)

    model = build_model(num_classes=NUM_OUTPUT_CHANNELS, width=args.model_width).to(device)
    criterion = LipSDFLoss(args.alpha_weight, args.sdf_weight, args.edge_weight, args.edge_pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    lower_is_better = args.best_metric in {"mae", "rmse", "loss", "sdf_mae", "edge_mae"}
    best_score = float("inf") if lower_is_better else -1.0
    history_path = output_dir / "history.json"
    history: list[dict[str, object]] = []
    start_epoch = 1
    checkpoint_path = args.resume_checkpoint or args.init_checkpoint
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        checkpoint_width = int(checkpoint.get("model_width", args.model_width))
        checkpoint_classes = int(checkpoint.get("num_classes", NUM_OUTPUT_CHANNELS))
        if checkpoint_width != args.model_width or checkpoint_classes != NUM_OUTPUT_CHANNELS:
            raise ValueError(
                "Checkpoint architecture does not match: "
                f"checkpoint width/classes={checkpoint_width}/{checkpoint_classes}, "
                f"requested={args.model_width}/{NUM_OUTPUT_CHANNELS}"
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
        train_metrics = run_epoch(model, train_loader, criterion, device, args.alpha_threshold, optimizer)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, criterion, device, args.alpha_threshold)

        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(
            f"train loss={train_metrics['loss']:.4f} iou={train_metrics['iou']:.4f} "
            f"mae={train_metrics['mae']:.4f} sdf={train_metrics['sdf_mae']:.4f} | "
            f"val loss={val_metrics['loss']:.4f} iou={val_metrics['iou']:.4f} "
            f"mae={val_metrics['mae']:.4f} sdf={val_metrics['sdf_mae']:.4f}"
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
