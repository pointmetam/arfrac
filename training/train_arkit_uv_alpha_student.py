from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from arkit_uv_alpha_dataset import ARKitUVAlphaDataset
from evaluate_arkit_uv_student import parse_crop_box
from model import build_model


NUM_OUTPUT_CHANNELS = 1


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class SoftAlphaLoss(nn.Module):
    def __init__(self, lip_weight: float, edge_weight: float, l1_weight: float) -> None:
        super().__init__()
        self.lip_weight = float(lip_weight)
        self.edge_weight = float(edge_weight)
        self.l1_weight = float(l1_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        alpha = targets.clamp(0, 1)
        edge = 4.0 * alpha * (1.0 - alpha)
        weights = 1.0 + (alpha * self.lip_weight) + (edge * self.edge_weight)
        denominator = weights.sum().clamp_min(1.0)
        loss = (bce * weights).sum() / denominator

        if self.l1_weight > 0:
            probabilities = torch.sigmoid(logits)
            l1 = ((probabilities - targets).abs() * weights).sum() / denominator
            loss = loss + (l1 * self.l1_weight)
        return loss


def alpha_stats(logits: torch.Tensor, targets: torch.Tensor, threshold: float) -> dict[str, float]:
    probabilities = torch.sigmoid(logits).clamp(0, 1)
    targets = targets.clamp(0, 1)
    pred_hard = probabilities >= threshold
    target_hard = targets >= threshold
    intersection = (pred_hard & target_hard).sum().item()
    union = (pred_hard | target_hard).sum().item()
    predicted = pred_hard.sum().item()
    labeled = target_hard.sum().item()
    mae = (probabilities - targets).abs().mean().item()
    rmse = torch.sqrt(((probabilities - targets) ** 2).mean()).item()
    return {
        "mae": mae,
        "rmse": rmse,
        "iou": intersection / union if union > 0 else 1.0,
        "precision": intersection / predicted if predicted > 0 else 0.0,
        "recall": intersection / labeled if labeled > 0 else 0.0,
        "pred_coverage": predicted / max(pred_hard.numel(), 1),
        "target_coverage": labeled / max(target_hard.numel(), 1),
    }


def make_loader(dataset: ARKitUVAlphaDataset, batch_size: int, shuffle: bool, workers: int) -> DataLoader:
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
            "uv_size": args.uv_size,
            "input_width": args.image_width or args.uv_size,
            "input_height": args.image_height or args.uv_size,
            "crop_box": args.crop_box,
            "model_width": args.model_width,
            "alpha_feather_radius": args.alpha_feather_radius,
            "alpha_threshold": args.alpha_threshold,
            "task": "arkit_uv_lip_alpha_student",
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/Users/lee/Datasets/ARKitLipUVTeacher")
    parser.add_argument("--output-dir", default="training/outputs/arkit_uv_alpha_student_128x96_w24")
    parser.add_argument("--uv-size", type=int, default=256)
    parser.add_argument("--image-width", type=int, default=None)
    parser.add_argument("--image-height", type=int, default=None)
    parser.add_argument("--crop-box", type=parse_crop_box, default=None)
    parser.add_argument("--model-width", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--init-checkpoint", default=None, help="Optional checkpoint to initialize model weights from")
    parser.add_argument("--alpha-feather-radius", type=float, default=3.0)
    parser.add_argument("--alpha-threshold", type=float, default=0.5)
    parser.add_argument("--lip-weight", type=float, default=5.0)
    parser.add_argument("--edge-weight", type=float, default=5.0)
    parser.add_argument("--l1-weight", type=float, default=0.35)
    parser.add_argument("--best-metric", choices=["mae", "rmse", "loss", "iou"], default="mae")
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--cache-uv-images", action="store_true")
    parser.add_argument("--cache-alpha-labels", action="store_true")
    parser.add_argument("--no-cull-backfaces", action="store_true")
    parser.add_argument("--front-face-winding", choices=["negative", "positive"], default="negative")
    parser.add_argument("--save-every", type=int, default=5)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device()
    print(f"Device: {device}")
    print(f"Alpha feather radius: {args.alpha_feather_radius}")
    print(f"Loss weights: lip={args.lip_weight} edge={args.edge_weight} l1={args.l1_weight}")

    train_dataset = ARKitUVAlphaDataset(
        args.root,
        "train",
        uv_size=args.uv_size,
        image_width=args.image_width,
        image_height=args.image_height,
        crop_box=args.crop_box,
        feather_radius=args.alpha_feather_radius,
        cache_alpha_labels=args.cache_alpha_labels,
        limit=args.limit_train,
        augment=True,
        cache_uv_images=args.cache_uv_images,
        cull_backfaces=not args.no_cull_backfaces,
        front_face_winding=args.front_face_winding,
    )
    val_dataset = ARKitUVAlphaDataset(
        args.root,
        "val",
        uv_size=args.uv_size,
        image_width=args.image_width,
        image_height=args.image_height,
        crop_box=args.crop_box,
        feather_radius=args.alpha_feather_radius,
        cache_alpha_labels=args.cache_alpha_labels,
        limit=args.limit_val,
        augment=False,
        cache_uv_images=args.cache_uv_images,
        cull_backfaces=not args.no_cull_backfaces,
        front_face_winding=args.front_face_winding,
    )
    train_loader = make_loader(train_dataset, args.batch_size, shuffle=True, workers=args.workers)
    val_loader = make_loader(val_dataset, args.batch_size, shuffle=False, workers=args.workers)

    model = build_model(num_classes=NUM_OUTPUT_CHANNELS, width=args.model_width).to(device)
    if args.init_checkpoint:
        init_checkpoint = torch.load(args.init_checkpoint, map_location="cpu")
        checkpoint_width = int(init_checkpoint.get("model_width", args.model_width))
        checkpoint_classes = int(init_checkpoint.get("num_classes", NUM_OUTPUT_CHANNELS))
        if checkpoint_width != args.model_width or checkpoint_classes != NUM_OUTPUT_CHANNELS:
            raise ValueError(
                "Init checkpoint architecture does not match: "
                f"checkpoint width/classes={checkpoint_width}/{checkpoint_classes}, "
                f"requested={args.model_width}/{NUM_OUTPUT_CHANNELS}"
            )
        model.load_state_dict(init_checkpoint["model_state"])
        print(f"Initialized from: {args.init_checkpoint}")
    criterion = SoftAlphaLoss(args.lip_weight, args.edge_weight, args.l1_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: list[dict[str, object]] = []
    lower_is_better = args.best_metric in {"mae", "rmse", "loss"}
    best_score = float("inf") if lower_is_better else -1.0
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
