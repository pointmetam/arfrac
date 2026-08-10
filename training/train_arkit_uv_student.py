from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from arkit_uv_student_dataset import ARKitUVTeacherDataset
from metrics import SegmentationMeter
from model import build_model


NUM_CLASSES = 4
BEST_METRICS = (
    "loss",
    "mean_iou",
    "foreground_mean_iou",
    "upper_lip_iou",
    "lower_lip_iou",
    "inner_mouth_iou",
    "lip_union_iou",
    "lip_precision",
    "lip_recall",
)
LOWER_IS_BETTER_METRICS = {"loss"}


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


def parse_class_weights(value: str | None) -> list[float] | None:
    if value is None or value.lower() in {"", "none"}:
        return None
    try:
        weights = [float(part) for part in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("--class-weights values must be numbers") from error
    if len(weights) != NUM_CLASSES:
        raise argparse.ArgumentTypeError(f"--class-weights must contain {NUM_CLASSES} comma-separated values")
    return weights


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def is_better_metric(metric: str, score: float, best_score: float) -> bool:
    if metric in LOWER_IS_BETTER_METRICS:
        return score < best_score
    return score > best_score


def initial_best_score(metric: str) -> float:
    if metric in LOWER_IS_BETTER_METRICS:
        return float("inf")
    return -1.0


class BoundaryWeightedCrossEntropy(nn.Module):
    def __init__(self, class_weights: torch.Tensor, boundary_weight: float, label_smoothing: float) -> None:
        super().__init__()
        self.register_buffer("class_weights", class_weights.detach().clone())
        self.boundary_weight = float(boundary_weight)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        pixel_loss = F.cross_entropy(
            logits,
            labels,
            weight=self.class_weights,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        class_weight_map = self.class_weights[labels]

        if self.boundary_weight > 0:
            boundary_multiplier = 1 + (self._label_boundary(labels) * self.boundary_weight)
            pixel_loss = pixel_loss * boundary_multiplier
            class_weight_map = class_weight_map * boundary_multiplier

        denominator = class_weight_map.sum().clamp_min(1)
        return pixel_loss.sum() / denominator

    @staticmethod
    def _label_boundary(labels: torch.Tensor) -> torch.Tensor:
        label_values = labels.float().unsqueeze(1)
        max_values = F.max_pool2d(label_values, kernel_size=3, stride=1, padding=1)
        min_values = -F.max_pool2d(-label_values, kernel_size=3, stride=1, padding=1)
        return (max_values != min_values).squeeze(1).float()


def make_loader(dataset: ARKitUVTeacherDataset, batch_size: int, shuffle: bool, workers: int) -> DataLoader:
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
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    meter = SegmentationMeter(NUM_CLASSES)
    batches = 0

    progress = tqdm(loader, leave=False)
    for batch in progress:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, labels)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += float(loss.detach().cpu())
        meter.update(logits.detach().cpu(), labels.detach().cpu())
        batches += 1
        stats = meter.compute()
        progress.set_postfix(
            loss=f"{total_loss / batches:.4f}",
            fg=f"{stats['foreground_mean_iou']:.4f}",
            lip=f"{stats['lip_union_iou']:.4f}",
        )

    metrics = meter.compute()
    return {"loss": total_loss / max(batches, 1), **metrics}


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
            "best_iou": metrics.get("mean_iou", 0.0),
            "num_classes": NUM_CLASSES,
            "uv_size": args.uv_size,
            "input_width": args.image_width or args.uv_size,
            "input_height": args.image_height or args.uv_size,
            "crop_box": args.crop_box,
            "model_width": args.model_width,
            "task": "arkit_uv_lip_student",
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/Users/lee/Datasets/ARKitLipUVTeacher")
    parser.add_argument("--output-dir", default="training/outputs/arkit_uv_student_256_w16")
    parser.add_argument("--uv-size", type=int, default=256)
    parser.add_argument("--image-width", type=int, default=None)
    parser.add_argument("--image-height", type=int, default=None)
    parser.add_argument("--crop-box", type=parse_crop_box, default=None, help="Optional UV crop as x0,y0,x1,y1")
    parser.add_argument("--model-width", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--cache-uv-images", action="store_true")
    parser.add_argument("--no-cull-backfaces", action="store_true")
    parser.add_argument("--front-face-winding", choices=["negative", "positive"], default="negative")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--class-weights", type=parse_class_weights, default=None)
    parser.add_argument("--boundary-weight", type=float, default=0.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--best-metric", choices=BEST_METRICS, default="foreground_mean_iou")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(__file__).with_name("lip_labels.json"), output_dir / "lip_labels.json")

    device = choose_device()
    print(f"Device: {device}")

    train_dataset = ARKitUVTeacherDataset(
        args.root,
        "train",
        uv_size=args.uv_size,
        image_width=args.image_width,
        image_height=args.image_height,
        crop_box=args.crop_box,
        limit=args.limit_train,
        augment=True,
        cache_uv_images=args.cache_uv_images,
        cull_backfaces=not args.no_cull_backfaces,
        front_face_winding=args.front_face_winding,
    )
    val_dataset = ARKitUVTeacherDataset(
        args.root,
        "val",
        uv_size=args.uv_size,
        image_width=args.image_width,
        image_height=args.image_height,
        crop_box=args.crop_box,
        limit=args.limit_val,
        augment=False,
        cache_uv_images=args.cache_uv_images,
        cull_backfaces=not args.no_cull_backfaces,
        front_face_winding=args.front_face_winding,
    )

    train_loader = make_loader(train_dataset, args.batch_size, shuffle=True, workers=args.workers)
    val_loader = make_loader(val_dataset, args.batch_size, shuffle=False, workers=args.workers)

    model = build_model(num_classes=NUM_CLASSES, width=args.model_width).to(device)
    class_weights = args.class_weights or [0.25, 6.0, 6.0, 3.0]
    print(f"Class weights: {class_weights}")
    weights = torch.tensor(class_weights, device=device)
    if args.boundary_weight > 0:
        print(f"Boundary weight: {args.boundary_weight}")
        if args.label_smoothing > 0:
            print(f"Label smoothing: {args.label_smoothing}")
        criterion = BoundaryWeightedCrossEntropy(weights, args.boundary_weight, args.label_smoothing)
    else:
        if args.label_smoothing > 0:
            print(f"Label smoothing: {args.label_smoothing}")
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: list[dict[str, object]] = []
    best_score = initial_best_score(args.best_metric)
    started_at = time.time()

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, criterion, device)

        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)
        print(
            f"train loss={train_metrics['loss']:.4f} fg={train_metrics['foreground_mean_iou']:.4f} "
            f"lip={train_metrics['lip_union_iou']:.4f} inner={train_metrics['inner_mouth_iou']:.4f} | "
            f"val loss={val_metrics['loss']:.4f} fg={val_metrics['foreground_mean_iou']:.4f} "
            f"lip={val_metrics['lip_union_iou']:.4f} inner={val_metrics['inner_mouth_iou']:.4f}"
        )

        score = val_metrics[args.best_metric]
        is_best = is_better_metric(args.best_metric, score, best_score)
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
