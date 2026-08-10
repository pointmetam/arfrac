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

from lip_dataset import LipRoiDataset
from lip_template_dataset import TemplateGuidedLipRoiDataset
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


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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


def initial_best_score(metric: str) -> float:
    if metric in LOWER_IS_BETTER_METRICS:
        return float("inf")
    return -1.0


def is_better_metric(metric: str, score: float, best_score: float) -> bool:
    if metric in LOWER_IS_BETTER_METRICS:
        return score < best_score
    return score > best_score


class BoundaryWeightedCrossEntropy(nn.Module):
    def __init__(
        self,
        class_weights: torch.Tensor,
        boundary_weight: float,
        boundary_radius: int,
        label_smoothing: float,
    ) -> None:
        super().__init__()
        self.register_buffer("class_weights", class_weights.detach().clone())
        self.boundary_weight = float(boundary_weight)
        self.boundary_radius = max(1, int(boundary_radius))
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

    def _label_boundary(self, labels: torch.Tensor) -> torch.Tensor:
        kernel_size = self.boundary_radius * 2 + 1
        label_values = labels.float().unsqueeze(1)
        max_values = F.max_pool2d(label_values, kernel_size=kernel_size, stride=1, padding=self.boundary_radius)
        min_values = -F.max_pool2d(-label_values, kernel_size=kernel_size, stride=1, padding=self.boundary_radius)
        return (max_values != min_values).squeeze(1).float()


def make_loader(dataset: LipRoiDataset | TemplateGuidedLipRoiDataset, batch_size: int, shuffle: bool, workers: int) -> DataLoader:
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

    return {"loss": total_loss / max(batches, 1), **meter.compute()}


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
            "input_width": args.image_width,
            "input_height": args.image_height,
            "input_channels": args.input_channels,
            "template_prior": bool(args.template_prior),
            "model_width": args.model_width,
        },
        path,
    )


def move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def load_history(path: Path, start_epoch: int) -> list[dict[str, object]]:
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8") as f:
        history = json.load(f)

    return [item for item in history if int(item.get("epoch", 0)) < start_epoch]


def best_score_from_history(history: list[dict[str, object]], metric: str) -> float:
    best = initial_best_score(metric)
    for item in history:
        val = item.get("val")
        if isinstance(val, dict) and metric in val:
            score = float(val[metric])
            if is_better_metric(metric, score, best):
                best = score
    return best


def _adapt_stem_input_channels(model: nn.Module, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    target = model.state_dict()
    key = "stem.block.0.weight"
    source_weight = state.get(key)
    target_weight = target.get(key)
    if source_weight is None or target_weight is None or source_weight.shape == target_weight.shape:
        return state

    if source_weight.ndim == 4 and target_weight.ndim == 4 and source_weight.shape[1] < target_weight.shape[1]:
        expanded = torch.zeros_like(target_weight)
        copied_channels = min(source_weight.shape[1], target_weight.shape[1])
        expanded[:, :copied_channels] = source_weight[:, :copied_channels]
        adapted = dict(state)
        adapted[key] = expanded
        return adapted

    return state


def load_initial_model_state(model: nn.Module, checkpoint_path: str | Path) -> None:
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
    model.load_state_dict(_adapt_stem_input_channels(model, checkpoint["model_state"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/Users/lee/Datasets/LaPaLip")
    parser.add_argument("--output-dir", default="training/outputs/lip_smoke")
    parser.add_argument("--image-width", type=int, default=192)
    parser.add_argument("--image-height", type=int, default=128)
    parser.add_argument("--model-width", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--resume", default=None, help="Checkpoint path to continue from.")
    parser.add_argument("--init-checkpoint", default=None, help="Checkpoint path to initialize model weights from.")
    parser.add_argument("--save-every", type=int, default=5, help="Save numbered checkpoint every N epochs.")
    parser.add_argument("--class-weights", type=parse_class_weights, default=None)
    parser.add_argument("--boundary-weight", type=float, default=0.0)
    parser.add_argument("--boundary-radius", type=int, default=1)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--best-metric", choices=BEST_METRICS, default="mean_iou")
    parser.add_argument("--template-prior", action="store_true", help="Append a canonical lip template as a 4th input channel.")
    args = parser.parse_args()
    args.input_channels = 4 if args.template_prior else 3

    if args.resume and args.init_checkpoint:
        raise ValueError("--resume and --init-checkpoint are mutually exclusive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(__file__).with_name("lip_labels.json"), output_dir / "lip_labels.json")

    device = choose_device()
    print(f"Device: {device}")

    dataset_cls = TemplateGuidedLipRoiDataset if args.template_prior else LipRoiDataset
    if args.template_prior:
        print("Template prior: enabled (RGB + canonical lip template)")

    train_dataset = dataset_cls(
        args.root,
        "train",
        image_width=args.image_width,
        image_height=args.image_height,
        limit=args.limit_train,
        augment=True,
    )
    val_dataset = dataset_cls(
        args.root,
        "val",
        image_width=args.image_width,
        image_height=args.image_height,
        limit=args.limit_val,
        augment=False,
    )

    train_loader = make_loader(train_dataset, args.batch_size, shuffle=True, workers=args.workers)
    val_loader = make_loader(val_dataset, args.batch_size, shuffle=False, workers=args.workers)

    model = build_model(num_classes=NUM_CLASSES, width=args.model_width, input_channels=args.input_channels).to(device)
    class_weights = args.class_weights or [0.35, 4.8, 4.8, 2.6]
    print(f"Class weights: {class_weights}")
    weights = torch.tensor(class_weights, device=device)
    if args.boundary_weight > 0:
        print(f"Boundary weight: {args.boundary_weight}")
        print(f"Boundary radius: {max(1, args.boundary_radius)}")
        if args.label_smoothing > 0:
            print(f"Label smoothing: {args.label_smoothing}")
        criterion = BoundaryWeightedCrossEntropy(
            weights,
            boundary_weight=args.boundary_weight,
            boundary_radius=args.boundary_radius,
            label_smoothing=args.label_smoothing,
        )
    else:
        if args.label_smoothing > 0:
            print(f"Label smoothing: {args.label_smoothing}")
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 1
    best_score = initial_best_score(args.best_metric)
    history: list[dict[str, object]] = []

    if args.init_checkpoint:
        load_initial_model_state(model, args.init_checkpoint)
        print(f"Initialized model weights from {args.init_checkpoint}")

    if args.resume:
        resume_path = Path(args.resume)
        checkpoint = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            move_optimizer_state(optimizer, device)
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        history = load_history(output_dir / "history.json", start_epoch)
        checkpoint_best = float(checkpoint.get("best_score", checkpoint.get("best_iou", initial_best_score(args.best_metric))))
        history_best = best_score_from_history(history, args.best_metric)
        best_score = checkpoint_best if is_better_metric(args.best_metric, checkpoint_best, history_best) else history_best
        print(f"Resumed from {resume_path} at epoch {start_epoch}; best val {args.best_metric}={best_score:.4f}")

    started_at = time.time()

    if start_epoch > args.epochs:
        print(f"Checkpoint is already at epoch {start_epoch - 1}; target epochs={args.epochs}. Nothing to train.")
        return

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, criterion, device)

        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)
        print(
            f"train loss={train_metrics['loss']:.4f} miou={train_metrics['mean_iou']:.4f} "
            f"fg={train_metrics['foreground_mean_iou']:.4f} lip={train_metrics['lip_union_iou']:.4f} | "
            f"val loss={val_metrics['loss']:.4f} miou={val_metrics['mean_iou']:.4f} "
            f"fg={val_metrics['foreground_mean_iou']:.4f} lip={val_metrics['lip_union_iou']:.4f}"
        )

        score = val_metrics[args.best_metric]
        is_best = is_better_metric(args.best_metric, score, best_score)
        if is_best:
            best_score = score

        save_checkpoint(output_dir / "last.pt", model, optimizer, epoch, args, val_metrics, best_score)
        if is_best:
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, args, val_metrics, best_score)

        if args.save_every > 0 and (epoch % args.save_every == 0 or epoch == args.epochs):
            save_checkpoint(
                output_dir / f"epoch_{epoch:03d}.pt",
                model,
                optimizer,
                epoch,
                args,
                val_metrics,
                best_score,
            )

        with (output_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    elapsed = time.time() - started_at
    print(f"\nDone in {elapsed / 60:.1f} min. Best val {args.best_metric}: {best_score:.4f}")
    print(f"Best checkpoint: {output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
