from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import LaPaDataset
from metrics import segmentation_stats
from model import build_model


NUM_CLASSES = 9


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_loader(dataset: LaPaDataset, batch_size: int, shuffle: bool, workers: int) -> DataLoader:
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
    total_accuracy = 0.0
    total_iou = 0.0
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

        stats = segmentation_stats(logits.detach().cpu(), labels.detach().cpu(), NUM_CLASSES)
        total_loss += float(loss.detach().cpu())
        total_accuracy += stats["pixel_accuracy"]
        total_iou += stats["mean_iou"]
        batches += 1
        progress.set_postfix(loss=f"{total_loss / batches:.4f}", miou=f"{total_iou / batches:.4f}")

    return {
        "loss": total_loss / max(batches, 1),
        "pixel_accuracy": total_accuracy / max(batches, 1),
        "mean_iou": total_iou / max(batches, 1),
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
    metrics: dict[str, float],
    best_iou: float,
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "args": vars(args),
            "metrics": metrics,
            "best_iou": best_iou,
            "num_classes": NUM_CLASSES,
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

    return [
        item for item in history
        if int(item.get("epoch", 0)) < start_epoch
    ]


def best_iou_from_history(history: list[dict[str, object]]) -> float:
    best = -1.0
    for item in history:
        val = item.get("val")
        if isinstance(val, dict) and "mean_iou" in val:
            best = max(best, float(val["mean_iou"]))
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/Users/lee/Datasets/LaPa")
    parser.add_argument("--output-dir", default="training/outputs/lapa_smoke")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--resume", default=None, help="Checkpoint path to continue from.")
    parser.add_argument("--save-every", type=int, default=5, help="Save numbered checkpoint every N epochs.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(__file__).with_name("labels.json"), output_dir / "labels.json")

    device = choose_device()
    print(f"Device: {device}")

    train_dataset = LaPaDataset(
        args.root,
        "train",
        image_size=args.image_size,
        limit=args.limit_train,
        augment=True,
    )
    val_dataset = LaPaDataset(
        args.root,
        "val",
        image_size=args.image_size,
        limit=args.limit_val,
        augment=False,
    )

    train_loader = make_loader(train_dataset, args.batch_size, shuffle=True, workers=args.workers)
    val_loader = make_loader(val_dataset, args.batch_size, shuffle=False, workers=args.workers)

    model = build_model(num_classes=NUM_CLASSES).to(device)
    weights = torch.tensor([0.35, 0.8, 2.0, 2.0, 1.4, 4.5, 4.5, 3.5, 1.5], device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 1
    best_iou = -1.0
    history: list[dict[str, object]] = []

    if args.resume:
        resume_path = Path(args.resume)
        checkpoint = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            move_optimizer_state(optimizer, device)
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        history = load_history(output_dir / "history.json", start_epoch)
        best_iou = max(
            float(checkpoint.get("best_iou", -1.0)),
            best_iou_from_history(history),
        )
        print(f"Resumed from {resume_path} at epoch {start_epoch}; best val mIoU={best_iou:.4f}")

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
            f"train loss={train_metrics['loss']:.4f} miou={train_metrics['mean_iou']:.4f} | "
            f"val loss={val_metrics['loss']:.4f} miou={val_metrics['mean_iou']:.4f}"
        )

        is_best = val_metrics["mean_iou"] > best_iou
        if is_best:
            best_iou = val_metrics["mean_iou"]

        save_checkpoint(output_dir / "last.pt", model, optimizer, epoch, args, val_metrics, best_iou)
        if is_best:
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, args, val_metrics, best_iou)

        if args.save_every > 0 and (epoch % args.save_every == 0 or epoch == args.epochs):
            save_checkpoint(
                output_dir / f"epoch_{epoch:03d}.pt",
                model,
                optimizer,
                epoch,
                args,
                val_metrics,
                best_iou,
            )

        with (output_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    elapsed = time.time() - started_at
    print(f"\nDone in {elapsed / 60:.1f} min. Best val mIoU: {best_iou:.4f}")
    print(f"Best checkpoint: {output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
