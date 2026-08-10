from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    history_path = run_dir / "history.json"
    if not history_path.exists():
        raise FileNotFoundError(history_path)

    with history_path.open("r", encoding="utf-8") as f:
        history = json.load(f)

    if not history:
        print(f"No epochs recorded yet: {history_path}")
        return

    latest = history[-1]
    best = max(history, key=lambda item: item["val"]["mean_iou"])

    print(f"Run: {run_dir}")
    print(
        "Latest: "
        f"epoch={latest['epoch']} "
        f"train_mIoU={latest['train']['mean_iou']:.4f} "
        f"val_mIoU={latest['val']['mean_iou']:.4f} "
        f"val_loss={latest['val']['loss']:.4f}"
    )
    print(
        "Best:   "
        f"epoch={best['epoch']} "
        f"val_mIoU={best['val']['mean_iou']:.4f} "
        f"val_loss={best['val']['loss']:.4f}"
    )
    print(f"Best checkpoint: {run_dir / 'best.pt'}")
    print(f"Last checkpoint: {run_dir / 'last.pt'}")


if __name__ == "__main__":
    main()
