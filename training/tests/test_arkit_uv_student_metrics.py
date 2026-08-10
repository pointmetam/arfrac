from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metrics import SegmentationMeter, segmentation_stats
from train_arkit_uv_student import is_better_metric


def logits_from_predictions(predictions: torch.Tensor, num_classes: int) -> torch.Tensor:
    logits = torch.full(
        (predictions.shape[0], num_classes, predictions.shape[1], predictions.shape[2]),
        -10.0,
        dtype=torch.float32,
    )
    return logits.scatter_(1, predictions.unsqueeze(1), 10.0)


class ARKitUVStudentMetricsTests(unittest.TestCase):
    def test_segmentation_stats_does_not_reward_absent_inner_mouth(self) -> None:
        labels = torch.tensor([[[0, 1, 1], [0, 2, 2]]], dtype=torch.long)
        predictions = torch.tensor([[[0, 1, 0], [0, 2, 2]]], dtype=torch.long)

        stats = segmentation_stats(logits_from_predictions(predictions, 4), labels, 4)

        self.assertAlmostEqual(stats["upper_lip_iou"], 0.5)
        self.assertAlmostEqual(stats["lower_lip_iou"], 1.0)
        self.assertAlmostEqual(stats["inner_mouth_iou"], 0.0)
        self.assertAlmostEqual(stats["foreground_mean_iou"], 0.75)

    def test_segmentation_meter_accumulates_binary_lip_union_metrics(self) -> None:
        labels = torch.tensor([[[0, 1, 1, 2], [0, 2, 3, 3]]], dtype=torch.long)
        predictions = torch.tensor([[[0, 1, 2, 2], [0, 2, 0, 3]]], dtype=torch.long)

        meter = SegmentationMeter(num_classes=4)
        meter.update(logits_from_predictions(predictions, 4), labels)
        stats = meter.compute()

        self.assertAlmostEqual(stats["upper_lip_iou"], 0.5)
        self.assertAlmostEqual(stats["lower_lip_iou"], 2 / 3)
        self.assertAlmostEqual(stats["inner_mouth_iou"], 0.5)
        self.assertAlmostEqual(stats["foreground_mean_iou"], (0.5 + (2 / 3) + 0.5) / 3)
        self.assertAlmostEqual(stats["lip_union_iou"], 1.0)
        self.assertAlmostEqual(stats["lip_precision"], 1.0)
        self.assertAlmostEqual(stats["lip_recall"], 1.0)

    def test_is_better_metric_uses_lower_direction_only_for_loss(self) -> None:
        self.assertTrue(is_better_metric("loss", 0.2, 0.3))
        self.assertFalse(is_better_metric("loss", 0.4, 0.3))
        self.assertTrue(is_better_metric("foreground_mean_iou", 0.6, 0.5))
        self.assertFalse(is_better_metric("foreground_mean_iou", 0.4, 0.5))


if __name__ == "__main__":
    unittest.main()
