from __future__ import annotations

import torch


def _to_float(value: torch.Tensor) -> float:
    return float(value.item())


def _safe_divide(numerator: torch.Tensor, denominator: torch.Tensor) -> float:
    if _to_float(denominator) <= 0:
        return 0.0
    return _to_float(numerator / denominator)


def _compute_stats(
    *,
    correct: torch.Tensor,
    total: torch.Tensor,
    intersections: torch.Tensor,
    unions: torch.Tensor,
    lip_intersection: torch.Tensor,
    lip_union: torch.Tensor,
    lip_predicted: torch.Tensor,
    lip_labeled: torch.Tensor,
    num_classes: int,
) -> dict[str, float]:
    per_class_iou: list[float] = []
    present_ious: list[float] = []
    for class_id in range(num_classes):
        union = unions[class_id]
        if _to_float(union) > 0:
            iou = _to_float(intersections[class_id] / union)
            present_ious.append(iou)
            per_class_iou.append(iou)
        else:
            per_class_iou.append(0.0)

    foreground_ious = [
        per_class_iou[class_id]
        for class_id in range(1, num_classes)
        if _to_float(unions[class_id]) > 0
    ]

    return {
        "pixel_accuracy": _safe_divide(correct, total),
        "mean_iou": sum(present_ious) / max(len(present_ious), 1),
        "foreground_mean_iou": sum(foreground_ious) / max(len(foreground_ious), 1),
        "upper_lip_iou": per_class_iou[1] if num_classes > 1 else 0.0,
        "lower_lip_iou": per_class_iou[2] if num_classes > 2 else 0.0,
        "inner_mouth_iou": per_class_iou[3] if num_classes > 3 else 0.0,
        "lip_union_iou": _safe_divide(lip_intersection, lip_union),
        "lip_precision": _safe_divide(lip_intersection, lip_predicted),
        "lip_recall": _safe_divide(lip_intersection, lip_labeled),
    }


class SegmentationMeter:
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.correct = torch.tensor(0.0, dtype=torch.float64)
        self.total = torch.tensor(0.0, dtype=torch.float64)
        self.intersections = torch.zeros(num_classes, dtype=torch.float64)
        self.unions = torch.zeros(num_classes, dtype=torch.float64)
        self.lip_intersection = torch.tensor(0.0, dtype=torch.float64)
        self.lip_union = torch.tensor(0.0, dtype=torch.float64)
        self.lip_predicted = torch.tensor(0.0, dtype=torch.float64)
        self.lip_labeled = torch.tensor(0.0, dtype=torch.float64)

    @torch.no_grad()
    def update(self, logits_or_predictions: torch.Tensor, labels: torch.Tensor) -> None:
        if logits_or_predictions.ndim == labels.ndim + 1:
            predictions = logits_or_predictions.argmax(dim=1)
        else:
            predictions = logits_or_predictions

        predictions = predictions.detach().cpu()
        labels = labels.detach().cpu()
        valid = labels >= 0

        self.correct += ((predictions == labels) & valid).sum().item()
        self.total += valid.sum().item()

        for class_id in range(self.num_classes):
            pred_mask = (predictions == class_id) & valid
            label_mask = (labels == class_id) & valid
            self.intersections[class_id] += (pred_mask & label_mask).sum().item()
            self.unions[class_id] += (pred_mask | label_mask).sum().item()

        pred_lip = ((predictions == 1) | (predictions == 2)) & valid
        label_lip = ((labels == 1) | (labels == 2)) & valid
        self.lip_intersection += (pred_lip & label_lip).sum().item()
        self.lip_union += (pred_lip | label_lip).sum().item()
        self.lip_predicted += pred_lip.sum().item()
        self.lip_labeled += label_lip.sum().item()

    def compute(self) -> dict[str, float]:
        return _compute_stats(
            correct=self.correct,
            total=self.total,
            intersections=self.intersections,
            unions=self.unions,
            lip_intersection=self.lip_intersection,
            lip_union=self.lip_union,
            lip_predicted=self.lip_predicted,
            lip_labeled=self.lip_labeled,
            num_classes=self.num_classes,
        )


@torch.no_grad()
def segmentation_stats(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> dict[str, float]:
    meter = SegmentationMeter(num_classes)
    meter.update(logits, labels)
    return meter.compute()
