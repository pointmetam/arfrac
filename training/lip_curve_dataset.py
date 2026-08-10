from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

from lip_dataset import LipRoiDataset


CURVE_NAMES = ("upper_outer", "upper_inner", "lower_inner", "lower_outer")
CURVE_COUNT = len(CURVE_NAMES)
UPPER_LIP_ID = 1
LOWER_LIP_ID = 2
INNER_MOUTH_ID = 3


def lip_union_mask_from_label(label: np.ndarray) -> np.ndarray:
    label = np.asarray(label, dtype=np.uint8)
    return np.isin(label, [UPPER_LIP_ID, LOWER_LIP_ID]).astype(np.uint8)


def _smooth_1d(values: np.ndarray, passes: int = 2) -> np.ndarray:
    output = values.astype(np.float32)
    if output.size < 3:
        return output
    kernel = np.asarray([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float32)
    kernel /= kernel.sum()
    for _ in range(max(0, passes)):
        padded = np.pad(output, (2, 2), mode="edge")
        output = np.convolve(padded, kernel, mode="valid").astype(np.float32)
    return output


def _boundary_y(mask: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray] | None:
    active_columns = np.where(mask.any(axis=0))[0]
    if active_columns.size < 2:
        return None

    if mode == "top":
        ys = np.argmax(mask[:, active_columns], axis=0).astype(np.float32)
    elif mode == "bottom":
        reversed_ys = np.argmax(mask[::-1, active_columns], axis=0)
        ys = (mask.shape[0] - 1 - reversed_ys).astype(np.float32)
    else:
        raise ValueError(f"Unsupported boundary mode: {mode}")

    return active_columns.astype(np.float32), ys


def _sample_boundary(mask: np.ndarray, mode: str, x_values: np.ndarray) -> np.ndarray | None:
    boundary = _boundary_y(mask, mode)
    if boundary is None:
        return None
    xs, ys = boundary
    sampled = np.interp(x_values, xs, ys).astype(np.float32)
    return _smooth_1d(sampled, passes=2)


def _template_curves(width: int, height: int, points_per_curve: int) -> np.ndarray:
    x_values = np.linspace(width * 0.24, width * 0.76, points_per_curve, dtype=np.float32)
    phase = np.linspace(-1.0, 1.0, points_per_curve, dtype=np.float32)
    arch = 1.0 - np.abs(phase)
    center_y = height * 0.50
    upper_outer = center_y - (height * (0.12 + 0.06 * arch))
    upper_inner = center_y - (height * (0.010 + 0.012 * arch))
    lower_inner = center_y + (height * (0.010 + 0.012 * arch))
    lower_outer = center_y + (height * (0.12 + 0.05 * arch))
    return np.stack(
        [
            np.stack([x_values, upper_outer], axis=1),
            np.stack([x_values, upper_inner], axis=1),
            np.stack([x_values, lower_inner], axis=1),
            np.stack([x_values, lower_outer], axis=1),
        ],
        axis=0,
    ).astype(np.float32)


def curves_from_label(label: np.ndarray, points_per_curve: int) -> np.ndarray:
    label = np.asarray(label, dtype=np.uint8)
    height, width = label.shape
    upper = label == UPPER_LIP_ID
    lower = label == LOWER_LIP_ID
    union = upper | lower

    if int(union.sum()) < 16:
        return _template_curves(width, height, points_per_curve)

    active_columns = np.where(union.any(axis=0))[0]
    if active_columns.size < 2:
        return _template_curves(width, height, points_per_curve)

    x_values = np.linspace(float(active_columns.min()), float(active_columns.max()), points_per_curve, dtype=np.float32)
    template = _template_curves(width, height, points_per_curve)
    curves = np.empty((CURVE_COUNT, points_per_curve, 2), dtype=np.float32)
    curves[:, :, 0] = x_values[None, :]

    boundaries = [
        _sample_boundary(upper, "top", x_values),
        _sample_boundary(upper, "bottom", x_values),
        _sample_boundary(lower, "top", x_values),
        _sample_boundary(lower, "bottom", x_values),
    ]
    for index, sampled in enumerate(boundaries):
        curves[index, :, 1] = template[index, :, 1] if sampled is None else sampled

    # Keep curves vertically ordered without forcing open-mouth frames closed.
    curves[1, :, 1] = np.maximum(curves[1, :, 1], curves[0, :, 1] + 1.0)
    curves[2, :, 1] = np.maximum(curves[2, :, 1], curves[1, :, 1])
    curves[3, :, 1] = np.maximum(curves[3, :, 1], curves[2, :, 1] + 1.0)
    curves[:, :, 0] = np.clip(curves[:, :, 0], 0.0, float(width - 1))
    curves[:, :, 1] = np.clip(curves[:, :, 1], 0.0, float(height - 1))
    return curves.astype(np.float32)


def normalize_curves(curves: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = np.asarray([max(width - 1, 1), max(height - 1, 1)], dtype=np.float32)
    return np.clip(curves.astype(np.float32) / scale[None, None, :], 0.0, 1.0)


def denormalize_curves(curves: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = np.asarray([max(width - 1, 1), max(height - 1, 1)], dtype=np.float32)
    return np.clip(curves.astype(np.float32), 0.0, 1.0) * scale[None, None, :]


def catmull_rom_open(points: np.ndarray, samples_per_segment: int = 8) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.shape[0] < 2:
        return points

    padded = np.concatenate([points[:1], points, points[-1:]], axis=0)
    samples: list[np.ndarray] = []
    for index in range(1, padded.shape[0] - 2):
        p0, p1, p2, p3 = padded[index - 1], padded[index], padded[index + 1], padded[index + 2]
        for sample_index in range(max(1, samples_per_segment)):
            t = sample_index / float(max(1, samples_per_segment))
            t2 = t * t
            t3 = t2 * t
            point = 0.5 * (
                (2.0 * p1)
                + ((-p0 + p2) * t)
                + (((2.0 * p0) - (5.0 * p1) + (4.0 * p2) - p3) * t2)
                + ((-p0 + (3.0 * p1) - (3.0 * p2) + p3) * t3)
            )
            samples.append(point)
    samples.append(points[-1])
    return np.asarray(samples, dtype=np.float32)


def curves_to_mask(curves: np.ndarray, width: int, height: int, *, smooth: bool = True) -> np.ndarray:
    curves = np.asarray(curves, dtype=np.float32)
    rendered: list[np.ndarray] = []
    for curve in curves:
        rendered.append(catmull_rom_open(curve, samples_per_segment=8) if smooth else curve)

    outer = np.concatenate([rendered[0], rendered[3][::-1]], axis=0)
    inner = np.concatenate([rendered[1], rendered[2][::-1]], axis=0)
    for points in (outer, inner):
        points[:, 0] = np.clip(points[:, 0], 0.0, float(width - 1))
        points[:, 1] = np.clip(points[:, 1], 0.0, float(height - 1))

    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon([(float(x), float(y)) for x, y in outer], fill=255)
    if float(np.mean(rendered[2][:, 1] - rendered[1][:, 1])) >= 2.0:
        draw.polygon([(float(x), float(y)) for x, y in inner], fill=0)
    return (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)


def mask_iou(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    union = prediction | target
    if not union.any():
        return 1.0
    return float((prediction & target).sum() / union.sum())


class LipCurveDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        image_width: int = 384,
        image_height: int = 256,
        points_per_curve: int = 24,
        limit: int | None = None,
        augment: bool = False,
    ) -> None:
        self.base = LipRoiDataset(
            root,
            split,
            image_width=image_width,
            image_height=image_height,
            limit=limit,
            augment=augment,
        )
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self.points_per_curve = int(points_per_curve)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.base[index]
        label_tensor = sample["label"]
        if not isinstance(label_tensor, torch.Tensor):
            raise TypeError("Expected LipRoiDataset label tensor")

        label = label_tensor.cpu().numpy().astype(np.uint8)
        curves = curves_from_label(label, self.points_per_curve)
        normalized_curves = normalize_curves(curves, self.image_width, self.image_height)
        lip_mask = lip_union_mask_from_label(label).astype(np.float32)

        return {
            **sample,
            "curves": torch.from_numpy(normalized_curves.astype(np.float32)),
            "lip_mask": torch.from_numpy(lip_mask).unsqueeze(0),
        }
