from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

from lip_dataset import LipRoiDataset


LIP_CLASS_IDS = {1, 2}


def _segment_lengths(points: np.ndarray) -> tuple[np.ndarray, float]:
    next_points = np.roll(points, -1, axis=0)
    lengths = np.linalg.norm(next_points - points, axis=1)
    return lengths, float(lengths.sum())


def resample_closed_contour(points: np.ndarray, count: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.shape[0] < 3:
        raise ValueError("A closed contour needs at least 3 points")

    lengths, total = _segment_lengths(points)
    if total <= 1e-6:
        raise ValueError("Cannot resample a zero-length contour")

    output = np.zeros((count, 2), dtype=np.float32)
    step = total / float(count)
    segment_index = 0
    segment_start_distance = 0.0

    for point_index in range(count):
        target = point_index * step
        while segment_index < points.shape[0] - 1 and segment_start_distance + lengths[segment_index] < target:
            segment_start_distance += float(lengths[segment_index])
            segment_index += 1

        start = points[segment_index]
        end = points[(segment_index + 1) % points.shape[0]]
        length = float(lengths[segment_index])
        t = 0.0 if length <= 1e-6 else (target - segment_start_distance) / length
        output[point_index] = start + ((end - start) * t)

    return output


def canonicalize_lip_contour(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    center_y = float(points[:, 1].mean())
    min_x = float(points[:, 0].min())
    left_candidates = np.where(points[:, 0] <= min_x + 1.0)[0]
    start_index = int(left_candidates[np.argmin(np.abs(points[left_candidates, 1] - center_y))])
    rolled = np.roll(points, -start_index, axis=0)

    if rolled.shape[0] >= 8 and rolled[rolled.shape[0] // 4, 1] > center_y:
        rolled = np.concatenate([rolled[:1], rolled[:0:-1]], axis=0)
    return rolled.astype(np.float32)


def lip_mask_from_label(label: np.ndarray) -> np.ndarray:
    label = np.asarray(label, dtype=np.uint8)
    return np.isin(label, list(LIP_CLASS_IDS)).astype(np.uint8)


def contour_points_from_label(label: np.ndarray, point_count: int) -> np.ndarray | None:
    mask = lip_mask_from_label(label)
    if int(mask.sum()) < 9:
        return None

    contours, _hierarchy = cv2.findContours(mask * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    if contour.shape[0] < 3 or cv2.contourArea(contour) < 4.0:
        return None

    points = contour[:, 0, :].astype(np.float32)
    points = canonicalize_lip_contour(points)
    return resample_closed_contour(points, point_count)


def normalize_points(points: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = np.array([max(width - 1, 1), max(height - 1, 1)], dtype=np.float32)
    return np.clip(points.astype(np.float32) / scale[None, :], 0.0, 1.0)


def denormalize_points(points: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = np.array([max(width - 1, 1), max(height - 1, 1)], dtype=np.float32)
    return np.clip(points.astype(np.float32), 0.0, 1.0) * scale[None, :]


def catmull_rom_closed(points: np.ndarray, samples_per_segment: int = 8) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.shape[0] < 3:
        return points

    samples: list[np.ndarray] = []
    segment_samples = max(1, int(samples_per_segment))
    for index in range(points.shape[0]):
        p0 = points[(index - 1) % points.shape[0]]
        p1 = points[index]
        p2 = points[(index + 1) % points.shape[0]]
        p3 = points[(index + 2) % points.shape[0]]
        for sample_index in range(segment_samples):
            t = sample_index / float(segment_samples)
            t2 = t * t
            t3 = t2 * t
            point = 0.5 * (
                (2.0 * p1)
                + ((-p0 + p2) * t)
                + (((2.0 * p0) - (5.0 * p1) + (4.0 * p2) - p3) * t2)
                + ((-p0 + (3.0 * p1) - (3.0 * p2) + p3) * t3)
            )
            samples.append(point)
    return np.asarray(samples, dtype=np.float32)


def points_to_mask(points: np.ndarray, width: int, height: int, *, smooth: bool = True) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).copy()
    if smooth:
        points = catmull_rom_closed(points, samples_per_segment=6)
    points[:, 0] = np.clip(points[:, 0], 0, width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, height - 1)
    mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask).polygon([(float(x), float(y)) for x, y in points], fill=255)
    return (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)


def mask_iou(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    union = prediction | target
    if not union.any():
        return 1.0
    return float((prediction & target).sum() / union.sum())


class LipContourDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        image_width: int = 256,
        image_height: int = 192,
        point_count: int = 64,
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
        self.point_count = int(point_count)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.base[index]
        label_tensor = sample["label"]
        if not isinstance(label_tensor, torch.Tensor):
            raise TypeError("Expected LipRoiDataset label tensor")

        label = label_tensor.cpu().numpy().astype(np.uint8)
        points = contour_points_from_label(label, self.point_count)
        if points is None:
            points = self._fallback_template_points(label)

        normalized_points = normalize_points(points, self.image_width, self.image_height)
        lip_mask = lip_mask_from_label(label).astype(np.float32)

        return {
            **sample,
            "points": torch.from_numpy(normalized_points.astype(np.float32)),
            "lip_mask": torch.from_numpy(lip_mask).unsqueeze(0),
        }

    def _fallback_template_points(self, label: np.ndarray) -> np.ndarray:
        width = self.image_width
        height = self.image_height
        center_x = (width - 1) * 0.5
        center_y = (height - 1) * 0.5
        radius_x = width * 0.28
        radius_y = height * 0.16
        points = []
        for index in range(self.point_count):
            angle = (2.0 * math.pi * index / self.point_count) + math.pi
            points.append((center_x + (math.cos(angle) * radius_x), center_y + (math.sin(angle) * radius_y)))
        return np.asarray(points, dtype=np.float32)
