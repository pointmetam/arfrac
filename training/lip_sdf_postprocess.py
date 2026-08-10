from __future__ import annotations

import cv2
import numpy as np

from PIL import Image, ImageDraw

from lip_contour_dataset import catmull_rom_closed


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=np.uint8)
    if mask.max() == 0:
        return np.zeros(mask.shape, dtype=np.uint8)

    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return mask.astype(np.uint8)

    areas = stats[1:, cv2.CC_STAT_AREA]
    component_id = int(np.argmax(areas) + 1)
    return (labels == component_id).astype(np.uint8)


def cleanup_lip_mask(
    mask: np.ndarray,
    *,
    close_kernel: int = 5,
    open_kernel: int = 3,
) -> np.ndarray:
    mask = largest_connected_component(mask)
    if mask.max() == 0:
        return mask

    if close_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    if open_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return largest_connected_component(mask)


def _resample_contour(contour: np.ndarray, point_count: int, samples_per_segment: int) -> np.ndarray | None:
    points = contour[:, 0, :].astype(np.float32)
    perimeter = cv2.arcLength(contour, closed=True)
    epsilon = max(perimeter * 0.0035, 0.5)
    simplified = cv2.approxPolyDP(contour, epsilon, closed=True)[:, 0, :].astype(np.float32)
    if simplified.shape[0] >= 6:
        points = simplified

    segment_lengths = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    total = float(segment_lengths.sum())
    if total <= 1e-6:
        return None

    resampled = np.zeros((point_count, 2), dtype=np.float32)
    segment_index = 0
    segment_start = 0.0
    for index in range(point_count):
        target = total * index / float(point_count)
        while segment_index < len(points) - 1 and segment_start + segment_lengths[segment_index] < target:
            segment_start += float(segment_lengths[segment_index])
            segment_index += 1
        start = points[segment_index]
        end = points[(segment_index + 1) % len(points)]
        length = float(segment_lengths[segment_index])
        t = 0.0 if length <= 1e-6 else (target - segment_start) / length
        resampled[index] = start + ((end - start) * t)
    return catmull_rom_closed(resampled, samples_per_segment=samples_per_segment)


def _draw_curve_polygon(mask: np.ndarray, curve: np.ndarray, fill: int) -> None:
    curve = np.asarray(curve, dtype=np.float32).copy()
    curve[:, 0] = np.clip(curve[:, 0], 0, mask.shape[1] - 1)
    curve[:, 1] = np.clip(curve[:, 1], 0, mask.shape[0] - 1)
    image = Image.fromarray((mask * 255).astype(np.uint8)).convert("L")
    ImageDraw.Draw(image).polygon([(float(x), float(y)) for x, y in curve], fill=int(fill))
    mask[:] = (np.asarray(image, dtype=np.uint8) > 0).astype(np.uint8)


def smooth_lip_mask(
    mask: np.ndarray,
    *,
    point_count: int = 96,
    samples_per_segment: int = 8,
    close_kernel: int = 5,
    open_kernel: int = 3,
) -> np.ndarray:
    mask = cleanup_lip_mask(mask, close_kernel=close_kernel, open_kernel=open_kernel)
    if mask.max() == 0:
        return mask

    contours, hierarchy = cv2.findContours(mask * 255, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not contours or hierarchy is None:
        return mask

    hierarchy = hierarchy[0]
    outer_indices = [index for index, item in enumerate(hierarchy) if int(item[3]) < 0]
    if not outer_indices:
        return mask

    outer_index = max(outer_indices, key=lambda index: cv2.contourArea(contours[index]))
    outer_curve = _resample_contour(contours[outer_index], point_count, samples_per_segment)
    if outer_curve is None:
        return mask

    output = np.zeros(mask.shape, dtype=np.uint8)
    _draw_curve_polygon(output, outer_curve, fill=255)

    child_index = int(hierarchy[outer_index][2])
    while child_index >= 0:
        if cv2.contourArea(contours[child_index]) >= 4.0:
            hole_curve = _resample_contour(contours[child_index], max(16, point_count // 2), samples_per_segment)
            if hole_curve is not None:
                _draw_curve_polygon(output, hole_curve, fill=0)
        child_index = int(hierarchy[child_index][0])

    return output


def alpha_to_smooth_mask(
    alpha: np.ndarray,
    *,
    threshold: float = 0.5,
    point_count: int = 96,
    close_kernel: int = 5,
    open_kernel: int = 3,
) -> np.ndarray:
    hard = (np.asarray(alpha, dtype=np.float32) >= float(threshold)).astype(np.uint8)
    return smooth_lip_mask(
        hard,
        point_count=point_count,
        close_kernel=close_kernel,
        open_kernel=open_kernel,
    )


def mask_iou(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction).astype(bool)
    target = np.asarray(target).astype(bool)
    union = prediction | target
    if not union.any():
        return 1.0
    return float((prediction & target).sum() / union.sum())
