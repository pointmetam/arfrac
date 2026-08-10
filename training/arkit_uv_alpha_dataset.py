from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from arkit_uv_student_dataset import ARKitUVTeacherDataset


def _edt_1d(values: np.ndarray) -> np.ndarray:
    """Squared Euclidean distance transform for one 1D array."""
    values = values.astype(np.float64, copy=False)
    count = values.shape[0]
    positions = np.zeros(count, dtype=np.int64)
    boundaries = np.zeros(count + 1, dtype=np.float64)
    output = np.zeros(count, dtype=np.float64)

    active = 0
    positions[0] = 0
    boundaries[0] = -np.inf
    boundaries[1] = np.inf

    for query in range(1, count):
        while True:
            previous = positions[active]
            intersection = (
                (values[query] + query * query) - (values[previous] + previous * previous)
            ) / (2 * query - 2 * previous)
            if intersection > boundaries[active]:
                break
            active -= 1
        active += 1
        positions[active] = query
        boundaries[active] = intersection
        boundaries[active + 1] = np.inf

    active = 0
    for query in range(count):
        while boundaries[active + 1] < query:
            active += 1
        nearest = positions[active]
        output[query] = (query - nearest) * (query - nearest) + values[nearest]
    return output


def distance_to_mask(mask: np.ndarray) -> np.ndarray:
    """Euclidean distance from every pixel to the nearest True pixel."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.full(mask.shape, 1e3, dtype=np.float32)

    large = float(mask.shape[0] * mask.shape[0] + mask.shape[1] * mask.shape[1] + 1)
    grid = np.where(mask, 0.0, large).astype(np.float64)
    vertical = np.empty_like(grid)
    for x in range(grid.shape[1]):
        vertical[:, x] = _edt_1d(grid[:, x])

    squared = np.empty_like(vertical)
    for y in range(vertical.shape[0]):
        squared[y, :] = _edt_1d(vertical[y, :])

    return np.sqrt(squared).astype(np.float32)


def label_to_lip_alpha(
    label: np.ndarray,
    feather_radius: float = 3.0,
    suppress_inner_mouth: bool = True,
) -> np.ndarray:
    """Convert hard 4-class UV labels into a continuous lipstick alpha target."""
    label = np.asarray(label, dtype=np.uint8)
    lip_mask = (label == 1) | (label == 2)
    if not lip_mask.any():
        return np.zeros(label.shape, dtype=np.float32)

    radius = max(float(feather_radius), 0.25)
    distance_to_lip = distance_to_mask(lip_mask)
    distance_to_background = distance_to_mask(~lip_mask)
    signed_distance = distance_to_background - distance_to_lip
    alpha = np.clip((signed_distance + radius) / (radius * 2.0), 0.0, 1.0)
    alpha = alpha * alpha * (3.0 - (2.0 * alpha))
    if suppress_inner_mouth:
        alpha[label == 3] = 0.0
    return alpha.astype(np.float32)


class ARKitUVAlphaDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        uv_size: int = 256,
        image_width: int | None = None,
        image_height: int | None = None,
        crop_box: tuple[int, int, int, int] | None = None,
        feather_radius: float = 3.0,
        suppress_inner_mouth: bool = True,
        cache_alpha_labels: bool = False,
        limit: int | None = None,
        augment: bool = False,
        cache_uv_images: bool = False,
        cull_backfaces: bool = True,
        front_face_winding: str = "negative",
    ) -> None:
        self.base = ARKitUVTeacherDataset(
            root,
            split,
            uv_size=uv_size,
            image_width=image_width,
            image_height=image_height,
            crop_box=crop_box,
            limit=limit,
            augment=augment,
            cache_uv_images=cache_uv_images,
            cull_backfaces=cull_backfaces,
            front_face_winding=front_face_winding,
        )
        self.feather_radius = float(feather_radius)
        self.suppress_inner_mouth = suppress_inner_mouth
        self.cache_alpha_labels = cache_alpha_labels
        self.alpha_cache_dir = self.base.root / self.base.split / self._alpha_cache_folder_name()
        if self.cache_alpha_labels:
            self.alpha_cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.base[index]
        label_tensor = sample["label"]
        if not isinstance(label_tensor, torch.Tensor):
            raise TypeError("Expected ARKitUVTeacherDataset label tensor")

        label = label_tensor.cpu().numpy().astype(np.uint8)
        alpha = self._load_or_build_alpha(str(sample["stem"]), label)
        return {
            **sample,
            "alpha": torch.from_numpy(alpha).unsqueeze(0),
            "hard_lip": torch.from_numpy(((label == 1) | (label == 2)).astype(np.float32)).unsqueeze(0),
        }

    def _alpha_cache_folder_name(self) -> str:
        crop = "full"
        if self.base.crop_box is not None:
            crop = "-".join(str(value) for value in self.base.crop_box)
        mouth = "mouth0" if self.suppress_inner_mouth else "mouthsoft"
        feather = f"f{self.feather_radius:g}".replace(".", "p")
        return f"alpha_labels_{self.base.image_width}x{self.base.image_height}_{crop}_{feather}_{mouth}"

    def _load_or_build_alpha(self, stem: str, label: np.ndarray) -> np.ndarray:
        cache_path = self.alpha_cache_dir / f"{stem}.png"
        if self.cache_alpha_labels and cache_path.exists():
            return (np.asarray(Image.open(cache_path).convert("L"), dtype=np.float32) / 255.0).astype(np.float32)

        alpha = label_to_lip_alpha(
            label,
            feather_radius=self.feather_radius,
            suppress_inner_mouth=self.suppress_inner_mouth,
        )
        if self.cache_alpha_labels:
            Image.fromarray(np.clip(alpha * 255.0, 0, 255).astype(np.uint8)).save(cache_path)
        return alpha
