from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from arkit_uv_alpha_dataset import distance_to_mask
from lip_dataset import LipRoiDataset


def _smoothstep(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - (2.0 * value))


def label_to_lip_sdf_alpha(
    label: np.ndarray,
    feather_radius: float = 3.0,
    shrink_pixels: float = 0.0,
    suppress_inner_mouth: bool = True,
) -> np.ndarray:
    """Convert 4-class lip labels into a continuous lipstick alpha target."""
    label = np.asarray(label, dtype=np.uint8)
    lip_mask = (label == 1) | (label == 2)
    if not lip_mask.any():
        return np.zeros(label.shape, dtype=np.float32)

    radius = max(float(feather_radius), 0.25)
    distance_to_lip = distance_to_mask(lip_mask)
    distance_to_background = distance_to_mask(~lip_mask)
    signed_distance = distance_to_background - distance_to_lip - float(shrink_pixels)
    alpha = _smoothstep((signed_distance + radius) / (radius * 2.0))
    if suppress_inner_mouth:
        alpha[label == 3] = 0.0
    return alpha.astype(np.float32)


class LipAlphaDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        image_width: int = 192,
        image_height: int = 128,
        feather_radius: float = 3.0,
        shrink_pixels: float = 0.0,
        suppress_inner_mouth: bool = True,
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
        self.feather_radius = float(feather_radius)
        self.shrink_pixels = float(shrink_pixels)
        self.suppress_inner_mouth = suppress_inner_mouth

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.base[index]
        label_tensor = sample["label"]
        if not isinstance(label_tensor, torch.Tensor):
            raise TypeError("Expected LipRoiDataset label tensor")

        label = label_tensor.cpu().numpy().astype(np.uint8)
        alpha = label_to_lip_sdf_alpha(
            label,
            feather_radius=self.feather_radius,
            shrink_pixels=self.shrink_pixels,
            suppress_inner_mouth=self.suppress_inner_mouth,
        )
        hard_lip = ((label == 1) | (label == 2)).astype(np.float32)

        return {
            **sample,
            "alpha": torch.from_numpy(alpha).unsqueeze(0),
            "hard_lip": torch.from_numpy(hard_lip).unsqueeze(0),
        }
