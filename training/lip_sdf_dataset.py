from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from arkit_uv_alpha_dataset import distance_to_mask
from lip_dataset import LipRoiDataset


def smoothstep(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - (2.0 * value))


def lip_mask_from_label(label: np.ndarray) -> np.ndarray:
    label = np.asarray(label, dtype=np.uint8)
    return ((label == 1) | (label == 2)).astype(np.uint8)


def signed_distance_from_lip_mask(lip_mask: np.ndarray) -> np.ndarray:
    lip_mask = np.asarray(lip_mask, dtype=bool)
    if not lip_mask.any():
        return np.full(lip_mask.shape, -1e3, dtype=np.float32)

    distance_to_lip = distance_to_mask(lip_mask)
    distance_to_background = distance_to_mask(~lip_mask)
    return (distance_to_background - distance_to_lip).astype(np.float32)


def label_to_lip_sdf_targets(
    label: np.ndarray,
    feather_radius: float = 3.0,
    sdf_radius: float = 16.0,
    boundary_radius: float = 2.0,
    shrink_pixels: float = 0.0,
    suppress_inner_mouth: bool = True,
) -> dict[str, np.ndarray]:
    label = np.asarray(label, dtype=np.uint8)
    lip_mask = lip_mask_from_label(label).astype(bool)
    signed_distance = signed_distance_from_lip_mask(lip_mask) - float(shrink_pixels)

    feather = max(float(feather_radius), 0.25)
    alpha = smoothstep((signed_distance + feather) / (feather * 2.0)).astype(np.float32)

    sdf = np.clip(signed_distance / max(float(sdf_radius), 1.0), -1.0, 1.0).astype(np.float32)
    edge = np.exp(-((np.abs(signed_distance) / max(float(boundary_radius), 0.25)) ** 2)).astype(np.float32)

    if suppress_inner_mouth:
        alpha[label == 3] = 0.0
        sdf[label == 3] = np.minimum(sdf[label == 3], 0.0)

    return {
        "alpha": alpha,
        "sdf": sdf,
        "edge": edge,
        "hard_lip": lip_mask.astype(np.float32),
    }


class LipSDFDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        image_width: int = 384,
        image_height: int = 256,
        feather_radius: float = 3.0,
        sdf_radius: float = 16.0,
        boundary_radius: float = 2.0,
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
        self.sdf_radius = float(sdf_radius)
        self.boundary_radius = float(boundary_radius)
        self.shrink_pixels = float(shrink_pixels)
        self.suppress_inner_mouth = bool(suppress_inner_mouth)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.base[index]
        label_tensor = sample["label"]
        if not isinstance(label_tensor, torch.Tensor):
            raise TypeError("Expected LipRoiDataset label tensor")

        label = label_tensor.cpu().numpy().astype(np.uint8)
        targets = label_to_lip_sdf_targets(
            label,
            feather_radius=self.feather_radius,
            sdf_radius=self.sdf_radius,
            boundary_radius=self.boundary_radius,
            shrink_pixels=self.shrink_pixels,
            suppress_inner_mouth=self.suppress_inner_mouth,
        )

        return {
            **sample,
            "alpha": torch.from_numpy(targets["alpha"]).unsqueeze(0),
            "sdf": torch.from_numpy(targets["sdf"]).unsqueeze(0),
            "edge": torch.from_numpy(targets["edge"]).unsqueeze(0),
            "hard_lip": torch.from_numpy(targets["hard_lip"]).unsqueeze(0),
            "target": torch.from_numpy(
                np.stack([targets["alpha"], targets["sdf"], targets["edge"]], axis=0).astype(np.float32)
            ),
        }
