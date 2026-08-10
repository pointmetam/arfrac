from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from arkit_uv_alpha_dataset import distance_to_mask
from lip_dataset import LipRoiDataset


def _smoothstep(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - (2.0 * value))


@dataclass(frozen=True)
class LipTemplatePriorConfig:
    scale_x: float = 1.0
    scale_y: float = 1.0
    shift_x: float = 0.0
    shift_y: float = 0.0
    feather_radius: float = 3.0


def canonical_lip_template(
    width: int,
    height: int,
    config: LipTemplatePriorConfig | None = None,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    shift_x: float = 0.0,
    shift_y: float = 0.0,
    feather_radius: float = 3.0,
) -> np.ndarray:
    """Build a soft canonical lip prior in normalized ROI coordinates."""
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")

    config = config or LipTemplatePriorConfig(
        scale_x=scale_x,
        scale_y=scale_y,
        shift_x=shift_x,
        shift_y=shift_y,
        feather_radius=feather_radius,
    )
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    center_x = (width - 1) * 0.5 + config.shift_x
    split_y = (height - 1) * 0.51 + config.shift_y

    radius_x = max(width * 0.34 * config.scale_x, 2.0)
    upper_radius_y = max(height * 0.18 * config.scale_y, 2.0)
    lower_radius_y = max(height * 0.23 * config.scale_y, 2.0)
    upper_center_y = split_y - (height * 0.085 * config.scale_y)
    lower_center_y = split_y + (height * 0.115 * config.scale_y)

    normalized_x = ((x - center_x) / radius_x) ** 2
    upper = normalized_x + ((y - upper_center_y) / upper_radius_y) ** 2 <= 1.0
    lower = normalized_x + ((y - lower_center_y) / lower_radius_y) ** 2 <= 1.0
    upper &= y <= split_y + (upper_radius_y * 0.24)
    lower &= y >= split_y - (lower_radius_y * 0.22)

    mask = upper | lower
    if not mask.any():
        return np.zeros((height, width), dtype=np.float32)

    radius = max(float(config.feather_radius), 0.25)
    distance_to_lip = distance_to_mask(mask)
    distance_to_background = distance_to_mask(~mask)
    signed_distance = distance_to_background - distance_to_lip
    return _smoothstep((signed_distance + radius) / (radius * 2.0)).astype(np.float32)


def random_lip_template_config(width: int, height: int) -> LipTemplatePriorConfig:
    return LipTemplatePriorConfig(
        scale_x=random.uniform(0.90, 1.10),
        scale_y=random.uniform(0.84, 1.16),
        shift_x=random.uniform(-0.045, 0.045) * width,
        shift_y=random.uniform(-0.055, 0.055) * height,
        feather_radius=random.uniform(2.0, 4.0),
    )


class TemplateGuidedLipRoiDataset(Dataset):
    input_channels = 4

    def __init__(
        self,
        root: str | Path,
        split: str,
        image_width: int = 192,
        image_height: int = 128,
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
        self.augment = augment

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.base[index]
        image = sample["image"]
        if not isinstance(image, torch.Tensor):
            raise TypeError("Expected LipRoiDataset image tensor")

        config = random_lip_template_config(self.image_width, self.image_height) if self.augment else None
        template = canonical_lip_template(self.image_width, self.image_height, config=config)
        template_tensor = torch.from_numpy(template).unsqueeze(0)

        return {
            **sample,
            "image": torch.cat([image, template_tensor], dim=0),
            "template": template_tensor,
        }
