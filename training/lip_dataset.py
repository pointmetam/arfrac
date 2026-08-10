from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset

from dataset import IMAGE_EXTENSIONS, IMAGENET_MEAN, IMAGENET_STD


class LipRoiDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        image_width: int = 192,
        image_height: int = 128,
        limit: int | None = None,
        augment: bool = False,
    ) -> None:
        self.root = Path(root).expanduser()
        self.split = split
        self.image_width = image_width
        self.image_height = image_height
        self.augment = augment

        split_root = self.root / split
        image_dir = split_root / "images"
        label_dir = split_root / "labels"
        if not image_dir.exists() or not label_dir.exists():
            raise FileNotFoundError(f"Missing lip ROI split folders under {split_root}")

        images_by_stem: dict[str, Path] = {}
        for extension in IMAGE_EXTENSIONS:
            for path in image_dir.glob(f"*{extension}"):
                images_by_stem[path.stem] = path

        samples: list[tuple[Path, Path]] = []
        for label_path in sorted(label_dir.glob("*.png")):
            image_path = images_by_stem.get(label_path.stem)
            if image_path is not None:
                samples.append((image_path, label_path))

        if limit is not None:
            samples = samples[:limit]
        if not samples:
            raise RuntimeError(f"No paired image/label samples found in {split_root}")

        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        image_path, label_path = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        label = Image.open(label_path).convert("L")

        if self.augment:
            image, label = self._augment(image, label)

        image = image.resize((self.image_width, self.image_height), Image.BILINEAR)
        label = label.resize((self.image_width, self.image_height), Image.NEAREST)

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        image_tensor = (image_tensor - IMAGENET_MEAN) / IMAGENET_STD

        label_array = np.asarray(label, dtype=np.uint8)
        label_tensor = torch.from_numpy(label_array.astype(np.int64))

        return {
            "image": image_tensor,
            "label": label_tensor,
            "image_path": str(image_path),
            "label_path": str(label_path),
        }

    def _augment(self, image: Image.Image, label: Image.Image) -> tuple[Image.Image, Image.Image]:
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)

        if random.random() < 0.85:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.72, 1.28))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.76, 1.24))
            image = ImageEnhance.Color(image).enhance(random.uniform(0.82, 1.18))

        if random.random() < 0.35:
            angle = random.uniform(-7.0, 7.0)
            image = image.rotate(angle, resample=Image.BILINEAR, fillcolor=(128, 128, 128))
            label = label.rotate(angle, resample=Image.NEAREST, fillcolor=0)

        return image, label
