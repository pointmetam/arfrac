from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_label_mapping(labels_path: str | Path | None = None) -> np.ndarray:
    if labels_path is None:
        labels_path = Path(__file__).with_name("labels.json")
    with Path(labels_path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    table = np.zeros(256, dtype=np.uint8)
    for source_id, target_id in data["lapa_to_app"].items():
        table[int(source_id)] = int(target_id)
    return table


class LaPaDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        image_size: int = 256,
        limit: int | None = None,
        augment: bool = False,
    ) -> None:
        self.root = Path(root).expanduser()
        self.split = split
        self.image_size = image_size
        self.augment = augment
        self.label_table = load_label_mapping()

        split_root = self.root / split
        image_dir = split_root / "images"
        label_dir = split_root / "labels"
        if not image_dir.exists() or not label_dir.exists():
            raise FileNotFoundError(f"Missing LaPa split folders under {split_root}")

        images_by_stem = {}
        for ext in IMAGE_EXTENSIONS:
            for path in image_dir.glob(f"*{ext}"):
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

        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        label = label.resize((self.image_size, self.image_size), Image.NEAREST)

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        image_tensor = (image_tensor - IMAGENET_MEAN) / IMAGENET_STD

        label_array = np.asarray(label, dtype=np.uint8)
        label_array = self.label_table[label_array]
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

        if random.random() < 0.8:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.85, 1.15))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.85, 1.15))
            image = ImageEnhance.Color(image).enhance(random.uniform(0.9, 1.1))

        return image, label
