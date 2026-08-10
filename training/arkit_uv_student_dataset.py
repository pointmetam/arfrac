from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset

from dataset import IMAGENET_MEAN, IMAGENET_STD
from teacher_2d_to_arkit_uv import barycentric_grid, project_vertices


def rasterize_portrait_image_to_uv(
    frame: dict[str, Any],
    portrait: Image.Image,
    uv_size: int,
    cull_backfaces: bool = False,
    front_face_winding: str = "negative",
) -> np.ndarray:
    portrait_points, _, valid_points = project_vertices(frame)
    geometry = frame["faceGeometry"]
    uvs = np.array([[point["x"], point["y"]] for point in geometry["textureCoordinates"]], dtype=np.float64)
    triangle_indices = np.array(geometry["triangleIndices"], dtype=np.int64)

    source = np.asarray(portrait.convert("RGB"), dtype=np.uint8)
    scale = np.array(
        [
            source.shape[1] / float(frame["viewport"]["width"]),
            source.shape[0] / float(frame["viewport"]["height"]),
        ],
        dtype=np.float64,
    )
    source_points = portrait_points * scale
    uv_pixels = uvs * float(uv_size - 1)

    output = np.zeros((uv_size, uv_size, 3), dtype=np.uint8)
    filled = np.zeros((uv_size, uv_size), dtype=bool)

    for offset in range(0, len(triangle_indices) - 2, 3):
        i0, i1, i2 = triangle_indices[offset : offset + 3]
        if not (valid_points[i0] and valid_points[i1] and valid_points[i2]):
            continue

        if cull_backfaces:
            s0, s1, s2 = portrait_points[i0], portrait_points[i1], portrait_points[i2]
            signed_area = ((s1[0] - s0[0]) * (s2[1] - s0[1])) - ((s1[1] - s0[1]) * (s2[0] - s0[0]))
            if front_face_winding == "negative":
                if signed_area >= -1e-6:
                    continue
            elif signed_area <= 1e-6:
                continue

        p0, p1, p2 = uv_pixels[i0], uv_pixels[i1], uv_pixels[i2]
        min_x = max(0, int(np.floor(min(p0[0], p1[0], p2[0]))))
        max_x = min(uv_size - 1, int(np.ceil(max(p0[0], p1[0], p2[0]))))
        min_y = max(0, int(np.floor(min(p0[1], p1[1], p2[1]))))
        max_y = min(uv_size - 1, int(np.ceil(max(p0[1], p1[1], p2[1]))))
        if min_x > max_x or min_y > max_y:
            continue

        w0, w1, w2, inside, indices = barycentric_grid(p0, p1, p2, min_x, max_x, min_y, max_y)
        if inside.size == 0 or not inside.any():
            continue

        src = (
            source_points[i0][None, None, :] * w0[:, :, None]
            + source_points[i1][None, None, :] * w1[:, :, None]
            + source_points[i2][None, None, :] * w2[:, :, None]
        )
        flat_src = src[inside]
        src_x = np.rint(flat_src[:, 0]).astype(np.int64)
        src_y = np.rint(flat_src[:, 1]).astype(np.int64)
        valid_samples = (src_x >= 0) & (src_x < source.shape[1]) & (src_y >= 0) & (src_y < source.shape[0])
        if not valid_samples.any():
            continue

        local_y = indices[0][inside][valid_samples]
        local_x = indices[1][inside][valid_samples]
        target_y = local_y + min_y
        target_x = local_x + min_x
        output[target_y, target_x] = source[src_y[valid_samples], src_x[valid_samples]]
        filled[target_y, target_x] = True

    if filled.any():
        mean_color = output[filled].mean(axis=0).astype(np.uint8)
        output[~filled] = mean_color
    return output


class ARKitUVTeacherDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        uv_size: int = 256,
        image_width: int | None = None,
        image_height: int | None = None,
        crop_box: tuple[int, int, int, int] | None = None,
        limit: int | None = None,
        augment: bool = False,
        cache_uv_images: bool = False,
        cull_backfaces: bool = True,
        front_face_winding: str = "negative",
    ) -> None:
        self.root = Path(root).expanduser()
        self.split = split
        self.uv_size = int(uv_size)
        self.image_width = int(image_width or self.uv_size)
        self.image_height = int(image_height or self.uv_size)
        self.crop_box = tuple(int(value) for value in crop_box) if crop_box is not None else None
        self.augment = augment
        self.cache_uv_images = cache_uv_images
        self.cull_backfaces = cull_backfaces
        self.front_face_winding = front_face_winding
        if self.crop_box is not None:
            x0, y0, x1, y1 = self.crop_box
            if x0 < 0 or y0 < 0 or x1 > self.uv_size or y1 > self.uv_size or x0 >= x1 or y0 >= y1:
                raise ValueError(f"Invalid crop_box {self.crop_box} for uv_size={self.uv_size}")

        split_root = self.root / split
        metadata_path = split_root / "metadata.jsonl"
        image_dir = split_root / "images"
        label_dir = split_root / "uv_labels"
        frame_dir = split_root / "frames"
        if not metadata_path.exists() or not image_dir.exists() or not label_dir.exists() or not frame_dir.exists():
            raise FileNotFoundError(f"Missing ARKit UV teacher split folders under {split_root}")

        samples: list[dict[str, Any]] = []
        for line in metadata_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            stem = str(payload["stem"])
            image_path = image_dir / f"{stem}.jpg"
            label_path = label_dir / f"{stem}.png"
            frame_path = frame_dir / f"{stem}.json"
            if image_path.exists() and label_path.exists() and frame_path.exists():
                samples.append(
                    {
                        "stem": stem,
                        "image_path": image_path,
                        "label_path": label_path,
                        "frame_path": frame_path,
                    }
                )

        if limit is not None:
            samples = samples[:limit]
        if not samples:
            raise RuntimeError(f"No paired ARKit UV teacher samples found in {split_root}")
        self.samples = samples
        cull_suffix = "_culled" if self.cull_backfaces else ""
        self.cache_dir = split_root / f"uv_images_{self.uv_size}{cull_suffix}"
        if self.cache_uv_images:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        portrait = Image.open(sample["image_path"]).convert("RGB")
        label = Image.open(sample["label_path"]).convert("L")
        frame = json.loads(Path(sample["frame_path"]).read_text(encoding="utf-8"))

        uv_image = self._load_or_build_uv_image(str(sample["stem"]), portrait, frame)
        label = label.resize((self.uv_size, self.uv_size), Image.Resampling.NEAREST)

        if self.crop_box is not None:
            uv_image = uv_image.crop(self.crop_box)
            label = label.crop(self.crop_box)

        target_size = (self.image_width, self.image_height)
        if uv_image.size != target_size:
            uv_image = uv_image.resize(target_size, Image.Resampling.BILINEAR)
        if label.size != target_size:
            label = label.resize(target_size, Image.Resampling.NEAREST)

        if self.augment:
            uv_image = self._augment_image(uv_image)

        image_array = np.asarray(uv_image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        image_tensor = (image_tensor - IMAGENET_MEAN) / IMAGENET_STD

        label_array = np.asarray(label, dtype=np.uint8)
        label_tensor = torch.from_numpy(label_array.astype(np.int64))

        return {
            "image": image_tensor,
            "label": label_tensor,
            "stem": str(sample["stem"]),
            "image_path": str(sample["image_path"]),
            "label_path": str(sample["label_path"]),
            "frame_path": str(sample["frame_path"]),
        }

    def _load_or_build_uv_image(self, stem: str, portrait: Image.Image, frame: dict[str, Any]) -> Image.Image:
        cache_path = self.cache_dir / f"{stem}.jpg"
        if self.cache_uv_images and cache_path.exists():
            return Image.open(cache_path).convert("RGB")

        uv_image = Image.fromarray(
            rasterize_portrait_image_to_uv(
                frame,
                portrait,
                self.uv_size,
                cull_backfaces=self.cull_backfaces,
                front_face_winding=self.front_face_winding,
            )
        )
        if self.cache_uv_images:
            uv_image.save(cache_path, quality=92)
        return uv_image

    def _augment_image(self, image: Image.Image) -> Image.Image:
        if random.random() < 0.85:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.72, 1.28))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.76, 1.24))
            image = ImageEnhance.Color(image).enhance(random.uniform(0.82, 1.18))
        return image
