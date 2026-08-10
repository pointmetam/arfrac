from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lip_contour_dataset import (
    LipContourDataset,
    contour_points_from_label,
    lip_mask_from_label,
    mask_iou,
    points_to_mask,
)


class LipContourDatasetTests(unittest.TestCase):
    def test_contour_points_from_label_returns_canonical_closed_lip_contour(self) -> None:
        label_image = Image.new("L", (96, 64), 0)
        ImageDraw.Draw(label_image).ellipse((18, 20, 78, 44), fill=1)
        label = np.asarray(label_image, dtype=np.uint8)

        points = contour_points_from_label(label, point_count=32)

        self.assertIsNotNone(points)
        assert points is not None
        self.assertEqual(points.shape, (32, 2))
        self.assertLessEqual(points[0, 0], points[:, 0].min() + 1.0)
        self.assertLess(points[8, 1], points[:, 1].mean())

        reconstructed = points_to_mask(points, width=96, height=64, smooth=True)
        self.assertGreater(mask_iou(reconstructed, lip_mask_from_label(label)), 0.86)

    def test_lip_contour_dataset_returns_image_points_and_lip_mask(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "lip"
            split_root = root / "train"
            (split_root / "images").mkdir(parents=True)
            (split_root / "labels").mkdir(parents=True)

            Image.new("RGB", (96, 64), (160, 120, 104)).save(split_root / "images" / "sample.jpg")
            label_image = Image.new("L", (96, 64), 0)
            ImageDraw.Draw(label_image).ellipse((18, 20, 78, 44), fill=2)
            label_image.save(split_root / "labels" / "sample.png")

            dataset = LipContourDataset(root, "train", image_width=96, image_height=64, point_count=40)
            sample = dataset[0]

            self.assertEqual(len(dataset), 1)
            self.assertEqual(sample["image"].shape, (3, 64, 96))
            self.assertEqual(sample["points"].shape, (40, 2))
            self.assertEqual(sample["lip_mask"].shape, (1, 64, 96))
            self.assertEqual(sample["points"].dtype, torch.float32)
            self.assertTrue(torch.all(sample["points"] >= 0))
            self.assertTrue(torch.all(sample["points"] <= 1))


if __name__ == "__main__":
    unittest.main()
