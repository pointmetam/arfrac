from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lip_curve_dataset import CURVE_COUNT, LipCurveDataset, curves_from_label, curves_to_mask, lip_union_mask_from_label, mask_iou


class LipCurveDatasetTests(unittest.TestCase):
    def test_curves_from_label_returns_ordered_four_curve_lip_structure(self) -> None:
        label_image = Image.new("L", (128, 96), 0)
        draw = ImageDraw.Draw(label_image)
        draw.polygon([(20, 48), (42, 26), (64, 22), (86, 26), (108, 48), (66, 44)], fill=1)
        draw.polygon([(20, 50), (44, 72), (64, 78), (84, 72), (108, 50), (66, 56)], fill=2)
        label = np.asarray(label_image, dtype=np.uint8)

        curves = curves_from_label(label, points_per_curve=24)

        self.assertEqual(curves.shape, (CURVE_COUNT, 24, 2))
        self.assertTrue(np.all(np.diff(curves[:, :, 0], axis=1) >= -1e-4))
        self.assertTrue(np.all(curves[0, :, 1] <= curves[1, :, 1]))
        self.assertTrue(np.all(curves[1, :, 1] <= curves[2, :, 1]))
        self.assertTrue(np.all(curves[2, :, 1] <= curves[3, :, 1]))

        reconstructed = curves_to_mask(curves, width=128, height=96, smooth=True)
        self.assertGreater(mask_iou(reconstructed, lip_union_mask_from_label(label)), 0.65)

    def test_lip_curve_dataset_returns_curves_and_lip_mask(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "lip"
            split_root = root / "train"
            (split_root / "images").mkdir(parents=True)
            (split_root / "labels").mkdir(parents=True)

            Image.new("RGB", (128, 96), (160, 120, 104)).save(split_root / "images" / "sample.jpg")
            label_image = Image.new("L", (128, 96), 0)
            draw = ImageDraw.Draw(label_image)
            draw.ellipse((20, 28, 108, 54), fill=1)
            draw.ellipse((20, 44, 108, 76), fill=2)
            label_image.save(split_root / "labels" / "sample.png")

            dataset = LipCurveDataset(root, "train", image_width=128, image_height=96, points_per_curve=20)
            sample = dataset[0]

            self.assertEqual(len(dataset), 1)
            self.assertEqual(sample["image"].shape, (3, 96, 128))
            self.assertEqual(sample["curves"].shape, (4, 20, 2))
            self.assertEqual(sample["lip_mask"].shape, (1, 96, 128))
            self.assertEqual(sample["curves"].dtype, torch.float32)
            self.assertTrue(torch.all(sample["curves"] >= 0))
            self.assertTrue(torch.all(sample["curves"] <= 1))


if __name__ == "__main__":
    unittest.main()
