from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lip_alpha_dataset import LipAlphaDataset, label_to_lip_sdf_alpha


class LipAlphaDatasetTests(unittest.TestCase):
    def test_label_to_lip_sdf_alpha_feathers_lip_edge_and_suppresses_inner_mouth(self) -> None:
        label = np.zeros((17, 17), dtype=np.uint8)
        label[5:12, 4:13] = 1
        label[8, 7:10] = 3

        alpha = label_to_lip_sdf_alpha(label, feather_radius=2.0, shrink_pixels=0.0)
        shrink = label_to_lip_sdf_alpha(label, feather_radius=2.0, shrink_pixels=1.5)

        self.assertEqual(alpha.shape, label.shape)
        self.assertEqual(alpha.dtype, np.float32)
        self.assertGreater(float(alpha[6, 5]), 0.90)
        self.assertEqual(float(alpha[8, 8]), 0.0)
        self.assertGreater(float(alpha[5, 5]), float(alpha[4, 5]))
        self.assertGreater(float(alpha[4, 5]), 0.0)
        self.assertLess(float(shrink.sum()), float(alpha.sum()))

    def test_lip_alpha_dataset_returns_roi_image_alpha_and_hard_lip_tensors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "lip"
            split_root = root / "train"
            (split_root / "images").mkdir(parents=True)
            (split_root / "labels").mkdir(parents=True)

            image = Image.new("RGB", (8, 8), (160, 120, 104))
            label = np.zeros((8, 8), dtype=np.uint8)
            label[2:6, 2:6] = 2
            label[3, 3] = 3
            image.save(split_root / "images" / "sample.jpg")
            Image.fromarray(label).save(split_root / "labels" / "sample.png")

            dataset = LipAlphaDataset(root, "train", image_width=16, image_height=12, feather_radius=2.0)
            sample = dataset[0]

            self.assertEqual(len(dataset), 1)
            self.assertEqual(sample["image"].shape, (3, 12, 16))
            self.assertEqual(sample["alpha"].shape, (1, 12, 16))
            self.assertEqual(sample["hard_lip"].shape, (1, 12, 16))
            self.assertEqual(sample["alpha"].dtype, torch.float32)
            self.assertEqual(sample["hard_lip"].dtype, torch.float32)
            self.assertEqual(sample["image_path"], str(split_root / "images" / "sample.jpg"))


if __name__ == "__main__":
    unittest.main()
