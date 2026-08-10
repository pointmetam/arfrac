from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lip_sdf_dataset import LipSDFDataset, label_to_lip_sdf_targets


class LipSDFDatasetTests(unittest.TestCase):
    def test_label_to_lip_sdf_targets_returns_alpha_sdf_edge(self) -> None:
        label = np.zeros((21, 21), dtype=np.uint8)
        label[6:15, 5:16] = 1
        label[10, 8:13] = 3

        targets = label_to_lip_sdf_targets(label, feather_radius=2.0, sdf_radius=8.0, boundary_radius=1.5)

        self.assertEqual(set(targets), {"alpha", "sdf", "edge", "hard_lip"})
        self.assertEqual(targets["alpha"].shape, label.shape)
        self.assertEqual(targets["sdf"].shape, label.shape)
        self.assertEqual(targets["edge"].shape, label.shape)
        self.assertEqual(targets["alpha"].dtype, np.float32)
        self.assertGreater(float(targets["alpha"][7, 7]), 0.9)
        self.assertLess(float(targets["alpha"][2, 2]), 0.1)
        self.assertGreater(float(targets["sdf"][7, 7]), 0.0)
        self.assertLess(float(targets["sdf"][2, 2]), 0.0)
        self.assertGreater(float(targets["edge"][6, 7]), float(targets["edge"][1, 1]))
        self.assertEqual(float(targets["alpha"][10, 10]), 0.0)

    def test_lip_sdf_dataset_returns_three_channel_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "lip"
            split_root = root / "train"
            (split_root / "images").mkdir(parents=True)
            (split_root / "labels").mkdir(parents=True)

            image = Image.new("RGB", (16, 16), (160, 120, 104))
            label = np.zeros((16, 16), dtype=np.uint8)
            label[5:11, 4:12] = 2
            image.save(split_root / "images" / "sample.jpg")
            Image.fromarray(label).save(split_root / "labels" / "sample.png")

            dataset = LipSDFDataset(root, "train", image_width=32, image_height=24)
            sample = dataset[0]

            self.assertEqual(len(dataset), 1)
            self.assertEqual(sample["image"].shape, (3, 24, 32))
            self.assertEqual(sample["target"].shape, (3, 24, 32))
            self.assertEqual(sample["alpha"].shape, (1, 24, 32))
            self.assertEqual(sample["sdf"].shape, (1, 24, 32))
            self.assertEqual(sample["edge"].shape, (1, 24, 32))
            self.assertEqual(sample["target"].dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
