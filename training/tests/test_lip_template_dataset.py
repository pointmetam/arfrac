from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lip_template_dataset import TemplateGuidedLipRoiDataset, canonical_lip_template


class LipTemplateDatasetTests(unittest.TestCase):
    def test_canonical_lip_template_is_soft_centered_prior(self) -> None:
        template = canonical_lip_template(width=16, height=12, feather_radius=2.0)

        self.assertEqual(template.shape, (12, 16))
        self.assertEqual(template.dtype, np.float32)
        self.assertGreaterEqual(float(template.min()), 0.0)
        self.assertLessEqual(float(template.max()), 1.0)
        self.assertGreater(float(template[6, 8]), float(template[0, 0]))

    def test_template_guided_lip_roi_dataset_appends_prior_channel(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "lip"
            split_root = root / "train"
            (split_root / "images").mkdir(parents=True)
            (split_root / "labels").mkdir(parents=True)

            Image.new("RGB", (8, 8), (160, 120, 104)).save(split_root / "images" / "sample.jpg")
            label = np.zeros((8, 8), dtype=np.uint8)
            label[2:5, 2:6] = 1
            Image.fromarray(label).save(split_root / "labels" / "sample.png")

            dataset = TemplateGuidedLipRoiDataset(root, "train", image_width=16, image_height=12, augment=False)
            sample = dataset[0]

            self.assertEqual(len(dataset), 1)
            self.assertEqual(tuple(sample["image"].shape), (4, 12, 16))
            self.assertEqual(tuple(sample["template"].shape), (1, 12, 16))
            self.assertEqual(sample["template"].dtype, torch.float32)
            self.assertTrue(torch.equal(sample["image"][3:4], sample["template"]))
            self.assertEqual(tuple(sample["label"].shape), (12, 16))
            self.assertEqual(sample["image_path"], str(split_root / "images" / "sample.jpg"))


if __name__ == "__main__":
    unittest.main()
