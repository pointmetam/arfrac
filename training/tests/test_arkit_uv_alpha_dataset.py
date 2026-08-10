from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from arkit_uv_alpha_dataset import ARKitUVAlphaDataset, label_to_lip_alpha
from test_arkit_uv_student_dataset import synthetic_frame


class ARKitUVAlphaDatasetTests(unittest.TestCase):
    def test_label_to_lip_alpha_creates_soft_boundary_and_mouth_hole(self) -> None:
        label = np.zeros((9, 9), dtype=np.uint8)
        label[2:7, 2:7] = 1
        label[4, 4] = 3

        alpha = label_to_lip_alpha(label, feather_radius=2.0)

        self.assertEqual(alpha.shape, label.shape)
        self.assertEqual(alpha.dtype, np.float32)
        self.assertGreater(float(alpha[3, 3]), 0.90)
        self.assertEqual(float(alpha[4, 4]), 0.0)
        self.assertGreater(float(alpha[2, 2]), float(alpha[1, 1]))
        self.assertGreater(float(alpha[1, 2]), 0.0)
        self.assertEqual(float(alpha[0, 0]), 0.0)

    def test_arkit_uv_alpha_dataset_returns_alpha_tensor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            split_root = root / "train"
            for folder in ("images", "uv_labels", "frames"):
                (split_root / folder).mkdir(parents=True)

            stem = "session_frame_000000"
            Image.new("RGB", (4, 4), (128, 64, 32)).save(split_root / "images" / f"{stem}.jpg")
            Image.fromarray(
                np.array([[0, 1, 1, 0], [0, 1, 2, 0], [0, 2, 3, 0], [0, 0, 0, 0]], dtype=np.uint8)
            ).save(split_root / "uv_labels" / f"{stem}.png")
            (split_root / "frames" / f"{stem}.json").write_text(json.dumps(synthetic_frame()), encoding="utf-8")
            (split_root / "metadata.jsonl").write_text(json.dumps({"stem": stem}) + "\n", encoding="utf-8")

            dataset = ARKitUVAlphaDataset(root, "train", uv_size=4, image_width=2, image_height=2, crop_box=(1, 1, 3, 3))
            sample = dataset[0]

            self.assertEqual(len(dataset), 1)
            self.assertEqual(sample["image"].shape, (3, 2, 2))
            self.assertEqual(sample["alpha"].shape, (1, 2, 2))
            self.assertEqual(sample["alpha"].dtype, torch.float32)
            self.assertEqual(sample["hard_lip"].shape, (1, 2, 2))
            self.assertEqual(sample["stem"], stem)


if __name__ == "__main__":
    unittest.main()
