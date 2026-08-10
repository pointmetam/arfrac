from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate_lip import build_lip_eval_dataset


class EvaluateLipTemplatePriorTests(unittest.TestCase):
    def test_build_lip_eval_dataset_uses_template_dataset_for_template_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "lip"
            split_root = root / "val"
            (split_root / "images").mkdir(parents=True)
            (split_root / "labels").mkdir(parents=True)

            Image.new("RGB", (8, 8), (160, 120, 104)).save(split_root / "images" / "sample.jpg")
            Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(split_root / "labels" / "sample.png")

            dataset = build_lip_eval_dataset(
                root=root,
                split="val",
                image_width=16,
                image_height=12,
                limit=None,
                checkpoint={"template_prior": True, "input_channels": 4},
            )

            sample = dataset[0]
            self.assertEqual(tuple(sample["image"].shape), (4, 12, 16))


if __name__ == "__main__":
    unittest.main()
