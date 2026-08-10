from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lip_contour_model import build_lip_contour_model


class LipContourModelTests(unittest.TestCase):
    def test_lip_contour_model_outputs_normalized_points(self) -> None:
        model = build_lip_contour_model(point_count=24, width=8)
        output = model(torch.zeros((2, 3, 64, 96), dtype=torch.float32))

        self.assertEqual(tuple(output.shape), (2, 24, 2))
        self.assertTrue(torch.all(output >= 0))
        self.assertTrue(torch.all(output <= 1))


if __name__ == "__main__":
    unittest.main()
