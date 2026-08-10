from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lip_curve_model import build_lip_curve_model


class LipCurveModelTests(unittest.TestCase):
    def test_lip_curve_model_outputs_normalized_four_curve_points(self) -> None:
        model = build_lip_curve_model(points_per_curve=18, width=8)
        output = model(torch.zeros((2, 3, 96, 128), dtype=torch.float32))

        self.assertEqual(tuple(output.shape), (2, 4, 18, 2))
        self.assertTrue(torch.all(output >= 0))
        self.assertTrue(torch.all(output <= 1))


if __name__ == "__main__":
    unittest.main()
