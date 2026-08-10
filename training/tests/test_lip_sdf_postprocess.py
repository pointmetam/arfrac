from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lip_sdf_postprocess import alpha_to_smooth_mask, largest_connected_component, mask_iou


class LipSDFPostprocessTests(unittest.TestCase):
    def test_largest_connected_component_removes_small_island(self) -> None:
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[8:20, 8:20] = 1
        mask[2:4, 2:4] = 1

        component = largest_connected_component(mask)

        self.assertEqual(int(component[10, 10]), 1)
        self.assertEqual(int(component[2, 2]), 0)

    def test_alpha_to_smooth_mask_keeps_main_lip_area(self) -> None:
        alpha = np.zeros((64, 96), dtype=np.float32)
        yy, xx = np.mgrid[:64, :96]
        ellipse = (((xx - 48) / 28.0) ** 2 + ((yy - 32) / 10.0) ** 2) <= 1.0
        alpha[ellipse] = 1.0
        alpha[10:12, 10:12] = 1.0

        smoothed = alpha_to_smooth_mask(alpha, threshold=0.5, point_count=64)

        self.assertEqual(smoothed.shape, alpha.shape)
        self.assertEqual(int(smoothed[11, 11]), 0)
        self.assertGreater(mask_iou(smoothed, ellipse.astype(np.uint8)), 0.84)

    def test_alpha_to_smooth_mask_preserves_inner_mouth_hole(self) -> None:
        alpha = np.zeros((80, 120), dtype=np.float32)
        yy, xx = np.mgrid[:80, :120]
        outer = (((xx - 60) / 38.0) ** 2 + ((yy - 40) / 18.0) ** 2) <= 1.0
        inner = (((xx - 60) / 26.0) ** 2 + ((yy - 39) / 7.0) ** 2) <= 1.0
        alpha[outer] = 1.0
        alpha[inner] = 0.0

        smoothed = alpha_to_smooth_mask(alpha, threshold=0.5, point_count=96)

        self.assertEqual(int(smoothed[40, 60]), 0)
        self.assertEqual(int(smoothed[40, 25]), 1)
        self.assertGreater(mask_iou(smoothed, (outer & ~inner).astype(np.uint8)), 0.80)


if __name__ == "__main__":
    unittest.main()
