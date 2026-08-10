from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sam2_lip_refine_preview import (
    assign_refined_lip_classes,
    best_sam2_mask,
    dilate_mask,
    prompt_points_from_label,
)


class Sam2LipRefinePreviewTests(unittest.TestCase):
    def test_assign_refined_lip_classes_expands_to_nearest_original_lip_class(self) -> None:
        label = np.zeros((8, 12), dtype=np.uint8)
        label[2:4, 4:8] = 1
        label[4:6, 4:8] = 2
        label[3:5, 5:7] = 3
        refined_lip_surface = np.zeros_like(label, dtype=bool)
        refined_lip_surface[1:7, 3:9] = True

        refined = assign_refined_lip_classes(label, refined_lip_surface)

        self.assertEqual(int(refined[1, 5]), 1)
        self.assertEqual(int(refined[6, 5]), 2)
        self.assertEqual(int(refined[3, 5]), 3)
        self.assertEqual(int(refined[0, 0]), 0)

    def test_prompt_points_include_lip_positives_and_inner_mouth_negative(self) -> None:
        label = np.zeros((10, 14), dtype=np.uint8)
        label[2:4, 4:10] = 1
        label[5:8, 3:11] = 2
        label[4:6, 6:8] = 3

        points, point_labels = prompt_points_from_label(label)

        self.assertGreaterEqual(int((point_labels == 1).sum()), 2)
        self.assertGreaterEqual(int((point_labels == 0).sum()), 1)
        self.assertTrue(np.all(points[:, 0] >= 0))
        self.assertTrue(np.all(points[:, 1] >= 0))

    def test_dilate_mask_respects_radius(self) -> None:
        mask = np.zeros((7, 7), dtype=bool)
        mask[3, 3] = True

        dilated = dilate_mask(mask, radius=1)

        self.assertTrue(dilated[3, 3])
        self.assertTrue(dilated[2, 3])
        self.assertTrue(dilated[3, 2])
        self.assertFalse(dilated[1, 3])
        self.assertFalse(dilated[2, 2])

    def test_best_sam2_mask_rejects_large_spill_and_prefers_overlap(self) -> None:
        initial = np.zeros((10, 14), dtype=bool)
        initial[3:7, 5:9] = True
        huge_spill = np.ones_like(initial)
        slight_refine = initial.copy()
        slight_refine[2, 6:8] = True
        slight_refine[6, 8] = False
        masks = np.stack([huge_spill, slight_refine])
        scores = np.asarray([0.99, 0.40], dtype=np.float32)

        selected = best_sam2_mask(
            masks,
            scores,
            initial,
            min_area_ratio=0.70,
            max_area_ratio=1.30,
            max_expand_px=1,
        )

        self.assertFalse(np.array_equal(selected, huge_spill))
        self.assertTrue(selected[2, 6])
        self.assertFalse(selected[0, 0])


if __name__ == "__main__":
    unittest.main()
