from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preview_uv_sdf_edges import compose_sdf_edge_preview, lip_sdf_alpha


class UVSDFEdgePreviewTests(unittest.TestCase):
    def test_lip_sdf_alpha_feathers_and_shrinks_hard_label(self) -> None:
        label = np.zeros((17, 17), dtype=np.uint8)
        label[4:13, 4:13] = 1
        label[8, 8] = 3

        soft = lip_sdf_alpha(label, feather_radius=2.0, shrink_pixels=0.0)
        shrink = lip_sdf_alpha(label, feather_radius=2.0, shrink_pixels=2.0)

        self.assertEqual(soft.shape, label.shape)
        self.assertEqual(soft.dtype, np.float32)
        self.assertEqual(float(soft[8, 8]), 0.0)
        self.assertGreater(float(soft[5, 5]), 0.9)
        self.assertGreater(float(soft[4, 4]), 0.1)
        self.assertLess(float(shrink.sum()), float(soft.sum()))
        self.assertLess(float(shrink[4, 4]), float(soft[4, 4]))

    def test_compose_sdf_edge_preview_writes_three_panel_image(self) -> None:
        label = np.zeros((64, 64), dtype=np.uint8)
        label[24:40, 18:46] = 1
        uv_image = Image.new("RGB", (64, 64), (128, 96, 78))

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "preview.jpg"

            preview = compose_sdf_edge_preview(
                uv_image,
                label,
                output_path=output_path,
                feather_radius=3.0,
                shrink_pixels=2.0,
                crop_padding=8,
            )

            self.assertTrue(output_path.exists())
            self.assertGreater(preview.width, preview.height)
            pixels = np.asarray(preview.convert("RGB"), dtype=np.uint8)
            red_pixels = (pixels[:, :, 0] > 150) & (pixels[:, :, 1] < 130)
            self.assertGreater(int(red_pixels.sum()), 100)


if __name__ == "__main__":
    unittest.main()
