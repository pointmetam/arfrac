from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from export_coreml import build_export_model, output_name_for_task
from model import build_model


class ExportCoreMLTests(unittest.TestCase):
    def test_lip_alpha_export_wraps_model_with_sigmoid_alpha_output(self) -> None:
        base = build_model(num_classes=1, width=8)
        wrapped = build_export_model(base, task="lip-alpha")
        sample = torch.randn(1, 3, 16, 24)

        output = wrapped(sample)

        self.assertEqual(tuple(output.shape), (1, 1, 16, 24))
        self.assertGreaterEqual(float(output.min()), 0.0)
        self.assertLessEqual(float(output.max()), 1.0)
        self.assertEqual(output_name_for_task("lip-alpha"), "alpha")

    def test_segmentation_export_keeps_logits_output(self) -> None:
        base = build_model(num_classes=4, width=8)
        wrapped = build_export_model(base, task="segmentation")
        sample = torch.randn(1, 3, 16, 24)

        output = wrapped(sample)

        self.assertEqual(tuple(output.shape), (1, 4, 16, 24))
        self.assertEqual(output_name_for_task("segmentation"), "logits")

    def test_lip_sdf_export_applies_channel_activations(self) -> None:
        base = build_model(num_classes=3, width=8)
        wrapped = build_export_model(base, task="lip-sdf")
        sample = torch.randn(1, 3, 16, 24)

        output = wrapped(sample)

        self.assertEqual(tuple(output.shape), (1, 3, 16, 24))
        self.assertGreaterEqual(float(output[:, 0].min()), 0.0)
        self.assertLessEqual(float(output[:, 0].max()), 1.0)
        self.assertGreaterEqual(float(output[:, 1].min()), -1.0)
        self.assertLessEqual(float(output[:, 1].max()), 1.0)
        self.assertGreaterEqual(float(output[:, 2].min()), 0.0)
        self.assertLessEqual(float(output[:, 2].max()), 1.0)
        self.assertEqual(output_name_for_task("lip-sdf"), "lip_sdf")

    def test_build_model_accepts_template_prior_input_channels(self) -> None:
        model = build_model(num_classes=4, width=8, input_channels=4)
        sample = torch.randn(1, 4, 16, 24)

        output = model(sample)

        self.assertEqual(tuple(output.shape), (1, 4, 16, 24))


if __name__ == "__main__":
    unittest.main()
