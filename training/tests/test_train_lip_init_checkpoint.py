from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import build_model
from train_lip import load_initial_model_state


class TrainLipInitCheckpointTests(unittest.TestCase):
    def test_load_initial_model_state_loads_weights_without_training_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "checkpoint.pt"
            source = build_model(num_classes=4, width=8)
            target = build_model(num_classes=4, width=8)

            for parameter in source.parameters():
                torch.nn.init.constant_(parameter, 0.25)
            for parameter in target.parameters():
                torch.nn.init.constant_(parameter, -0.5)

            torch.save({"model_state": source.state_dict(), "epoch": 40, "best_iou": 0.9}, checkpoint_path)
            load_initial_model_state(target, checkpoint_path)

            for source_parameter, target_parameter in zip(source.parameters(), target.parameters()):
                self.assertTrue(torch.equal(source_parameter, target_parameter))

    def test_load_initial_model_state_expands_rgb_stem_for_template_prior_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "checkpoint.pt"
            source = build_model(num_classes=4, width=8, input_channels=3)
            target = build_model(num_classes=4, width=8, input_channels=4)

            torch.nn.init.constant_(source.stem.block[0].weight, 0.25)
            torch.save({"model_state": source.state_dict(), "epoch": 40, "best_iou": 0.9}, checkpoint_path)

            load_initial_model_state(target, checkpoint_path)

            loaded_stem = target.stem.block[0].weight
            self.assertTrue(torch.equal(loaded_stem[:, :3], source.stem.block[0].weight))
            self.assertTrue(torch.equal(loaded_stem[:, 3:], torch.zeros_like(loaded_stem[:, 3:])))


if __name__ == "__main__":
    unittest.main()
