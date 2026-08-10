from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import build_model
from train_lip_alpha_student import load_initial_model_state


class TrainLipAlphaInitCheckpointTests(unittest.TestCase):
    def test_load_initial_model_state_loads_model_weights_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "checkpoint.pt"
            source = build_model(num_classes=1, width=8)
            target = build_model(num_classes=1, width=8)

            for parameter in source.parameters():
                torch.nn.init.constant_(parameter, 0.25)
            for parameter in target.parameters():
                torch.nn.init.constant_(parameter, -0.5)

            torch.save(
                {
                    "model_state": source.state_dict(),
                    "optimizer_state": {"old": "state"},
                    "epoch": 80,
                    "best_score": 0.0119,
                },
                checkpoint_path,
            )
            load_initial_model_state(target, checkpoint_path)

            for source_parameter, target_parameter in zip(source.parameters(), target.parameters()):
                self.assertTrue(torch.equal(source_parameter, target_parameter))


if __name__ == "__main__":
    unittest.main()
