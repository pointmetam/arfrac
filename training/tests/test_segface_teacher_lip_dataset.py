from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prepare_segface_teacher_lip_dataset import prepare_segface_teacher_split


class SegFaceTeacherLipDatasetTests(unittest.TestCase):
    def test_prepare_split_uses_lapa_bbox_but_writes_teacher_lip_label(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "LaPa"
            image_dir = root / "train" / "images"
            label_dir = root / "train" / "labels"
            landmark_dir = root / "train" / "landmarks"
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            landmark_dir.mkdir(parents=True)

            Image.new("RGB", (40, 30), (120, 100, 90)).save(image_dir / "sample.jpg")
            lapa_label = np.zeros((30, 40), dtype=np.uint8)
            lapa_label[10:14, 13:27] = 7
            lapa_label[15:22, 10:30] = 9
            Image.fromarray(lapa_label).save(label_dir / "sample.png")
            landmark_dir.joinpath("sample.txt").write_text("106 " + " ".join(["0"] * 212), encoding="utf-8")

            teacher_label = np.zeros((30, 40), dtype=np.uint8)
            teacher_label[9:13, 12:28] = 1
            teacher_label[14:22, 9:31] = 2
            teacher_label[13:16, 16:24] = 3

            output = Path(directory) / "LaPaLipTeacherSegFace"
            summary = prepare_segface_teacher_split(
                source_root=root,
                output_root=output,
                split="train",
                width=20,
                height=12,
                padding_x=0.2,
                padding_y=0.2,
                limit=None,
                predict_lip_label=lambda _image, _landmarks: teacher_label,
                teacher_name="segface/test",
                teacher_checkpoint="/tmp/model.pt",
                teacher_resolution=512,
            )

            self.assertEqual(summary["kept"], 1)
            label = np.asarray(Image.open(output / "train" / "labels" / "sample.png").convert("L"), dtype=np.uint8)
            self.assertEqual(set(np.unique(label).tolist()), {0, 1, 2, 3})

            metadata = json.loads((output / "train" / "metadata.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(metadata["teacher"], "segface/test")
            self.assertEqual(metadata["teacher_checkpoint"], "/tmp/model.pt")
            self.assertEqual(metadata["source_stem"], "sample")
            self.assertEqual(metadata["teacher_mapping"], {"7": 1, "9": 2, "8": 3})


if __name__ == "__main__":
    unittest.main()
