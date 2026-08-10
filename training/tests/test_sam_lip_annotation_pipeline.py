from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from combine_sam_lip_masks import CLASS_FOLDERS, combine_class_masks, combine_mask_directory
from prepare_sam_lip_annotation import prepare_annotation_session
from test_arkit_uv_student_dataset import synthetic_frame


def write_session(root: Path, frame_count: int = 4) -> Path:
    session = root / "20260530-annotation"
    (session / "images").mkdir(parents=True)
    (session / "frames").mkdir(parents=True)
    (session / "session.json").write_text(json.dumps({"schemaVersion": 1}), encoding="utf-8")
    for index in range(frame_count):
        sample_id = f"frame_{index:06d}"
        image = Image.new("RGB", (4, 4), (index * 20, 80, 120))
        image.save(session / "images" / f"{sample_id}.jpg")
        frame = synthetic_frame()
        frame["sampleID"] = sample_id
        frame["image"]["file"] = f"images/{sample_id}.jpg"
        (session / "frames" / f"{sample_id}.json").write_text(json.dumps(frame), encoding="utf-8")
    return session


class SAMLipAnnotationPipelineTests(unittest.TestCase):
    def test_prepare_annotation_session_exports_portrait_frames_and_mask_folders(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source_root = Path(directory) / "source"
            session = write_session(source_root, frame_count=5)
            output = Path(directory) / "annotation"

            summary = prepare_annotation_session(session, output, stride=2, max_frames=2)

            self.assertEqual(summary["count"], 2)
            self.assertEqual([row["sample_id"] for row in summary["frames"]], ["frame_000000", "frame_000002"])
            self.assertTrue((output / "images" / "frame_000000.jpg").exists())
            self.assertTrue((output / "frames" / "frame_000000.json").exists())
            self.assertTrue((output / "manifest.jsonl").exists())
            for folder in CLASS_FOLDERS:
                self.assertTrue((output / "masks" / folder).is_dir())

    def test_combine_class_masks_uses_lip_classes_and_inner_mouth_precedence(self) -> None:
        masks = {
            "upper_lip": np.zeros((5, 5), dtype=np.uint8),
            "lower_lip": np.zeros((5, 5), dtype=np.uint8),
            "inner_mouth": np.zeros((5, 5), dtype=np.uint8),
        }
        masks["upper_lip"][1:3, 1:4] = 255
        masks["lower_lip"][2:4, 1:4] = 255
        masks["inner_mouth"][2, 2] = 255

        label = combine_class_masks(masks)

        self.assertEqual(label[1, 2], 1)
        self.assertEqual(label[3, 2], 2)
        self.assertEqual(label[2, 2], 3)
        self.assertEqual(label[0, 0], 0)

    def test_combine_mask_directory_writes_labels_and_previews(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "masks"
            image_dir = root / "images"
            output_dir = root / "labels"
            preview_dir = root / "previews"
            image_dir.mkdir()
            for folder in CLASS_FOLDERS:
                (input_dir / folder).mkdir(parents=True)
            Image.new("RGB", (5, 5), (100, 80, 60)).save(image_dir / "frame_000000.jpg")
            Image.fromarray(np.pad(np.ones((2, 3), dtype=np.uint8) * 255, ((1, 2), (1, 1)))).save(
                input_dir / "upper_lip" / "frame_000000.png"
            )

            summary = combine_mask_directory(input_dir, output_dir, image_dir=image_dir, preview_dir=preview_dir)

            self.assertEqual(summary["count_saved"], 1)
            label = np.asarray(Image.open(output_dir / "frame_000000.png").convert("L"), dtype=np.uint8)
            self.assertEqual(int(label.max()), 1)
            self.assertTrue((preview_dir / "preview_001_frame_000000.jpg").exists())


if __name__ == "__main__":
    unittest.main()
