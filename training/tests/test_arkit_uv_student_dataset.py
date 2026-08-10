from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arkit_uv_student_dataset import ARKitUVTeacherDataset, rasterize_portrait_image_to_uv


def synthetic_frame() -> dict:
    return {
        "sampleID": "sample_000",
        "viewport": {"width": 4, "height": 4},
        "image": {"width": 4, "height": 4},
        "displayTransformPortrait": {"a": 1.0, "b": 0.0, "c": 0.0, "d": 1.0, "tx": 0.0, "ty": 0.0},
        "faceTransform": np.eye(4, dtype=np.float64).reshape(-1).tolist(),
        "viewMatrixPortrait": np.eye(4, dtype=np.float64).reshape(-1).tolist(),
        "projectionMatrixPortrait": np.eye(4, dtype=np.float64).reshape(-1).tolist(),
        "faceGeometry": {
            "vertices": [
                {"x": -1.0, "y": 1.0, "z": 0.0},
                {"x": 1.0, "y": 1.0, "z": 0.0},
                {"x": -1.0, "y": -1.0, "z": 0.0},
            ],
            "textureCoordinates": [
                {"x": 0.0, "y": 0.0},
                {"x": 1.0, "y": 0.0},
                {"x": 0.0, "y": 1.0},
            ],
            "triangleIndices": [0, 1, 2],
        },
    }


class ARKitUVStudentDatasetTests(unittest.TestCase):
    def test_rasterize_portrait_image_to_uv_samples_triangle_pixels(self) -> None:
        image = Image.fromarray(
            np.array(
                [
                    [[255, 0, 0], [255, 0, 0], [0, 255, 0], [0, 255, 0]],
                    [[255, 0, 0], [255, 0, 0], [0, 255, 0], [0, 255, 0]],
                    [[0, 0, 255], [0, 0, 255], [255, 255, 0], [255, 255, 0]],
                    [[0, 0, 255], [0, 0, 255], [255, 255, 0], [255, 255, 0]],
                ],
                dtype=np.uint8,
            )
        )

        uv = rasterize_portrait_image_to_uv(synthetic_frame(), image, uv_size=4)

        self.assertEqual(uv.shape, (4, 4, 3))
        self.assertEqual(uv[0, 0].tolist(), [255, 0, 0])
        self.assertEqual(uv[0, 2].tolist(), [0, 255, 0])
        self.assertEqual(uv[2, 0].tolist(), [0, 0, 255])

    def test_arkit_uv_teacher_dataset_returns_uv_tensors(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            split_root = root / "train"
            for folder in ("images", "uv_labels", "frames"):
                (split_root / folder).mkdir(parents=True)

            stem = "session_frame_000000"
            Image.new("RGB", (4, 4), (128, 64, 32)).save(split_root / "images" / f"{stem}.jpg")
            Image.fromarray(
                np.array([[0, 1, 1, 0], [0, 1, 2, 0], [0, 2, 3, 0], [0, 0, 0, 0]], dtype=np.uint8)
            ).save(split_root / "uv_labels" / f"{stem}.png")
            (split_root / "frames" / f"{stem}.json").write_text(json.dumps(synthetic_frame()), encoding="utf-8")
            (split_root / "metadata.jsonl").write_text(json.dumps({"stem": stem}) + "\n", encoding="utf-8")

            dataset = ARKitUVTeacherDataset(root, "train", uv_size=4, image_width=2, image_height=2, crop_box=(1, 1, 3, 3))
            sample = dataset[0]

            self.assertEqual(len(dataset), 1)
            self.assertEqual(sample["image"].shape, (3, 2, 2))
            self.assertEqual(sample["label"].shape, (2, 2))
            self.assertEqual(sample["label"].dtype, torch.int64)
            self.assertEqual(sample["stem"], stem)


if __name__ == "__main__":
    unittest.main()
