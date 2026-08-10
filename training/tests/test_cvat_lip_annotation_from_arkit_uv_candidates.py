from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prepare_cvat_lip_annotation_from_arkit_uv_candidates import (
    portrait_scale_for_target_long_edge,
    prepare_workspace,
)


def write_manifest_with_missing_source(root: Path) -> Path:
    dataset = root / "dataset"
    split = dataset / "train"
    for folder in ("images", "frames", "uv_labels"):
        (split / folder).mkdir(parents=True, exist_ok=True)

    stem = "session_frame_000000"
    Image.new("RGB", (390, 844), (120, 90, 80)).save(split / "images" / f"{stem}.jpg")
    Image.new("L", (512, 512), 0).save(split / "uv_labels" / f"{stem}.png")
    frame = {
        "sampleID": "frame_000000",
        "viewport": {"width": 390, "height": 844},
        "image": {"width": 1440, "height": 1080, "file": "images/frame_000000.jpg"},
    }
    (split / "frames" / f"{stem}.json").write_text(json.dumps(frame), encoding="utf-8")

    row = {
        "stem": stem,
        "split": "train",
        "image": str(split / "images" / f"{stem}.jpg"),
        "frame": str(split / "frames" / f"{stem}.json"),
        "uv_label": str(split / "uv_labels" / f"{stem}.png"),
        "metadata": {
            "source_image": str(root / "missing_recording" / "images" / "frame_000000.jpg"),
            "source_frame": str(split / "frames" / f"{stem}.json"),
        },
    }
    manifest = root / "candidates.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return manifest


class CVATLipAnnotationFromARKitUVCandidatesTests(unittest.TestCase):
    def test_portrait_target_long_edge_uses_raw_resolution_without_upscaling(self) -> None:
        frame = {"viewport": {"width": 390, "height": 844}}

        scale = portrait_scale_for_target_long_edge((1440, 1080), frame, target_long_edge=1440)

        self.assertAlmostEqual(scale, 1080 / 844, places=4)

    def test_require_source_image_fails_instead_of_silently_using_low_res_dataset_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = write_manifest_with_missing_source(root)

            with self.assertRaises(FileNotFoundError):
                prepare_workspace(
                    manifest=manifest,
                    output_dir=root / "out",
                    task_name="missing_source",
                    limit=1,
                    points_per_edge=8,
                    smooth_window=5,
                    panel_width=400,
                    portrait_scale=1.0,
                    portrait_target_long_edge=1080,
                    mouth_crop=True,
                    crop_padding=20,
                    min_crop_width=120,
                    min_crop_height=80,
                    add_keypoints=False,
                    require_source_image=True,
                    overwrite=False,
                )


if __name__ == "__main__":
    unittest.main()
