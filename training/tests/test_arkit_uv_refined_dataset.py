from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prepare_arkit_uv_refined_dataset import prepare_refined_dataset
from select_arkit_uv_hard_frames import (
    Candidate,
    build_parser,
    expanded_bbox_from_mask,
    score_candidate,
    select_candidates,
    write_hires_preview,
)


def write_sample_split(root: Path, split: str, stem: str) -> None:
    split_root = root / split
    for folder in ("images", "uv_labels", "frames", "uv_images_128_culled"):
        (split_root / folder).mkdir(parents=True, exist_ok=True)

    Image.new("RGB", (8, 8), (120, 90, 70)).save(split_root / "images" / f"{stem}.jpg")
    Image.new("RGB", (8, 8), (110, 80, 60)).save(split_root / "uv_images_128_culled" / f"{stem}.jpg")
    label = np.zeros((8, 8), dtype=np.uint8)
    label[2:4, 2:6] = 1
    label[4:6, 2:6] = 2
    Image.fromarray(label).save(split_root / "uv_labels" / f"{stem}.png")
    (split_root / "frames" / f"{stem}.json").write_text(json.dumps({"sampleID": stem}), encoding="utf-8")
    metadata = {
        "stem": stem,
        "split": split,
        "prev_binary_iou": 0.91,
        "centroid_delta_uv_px": 2.0,
        "yaw_deg": 4.0,
        "jaw_open": 0.02,
        "lip_area_px": 64,
    }
    (split_root / "metadata.jsonl").write_text(json.dumps(metadata) + "\n", encoding="utf-8")


def write_large_label_sample(root: Path, split: str, stem: str) -> None:
    split_root = root / split
    for folder in ("images", "uv_labels", "frames", "uv_images_128_culled"):
        (split_root / folder).mkdir(parents=True, exist_ok=True)

    Image.new("RGB", (48, 64), (120, 90, 70)).save(split_root / "images" / f"{stem}.jpg")
    Image.new("RGB", (64, 64), (110, 80, 60)).save(split_root / "uv_images_128_culled" / f"{stem}.jpg")
    label = np.zeros((256, 256), dtype=np.uint8)
    label[116:128, 102:154] = 1
    label[128:144, 102:154] = 2
    Image.fromarray(label).save(split_root / "uv_labels" / f"{stem}.png")
    (split_root / "frames" / f"{stem}.json").write_text(json.dumps({"sampleID": stem}), encoding="utf-8")


class ARKitUVRefinedDatasetTests(unittest.TestCase):
    def test_prepare_refined_dataset_copies_source_without_overwriting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            output = Path(directory) / "refined"
            write_sample_split(source, "train", "session_frame_000000")
            (source / "summary.json").write_text(json.dumps({"counts": {"train": 1}}), encoding="utf-8")

            summary = prepare_refined_dataset(source, output, splits=("train",))

            self.assertEqual(summary["source_root"], str(source))
            self.assertEqual(summary["output_root"], str(output))
            self.assertEqual(summary["splits"]["train"]["metadata_rows"], 1)
            self.assertTrue((output / "train" / "images" / "session_frame_000000.jpg").exists())
            self.assertTrue((output / "train" / "uv_labels" / "session_frame_000000.png").exists())
            self.assertTrue((output / "summary.json").exists())

            with self.assertRaises(FileExistsError):
                prepare_refined_dataset(source, output, splits=("train",))

    def test_score_candidate_prioritizes_unstable_visible_mouth_frames(self) -> None:
        easy = {
            "stem": "easy",
            "prev_binary_iou": 0.95,
            "centroid_delta_uv_px": 1.0,
            "yaw_deg": 3.0,
            "jaw_open": 0.01,
            "lip_area_px": 3200,
        }
        hard = {
            "stem": "hard",
            "prev_binary_iou": 0.42,
            "centroid_delta_uv_px": 18.0,
            "yaw_deg": -31.0,
            "jaw_open": 0.12,
            "lip_area_px": 900,
        }

        hard_score, hard_reasons = score_candidate(hard, median_lip_area=3200.0)
        easy_score, easy_reasons = score_candidate(easy, median_lip_area=3200.0)

        self.assertGreater(hard_score, easy_score)
        self.assertIn("low_prev_iou", hard_reasons)
        self.assertIn("large_centroid_jump", hard_reasons)
        self.assertIn("yaw", hard_reasons)
        self.assertIn("open_mouth", hard_reasons)
        self.assertIn("lip_area_outlier", hard_reasons)
        self.assertEqual(easy_reasons, [])

    def test_select_candidates_returns_ranked_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            write_sample_split(root, "train", "easy")
            hard_payload = {
                "stem": "hard",
                "split": "train",
                "prev_binary_iou": 0.4,
                "centroid_delta_uv_px": 16.0,
                "yaw_deg": 28.0,
                "jaw_open": 0.11,
                "lip_area_px": 900,
            }
            metadata_path = root / "train" / "metadata.jsonl"
            metadata_path.write_text(metadata_path.read_text(encoding="utf-8") + json.dumps(hard_payload) + "\n", encoding="utf-8")

            candidates = select_candidates(root, "train", count=2)

            self.assertEqual([candidate.payload["stem"] for candidate in candidates], ["hard", "easy"])
            self.assertGreater(candidates[0].score, candidates[1].score)

    def test_expanded_bbox_from_mask_pads_and_clamps(self) -> None:
        mask = np.zeros((10, 20), dtype=bool)
        mask[4:6, 8:12] = True

        bbox = expanded_bbox_from_mask(mask, padding=3)

        self.assertEqual(bbox, (5, 1, 15, 9))

    def test_parser_accepts_hires_flag(self) -> None:
        args = build_parser().parse_args(["--hires", "--no-previews"])

        self.assertTrue(args.hires)

    def test_hires_preview_keeps_uv_label_visible_when_uv_cache_is_smaller(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            stem = "large_label"
            write_large_label_sample(root, "train", stem)
            output_path = Path(directory) / "preview.jpg"
            candidate = Candidate(split="train", payload={"stem": stem}, score=1.0, reasons=[])

            write_hires_preview(root, candidate, output_path)

            preview = np.asarray(Image.open(output_path).convert("RGB"), dtype=np.uint8)
            pink_pixels = (preview[:, :, 0] > 200) & (preview[:, :, 1] < 140) & (preview[:, :, 2] < 180)
            self.assertGreater(int(pink_pixels.sum()), 100)


if __name__ == "__main__":
    unittest.main()
