from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from combine_lip_datasets import combine_lip_datasets
from prepare_ibugmask_lip_dataset import prepare_ibugmask_lip_dataset
from prepare_public_lip_dataset import prepare_public_lip_dataset


def write_binary_mask(path: Path, size: tuple[int, int], box: tuple[int, int, int, int]) -> None:
    mask = np.zeros((size[1], size[0]), dtype=np.uint8)
    left, top, right, bottom = box
    mask[top:bottom, left:right] = 255
    Image.fromarray(mask).save(path)


def write_ibug_label(path: Path, size: tuple[int, int]) -> None:
    label = np.zeros((size[1], size[0]), dtype=np.uint8)
    label[10:14, 12:28] = 7
    label[14:17, 16:24] = 8
    label[17:23, 10:30] = 9
    Image.fromarray(label).save(path)


class PublicLipDatasetConversionTests(unittest.TestCase):
    def test_prepare_public_lip_dataset_maps_celebamaskhq_class_masks_to_lip_roi_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "CelebAMask-HQ"
            image_dir = root / "CelebA-HQ-img"
            mask_dir = root / "CelebAMask-HQ-mask-anno" / "0"
            image_dir.mkdir(parents=True)
            mask_dir.mkdir(parents=True)

            Image.new("RGB", (40, 30), (120, 100, 90)).save(image_dir / "0.jpg")
            write_binary_mask(mask_dir / "00000_u_lip.png", (40, 30), (12, 10, 28, 15))
            write_binary_mask(mask_dir / "00000_l_lip.png", (40, 30), (10, 15, 30, 22))
            write_binary_mask(mask_dir / "00000_mouth.png", (40, 30), (16, 14, 24, 18))

            output = Path(directory) / "CelebAMaskHQLip"
            summary = prepare_public_lip_dataset(
                root=root,
                output_root=output,
                image_dir_name="CelebA-HQ-img",
                mask_dir_name="CelebAMask-HQ-mask-anno",
                width=16,
                height=10,
                split_ratios=(1.0, 0.0, 0.0),
            )

            self.assertEqual(summary["train"]["kept"], 1)
            label = np.asarray(Image.open(output / "train" / "labels" / "00000.png").convert("L"), dtype=np.uint8)
            self.assertEqual(set(np.unique(label).tolist()), {0, 1, 2, 3})
            metadata = json.loads((output / "train" / "metadata.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(metadata["source_dataset"], "public_class_masks")
            self.assertEqual(metadata["source_stem"], "00000")

    def test_combine_lip_datasets_prefixes_samples_and_preserves_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_a = root / "LaPaLip"
            source_b = root / "CelebLip"
            for source, color, label_value in [(source_a, (10, 20, 30), 1), (source_b, (40, 50, 60), 2)]:
                (source / "train" / "images").mkdir(parents=True)
                (source / "train" / "labels").mkdir(parents=True)
                Image.new("RGB", (4, 3), color).save(source / "train" / "images" / "sample.jpg")
                Image.fromarray(np.full((3, 4), label_value, dtype=np.uint8)).save(
                    source / "train" / "labels" / "sample.png"
                )

            output = root / "MixedLip"
            summary = combine_lip_datasets(
                sources={"lapa": source_a, "celeb": source_b},
                output_root=output,
                splits=["train"],
            )

            self.assertEqual(summary["train"]["kept"], 2)
            self.assertTrue((output / "train" / "images" / "lapa_sample.jpg").exists())
            self.assertTrue((output / "train" / "labels" / "celeb_sample.png").exists())
            label = np.asarray(Image.open(output / "train" / "labels" / "celeb_sample.png").convert("L"))
            self.assertEqual(int(label.max()), 2)

    def test_prepare_ibugmask_lip_dataset_maps_single_label_pngs_to_lip_roi_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "ibugmask_release"
            for split, stems in {"train": ["sample_a", "sample_b"], "test": ["sample_c"]}.items():
                split_dir = root / split
                split_dir.mkdir(parents=True)
                for stem in stems:
                    Image.new("RGB", (40, 30), (120, 100, 90)).save(split_dir / f"{stem}.jpg")
                    write_ibug_label(split_dir / f"{stem}.png", (40, 30))

            output = Path(directory) / "iBugMaskLip"
            summary = prepare_ibugmask_lip_dataset(
                root=root,
                output_root=output,
                width=16,
                height=10,
                val_ratio=0.5,
            )

            self.assertEqual(summary["train"]["kept"], 1)
            self.assertEqual(summary["val"]["kept"], 1)
            self.assertEqual(summary["test"]["kept"], 1)
            label = np.asarray(Image.open(output / "test" / "labels" / "sample_c.png").convert("L"), dtype=np.uint8)
            self.assertEqual(set(np.unique(label).tolist()), {0, 1, 2, 3})
            metadata = json.loads((output / "test" / "metadata.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(metadata["source_dataset"], "ibugmask")
            self.assertEqual(metadata["source_stem"], "sample_c")


if __name__ == "__main__":
    unittest.main()
