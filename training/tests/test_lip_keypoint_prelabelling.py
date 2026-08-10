from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smooth_lip_cvat_labels import build_cvat_xml, extract_lip_keypoints


class LipKeypointPrelabellingTests(unittest.TestCase):
    def test_extract_lip_keypoints_from_three_class_mask(self) -> None:
        label = np.zeros((40, 80), dtype=np.uint8)
        label[12:23, 20:61] = 1
        label[21:32, 18:63] = 2
        label[20:24, 25:56] = 3
        label[10:16, 31:38] = 1
        label[10:16, 43:50] = 1

        points = extract_lip_keypoints(label)

        self.assertIn("left_corner", points)
        self.assertIn("right_corner", points)
        self.assertIn("upper_left_peak", points)
        self.assertIn("upper_cupid_bow", points)
        self.assertIn("upper_right_peak", points)
        self.assertIn("lower_outer_mid", points)
        self.assertIn("inner_left_corner", points)
        self.assertIn("inner_right_corner", points)
        self.assertAlmostEqual(points["left_corner"][0], 18.0, delta=2.0)
        self.assertAlmostEqual(points["right_corner"][0], 62.0, delta=2.0)
        self.assertLess(points["upper_left_peak"][1], points["upper_cupid_bow"][1])
        self.assertLess(points["upper_right_peak"][1], points["upper_cupid_bow"][1])
        self.assertGreater(points["lower_outer_mid"][1], points["inner_lower_mid"][1])

    def test_build_cvat_xml_writes_point_labels_when_keypoints_are_provided(self) -> None:
        images = [{"stem": "frame_000000", "name": "frame_000000.jpg", "width": 80, "height": 40}]
        polygons = {"frame_000000": {1: [(20, 12), (60, 12), (60, 22), (20, 22)]}}
        keypoints = {"frame_000000": {"left_corner": (18.0, 22.0), "right_corner": (62.0, 22.0)}}

        tree = build_cvat_xml("lip_task", images, polygons, keypoints_by_stem=keypoints)
        root = tree.getroot()

        label_types = {
            label.findtext("name"): label.findtext("type")
            for label in root.findall("./meta/task/labels/label")
        }
        self.assertEqual(label_types["upper_lip"], "polygon")
        self.assertEqual(label_types["left_corner"], "points")
        self.assertEqual(label_types["right_corner"], "points")

        point_nodes = root.findall("./image/points")
        self.assertEqual(len(point_nodes), 2)
        self.assertEqual({node.attrib["label"] for node in point_nodes}, {"left_corner", "right_corner"})


if __name__ == "__main__":
    unittest.main()
