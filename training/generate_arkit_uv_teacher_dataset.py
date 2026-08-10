from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm

from arkit_uv_stability_preview import face_yaw_deg, mask_bbox
from teacher_2d_to_arkit_uv import portrait_image


def parse_csv(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def split_for_session(session_name: str, val_sessions: set[str], test_sessions: set[str]) -> str:
    if session_name in test_sessions:
        return "test"
    if session_name in val_sessions:
        return "val"
    return "train"


def default_eval_sessions(sessions: list[Path]) -> tuple[set[str], set[str]]:
    names = [session.name for session in sessions]
    if len(names) < 3:
        return set(), set(names[-1:])
    return {names[-2]}, {names[-1]}


def read_metrics(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8") as file:
        return {row["sample_id"]: row for row in csv.DictReader(file)}


def float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def lip_area(label: np.ndarray) -> int:
    return int((label > 0).sum())


def write_jsonl_line(file: Any, payload: dict[str, Any]) -> None:
    file.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")


def prepare_split_dirs(output_root: Path) -> dict[str, dict[str, Path]]:
    result: dict[str, dict[str, Path]] = {}
    for split in ("train", "val", "test"):
        split_root = output_root / split
        paths = {
            "images": split_root / "images",
            "uv_labels": split_root / "uv_labels",
            "frames": split_root / "frames",
        }
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)
        result[split] = paths
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions-root", default="/private/tmp/arface-uv-recordings-new")
    parser.add_argument("--output-root", default="/Users/lee/Datasets/ARKitLipUVTeacher")
    parser.add_argument("--label-root-template", default="training/outputs/arkit_uv_stability_segface512_culled_{session}/uv_labels")
    parser.add_argument("--metrics-template", default="training/outputs/arkit_uv_stability_segface512_culled_{session}/metrics.csv")
    parser.add_argument("--session-prefix", default="20260528-")
    parser.add_argument("--val-sessions", default="")
    parser.add_argument("--test-sessions", default="")
    parser.add_argument("--min-lip-area", type=int, default=256)
    parser.add_argument("--max-centroid-delta", type=float, default=80.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    sessions_root = Path(args.sessions_root).expanduser()
    output_root = Path(args.output_root).expanduser()
    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    sessions = sorted(
        session
        for session in sessions_root.iterdir()
        if session.is_dir() and session.name.startswith(args.session_prefix) and (session / "session.json").exists()
    )
    if not sessions:
        raise SystemExit(f"No sessions found under {sessions_root} with prefix {args.session_prefix!r}")

    val_sessions = parse_csv(args.val_sessions)
    test_sessions = parse_csv(args.test_sessions)
    if not val_sessions and not test_sessions:
        val_sessions, test_sessions = default_eval_sessions(sessions)

    split_dirs = prepare_split_dirs(output_root)
    metadata_files = {
        split: (output_root / split / "metadata.jsonl").open("w", encoding="utf-8")
        for split in ("train", "val", "test")
    }

    counts = {
        split: {"kept": 0, "skipped_missing": 0, "skipped_area": 0, "skipped_jump": 0}
        for split in ("train", "val", "test")
    }

    try:
        for session in sessions:
            session_name = session.name
            split = split_for_session(session_name, val_sessions, test_sessions)
            label_dir = Path(args.label_root_template.format(session=session_name)).expanduser()
            metrics = read_metrics(Path(args.metrics_template.format(session=session_name)).expanduser())
            if not label_dir.exists():
                raise FileNotFoundError(f"Missing UV label dir for {session_name}: {label_dir}")

            frame_paths = sorted((session / "frames").glob("*.json"))
            for frame_path in tqdm(frame_paths, desc=f"{split}:{session_name}"):
                frame = json.loads(frame_path.read_text(encoding="utf-8"))
                sample_id = str(frame["sampleID"])
                label_path = label_dir / f"{sample_id}.png"
                raw_image_path = session / frame["image"]["file"]
                if not label_path.exists() or not raw_image_path.exists():
                    counts[split]["skipped_missing"] += 1
                    continue

                uv_label = np.asarray(Image.open(label_path).convert("L"), dtype=np.uint8)
                area = lip_area(uv_label)
                if area < args.min_lip_area:
                    counts[split]["skipped_area"] += 1
                    continue

                metric = metrics.get(sample_id, {})
                centroid_delta = float_or_none(metric.get("centroid_delta_uv_px"))
                if centroid_delta is not None and centroid_delta > args.max_centroid_delta:
                    counts[split]["skipped_jump"] += 1
                    continue

                output_stem = f"{session_name}_{sample_id}"
                raw_image = Image.open(raw_image_path).convert("RGB")
                portrait = portrait_image(raw_image, frame, scale=1.0)

                image_out = split_dirs[split]["images"] / f"{output_stem}.jpg"
                label_out = split_dirs[split]["uv_labels"] / f"{output_stem}.png"
                frame_out = split_dirs[split]["frames"] / f"{output_stem}.json"
                portrait.save(image_out, quality=92)
                Image.fromarray(uv_label).save(label_out)
                shutil.copy2(frame_path, frame_out)

                write_jsonl_line(
                    metadata_files[split],
                    {
                        "stem": output_stem,
                        "split": split,
                        "session": session_name,
                        "sample_id": sample_id,
                        "source_frame": str(frame_path),
                        "source_image": str(raw_image_path),
                        "source_uv_label": str(label_path),
                        "image": str(image_out.relative_to(output_root)),
                        "uv_label": str(label_out.relative_to(output_root)),
                        "frame": str(frame_out.relative_to(output_root)),
                        "yaw_deg": face_yaw_deg(frame),
                        "jaw_open": float(frame.get("blendShapes", {}).get("jawOpen", 0.0) or 0.0),
                        "lip_area_px": area,
                        "bbox_uv_px": mask_bbox(uv_label),
                        "prev_binary_iou": float_or_none(metric.get("prev_binary_iou")),
                        "centroid_delta_uv_px": centroid_delta,
                    },
                )
                counts[split]["kept"] += 1
    finally:
        for file in metadata_files.values():
            file.close()

    summary = {
        "output_root": str(output_root),
        "sessions_root": str(sessions_root),
        "sessions": [session.name for session in sessions],
        "val_sessions": sorted(val_sessions),
        "test_sessions": sorted(test_sessions),
        "min_lip_area": args.min_lip_area,
        "max_centroid_delta": args.max_centroid_delta,
        "counts": counts,
        "labels": {
            "0": "background_or_non_lip_uv",
            "1": "upper_lip",
            "2": "lower_lip",
            "3": "inner_mouth",
        },
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
