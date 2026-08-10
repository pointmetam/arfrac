from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from PIL import Image

from teacher_2d_to_arkit_uv import portrait_image


CLASS_FOLDERS = ("upper_lip", "lower_lip", "inner_mouth")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def selected_frame_paths(session_dir: Path, start: int = 0, stride: int = 1, max_frames: int | None = None) -> list[Path]:
    frame_paths = sorted((session_dir / "frames").glob("*.json"))
    frame_paths = frame_paths[max(0, start) :: max(1, stride)]
    if max_frames is not None:
        frame_paths = frame_paths[:max_frames]
    return frame_paths


def prepare_annotation_session(
    session_dir: str | Path,
    output_dir: str | Path,
    start: int = 0,
    stride: int = 1,
    max_frames: int | None = None,
    portrait_scale: float = 1.0,
    overwrite: bool = False,
) -> dict[str, Any]:
    session_dir = Path(session_dir).expanduser()
    output_dir = Path(output_dir)
    if not (session_dir / "frames").exists() or not (session_dir / "images").exists():
        raise FileNotFoundError(f"Missing ARKit recording folders under {session_dir}")
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Annotation output already exists: {output_dir}")

    images_dir = output_dir / "images"
    frames_dir = output_dir / "frames"
    masks_dir = output_dir / "masks"
    images_dir.mkdir(parents=True)
    frames_dir.mkdir(parents=True)
    for folder in CLASS_FOLDERS:
        (masks_dir / folder).mkdir(parents=True)

    rows: list[dict[str, Any]] = []
    for frame_path in selected_frame_paths(session_dir, start=start, stride=stride, max_frames=max_frames):
        frame = json.loads(frame_path.read_text(encoding="utf-8"))
        sample_id = str(frame["sampleID"])
        raw_image_path = session_dir / frame["image"]["file"]
        raw_image = Image.open(raw_image_path).convert("RGB")
        portrait = portrait_image(raw_image, frame, scale=portrait_scale)

        output_image = images_dir / f"{sample_id}.jpg"
        output_frame = frames_dir / f"{sample_id}.json"
        portrait.save(output_image, quality=95)
        shutil.copy2(frame_path, output_frame)
        rows.append(
            {
                "sample_id": sample_id,
                "image": str(output_image),
                "frame": str(output_frame),
                "source_image": str(raw_image_path),
                "source_frame": str(frame_path),
                "mask_space": "portrait",
                "portrait_scale": portrait_scale,
            }
        )

    write_jsonl(output_dir / "manifest.jsonl", rows)
    classes = {
        "0": "background",
        "1": "upper_lip",
        "2": "lower_lip",
        "3": "inner_mouth",
        "mask_folders": list(CLASS_FOLDERS),
    }
    (output_dir / "classes.json").write_text(json.dumps(classes, indent=2, sort_keys=True), encoding="utf-8")
    readme = (
        "# SAM2 Lip Annotation Workspace\n\n"
        "Annotate the images in `images/` in portrait orientation.\n\n"
        "Export binary masks with these filenames:\n"
        "- `masks/upper_lip/<sample_id>.png`\n"
        "- `masks/lower_lip/<sample_id>.png`\n"
        "- `masks/inner_mouth/<sample_id>.png`\n\n"
        "Then run `training/combine_sam_lip_masks.py` to produce 4-class masks for UV projection.\n"
    )
    (output_dir / "README.md").write_text(readme, encoding="utf-8")

    summary: dict[str, Any] = {
        "session": str(session_dir),
        "output_dir": str(output_dir),
        "mask_space": "portrait",
        "portrait_scale": portrait_scale,
        "start": start,
        "stride": stride,
        "count": len(rows),
        "frames": rows,
        "classes": classes,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare portrait frames for SAM2/CVAT lip mask annotation.")
    parser.add_argument("--session", required=True, help="ARKit recording session directory")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--portrait-scale", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = prepare_annotation_session(
        args.session,
        args.output_dir,
        start=args.start,
        stride=args.stride,
        max_frames=args.max_frames,
        portrait_scale=args.portrait_scale,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
