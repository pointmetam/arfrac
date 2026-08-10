from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from teacher_2d_to_arkit_uv import label_to_rgba, render_uv_label_to_portrait


DEFAULT_ROOT = Path("/Users/lee/Datasets/ARKitLipUVTeacherRefined")
DEFAULT_OUTPUT_DIR = Path("training/outputs/arkit_uv_label_review")


@dataclass(frozen=True)
class Candidate:
    split: str
    payload: dict[str, Any]
    score: float
    reasons: list[str]

    def to_json(self, root: Path) -> dict[str, Any]:
        stem = str(self.payload["stem"])
        split_root = root / self.split
        return {
            "split": self.split,
            "stem": stem,
            "score": round(self.score, 4),
            "reasons": self.reasons,
            "image": str(split_root / "images" / f"{stem}.jpg"),
            "uv_label": str(split_root / "uv_labels" / f"{stem}.png"),
            "frame": str(split_root / "frames" / f"{stem}.json"),
            "metadata": self.payload,
        }


def numeric(payload: dict[str, Any], key: str, default: float | None = 0.0) -> float | None:
    value = payload.get(key, default)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def score_candidate(payload: dict[str, Any], median_lip_area: float | None) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []

    prev_iou = numeric(payload, "prev_binary_iou", None)
    if prev_iou is not None and prev_iou < 0.85:
        score += (0.85 - prev_iou) * 70.0
        if prev_iou < 0.75:
            reasons.append("low_prev_iou")

    centroid_delta = numeric(payload, "centroid_delta_uv_px", 0.0) or 0.0
    if centroid_delta > 8.0:
        score += (centroid_delta - 8.0) * 1.6
        reasons.append("large_centroid_jump")

    yaw = min(abs(numeric(payload, "yaw_deg", 0.0) or 0.0), 45.0)
    if yaw > 18.0:
        score += (yaw - 18.0) * 0.9
        reasons.append("yaw")

    jaw_open = min(numeric(payload, "jaw_open", 0.0) or 0.0, 0.25)
    if jaw_open > 0.055:
        score += (jaw_open - 0.055) * 260.0
        reasons.append("open_mouth")

    lip_area = numeric(payload, "lip_area_px", None)
    if median_lip_area and lip_area and median_lip_area > 0:
        ratio = lip_area / median_lip_area
        if ratio < 0.55:
            score += (0.55 - ratio) * 34.0
            reasons.append("lip_area_outlier")
        elif ratio > 1.65:
            score += (ratio - 1.65) * 16.0
            reasons.append("lip_area_outlier")

    return score, reasons


def load_metadata(root: Path, split: str) -> list[dict[str, Any]]:
    metadata_path = root / split / "metadata.jsonl"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata: {metadata_path}")
    rows: list[dict[str, Any]] = []
    for line in metadata_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def median_lip_area(rows: Iterable[dict[str, Any]]) -> float | None:
    areas = [numeric(row, "lip_area_px", None) for row in rows]
    valid = np.array([area for area in areas if area is not None and area > 0], dtype=np.float64)
    if valid.size == 0:
        return None
    return float(np.median(valid))


def select_candidates(root: str | Path, split: str, count: int = 48, max_per_session: int | None = None) -> list[Candidate]:
    root = Path(root).expanduser()
    rows = load_metadata(root, split)
    median_area = median_lip_area(rows)
    candidates = [
        Candidate(split=split, payload=row, score=score, reasons=reasons)
        for row in rows
        for score, reasons in [score_candidate(row, median_area)]
    ]
    candidates.sort(key=lambda candidate: (-candidate.score, str(candidate.payload.get("stem", ""))))
    if max_per_session is None or max_per_session <= 0:
        return candidates[:count]

    selected: list[Candidate] = []
    session_counts: dict[str, int] = {}
    for candidate in candidates:
        session = str(candidate.payload.get("session", "unknown"))
        if session_counts.get(session, 0) >= max_per_session:
            continue
        selected.append(candidate)
        session_counts[session] = session_counts.get(session, 0) + 1
        if len(selected) >= count:
            break
    return selected


def titled(image: Image.Image, title: str) -> Image.Image:
    image = image.convert("RGB")
    title_height = 24
    output = Image.new("RGB", (image.width, image.height + title_height), "white")
    output.paste(image, (0, title_height))
    draw = ImageDraw.Draw(output)
    try:
        font = ImageFont.truetype("Arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    draw.text((6, 5), title, fill=(20, 20, 20), font=font)
    return output


def hstack(images: list[Image.Image]) -> Image.Image:
    width = sum(image.width for image in images)
    height = max(image.height for image in images)
    output = Image.new("RGB", (width, height), "white")
    x = 0
    for image in images:
        output.paste(image, (x, 0))
        x += image.width
    return output


def expanded_bbox_from_mask(
    mask: np.ndarray,
    padding: int = 0,
    min_size: tuple[int, int] | None = None,
) -> tuple[int, int, int, int]:
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape[:2]
    ys, xs = np.nonzero(mask)
    if xs.size == 0 or ys.size == 0:
        return (0, 0, width, height)

    x0 = int(xs.min()) - padding
    y0 = int(ys.min()) - padding
    x1 = int(xs.max()) + 1 + padding
    y1 = int(ys.max()) + 1 + padding

    if min_size is not None:
        min_width, min_height = min_size
        center_x = (x0 + x1) / 2.0
        center_y = (y0 + y1) / 2.0
        box_width = max(x1 - x0, min_width)
        box_height = max(y1 - y0, min_height)
        x0 = int(np.floor(center_x - box_width / 2.0))
        x1 = int(np.ceil(center_x + box_width / 2.0))
        y0 = int(np.floor(center_y - box_height / 2.0))
        y1 = int(np.ceil(center_y + box_height / 2.0))

    x0, x1 = clamp_interval(x0, x1, width)
    y0, y1 = clamp_interval(y0, y1, height)
    return (x0, y0, x1, y1)


def clamp_interval(start: int, end: int, limit: int) -> tuple[int, int]:
    length = end - start
    if length >= limit:
        return 0, limit
    if start < 0:
        end -= start
        start = 0
    if end > limit:
        start -= end - limit
        end = limit
    return max(0, start), min(limit, end)


def resize_to_height(image: Image.Image, target_height: int, resample: Image.Resampling) -> Image.Image:
    scale = target_height / float(image.height)
    return image.resize((max(1, int(image.width * scale)), target_height), resample)


def overlay_label(image: Image.Image, label: np.ndarray) -> Image.Image:
    if label.shape[:2] != (image.height, image.width):
        label = np.asarray(Image.fromarray(label.astype(np.uint8)).resize(image.size, Image.Resampling.NEAREST))
    return Image.alpha_composite(image.convert("RGBA"), label_to_rgba(label)).convert("RGB")


def label_mask_image(label: np.ndarray, size: tuple[int, int]) -> Image.Image:
    image = label_to_rgba(label).convert("RGB")
    if image.size != size:
        image = image.resize(size, Image.Resampling.NEAREST)
    return image


def render_portrait_label(frame_path: Path, portrait: Image.Image, uv_label: np.ndarray) -> np.ndarray | None:
    try:
        frame = json.loads(frame_path.read_text(encoding="utf-8"))
        return render_uv_label_to_portrait(frame, uv_label, portrait.size)
    except Exception:
        # Some tests and partial exports do not contain full ARKit projection data.
        return None


def render_portrait_overlay(frame_path: Path, portrait: Image.Image, uv_label: np.ndarray) -> Image.Image:
    portrait_label = render_portrait_label(frame_path, portrait, uv_label)
    if portrait_label is None:
        return portrait.copy()
    return overlay_label(portrait, portrait_label)


def write_preview(root: Path, candidate: Candidate, output_path: Path, uv_image_dir: str = "uv_images_128_culled") -> None:
    stem = str(candidate.payload["stem"])
    split_root = root / candidate.split
    image_path = split_root / "images" / f"{stem}.jpg"
    label_path = split_root / "uv_labels" / f"{stem}.png"
    frame_path = split_root / "frames" / f"{stem}.json"
    uv_image_path = split_root / uv_image_dir / f"{stem}.jpg"

    portrait = Image.open(image_path).convert("RGB")
    uv_label = np.asarray(Image.open(label_path).convert("L"), dtype=np.uint8)
    portrait_overlay = render_portrait_overlay(frame_path, portrait, uv_label)

    if uv_image_path.exists():
        uv_image = Image.open(uv_image_path).convert("RGB")
    else:
        uv_image = Image.new("RGB", (uv_label.shape[1], uv_label.shape[0]), (96, 80, 70))
    uv_overlay = overlay_label(uv_image, uv_label)
    uv_mask = label_mask_image(uv_label, uv_image.size)

    target_height = 220
    panels = [portrait, portrait_overlay, uv_overlay, uv_mask]
    resized: list[Image.Image] = []
    for panel in panels:
        scale = target_height / float(panel.height)
        resized.append(panel.resize((max(1, int(panel.width * scale)), target_height), Image.Resampling.BILINEAR))

    reasons = ",".join(candidate.reasons) if candidate.reasons else "baseline"
    output = hstack(
        [
            titled(resized[0], "portrait"),
            titled(resized[1], f"projected label score={candidate.score:.1f}"),
            titled(resized[2], reasons),
            titled(resized[3], "uv label"),
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.save(output_path, quality=92)


def write_hires_preview(root: Path, candidate: Candidate, output_path: Path, uv_image_dir: str = "uv_images_128_culled") -> None:
    stem = str(candidate.payload["stem"])
    split_root = root / candidate.split
    image_path = split_root / "images" / f"{stem}.jpg"
    label_path = split_root / "uv_labels" / f"{stem}.png"
    frame_path = split_root / "frames" / f"{stem}.json"
    uv_image_path = split_root / uv_image_dir / f"{stem}.jpg"

    portrait = Image.open(image_path).convert("RGB")
    uv_label = np.asarray(Image.open(label_path).convert("L"), dtype=np.uint8)
    portrait_label = render_portrait_label(frame_path, portrait, uv_label)
    if portrait_label is None:
        portrait_label = np.zeros((portrait.height, portrait.width), dtype=np.uint8)
    portrait_overlay = overlay_label(portrait, portrait_label)

    if (portrait_label > 0).any():
        mouth_box = expanded_bbox_from_mask(
            portrait_label > 0,
            padding=max(32, int(min(portrait.width, portrait.height) * 0.05)),
            min_size=(360, 260),
        )
    else:
        center_x = portrait.width // 2
        center_y = int(portrait.height * 0.58)
        mouth_box = (
            max(0, center_x - 180),
            max(0, center_y - 130),
            min(portrait.width, center_x + 180),
            min(portrait.height, center_y + 130),
        )

    mouth_crop = portrait.crop(mouth_box)
    mouth_overlay_crop = portrait_overlay.crop(mouth_box)

    if uv_image_path.exists():
        uv_image = Image.open(uv_image_path).convert("RGB")
    else:
        uv_image = Image.new("RGB", (uv_label.shape[1], uv_label.shape[0]), (96, 80, 70))
    uv_label_for_image = np.asarray(
        Image.fromarray(uv_label.astype(np.uint8)).resize(uv_image.size, Image.Resampling.NEAREST),
        dtype=np.uint8,
    )
    uv_box = expanded_bbox_from_mask(
        uv_label_for_image > 0,
        padding=max(8, int(min(uv_image.size) * 0.12)),
        min_size=(min(140, uv_image.width), min(110, uv_image.height)),
    )
    uv_overlay_crop = overlay_label(uv_image, uv_label_for_image).crop(uv_box)
    uv_mask_crop = label_mask_image(uv_label_for_image, uv_image.size).crop(uv_box)

    target_height = 420
    panels = [
        titled(resize_to_height(mouth_crop, target_height, Image.Resampling.BILINEAR), "mouth crop"),
        titled(resize_to_height(mouth_overlay_crop, target_height, Image.Resampling.BILINEAR), f"projected label score={candidate.score:.1f}"),
        titled(resize_to_height(uv_overlay_crop, target_height, Image.Resampling.BILINEAR), "uv crop + label"),
        titled(resize_to_height(uv_mask_crop, target_height, Image.Resampling.NEAREST), "uv label crop"),
    ]
    output = hstack(panels)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.save(output_path, quality=95)


def write_manifest(root: Path, candidates: list[Candidate], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(candidate.to_json(root), sort_keys=True) for candidate in candidates]
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def parse_splits(value: str) -> tuple[str, ...]:
    splits = tuple(part.strip() for part in value.split(",") if part.strip())
    if not splits:
        raise argparse.ArgumentTypeError("--splits must contain at least one split")
    return splits


def parse_exclude_stems(value: str | None) -> set[str]:
    if not value:
        return set()
    return {part.strip() for part in value.split(",") if part.strip()}


def load_exclude_stems(value: str | None, file_path: str | None) -> set[str]:
    stems = parse_exclude_stems(value)
    if file_path:
        path = Path(file_path).expanduser()
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                stems.add(line)
    return stems


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rank ARKit UV lip frames that are worth label refinement.")
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--splits", type=parse_splits, default=("train",))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--count", type=int, default=48)
    parser.add_argument("--preview-count", type=int, default=None)
    parser.add_argument("--max-per-session", type=int, default=8)
    parser.add_argument("--uv-image-dir", default="uv_images_128_culled")
    parser.add_argument("--hires", action="store_true", help="Write zoomed mouth/UV review images instead of compact full-face previews.")
    parser.add_argument("--no-previews", action="store_true")
    parser.add_argument("--exclude-stems", default=None, help="Comma-separated stems to exclude from selection.")
    parser.add_argument("--exclude-stems-file", default=None, help="Text file with one stem per line to exclude from selection.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    root = Path(args.root).expanduser()
    output_dir = Path(args.output_dir)
    preview_count = args.count if args.preview_count is None else args.preview_count

    all_candidates: list[Candidate] = []
    for split in args.splits:
        all_candidates.extend(select_candidates(root, split, count=args.count, max_per_session=args.max_per_session))
    all_candidates.sort(key=lambda candidate: (-candidate.score, candidate.split, str(candidate.payload.get("stem", ""))))
    exclude_stems = load_exclude_stems(args.exclude_stems, args.exclude_stems_file)
    if exclude_stems:
        all_candidates = [candidate for candidate in all_candidates if str(candidate.payload.get("stem", "")) not in exclude_stems]
    all_candidates = all_candidates[: args.count]

    manifest_path = output_dir / "candidates.jsonl"
    write_manifest(root, all_candidates, manifest_path)

    if not args.no_previews:
        preview_dir = output_dir / ("hires" if args.hires else "previews")
        for index, candidate in enumerate(all_candidates[:preview_count], start=1):
            stem = str(candidate.payload["stem"])
            preview_path = preview_dir / f"candidate_{index:03d}_{candidate.split}_{stem}.jpg"
            if args.hires:
                write_hires_preview(root, candidate, preview_path, uv_image_dir=args.uv_image_dir)
            else:
                write_preview(root, candidate, preview_path, uv_image_dir=args.uv_image_dir)

    summary = {
        "root": str(root),
        "splits": list(args.splits),
        "count": len(all_candidates),
        "exclude_count": len(exclude_stems),
        "manifest": str(manifest_path),
        "preview_dir": None if args.no_previews else str(output_dir / ("hires" if args.hires else "previews")),
        "top": [candidate.to_json(root) for candidate in all_candidates[:5]],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
