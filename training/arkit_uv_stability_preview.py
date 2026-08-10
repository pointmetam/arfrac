from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from teacher_2d_to_arkit_uv import (
    debug_lip_roi_label,
    label_to_rgba,
    overlay_label,
    portrait_image,
    rasterize_2d_label_to_uv,
    render_uv_label_to_portrait,
)


SourceKind = Literal["debug-roi", "model", "segface"]

CLASS_NAMES = {
    1: "upper_lip",
    2: "lower_lip",
    3: "inner_mouth",
}


def binary_iou(a: np.ndarray, b: np.ndarray) -> float | None:
    a_mask = a > 0
    b_mask = b > 0
    union = np.logical_or(a_mask, b_mask).sum()
    if union == 0:
        return None
    return float(np.logical_and(a_mask, b_mask).sum() / union)


def class_iou(a: np.ndarray, b: np.ndarray, class_id: int) -> float | None:
    a_mask = a == class_id
    b_mask = b == class_id
    union = np.logical_or(a_mask, b_mask).sum()
    if union == 0:
        return None
    return float(np.logical_and(a_mask, b_mask).sum() / union)


def mask_centroid(mask: np.ndarray) -> tuple[float | None, float | None]:
    ys, xs = np.nonzero(mask > 0)
    if xs.size == 0:
        return None, None
    return float(xs.mean()), float(ys.mean())


def mask_bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(mask > 0)
    if xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def face_yaw_deg(frame: dict[str, Any]) -> float:
    transform = np.array(frame["faceTransform"], dtype=np.float64).reshape(4, 4)
    rotation = transform[:3, :3]
    return float(math.degrees(math.atan2(rotation[0, 2], rotation[2, 2])))


def blend_shape(frame: dict[str, Any], key: str) -> float:
    value = frame.get("blendShapes", {}).get(key)
    return float(value) if value is not None else 0.0


def stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "median": None, "mean": None, "max": None}
    array = np.array(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "max": float(array.max()),
    }


def coefficient_of_variation(values: list[float]) -> float | None:
    if not values:
        return None
    array = np.array(values, dtype=np.float64)
    mean = float(array.mean())
    if abs(mean) < 1e-8:
        return None
    return float(array.std() / mean)


def load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def annotate(image: Image.Image, lines: list[str]) -> Image.Image:
    font = load_font(16)
    line_height = 20
    pad = 8
    header_height = pad * 2 + line_height * len(lines)
    output = Image.new("RGB", (image.width, image.height + header_height), "white")
    draw = ImageDraw.Draw(output)
    for index, line in enumerate(lines):
        draw.text((pad, pad + index * line_height), line, fill=(20, 20, 20), font=font)
    output.paste(image.convert("RGB"), (0, header_height))
    return output


def fit_width(image: Image.Image, width: int) -> Image.Image:
    if image.width == width:
        return image
    height = max(1, int(round(image.height * width / image.width)))
    return image.resize((width, height), Image.Resampling.BILINEAR)


def make_grid(images: list[Image.Image], columns: int, background: str = "white") -> Image.Image:
    if not images:
        return Image.new("RGB", (1, 1), background)
    cell_width = max(image.width for image in images)
    cell_height = max(image.height for image in images)
    rows = int(math.ceil(len(images) / columns))
    output = Image.new("RGB", (cell_width * columns, cell_height * rows), background)
    for index, image in enumerate(images):
        x = (index % columns) * cell_width
        y = (index // columns) * cell_height
        output.paste(image.convert("RGB"), (x, y))
    return output


def make_panel(
    portrait: Image.Image,
    source_overlay: Image.Image,
    reprojected: np.ndarray,
    uv_label: np.ndarray,
    metric: dict[str, Any],
    panel_width: int,
) -> Image.Image:
    uv_overlay = overlay_label(portrait, reprojected)
    uv_rgb = label_to_rgba(uv_label).convert("RGB")
    preview_width = min(portrait.width, max(1, portrait.height * 2 // 3))
    uv_rgb = uv_rgb.resize((preview_width, preview_width), Image.Resampling.NEAREST)
    combined_width = portrait.width * 2 + preview_width
    combined = Image.new("RGB", (combined_width, max(source_overlay.height, uv_overlay.height, uv_rgb.height)), "white")
    combined.paste(source_overlay.convert("RGB"), (0, 0))
    combined.paste(uv_overlay, (portrait.width, 0))
    combined.paste(uv_rgb, (portrait.width * 2, 0))

    iou_text = "n/a" if metric["prev_binary_iou"] is None else f"{metric['prev_binary_iou']:.3f}"
    delta_text = "n/a" if metric["centroid_delta_uv_px"] is None else f"{metric['centroid_delta_uv_px']:.1f}px"
    annotated = annotate(
        combined,
        [
            f"{metric['sample_id']} yaw {metric['yaw_deg']:+.1f} jaw {metric['jaw_open']:.2f}",
            f"area {metric['lip_area_px']} px  prev IoU {iou_text}  centroid d {delta_text}",
        ],
    )
    return fit_width(annotated, panel_width)


def select_evenly(items: list[Image.Image], count: int) -> list[Image.Image]:
    if count <= 0 or len(items) <= count:
        return items
    indices = np.linspace(0, len(items) - 1, count).round().astype(int)
    return [items[int(index)] for index in indices]


class LipModelRunner:
    def __init__(
        self,
        checkpoint_path: Path,
        input_width: int | None,
        input_height: int | None,
        model_width: int | None,
        device_name: str,
    ) -> None:
        import torch
        from model import build_model

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.input_width = int(input_width or checkpoint.get("input_width", 128))
        self.input_height = int(input_height or checkpoint.get("input_height", 96))
        self.num_classes = int(checkpoint.get("num_classes", 4))
        checkpoint_args = checkpoint.get("args") if isinstance(checkpoint.get("args"), dict) else {}
        inferred_width = int(checkpoint["model_state"]["stem.block.0.weight"].shape[0])
        self.model_width = int(model_width or checkpoint.get("model_width") or checkpoint_args.get("model_width") or inferred_width)

        if device_name == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device_name)

        self.torch = torch
        self.model = build_model(num_classes=self.num_classes, width=self.model_width).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

    def predict_crop(self, crop: Image.Image) -> np.ndarray:
        image = crop.convert("RGB").resize((self.input_width, self.input_height), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32).copy() / 255.0
        tensor = self.torch.from_numpy(array).permute(2, 0, 1)
        tensor = (tensor - self.mean) / self.std
        tensor = tensor.unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            logits = self.model(tensor)
            prediction = logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)
        return prediction


class SegFaceRunner:
    def __init__(
        self,
        repo_dir: Path,
        checkpoint_path: Path,
        resolution: int,
        device_name: str,
    ) -> None:
        import torch

        os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
        import facer  # type: ignore

        if device_name == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device_name)

        sys.path.insert(0, str(repo_dir))
        from network import get_model  # type: ignore

        self.torch = torch
        self.resolution = int(resolution)
        self.detector = facer.face_detector("retinaface/mobilenet", device=self.device)
        self.model = get_model("segface_lapa", self.resolution, "swin_base").to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(checkpoint["state_dict_backbone"])
        self.model.eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

    @staticmethod
    def _select_face_index(faces: dict[str, Any]) -> int:
        import torch

        rects = faces.get("rects", faces.get("boxes"))
        if torch.is_tensor(rects) and rects.ndim == 2 and rects.shape[0] > 0 and rects.shape[1] >= 4:
            rects_cpu = rects.detach().cpu()
            widths = (rects_cpu[:, 2] - rects_cpu[:, 0]).clamp_min(0)
            heights = (rects_cpu[:, 3] - rects_cpu[:, 1]).clamp_min(0)
            return int(torch.argmax(widths * heights).item())
        return 0

    def _image_tensor(self, image: Image.Image) -> Any:
        resized = image.convert("RGB").resize((self.resolution, self.resolution), Image.Resampling.BICUBIC)
        array = np.asarray(resized, dtype=np.float32).copy() / 255.0
        tensor = self.torch.from_numpy(array).permute(2, 0, 1)
        tensor = (tensor - self.mean) / self.std
        return tensor.unsqueeze(0).to(self.device)

    def predict(self, image: Image.Image) -> np.ndarray | None:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        detector_input = self.torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with self.torch.no_grad():
            faces = self.detector(detector_input)
            rects = faces.get("rects", faces.get("boxes")) if isinstance(faces, dict) else None
            points = faces.get("points") if isinstance(faces, dict) else None
            if rects is None or points is None or int(rects.shape[0]) == 0:
                return None

            face_index = min(self._select_face_index(faces), points.shape[0] - 1)
            five_points = points[face_index : face_index + 1].float()
            five_points = five_points * self.torch.tensor(
                [self.resolution / float(image.width), self.resolution / float(image.height)],
                dtype=self.torch.float32,
                device=five_points.device,
            )

            labels = {"lnm_seg": five_points.to(self.device)}
            dataset = self.torch.ones((1,), dtype=self.torch.long, device=self.device)
            logits = self.model(self._image_tensor(image), labels, dataset)
            logits = self.torch.nn.functional.interpolate(
                logits,
                size=(self.resolution, self.resolution),
                mode="bilinear",
                align_corners=False,
            )
            prediction = logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)

        if prediction.shape != (image.height, image.width):
            prediction = np.asarray(
                Image.fromarray(prediction).resize(image.size, Image.Resampling.NEAREST),
                dtype=np.uint8,
            )

        label = np.zeros(prediction.shape, dtype=np.uint8)
        label[prediction == 7] = 1
        label[prediction == 9] = 2
        label[prediction == 8] = 3
        return label


def clamp_roi(roi: dict[str, Any], image_size: tuple[int, int]) -> tuple[int, int, int, int] | None:
    image_width, image_height = image_size
    x0 = int(math.floor(float(roi["x"])))
    y0 = int(math.floor(float(roi["y"])))
    x1 = int(math.ceil(float(roi["x"]) + float(roi["width"])))
    y1 = int(math.ceil(float(roi["y"]) + float(roi["height"])))
    x0 = min(max(x0, 0), image_width)
    y0 = min(max(y0, 0), image_height)
    x1 = min(max(x1, 0), image_width)
    y1 = min(max(y1, 0), image_height)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    return x0, y0, x1, y1


def model_lip_label(frame: dict[str, Any], portrait: Image.Image, runner: LipModelRunner) -> np.ndarray | None:
    roi = frame.get("lipROI")
    if not roi:
        return None
    crop_box = clamp_roi(roi, portrait.size)
    if crop_box is None:
        return None

    crop = portrait.crop(crop_box)
    crop_label = runner.predict_crop(crop)
    crop_label_image = Image.fromarray(crop_label).resize(crop.size, Image.Resampling.NEAREST)
    label = Image.new("L", portrait.size, 0)
    label.paste(crop_label_image, crop_box[:2])
    return np.asarray(label, dtype=np.uint8)


def build_source_label(
    source: SourceKind,
    frame: dict[str, Any],
    portrait: Image.Image,
    portrait_scale: float,
    model_runner: LipModelRunner | None,
    segface_runner: SegFaceRunner | None,
) -> np.ndarray | None:
    if source == "debug-roi":
        return debug_lip_roi_label(frame, scale=portrait_scale)
    if source == "model":
        if model_runner is None:
            raise RuntimeError("--source model requires --checkpoint")
        return model_lip_label(frame, portrait, model_runner)
    if source == "segface":
        if segface_runner is None:
            raise RuntimeError("--source segface requires SegFace runner")
        return segface_runner.predict(portrait)
    raise RuntimeError(f"Unknown source: {source}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True, help="ARKitLipUVRecordings/<session> directory")
    parser.add_argument("--output-dir", default="training/outputs/arkit_uv_stability_debug")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=0, help="0 means all frames after --start")
    parser.add_argument("--uv-size", type=int, default=512)
    parser.add_argument("--portrait-scale", type=float, default=1.0)
    parser.add_argument("--cull-backfaces", action="store_true")
    parser.add_argument("--front-face-winding", choices=["negative", "positive"], default="negative")
    parser.add_argument("--source", choices=["debug-roi", "model", "segface"], default="debug-roi")
    parser.add_argument("--checkpoint", default="training/outputs/lip_realtime_128x96_w16/best.pt")
    parser.add_argument("--model-device", default="auto")
    parser.add_argument("--model-input-width", type=int, default=None)
    parser.add_argument("--model-input-height", type=int, default=None)
    parser.add_argument("--model-width", type=int, default=None)
    parser.add_argument("--segface-root", default="third_party/SegFace")
    parser.add_argument("--segface-checkpoint", default="/Users/lee/Models/SegFace/swinb_lapa_512/model_299.pt")
    parser.add_argument("--segface-resolution", type=int, default=512)
    parser.add_argument("--segface-device", default="auto")
    parser.add_argument("--gif-count", type=int, default=32)
    parser.add_argument("--contact-count", type=int, default=16)
    parser.add_argument("--panel-width", type=int, default=720)
    args = parser.parse_args()

    session_dir = Path(args.session).expanduser()
    output_dir = Path(args.output_dir)
    label_dir = output_dir / "uv_labels"
    frame_dir = output_dir / "frames"
    output_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    frame_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted((session_dir / "frames").glob("*.json"))
    if args.count > 0:
        frame_paths = frame_paths[args.start : args.start + args.count]
    else:
        frame_paths = frame_paths[args.start :]

    model_runner = None
    if args.source == "model":
        model_runner = LipModelRunner(
            Path(args.checkpoint).expanduser(),
            args.model_input_width,
            args.model_input_height,
            args.model_width,
            args.model_device,
        )
        print(
            "Model source: "
            f"{args.checkpoint} input={model_runner.input_width}x{model_runner.input_height} "
            f"width={model_runner.model_width} device={model_runner.device}"
        )

    segface_runner = None
    if args.source == "segface":
        segface_runner = SegFaceRunner(
            Path(args.segface_root).expanduser(),
            Path(args.segface_checkpoint).expanduser(),
            args.segface_resolution,
            args.segface_device,
        )
        print(
            "SegFace source: "
            f"{args.segface_checkpoint} resolution={segface_runner.resolution} "
            f"device={segface_runner.device}"
        )

    metrics: list[dict[str, Any]] = []
    panels: list[Image.Image] = []
    prev_label: np.ndarray | None = None
    prev_centroid: tuple[float | None, float | None] = (None, None)

    for frame_path in tqdm(frame_paths):
        frame = json.loads(frame_path.read_text(encoding="utf-8"))
        image_path = session_dir / frame["image"]["file"]
        raw_image = Image.open(image_path).convert("RGB")
        portrait = portrait_image(raw_image, frame, scale=args.portrait_scale)
        source_label = build_source_label(args.source, frame, portrait, args.portrait_scale, model_runner, segface_runner)
        if source_label is None:
            continue
        uv_label = rasterize_2d_label_to_uv(
            frame,
            source_label,
            "portrait",
            args.uv_size,
            cull_backfaces=args.cull_backfaces,
            front_face_winding=args.front_face_winding,
        )

        sample_id = str(frame["sampleID"])
        Image.fromarray(uv_label).save(label_dir / f"{sample_id}.png")

        centroid = mask_centroid(uv_label)
        centroid_delta = None
        if centroid[0] is not None and prev_centroid[0] is not None:
            centroid_delta = float(math.hypot(centroid[0] - prev_centroid[0], centroid[1] - prev_centroid[1]))

        metric = {
            "sample_id": sample_id,
            "timestamp": float(frame.get("timestamp", 0.0)),
            "yaw_deg": face_yaw_deg(frame),
            "jaw_open": blend_shape(frame, "jawOpen"),
            "mouth_close": blend_shape(frame, "mouthClose"),
            "lip_area_px": int((uv_label > 0).sum()),
            "upper_lip_area_px": int((uv_label == 1).sum()),
            "lower_lip_area_px": int((uv_label == 2).sum()),
            "inner_mouth_area_px": int((uv_label == 3).sum()),
            "centroid_uv_px": [centroid[0], centroid[1]],
            "centroid_uv_norm": [
                None if centroid[0] is None else centroid[0] / float(args.uv_size),
                None if centroid[1] is None else centroid[1] / float(args.uv_size),
            ],
            "centroid_delta_uv_px": centroid_delta,
            "bbox_uv_px": mask_bbox(uv_label),
            "prev_binary_iou": None if prev_label is None else binary_iou(prev_label, uv_label),
            "prev_upper_lip_iou": None if prev_label is None else class_iou(prev_label, uv_label, 1),
            "prev_lower_lip_iou": None if prev_label is None else class_iou(prev_label, uv_label, 2),
            "prev_inner_mouth_iou": None if prev_label is None else class_iou(prev_label, uv_label, 3),
        }
        metrics.append(metric)

        reprojected = render_uv_label_to_portrait(frame, uv_label, portrait.size)
        source_overlay = overlay_label(portrait, source_label)
        panel = make_panel(portrait, source_overlay, reprojected, uv_label, metric, args.panel_width)
        frame_path_out = frame_dir / f"{sample_id}.jpg"
        panel.save(frame_path_out, quality=92)
        panels.append(panel)

        prev_label = uv_label
        prev_centroid = centroid

    csv_path = output_dir / "metrics.csv"
    if metrics:
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(metrics[0].keys()))
            writer.writeheader()
            writer.writerows(metrics)

    ious = [value for item in metrics for value in [item["prev_binary_iou"]] if value is not None]
    deltas = [value for item in metrics for value in [item["centroid_delta_uv_px"]] if value is not None]
    areas = [float(item["lip_area_px"]) for item in metrics if item["lip_area_px"] > 0]
    summary = {
        "session": str(session_dir),
        "source": args.source,
        "checkpoint": str(Path(args.checkpoint).expanduser()) if args.source == "model" else None,
        "segface_checkpoint": str(Path(args.segface_checkpoint).expanduser()) if args.source == "segface" else None,
        "segface_resolution": args.segface_resolution if args.source == "segface" else None,
        "uv_size": args.uv_size,
        "cull_backfaces": args.cull_backfaces,
        "front_face_winding": args.front_face_winding if args.cull_backfaces else None,
        "frame_count": len(metrics),
        "yaw_deg": stats([float(item["yaw_deg"]) for item in metrics]),
        "jaw_open": stats([float(item["jaw_open"]) for item in metrics]),
        "lip_area_px": stats(areas),
        "lip_area_cv": coefficient_of_variation(areas),
        "prev_binary_iou": stats(ious),
        "centroid_delta_uv_px": stats(deltas),
        "labels_dir": str(label_dir),
        "frames_dir": str(frame_dir),
        "metrics_csv": str(csv_path),
    }

    contact_frames = select_evenly(panels, args.contact_count)
    contact_path = output_dir / "contact_sheet.jpg"
    make_grid(contact_frames, columns=4).save(contact_path, quality=92)
    summary["contact_sheet"] = str(contact_path)

    gif_frames = select_evenly(panels, args.gif_count)
    gif_path = output_dir / "stability_preview.gif"
    if gif_frames:
        gif_frames[0].save(
            gif_path,
            save_all=True,
            append_images=gif_frames[1:],
            duration=180,
            loop=0,
            optimize=True,
        )
    summary["gif"] = str(gif_path)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
