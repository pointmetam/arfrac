from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


CLASS_COLORS = {
    1: (236, 36, 82, 220),
    2: (255, 92, 128, 220),
    3: (24, 24, 28, 230),
}

SourceSpace = Literal["image", "portrait"]


def affine_matrix(payload: dict[str, float]) -> np.ndarray:
    return np.array(
        [
            [payload["a"], payload["c"], payload["tx"]],
            [payload["b"], payload["d"], payload["ty"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float64)], axis=1)
    transformed = np.einsum("ij,nj->ni", matrix, homogeneous)
    return transformed[:, :2]


def project_vertices(frame: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    geometry = frame["faceGeometry"]
    vertices = np.array(
        [[vertex["x"], vertex["y"], vertex["z"], 1.0] for vertex in geometry["vertices"]],
        dtype=np.float64,
    )
    face_transform = np.array(frame["faceTransform"], dtype=np.float64).reshape(4, 4)
    view_matrix = np.array(frame["viewMatrixPortrait"], dtype=np.float64).reshape(4, 4)
    projection_matrix = np.array(frame["projectionMatrixPortrait"], dtype=np.float64).reshape(4, 4)

    model_view_projection = projection_matrix @ view_matrix @ face_transform
    clip = np.einsum("ij,nj->ni", model_view_projection, vertices)
    valid = np.isfinite(clip).all(axis=1) & (np.abs(clip[:, 3]) > 1e-8)
    ndc = np.zeros((vertices.shape[0], 3), dtype=np.float64)
    ndc[valid] = clip[valid, :3] / clip[valid, 3:4]

    viewport_width = float(frame["viewport"]["width"])
    viewport_height = float(frame["viewport"]["height"])
    portrait_points = np.zeros((vertices.shape[0], 2), dtype=np.float64)
    portrait_points[:, 0] = (ndc[:, 0] + 1.0) * 0.5 * viewport_width
    portrait_points[:, 1] = (1.0 - ndc[:, 1]) * 0.5 * viewport_height

    view_norm = portrait_points / np.array([viewport_width, viewport_height], dtype=np.float64)
    image_norm = transform_points(np.linalg.inv(affine_matrix(frame["displayTransformPortrait"])), view_norm)
    image_points = image_norm * np.array([frame["image"]["width"], frame["image"]["height"]], dtype=np.float64)
    valid &= np.isfinite(portrait_points).all(axis=1) & np.isfinite(image_points).all(axis=1)
    return portrait_points, image_points, valid


def barycentric_grid(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    min_x: int,
    max_x: int,
    min_y: int,
    max_y: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs = np.arange(min_x, max_x + 1, dtype=np.float64) + 0.5
    ys = np.arange(min_y, max_y + 1, dtype=np.float64) + 0.5
    grid_x, grid_y = np.meshgrid(xs, ys)

    denominator = ((p1[1] - p2[1]) * (p0[0] - p2[0])) + ((p2[0] - p1[0]) * (p0[1] - p2[1]))
    if abs(denominator) < 1e-8:
        empty = np.zeros((0, 0), dtype=np.float64)
        return empty, empty, empty, empty.astype(bool), empty.astype(int)

    w0 = (((p1[1] - p2[1]) * (grid_x - p2[0])) + ((p2[0] - p1[0]) * (grid_y - p2[1]))) / denominator
    w1 = (((p2[1] - p0[1]) * (grid_x - p2[0])) + ((p0[0] - p2[0]) * (grid_y - p2[1]))) / denominator
    w2 = 1.0 - w0 - w1
    inside = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
    return w0, w1, w2, inside, np.indices(inside.shape)


def sample_label_nearest(label: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xs = np.rint(points[:, 0]).astype(np.int64)
    ys = np.rint(points[:, 1]).astype(np.int64)
    valid = (xs >= 0) & (xs < label.shape[1]) & (ys >= 0) & (ys < label.shape[0])
    values = np.zeros(points.shape[0], dtype=np.uint8)
    values[valid] = label[ys[valid], xs[valid]]
    return values, valid


def rasterize_2d_label_to_uv(
    frame: dict[str, Any],
    source_label: np.ndarray,
    source_space: SourceSpace,
    uv_size: int,
    cull_backfaces: bool = False,
    front_face_winding: Literal["negative", "positive"] = "negative",
) -> np.ndarray:
    portrait_points, image_points, valid_points = project_vertices(frame)
    geometry = frame["faceGeometry"]
    uvs = np.array(
        [[point["x"], point["y"]] for point in geometry["textureCoordinates"]],
        dtype=np.float64,
    )
    triangle_indices = np.array(geometry["triangleIndices"], dtype=np.int64)
    uv_label = np.zeros((uv_size, uv_size), dtype=np.uint8)

    if source_space == "portrait":
        scale = np.array(
            [
                source_label.shape[1] / float(frame["viewport"]["width"]),
                source_label.shape[0] / float(frame["viewport"]["height"]),
            ],
            dtype=np.float64,
        )
        source_points = portrait_points * scale
    else:
        scale = np.array(
            [
                source_label.shape[1] / float(frame["image"]["width"]),
                source_label.shape[0] / float(frame["image"]["height"]),
            ],
            dtype=np.float64,
        )
        source_points = image_points * scale

    uv_pixels = uvs * float(uv_size - 1)
    for offset in range(0, len(triangle_indices) - 2, 3):
        i0, i1, i2 = triangle_indices[offset : offset + 3]
        if not (valid_points[i0] and valid_points[i1] and valid_points[i2]):
            continue

        if cull_backfaces:
            s0, s1, s2 = portrait_points[i0], portrait_points[i1], portrait_points[i2]
            signed_area = ((s1[0] - s0[0]) * (s2[1] - s0[1])) - ((s1[1] - s0[1]) * (s2[0] - s0[0]))
            if front_face_winding == "negative":
                if signed_area >= -1e-6:
                    continue
            elif signed_area <= 1e-6:
                continue

        p0, p1, p2 = uv_pixels[i0], uv_pixels[i1], uv_pixels[i2]
        min_x = max(0, int(np.floor(min(p0[0], p1[0], p2[0]))))
        max_x = min(uv_size - 1, int(np.ceil(max(p0[0], p1[0], p2[0]))))
        min_y = max(0, int(np.floor(min(p0[1], p1[1], p2[1]))))
        max_y = min(uv_size - 1, int(np.ceil(max(p0[1], p1[1], p2[1]))))
        if min_x > max_x or min_y > max_y:
            continue

        w0, w1, w2, inside, indices = barycentric_grid(p0, p1, p2, min_x, max_x, min_y, max_y)
        if inside.size == 0 or not inside.any():
            continue

        src = (
            source_points[i0][None, None, :] * w0[:, :, None]
            + source_points[i1][None, None, :] * w1[:, :, None]
            + source_points[i2][None, None, :] * w2[:, :, None]
        )
        flat_src = src[inside]
        values, valid_samples = sample_label_nearest(source_label, flat_src)
        values = values[valid_samples]
        if values.size == 0:
            continue

        local_y = indices[0][inside][valid_samples]
        local_x = indices[1][inside][valid_samples]
        target_y = local_y + min_y
        target_x = local_x + min_x
        nonzero = values > 0
        uv_label[target_y[nonzero], target_x[nonzero]] = np.maximum(
            uv_label[target_y[nonzero], target_x[nonzero]],
            values[nonzero],
        )

    return uv_label


def render_uv_label_to_portrait(
    frame: dict[str, Any],
    uv_label: np.ndarray,
    output_size: tuple[int, int],
) -> np.ndarray:
    portrait_points, _, valid_points = project_vertices(frame)
    geometry = frame["faceGeometry"]
    uvs = np.array(
        [[point["x"], point["y"]] for point in geometry["textureCoordinates"]],
        dtype=np.float64,
    )
    triangle_indices = np.array(geometry["triangleIndices"], dtype=np.int64)
    output = np.zeros((output_size[1], output_size[0]), dtype=np.uint8)
    scale = np.array(
        [
            output_size[0] / float(frame["viewport"]["width"]),
            output_size[1] / float(frame["viewport"]["height"]),
        ],
        dtype=np.float64,
    )
    screen_pixels = portrait_points * scale
    uv_pixels = uvs * np.array([uv_label.shape[1] - 1, uv_label.shape[0] - 1], dtype=np.float64)

    for offset in range(0, len(triangle_indices) - 2, 3):
        i0, i1, i2 = triangle_indices[offset : offset + 3]
        if not (valid_points[i0] and valid_points[i1] and valid_points[i2]):
            continue

        p0, p1, p2 = screen_pixels[i0], screen_pixels[i1], screen_pixels[i2]
        min_x = max(0, int(np.floor(min(p0[0], p1[0], p2[0]))))
        max_x = min(output_size[0] - 1, int(np.ceil(max(p0[0], p1[0], p2[0]))))
        min_y = max(0, int(np.floor(min(p0[1], p1[1], p2[1]))))
        max_y = min(output_size[1] - 1, int(np.ceil(max(p0[1], p1[1], p2[1]))))
        if min_x > max_x or min_y > max_y:
            continue

        w0, w1, w2, inside, indices = barycentric_grid(p0, p1, p2, min_x, max_x, min_y, max_y)
        if inside.size == 0 or not inside.any():
            continue

        uv = (
            uv_pixels[i0][None, None, :] * w0[:, :, None]
            + uv_pixels[i1][None, None, :] * w1[:, :, None]
            + uv_pixels[i2][None, None, :] * w2[:, :, None]
        )
        flat_uv = uv[inside]
        values, valid_samples = sample_label_nearest(uv_label, flat_uv)
        values = values[valid_samples]
        if values.size == 0:
            continue

        local_y = indices[0][inside][valid_samples]
        local_x = indices[1][inside][valid_samples]
        target_y = local_y + min_y
        target_x = local_x + min_x
        nonzero = values > 0
        output[target_y[nonzero], target_x[nonzero]] = np.maximum(output[target_y[nonzero], target_x[nonzero]], values[nonzero])

    return output


def inverse_display_coefficients(frame: dict[str, Any], output_size: tuple[int, int]) -> tuple[float, float, float, float, float, float]:
    inverse = np.linalg.inv(affine_matrix(frame["displayTransformPortrait"]))
    image_width = float(frame["image"]["width"])
    image_height = float(frame["image"]["height"])
    output_width = float(output_size[0])
    output_height = float(output_size[1])
    return (
        image_width * inverse[0, 0] / output_width,
        image_width * inverse[0, 1] / output_height,
        image_width * inverse[0, 2],
        image_height * inverse[1, 0] / output_width,
        image_height * inverse[1, 1] / output_height,
        image_height * inverse[1, 2],
    )


def portrait_image(raw_image: Image.Image, frame: dict[str, Any], scale: float = 1.0) -> Image.Image:
    output_size = (
        max(1, int(round(float(frame["viewport"]["width"]) * scale))),
        max(1, int(round(float(frame["viewport"]["height"]) * scale))),
    )
    return raw_image.transform(
        output_size,
        Image.Transform.AFFINE,
        inverse_display_coefficients(frame, output_size),
        resample=Image.Resampling.BILINEAR,
        fillcolor=(0, 0, 0),
    )


def debug_lip_roi_label(frame: dict[str, Any], scale: float = 1.0) -> np.ndarray:
    width = max(1, int(round(float(frame["viewport"]["width"]) * scale)))
    height = max(1, int(round(float(frame["viewport"]["height"]) * scale)))
    label = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(label)
    roi = frame.get("lipROI")
    if not roi:
        return np.asarray(label, dtype=np.uint8)

    x0 = int(round(float(roi["x"]) * scale))
    y0 = int(round(float(roi["y"]) * scale))
    x1 = int(round((float(roi["x"]) + float(roi["width"])) * scale))
    y1 = int(round((float(roi["y"]) + float(roi["height"])) * scale))
    mid_y = int(round(y0 + (y1 - y0) * 0.48))
    draw.ellipse((x0, y0, x1, mid_y + int((y1 - y0) * 0.10)), fill=1)
    draw.ellipse((x0, mid_y - int((y1 - y0) * 0.10), x1, y1), fill=2)
    crease_h = max(2, int((y1 - y0) * 0.08))
    draw.ellipse((x0 + int((x1 - x0) * 0.14), mid_y - crease_h, x1 - int((x1 - x0) * 0.14), mid_y + crease_h), fill=3)
    return np.asarray(label, dtype=np.uint8)


def load_mask(path: Path) -> np.ndarray:
    label = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    if label.max(initial=0) > 3:
        label = (label > 0).astype(np.uint8)
    return label


def find_mask(mask_dir: Path, sample_id: str) -> Path | None:
    for suffix in (".png", ".jpg", ".jpeg", ".bmp"):
        candidate = mask_dir / f"{sample_id}{suffix}"
        if candidate.exists():
            return candidate
    return None


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def labels_from_parser(face_parser: Any) -> list[str] | None:
    for attr in ("labels", "label_names", "classes", "class_names"):
        value = getattr(face_parser, attr, None)
        if isinstance(value, (list, tuple)) and value:
            return [str(item) for item in value]
    config = getattr(face_parser, "config", None)
    if isinstance(config, dict):
        for key in ("labels", "label_names", "classes", "class_names"):
            value = config.get(key)
            if isinstance(value, (list, tuple)) and value:
                return [str(item) for item in value]
    return None


def lip_id_map(face_parser: Any, num_classes: int) -> dict[int, int]:
    labels = labels_from_parser(face_parser)
    if labels is None:
        return {7: 1, 9: 2, 8: 3}

    normalized = [normalize_name(label) for label in labels]
    candidates = {
        1: {"upper_lip", "u_lip", "ulip", "upperlip"},
        2: {"lower_lip", "l_lip", "llip", "lowerlip"},
        3: {"inner_mouth", "mouth", "i_mouth", "imouth", "oral"},
    }
    result: dict[int, int] = {}
    for output_id, names in candidates.items():
        for class_id, name in enumerate(normalized):
            if name in names:
                result[class_id] = output_id
                break
    return result if len(result) == 3 else {7: 1, 9: 2, 8: 3}


def select_face_index(faces: dict[str, Any]) -> int:
    import torch

    rects = faces.get("rects", faces.get("boxes"))
    if torch.is_tensor(rects) and rects.ndim == 2 and rects.shape[0] > 0 and rects.shape[1] >= 4:
        rects_cpu = rects.detach().cpu()
        widths = (rects_cpu[:, 2] - rects_cpu[:, 0]).clamp_min(0)
        heights = (rects_cpu[:, 3] - rects_cpu[:, 1]).clamp_min(0)
        return int(torch.argmax(widths * heights).item())
    return 0


def teacher_facer_label(
    image: Image.Image,
    face_detector: Any,
    face_parser: Any,
    device: Any,
) -> np.ndarray | None:
    import torch

    array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    image_tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        faces = face_detector(image_tensor)
        rects = faces.get("rects", faces.get("boxes")) if isinstance(faces, dict) else None
        if rects is None or int(rects.shape[0]) == 0:
            return None
        faces = face_parser(image_tensor, faces)

    seg_logits = faces.get("seg", {}).get("logits")
    if not torch.is_tensor(seg_logits) or seg_logits.ndim != 4:
        return None

    face_index = min(select_face_index(faces), seg_logits.shape[0] - 1)
    prediction = seg_logits.argmax(dim=1)[face_index].detach().cpu().numpy().astype(np.uint8)
    if prediction.shape != (image.height, image.width):
        prediction = np.asarray(Image.fromarray(prediction).resize(image.size, Image.Resampling.NEAREST), dtype=np.uint8)

    mapping = lip_id_map(face_parser, int(seg_logits.shape[1]))
    output = np.zeros(prediction.shape, dtype=np.uint8)
    for source_id, lip_id in mapping.items():
        output[prediction == source_id] = lip_id
    return output


def import_facer_teacher(device_name: str) -> tuple[Any, Any, Any]:
    import torch

    try:
        import facer  # type: ignore
    except ImportError as error:
        raise SystemExit(
            "Missing teacher dependency: facer.\n"
            "Create/activate .venv-teacher, then run:\n"
            "  pip install -r training/requirements-teacher.txt"
        ) from error

    if device_name == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_name)

    print(f"Loading FacePerceiver/facer teacher on {device}: farl/lapa/448")
    face_detector = facer.face_detector("retinaface/mobilenet", device=device)
    face_parser = facer.face_parser("farl/lapa/448", device=device)
    return face_detector, face_parser, device


def label_to_rgba(label: np.ndarray) -> Image.Image:
    rgba = np.zeros((label.shape[0], label.shape[1], 4), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        rgba[label == class_id] = color
    return Image.fromarray(rgba)


def overlay_label(image: Image.Image, label: np.ndarray) -> Image.Image:
    if label.shape[:2] != (image.height, image.width):
        label = np.asarray(Image.fromarray(label).resize(image.size, Image.Resampling.NEAREST), dtype=np.uint8)
    return Image.alpha_composite(image.convert("RGBA"), label_to_rgba(label)).convert("RGB")


def titled(image: Image.Image, title: str) -> Image.Image:
    title_height = 30
    output = Image.new("RGB", (image.width, image.height + title_height), "white")
    output.paste(image.convert("RGB"), (0, title_height))
    draw = ImageDraw.Draw(output)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    draw.text((8, 8), title, fill=(30, 30, 30), font=font)
    return output


def make_grid(columns: list[Image.Image]) -> Image.Image:
    height = max(column.height for column in columns)
    width = sum(column.width for column in columns)
    output = Image.new("RGB", (width, height), "white")
    x = 0
    for column in columns:
        output.paste(column, (x, 0))
        x += column.width
    return output


def build_source_label(
    args: argparse.Namespace,
    frame: dict[str, Any],
    raw_image: Image.Image,
    mask_dir: Path | None,
    teacher: tuple[Any, Any, Any] | None,
) -> tuple[np.ndarray | None, SourceSpace, Image.Image]:
    if args.debug_lip_roi_mask:
        label = debug_lip_roi_label(frame, scale=args.portrait_scale)
        return label, "portrait", portrait_image(raw_image, frame, scale=args.portrait_scale)

    if mask_dir is not None:
        mask_path = find_mask(mask_dir, frame["sampleID"])
        if mask_path is None:
            return None, args.mask_space, portrait_image(raw_image, frame, scale=args.portrait_scale)
        return load_mask(mask_path), args.mask_space, portrait_image(raw_image, frame, scale=args.portrait_scale)

    if teacher is not None:
        portrait = portrait_image(raw_image, frame, scale=args.teacher_scale)
        face_detector, face_parser, device = teacher
        label = teacher_facer_label(portrait, face_detector, face_parser, device)
        return label, "portrait", portrait

    raise SystemExit("Choose one source: --debug-lip-roi-mask, --mask-dir, or --teacher facer")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True, help="ARKitLipUVRecordings/<session> directory")
    parser.add_argument("--output-dir", default="training/outputs/arkit_uv_bridge")
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--uv-size", type=int, default=512)
    parser.add_argument("--preview-count", type=int, default=12)
    parser.add_argument("--portrait-scale", type=float, default=1.0)
    parser.add_argument("--cull-backfaces", action="store_true")
    parser.add_argument("--front-face-winding", choices=["negative", "positive"], default="negative")
    parser.add_argument("--debug-lip-roi-mask", action="store_true")
    parser.add_argument("--mask-dir")
    parser.add_argument("--mask-space", choices=["image", "portrait"], default="image")
    parser.add_argument("--teacher", choices=["none", "facer"], default="none")
    parser.add_argument("--teacher-device", default="auto")
    parser.add_argument("--teacher-scale", type=float, default=2.0)
    args = parser.parse_args()

    session_dir = Path(args.session).expanduser()
    output_dir = Path(args.output_dir)
    label_dir = output_dir / "uv_labels"
    preview_dir = output_dir / "previews"
    label_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    mask_dir = Path(args.mask_dir).expanduser() if args.mask_dir else None
    teacher = import_facer_teacher(args.teacher_device) if args.teacher == "facer" else None

    frame_paths = sorted((session_dir / "frames").glob("*.json"))
    frame_paths = frame_paths[args.start : args.start + args.count]

    saved_labels: list[str] = []
    saved_previews: list[str] = []
    skipped: list[str] = []

    for output_index, frame_path in enumerate(tqdm(frame_paths), start=1):
        frame = json.loads(frame_path.read_text(encoding="utf-8"))
        image_path = session_dir / frame["image"]["file"]
        raw_image = Image.open(image_path).convert("RGB")
        source_label, source_space, source_image = build_source_label(args, frame, raw_image, mask_dir, teacher)
        if source_label is None:
            skipped.append(frame["sampleID"])
            continue

        uv_label = rasterize_2d_label_to_uv(
            frame,
            source_label,
            source_space,
            args.uv_size,
            cull_backfaces=args.cull_backfaces,
            front_face_winding=args.front_face_winding,
        )
        label_path = label_dir / f"{frame['sampleID']}.png"
        Image.fromarray(uv_label).save(label_path)
        saved_labels.append(str(label_path))

        if output_index <= args.preview_count:
            portrait = portrait_image(raw_image, frame, scale=args.portrait_scale)
            reprojected = render_uv_label_to_portrait(frame, uv_label, portrait.size)
            uv_preview = label_to_rgba(uv_label).convert("RGB").resize(portrait.size, Image.Resampling.NEAREST)
            source_overlay = overlay_label(source_image, source_label).resize(portrait.size, Image.Resampling.BILINEAR)
            reproject_overlay = overlay_label(portrait, reprojected)
            grid = make_grid(
                [
                    titled(portrait, "portrait frame"),
                    titled(source_overlay, f"2D source ({source_space})"),
                    titled(uv_preview, "UV label"),
                    titled(reproject_overlay, "UV back on 3D"),
                ]
            )
            preview_path = preview_dir / f"preview_{output_index:02d}_{frame['sampleID']}.jpg"
            grid.save(preview_path, quality=94)
            saved_previews.append(str(preview_path))

    summary = {
        "session": str(session_dir),
        "uv_size": args.uv_size,
        "cull_backfaces": args.cull_backfaces,
        "front_face_winding": args.front_face_winding if args.cull_backfaces else None,
        "count_requested": args.count,
        "count_saved": len(saved_labels),
        "skipped": skipped,
        "source": "debug_lip_roi_mask" if args.debug_lip_roi_mask else ("facer" if teacher else str(mask_dir)),
        "labels": saved_labels,
        "previews": saved_previews,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
