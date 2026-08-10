from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from lip_curve_model import build_lip_curve_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="arface/Models/LipCurveStudent384.mlpackage")
    parser.add_argument("--image-width", type=int, default=None)
    parser.add_argument("--image-height", type=int, default=None)
    args = parser.parse_args()

    import coremltools as ct

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_width = int(checkpoint.get("model_width", checkpoint.get("args", {}).get("model_width", 32)))
    points_per_curve = int(checkpoint.get("points_per_curve", checkpoint.get("args", {}).get("points_per_curve", 24)))
    curve_count = int(checkpoint.get("curve_count", 4))
    image_width = int(args.image_width or checkpoint.get("input_width", checkpoint.get("args", {}).get("image_width", 384)))
    image_height = int(args.image_height or checkpoint.get("input_height", checkpoint.get("args", {}).get("image_height", 256)))

    model = build_lip_curve_model(
        curve_count=curve_count,
        points_per_curve=points_per_curve,
        width=model_width,
        input_channels=3,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    example = torch.randn(1, 3, image_height, image_width)
    traced = torch.jit.trace(model, example)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(
                name="image",
                shape=example.shape,
                dtype=np.float32,
            )
        ],
        outputs=[ct.TensorType(name="curves", dtype=np.float32)],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
    )
    mlmodel.short_description = "Lip curve control-point model for smooth lipstick boundaries."
    mlmodel.input_description["image"] = f"Normalized RGB lip ROI tensor, shape 1x3x{image_height}x{image_width}."
    mlmodel.output_description["curves"] = (
        f"Normalized lip curves, shape 1x{curve_count}x{points_per_curve}x2. "
        "Curve order: upper_outer, upper_inner, lower_inner, lower_outer."
    )
    mlmodel.user_defined_metadata["task"] = "lip_curve_student"
    mlmodel.user_defined_metadata["curve_count"] = str(curve_count)
    mlmodel.user_defined_metadata["points_per_curve"] = str(points_per_curve)
    mlmodel.user_defined_metadata["input_width"] = str(image_width)
    mlmodel.user_defined_metadata["input_height"] = str(image_height)
    mlmodel.save(str(output_path))
    print(f"Saved Core ML model: {output_path}")
    print(f"Input: 1x3x{image_height}x{image_width}")
    print(f"Output curves: 1x{curve_count}x{points_per_curve}x2")


if __name__ == "__main__":
    main()
