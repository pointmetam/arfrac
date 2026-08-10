from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from model import build_model


class SigmoidOutputWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(image))


class LipSDFOutputWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        logits = self.model(image)
        alpha = torch.sigmoid(logits[:, 0:1])
        sdf = torch.tanh(logits[:, 1:2])
        edge = torch.sigmoid(logits[:, 2:3])
        return torch.cat([alpha, sdf, edge], dim=1)


def normalize_task(task: str) -> str:
    normalized = task.strip().lower().replace("_", "-")
    if normalized in {"seg", "segmentation", "logits"}:
        return "segmentation"
    if normalized in {"lip-alpha", "lip-alpha-student", "alpha", "sdf"}:
        return "lip-alpha"
    if normalized in {"lip-sdf", "lip-sdf-student", "sdf-alpha", "alpha-sdf", "boundary-sdf"}:
        return "lip-sdf"
    raise ValueError(f"Unsupported export task: {task}")


def build_export_model(model: nn.Module, task: str) -> nn.Module:
    task = normalize_task(task)
    if task == "lip-alpha":
        return SigmoidOutputWrapper(model)
    if task == "lip-sdf":
        return LipSDFOutputWrapper(model)
    return model


def output_name_for_task(task: str) -> str:
    task = normalize_task(task)
    if task == "lip-sdf":
        return "lip_sdf"
    return "alpha" if task == "lip-alpha" else "logits"


def short_description_for_task(task: str) -> str:
    task = normalize_task(task)
    if task == "lip-sdf":
        return "Lip ROI alpha/SDF/edge model for cleaner lipstick boundaries."
    if task == "lip-alpha":
        return "Lightweight lip ROI soft alpha model for lipstick boundaries."
    return "Lightweight face parsing model for AR makeup masks."


def output_description_for_task(task: str, num_classes: int) -> str:
    task = normalize_task(task)
    if task == "lip-sdf":
        return "Lip outputs after activation: alpha, signed distance, edge heatmap; shape 1x3xHxW."
    if task == "lip-alpha":
        return "Lipstick alpha probabilities after sigmoid, shape 1x1xHxW."
    return f"Segmentation logits, shape 1x{num_classes}xHxW."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="arface/Models/MakeupParsingSmall.mlpackage")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--image-width", type=int, default=None)
    parser.add_argument("--image-height", type=int, default=None)
    parser.add_argument("--task", choices=["segmentation", "lip-alpha", "lip-sdf"], default=None)
    args = parser.parse_args()

    import coremltools as ct

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    task = normalize_task(args.task or checkpoint.get("task", "segmentation"))
    model = build_model(
        num_classes=checkpoint.get("num_classes", 9),
        width=int(checkpoint.get("model_width", 32)),
        input_channels=int(checkpoint.get("input_channels", 3)),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    export_model = build_export_model(model, task)
    export_model.eval()

    image_width = args.image_width or int(checkpoint.get("input_width", args.image_size))
    image_height = args.image_height or int(checkpoint.get("input_height", args.image_size))
    num_classes = int(checkpoint.get("num_classes", 9))
    input_channels = int(checkpoint.get("input_channels", 3))

    example = torch.randn(1, input_channels, image_height, image_width)
    traced = torch.jit.trace(export_model, example)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_name = output_name_for_task(task)

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
        outputs=[ct.TensorType(name=output_name, dtype=np.float32)],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
    )
    spec = mlmodel.get_spec()
    input_names = [item.name for item in spec.description.input]
    output_names = [item.name for item in spec.description.output]

    mlmodel.short_description = short_description_for_task(task)
    if input_names:
        if input_channels == 4:
            mlmodel.input_description[input_names[0]] = (
                "Normalized RGB face ROI tensor plus canonical lip template prior, shape 1x4xHxW."
            )
        else:
            mlmodel.input_description[input_names[0]] = "Normalized RGB face ROI tensor, shape 1x3xHxW."
    if output_names:
        mlmodel.output_description[output_names[0]] = output_description_for_task(task, num_classes)
    mlmodel.save(str(output_path))
    print(f"Saved Core ML model: {output_path}")


if __name__ == "__main__":
    main()
