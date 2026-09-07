#!/usr/bin/env python3
"""Measure whether anchored gates actually vary across validation events."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from train_icbhi import _model_from_config

from respiratory_sound.data.audio import (
    FeatureConfig,
    FeatureNormalization,
    ICBHICycleDataset,
)
from respiratory_sound.models import AnchoredChannelGate
from respiratory_sound.runtime import select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split-column", default="development_split")
    parser.add_argument("--split-value", default="validation")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    device = select_device(args.device)
    if args.device == "mps" and device.type != "mps":
        raise SystemExit("MPS was requested but is unavailable")
    checkpoint = torch.load(
        (root / args.checkpoint).resolve(),
        map_location=device,
        weights_only=False,
    )
    model = _model_from_config(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    feature_config = FeatureConfig(**checkpoint["feature_config"])
    normalization = FeatureNormalization.from_json(
        (root / checkpoint["normalization_path"]).resolve()
    )
    dataset = ICBHICycleDataset(
        manifest_path=(root / args.manifest).resolve(),
        project_root=root,
        split_column=args.split_column,
        split_value=args.split_value,
        feature_config=feature_config,
        normalization=normalization,
        training=False,
        num_views=1,
        label_column=str(checkpoint["label_column"]),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    gates = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, AnchoredChannelGate)
    }
    collected = {name: [] for name in gates}
    targets: list[torch.Tensor] = []
    for views, frame_masks, batch_targets, _ in loader:
        model(
            views[:, 0].to(device),
            frame_masks=frame_masks[:, 0].to(device),
        )
        targets.append(batch_targets)
        for name, gate in gates.items():
            if gate.last_scale is None:
                raise RuntimeError(f"Gate {name} did not record a scale")
            collected[name].append(gate.last_scale.cpu())

    target_array = torch.cat(targets).numpy()
    gate_payload: dict[str, object] = {}
    all_sample_descriptors: list[np.ndarray] = []
    for name, batches in collected.items():
        scales = torch.cat(batches).numpy()
        sample_means = scales.mean(axis=1)
        all_sample_descriptors.append(scales)
        class_means = {
            str(int(label)): float(sample_means[target_array == label].mean())
            for label in np.unique(target_array)
        }
        class_channel_means = {
            int(label): scales[target_array == label].mean(axis=0)
            for label in np.unique(target_array)
        }
        class_channel_difference = class_channel_means.get(
            1,
            np.full(scales.shape[1], np.nan),
        ) - class_channel_means.get(
            0,
            np.full(scales.shape[1], np.nan),
        )
        gate_payload[name] = {
            "conditioning": gates[name].conditioning,
            "channels": int(scales.shape[1]),
            "scale_mean": float(scales.mean()),
            "scale_standard_deviation": float(scales.std(ddof=1)),
            "scale_minimum": float(scales.min()),
            "scale_maximum": float(scales.max()),
            "between_sample_mean_scale_standard_deviation": float(
                sample_means.std(ddof=1)
            ),
            "mean_within_sample_channel_standard_deviation": float(
                scales.std(axis=1, ddof=1).mean()
            ),
            "class_mean_scales": class_means,
            "class_1_minus_class_0_mean_scale": (
                class_means.get("1", float("nan"))
                - class_means.get("0", float("nan"))
            ),
            "mean_absolute_class_channel_scale_difference": float(
                np.abs(class_channel_difference).mean()
            ),
            "maximum_absolute_class_channel_scale_difference": float(
                np.abs(class_channel_difference).max()
            ),
        }

    concatenated = np.concatenate(all_sample_descriptors, axis=1)
    payload = {
        "checkpoint": str(args.checkpoint),
        "sample_count": len(dataset),
        "gate_count": len(gates),
        "gate_conditioning": sorted({gate.conditioning for gate in gates.values()}),
        "concatenated_scale_standard_deviation": float(
            concatenated.std(ddof=1)
        ),
        "mean_between_sample_channelwise_standard_deviation": float(
            concatenated.std(axis=0, ddof=1).mean()
        ),
        "gates": gate_payload,
    }
    output = (root / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
