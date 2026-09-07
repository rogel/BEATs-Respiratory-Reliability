#!/usr/bin/env python3
"""Summarize HM-RDCR scale routing on a frozen development split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from evaluate_checkpoint import _model_from_config
from torch.utils.data import DataLoader

from respiratory_sound.data.audio import (
    FeatureConfig,
    FeatureNormalization,
    ICBHICycleDataset,
)
from respiratory_sound.data.sprsound import morphology_target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/sprsound2022_events.csv"),
    )
    parser.add_argument("--split-column", default="development_split")
    parser.add_argument("--split-value", default="validation")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def route_statistics(values: np.ndarray) -> dict[str, object]:
    return {
        "mean": values.mean(axis=0).tolist(),
        "standard_deviation": values.std(axis=0).tolist(),
        "mean_entropy": float(
            (-(values * np.log(values.clip(min=1.0e-8))).sum(axis=1)).mean()
        ),
    }


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    checkpoint_path = (root / args.checkpoint).resolve()
    manifest_path = (root / args.manifest).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = _model_from_config(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    if model.morphology_routing != "softmax":
        raise SystemExit("Checkpoint does not use softmax morphology routing")

    normalization_path = (root / checkpoint["normalization_path"]).resolve()
    dataset = ICBHICycleDataset(
        manifest_path=manifest_path,
        project_root=root,
        split_column=args.split_column,
        split_value=args.split_value,
        feature_config=FeatureConfig(**checkpoint["feature_config"]),
        normalization=FeatureNormalization.from_json(normalization_path),
        training=False,
        num_views=1,
        label_column=str(checkpoint["label_column"]),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    manifest = pd.read_csv(manifest_path)
    label_lookup = manifest.set_index("sample_id")["label_name"].to_dict()

    routes: dict[str, list[np.ndarray]] = {"transient": [], "continuous": []}
    coarse_targets: list[np.ndarray] = []
    morphology_targets: list[np.ndarray] = []
    with torch.inference_mode():
        for views, frame_masks, targets, sample_ids in loader:
            model(
                views[:, 0],
                frame_masks=frame_masks[:, 0],
                return_aux=True,
            )
            if model.last_morphology_routing is None:
                raise RuntimeError("Model did not expose morphology routing weights")
            for attribute in routes:
                routes[attribute].append(
                    model.last_morphology_routing[attribute].cpu().numpy()
                )
            coarse_targets.append(targets.numpy())
            morphology_targets.append(
                np.asarray(
                    [
                        morphology_target(str(label_lookup[sample_id]))
                        for sample_id in sample_ids
                    ]
                )
            )

    route_arrays = {
        attribute: np.concatenate(values, axis=0)
        for attribute, values in routes.items()
    }
    coarse_array = np.concatenate(coarse_targets)
    morphology_array = np.concatenate(morphology_targets)
    payload: dict[str, object] = {
        "checkpoint": str(checkpoint_path.relative_to(root)),
        "sample_count": len(dataset),
        "stages_zero_based": list(model.morphology_stages),
        "hierarchy_weight": float(model.hierarchy_weight().detach()),
        "routing": {},
    }
    routing_payload: dict[str, object] = {}
    for attribute_index, attribute in enumerate(("transient", "continuous")):
        values = route_arrays[attribute]
        routing_payload[attribute] = {
            "overall": route_statistics(values),
            "normal": route_statistics(values[coarse_array == 0]),
            "adventitious": route_statistics(values[coarse_array == 1]),
            "attribute_negative": route_statistics(
                values[morphology_array[:, attribute_index] == 0]
            ),
            "attribute_positive": route_statistics(
                values[morphology_array[:, attribute_index] == 1]
            ),
        }
    payload["routing"] = routing_payload
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output is not None:
        output_path = (root / args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
