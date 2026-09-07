#!/usr/bin/env python3
"""Evaluate a frozen checkpoint with patient-clustered confidence intervals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from respiratory_sound.data.audio import (
    FeatureConfig,
    FeatureNormalization,
    ICBHICycleDataset,
)
from respiratory_sound.metrics import patient_bootstrap_intervals
from respiratory_sound.models.network import TFDCRNet
from respiratory_sound.runtime import select_device
from respiratory_sound.training import evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/icbhi2017_cycles.csv"),
    )
    parser.add_argument("--feature-stats", type=Path)
    parser.add_argument("--split-column", required=True)
    parser.add_argument("--split-value", required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--bootstrap-iterations", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20_260_727)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--filter-column")
    parser.add_argument("--filter-value")
    parser.add_argument(
        "--acknowledge-frozen",
        action="store_true",
        help="Required for a test split to prevent premature test-set inspection.",
    )
    return parser.parse_args()


def _model_from_config(config: dict[str, object]) -> TFDCRNet:
    return TFDCRNet(
        mode=str(config["mode"]),
        input_channels=int(config["input_channels"]),
        channels=tuple(int(value) for value in config["channels"]),
        depths=tuple(int(value) for value in config["depths"]),
        temporal_dilations=tuple(
            tuple(int(value) for value in stage) for stage in config["temporal_dilations"]
        ),
        expansion_ratio=int(config["expansion_ratio"]),
        num_classes=int(config["num_classes"]),
        include_joint=bool(config.get("include_joint", False)),
        branch_types=(
            tuple(str(branch) for branch in config["branch_types"])
            if "branch_types" in config
            else None
        ),
        pooling=str(config.get("pooling", "gap")),
        lse_temperature=float(config.get("lse_temperature", 1.0)),
        gate_conditioning=str(config.get("gate_conditioning", "none")),
        gate_stages=tuple(int(stage) for stage in config.get("gate_stages", ())),
        gate_reduction=int(config.get("gate_reduction", 4)),
        gate_max_delta=float(config.get("gate_max_delta", 0.5)),
        morphology_supervision=str(config.get("morphology_supervision", "none")),
        morphology_stages=tuple(
            int(stage) for stage in config.get("morphology_stages", ())
        ),
        morphology_branch_indices=tuple(
            int(index) for index in config.get("morphology_branch_indices", (1, 2))
        ),
        morphology_source=str(config.get("morphology_source", "branch_response")),
        morphology_routing=str(config.get("morphology_routing", "mean")),
        hierarchy_fusion=str(config.get("hierarchy_fusion", "none")),
        hierarchy_max_weight=float(config.get("hierarchy_max_weight", 1.0)),
        input_normalization=str(config.get("input_normalization", "none")),
        relaxed_frequency_weight=float(
            config.get("relaxed_frequency_weight", 0.5)
        ),
        normalization_eps=float(config.get("normalization_eps", 1.0e-5)),
    )


def main() -> None:
    args = parse_args()
    split_value = args.split_value.lower()
    locked_split = split_value in {"test", "testing_1", "testing_2"} or split_value.startswith(
        "locked_"
    )
    if locked_split and not args.acknowledge_frozen:
        raise SystemExit(
            "Test evaluation is locked. Freeze the method and rerun with --acknowledge-frozen."
        )
    root = args.project_root.resolve()
    checkpoint_path = (root / args.checkpoint).resolve()
    manifest_path = (root / args.manifest).resolve()
    output_prefix = (root / args.output_prefix).resolve()
    device = select_device(args.device)
    if args.device == "mps" and device.type != "mps":
        raise SystemExit("MPS was requested but is unavailable")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint["model_config"]
    feature_config = FeatureConfig(**checkpoint["feature_config"])
    stats_relative = (
        args.feature_stats
        if args.feature_stats is not None
        else Path(checkpoint["normalization_path"])
    )
    normalization = FeatureNormalization.from_json((root / stats_relative).resolve())
    row_filters = (
        {args.filter_column: args.filter_value}
        if args.filter_column is not None
        else None
    )
    dataset = ICBHICycleDataset(
        manifest_path=manifest_path,
        project_root=root,
        split_column=args.split_column,
        split_value=args.split_value,
        feature_config=feature_config,
        normalization=normalization,
        training=False,
        num_views=1,
        label_column=str(checkpoint.get("label_column", "label_id")),
        row_filters=row_filters,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    model = _model_from_config(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    class_names = tuple(checkpoint.get("class_names", ("normal", "crackle", "wheeze", "both")))
    result = evaluate(
        model,
        loader,
        device,
        nn.CrossEntropyLoss(),
        class_names=class_names,
    )

    manifest = pd.read_csv(manifest_path, dtype={"patient_id": str})
    patient_lookup = manifest.set_index("sample_id")["patient_id"]
    patient_ids = np.asarray([patient_lookup[sample_id] for sample_id in result.sample_ids])
    intervals = patient_bootstrap_intervals(
        result.targets,
        result.predictions,
        patient_ids,
        iterations=args.bootstrap_iterations,
        seed=args.seed,
        class_names=class_names,
    )
    predictions = pd.DataFrame(
        {
            "sample_id": result.sample_ids,
            "patient_id": patient_ids,
            "target": result.targets,
            "prediction": result.predictions,
            **{
                f"probability_{class_index}": result.probabilities[:, class_index]
                for class_index in range(result.probabilities.shape[1])
            },
        }
    )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_prefix.with_suffix(".csv"), index=False)
    payload = {
        "checkpoint": str(checkpoint_path.relative_to(root)),
        "split_column": args.split_column,
        "split_value": args.split_value,
        "row_filters": row_filters,
        "sample_count": len(dataset),
        "patient_count": int(len(np.unique(patient_ids))),
        "metrics": result.metrics,
        "patient_bootstrap_intervals": intervals,
    }
    output_prefix.with_suffix(".json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
