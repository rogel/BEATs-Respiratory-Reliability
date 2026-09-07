#!/usr/bin/env python3
"""Calibrate Gate 9B drift loss on training batches without validation access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import yaml
from compute_feature_stats import feature_config_from_yaml
from torch import nn
from torch.utils.data import DataLoader
from train_gate9a_pretrained import _assert_development_protocol, _augmentation
from train_gate9b_adaptation import (
    _forward_batch,
    _sha256,
    _trainable_parameter_groups,
)

from respiratory_sound.data.audio import ICBHICycleDataset
from respiratory_sound.data.sampling import DomainClassEventBatchSampler
from respiratory_sound.models.beats_adaptation import (
    configure_beats_adaptation,
    projection_drift_regularization,
)
from respiratory_sound.models.pretrained_audio import load_beats_transfer
from respiratory_sound.runtime import select_device
from respiratory_sound.training import (
    jensen_shannon_consistency,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batches < 2:
        raise ValueError("Use at least two batches so post-update drift is observable")
    root = args.project_root.resolve()
    manifest_path = (root / args.manifest).resolve()
    data_config_path = (root / args.data_config).resolve()
    model_config = yaml.safe_load(
        (root / args.model_config).read_text(encoding="utf-8")
    )
    experiment = yaml.safe_load(
        (root / args.experiment_config).read_text(encoding="utf-8")
    )
    seed = int(experiment["seed"])
    seed_everything(seed)
    device = select_device(args.device)
    if args.device == "mps" and device.type != "mps":
        raise SystemExit("MPS was requested but is unavailable")

    manifest = pd.read_csv(manifest_path, dtype={"patient_id": str})
    train_role = str(experiment.get("train_value", "train_fit"))
    validation_role = str(experiment.get("validation_value", "validation_select"))
    _assert_development_protocol(manifest, train_role, validation_role)
    feature_config = feature_config_from_yaml(data_config_path)
    consistency = experiment["consistency"]
    train_dataset = ICBHICycleDataset(
        manifest_path=manifest_path,
        project_root=root,
        split_column="protocol_role",
        split_value=train_role,
        feature_config=feature_config,
        training=True,
        num_views=2 if bool(consistency["enabled"]) else 1,
        augmentation=_augmentation(experiment),
        label_column=str(experiment.get("label_column", "binary_label_id")),
        return_waveform=True,
        waveform_only=True,
    )
    batch_size = int(experiment["batch_size"])
    sampler = DomainClassEventBatchSampler(
        train_dataset.rows,
        batch_size=batch_size,
        samples_per_epoch=batch_size * args.batches,
        seed=seed,
        class_column=str(experiment.get("label_column", "binary_label_id")),
    )
    loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        num_workers=0,
        pin_memory=False,
    )

    checkpoint = (root / str(model_config["checkpoint"])).resolve()
    source_dir = (root / str(model_config["source_dir"])).resolve()
    model = load_beats_transfer(checkpoint, source_dir)
    adaptation = model_config["adaptation"]
    audit = configure_beats_adaptation(
        model,
        strategy=str(adaptation["strategy"]),
        lora_last_n_layers=int(adaptation.get("last_n_layers", 4)),
        lora_rank=int(adaptation.get("rank", 8)),
        lora_alpha=float(adaptation.get("alpha", 16.0)),
        lora_dropout=float(adaptation.get("dropout", 0.05)),
    )
    adaptation_parameters, classifier_parameters = _trainable_parameter_groups(model)
    optimizer_config = experiment["optimizer"]
    parameter_groups: list[dict[str, object]] = []
    if adaptation_parameters:
        parameter_groups.append(
            {
                "params": adaptation_parameters,
                "lr": float(optimizer_config["adaptation_learning_rate"]),
            }
        )
    parameter_groups.append(
        {
            "params": classifier_parameters,
            "lr": float(optimizer_config["head_learning_rate"]),
        }
    )
    optimizer = torch.optim.AdamW(
        parameter_groups,
        weight_decay=float(optimizer_config["weight_decay"]),
    )
    model = model.to(device)
    model.train()

    rows: list[dict[str, float | int]] = []
    for step, (waveforms, sample_masks, targets, _) in enumerate(loader, start=1):
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = _forward_batch(model, waveforms, sample_masks, device)
        repeated_targets = targets[:, None].expand(-1, logits.shape[1]).reshape(-1)
        classification_loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            repeated_targets,
        )
        consistency_loss = jensen_shannon_consistency(logits)
        drift = projection_drift_regularization(model)
        loss = (
            classification_loss
            + float(consistency["max_weight"]) * consistency_loss
        )
        loss.backward()
        nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            max_norm=5.0,
        )
        optimizer.step()
        row: dict[str, float | int] = {
            "step": step,
            "classification_loss": float(classification_loss.detach().cpu()),
            "consistency_loss": float(consistency_loss.detach().cpu()),
            "relative_rms_projection_drift": float(drift.detach().cpu()),
        }
        if device.type == "mps":
            row["mps_allocated_gib"] = (
                float(torch.mps.current_allocated_memory()) / 1024**3
            )
            row["mps_driver_allocated_gib"] = (
                float(torch.mps.driver_allocated_memory()) / 1024**3
            )
        rows.append(row)
        print(json.dumps(row), flush=True)

    nonzero_drifts = [
        float(row["relative_rms_projection_drift"])
        for row in rows
        if float(row["relative_rms_projection_drift"]) > 0
    ]
    result = {
        "gate": "9B",
        "purpose": "training_only_loss_scale_and_resource_calibration",
        "validation_accessed": False,
        "locked_test_accessed": False,
        "candidate": str(model_config["candidate"]),
        "seed": seed,
        "batches": args.batches,
        "formal_status": "training_only_exact_token_mask_calibration",
        "code_audit": {
            "beats_backbone_source_sha256": _sha256(source_dir / "backbone.py"),
            "beats_transfer_wrapper_sha256": _sha256(
                root / "src/respiratory_sound/models/pretrained_audio.py"
            ),
            "exact_token_mask_amendment": (
                "artifacts/gate9b_exact_token_mask_amendment.json"
            ),
        },
        "adaptation_audit": audit.to_dict(),
        "rows": rows,
        "maximum_observed_projection_drift": max(nonzero_drifts, default=0.0),
        "median_nonzero_projection_drift": (
            float(torch.tensor(nonzero_drifts).median())
            if nonzero_drifts
            else 0.0
        ),
    }
    if args.output is not None:
        output_path = (root / args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
