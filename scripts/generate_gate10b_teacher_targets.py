#!/usr/bin/env python3
"""Generate deterministic train_fit targets from the frozen Gate 10A teacher."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from compute_feature_stats import feature_config_from_yaml
from torch.utils.data import DataLoader
from train_gate9a_pretrained import _evaluate, _sha256

from respiratory_sound.data.audio import ICBHICycleDataset
from respiratory_sound.models.beats_adaptation import configure_beats_adaptation
from respiratory_sound.models.pretrained_audio import load_beats_transfer
from respiratory_sound.runtime import select_device
from respiratory_sound.training import seed_everything
from respiratory_sound.training_state import load_trainable_parameter_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--runs", type=Path, nargs=6, required=True)
    parser.add_argument(
        "--modes",
        nargs=6,
        required=True,
        choices=("domain_class_event", "event_random"),
    )
    parser.add_argument("--seeds", type=int, nargs=6, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    return parser.parse_args()


def _sampling_mode(configuration: dict[str, object]) -> str:
    if "sampling_mode" in configuration:
        return str(configuration["sampling_mode"])
    experiment = configuration["experiment"]
    if not isinstance(experiment, dict):
        raise ValueError("Invalid experiment configuration")
    sampling = experiment["sampling"]
    if not isinstance(sampling, dict):
        raise ValueError("Invalid sampling configuration")
    return str(sampling["mode"])


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    manifest_path = (root / args.manifest).resolve()
    data_config_path = (root / args.data_config).resolve()
    model_config_path = (root / args.model_config).resolve()
    output_path = (root / args.output).resolve()
    audit_path = (root / args.audit_output).resolve()
    seed_everything(20_260_731)
    device = select_device(args.device)
    if args.device == "mps" and device.type != "mps":
        raise SystemExit("MPS was requested but is unavailable")

    model_config = yaml.safe_load(model_config_path.read_text(encoding="utf-8"))
    feature_config = feature_config_from_yaml(data_config_path)
    dataset = ICBHICycleDataset(
        manifest_path=manifest_path,
        project_root=root,
        split_column="protocol_role",
        split_value="train_fit",
        feature_config=feature_config,
        training=False,
        num_views=1,
        label_column="binary_label_id",
        return_waveform=True,
        waveform_only=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    checkpoint = (root / str(model_config["checkpoint"])).resolve()
    source_dir = (root / str(model_config["source_dir"])).resolve()
    model = load_beats_transfer(checkpoint, source_dir)
    adaptation = model_config["adaptation"]
    configure_beats_adaptation(
        model,
        strategy=str(adaptation["strategy"]),
        lora_last_n_layers=int(adaptation["last_n_layers"]),
        lora_rank=int(adaptation["rank"]),
        lora_alpha=float(adaptation["alpha"]),
        lora_dropout=float(adaptation["dropout"]),
    )
    model = model.to(device)

    base: pd.DataFrame | None = None
    probability_columns = []
    input_audit = []
    for index, (run_relative, expected_mode, expected_seed) in enumerate(
        zip(args.runs, args.modes, args.seeds, strict=True)
    ):
        run_dir = (root / run_relative).resolve()
        configuration = json.loads(
            (run_dir / "configuration.json").read_text(encoding="utf-8")
        )
        if int(configuration["seed"]) != expected_seed:
            raise ValueError(f"Seed mismatch in {run_dir}")
        if _sampling_mode(configuration) != expected_mode:
            raise ValueError(f"Sampling mode mismatch in {run_dir}")
        if bool(configuration["locked_test_accessed"]):
            raise ValueError(f"Locked test was accessed in {run_dir}")
        best_path = run_dir / "best.pt"
        payload = torch.load(best_path, map_location="cpu", weights_only=False)
        if payload["model_config"] != model_config:
            raise ValueError(f"Model configuration mismatch in {run_dir}")
        load_trainable_parameter_state(
            model,
            payload["trainable_state_dict"],
            list(payload["trainable_parameter_names"]),
        )
        result = _evaluate(model, loader, device)
        probability_column = f"probability_1_model_{index}"
        frame = pd.DataFrame(
            {
                "sample_id": result.sample_ids,
                "target": result.targets,
                probability_column: result.probabilities[:, 1],
            }
        )
        if base is None:
            base = frame
        else:
            base = base.merge(
                frame,
                on=["sample_id", "target"],
                how="inner",
                validate="one_to_one",
            )
        probability_columns.append(probability_column)
        input_audit.append(
            {
                "run": str(run_dir.relative_to(root)),
                "seed": expected_seed,
                "sampling_mode": expected_mode,
                "best_checkpoint_sha256": _sha256(best_path),
                "events": len(result.sample_ids),
            }
        )
        print(
            json.dumps(
                {
                    "completed_teacher": index + 1,
                    "total_teachers": len(args.runs),
                    "run": str(run_dir.relative_to(root)),
                }
            ),
            flush=True,
        )

    if base is None or len(base) != len(dataset):
        raise ValueError("Teacher predictions do not exactly cover train_fit")
    base["teacher_probability_1"] = base[probability_columns].mean(axis=1)
    output = base[["sample_id", "target", "teacher_probability_1"]].copy()
    if output["sample_id"].duplicated().any():
        raise ValueError("Teacher output contains duplicate sample IDs")
    probabilities = output["teacher_probability_1"].to_numpy(dtype=float)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    audit = {
        "gate": "10B-preparation",
        "teacher": "Gate 10A fixed six-checkpoint equal-probability ensemble",
        "role": "train_fit",
        "events": len(output),
        "unique_samples": int(output["sample_id"].nunique()),
        "class_counts": {
            str(int(label)): int(count)
            for label, count in output["target"].value_counts().sort_index().items()
        },
        "probability_summary": {
            "minimum": float(probabilities.min()),
            "maximum": float(probabilities.max()),
            "mean": float(probabilities.mean()),
            "mean_entropy": float(
                np.mean(
                    -probabilities * np.log(np.clip(probabilities, 1.0e-12, 1.0))
                    -(1.0 - probabilities)
                    * np.log(np.clip(1.0 - probabilities, 1.0e-12, 1.0))
                )
            ),
        },
        "inputs": input_audit,
        "output": str(output_path.relative_to(root)),
        "output_sha256": _sha256(output_path),
        "deterministic_preprocessing": {
            "training": False,
            "num_views": 1,
            "augmentation": False,
            "duration_placement": "center",
            "exact_token_mask": True
        },
        "calibration_accessed": False,
        "locked_tests_accessed": False,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
