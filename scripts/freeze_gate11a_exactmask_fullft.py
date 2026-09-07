#!/usr/bin/env python3
"""Seal the Gate 11A protocol before any formal validation inference."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import torch
import yaml

from respiratory_sound.gate11a import (
    ALLOWED_SEEDS,
    SINGLE_SEED_THRESHOLDS,
    THREE_SEED_THRESHOLDS,
    TRAIN_ROLE,
    VALIDATION_ROLE,
    assert_gate11a_protocol,
    gate11a_hashes,
)

FORMAL_RESULT_NAMES = {
    "history.csv",
    "summary.json",
    "decision.json",
    "best.pt",
    "resume_state.pt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/cross_domain_binary.csv"),
    )
    parser.add_argument(
        "--data-config",
        type=Path,
        default=Path("configs/data/gate9a_beats.yaml"),
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("configs/model/gate9a_beats_as2m_cpt2.yaml"),
    )
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=Path(
            "configs/experiment/"
            "gate11a_beats_exactmask_fullft_seed20260729.yaml"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/gate11a_exactmask_fullft_freeze.json"),
    )
    return parser.parse_args()


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _assert_no_formal_results(root: Path) -> None:
    violations: list[str] = []
    for run_dir in sorted((root / "runs").glob("gate11a_exactmask_fullft_seed*")):
        for path in run_dir.iterdir():
            if (
                path.name in FORMAL_RESULT_NAMES
                or path.name.startswith("best_validation_predictions_")
            ):
                violations.append(str(path.relative_to(root)))
    if violations:
        raise RuntimeError(
            "Gate 11A validation artifacts already exist before freeze: "
            f"{violations}"
        )


def _assert_frozen_method(
    data_config: dict[str, Any],
    model_config: dict[str, Any],
    experiment: dict[str, Any],
) -> None:
    expected_experiment_values = {
        "gate": "11A",
        "formal_status": "exact_token_mask_full_finetuning_confirmation",
        "seed": ALLOWED_SEEDS[0],
        "allowed_seeds": list(ALLOWED_SEEDS),
        "train_value": TRAIN_ROLE,
        "validation_value": VALIDATION_ROLE,
        "label_column": "binary_label_id",
        "batch_size": 8,
        "samples_per_epoch": 7_515,
        "num_workers": 4,
        "sampling.mode": "domain_class_event",
        "augmentation.gain_min": 0.8,
        "augmentation.gain_max": 1.2,
        "augmentation.noise_probability": 0.5,
        "augmentation.noise_snr_min_db": 12.0,
        "augmentation.noise_snr_max_db": 30.0,
        "augmentation.shift_probability": 0.5,
        "augmentation.max_shift_fraction": 0.1,
        "optimizer.name": "adamw",
        "optimizer.backbone_learning_rate": 1.0e-5,
        "optimizer.head_learning_rate": 1.0e-4,
        "optimizer.weight_decay": 0.01,
        "schedule.warmup_epochs": 3,
        "schedule.max_epochs": 20,
        "schedule.early_stopping_patience": 5,
        "consistency.enabled": True,
        "consistency.name": "jensen_shannon",
        "consistency.max_weight": 0.4,
        "consistency.warmup_epochs": 20,
        "checkpoint_selection.primary": "minimum_domain_average_score",
        "checkpoint_selection.tie_breaker": "mean_domain_average_score",
        "checkpoint_selection.tolerance": 1.0e-12,
        "classification.threshold": 0.5,
        "data_access.calibration_allowed": False,
        "data_access.locked_allowed": False,
    }

    def nested(payload: dict[str, Any], name: str) -> Any:
        value: Any = payload
        for part in name.split("."):
            value = value[part]
        return value

    mismatches = {
        name: {"found": nested(experiment, name), "expected": expected}
        for name, expected in expected_experiment_values.items()
        if nested(experiment, name) != expected
    }
    if mismatches:
        raise ValueError(f"Gate 11A experiment differs from frozen method: {mismatches}")
    if (
        str(model_config["type"]) != "beats"
        or int(model_config["num_classes"]) != 2
        or str(model_config["source_dir"]) != "third_party/beats"
        or str(model_config["checkpoint"])
        != (
            "checkpoints/pretrained/beats_as2m_cpt2/"
            "BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"
        )
    ):
        raise ValueError("Gate 11A requires the frozen binary BEATs model")
    if (
        int(data_config["sample_rate"]) != 16_000
        or float(data_config["clip_seconds"]) != 8.0
        or str(data_config["duration_fit"]) != "zero_pad"
        or bool(data_config["random_pad_position"]) is not True
    ):
        raise ValueError("Gate 11A requires the exact zero-padded waveform protocol")


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    manifest_path = _resolve(root, args.manifest).resolve()
    data_config_path = _resolve(root, args.data_config).resolve()
    model_config_path = _resolve(root, args.model_config).resolve()
    experiment_config_path = _resolve(root, args.experiment_config).resolve()
    output_path = _resolve(root, args.output).resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite frozen artifact: {output_path}")

    _assert_no_formal_results(root)
    manifest = pd.read_csv(manifest_path, dtype={"patient_id": str})
    protocol_audit = assert_gate11a_protocol(
        manifest,
        train_role=TRAIN_ROLE,
        validation_role=VALIDATION_ROLE,
    )
    data_config = yaml.safe_load(data_config_path.read_text(encoding="utf-8"))
    model_config = yaml.safe_load(model_config_path.read_text(encoding="utf-8"))
    experiment = yaml.safe_load(
        experiment_config_path.read_text(encoding="utf-8")
    )
    _assert_frozen_method(data_config, model_config, experiment)

    checkpoint_path = (root / str(model_config["checkpoint"])).resolve()
    beats_source_dir = (root / str(model_config["source_dir"])).resolve()
    hashes = gate11a_hashes(
        root,
        manifest_path=manifest_path,
        data_config_path=data_config_path,
        model_config_path=model_config_path,
        experiment_config_path=experiment_config_path,
        checkpoint_path=checkpoint_path,
        beats_source_dir=beats_source_dir,
    )
    free_space_bytes = shutil.disk_usage(root).free
    if free_space_bytes < 5 * 1024**3:
        raise RuntimeError("Gate 11A requires at least 5 GiB free disk space")

    payload = {
        "schema_version": 1,
        "gate": "11A",
        "status": "sealed_before_formal_validation",
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "purpose": (
            "Exact-token-mask BEATs full-finetuning three-seed confirmation"
        ),
        "user_confirmation": {
            "confirmed_in_codex_task": True,
            "confirmation_date": "2026-07-31",
        },
        "development_data_boundary": {
            "train_role": TRAIN_ROLE,
            "validation_role": VALIDATION_ROLE,
            "calibration_allowed": False,
            "locked_allowed": False,
            "protocol_audit": protocol_audit,
        },
        "fixed_method": {
            "model": model_config,
            "data": data_config,
            "experiment": experiment,
            "full_backbone_and_binary_head_trainable": True,
            "exact_token_mask_required": True,
            "fixed_classification_threshold": 0.5,
            "checkpoint_selection": {
                "primary": "minimum_domain_average_score",
                "tie_breaker": "mean_domain_average_score",
                "tolerance": 1.0e-12,
            },
        },
        "allowed_seeds": list(ALLOWED_SEEDS),
        "single_seed_thresholds": SINGLE_SEED_THRESHOLDS,
        "three_seed_thresholds": THREE_SEED_THRESHOLDS,
        "stop_rules": {
            "first_seed_failure_without_reproducible_implementation_error": (
                "stop_route"
            ),
            "nonfinite_loss_gradient_or_output": "abort_and_audit",
            "data_boundary_violation": "abort_immediately",
            "exact_mask_assertion_failure": "abort_immediately",
            "post_hoc_threshold_sampler_augmentation_or_seed_change": "forbidden",
            "implementation_error": (
                "record_amendment_then_rerun_unchanged_scientific_protocol"
            ),
        },
        "three_seed_inference_rule": {
            "gate_decision_uses_individual_seeds_and_arithmetic_means": True,
            "best_seed_selection_forbidden": True,
            "post_gate_primary_model": (
                "equal_positive-class-probability ensemble of three checkpoints"
            ),
            "ensemble_weights": [1 / 3, 1 / 3, 1 / 3],
        },
        "matched_reference": {
            "method": (
                "exact-mask rank-8 Q/V LoRA, blocks 8-11, "
                "domain_class_event, same three seeds"
            ),
            "artifact": "artifacts/gate9d_multiseed_final.json",
            "purpose": (
                "paired patient-cluster bootstrap; does not replace absolute gate"
            ),
        },
        "bootstrap": {
            "iterations": 4_000,
            "seed": 20_260_729,
            "confidence": 0.95,
            "interval": "percentile",
            "cluster": "patient_id within database",
            "paired_reference_resampling": True,
        },
        "exact_token_mask_mapping": {
            "sample_frame_length": 400,
            "sample_frame_shift": 160,
            "fbank_time_frames_before_tail_padding": 798,
            "fbank_time_frames_after_tail_padding": 800,
            "patch_size": 16,
            "frequency_patches": 8,
            "tokens": 400,
            "supports_arbitrary_valid_audio_position": True,
            "nonempty_waveform_must_have_valid_tokens": True,
        },
        "hashes": hashes,
        "environment": {
            "python": ".".join(map(str, __import__("sys").version_info[:3])),
            "torch": torch.__version__,
            "mps_built": torch.backends.mps.is_built(),
            "mps_available_at_freeze": torch.backends.mps.is_available(),
            "free_space_bytes": free_space_bytes,
            "project_is_git_repository": (root / ".git").is_dir(),
        },
        "access_history_at_freeze": {
            "gate11a_validation_results_exist": False,
            "calibration_accessed_by_gate11a": False,
            "locked_tests_accessed_by_gate11a": False,
            "locked_metadata_and_label_counts_previously_audited": True,
            "calibration_patients_accessed_by_abandoned_earlier_routes": True,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(output_path)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
