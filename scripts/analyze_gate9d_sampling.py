#!/usr/bin/env python3
"""Analyze the frozen Gate 9D single-seed matched sampling ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from respiratory_sound.gate9d import sampling_ablation_decision

DOMAINS = ("icbhi2017", "sprsound2022")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--balanced-run", type=Path, required=True)
    parser.add_argument("--event-random-run", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=4_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_729)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _average_score(targets: np.ndarray, predictions: np.ndarray) -> float:
    normal = targets == 0
    adventitious = targets == 1
    if not normal.any() or not adventitious.any():
        raise ValueError("Both classes are required for Average Score")
    specificity = float((predictions[normal] == 0).mean())
    sensitivity = float((predictions[adventitious] == 1).mean())
    return (specificity + sensitivity) / 2.0


def _load_run(
    run_dir: Path,
    *,
    expected_mode: str,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame], dict[str, Any]]:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    configuration = json.loads(
        (run_dir / "configuration.json").read_text(encoding="utf-8")
    )
    mode = configuration.get(
        "sampling_mode",
        configuration["experiment"]["sampling"]["mode"],
    )
    if str(mode) != expected_mode:
        raise ValueError(f"Unexpected sampling mode in {run_dir}: {mode}")
    if bool(configuration["locked_test_accessed"]):
        raise ValueError(f"Locked test was accessed in {run_dir}")
    predictions = {
        domain: pd.read_csv(
            run_dir / f"best_validation_predictions_{domain}.csv"
        )
        for domain in DOMAINS
    }
    return summary, predictions, configuration


def _paired_frame(
    candidate: pd.DataFrame,
    control: pd.DataFrame,
    patient_lookup: pd.DataFrame,
) -> pd.DataFrame:
    required = {"sample_id", "target", "prediction", "probability_1"}
    for frame in (candidate, control):
        if not required.issubset(frame):
            raise ValueError("Prediction file is missing required columns")
        expected_predictions = (frame["probability_1"].to_numpy() >= 0.5).astype(int)
        if not np.array_equal(expected_predictions, frame["prediction"].to_numpy()):
            raise ValueError("Stored predictions do not use the fixed 0.5 threshold")
    merged = candidate[list(required)].merge(
        control[list(required)],
        on=["sample_id", "target"],
        suffixes=("_balanced", "_event_random"),
        validate="one_to_one",
    )
    if len(merged) != len(candidate) or len(merged) != len(control):
        raise ValueError("Balanced and event-random predictions are not fully paired")
    merged = merged.merge(
        patient_lookup,
        on="sample_id",
        how="left",
        validate="many_to_one",
    )
    if merged["patient_id"].isna().any():
        raise ValueError("Patient lookup is incomplete")
    return merged


def _bootstrap_gain(
    paired: pd.DataFrame,
    *,
    iterations: int,
    random_generator: np.random.Generator,
) -> np.ndarray:
    patient_values = paired["patient_id"].astype(str).to_numpy()
    patients = np.unique(patient_values)
    rows_by_patient = {
        patient: np.flatnonzero(patient_values == patient)
        for patient in patients
    }
    targets = paired["target"].to_numpy(dtype=int)
    balanced = paired["prediction_balanced"].to_numpy(dtype=int)
    event_random = paired["prediction_event_random"].to_numpy(dtype=int)
    gains = np.empty(iterations, dtype=np.float64)
    completed = 0
    while completed < iterations:
        sampled_patients = random_generator.choice(
            patients,
            size=len(patients),
            replace=True,
        )
        indices = np.concatenate(
            [rows_by_patient[str(patient)] for patient in sampled_patients]
        )
        sampled_targets = targets[indices]
        if not (
            np.any(sampled_targets == 0)
            and np.any(sampled_targets == 1)
        ):
            continue
        gains[completed] = (
            _average_score(sampled_targets, balanced[indices])
            - _average_score(sampled_targets, event_random[indices])
        )
        completed += 1
    return gains


def _bootstrap_summary(gains: np.ndarray) -> dict[str, Any]:
    return {
        "mean_gain": float(gains.mean()),
        "confidence_interval_95": [
            float(np.quantile(gains, 0.025)),
            float(np.quantile(gains, 0.975)),
        ],
        "probability_gain_above_zero": float(np.mean(gains > 0.0)),
    }


def main() -> None:
    args = parse_args()
    if args.bootstrap_iterations < 100:
        raise ValueError("At least 100 bootstrap iterations are required")
    balanced_summary, balanced_predictions, balanced_config = _load_run(
        args.balanced_run,
        expected_mode="domain_class_event",
    )
    random_summary, random_predictions, random_config = _load_run(
        args.event_random_run,
        expected_mode="event_random",
    )
    if int(balanced_config["seed"]) != int(random_config["seed"]):
        raise ValueError("Gate 9D runs do not use the same seed")
    manifest = pd.read_csv(args.manifest, dtype={"patient_id": str})
    validation = manifest.loc[
        manifest["protocol_role"].eq("validation_select"),
        ["sample_id", "patient_id", "dataset"],
    ]
    random_generator = np.random.default_rng(args.bootstrap_seed)
    bootstrap = {
        "per_domain": {},
        "iterations": args.bootstrap_iterations,
        "seed": args.bootstrap_seed,
    }
    gain_samples = []
    pairing_audit = {}
    for domain in DOMAINS:
        lookup = validation.loc[
            validation["dataset"].astype(str).eq(domain),
            ["sample_id", "patient_id"],
        ]
        paired = _paired_frame(
            balanced_predictions[domain],
            random_predictions[domain],
            lookup,
        )
        gains = _bootstrap_gain(
            paired,
            iterations=args.bootstrap_iterations,
            random_generator=random_generator,
        )
        gain_samples.append(gains)
        bootstrap["per_domain"][domain] = _bootstrap_summary(gains)
        pairing_audit[domain] = {
            "events": len(paired),
            "patients": int(paired["patient_id"].nunique()),
            "targets_identical": True,
            "fixed_threshold_predictions_verified": True,
        }
    mean_gains = np.mean(np.stack(gain_samples), axis=0)
    bootstrap["mean_domain"] = _bootstrap_summary(mean_gains)
    decision = sampling_ablation_decision(
        balanced_summary,
        random_summary,
        bootstrap,
    )
    result = {
        "gate": "9D",
        "status": (
            "single_seed_sampling_ablation_passed"
            if decision["passed"]
            else "single_seed_sampling_ablation_failed"
        ),
        "seed": int(balanced_config["seed"]),
        "development_only": True,
        "calibration_accessed": False,
        "locked_tests_accessed": False,
        "balanced_run": str(args.balanced_run),
        "event_random_run": str(args.event_random_run),
        "pairing_audit": pairing_audit,
        "bootstrap": bootstrap,
        "decision": decision,
        "freeze": "artifacts/gate9d_sampling_ablation_freeze.json",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
