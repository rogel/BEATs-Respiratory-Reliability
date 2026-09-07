#!/usr/bin/env python3
"""Evaluate the frozen, untuned six-checkpoint Gate 10A ensemble."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from respiratory_sound.gate10a import sampling_diverse_ensemble_decision

DOMAINS = ("icbhi2017", "sprsound2022")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--balanced-runs", type=Path, nargs=3, required=True)
    parser.add_argument("--event-random-runs", type=Path, nargs=3, required=True)
    parser.add_argument("--seeds", type=int, nargs=3, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=4_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_731)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _sampling_mode(configuration: dict[str, Any]) -> str:
    if "sampling_mode" in configuration:
        return str(configuration["sampling_mode"])
    return str(configuration["experiment"]["sampling"]["mode"])


def _load_predictions(
    run_dir: Path,
    *,
    expected_seed: int,
    expected_mode: str,
) -> dict[str, pd.DataFrame]:
    configuration = json.loads(
        (run_dir / "configuration.json").read_text(encoding="utf-8")
    )
    if int(configuration["seed"]) != expected_seed:
        raise ValueError(f"Seed mismatch in {run_dir}")
    if _sampling_mode(configuration) != expected_mode:
        raise ValueError(f"Sampling-mode mismatch in {run_dir}")
    if bool(configuration["locked_test_accessed"]):
        raise ValueError(f"Locked test was accessed in {run_dir}")
    return {
        domain: pd.read_csv(
            run_dir / f"best_validation_predictions_{domain}.csv"
        )
        for domain in DOMAINS
    }


def _ensemble_frame(frames: list[pd.DataFrame]) -> pd.DataFrame:
    required = {"sample_id", "target", "prediction", "probability_1"}
    merged: pd.DataFrame | None = None
    probability_columns = []
    for index, frame in enumerate(frames):
        if not required.issubset(frame):
            raise ValueError("Prediction file is missing required columns")
        expected = (frame["probability_1"].to_numpy() >= 0.5).astype(int)
        if not np.array_equal(expected, frame["prediction"].to_numpy(dtype=int)):
            raise ValueError("Stored prediction does not use threshold 0.5")
        probability_column = f"probability_1_{index}"
        selected = frame[["sample_id", "target", "probability_1"]].rename(
            columns={"probability_1": probability_column}
        )
        if merged is None:
            merged = selected
        else:
            merged = merged.merge(
                selected,
                on=["sample_id", "target"],
                how="inner",
                validate="one_to_one",
            )
        probability_columns.append(probability_column)
    if merged is None or len(merged) != len(frames[0]):
        raise ValueError("Prediction files do not contain identical samples")
    merged["probability_1"] = merged[probability_columns].mean(axis=1)
    merged["prediction"] = (merged["probability_1"] >= 0.5).astype(int)
    return merged[
        ["sample_id", "target", "probability_1", "prediction"]
    ].copy()


def _metrics(frame: pd.DataFrame) -> dict[str, float]:
    target = frame["target"].to_numpy(dtype=int)
    prediction = frame["prediction"].to_numpy(dtype=int)
    normal = target == 0
    adventitious = target == 1
    if not normal.any() or not adventitious.any():
        raise ValueError("Both classes are required")
    specificity = float((prediction[normal] == 0).mean())
    sensitivity = float((prediction[adventitious] == 1).mean())
    accuracy = float((prediction == target).mean())
    return {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "average_score": (sensitivity + specificity) / 2.0,
    }


def _pair_with_patients(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    patient_lookup: pd.DataFrame,
) -> pd.DataFrame:
    paired = candidate.merge(
        reference,
        on=["sample_id", "target"],
        suffixes=("_candidate", "_reference"),
        validate="one_to_one",
    )
    if len(paired) != len(candidate) or len(paired) != len(reference):
        raise ValueError("Candidate and reference predictions are not fully paired")
    paired = paired.merge(
        patient_lookup,
        on="sample_id",
        how="left",
        validate="many_to_one",
    )
    if paired["patient_id"].isna().any():
        raise ValueError("Patient lookup is incomplete")
    return paired


def _average_score(target: np.ndarray, prediction: np.ndarray) -> float:
    normal = target == 0
    adventitious = target == 1
    if not normal.any() or not adventitious.any():
        raise ValueError("Both classes are required")
    return float(
        (
            (prediction[normal] == 0).mean()
            + (prediction[adventitious] == 1).mean()
        )
        / 2.0
    )


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
    target = paired["target"].to_numpy(dtype=int)
    candidate = paired["prediction_candidate"].to_numpy(dtype=int)
    reference = paired["prediction_reference"].to_numpy(dtype=int)
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
        sampled_target = target[indices]
        if not (
            np.any(sampled_target == 0)
            and np.any(sampled_target == 1)
        ):
            continue
        gains[completed] = (
            _average_score(sampled_target, candidate[indices])
            - _average_score(sampled_target, reference[indices])
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


def _parent_diversity(
    balanced: dict[str, pd.DataFrame],
    event_random: dict[str, pd.DataFrame],
) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = []
    by_domain = {}
    for domain in DOMAINS:
        paired = balanced[domain].merge(
            event_random[domain],
            on=["sample_id", "target"],
            suffixes=("_balanced", "_event_random"),
            validate="one_to_one",
        )
        target = paired["target"].to_numpy(dtype=int)
        balanced_prediction = paired["prediction_balanced"].to_numpy(dtype=int)
        random_prediction = paired["prediction_event_random"].to_numpy(dtype=int)
        disagreement = balanced_prediction != random_prediction
        balanced_unique = (balanced_prediction == target) & (
            random_prediction != target
        )
        random_unique = (random_prediction == target) & (
            balanced_prediction != target
        )
        by_domain[domain] = {
            "events": len(paired),
            "disagreement_fraction": float(disagreement.mean()),
            "balanced_unique_correct_fraction": float(balanced_unique.mean()),
            "event_random_unique_correct_fraction": float(random_unique.mean()),
        }
        rows.append(
            pd.DataFrame(
                {
                    "disagreement": disagreement,
                    "balanced_unique": balanced_unique,
                    "random_unique": random_unique,
                }
            )
        )
    combined = pd.concat(rows, ignore_index=True)
    overall = {
        "events": len(combined),
        "disagreement_fraction": float(combined["disagreement"].mean()),
        "balanced_unique_correct_fraction": float(
            combined["balanced_unique"].mean()
        ),
        "event_random_unique_correct_fraction": float(
            combined["random_unique"].mean()
        ),
    }
    return overall, by_domain


def main() -> None:
    args = parse_args()
    if args.bootstrap_iterations < 100:
        raise ValueError("At least 100 bootstrap iterations are required")
    if len(set(args.seeds)) != 3:
        raise ValueError("Gate 10A requires three distinct seeds")

    balanced_seed_frames = {domain: [] for domain in DOMAINS}
    random_seed_frames = {domain: [] for domain in DOMAINS}
    for seed, balanced_run, random_run in zip(
        args.seeds,
        args.balanced_runs,
        args.event_random_runs,
        strict=True,
    ):
        balanced = _load_predictions(
            balanced_run,
            expected_seed=seed,
            expected_mode="domain_class_event",
        )
        event_random = _load_predictions(
            random_run,
            expected_seed=seed,
            expected_mode="event_random",
        )
        for domain in DOMAINS:
            balanced_seed_frames[domain].append(balanced[domain])
            random_seed_frames[domain].append(event_random[domain])

    balanced_parent = {
        domain: _ensemble_frame(balanced_seed_frames[domain])
        for domain in DOMAINS
    }
    random_parent = {
        domain: _ensemble_frame(random_seed_frames[domain])
        for domain in DOMAINS
    }
    candidate = {}
    for domain in DOMAINS:
        frame = balanced_parent[domain][
            ["sample_id", "target", "probability_1"]
        ].merge(
            random_parent[domain][
                ["sample_id", "target", "probability_1"]
            ],
            on=["sample_id", "target"],
            suffixes=("_balanced", "_event_random"),
            validate="one_to_one",
        )
        frame["probability_1"] = (
            frame["probability_1_balanced"]
            + frame["probability_1_event_random"]
        ) / 2.0
        frame["prediction"] = (frame["probability_1"] >= 0.5).astype(int)
        candidate[domain] = frame[
            ["sample_id", "target", "probability_1", "prediction"]
        ]

    parent_metrics = {
        "balanced_parent": {
            domain: _metrics(balanced_parent[domain])
            for domain in DOMAINS
        },
        "event_random_parent": {
            domain: _metrics(random_parent[domain])
            for domain in DOMAINS
        },
    }
    parent_scores = {}
    for name, metrics in parent_metrics.items():
        scores = [metrics[domain]["average_score"] for domain in DOMAINS]
        parent_scores[name] = {
            "worst_database": min(scores),
            "mean_database": float(np.mean(scores)),
        }
    reference_name = max(
        parent_scores,
        key=lambda name: (
            parent_scores[name]["worst_database"],
            parent_scores[name]["mean_database"],
        ),
    )
    reference = (
        balanced_parent
        if reference_name == "balanced_parent"
        else random_parent
    )
    reference_metrics = parent_metrics[reference_name]
    candidate_metrics = {
        domain: _metrics(candidate[domain]) for domain in DOMAINS
    }

    manifest = pd.read_csv(args.manifest, dtype={"patient_id": str})
    validation = manifest.loc[
        manifest["protocol_role"].eq("validation_select"),
        ["sample_id", "patient_id", "dataset"],
    ]
    random_generator = np.random.default_rng(args.bootstrap_seed)
    bootstrap: dict[str, Any] = {
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
        paired = _pair_with_patients(
            candidate[domain],
            reference[domain],
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
            "all_six_checkpoints_have_identical_samples_and_targets": True,
            "fixed_threshold": 0.5,
        }
    mean_gains = np.mean(np.stack(gain_samples), axis=0)
    bootstrap["mean_domain"] = _bootstrap_summary(mean_gains)

    diversity_overall, diversity_by_domain = _parent_diversity(
        balanced_parent,
        random_parent,
    )
    decision = sampling_diverse_ensemble_decision(
        candidate_metrics=candidate_metrics,
        reference_metrics=reference_metrics,
        parent_diversity=diversity_overall,
        bootstrap=bootstrap,
        reference_name=reference_name,
    )
    result = {
        "gate": "10A",
        "status": (
            "gate_10a_sampling_diverse_ensemble_passed"
            if decision["passed"]
            else "gate_10a_sampling_diverse_ensemble_failed"
        ),
        "development_only": True,
        "calibration_accessed": False,
        "locked_tests_accessed": False,
        "seeds": args.seeds,
        "balanced_runs": [str(path) for path in args.balanced_runs],
        "event_random_runs": [str(path) for path in args.event_random_runs],
        "pairing_audit": pairing_audit,
        "parent_metrics": parent_metrics,
        "parent_selection_scores": parent_scores,
        "candidate_metrics": candidate_metrics,
        "parent_diversity": {
            "overall": diversity_overall,
            "per_domain": diversity_by_domain,
        },
        "patient_cluster_bootstrap": bootstrap,
        "decision": decision,
        "freeze": "artifacts/gate10a_sampling_diverse_ensemble_freeze.json",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
