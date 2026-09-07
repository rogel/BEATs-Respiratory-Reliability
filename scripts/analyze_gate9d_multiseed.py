#!/usr/bin/env python3
"""Analyze the frozen Gate 9D three-seed matched sampling ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from respiratory_sound.gate9d import multiseed_sampling_decision

DOMAINS = ("icbhi2017", "sprsound2022")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--balanced-runs", type=Path, nargs=3, required=True)
    parser.add_argument("--event-random-runs", type=Path, nargs=3, required=True)
    parser.add_argument("--seeds", type=int, nargs=3, required=True)
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


def _sampling_mode(configuration: dict[str, Any]) -> str:
    if "sampling_mode" in configuration:
        return str(configuration["sampling_mode"])
    return str(configuration["experiment"]["sampling"]["mode"])


def _load_run(
    run_dir: Path,
    *,
    expected_seed: int,
    expected_mode: str,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    configuration = json.loads(
        (run_dir / "configuration.json").read_text(encoding="utf-8")
    )
    if int(configuration["seed"]) != expected_seed:
        raise ValueError(f"Seed mismatch in {run_dir}")
    if _sampling_mode(configuration) != expected_mode:
        raise ValueError(f"Sampling-mode mismatch in {run_dir}")
    if bool(configuration["locked_test_accessed"]):
        raise ValueError(f"Locked test was accessed in {run_dir}")
    predictions = {
        domain: pd.read_csv(
            run_dir / f"best_validation_predictions_{domain}.csv"
        )
        for domain in DOMAINS
    }
    return summary, predictions


def _run_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    metrics = {
        domain: {
            metric: float(summary["best_validation_metrics"][domain][metric])
            for metric in (
                "average_score",
                "sensitivity",
                "specificity",
                "macro_f1",
            )
        }
        for domain in DOMAINS
    }
    scores = [metrics[domain]["average_score"] for domain in DOMAINS]
    return {
        "domain_metrics": metrics,
        "worst_database_average_score": min(scores),
        "mean_database_average_score": float(np.mean(scores)),
    }


def _ensemble_predictions(
    frames: list[pd.DataFrame],
    patient_lookup: pd.DataFrame,
) -> pd.DataFrame:
    required = {"sample_id", "target", "prediction", "probability_1"}
    base: pd.DataFrame | None = None
    probability_columns: list[str] = []
    for index, frame in enumerate(frames):
        if not required.issubset(frame):
            raise ValueError("Prediction file is missing required columns")
        expected = (frame["probability_1"].to_numpy() >= 0.5).astype(int)
        if not np.array_equal(expected, frame["prediction"].to_numpy(dtype=int)):
            raise ValueError("Stored predictions do not use threshold 0.5")
        probability_column = f"probability_1_seed_{index}"
        candidate = frame[["sample_id", "target", "probability_1"]].rename(
            columns={"probability_1": probability_column}
        )
        if base is None:
            base = candidate
        else:
            base = base.merge(
                candidate,
                on=["sample_id", "target"],
                how="inner",
                validate="one_to_one",
            )
        probability_columns.append(probability_column)
    if base is None or len(base) != len(frames[0]):
        raise ValueError("Seed prediction files do not contain identical samples")
    base["probability_1"] = base[probability_columns].mean(axis=1)
    base = base.merge(
        patient_lookup,
        on="sample_id",
        how="left",
        validate="many_to_one",
    )
    if base["patient_id"].isna().any():
        raise ValueError("Patient lookup is incomplete")
    return base[["sample_id", "patient_id", "target", "probability_1"]]


def _paired_bootstrap_gain(
    balanced: pd.DataFrame,
    event_random: pd.DataFrame,
    *,
    iterations: int,
    random_generator: np.random.Generator,
) -> np.ndarray:
    paired = balanced.merge(
        event_random,
        on=["sample_id", "patient_id", "target"],
        suffixes=("_balanced", "_event_random"),
        validate="one_to_one",
    )
    if len(paired) != len(balanced) or len(paired) != len(event_random):
        raise ValueError("Balanced and event-random predictions are not fully paired")
    patient_values = paired["patient_id"].astype(str).to_numpy()
    patients = np.unique(patient_values)
    rows_by_patient = {
        patient: np.flatnonzero(patient_values == patient)
        for patient in patients
    }
    targets = paired["target"].to_numpy(dtype=int)
    balanced_predictions = (
        paired["probability_1_balanced"].to_numpy(dtype=float) >= 0.5
    ).astype(int)
    random_predictions = (
        paired["probability_1_event_random"].to_numpy(dtype=float) >= 0.5
    ).astype(int)
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
            _average_score(sampled_targets, balanced_predictions[indices])
            - _average_score(sampled_targets, random_predictions[indices])
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
    if len(set(args.seeds)) != 3:
        raise ValueError("Gate 9D requires three distinct seeds")

    balanced_runs: list[dict[str, Any]] = []
    random_runs: list[dict[str, Any]] = []
    balanced_predictions = {domain: [] for domain in DOMAINS}
    random_predictions = {domain: [] for domain in DOMAINS}
    for seed, balanced_path, random_path in zip(
        args.seeds,
        args.balanced_runs,
        args.event_random_runs,
        strict=True,
    ):
        balanced_summary, balanced_frames = _load_run(
            balanced_path,
            expected_seed=seed,
            expected_mode="domain_class_event",
        )
        random_summary, random_frames = _load_run(
            random_path,
            expected_seed=seed,
            expected_mode="event_random",
        )
        balanced_runs.append(_run_metrics(balanced_summary))
        random_runs.append(_run_metrics(random_summary))
        for domain in DOMAINS:
            balanced_predictions[domain].append(balanced_frames[domain])
            random_predictions[domain].append(random_frames[domain])

    paired_seed_results = []
    for seed, balanced, random in zip(
        args.seeds,
        balanced_runs,
        random_runs,
        strict=True,
    ):
        gains = {
            domain: (
                balanced["domain_metrics"][domain]["average_score"]
                - random["domain_metrics"][domain]["average_score"]
            )
            for domain in DOMAINS
        }
        gains["worst_database"] = (
            balanced["worst_database_average_score"]
            - random["worst_database_average_score"]
        )
        gains["mean_database"] = (
            balanced["mean_database_average_score"]
            - random["mean_database_average_score"]
        )
        paired_seed_results.append(
            {
                "seed": seed,
                "balanced": balanced,
                "event_random": random,
                "gains": gains,
            }
        )

    gain_keys = (*DOMAINS, "worst_database", "mean_database")
    mean_paired_gains = {
        key: float(
            np.mean([row["gains"][key] for row in paired_seed_results])
        )
        for key in gain_keys
    }
    seed_wins = {
        key: sum(row["gains"][key] > 0 for row in paired_seed_results)
        for key in ("worst_database", "mean_database")
    }
    balanced_safety = all(
        run["domain_metrics"][domain][metric] >= 0.5
        for run in balanced_runs
        for domain in DOMAINS
        for metric in ("sensitivity", "specificity")
    )
    mean_balanced_scores = {
        domain: float(
            np.mean(
                [
                    run["domain_metrics"][domain]["average_score"]
                    for run in balanced_runs
                ]
            )
        )
        for domain in DOMAINS
    }
    weakest_database = min(mean_balanced_scores, key=mean_balanced_scores.get)

    manifest = pd.read_csv(args.manifest, dtype={"patient_id": str})
    validation = manifest.loc[
        manifest["protocol_role"].eq("validation_select"),
        ["sample_id", "patient_id", "dataset"],
    ]
    random_generator = np.random.default_rng(args.bootstrap_seed)
    bootstrap: dict[str, Any] = {
        "method": (
            "paired patient-cluster bootstrap of three-seed probability "
            "ensembles within each sampling arm"
        ),
        "per_domain": {},
        "iterations": args.bootstrap_iterations,
        "seed": args.bootstrap_seed,
    }
    gain_samples = []
    pairing_audit = {}
    for domain in DOMAINS:
        patient_lookup = validation.loc[
            validation["dataset"].astype(str).eq(domain),
            ["sample_id", "patient_id"],
        ]
        balanced_ensemble = _ensemble_predictions(
            balanced_predictions[domain],
            patient_lookup,
        )
        random_ensemble = _ensemble_predictions(
            random_predictions[domain],
            patient_lookup,
        )
        gains = _paired_bootstrap_gain(
            balanced_ensemble,
            random_ensemble,
            iterations=args.bootstrap_iterations,
            random_generator=random_generator,
        )
        gain_samples.append(gains)
        bootstrap["per_domain"][domain] = _bootstrap_summary(gains)
        pairing_audit[domain] = {
            "events": len(balanced_ensemble),
            "patients": int(balanced_ensemble["patient_id"].nunique()),
            "three_seeds_identical_samples_and_targets": True,
            "arms_fully_paired": True,
            "fixed_threshold_predictions_verified": True,
        }
    mean_domain_gains = np.mean(np.stack(gain_samples), axis=0)
    bootstrap["mean_domain"] = _bootstrap_summary(mean_domain_gains)

    decision = multiseed_sampling_decision(
        paired_seed_results=paired_seed_results,
        mean_paired_gains=mean_paired_gains,
        seed_wins=seed_wins,
        balanced_safety_every_seed=balanced_safety,
        weakest_database=weakest_database,
        bootstrap=bootstrap,
    )
    result = {
        "gate": "9D-multiseed",
        "status": (
            "gate_9d_multiseed_passed"
            if decision["passed"]
            else "gate_9d_multiseed_failed"
        ),
        "seeds": args.seeds,
        "development_only": True,
        "calibration_accessed": False,
        "locked_tests_accessed": False,
        "balanced_runs": [str(path) for path in args.balanced_runs],
        "event_random_runs": [str(path) for path in args.event_random_runs],
        "pairing_audit": pairing_audit,
        "paired_seed_results": paired_seed_results,
        "mean_balanced_scores": mean_balanced_scores,
        "mean_paired_gains": mean_paired_gains,
        "seed_wins": seed_wins,
        "patient_cluster_bootstrap": bootstrap,
        "decision": decision,
        "freeze": "artifacts/gate9d_multiseed_freeze.json",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
