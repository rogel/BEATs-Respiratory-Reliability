#!/usr/bin/env python3
"""Summarize paired Gate 5D runs with patient-cluster bootstrap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DOMAINS = ("icbhi2017", "sprsound2022")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--candidate-runs",
        type=Path,
        nargs=3,
        required=True,
    )
    parser.add_argument(
        "--control-runs",
        type=Path,
        nargs=3,
        required=True,
    )
    parser.add_argument("--seeds", type=int, nargs=3, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=4_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_729)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _average_score(targets: np.ndarray, predictions: np.ndarray) -> float:
    normal = targets == 0
    adventitious = targets == 1
    if not normal.any() or not adventitious.any():
        raise ValueError("Both classes are required to calculate Average Score")
    specificity = float((predictions[normal] == 0).mean())
    sensitivity = float((predictions[adventitious] == 1).mean())
    return (specificity + sensitivity) / 2.0


def _load_run(
    run_dir: Path,
    expected_seed: int,
    expected_sampling_mode: str,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    configuration = json.loads(
        (run_dir / "configuration.json").read_text(encoding="utf-8")
    )
    if int(configuration["seed"]) != expected_seed:
        raise ValueError(f"Seed mismatch in {run_dir}")
    if configuration["sampling_mode"] != expected_sampling_mode:
        raise ValueError(f"Sampling-mode mismatch in {run_dir}")
    if configuration["locked_test_accessed"]:
        raise ValueError(f"Locked test was accessed in {run_dir}")
    predictions = {
        domain: pd.read_csv(
            run_dir / f"best_validation_predictions_{domain}.csv"
        )
        for domain in DOMAINS
    }
    return summary, predictions


def _run_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    domain_scores = {
        domain: float(
            summary["best_validation_metrics"][domain]["average_score"]
        )
        for domain in DOMAINS
    }
    return {
        "domain_metrics": {
            domain: {
                metric: float(
                    summary["best_validation_metrics"][domain][metric]
                )
                for metric in (
                    "average_score",
                    "sensitivity",
                    "specificity",
                    "macro_f1",
                )
            }
            for domain in DOMAINS
        },
        "worst_domain_average_score": min(domain_scores.values()),
        "mean_domain_average_score": sum(domain_scores.values()) / len(DOMAINS),
    }


def _ensemble_predictions(
    frames: list[pd.DataFrame],
    patient_lookup: pd.DataFrame,
) -> pd.DataFrame:
    base = frames[0][["sample_id", "target"]].copy()
    probability_columns = []
    for index, frame in enumerate(frames):
        probability_column = f"probability_1_seed_{index}"
        candidate = frame[["sample_id", "target", "probability_1"]].rename(
            columns={"probability_1": probability_column}
        )
        base = base.merge(
            candidate,
            on=["sample_id", "target"],
            how="inner",
            validate="one_to_one",
        )
        probability_columns.append(probability_column)
    if len(base) != len(frames[0]):
        raise ValueError("Prediction files do not contain identical samples")
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


def _bootstrap_gain(
    candidate: pd.DataFrame,
    control: pd.DataFrame,
    iterations: int,
    random_generator: np.random.Generator,
) -> np.ndarray:
    merged = candidate.merge(
        control,
        on=["sample_id", "patient_id", "target"],
        suffixes=("_candidate", "_control"),
        validate="one_to_one",
    )
    patients = merged["patient_id"].astype(str).unique()
    patient_values = merged["patient_id"].astype(str).to_numpy()
    patient_rows = {
        patient: np.flatnonzero(patient_values == patient) for patient in patients
    }
    targets = merged["target"].to_numpy(dtype=int)
    candidate_predictions = (
        merged["probability_1_candidate"].to_numpy(dtype=float) >= 0.5
    ).astype(int)
    control_predictions = (
        merged["probability_1_control"].to_numpy(dtype=float) >= 0.5
    ).astype(int)
    gains = np.empty(iterations, dtype=float)
    completed = 0
    while completed < iterations:
        sampled_patients = random_generator.choice(
            patients,
            size=len(patients),
            replace=True,
        )
        indices = np.concatenate(
            [patient_rows[str(patient)] for patient in sampled_patients]
        )
        sampled_targets = targets[indices]
        if not (sampled_targets == 0).any() or not (sampled_targets == 1).any():
            continue
        gains[completed] = _average_score(
            targets[indices],
            candidate_predictions[indices],
        ) - _average_score(
            targets[indices],
            control_predictions[indices],
        )
        completed += 1
    return gains


def main() -> None:
    args = parse_args()
    if args.bootstrap_iterations < 100:
        raise ValueError("At least 100 bootstrap iterations are required")
    manifest = pd.read_csv(args.manifest)
    patient_lookup = manifest[
        ["sample_id", "patient_id", "dataset", "protocol_role"]
    ].copy()
    patient_lookup = patient_lookup.loc[
        patient_lookup["protocol_role"].eq("validation_select")
    ]

    candidate_runs = []
    control_runs = []
    candidate_predictions = {domain: [] for domain in DOMAINS}
    control_predictions = {domain: [] for domain in DOMAINS}
    for seed, candidate_path, control_path in zip(
        args.seeds,
        args.candidate_runs,
        args.control_runs,
        strict=True,
    ):
        candidate_summary, candidate_frames = _load_run(
            candidate_path,
            seed,
            "domain_class_event",
        )
        control_summary, control_frames = _load_run(
            control_path,
            seed,
            "event_random",
        )
        candidate_runs.append(_run_metrics(candidate_summary))
        control_runs.append(_run_metrics(control_summary))
        for domain in DOMAINS:
            candidate_predictions[domain].append(candidate_frames[domain])
            control_predictions[domain].append(control_frames[domain])

    paired_rows = []
    for seed, candidate, control in zip(
        args.seeds,
        candidate_runs,
        control_runs,
        strict=True,
    ):
        paired_rows.append(
            {
                "seed": seed,
                "candidate": candidate,
                "control": control,
                "gains": {
                    domain: (
                        candidate["domain_metrics"][domain]["average_score"]
                        - control["domain_metrics"][domain]["average_score"]
                    )
                    for domain in DOMAINS
                }
                | {
                    "worst_domain": (
                        candidate["worst_domain_average_score"]
                        - control["worst_domain_average_score"]
                    ),
                    "mean_domain": (
                        candidate["mean_domain_average_score"]
                        - control["mean_domain_average_score"]
                    ),
                },
            }
        )

    mean_gains = {
        key: float(np.mean([row["gains"][key] for row in paired_rows]))
        for key in (*DOMAINS, "worst_domain", "mean_domain")
    }
    wins = {
        metric: sum(row["gains"][metric] > 0 for row in paired_rows)
        for metric in ("worst_domain", "mean_domain")
    }
    safety = all(
        run["domain_metrics"][domain][metric] >= 0.5
        for run in candidate_runs
        for domain in DOMAINS
        for metric in ("sensitivity", "specificity")
    )

    random_generator = np.random.default_rng(args.bootstrap_seed)
    bootstrap_by_domain = {}
    domain_gain_samples = []
    for domain in DOMAINS:
        lookup = patient_lookup.loc[
            patient_lookup["dataset"].astype(str).eq(domain),
            ["sample_id", "patient_id"],
        ]
        candidate_ensemble = _ensemble_predictions(
            candidate_predictions[domain],
            lookup,
        )
        control_ensemble = _ensemble_predictions(
            control_predictions[domain],
            lookup,
        )
        gains = _bootstrap_gain(
            candidate_ensemble,
            control_ensemble,
            args.bootstrap_iterations,
            random_generator,
        )
        domain_gain_samples.append(gains)
        bootstrap_by_domain[domain] = {
            "mean_gain": float(gains.mean()),
            "confidence_interval_95": [
                float(np.quantile(gains, 0.025)),
                float(np.quantile(gains, 0.975)),
            ],
            "probability_gain_above_zero": float((gains > 0).mean()),
        }
    mean_domain_bootstrap = np.mean(np.stack(domain_gain_samples), axis=0)
    bootstrap_summary = {
        "per_domain": bootstrap_by_domain,
        "mean_domain": {
            "mean_gain": float(mean_domain_bootstrap.mean()),
            "confidence_interval_95": [
                float(np.quantile(mean_domain_bootstrap, 0.025)),
                float(np.quantile(mean_domain_bootstrap, 0.975)),
            ],
            "probability_gain_above_zero": float(
                (mean_domain_bootstrap > 0).mean()
            ),
        },
        "iterations": args.bootstrap_iterations,
        "seed": args.bootstrap_seed,
    }

    gates = {
        "mean_domain_gain_at_least_0_015": mean_gains["mean_domain"] >= 0.015,
        "worst_domain_gain_at_least_0_015": (
            mean_gains["worst_domain"] >= 0.015
        ),
        "mean_and_worst_wins_at_least_2_of_3": (
            wins["mean_domain"] >= 2 and wins["worst_domain"] >= 2
        ),
        "no_database_mean_drop_greater_than_0_005": all(
            mean_gains[domain] >= -0.005 for domain in DOMAINS
        ),
        "candidate_safety_every_seed": safety,
        "patient_bootstrap_probability_at_least_0_90": (
            bootstrap_summary["mean_domain"]["probability_gain_above_zero"]
            >= 0.9
        ),
    }
    gates["passed"] = all(gates.values())
    output = {
        "status": "gate_5d_passed" if gates["passed"] else "gate_5d_rejected",
        "seeds": args.seeds,
        "development_only": True,
        "locked_tests_accessed": False,
        "paired_seed_results": paired_rows,
        "mean_paired_gains": mean_gains,
        "seed_wins": wins,
        "patient_cluster_bootstrap": bootstrap_summary,
        "gate": gates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
