#!/usr/bin/env python3
"""Select and evaluate a three-seed consensus ensemble without test access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from respiratory_sound.calibration import probability_metrics
from respiratory_sound.metrics import respiratory_metrics

CLASS_NAMES = ("normal", "adventitious")
DOMAINS = ("icbhi2017", "sprsound2022")
FAMILIES = ("event_random", "domain_class_event")
SEEDS = (20_260_729, 20_260_730, 20_260_731)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/cross_domain_binary.csv"),
    )
    parser.add_argument(
        "--calibration-root",
        type=Path,
        default=Path("artifacts/gate6a_predictions/calibration"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/gate6a_predictions/ensembles"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("artifacts/gate6a_seed_ensemble_final.json"),
    )
    return parser.parse_args()


def _run_name(family: str, seed: int) -> str:
    prefix = "gate5c" if seed == SEEDS[0] else "gate5d"
    return f"{prefix}_{family}_plain_seed{seed}"


def _load_prediction(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"patient_id": str})
    required = {"sample_id", "target", "probability_1"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if frame["sample_id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate sample IDs")
    return frame


def _prediction_path(
    root: Path,
    calibration_root: Path,
    family: str,
    seed: int,
    domain: str,
    role: str,
) -> Path:
    if role == "calibration":
        return calibration_root / f"{family}_seed{seed}_{domain}.csv"
    if role == "validation_select":
        return (
            root
            / "runs"
            / _run_name(family, seed)
            / f"best_validation_predictions_{domain}.csv"
        )
    raise ValueError(f"Unsupported role: {role}")


def _aggregate(
    frames: list[pd.DataFrame],
    lookup: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    base = frames[0][["sample_id", "target"]].copy()
    probability_columns = []
    individual_metrics = []
    for seed, frame in zip(SEEDS, frames, strict=True):
        column = f"probability_1_seed{seed}"
        candidate = frame[["sample_id", "target", "probability_1"]].rename(
            columns={"probability_1": column}
        )
        base = base.merge(
            candidate,
            on=["sample_id", "target"],
            how="inner",
            validate="one_to_one",
        )
        probability_columns.append(column)
        targets = frame["target"].to_numpy(dtype=int)
        probabilities = frame["probability_1"].to_numpy(dtype=float)
        individual_metrics.append(
            {
                "seed": seed,
                "classification": respiratory_metrics(
                    targets,
                    (probabilities >= 0.5).astype(int),
                    CLASS_NAMES,
                ),
                "probability": probability_metrics(targets, probabilities),
            }
        )
    if len(base) != len(frames[0]):
        raise ValueError("Seed prediction files contain different samples")
    base = base.merge(
        lookup,
        on="sample_id",
        how="left",
        validate="many_to_one",
    )
    if base[["patient_id", "dataset", "protocol_role"]].isna().any().any():
        raise ValueError("Manifest lookup is incomplete")
    base["probability_1"] = base[probability_columns].mean(axis=1)
    base["seed_disagreement"] = base[probability_columns].std(
        axis=1,
        ddof=0,
    )
    base["prediction"] = (base["probability_1"] >= 0.5).astype(int)
    targets = base["target"].to_numpy(dtype=int)
    probabilities = base["probability_1"].to_numpy(dtype=float)
    metrics = {
        "classification": respiratory_metrics(
            targets,
            base["prediction"].to_numpy(dtype=int),
            CLASS_NAMES,
        ),
        "probability": probability_metrics(targets, probabilities),
        "individual_models": individual_metrics,
    }
    return base, metrics


def _family_summary(
    role_metrics: dict[str, dict[str, Any]],
) -> dict[str, float]:
    domain_scores = {
        domain: float(
            role_metrics[domain]["classification"]["average_score"]
        )
        for domain in DOMAINS
    }
    domain_nll = {
        domain: float(
            role_metrics[domain]["probability"]["negative_log_likelihood"]
        )
        for domain in DOMAINS
    }
    return {
        "worst_domain_average_score": min(domain_scores.values()),
        "mean_domain_average_score": float(np.mean(list(domain_scores.values()))),
        "mean_domain_negative_log_likelihood": float(
            np.mean(list(domain_nll.values()))
        ),
    }


def _select_family(
    summaries: dict[str, dict[str, float]],
) -> str:
    return max(
        FAMILIES,
        key=lambda family: (
            summaries[family]["worst_domain_average_score"],
            summaries[family]["mean_domain_average_score"],
            -summaries[family]["mean_domain_negative_log_likelihood"],
            family == "event_random",
        ),
    )


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    manifest_path = (root / args.manifest).resolve()
    calibration_root = (root / args.calibration_root).resolve()
    output_root = (root / args.output_root).resolve()
    output_json = (root / args.output_json).resolve()
    manifest = pd.read_csv(manifest_path, dtype={"patient_id": str})
    lookup = manifest[
        ["sample_id", "patient_id", "dataset", "protocol_role"]
    ].copy()

    all_metrics: dict[str, dict[str, dict[str, Any]]] = {
        family: {} for family in FAMILIES
    }
    aggregate_frames: dict[str, dict[str, dict[str, pd.DataFrame]]] = {
        family: {} for family in FAMILIES
    }
    for family in FAMILIES:
        for role in ("calibration", "validation_select"):
            all_metrics[family][role] = {}
            aggregate_frames[family][role] = {}
            for domain in DOMAINS:
                frames = [
                    _load_prediction(
                        _prediction_path(
                            root,
                            calibration_root,
                            family,
                            seed,
                            domain,
                            role,
                        )
                    )
                    for seed in SEEDS
                ]
                role_lookup = lookup.loc[
                    lookup["dataset"].astype(str).eq(domain)
                    & lookup["protocol_role"].eq(role)
                ]
                aggregate, metrics = _aggregate(frames, role_lookup)
                all_metrics[family][role][domain] = metrics
                aggregate_frames[family][role][domain] = aggregate

    calibration_summaries = {
        family: _family_summary(all_metrics[family]["calibration"])
        for family in FAMILIES
    }
    selected_family = _select_family(calibration_summaries)
    selected_validation = all_metrics[selected_family]["validation_select"]
    validation_ensemble_summary = _family_summary(selected_validation)

    individual_worst_scores = []
    individual_mean_scores = []
    individual_domain_scores = {domain: [] for domain in DOMAINS}
    for model_index in range(len(SEEDS)):
        scores = {
            domain: float(
                selected_validation[domain]["individual_models"][model_index][
                    "classification"
                ]["average_score"]
            )
            for domain in DOMAINS
        }
        for domain in DOMAINS:
            individual_domain_scores[domain].append(scores[domain])
        individual_worst_scores.append(min(scores.values()))
        individual_mean_scores.append(float(np.mean(list(scores.values()))))
    mean_individual = {
        "worst_domain_average_score": float(np.mean(individual_worst_scores)),
        "mean_domain_average_score": float(np.mean(individual_mean_scores)),
        "domain_average_scores": {
            domain: float(np.mean(scores))
            for domain, scores in individual_domain_scores.items()
        },
    }
    gains = {
        "worst_domain": (
            validation_ensemble_summary["worst_domain_average_score"]
            - mean_individual["worst_domain_average_score"]
        ),
        "mean_domain": (
            validation_ensemble_summary["mean_domain_average_score"]
            - mean_individual["mean_domain_average_score"]
        ),
        **{
            domain: (
                selected_validation[domain]["classification"]["average_score"]
                - mean_individual["domain_average_scores"][domain]
            )
            for domain in DOMAINS
        },
    }
    selected_calibration_patients = set(
        aggregate_frames[selected_family]["calibration"][DOMAINS[0]][
            "patient_id"
        ]
    ).union(
        aggregate_frames[selected_family]["calibration"][DOMAINS[1]][
            "patient_id"
        ]
    )
    selected_validation_patients = set(
        aggregate_frames[selected_family]["validation_select"][DOMAINS[0]][
            "patient_id"
        ]
    ).union(
        aggregate_frames[selected_family]["validation_select"][DOMAINS[1]][
            "patient_id"
        ]
    )
    overlap = sorted(
        selected_calibration_patients.intersection(selected_validation_patients)
    )
    gates = {
        "worst_domain_gain_at_least_0_005": gains["worst_domain"] >= 0.005,
        "mean_domain_gain_at_least_0_005": gains["mean_domain"] >= 0.005,
        "no_domain_drop_greater_than_0_005": all(
            gains[domain] >= -0.005 for domain in DOMAINS
        ),
        "sensitivity_specificity_each_domain_at_least_0_50": all(
            selected_validation[domain]["classification"][metric] >= 0.5
            for domain in DOMAINS
            for metric in ("sensitivity", "specificity")
        ),
        "deployed_parameters_at_most_2m": 3 * 539_482 <= 2_000_000,
        "calibration_validation_patient_overlap_zero": not overlap,
    }
    gates["passed"] = all(gates.values())

    output_root.mkdir(parents=True, exist_ok=True)
    for role in ("calibration", "validation_select"):
        for domain in DOMAINS:
            aggregate_frames[selected_family][role][domain].to_csv(
                output_root / f"{selected_family}_{role}_{domain}.csv",
                index=False,
            )
    payload = {
        "status": (
            "gate_6a_passed" if gates["passed"] else "gate_6a_rejected"
        ),
        "selected_family": selected_family,
        "family_selection_role": "calibration",
        "calibration_family_summaries": calibration_summaries,
        "validation_selected_ensemble": {
            "summary": validation_ensemble_summary,
            "domain_metrics": selected_validation,
        },
        "validation_mean_individual_comparator": mean_individual,
        "validation_ensemble_gains": gains,
        "patient_overlap": overlap,
        "deployed_parameters": 3 * 539_482,
        "additional_training": False,
        "locked_tests_accessed": False,
        "gate": gates,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
