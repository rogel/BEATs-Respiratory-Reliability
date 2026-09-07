#!/usr/bin/env python3
"""Diagnose whether patient-transferable threshold calibration is plausible."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from respiratory_sound.metrics import (
    paired_patient_bootstrap_difference,
    respiratory_metrics,
)
from respiratory_sound.threshold_diagnostics import (
    leave_one_patient_out_thresholds,
    patient_class_balanced_accuracy,
    threshold_summary,
)

CLASS_NAMES = ("normal", "adventitious")
DOMAINS = ("icbhi2017", "sprsound2022")
FAMILIES = ("domain_class_event", "event_random")
SEEDS = (20_260_729, 20_260_730, 20_260_731)
STRATEGIES = ("fixed", "global", "domain_conditioned")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction-root",
        type=Path,
        default=Path("artifacts/gate6a_predictions/calibration"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/gate7a0_predictions"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("artifacts/gate7a0_route_convergence_final.json"),
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=4_000)
    parser.add_argument("--seed", type=int, default=20_260_729)
    return parser.parse_args()


def _load_ensemble(
    prediction_root: Path,
    family: str,
    domain: str,
) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    probability_columns = []
    for seed in SEEDS:
        path = prediction_root / f"{family}_seed{seed}_{domain}.csv"
        frame = pd.read_csv(path, dtype={"patient_id": str})
        required = {"sample_id", "patient_id", "target", "probability_1"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        if frame["sample_id"].duplicated().any():
            raise ValueError(f"{path} contains duplicate sample IDs")
        probability_column = f"probability_1_seed{seed}"
        seed_frame = frame[
            ["sample_id", "patient_id", "target", "probability_1"]
        ].rename(columns={"probability_1": probability_column})
        if merged is None:
            merged = seed_frame
        else:
            merged = merged.merge(
                seed_frame,
                on=["sample_id", "patient_id", "target"],
                how="inner",
                validate="one_to_one",
            )
        probability_columns.append(probability_column)
    if merged is None:
        raise RuntimeError("no seed predictions loaded")
    merged["dataset"] = domain
    merged["probability_1"] = merged[probability_columns].mean(axis=1)
    merged["seed_disagreement"] = merged[probability_columns].std(axis=1, ddof=0)
    return merged


def _error_disagreement(
    targets: np.ndarray,
    predictions: np.ndarray,
    disagreements: np.ndarray,
) -> dict[str, float]:
    errors = (targets != predictions).astype(np.int64)
    payload = {
        "error_rate": float(errors.mean()),
        "median_disagreement_correct": float(np.median(disagreements[errors == 0])),
        "median_disagreement_error": float(np.median(disagreements[errors == 1])),
    }
    payload["error_detection_auroc"] = (
        float(roc_auc_score(errors, disagreements))
        if np.unique(errors).size == 2
        else float("nan")
    )
    return payload


def _analyze_family(
    frame: pd.DataFrame,
    bootstrap_iterations: int,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    analyzed = frame.copy()
    patient_thresholds: dict[str, dict[str, float]] = {}
    for strategy in STRATEGIES:
        predictions, thresholds, by_patient = leave_one_patient_out_thresholds(
            analyzed,
            strategy,
        )
        analyzed[f"threshold_{strategy}"] = thresholds
        analyzed[f"prediction_{strategy}"] = predictions
        patient_thresholds[strategy] = by_patient

    domain_payload: dict[str, Any] = {}
    for domain in DOMAINS:
        domain_mask = analyzed["dataset"].eq(domain).to_numpy()
        domain_frame = analyzed.loc[domain_mask]
        targets = domain_frame["target"].to_numpy(dtype=np.int64)
        probabilities = domain_frame["probability_1"].to_numpy(dtype=np.float64)
        patients = domain_frame["patient_id"].astype(str).to_numpy()
        disagreements = domain_frame["seed_disagreement"].to_numpy(dtype=np.float64)
        strategies: dict[str, Any] = {}
        for strategy in STRATEGIES:
            predictions = domain_frame[f"prediction_{strategy}"].to_numpy(
                dtype=np.int64
            )
            strategies[strategy] = {
                "event_metrics": respiratory_metrics(
                    targets,
                    predictions,
                    CLASS_NAMES,
                ),
                "patient_class_balanced_average_score": (
                    patient_class_balanced_accuracy(
                        targets,
                        predictions,
                        patients,
                    )
                ),
                "error_disagreement": _error_disagreement(
                    targets,
                    predictions,
                    disagreements,
                ),
                "threshold_distribution": threshold_summary(
                    patient_thresholds[strategy],
                    patient_filter=lambda patient, prefix=f"{domain}::": (
                        patient.startswith(prefix)
                    ),
                ),
            }
        fixed_predictions = domain_frame["prediction_fixed"].to_numpy(dtype=np.int64)
        for strategy in ("global", "domain_conditioned"):
            candidate = domain_frame[f"prediction_{strategy}"].to_numpy(
                dtype=np.int64
            )
            strategies[strategy]["paired_patient_bootstrap_vs_fixed"] = (
                paired_patient_bootstrap_difference(
                    targets,
                    candidate,
                    fixed_predictions,
                    patients,
                    metric="average_score",
                    iterations=bootstrap_iterations,
                    seed=seed,
                    class_names=CLASS_NAMES,
                )
            )
        domain_payload[domain] = {
            "samples": int(len(domain_frame)),
            "patients": int(domain_frame["patient_id"].nunique()),
            "ranking": {
                "auroc": float(roc_auc_score(targets, probabilities)),
                "auprc": float(average_precision_score(targets, probabilities)),
            },
            "strategies": strategies,
        }

    domain_gains = {
        domain: float(
            domain_payload[domain]["strategies"]["domain_conditioned"][
                "event_metrics"
            ]["average_score"]
            - domain_payload[domain]["strategies"]["fixed"]["event_metrics"][
                "average_score"
            ]
        )
        for domain in DOMAINS
    }
    domain_vs_global = {
        domain: float(
            domain_payload[domain]["strategies"]["domain_conditioned"][
                "event_metrics"
            ]["average_score"]
            - domain_payload[domain]["strategies"]["global"]["event_metrics"][
                "average_score"
            ]
        )
        for domain in DOMAINS
    }
    summary = {
        "domain_conditioned_gains_vs_fixed": domain_gains,
        "mean_domain_conditioned_gain_vs_fixed": float(
            np.mean(list(domain_gains.values()))
        ),
        "domain_conditioned_gains_vs_global": domain_vs_global,
        "mean_domain_conditioned_gain_vs_global": float(
            np.mean(list(domain_vs_global.values()))
        ),
    }
    return {"domains": domain_payload, "summary": summary}, analyzed


def main() -> None:
    args = parse_args()
    prediction_root = args.prediction_root.resolve()
    output_root = args.output_root.resolve()
    output_json = args.output_json.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    family_results: dict[str, Any] = {}
    for family in FAMILIES:
        family_frame = pd.concat(
            [
                _load_ensemble(prediction_root, family, domain)
                for domain in DOMAINS
            ],
            ignore_index=True,
        )
        result, analyzed = _analyze_family(
            family_frame,
            bootstrap_iterations=args.bootstrap_iterations,
            seed=args.seed,
        )
        family_results[family] = result
        analyzed.to_csv(output_root / f"{family}_calibration_oof.csv", index=False)

    primary = family_results["domain_class_event"]
    control = family_results["event_random"]
    primary_domain_gains = primary["summary"][
        "domain_conditioned_gains_vs_fixed"
    ]
    primary_gate = {
        "auroc_each_domain_at_least_0_70": bool(
            all(
                primary["domains"][domain]["ranking"]["auroc"] >= 0.7
                for domain in DOMAINS
            )
        ),
        "domain_gain_each_at_least_0_01": bool(
            all(primary_domain_gains[domain] >= 0.01 for domain in DOMAINS)
        ),
        "mean_domain_gain_at_least_0_015": bool(
            primary["summary"]["mean_domain_conditioned_gain_vs_fixed"] >= 0.015
        ),
        "bootstrap_probability_each_domain_at_least_0_90": bool(
            all(
                primary["domains"][domain]["strategies"]["domain_conditioned"][
                    "paired_patient_bootstrap_vs_fixed"
                ]["probability_a_greater_than_b"]
                >= 0.9
                for domain in DOMAINS
            )
        ),
        "sensitivity_specificity_each_domain_at_least_0_50": bool(
            all(
                min(
                    primary["domains"][domain]["strategies"][
                        "domain_conditioned"
                    ]["event_metrics"]["sensitivity"],
                    primary["domains"][domain]["strategies"][
                        "domain_conditioned"
                    ]["event_metrics"]["specificity"],
                )
                >= 0.5
                for domain in DOMAINS
            )
        ),
        "mean_gain_over_global_at_least_0_005": bool(
            primary["summary"]["mean_domain_conditioned_gain_vs_global"]
            >= 0.005
        ),
    }
    primary_gate["passed"] = bool(all(primary_gate.values()))
    control_gains = control["summary"]["domain_conditioned_gains_vs_fixed"]
    control_gate = {
        "mean_domain_gain_nonnegative": bool(
            control["summary"]["mean_domain_conditioned_gain_vs_fixed"] >= 0.0
        ),
        "no_domain_drop_greater_than_0_005": bool(
            all(gain >= -0.005 for gain in control_gains.values())
        ),
    }
    control_gate["passed"] = bool(all(control_gate.values()))
    gate_passed = bool(primary_gate["passed"] and control_gate["passed"])
    payload = {
        "status": "gate_7a0_passed" if gate_passed else "gate_7a0_rejected",
        "data_role": "calibration",
        "additional_training": False,
        "validation_select_accessed": False,
        "locked_tests_accessed": False,
        "families": family_results,
        "gate": {
            "primary_family": "domain_class_event",
            "primary": primary_gate,
            "robustness_control_family": "event_random",
            "robustness_control": control_gate,
            "passed": gate_passed,
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
