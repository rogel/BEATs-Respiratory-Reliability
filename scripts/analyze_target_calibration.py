#!/usr/bin/env python3
"""Fit target-patient calibration and evaluate it on disjoint target patients."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from respiratory_sound.calibration import (
    fit_positive_platt,
    fit_temperature,
    probabilities_to_logits,
    probability_metrics,
    select_balanced_threshold,
    sigmoid,
)
from respiratory_sound.metrics import (
    paired_patient_bootstrap_difference,
    patient_bootstrap_intervals,
    respiratory_metrics,
)

CLASS_NAMES = ("normal", "adventitious")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-predictions", type=Path, required=True)
    parser.add_argument("--evaluation-predictions", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20_260_728)
    return parser.parse_args()


def _load_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"patient_id": str})
    required = {"sample_id", "patient_id", "target", "probability_1"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if frame["sample_id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate sample_id values")
    if set(frame["target"].unique()) != {0, 1}:
        raise ValueError(f"{path} must contain both binary classes")
    return frame


def _classification_payload(
    targets: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    predictions = (probabilities >= threshold).astype(np.int64)
    payload = respiratory_metrics(targets, predictions, CLASS_NAMES)
    payload.update(
        {
            "threshold": float(threshold),
            "auroc": float(roc_auc_score(targets, probabilities)),
            "auprc": float(average_precision_score(targets, probabilities)),
        }
    )
    return payload


def main() -> None:
    args = parse_args()
    calibration = _load_predictions(args.calibration_predictions.resolve())
    evaluation = _load_predictions(args.evaluation_predictions.resolve())
    overlapping_patients = sorted(
        set(calibration["patient_id"]).intersection(evaluation["patient_id"])
    )
    if overlapping_patients:
        raise ValueError(
            "Calibration/evaluation patient overlap: "
            + ", ".join(overlapping_patients[:10])
        )

    calibration_targets = calibration["target"].to_numpy(dtype=np.int64)
    evaluation_targets = evaluation["target"].to_numpy(dtype=np.int64)
    calibration_raw = calibration["probability_1"].to_numpy(dtype=np.float64)
    evaluation_raw = evaluation["probability_1"].to_numpy(dtype=np.float64)

    temperature = fit_temperature(calibration_targets, calibration_raw)
    platt = fit_positive_platt(calibration_targets, calibration_raw)
    calibration_temperature = temperature.transform(calibration_raw)
    evaluation_temperature = temperature.transform(evaluation_raw)
    calibration_platt = platt.transform(calibration_raw)
    evaluation_platt = platt.transform(evaluation_raw)
    if not np.array_equal(calibration_raw >= 0.5, calibration_temperature >= 0.5):
        raise RuntimeError("Temperature scaling unexpectedly changed predictions")
    if not np.array_equal(evaluation_raw >= 0.5, evaluation_temperature >= 0.5):
        raise RuntimeError("Temperature scaling unexpectedly changed predictions")

    threshold, calibration_threshold_score = select_balanced_threshold(
        calibration_targets,
        calibration_platt,
    )
    raw_logit_threshold = (
        probabilities_to_logits(np.asarray((threshold,)))[0] - platt.intercept
    ) / platt.slope
    equivalent_raw_threshold = float(sigmoid(np.asarray((raw_logit_threshold,)))[0])

    raw_evaluation_predictions = (evaluation_raw >= 0.5).astype(np.int64)
    calibrated_evaluation_predictions = (evaluation_platt >= threshold).astype(np.int64)
    patient_ids = evaluation["patient_id"].to_numpy()
    fixed_threshold_intervals = patient_bootstrap_intervals(
        evaluation_targets,
        calibrated_evaluation_predictions,
        patient_ids,
        iterations=args.bootstrap_iterations,
        seed=args.seed,
        class_names=CLASS_NAMES,
    )
    paired_difference = paired_patient_bootstrap_difference(
        evaluation_targets,
        calibrated_evaluation_predictions,
        raw_evaluation_predictions,
        patient_ids,
        metric="average_score",
        iterations=args.bootstrap_iterations,
        seed=args.seed,
        class_names=CLASS_NAMES,
    )

    calibration_probability = {
        "raw": probability_metrics(calibration_targets, calibration_raw),
        "temperature": probability_metrics(
            calibration_targets,
            calibration_temperature,
        ),
        "platt": probability_metrics(calibration_targets, calibration_platt),
    }
    evaluation_probability = {
        "raw": probability_metrics(evaluation_targets, evaluation_raw),
        "temperature": probability_metrics(
            evaluation_targets,
            evaluation_temperature,
        ),
        "platt": probability_metrics(evaluation_targets, evaluation_platt),
    }
    evaluation_classification = {
        "raw": _classification_payload(evaluation_targets, evaluation_raw, 0.5),
        "temperature": _classification_payload(
            evaluation_targets,
            evaluation_temperature,
            0.5,
        ),
        "platt_fixed_operating_point": _classification_payload(
            evaluation_targets,
            evaluation_platt,
            threshold,
        ),
    }
    calibrated_metrics = evaluation_classification["platt_fixed_operating_point"]
    direction_gate = {
        "average_score_at_least_0_60": bool(
            calibrated_metrics["average_score"] >= 0.6
        ),
        "sensitivity_at_least_0_50": bool(
            calibrated_metrics["sensitivity"] >= 0.5
        ),
        "specificity_at_least_0_50": bool(
            calibrated_metrics["specificity"] >= 0.5
        ),
        "platt_nll_improved": bool(
            evaluation_probability["platt"]["negative_log_likelihood"]
            < evaluation_probability["raw"]["negative_log_likelihood"]
        ),
        "platt_brier_improved": bool(
            evaluation_probability["platt"]["brier_score"]
            < evaluation_probability["raw"]["brier_score"]
        ),
    }
    direction_gate["passed"] = bool(all(direction_gate.values()))

    payload = {
        "calibration_predictions": str(args.calibration_predictions),
        "evaluation_predictions": str(args.evaluation_predictions),
        "patient_overlap": overlapping_patients,
        "calibration_samples": int(len(calibration)),
        "calibration_patients": int(calibration["patient_id"].nunique()),
        "evaluation_samples": int(len(evaluation)),
        "evaluation_patients": int(evaluation["patient_id"].nunique()),
        "fitted_parameters": {
            "temperature": temperature.temperature,
            "platt_slope": platt.slope,
            "platt_intercept": platt.intercept,
            "platt_probability_threshold": threshold,
            "equivalent_raw_probability_threshold": equivalent_raw_threshold,
            "calibration_threshold_average_score": calibration_threshold_score,
        },
        "calibration_probability_metrics": calibration_probability,
        "evaluation_probability_metrics": evaluation_probability,
        "evaluation_classification_metrics": evaluation_classification,
        "fixed_threshold_patient_bootstrap": fixed_threshold_intervals,
        "paired_average_score_difference_vs_raw": paired_difference,
        "direction_gate": direction_gate,
        "locked_test_accessed": False,
    }

    output_prefix = args.output_prefix.resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    output_frame = evaluation[
        ["sample_id", "patient_id", "target", "probability_1"]
    ].rename(columns={"probability_1": "raw_probability"})
    output_frame["temperature_probability"] = evaluation_temperature
    output_frame["platt_probability"] = evaluation_platt
    output_frame["raw_prediction"] = raw_evaluation_predictions
    output_frame["calibrated_prediction"] = calibrated_evaluation_predictions
    output_frame.to_csv(output_prefix.with_suffix(".csv"), index=False)
    output_prefix.with_suffix(".json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
