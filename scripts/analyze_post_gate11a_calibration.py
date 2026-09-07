#!/usr/bin/env python3
"""Fit the predeclared Gate-11A temperature and selective-prediction rules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from respiratory_sound.calibration import fit_temperature, probability_metrics
from respiratory_sound.gate11a import CLASS_NAMES, sha256_file
from respiratory_sound.metrics import respiratory_metrics
from respiratory_sound.post_gate11a import (
    DOMAINS,
    FULLFT_RUNS,
    probability_ensemble,
)
from respiratory_sound.selective_prediction import (
    equal_frequency_reliability_table,
    fit_uncertainty_cutoff,
    normalized_binary_entropy,
    risk_coverage_curve,
    selective_operating_point,
)

TARGET_COVERAGES = (0.80, 0.90)
ECE_BINS = 15
CLASS_COVERAGE_TOLERANCE = 0.15
CLASS_COVERAGE_GAP_MAXIMUM = 0.20
SELECTIVE_RISK_TOLERANCE = 0.02
NLL_DOMAIN_WORSENING_TOLERANCE = 0.02


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--prediction-root",
        type=Path,
        default=Path("artifacts/post_gate11a/calibration_predictions"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/post_gate11a/calibration_analysis"),
    )
    return parser.parse_args()


def _classification(targets: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    predictions = (probabilities >= 0.5).astype(np.int64)
    result = respiratory_metrics(targets, predictions, CLASS_NAMES)
    result["auroc"] = float(roc_auc_score(targets, probabilities))
    return result


def _load_calibration(prediction_root: Path, domain: str) -> pd.DataFrame:
    path = prediction_root / f"fullft_ensemble_{domain}_calibration.csv"
    frame = pd.read_csv(path, dtype={"patient_id": str})
    if set(frame["protocol_role"].astype(str)) != {"calibration"}:
        raise ValueError("Calibration predictions crossed the role boundary")
    if frame["locked"].astype(str).str.lower().isin({"true", "1"}).any():
        raise ValueError("Calibration predictions contain locked rows")
    return frame


def _load_validation(root: Path, domain: str) -> pd.DataFrame:
    frames = [
        pd.read_csv(
            root / run / f"best_validation_predictions_{domain}.csv",
            dtype={"patient_id": str},
        )
        for run in FULLFT_RUNS.values()
    ]
    frame = probability_ensemble(frames)
    if set(frame["protocol_role"].astype(str)) != {"validation_select"}:
        raise ValueError("Validation predictions crossed the role boundary")
    return frame


def _safety_checks(
    point: dict[str, float | int],
    *,
    target: float,
    full_risk: float,
) -> dict[str, bool]:
    return {
        "normal_coverage_at_least_target_minus_0_15": bool(
            float(point["normal_coverage"]) >= target - CLASS_COVERAGE_TOLERANCE
        ),
        "adventitious_coverage_at_least_target_minus_0_15": bool(
            float(point["adventitious_coverage"])
            >= target - CLASS_COVERAGE_TOLERANCE
        ),
        "class_coverage_gap_at_most_0_20": bool(
            float(point["class_coverage_gap"]) <= CLASS_COVERAGE_GAP_MAXIMUM
        ),
        "selective_error_not_worse_by_more_than_0_02": bool(
            float(point["selective_error_risk"])
            <= full_risk + SELECTIVE_RISK_TOLERANCE
        ),
    }


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    prediction_root = (root / args.prediction_root).resolve()
    output_root = (root / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    freeze_path = root / "artifacts/post_gate11a_development_freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("status") != "sealed_before_calibration":
        raise ValueError("Post-Gate-11A development freeze is invalid")

    domains: dict[str, Any] = {}
    validation_nll_deltas = []
    temperature_predictions_unchanged = True
    all_selective_checks: list[bool] = []
    for domain in DOMAINS:
        calibration = _load_calibration(prediction_root, domain)
        validation = _load_validation(root, domain)
        calibration_targets = calibration["target"].to_numpy(dtype=np.int64)
        validation_targets = validation["target"].to_numpy(dtype=np.int64)
        calibration_raw = calibration["probability_1"].to_numpy(dtype=np.float64)
        validation_raw = validation["probability_1"].to_numpy(dtype=np.float64)
        temperature = fit_temperature(calibration_targets, calibration_raw)
        calibration_scaled = temperature.transform(calibration_raw)
        validation_scaled = temperature.transform(validation_raw)
        unchanged = bool(
            np.array_equal(calibration_raw >= 0.5, calibration_scaled >= 0.5)
            and np.array_equal(validation_raw >= 0.5, validation_scaled >= 0.5)
        )
        temperature_predictions_unchanged &= unchanged
        probability = {
            "calibration": {
                "raw": probability_metrics(
                    calibration_targets, calibration_raw, bins=ECE_BINS
                ),
                "temperature": probability_metrics(
                    calibration_targets, calibration_scaled, bins=ECE_BINS
                ),
            },
            "validation_select": {
                "raw": probability_metrics(
                    validation_targets, validation_raw, bins=ECE_BINS
                ),
                "temperature": probability_metrics(
                    validation_targets, validation_scaled, bins=ECE_BINS
                ),
            },
        }
        delta = (
            probability["validation_select"]["temperature"]["negative_log_likelihood"]
            - probability["validation_select"]["raw"]["negative_log_likelihood"]
        )
        validation_nll_deltas.append(float(delta))

        selective: dict[str, Any] = {}
        calibration_uncertainty = normalized_binary_entropy(calibration_scaled)
        for target in TARGET_COVERAGES:
            cutoff = fit_uncertainty_cutoff(calibration_uncertainty, target)
            calibration_point = selective_operating_point(
                calibration_targets, calibration_scaled, cutoff
            )
            validation_point = selective_operating_point(
                validation_targets, validation_scaled, cutoff
            )
            calibration_full_risk = float(
                ((calibration_scaled >= 0.5) != calibration_targets).mean()
            )
            validation_full_risk = float(
                ((validation_scaled >= 0.5) != validation_targets).mean()
            )
            checks = {
                "calibration": _safety_checks(
                    calibration_point,
                    target=target,
                    full_risk=calibration_full_risk,
                ),
                "validation_select": _safety_checks(
                    validation_point,
                    target=target,
                    full_risk=validation_full_risk,
                ),
            }
            passed = all(value for split in checks.values() for value in split.values())
            all_selective_checks.append(bool(passed))
            selective[f"coverage_{int(target * 100)}"] = {
                "target": target,
                "uncertainty_cutoff": cutoff,
                "calibration": calibration_point,
                "validation_select": validation_point,
                "checks": checks,
                "passed": bool(passed),
            }

        reliability = {
            "calibration_raw": equal_frequency_reliability_table(
                calibration_targets, calibration_raw, bins=ECE_BINS
            ),
            "calibration_temperature": equal_frequency_reliability_table(
                calibration_targets, calibration_scaled, bins=ECE_BINS
            ),
            "validation_raw": equal_frequency_reliability_table(
                validation_targets, validation_raw, bins=ECE_BINS
            ),
            "validation_temperature": equal_frequency_reliability_table(
                validation_targets, validation_scaled, bins=ECE_BINS
            ),
        }
        for name, rows in reliability.items():
            pd.DataFrame(rows).to_csv(output_root / f"{domain}_{name}.csv", index=False)
        risk_summary, risk_curve = risk_coverage_curve(
            validation_targets, validation_scaled
        )
        pd.DataFrame(
            risk_curve,
            columns=("coverage", "error_risk", "uncertainty"),
        ).to_csv(output_root / f"{domain}_risk_coverage.csv", index=False)
        domains[domain] = {
            "calibration_samples": int(len(calibration)),
            "calibration_patients": int(calibration["patient_id"].nunique()),
            "validation_samples": int(len(validation)),
            "validation_patients": int(validation["patient_id"].nunique()),
            "temperature": temperature.temperature,
            "classification_predictions_unchanged": unchanged,
            "classification": {
                "calibration_raw": _classification(calibration_targets, calibration_raw),
                "calibration_temperature": _classification(
                    calibration_targets, calibration_scaled
                ),
                "validation_raw": _classification(validation_targets, validation_raw),
                "validation_temperature": _classification(
                    validation_targets, validation_scaled
                ),
            },
            "probability_metrics": probability,
            "validation_temperature_minus_raw_nll": float(delta),
            "selective": selective,
            "risk_coverage": risk_summary,
        }

    temperature_checks = {
        "classification_predictions_unchanged": temperature_predictions_unchanged,
        "mean_validation_nll_not_worse": bool(np.mean(validation_nll_deltas) <= 0.0),
        "no_domain_validation_nll_worsens_more_than_0_02": bool(
            max(validation_nll_deltas) <= NLL_DOMAIN_WORSENING_TOLERANCE
        ),
    }
    temperature_enabled = bool(all(temperature_checks.values()))
    selective_enabled = bool(temperature_enabled and all(all_selective_checks))
    payload = {
        "stage": "post_gate11a_calibration_and_selective_sanity",
        "development_freeze": str(freeze_path.relative_to(root)),
        "development_freeze_sha256": sha256_file(freeze_path),
        "primary_classifier": "equal-weight three-seed full-FT probability ensemble",
        "classification_threshold": 0.5,
        "ece_bins": ECE_BINS,
        "uncertainty_score": "normalized binary predictive entropy",
        "target_coverages": list(TARGET_COVERAGES),
        "domains": domains,
        "decision": {
            "temperature_checks": temperature_checks,
            "temperature_enabled_for_locked": temperature_enabled,
            "selective_all_domain_target_checks_passed": bool(
                all(all_selective_checks)
            ),
            "selective_enabled_for_locked": selective_enabled,
            "failure_action": (
                "do not search alternative calibration/rejection methods; "
                "fall back to raw fixed-threshold classification and report negative result"
            ),
        },
        "locked_parameters": {
            domain: {
                "temperature": (
                    float(domains[domain]["temperature"])
                    if temperature_enabled
                    else 1.0
                ),
                "selective_enabled": selective_enabled,
                "uncertainty_cutoffs": {
                    key: float(value["uncertainty_cutoff"])
                    for key, value in domains[domain]["selective"].items()
                }
                if selective_enabled
                else {},
            }
            for domain in DOMAINS
        },
        "calibration_history_disclosure": (
            "These calibration patients were used in earlier abandoned routes and are "
            "treated only as tuning/calibration data, never as independent evidence."
        ),
        "calibration_accessed": True,
        "locked_tests_accessed": False,
    }
    output = output_root / "post_gate11a_calibration_final.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
