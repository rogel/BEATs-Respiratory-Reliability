#!/usr/bin/env python3
"""Analyze the single frozen locked-data access episode after Gate 11A."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from respiratory_sound.calibration import TemperatureScaler, probability_metrics
from respiratory_sound.gate11a import CLASS_NAMES, sha256_file
from respiratory_sound.metrics import (
    paired_patient_bootstrap_difference,
    patient_bootstrap_intervals,
    respiratory_metrics,
)
from respiratory_sound.post_gate11a import SEEDS
from respiratory_sound.selective_prediction import risk_coverage_curve

PRIMARY = {
    "icbhi2017": ("icbhi2017", "locked_test"),
    "sprsound2022": ("sprsound2022", "locked_inter_test"),
}
DESCRIPTIVE = {
    "sprsound2022_intra": ("sprsound2022", "locked_intra_test"),
}
ITERATIONS = 4000
BOOTSTRAP_SEED = 20_260_729
CONFIDENCE = 0.95


def _load(root: Path, family: str, domain: str, role: str, member: str) -> pd.DataFrame:
    path = root / f"{family}_{member}_{domain}_{role}.csv"
    frame = pd.read_csv(path, dtype={"patient_id": str})
    if set(frame["dataset"].astype(str)) != {domain}:
        raise ValueError(f"Domain mismatch in {path}")
    if set(frame["protocol_role"].astype(str)) != {role}:
        raise ValueError(f"Role mismatch in {path}")
    if not frame["locked"].astype(str).str.lower().isin({"true", "1"}).all():
        raise ValueError(f"Non-locked row in {path}")
    if frame["sample_id"].duplicated().any():
        raise ValueError(f"Duplicate sample in {path}")
    if not np.array_equal(frame["binary_label_id"].astype(int), frame["target"].astype(int)):
        raise ValueError(f"Target mismatch in {path}")
    return frame.sort_values("sample_id").reset_index(drop=True)


def _metrics(frame: pd.DataFrame, probabilities: np.ndarray | None = None) -> dict[str, Any]:
    targets = frame["target"].to_numpy(dtype=np.int64)
    values = (
        frame["probability_1"].to_numpy(dtype=np.float64)
        if probabilities is None
        else np.asarray(probabilities, dtype=np.float64)
    )
    predictions = (values >= 0.5).astype(np.int64)
    result = respiratory_metrics(targets, predictions, CLASS_NAMES)
    result["auroc"] = float(roc_auc_score(targets, values))
    result.update(probability_metrics(targets, values, bins=15))
    return result


def _paired_mean_bootstrap(
    fullft: dict[str, pd.DataFrame],
    lora: dict[str, pd.DataFrame],
) -> dict[str, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    prepared: dict[str, dict[str, Any]] = {}
    for key in PRIMARY:
        left = fullft[key]
        right = lora[key]
        if not left[["sample_id", "patient_id", "target"]].equals(
            right[["sample_id", "patient_id", "target"]]
        ):
            raise ValueError("Full-FT and LoRA locked tables are not paired")
        patients = left["patient_id"].astype(str).to_numpy()
        unique = np.unique(patients)
        prepared[key] = {
            "left": left,
            "right": right,
            "patients": unique,
            "indices": {patient: np.flatnonzero(patients == patient) for patient in unique},
        }

    def gain(key: str, indices: np.ndarray) -> float:
        item = prepared[key]
        targets = item["left"]["target"].to_numpy(dtype=int)[indices]
        left = (
            item["left"]["probability_1"].to_numpy(dtype=float)[indices] >= 0.5
        ).astype(int)
        right = (
            item["right"]["probability_1"].to_numpy(dtype=float)[indices] >= 0.5
        ).astype(int)
        return float(
            respiratory_metrics(targets, left, CLASS_NAMES)["average_score"]
            - respiratory_metrics(targets, right, CLASS_NAMES)["average_score"]
        )

    point = float(
        np.mean(
            [gain(key, np.arange(len(prepared[key]["left"]))) for key in PRIMARY]
        )
    )
    samples = []
    for _ in range(ITERATIONS):
        gains = []
        for key in PRIMARY:
            item = prepared[key]
            sampled = rng.choice(item["patients"], size=len(item["patients"]), replace=True)
            indices = np.concatenate([item["indices"][patient] for patient in sampled])
            gains.append(gain(key, indices))
        samples.append(float(np.mean(gains)))
    values = np.asarray(samples)
    alpha = (1.0 - CONFIDENCE) / 2.0
    return {
        "estimate": point,
        "lower": float(np.quantile(values, alpha)),
        "upper": float(np.quantile(values, 1.0 - alpha)),
        "probability_fullft_greater_than_lora": float(np.mean(values > 0)),
    }


def main() -> None:
    root = Path.cwd().resolve()
    prediction_root = root / "artifacts/post_gate11a/locked_predictions"
    output = root / "artifacts/post_gate11a/locked_final.json"
    if output.exists():
        raise FileExistsError(f"Locked analysis already exists: {output}")
    freeze_path = root / "artifacts/post_gate11a_locked_freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("status") != "sealed_before_locked_access":
        raise ValueError("Final locked freeze is invalid")
    access_path = root / "artifacts/post_gate11a_locked_access.json"
    access = json.loads(access_path.read_text(encoding="utf-8"))
    if access.get("status") != "predictions_complete":
        raise ValueError("Locked predictions are not complete")
    calibration_path = (
        root
        / "artifacts/post_gate11a/calibration_analysis/post_gate11a_calibration_final.json"
    )
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))

    all_specs = {**PRIMARY, **DESCRIPTIVE}
    ensembles: dict[str, dict[str, pd.DataFrame]] = {"fullft": {}, "lora": {}}
    family_results: dict[str, Any] = {"fullft": {}, "lora": {}}
    for family in ("fullft", "lora"):
        for key, (domain, role) in all_specs.items():
            seed_frames = [
                _load(prediction_root, family, domain, role, f"seed{seed}")
                for seed in SEEDS
            ]
            ensemble = _load(prediction_root, family, domain, role, "ensemble")
            ensembles[family][key] = ensemble
            seed_metrics = [_metrics(frame) for frame in seed_frames]
            scores = np.asarray(
                [item["average_score"] for item in seed_metrics], dtype=np.float64
            )
            family_results[family][key] = {
                "role": role,
                "independent_primary": key in PRIMARY,
                "seed_metrics": {
                    str(seed): metrics
                    for seed, metrics in zip(SEEDS, seed_metrics, strict=True)
                },
                "seed_average_score_mean": float(scores.mean()),
                "seed_average_score_sample_sd": float(scores.std(ddof=1)),
                "ensemble_metrics": _metrics(ensemble),
            }

    primary_reliability: dict[str, Any] = {}
    absolute_bootstrap: dict[str, Any] = {}
    paired_bootstrap: dict[str, Any] = {}
    nll_deltas = []
    for key, (domain, _) in PRIMARY.items():
        full = ensembles["fullft"][key]
        lora = ensembles["lora"][key]
        targets = full["target"].to_numpy(dtype=int)
        raw = full["probability_1"].to_numpy(dtype=float)
        temperature = float(calibration["locked_parameters"][domain]["temperature"])
        scaled = TemperatureScaler(temperature).transform(raw)
        raw_probability = probability_metrics(targets, raw, bins=15)
        scaled_probability = probability_metrics(targets, scaled, bins=15)
        nll_delta = (
            scaled_probability["negative_log_likelihood"]
            - raw_probability["negative_log_likelihood"]
        )
        nll_deltas.append(float(nll_delta))
        curve_summary, curve = risk_coverage_curve(targets, scaled)
        pd.DataFrame(
            curve, columns=("coverage", "error_risk", "uncertainty")
        ).to_csv(
            root / f"artifacts/post_gate11a/{key}_locked_risk_coverage.csv",
            index=False,
        )
        primary_reliability[key] = {
            "temperature": temperature,
            "raw_probability_metrics": raw_probability,
            "temperature_probability_metrics": scaled_probability,
            "temperature_minus_raw_nll": float(nll_delta),
            "classification_predictions_unchanged": bool(
                np.array_equal(raw >= 0.5, scaled >= 0.5)
            ),
            "risk_coverage_descriptive_only": curve_summary,
            "selective_operating_claim_enabled": False,
        }
        predictions = (raw >= 0.5).astype(int)
        lora_predictions = (lora["probability_1"].to_numpy(dtype=float) >= 0.5).astype(int)
        patients = full["patient_id"].astype(str).to_numpy()
        absolute_bootstrap[key] = patient_bootstrap_intervals(
            targets,
            predictions,
            patients,
            iterations=ITERATIONS,
            seed=BOOTSTRAP_SEED,
            confidence=CONFIDENCE,
            class_names=CLASS_NAMES,
        )
        paired_bootstrap[key] = paired_patient_bootstrap_difference(
            targets,
            predictions,
            lora_predictions,
            patients,
            metric="average_score",
            iterations=ITERATIONS,
            seed=BOOTSTRAP_SEED,
            confidence=CONFIDENCE,
            class_names=CLASS_NAMES,
        )
    paired_bootstrap["two_domain_mean"] = _paired_mean_bootstrap(
        ensembles["fullft"], ensembles["lora"]
    )

    full_scores = {
        key: float(family_results["fullft"][key]["ensemble_metrics"]["average_score"])
        for key in PRIMARY
    }
    full_metrics = {
        key: family_results["fullft"][key]["ensemble_metrics"] for key in PRIMARY
    }
    locked_checks = {
        "icbhi_average_score_at_least_0_64": bool(
            full_scores["icbhi2017"] >= 0.64
        ),
        "sprsound_inter_average_score_at_least_0_82": (
            bool(full_scores["sprsound2022"] >= 0.82)
        ),
        "two_domain_mean_at_least_0_75": bool(
            np.mean(list(full_scores.values())) >= 0.75
        ),
        "sensitivity_and_specificity_each_at_least_0_50": all(
            float(full_metrics[key][metric]) >= 0.50
            for key in PRIMARY
            for metric in ("sensitivity", "specificity")
        ),
    }
    calibration_claim_checks = {
        "classification_predictions_unchanged": all(
            value["classification_predictions_unchanged"]
            for value in primary_reliability.values()
        ),
        "mean_locked_nll_not_worse": bool(np.mean(nll_deltas) <= 0.0),
        "no_domain_locked_nll_worsens_more_than_0_02": bool(max(nll_deltas) <= 0.02),
    }
    payload = {
        "stage": "post_gate11a_one_time_locked_evaluation",
        "status": (
            "locked_robustness_passed"
            if all(locked_checks.values())
            else "locked_robustness_failed"
        ),
        "locked_freeze": str(freeze_path.relative_to(root)),
        "locked_freeze_sha256": sha256_file(freeze_path),
        "prediction_manifest_sha256": sha256_file(
            prediction_root / "prediction_manifest.json"
        ),
        "families": family_results,
        "primary_fullft_reliability": primary_reliability,
        "patient_cluster_absolute_bootstrap": absolute_bootstrap,
        "paired_fullft_minus_lora_bootstrap": paired_bootstrap,
        "locked_robustness_checks": locked_checks,
        "calibration_claim_checks": calibration_claim_checks,
        "calibration_claim_supported_on_locked": bool(
            all(calibration_claim_checks.values())
        ),
        "selective_claim_supported": False,
        "sprsound_locked_intra_interpretation": (
            "descriptive same-patient result only; not independent generalization evidence"
        ),
        "bootstrap": {
            "iterations": ITERATIONS,
            "seed": BOOTSTRAP_SEED,
            "confidence": CONFIDENCE,
            "cluster": "patient",
        },
        "calibration_accessed": True,
        "locked_tests_accessed": True,
        "no_return_rule": (
            "These results may not be used to alter or rerun any model, threshold, "
            "temperature, rejection rule, or method."
        ),
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    access["status"] = "analysis_complete"
    access["locked_result_sha256"] = sha256_file(output)
    access_path.write_text(json.dumps(access, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
