#!/usr/bin/env python3
"""Analyze all frozen BEATS revision-stage model comparisons."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score

from respiratory_sound.calibration import (
    TemperatureScaler,
    fit_temperature,
    probabilities_to_logits,
    probability_metrics,
)
from respiratory_sound.gate11a import ALLOWED_SEEDS, CLASS_NAMES, sha256_file
from respiratory_sound.metrics import respiratory_metrics
from respiratory_sound.post_gate11a import probability_ensemble

ITERATIONS = 4_000
BOOTSTRAP_SEED = 20_260_729
CONFIDENCE = 0.95
FAMILIES = ("exact", "legacy-mask", "no-js", "matched-lora")
PRIMARY = {
    "icbhi2017": "locked_test",
    "sprsound2022": "locked_inter_test",
}
ROLES = {
    "icbhi2017": ("validation_select", "calibration", "locked_test"),
    "sprsound2022": (
        "validation_select",
        "calibration",
        "locked_inter_test",
        "locked_intra_test",
    ),
}
CONTRASTS = {
    "exact_minus_legacy": ("exact", "legacy-mask"),
    "js_on_minus_no_js": ("exact", "no-js"),
    "fullft_minus_matched_lora": ("exact", "matched-lora"),
}


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _metadata(manifest: pd.DataFrame, domain: str, role: str) -> pd.DataFrame:
    columns = [
        "sample_id",
        "dataset",
        "patient_id",
        "protocol_role",
        "locked",
        "binary_label_id",
        "fine_label_name",
        "event_duration_seconds",
    ]
    frame = manifest.loc[
        manifest["dataset"].astype(str).eq(domain)
        & manifest["protocol_role"].astype(str).eq(role),
        columns,
    ].copy()
    if frame.empty or frame["sample_id"].duplicated().any():
        raise ValueError(f"Invalid manifest role: {domain}/{role}")
    return frame


def _exact_path(root: Path, seed: int, domain: str, role: str) -> Path:
    if role == "validation_select":
        return (
            root
            / f"runs/gate11a_exactmask_fullft_seed{seed}"
            / f"best_validation_predictions_{domain}.csv"
        )
    if role == "calibration":
        return (
            root
            / "artifacts/post_gate11a/calibration_predictions"
            / f"fullft_seed{seed}_{domain}_{role}.csv"
        )
    return (
        root
        / "artifacts/post_gate11a/locked_predictions"
        / f"fullft_seed{seed}_{domain}_{role}.csv"
    )


def _revision_path(root: Path, family: str, seed: int, domain: str, role: str) -> Path:
    return (
        root
        / "artifacts/revision_2026_09_03/predictions"
        / family.replace("-", "_")
        / f"seed{seed}_{domain}_{role}.csv"
    )


def _load_seed(
    root: Path,
    manifest: pd.DataFrame,
    *,
    family: str,
    seed: int,
    domain: str,
    role: str,
) -> tuple[pd.DataFrame, str]:
    path = (
        _exact_path(root, seed, domain, role)
        if family == "exact"
        else _revision_path(root, family, seed, domain, role)
    )
    predictions = pd.read_csv(path, dtype={"patient_id": str})
    if predictions["sample_id"].duplicated().any():
        raise ValueError(f"Duplicate prediction sample: {path}")
    base = _metadata(manifest, domain, role)
    required = ["sample_id", "target", "probability_1"]
    merged = base.merge(
        predictions[required],
        on="sample_id",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(base) or len(merged) != len(predictions):
        raise ValueError(f"Prediction coverage mismatch: {path}")
    if not np.array_equal(
        merged["binary_label_id"].to_numpy(dtype=int),
        merged["target"].to_numpy(dtype=int),
    ):
        raise ValueError(f"Prediction target mismatch: {path}")
    probabilities = merged["probability_1"].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or not (
        (probabilities >= 0.0) & (probabilities <= 1.0)
    ).all():
        raise ValueError(f"Invalid probabilities: {path}")
    merged["probability_0"] = 1.0 - probabilities
    merged["prediction"] = (probabilities >= 0.5).astype(int)
    return merged.sort_values("sample_id").reset_index(drop=True), sha256_file(path)


def _classification(frame: pd.DataFrame, probabilities: np.ndarray) -> dict[str, Any]:
    targets = frame["target"].to_numpy(dtype=int)
    predictions = (probabilities >= 0.5).astype(int)
    result = respiratory_metrics(targets, predictions, CLASS_NAMES)
    result["auroc"] = float(roc_auc_score(targets, probabilities))
    result["predicted_adventitious_fraction"] = float(predictions.mean())
    return result


def _calibration_regression(
    targets: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    labels = np.asarray(targets, dtype=float)
    logits = probabilities_to_logits(probabilities)
    design = np.column_stack((np.ones(len(logits)), logits))
    beta = np.asarray((0.0, 1.0), dtype=float)
    for _ in range(100):
        fitted = expit(design @ beta)
        weights = np.maximum(fitted * (1.0 - fitted), 1.0e-9)
        gradient = design.T @ (labels - fitted)
        information = (design.T * weights) @ design
        information.flat[::3] += 1.0e-9
        step = np.linalg.solve(information, gradient)
        beta += step
        if float(np.max(np.abs(step))) < 1.0e-10:
            break
    if not np.isfinite(beta).all():
        raise FloatingPointError("Calibration regression became non-finite")
    return {"calibration_intercept": float(beta[0]), "calibration_slope": float(beta[1])}


def _probability_summary(
    targets: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    result = probability_metrics(targets, probabilities, bins=15)
    result.update(_calibration_regression(targets, probabilities))
    return result


def _patient_resamples(
    patient_ids: np.ndarray,
    generator: np.random.Generator | None = None,
) -> list[np.ndarray]:
    patients = np.asarray(patient_ids, dtype=str)
    unique = np.unique(patients)
    indices = {patient: np.flatnonzero(patients == patient) for patient in unique}
    if generator is None:
        generator = np.random.default_rng(BOOTSTRAP_SEED)
    return [
        np.concatenate(
            [indices[patient] for patient in generator.choice(unique, len(unique), True)]
        )
        for _ in range(ITERATIONS)
    ]


def _interval(values: list[float], estimate: float) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    alpha = (1.0 - CONFIDENCE) / 2.0
    return {
        "estimate": float(estimate),
        "lower": float(np.quantile(array, alpha)),
        "upper": float(np.quantile(array, 1.0 - alpha)),
        "probability_greater_than_zero": float(np.mean(array > 0.0)),
    }


def _as(targets: np.ndarray, probabilities: np.ndarray) -> float:
    return float(
        respiratory_metrics(
            targets,
            (probabilities >= 0.5).astype(int),
            CLASS_NAMES,
        )["average_score"]
    )


def _paired_as_interval(
    frame: pd.DataFrame,
    left: np.ndarray,
    right: np.ndarray,
) -> dict[str, float]:
    targets = frame["target"].to_numpy(dtype=int)
    resamples = _patient_resamples(frame["patient_id"].to_numpy(dtype=str))
    estimate = _as(targets, left) - _as(targets, right)
    values = [
        _as(targets[index], left[index]) - _as(targets[index], right[index])
        for index in resamples
    ]
    return _interval(values, estimate)


def _paired_probability_intervals(
    frame: pd.DataFrame,
    left: np.ndarray,
    right: np.ndarray,
) -> dict[str, dict[str, float]]:
    targets = frame["target"].to_numpy(dtype=int)
    resamples = _patient_resamples(frame["patient_id"].to_numpy(dtype=str))
    left_point = _probability_summary(targets, left)
    right_point = _probability_summary(targets, right)
    metric_names = (
        "negative_log_likelihood",
        "brier_score",
        "equal_frequency_ece",
    )
    samples: dict[str, list[float]] = {name: [] for name in metric_names}
    for index in resamples:
        left_sample = _probability_summary(targets[index], left[index])
        right_sample = _probability_summary(targets[index], right[index])
        for name in metric_names:
            samples[name].append(left_sample[name] - right_sample[name])
    return {
        name: _interval(samples[name], left_point[name] - right_point[name])
        for name in metric_names
    }


def _threshold_rows(frame: pd.DataFrame, probabilities: np.ndarray) -> list[dict[str, Any]]:
    targets = frame["target"].to_numpy(dtype=int)
    rows = []
    for threshold in np.arange(0.10, 0.901, 0.05):
        predictions = (probabilities >= threshold).astype(int)
        metrics = respiratory_metrics(targets, predictions, CLASS_NAMES)
        rows.append(
            {
                "dataset": str(frame["dataset"].iloc[0]),
                "role": str(frame["protocol_role"].iloc[0]),
                "threshold": float(round(threshold, 2)),
                "sensitivity": metrics["sensitivity"],
                "specificity": metrics["specificity"],
                "average_score": metrics["average_score"],
                "macro_f1": metrics["macro_f1"],
                "prediction_prevalence": float(predictions.mean()),
                "normal_error": float(predictions[targets == 0].mean()),
                "adventitious_error": float((1 - predictions[targets == 1]).mean()),
            }
        )
    return rows


def main() -> None:
    root = Path.cwd().resolve()
    output_root = root / "artifacts/revision_2026_09_03/final_analysis"
    output_root.mkdir(parents=True, exist_ok=False)
    manifest_path = root / "data/manifests/cross_domain_binary.csv"
    manifest = pd.read_csv(manifest_path, dtype={"patient_id": str})
    ensembles: dict[str, dict[tuple[str, str], pd.DataFrame]] = {
        family: {} for family in FAMILIES
    }
    probability_by_family: dict[str, dict[tuple[str, str], np.ndarray]] = {
        family: {} for family in FAMILIES
    }
    prediction_hashes: dict[str, str] = {}
    classification_rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        for domain, roles in ROLES.items():
            for role in roles:
                seed_frames = []
                for seed in ALLOWED_SEEDS:
                    frame, digest = _load_seed(
                        root,
                        manifest,
                        family=family,
                        seed=seed,
                        domain=domain,
                        role=role,
                    )
                    seed_frames.append(frame)
                    prediction_hashes[f"{family}/{seed}/{domain}/{role}"] = digest
                    metrics = _classification(
                        frame,
                        frame["probability_1"].to_numpy(dtype=float),
                    )
                    classification_rows.append(
                        {
                            "family": family,
                            "member": f"seed{seed}",
                            "dataset": domain,
                            "role": role,
                            **{
                                key: value
                                for key, value in metrics.items()
                                if isinstance(value, float)
                            },
                        }
                    )
                ensemble = probability_ensemble(seed_frames)
                ensembles[family][(domain, role)] = ensemble
                probability_by_family[family][(domain, role)] = ensemble[
                    "probability_1"
                ].to_numpy(dtype=float)
                metrics = _classification(
                    ensemble,
                    probability_by_family[family][(domain, role)],
                )
                classification_rows.append(
                    {
                        "family": family,
                        "member": "ensemble",
                        "dataset": domain,
                        "role": role,
                        **{
                            key: value
                            for key, value in metrics.items()
                            if isinstance(value, float)
                        },
                    }
                )
    pd.DataFrame(classification_rows).to_csv(
        output_root / "classification_metrics.csv", index=False
    )

    protected = json.loads(
        (
            root
            / "artifacts/post_gate11a/calibration_analysis/post_gate11a_calibration_final.json"
        ).read_text(encoding="utf-8")
    )
    temperatures: dict[str, dict[str, float]] = {family: {} for family in FAMILIES}
    calibration_rows: list[dict[str, Any]] = []
    calibrated: dict[str, dict[tuple[str, str], np.ndarray]] = {
        family: {} for family in FAMILIES
    }
    raw_scaled_intervals: dict[str, Any] = {}
    for family in FAMILIES:
        for domain, roles in ROLES.items():
            calibration = ensembles[family][(domain, "calibration")]
            targets = calibration["target"].to_numpy(dtype=int)
            raw = probability_by_family[family][(domain, "calibration")]
            fitted = fit_temperature(targets, raw).temperature
            temperature = (
                float(protected["locked_parameters"][domain]["temperature"])
                if family == "exact"
                else fitted
            )
            temperatures[family][domain] = temperature
            if family == "exact" and not np.isclose(fitted, temperature, atol=1.0e-8):
                raise ValueError("Refitted exact temperature differs from protected value")
            scaler = TemperatureScaler(temperature)
            for role in roles:
                frame = ensembles[family][(domain, role)]
                role_targets = frame["target"].to_numpy(dtype=int)
                role_raw = probability_by_family[family][(domain, role)]
                role_scaled = scaler.transform(role_raw)
                calibrated[family][(domain, role)] = role_scaled
                if not np.array_equal(role_raw >= 0.5, role_scaled >= 0.5):
                    raise ValueError("Positive temperature changed a 0.5 decision")
                for scale, values in (("raw", role_raw), ("temperature", role_scaled)):
                    calibration_rows.append(
                        {
                            "family": family,
                            "dataset": domain,
                            "role": role,
                            "scale": scale,
                            "temperature": temperature,
                            "samples": len(frame),
                            "patients": frame["patient_id"].nunique(),
                            **_probability_summary(role_targets, values),
                        }
                    )
                if family in {"exact", "legacy-mask"}:
                    raw_scaled_intervals[f"{family}/{domain}/{role}"] = (
                        _paired_probability_intervals(frame, role_raw, role_scaled)
                    )
    pd.DataFrame(calibration_rows).to_csv(
        output_root / "calibration_metrics.csv", index=False
    )

    contrast_results: dict[str, Any] = {}
    for contrast, (left_family, right_family) in CONTRASTS.items():
        domains: dict[str, Any] = {}
        domain_bootstrap_values: dict[str, list[float]] = {}
        domain_estimates = []
        child_sequences = np.random.SeedSequence(BOOTSTRAP_SEED).spawn(len(PRIMARY))
        domain_generators = {
            domain: np.random.default_rng(sequence)
            for domain, sequence in zip(PRIMARY, child_sequences, strict=True)
        }
        for domain, role in PRIMARY.items():
            frame = ensembles[left_family][(domain, role)]
            right_frame = ensembles[right_family][(domain, role)]
            if not frame[["sample_id", "target", "patient_id"]].equals(
                right_frame[["sample_id", "target", "patient_id"]]
            ):
                raise ValueError(f"Unpaired primary contrast: {contrast}/{domain}")
            left = probability_by_family[left_family][(domain, role)]
            right = probability_by_family[right_family][(domain, role)]
            as_interval = _paired_as_interval(frame, left, right)
            domains[domain] = {
                "role": role,
                "ensemble_as_left": _as(frame["target"].to_numpy(dtype=int), left),
                "ensemble_as_right": _as(frame["target"].to_numpy(dtype=int), right),
                "as_left_minus_right": as_interval,
            }
            domain_estimates.append(as_interval["estimate"])
            targets = frame["target"].to_numpy(dtype=int)
            resamples = _patient_resamples(
                frame["patient_id"].to_numpy(dtype=str),
                generator=domain_generators[domain],
            )
            domain_bootstrap_values[domain] = [
                _as(targets[index], left[index]) - _as(targets[index], right[index])
                for index in resamples
            ]
            if contrast == "exact_minus_legacy":
                domains[domain]["raw_probability_metric_intervals"] = (
                    _paired_probability_intervals(frame, left, right)
                )
                domains[domain]["temperature_probability_metric_intervals"] = (
                    _paired_probability_intervals(
                        frame,
                        calibrated[left_family][(domain, role)],
                        calibrated[right_family][(domain, role)],
                    )
                )
        mean_values = np.mean(
            np.column_stack(
                [domain_bootstrap_values[domain] for domain in PRIMARY]
            ),
            axis=1,
        )
        domains["equal_database_mean"] = _interval(
            mean_values.tolist(), float(np.mean(domain_estimates))
        )
        contrast_results[contrast] = domains

    threshold_rows = []
    for domain in ROLES:
        frame = ensembles["exact"][(domain, "validation_select")]
        threshold_rows.extend(
            _threshold_rows(
                frame,
                probability_by_family["exact"][(domain, "validation_select")],
            )
        )
    pd.DataFrame(threshold_rows).to_csv(
        output_root / "exact_validation_threshold_sensitivity.csv", index=False
    )

    role_patients = {
        f"{domain}/{role}": sorted(
            _metadata(manifest, domain, role)["patient_id"].astype(str).unique()
        )
        for domain, roles in ROLES.items()
        for role in roles
    }
    prohibited_overlap = {}
    for domain in ROLES:
        keys = [key for key in role_patients if key.startswith(f"{domain}/")]
        for index, left in enumerate(keys):
            for right in keys[index + 1 :]:
                overlap = sorted(set(role_patients[left]).intersection(role_patients[right]))
                expected = {left.rsplit("/", 1)[1], right.rsplit("/", 1)[1]} == {
                    "locked_inter_test",
                    "locked_intra_test",
                }
                if overlap and not expected:
                    prohibited_overlap[f"{left}__{right}"] = overlap
    if prohibited_overlap:
        raise ValueError(f"Prohibited patient overlap: {prohibited_overlap}")

    payload = {
        "schema_version": 1,
        "stage": "BEATS_revision_final_analysis",
        "families": FAMILIES,
        "primary_roles": PRIMARY,
        "threshold": 0.5,
        "threshold_interpretation": (
            "prespecified research-protocol threshold; "
            "not a clinical operating threshold"
        ),
        "temperature_positive_and_decision_invariant": True,
        "temperatures": temperatures,
        "contrasts": contrast_results,
        "raw_minus_temperature_probability_intervals": raw_scaled_intervals,
        "calibration_interval_scope": (
            "Patient-cluster intervals cover NLL, Brier score, and 15-bin "
            "equal-frequency ECE. Calibration intercept and slope are reported "
            "as descriptive point estimates because cluster resamples may be "
            "quasi-separated."
        ),
        "bootstrap": {
            "iterations": ITERATIONS,
            "seed": BOOTSTRAP_SEED,
            "confidence": CONFIDENCE,
            "cluster": "patient",
            "equal_database_resampling": "independent domain resamples paired by replicate index",
        },
        "manifest_sha256": sha256_file(manifest_path),
        "prediction_hashes": prediction_hashes,
        "patient_overlap_audit_passed": True,
        "finite_value_audit_passed": True,
        "locked_results_used_for_selection": False,
    }
    _atomic_json(payload, output_root / "revision_analysis_final.json")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
