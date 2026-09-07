#!/usr/bin/env python3
"""Analyze frozen Analysis A exact-versus-Legacy matched completers.

This script implements amendment BEATS-ABLATION-2026-09-03-A1.  It uses only
the two prespecified completed seed pairs (20260729 and 20260731), labels their
equal-weight ensemble as diagnostic, and never substitutes it for the protected
three-seed Exact primary ensemble.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analyze_revision_ablation import (
    BOOTSTRAP_SEED,
    CONFIDENCE,
    ITERATIONS,
    PRIMARY,
    ROLES,
    _as,
    _atomic_json,
    _calibration_regression,
    _classification,
    _interval,
    _load_seed,
    _metadata,
    _patient_resamples,
    _probability_summary,
)
from respiratory_sound.calibration import (
    TemperatureScaler,
    fit_temperature,
    probability_metrics,
)
from respiratory_sound.gate11a import sha256_file


PLAN_SHA256 = "18b6e39bba441e5be1ec870591b9dbf2846b7e6ab6293be9d211aab12809be39"
AMENDMENT_SHA256 = "9b2b7e19ae0f14fc8e3107b579d45987e9a729753507f858a2ca12ce94aa91e4"
MANIFEST_SHA256 = "2dd168129fc0fc159db201f2990c4d7074cb746fc9bc2c13bd5031e47dbd1419"
MATCHED_SEEDS = (20_260_729, 20_260_731)
FAMILIES = ("exact", "legacy-mask")
BASE_STATISTICAL_FREEZE_SHA256 = (
    "f7d8327bc5c5d2490c814f338733221e597fceb81312f36514e801436bafd089"
)
FIRST_IMPLEMENTATION_AMENDMENT_SHA256 = (
    "128cb041b42a8c2eda8ab0d183845db5bb2fa01d71c8ac9ad08e95746b04fd16"
)


def _matched_completer_ensemble(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Arithmetic-mean ensemble for the two frozen completed seed pairs."""

    if len(frames) != len(MATCHED_SEEDS):
        raise ValueError("Matched-completer ensemble requires exactly two frames")
    keys = [
        "sample_id",
        "dataset",
        "patient_id",
        "protocol_role",
        "locked",
        "binary_label_id",
        "fine_label_name",
        "event_duration_seconds",
        "target",
    ]
    merged = frames[0][keys + ["probability_1"]].rename(
        columns={"probability_1": f"probability_1_seed{MATCHED_SEEDS[0]}"}
    )
    merged = merged.merge(
        frames[1][keys + ["probability_1"]].rename(
            columns={"probability_1": f"probability_1_seed{MATCHED_SEEDS[1]}"}
        ),
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    seed_columns = [f"probability_1_seed{seed}" for seed in MATCHED_SEEDS]
    merged["probability_1"] = merged[seed_columns].mean(axis=1)
    merged["probability_0"] = 1.0 - merged["probability_1"]
    merged["prediction"] = (merged["probability_1"] >= 0.5).astype(int)
    if len(merged) != len(frames[0]) or len(merged) != len(frames[1]):
        raise ValueError("Matched-completer ensemble lost samples")
    return merged


def _paired_metric_intervals(
    frame: pd.DataFrame,
    left: np.ndarray,
    right: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Patient-cluster intervals for NLL, Brier, and equal-frequency ECE."""

    targets = frame["target"].to_numpy(dtype=int)
    left_point = probability_metrics(targets, left, bins=15)
    right_point = probability_metrics(targets, right, bins=15)
    metric_names = (
        "negative_log_likelihood",
        "brier_score",
        "equal_frequency_ece",
    )
    samples: dict[str, list[float]] = {name: [] for name in metric_names}
    for index in _patient_resamples(frame["patient_id"].to_numpy(dtype=str)):
        left_sample = probability_metrics(targets[index], left[index], bins=15)
        right_sample = probability_metrics(targets[index], right[index], bins=15)
        for name in metric_names:
            samples[name].append(left_sample[name] - right_sample[name])
    return {
        name: _interval(samples[name], left_point[name] - right_point[name])
        for name in metric_names
    }


def _verify_freeze(root: Path) -> tuple[Path, dict[str, Any], dict[str, str]]:
    freeze_path = (
        root
        / "artifacts/revision_2026_09_03/analysis_a_statistical_freeze.json"
    )
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("status") != "frozen_before_analysis_a_statistics":
        raise ValueError("Analysis A statistical freeze has invalid status")
    if freeze.get("analysis_plan_sha256") != PLAN_SHA256:
        raise ValueError("Analysis-plan identity changed")
    if freeze.get("legacy_failure_amendment_sha256") != AMENDMENT_SHA256:
        raise ValueError("Legacy-failure amendment identity changed")
    if freeze.get("manifest_sha256") != MANIFEST_SHA256:
        raise ValueError("Manifest identity changed")
    if tuple(freeze.get("matched_completed_seeds", ())) != MATCHED_SEEDS:
        raise ValueError("Matched-completer seed set changed")

    amendment_path = (
        root
        / "artifacts/revision_2026_09_03/"
        "analysis_a_statistical_freeze_amendment_2026-09-04.json"
    )
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    if amendment.get("status") != "frozen_implementation_only_correction":
        raise ValueError("Analysis A statistical-freeze amendment is invalid")
    if amendment.get("base_statistical_freeze_sha256") != BASE_STATISTICAL_FREEZE_SHA256:
        raise ValueError("Analysis A base statistical freeze changed")
    second_amendment_path = (
        root
        / "artifacts/revision_2026_09_03/"
        "analysis_a_statistical_freeze_amendment2_2026-09-04.json"
    )
    second_amendment = json.loads(
        second_amendment_path.read_text(encoding="utf-8")
    )
    if second_amendment.get("status") != "frozen_role_audit_correction":
        raise ValueError("Analysis A role-audit amendment is invalid")
    if (
        second_amendment.get("previous_amendment_sha256")
        != FIRST_IMPLEMENTATION_AMENDMENT_SHA256
    ):
        raise ValueError("First Analysis A implementation amendment changed")
    if sha256_file(Path(__file__).resolve()) != second_amendment.get(
        "corrected_analysis_script_sha256"
    ):
        raise ValueError("Corrected Analysis A script hash changed")

    expected_files = dict(freeze["frozen_files"])
    expected_files.pop("scripts/analyze_revision_analysis_a.py")
    expected_files[
        "artifacts/revision_2026_09_03/"
        "analysis_a_statistical_freeze_amendment_2026-09-04.json"
    ] = sha256_file(amendment_path)
    expected_files[
        "artifacts/revision_2026_09_03/"
        "analysis_a_statistical_freeze_amendment2_2026-09-04.json"
    ] = sha256_file(second_amendment_path)
    for relative, expected in expected_files.items():
        observed = sha256_file(root / relative)
        if observed != expected:
            raise ValueError(
                f"Frozen Analysis A file changed: {relative}: {observed} != {expected}"
            )
    return freeze_path, freeze, expected_files


def _verify_source_manifests(root: Path) -> None:
    legacy_root = root / "artifacts/revision_2026_09_03/predictions/legacy_mask"
    legacy = json.loads(
        (legacy_root / "prediction_manifest.json").read_text(encoding="utf-8")
    )
    if tuple(legacy["seeds"]) != MATCHED_SEEDS:
        raise ValueError("Legacy prediction manifest seed set changed")
    if not legacy["calibration_accessed"] or not legacy["locked_tests_accessed"]:
        raise ValueError("Legacy prediction manifest has incomplete role access")
    if legacy["selection_after_locked_access"]:
        raise ValueError("Legacy manifest reports post-held-out selection")
    for name, expected in legacy["files"].items():
        if sha256_file(legacy_root / name) != expected:
            raise ValueError(f"Legacy prediction hash changed: {name}")

    source_manifests = (
        root / "artifacts/post_gate11a/calibration_predictions/prediction_manifest.json",
        root / "artifacts/post_gate11a/locked_predictions/prediction_manifest.json",
    )
    for manifest_path in source_manifests:
        source = json.loads(manifest_path.read_text(encoding="utf-8"))
        for name, expected in source["files"].items():
            if not name.startswith("fullft_seed"):
                continue
            seed = int(name.split("seed", 1)[1].split("_", 1)[0])
            if seed not in MATCHED_SEEDS:
                continue
            if sha256_file(manifest_path.parent / name) != expected:
                raise ValueError(f"Protected Exact prediction hash changed: {name}")


def _patient_overlap_audit(manifest: pd.DataFrame) -> dict[str, Any]:
    audited_roles = {
        domain: ("train_fit", *roles) for domain, roles in ROLES.items()
    }
    role_patients = {
        f"{domain}/{role}": sorted(
            _metadata(manifest, domain, role)["patient_id"].astype(str).unique()
        )
        for domain, roles in audited_roles.items()
        for role in roles
    }
    audited: dict[str, int] = {}
    prohibited: dict[str, list[str]] = {}
    expected_descriptive_intra_overlap = {
        "sprsound2022/train_fit__sprsound2022/locked_intra_test": 111,
        "sprsound2022/validation_select__sprsound2022/locked_intra_test": 19,
        "sprsound2022/calibration__sprsound2022/locked_intra_test": 26,
        "sprsound2022/locked_inter_test__sprsound2022/locked_intra_test": 0,
    }
    for domain in audited_roles:
        keys = [key for key in role_patients if key.startswith(f"{domain}/")]
        for offset, left in enumerate(keys):
            for right in keys[offset + 1 :]:
                overlap = sorted(
                    set(role_patients[left]).intersection(role_patients[right])
                )
                label = f"{left}__{right}"
                audited[label] = len(overlap)
                roles_in_pair = {
                    left.rsplit("/", 1)[1], right.rsplit("/", 1)[1]
                }
                allowed_pair = (
                    domain == "sprsound2022"
                    and "locked_intra_test" in roles_in_pair
                )
                if overlap and not allowed_pair:
                    prohibited[label] = overlap
    if prohibited:
        raise ValueError(f"Prohibited patient overlap: {prohibited}")
    for label, expected in expected_descriptive_intra_overlap.items():
        if audited.get(label) != expected:
            raise ValueError(
                f"Descriptive intra-patient overlap changed: {label}: "
                f"{audited.get(label)} != {expected}"
            )
    return {
        "pairwise_overlap_counts": audited,
        "expected_descriptive_intra_overlap": expected_descriptive_intra_overlap,
        "prohibited_overlap": prohibited,
        "primary_patient_disjoint_roles_passed": True,
    }


def main() -> None:
    root = Path.cwd().resolve()
    freeze_path, freeze, frozen_files = _verify_freeze(root)
    _verify_source_manifests(root)
    manifest_path = root / "data/manifests/cross_domain_binary.csv"
    if sha256_file(manifest_path) != MANIFEST_SHA256:
        raise ValueError("Analysis manifest hash changed")
    manifest = pd.read_csv(manifest_path, dtype={"patient_id": str})

    output_root = root / "artifacts/revision_2026_09_03/analysis_a_final"
    output_root.mkdir(parents=True, exist_ok=False)

    ensembles: dict[str, dict[tuple[str, str], pd.DataFrame]] = {
        family: {} for family in FAMILIES
    }
    probabilities: dict[str, dict[tuple[str, str], np.ndarray]] = {
        family: {} for family in FAMILIES
    }
    prediction_hashes: dict[str, str] = {}
    classification_rows: list[dict[str, Any]] = []

    for family in FAMILIES:
        for domain, roles in ROLES.items():
            for role in roles:
                seed_frames = []
                for seed in MATCHED_SEEDS:
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
                        frame, frame["probability_1"].to_numpy(dtype=float)
                    )
                    classification_rows.append(
                        {
                            "family": family,
                            "member": f"seed{seed}",
                            "ensemble_scope": "single_seed",
                            "dataset": domain,
                            "role": role,
                            "samples": len(frame),
                            "patients": frame["patient_id"].nunique(),
                            **{
                                key: value
                                for key, value in metrics.items()
                                if isinstance(value, float)
                            },
                        }
                    )
                ensemble = _matched_completer_ensemble(seed_frames)
                ensembles[family][(domain, role)] = ensemble
                values = ensemble["probability_1"].to_numpy(dtype=float)
                probabilities[family][(domain, role)] = values
                metrics = _classification(ensemble, values)
                classification_rows.append(
                    {
                        "family": family,
                        "member": "matched_completer_ensemble",
                        "ensemble_scope": "diagnostic_two_seed_20260729_20260731",
                        "dataset": domain,
                        "role": role,
                        "samples": len(ensemble),
                        "patients": ensemble["patient_id"].nunique(),
                        **{
                            key: value
                            for key, value in metrics.items()
                            if isinstance(value, float)
                        },
                    }
                )

    classification_path = output_root / "classification_metrics.csv"
    pd.DataFrame(classification_rows).to_csv(classification_path, index=False)

    temperatures: dict[str, dict[str, float]] = {family: {} for family in FAMILIES}
    calibrated: dict[str, dict[tuple[str, str], np.ndarray]] = {
        family: {} for family in FAMILIES
    }
    calibration_rows: list[dict[str, Any]] = []
    raw_minus_temperature: dict[str, Any] = {}
    for family in FAMILIES:
        for domain, roles in ROLES.items():
            calibration_frame = ensembles[family][(domain, "calibration")]
            calibration_targets = calibration_frame["target"].to_numpy(dtype=int)
            calibration_raw = probabilities[family][(domain, "calibration")]
            temperature = float(
                fit_temperature(calibration_targets, calibration_raw).temperature
            )
            if not np.isfinite(temperature) or temperature <= 0.0:
                raise FloatingPointError("Temperature must be positive and finite")
            temperatures[family][domain] = temperature
            scaler = TemperatureScaler(temperature)
            for role in roles:
                frame = ensembles[family][(domain, role)]
                targets = frame["target"].to_numpy(dtype=int)
                raw = probabilities[family][(domain, role)]
                scaled = scaler.transform(raw)
                if not np.isfinite(scaled).all():
                    raise FloatingPointError("Temperature-scaled probability is non-finite")
                if not np.array_equal(raw >= 0.5, scaled >= 0.5):
                    raise ValueError("Positive temperature changed the 0.5 decision")
                calibrated[family][(domain, role)] = scaled
                for scale, values in (("raw", raw), ("temperature", scaled)):
                    calibration_rows.append(
                        {
                            "family": family,
                            "member": "matched_completer_ensemble",
                            "ensemble_scope": "diagnostic_two_seed_20260729_20260731",
                            "dataset": domain,
                            "role": role,
                            "scale": scale,
                            "temperature": temperature,
                            "samples": len(frame),
                            "patients": frame["patient_id"].nunique(),
                            **_probability_summary(targets, values),
                        }
                    )
                raw_minus_temperature[f"{family}/{domain}/{role}"] = (
                    _paired_metric_intervals(frame, raw, scaled)
                )

    calibration_path = output_root / "calibration_metrics.csv"
    pd.DataFrame(calibration_rows).to_csv(calibration_path, index=False)

    contrast_domains: dict[str, Any] = {}
    domain_bootstrap_values: dict[str, list[float]] = {}
    domain_estimates: list[float] = []
    child_sequences = np.random.SeedSequence(BOOTSTRAP_SEED).spawn(len(PRIMARY))
    domain_generators = {
        domain: np.random.default_rng(sequence)
        for domain, sequence in zip(PRIMARY, child_sequences, strict=True)
    }
    for domain, role in PRIMARY.items():
        exact_frame = ensembles["exact"][(domain, role)]
        legacy_frame = ensembles["legacy-mask"][(domain, role)]
        pairing_columns = ["sample_id", "target", "patient_id"]
        if not exact_frame[pairing_columns].equals(legacy_frame[pairing_columns]):
            raise ValueError(f"Unpaired Exact/Legacy comparison: {domain}/{role}")
        exact_raw = probabilities["exact"][(domain, role)]
        legacy_raw = probabilities["legacy-mask"][(domain, role)]
        targets = exact_frame["target"].to_numpy(dtype=int)
        resamples = _patient_resamples(
            exact_frame["patient_id"].to_numpy(dtype=str),
            generator=domain_generators[domain],
        )
        point = _as(targets, exact_raw) - _as(targets, legacy_raw)
        bootstrap_values = [
            _as(targets[index], exact_raw[index])
            - _as(targets[index], legacy_raw[index])
            for index in resamples
        ]
        domain_bootstrap_values[domain] = bootstrap_values
        domain_estimates.append(point)
        contrast_domains[domain] = {
            "role": role,
            "ensemble_scope": "diagnostic_two_seed_20260729_20260731",
            "exact_average_score": _as(targets, exact_raw),
            "legacy_average_score": _as(targets, legacy_raw),
            "exact_minus_legacy_average_score": _interval(bootstrap_values, point),
            "raw_probability_metric_intervals": _paired_metric_intervals(
                exact_frame, exact_raw, legacy_raw
            ),
            "temperature_probability_metric_intervals": _paired_metric_intervals(
                exact_frame,
                calibrated["exact"][(domain, role)],
                calibrated["legacy-mask"][(domain, role)],
            ),
        }

    equal_database_values = np.mean(
        np.column_stack(
            [domain_bootstrap_values[domain] for domain in PRIMARY]
        ),
        axis=1,
    )
    contrast_domains["equal_database_mean"] = _interval(
        equal_database_values.tolist(), float(np.mean(domain_estimates))
    )

    overlap_audit = _patient_overlap_audit(manifest)
    payload = {
        "schema_version": 1,
        "stage": "BEATS_revision_analysis_a_matched_completers",
        "status": "complete",
        "analysis_scope": (
            "diagnostic two-seed matched-completer comparison conditional on "
            "Legacy fixed-prefix completion; not the protected three-seed Exact primary"
        ),
        "families": FAMILIES,
        "matched_completed_seeds": MATCHED_SEEDS,
        "failed_legacy_seed_retained": 20_260_730,
        "failed_seed_used_for_classification_or_calibration": False,
        "threshold": 0.5,
        "temperature_positive_and_decision_invariant": True,
        "temperatures": temperatures,
        "exact_minus_legacy": contrast_domains,
        "raw_minus_temperature_probability_intervals": raw_minus_temperature,
        "bootstrap": {
            "iterations": ITERATIONS,
            "seed": BOOTSTRAP_SEED,
            "confidence": CONFIDENCE,
            "cluster": "patient",
            "equal_database_resampling": (
                "independent domain resamples paired by replicate index"
            ),
        },
        "calibration_metrics": (
            "NLL, Brier score, 15-bin equal-frequency ECE, intercept, and slope"
        ),
        "calibration_intercept_and_slope_interval_status": (
            "descriptive point estimates; not bootstrapped because clustered "
            "resamples may be quasi-separated"
        ),
        "manifest_sha256": sha256_file(manifest_path),
        "statistical_freeze_sha256": sha256_file(freeze_path),
        "frozen_files": frozen_files,
        "prediction_hashes": prediction_hashes,
        "output_hashes": {
            "classification_metrics.csv": sha256_file(classification_path),
            "calibration_metrics.csv": sha256_file(calibration_path),
        },
        "patient_overlap_audit": overlap_audit,
        "finite_value_audit_passed": True,
        "prediction_coverage_and_label_audit_passed": True,
        "locked_results_used_for_selection": False,
        "selection_after_locked_access": False,
        "analysis_freeze_methods": freeze["methods"],
    }
    result_path = output_root / "analysis_a_final.json"
    _atomic_json(payload, result_path)
    digest_path = output_root / "analysis_a_final.sha256"
    digest_path.write_text(
        f"{sha256_file(result_path)}  {result_path.name}\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
