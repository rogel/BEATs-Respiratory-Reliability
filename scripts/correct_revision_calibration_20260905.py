#!/usr/bin/env python3
"""Freeze and execute an additive calibration solver correction, never overwrite."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import shutil
import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from analyze_revision_ablation import (
    ITERATIONS, PRIMARY, ROLES, _as, _load_seed, _patient_resamples,
)
from analyze_revision_analysis_a import (
    FAMILIES, MANIFEST_SHA256, MATCHED_SEEDS, _matched_completer_ensemble,
    _patient_overlap_audit, _verify_freeze, _verify_source_manifests,
)
from respiratory_sound.calibration import (
    TemperatureScaler, probabilities_to_logits, probability_metrics,
)
from respiratory_sound.gate11a import sha256_file
from respiratory_sound.revision_calibration_regression import (
    CalibrationRegressionError, fit_calibration_regression,
)

FREEZE_NAME = "calibration_solver_correction_freeze_2026_09_05.json"
OLD_RESULT_SHA = "8f2d39a665e2678ee20a00afe96ae1b5a81422b18471360cbee566220df1caa9"
COEFFICIENTS = ("calibration_intercept", "calibration_slope")


def write_hashed_json(path, payload):
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    path.with_suffix(".sha256").write_text(f"{sha256_file(path)}  {path.name}\n")


def interval(values, point):
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    complete = bool(valid.all())
    bounds = np.quantile(values, [.025, .975]) if complete else (None, None)
    return {"estimate": float(point), "lower": None if bounds[0] is None else float(bounds[0]),
            "upper": None if bounds[1] is None else float(bounds[1]),
            "valid_replicates": int(valid.sum()), "invalid_replicates": int((~valid).sum()),
            "status": "complete" if complete else "withheld_due_to_nonestimable_replicates"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    base = root / "artifacts/revision_2026_09_03"
    original = base / "analysis_a_final"
    result_path = original / "analysis_a_final.json"
    assert sha256_file(result_path) == OLD_RESULT_SHA
    _verify_freeze(root)
    _verify_source_manifests(root)
    old = json.loads(result_path.read_text())
    for name, digest in old["output_hashes"].items():
        assert sha256_file(original / name) == digest
    freeze_path = base / FREEZE_NAME
    files = [Path(__file__).resolve(),
             root / "src/respiratory_sound/revision_calibration_regression.py",
             root / "tests/test_revision_calibration_regression.py",
             root.parent / "experiment_plans/2026-09-03/05_CALIBRATION_IMPLEMENTATION_CORRECTION_2026-09-05.md",
             result_path, original / "classification_metrics.csv", original / "calibration_metrics.csv",
             base / "review_2026_09_05/independent_review_diagnostic.json"]
    if args.freeze_only:
        write_hashed_json(freeze_path, {
            "status": "frozen_before_corrected_formal_statistics",
            "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "scope": "implementation correction of the same unpenalized model; original predictions and training frozen",
            "files": {str(p.relative_to(root) if p.is_relative_to(root) else p): sha256_file(p) for p in files},
            "original_result_sha256": OLD_RESULT_SHA,
            "bootstrap_iterations": 4000, "bootstrap_seed": 20260729,
            "invalid_replicate_policy": "retain and enumerate; no replacement; withhold affected interval unless all 4000 valid",
        })
        print(json.dumps({"freeze": str(freeze_path), "sha256": sha256_file(freeze_path)}, indent=2))
        return
    freeze = json.loads(freeze_path.read_text())
    for name, digest in freeze["files"].items():
        path = Path(name) if Path(name).is_absolute() else root / name
        assert sha256_file(path) == digest, name
    manifest_path = root / "data/manifests/cross_domain_binary.csv"
    assert sha256_file(manifest_path) == MANIFEST_SHA256
    manifest = pd.read_csv(manifest_path, dtype={"patient_id": str})
    overlap = _patient_overlap_audit(manifest)
    previous_table = pd.read_csv(original / "calibration_metrics.csv")
    corrected_table = previous_table.copy()
    point_diagnostics, bootstrap_results, corrected_rows = {}, {}, []
    metric_matches, as_matches = 0, 0
    started = time.monotonic()
    for domain, roles in ROLES.items():
        for role in roles:
            frames, points, temperatures = {}, {}, {}
            for family in FAMILIES:
                frames[family] = _matched_completer_ensemble([
                    _load_seed(root, manifest, family=family, seed=seed, domain=domain, role=role)[0]
                    for seed in MATCHED_SEEDS
                ])
                frame = frames[family]
                y, p = frame["target"].to_numpy(dtype=int), frame["probability_1"].to_numpy(dtype=float)
                temperature = old["temperatures"][family][domain]
                temperatures[family] = temperature
                scaled = TemperatureScaler(temperature).transform(p)
                assert np.array_equal(p >= .5, scaled >= .5)
                np.testing.assert_allclose(probabilities_to_logits(scaled), probabilities_to_logits(p) / temperature, atol=1e-10)
                for scale, values in (("raw", p), ("temperature", scaled)):
                    key = f"{family}/{domain}/{role}/{scale}"
                    mask = ((previous_table.family == family) & (previous_table.dataset == domain)
                            & (previous_table.role == role) & (previous_table.scale == scale))
                    assert int(mask.sum()) == 1
                    index = previous_table.index[mask][0]
                    metrics = probability_metrics(y, values, bins=15)
                    for name, value in metrics.items():
                        if name in previous_table.columns:
                            np.testing.assert_allclose(value, previous_table.at[index, name], rtol=1e-12, atol=1e-14)
                            metric_matches += 1
                    fit = fit_calibration_regression(y, values, independent_check=True)
                    point_diagnostics[key] = fit
                    for name in COEFFICIENTS:
                        corrected_rows.append({"key": key, "coefficient": name,
                                               "old_value": float(previous_table.at[index, name]),
                                               "corrected_value": fit[name]})
                        corrected_table.at[index, name] = fit[name]
                    if scale == "raw":
                        points[family] = np.array([fit[name] for name in COEFFICIENTS])
                    else:
                        np.testing.assert_allclose([fit[name] for name in COEFFICIENTS],
                                                   points[family] * [1, temperature], atol=1e-6, rtol=1e-6)
                if role == PRIMARY[domain]:
                    expected = old["exact_minus_legacy"][domain][
                        "exact_average_score" if family == "exact" else "legacy_average_score"]
                    assert abs(_as(y, p) - expected) < 1e-14
                    as_matches += 1
            assert frames["exact"][["sample_id", "target", "patient_id"]].equals(
                frames["legacy-mask"][["sample_id", "target", "patient_id"]])
            frame = frames["exact"]
            y = frame["target"].to_numpy(dtype=int)
            resamples = _patient_resamples(frame["patient_id"].to_numpy(dtype=str))
            draws = {family: np.full((ITERATIONS, 2), np.nan) for family in FAMILIES}
            invalid = {family: [] for family in FAMILIES}
            for family in FAMILIES:
                p = frames[family]["probability_1"].to_numpy(dtype=float)
                for iteration, index in enumerate(resamples):
                    try:
                        fit = fit_calibration_regression(y[index], p[index])
                        draws[family][iteration] = [fit[name] for name in COEFFICIENTS]
                    except CalibrationRegressionError as exc:
                        invalid[family].append({"replicate": iteration, "reason": exc.reason})
                    if (iteration + 1) % 1000 == 0:
                        print(json.dumps({"family": family, "dataset": domain, "role": role,
                                          "replicates_completed": iteration + 1,
                                          "invalid": len(invalid[family])}), flush=True)
            role_result = {"families": {}, "exact_minus_legacy": {}}
            for family in FAMILIES:
                temperature = temperatures[family]
                family_result = {"raw": {}, "temperature": {}, "raw_minus_temperature": {},
                                 "nonestimable_replicates": invalid[family],
                                 "nonestimability_reason_counts": dict(Counter(x["reason"] for x in invalid[family]))}
                for j, name in enumerate(COEFFICIENTS):
                    multiplier = 1 if j == 0 else temperature
                    family_result["raw"][name] = interval(draws[family][:, j], points[family][j])
                    family_result["temperature"][name] = interval(draws[family][:, j] * multiplier, points[family][j] * multiplier)
                    family_result["raw_minus_temperature"][name] = interval(
                        draws[family][:, j] * (1 - multiplier), points[family][j] * (1 - multiplier))
                role_result["families"][family] = family_result
            for scale in ("raw", "temperature"):
                role_result["exact_minus_legacy"][scale] = {}
                for j, name in enumerate(COEFFICIENTS):
                    le = temperatures["exact"] if scale == "temperature" and j == 1 else 1
                    ri = temperatures["legacy-mask"] if scale == "temperature" and j == 1 else 1
                    role_result["exact_minus_legacy"][scale][name] = interval(
                        draws["exact"][:, j] * le - draws["legacy-mask"][:, j] * ri,
                        points["exact"][j] * le - points["legacy-mask"][j] * ri)
            bootstrap_results[f"{domain}/{role}"] = role_result
    unchanged = [column for column in previous_table if column not in COEFFICIENTS]
    pd.testing.assert_frame_equal(previous_table[unchanged], corrected_table[unchanged])
    _verify_freeze(root)
    _verify_source_manifests(root)
    output = base / "analysis_a_corrected_2026_09_05"
    output.mkdir(exist_ok=False)
    shutil.copy2(original / "classification_metrics.csv", output / "classification_metrics.csv")
    corrected_table.to_csv(output / "calibration_metrics.csv", index=False)
    diagnostics = {"point_fits": point_diagnostics, "coefficient_changes": corrected_rows,
                   "bootstrap": bootstrap_results, "bootstrap_seed": 20260729,
                   "bootstrap_iterations": ITERATIONS, "regression": "unpenalized joint intercept and slope; not calibration-in-the-large",
                   "invalid_replicate_policy": freeze["invalid_replicate_policy"],
                   "point_estimates_all_convergence_and_independent_solver_checks_passed": True}
    write_hashed_json(output / "calibration_regression_diagnostics.json", diagnostics)
    new = deepcopy(old)
    new.update({
        "schema_version": 2, "status": "complete_with_audited_calibration_solver_correction",
        "supersedes_result_sha256": OLD_RESULT_SHA,
        "supersession_scope": "correct calibration regression coefficients and interval omission only; original files preserved",
        "correction_freeze_sha256": sha256_file(freeze_path),
        "corrected_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "calibration_intercept_and_slope_interval_status": "4000 patient-cluster draws; see diagnostics for intervals and any explicitly enumerated nonestimable replicates",
        "calibration_regression_qa_passed": True,
        "original_nonregression_calibration_cells_verified": metric_matches,
        "original_primary_contrast_as_points_verified": as_matches,
        "classification_table_byte_identical": sha256_file(output / "classification_metrics.csv") == old["output_hashes"]["classification_metrics.csv"],
        "patient_overlap_audit": overlap,
        "original_intervals_preserved_without_reselection": True,
        "correction_elapsed_seconds": time.monotonic() - started,
        "output_hashes": {p.name: sha256_file(p) for p in output.iterdir() if p.suffix in (".csv", ".json")},
    })
    assert new["classification_table_byte_identical"]
    write_hashed_json(output / "analysis_a_final.json", new)
    print(json.dumps({"output": str(output), "result_sha256": sha256_file(output / "analysis_a_final.json"),
                      "point_fits_passed": len(point_diagnostics), "nonregression_cells_verified": metric_matches,
                      "invalid_by_role": {role: {family: len(info["families"][family]["nonestimable_replicates"])
                                                for family in FAMILIES} for role, info in bootstrap_results.items()},
                      "elapsed_seconds": time.monotonic() - started}, indent=2), flush=True)


if __name__ == "__main__":
    main()
