#!/usr/bin/env python3
"""Additive, separately frozen no-JS inference and analysis; no model selection."""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import json
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch

from analyze_revision_ablation import (
    BOOTSTRAP_SEED, ITERATIONS, PRIMARY, ROLES, _as, _classification,
    _exact_path, _metadata, _patient_resamples, _threshold_rows,
)
from analyze_revision_analysis_a import MANIFEST_SHA256, PLAN_SHA256, _patient_overlap_audit
from compute_feature_stats import feature_config_from_yaml
from predict_revision_ablation import _load_model, _run_dir
from respiratory_sound.calibration import (
    TemperatureScaler, fit_temperature, probability_metrics, probabilities_to_logits,
)
from respiratory_sound.gate11a import ALLOWED_SEEDS, sha256_file
from respiratory_sound.post_gate11a import build_waveform_dataset, infer_probabilities, probability_ensemble
from respiratory_sound.revision_calibration_regression import (
    CalibrationRegressionError, fit_calibration_regression,
)
from respiratory_sound.runtime import select_device

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/revision_2026_09_03"
FREEZE = BASE / "analysis_b_execution_freeze_2026_09_05.json"
PRED = BASE / "predictions/no_js"
OUTPUT = BASE / "analysis_b_final"
JOB = BASE / "jobs_2026_09_05/analysis_b"
EXACT_SHA = dict(zip(ALLOWED_SEEDS, (
    "3f7942419370cc8822d447c9e1b2789cbfa58d2b150f6466c94da46c5948f6a8",
    "a9ee6200147b9979bcd37a5be62a60156ab5e146707982ed66e035727a47ef47",
    "093519ab419cf1c63c4d62047b858c75b62b6da3c1aaf74ccfca261fd3b23d60",
), strict=True))
PROTECTED_T = {"icbhi2017": 1.6464554911106857, "sprsound2022": 1.0350845956825019}
COEFFICIENTS = ("calibration_intercept", "calibration_slope")
NONESTIMABLE = {"single_class", "rank_deficient_predictor", "complete_separation", "quasi_complete_separation"}


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def write_json(path, value, *, replace=False):
    if path.exists() and not replace:
        raise FileExistsError(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    if not replace:
        path.with_suffix(".sha256").write_text(f"{sha256_file(path)}  {path.name}\n")


def read_json(path):
    return json.loads(path.read_text())


def checked_frame(frame, manifest, domain, role):
    expected = _metadata(manifest, domain, role).sort_values("sample_id").reset_index(drop=True)
    actual = frame.sort_values("sample_id").reset_index(drop=True)
    assert len(actual) == len(expected) and not actual.sample_id.duplicated().any()
    for column in expected.columns:
        if column == "event_duration_seconds":
            np.testing.assert_allclose(actual[column], expected[column], rtol=1e-12, atol=1e-12)
        else:
            assert actual[column].astype(str).equals(expected[column].astype(str)), (domain, role, column)
    y, p = actual.target.to_numpy(dtype=int), actual.probability_1.to_numpy(dtype=float)
    assert np.array_equal(y, expected.binary_label_id)
    assert np.isfinite(p).all() and np.all((p >= 0) & (p <= 1))
    assert np.array_equal(actual.prediction, (p >= .5).astype(int))
    assert np.isfinite(actual.probability_0).all()
    np.testing.assert_allclose(actual.probability_0 + p, 1, rtol=0, atol=2e-7)
    return actual


def interval(draws, point):
    values = np.asarray(draws, dtype=float)
    valid = np.isfinite(values)
    complete = bool(valid.all())
    bounds = np.quantile(values, [.025, .975]) if complete else (None, None)
    return {"estimate": None if point is None else float(point),
            "lower": None if bounds[0] is None else float(bounds[0]),
            "upper": None if bounds[1] is None else float(bounds[1]),
            "valid_replicates": int(valid.sum()), "invalid_replicates": int((~valid).sum()),
            "status": "complete" if complete else "withheld_due_to_nonestimable_replicates"}


def regression(y, p, *, independent=False):
    try:
        return fit_calibration_regression(y, p, independent_check=independent)
    except CalibrationRegressionError as exc:
        if exc.reason not in NONESTIMABLE:
            raise
        return {"status": "nonestimable", "reason": exc.reason,
                **{name: None for name in COEFFICIENTS}}


def freeze():
    assert not FREEZE.exists() and not PRED.exists() and not OUTPUT.exists() and not JOB.exists()
    assert sha256_file(ROOT / "data/manifests/cross_domain_binary.csv") == MANIFEST_SHA256
    files = dict(read_json(BASE / "implementation_freeze.json")["files"])
    for relative, expected in files.items():
        assert sha256_file(ROOT / relative) == expected, relative
    paths = set(ROOT / p for p in files)
    plan = ROOT.parent / "experiment_plans/2026-09-03/01_REVISION_ANALYSIS_PLAN_FROZEN_2026-09-03.md"
    upstream = ROOT / "checkpoints/pretrained/beats_as2m_cpt2/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"
    assert sha256_file(plan) == PLAN_SHA256
    assert sha256_file(upstream) == "e5815275a04b6885e7b8af63d120b29bffae2cd2225cf4915e1ec6d819d3022c"
    paths.update((plan, upstream, ROOT / "data/manifests/cross_domain_binary.csv"))
    paths.update((ROOT / "src/respiratory_sound").rglob("*.py"))
    paths.update((ROOT / "third_party").rglob("*.py"))
    paths.update(ROOT / "scripts" / name for name in (
        "run_revision_analysis_b.py", "analyze_revision_ablation.py", "analyze_revision_analysis_a.py",
        "predict_revision_ablation.py", "compute_feature_stats.py", "audit_completed_revision_run.py"))
    paths.update((ROOT / "tests/test_revision_analysis_b.py", ROOT / "tests/test_revision_calibration_regression.py",
                  BASE / "implementation_freeze.json", BASE / "calibration_solver_correction_freeze_2026_09_05.json",
                  ROOT.parent / "experiment_plans/2026-09-03/10_ANALYSIS_B_EXECUTION_SPEC_2026-09-05.md"))
    runs = {}
    for seed in ALLOWED_SEEDS:
        run = _run_dir(ROOT, "no-js", seed)
        audit_path = BASE / "post_training_audits" / f"no_js_seed{seed}.json"
        audit = read_json(audit_path)
        assert audit["status"] == "post_training_audit_passed"
        assert not audit["calibration_accessed"] and not audit["locked_tests_accessed"]
        for name, digest in audit["run_files_sha256"].items():
            assert sha256_file(run / name) == digest, (seed, name)
            paths.add(run / name)
        paths.add(audit_path)
        runs[str(seed)] = {"best_checkpoint_sha256": audit["run_files_sha256"]["best.pt"],
                           "selected_epoch": audit["selected_epoch"], "epochs_completed": audit["epochs_completed"],
                           "audit_sha256": sha256_file(audit_path)}
        exact_run = ROOT / f"runs/gate11a_exactmask_fullft_seed{seed}"
        assert sha256_file(exact_run / "best.pt") == EXACT_SHA[seed]
        paths.update(exact_run / name for name in ("best.pt", "configuration.json", "summary.json", "history.csv"))
        paths.update(_exact_path(ROOT, seed, domain, role) for domain, roles in ROLES.items() for role in roles)
    for subdir in ("calibration_predictions", "locked_predictions"):
        source = ROOT / "artifacts/post_gate11a" / subdir / "prediction_manifest.json"
        paths.add(source)
        for name, digest in read_json(source)["files"].items():
            if name.startswith("fullft_"):
                assert sha256_file(source.parent / name) == digest
                paths.add(source.parent / name)
    protected = ROOT / "artifacts/post_gate11a/calibration_analysis/post_gate11a_calibration_final.json"
    paths.add(protected)
    assert {d: read_json(protected)["locked_parameters"][d]["temperature"] for d in ROLES} == PROTECTED_T
    for name, digest in read_json(BASE / "calibration_solver_correction_freeze_2026_09_05.json")["files"].items():
        path = ROOT / name
        assert sha256_file(path) == digest, name
        paths.add(path)
    manifest = pd.read_csv(ROOT / "data/manifests/cross_domain_binary.csv", dtype={"patient_id": str})
    payload = {"status": "frozen_before_no_js_calibration_and_heldout_inference", "frozen_at": now(),
               "analysis_plan_sha256": PLAN_SHA256, "seeds": list(ALLOWED_SEEDS), "no_js_runs": runs,
               "primary_exact_checkpoints": EXACT_SHA, "patient_overlap_audit": _patient_overlap_audit(manifest),
               "files": {str(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p): sha256_file(p) for p in sorted(paths)},
               "threshold": .5, "primary_temperatures": PROTECTED_T, "bootstrap_seed": BOOTSTRAP_SEED,
               "bootstrap_iterations": ITERATIONS, "selection_after_locked_access": False,
               "runtime": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__}}
    write_json(FREEZE, payload)
    print(json.dumps({"freeze": str(FREEZE), "sha256": sha256_file(FREEZE), "files": len(payload["files"])}, indent=2))


def verify_freeze():
    frozen = read_json(FREEZE)
    assert frozen["status"] == "frozen_before_no_js_calibration_and_heldout_inference"
    assert frozen["seeds"] == list(ALLOWED_SEEDS)
    for path, digest in frozen["files"].items():
        assert sha256_file(ROOT / path) == digest, path
    return frozen


def predict():
    verify_freeze()
    device = select_device("mps")
    assert device.type == "mps"
    PRED.mkdir(parents=True, exist_ok=False)
    manifest_path = ROOT / "data/manifests/cross_domain_binary.csv"
    manifest = pd.read_csv(manifest_path, dtype={"patient_id": str})
    feature = feature_config_from_yaml(ROOT / "configs/data/gate9a_beats.yaml")
    files, run_audits, replay = {}, {}, {}
    # Complete validation replay for all three seeds before any new calibration/test inference.
    for phase in ("validation", "calibration_and_test"):
        for seed in ALLOWED_SEEDS:
            model, audit = _load_model(ROOT, family="no-js", seed=seed, device=device)
            run_audits[str(seed)] = audit
            for domain, roles in ROLES.items():
                for role in roles:
                    if (role == "validation_select") != (phase == "validation"):
                        continue
                    dataset = build_waveform_dataset(manifest_path=manifest_path, project_root=ROOT,
                        feature_config=feature, role=role, domain=domain)
                    frame = checked_frame(infer_probabilities(model, dataset, device=device, batch_size=8),
                                          manifest, domain, role)
                    if phase == "validation":
                        old = checked_frame(pd.read_csv(_run_dir(ROOT, "no-js", seed) /
                            f"best_validation_predictions_{domain}.csv", dtype={"patient_id": str}), manifest, domain, role)
                        error = float(np.max(np.abs(old.probability_1 - frame.probability_1)))
                        assert error <= 1e-6 and np.array_equal(old.prediction, frame.prediction), (seed, domain, error)
                        replay[f"{seed}/{domain}"] = {"maximum_absolute_probability_difference": error,
                                                     "classification_decisions_identical": True}
                    path = PRED / f"seed{seed}_{domain}_{role}.csv"
                    frame.to_csv(path, index=False)
                    files[path.name] = sha256_file(path)
                    print(json.dumps({"phase": phase, "seed": seed, "dataset": domain, "role": role,
                                      "events": len(frame), "sha256": files[path.name]}), flush=True)
            del model
            torch.mps.empty_cache()
        if phase == "validation":
            assert len(replay) == 6
            write_json(PRED / "validation_replay_audit.json", replay)
    for domain, roles in ROLES.items():
        for role in roles:
            frames = [pd.read_csv(PRED / f"seed{s}_{domain}_{role}.csv", dtype={"patient_id": str}) for s in ALLOWED_SEEDS]
            ensemble = checked_frame(probability_ensemble(frames), manifest, domain, role)
            path = PRED / f"ensemble_{domain}_{role}.csv"
            ensemble.to_csv(path, index=False)
            files[path.name] = sha256_file(path)
    verify_freeze()
    write_json(PRED / "prediction_manifest.json", {"status": "complete_pending_analysis_audit", "family": "no-js",
        "seeds": list(ALLOWED_SEEDS), "roles": ROLES, "device": "mps", "batch_size": 8, "runs": run_audits,
        "freeze_sha256": sha256_file(FREEZE), "files": files, "validation_replay": replay,
        "calibration_accessed": True, "locked_tests_accessed": True, "selection_after_locked_access": False})


def analyze():
    verify_freeze()
    sources = read_json(PRED / "prediction_manifest.json")
    assert sources["seeds"] == list(ALLOWED_SEEDS) and sources["freeze_sha256"] == sha256_file(FREEZE)
    for name, digest in sources["files"].items():
        assert sha256_file(PRED / name) == digest, name
    OUTPUT.mkdir(exist_ok=False)
    manifest = pd.read_csv(ROOT / "data/manifests/cross_domain_binary.csv", dtype={"patient_id": str})
    ensembles, classification_rows, hashes, primary_replay = {}, [], {}, {}
    for family in ("exact", "no-js"):
        for domain, roles in ROLES.items():
            for role in roles:
                frames = []
                for seed in ALLOWED_SEEDS:
                    path = _exact_path(ROOT, seed, domain, role) if family == "exact" else PRED / f"seed{seed}_{domain}_{role}.csv"
                    frame = checked_frame(pd.read_csv(path, dtype={"patient_id": str}), manifest, domain, role)
                    frames.append(frame)
                    hashes[str(path.relative_to(ROOT))] = sha256_file(path)
                    metrics = _classification(frame, frame.probability_1.to_numpy())
                    classification_rows.append({"family": family, "member": f"seed{seed}", "dataset": domain,
                        "role": role, "samples": len(frame), "patients": frame.patient_id.nunique(),
                        **{k: v for k, v in metrics.items() if isinstance(v, float)}})
                ensemble = checked_frame(probability_ensemble(frames), manifest, domain, role)
                ensembles[family, domain, role] = ensemble
                classification_rows.append({"family": family, "member": "ensemble", "dataset": domain,
                    "role": role, "samples": len(ensemble), "patients": ensemble.patient_id.nunique(),
                    **{k: v for k, v in _classification(ensemble, ensemble.probability_1.to_numpy()).items() if isinstance(v, float)}})
                if family == "exact" and role != "validation_select":
                    subdir = "calibration_predictions" if role == "calibration" else "locked_predictions"
                    archived = checked_frame(pd.read_csv(ROOT / "artifacts/post_gate11a" / subdir /
                        f"fullft_ensemble_{domain}_{role}.csv", dtype={"patient_id": str}), manifest, domain, role)
                    np.testing.assert_allclose(ensemble.probability_1, archived.probability_1, rtol=0, atol=1e-14)
                    assert np.array_equal(ensemble.prediction, archived.prediction)
                    primary_replay[f"{domain}/{role}"] = True
                if family == "no-js":
                    archived = checked_frame(pd.read_csv(PRED / f"ensemble_{domain}_{role}.csv", dtype={"patient_id": str}), manifest, domain, role)
                    np.testing.assert_allclose(ensemble.probability_1, archived.probability_1, rtol=0, atol=1e-14)
    pd.DataFrame(classification_rows).to_csv(OUTPUT / "classification_metrics.csv", index=False)
    contrasts, boot_draws = {}, {}
    generators = [np.random.default_rng(s) for s in np.random.SeedSequence(BOOTSTRAP_SEED).spawn(2)]
    for (domain, role), generator in zip(PRIMARY.items(), generators, strict=True):
        left, right = ensembles["exact", domain, role], ensembles["no-js", domain, role]
        assert left[["sample_id", "target", "patient_id"]].equals(right[["sample_id", "target", "patient_id"]])
        y, p, q = left.target.to_numpy(), left.probability_1.to_numpy(), right.probability_1.to_numpy()
        point = _as(y, p) - _as(y, q)
        values = np.full(ITERATIONS, np.nan)
        for iteration, indices in enumerate(_patient_resamples(left.patient_id.to_numpy(), generator=generator)):
            if len(np.unique(y[indices])) == 2:
                values[iteration] = _as(y[indices], p[indices]) - _as(y[indices], q[indices])
        boot_draws[domain] = values
        contrasts[domain] = {"role": role, "js_on_average_score": _as(y, p), "no_js_average_score": _as(y, q),
                             "js_on_minus_no_js": interval(values, point)}
    boot_draws["equal_database_mean"] = np.mean(np.stack(list(boot_draws.values())), axis=0)
    contrasts["equal_database_mean"] = interval(boot_draws["equal_database_mean"],
        np.mean([contrasts[d]["js_on_minus_no_js"]["estimate"] for d in PRIMARY]))
    np.savez_compressed(OUTPUT / "classification_bootstrap_draws.npz", **boot_draws)
    print(json.dumps({"stage": "primary_contrast_computed_pending_full_QA", "contrast": contrasts}), flush=True)

    temperatures, cal_rows, diagnostics, raw_scaled_intervals, coefficient_intervals = {}, [], {}, {}, {}
    for family in ("exact", "no-js"):
        for domain, roles in ROLES.items():
            calibration = ensembles[family, domain, "calibration"]
            temperature = PROTECTED_T[domain] if family == "exact" else float(fit_temperature(
                calibration.target.to_numpy(), calibration.probability_1.to_numpy()).temperature)
            assert np.isfinite(temperature) and temperature > 0
            temperatures[f"{family}/{domain}"] = temperature
            for role in roles:
                frame = ensembles[family, domain, role]
                y, p = frame.target.to_numpy(), frame.probability_1.to_numpy()
                scaled = TemperatureScaler(temperature).transform(p)
                assert np.isfinite(scaled).all() and np.array_equal(p >= .5, scaled >= .5)
                points, fits = {}, {}
                key = f"{family}/{domain}/{role}"
                for scale, values in (("raw", p), ("temperature", scaled)):
                    points[scale] = probability_metrics(y, values, bins=15)
                    fits[scale] = regression(y, values, independent=True)
                    diagnostics[f"{key}/{scale}"] = fits[scale]
                    cal_rows.append({"family": family, "dataset": domain, "role": role, "scale": scale,
                        "temperature": temperature, "samples": len(frame), "patients": frame.patient_id.nunique(),
                        **points[scale], **{n: fits[scale][n] for n in COEFFICIENTS},
                        "regression_status": fits[scale].get("status", "finite_independently_verified")})
                samples = {name: np.full(ITERATIONS, np.nan) for name in points["raw"]}
                coefficients = np.full((ITERATIONS, 2), np.nan)
                invalid = []
                if family == "exact":
                    np.testing.assert_allclose(probabilities_to_logits(scaled), probabilities_to_logits(p) / temperature, rtol=0, atol=1e-10)
                    if fits["raw"].get("status") != "nonestimable":
                        np.testing.assert_allclose([fits["temperature"][n] for n in COEFFICIENTS],
                            [fits["raw"][COEFFICIENTS[0]], fits["raw"][COEFFICIENTS[1]] * temperature], atol=1e-6, rtol=1e-6)
                for iteration, indices in enumerate(_patient_resamples(frame.patient_id.to_numpy())):
                    if len(np.unique(y[indices])) == 2:
                        raw_metrics = probability_metrics(y[indices], p[indices], bins=15)
                        scaled_metrics = probability_metrics(y[indices], scaled[indices], bins=15)
                        for name in samples:
                            samples[name][iteration] = raw_metrics[name] - scaled_metrics[name]
                    if family == "exact":
                        fitted = regression(y[indices], p[indices])
                        if fitted.get("status") == "nonestimable":
                            invalid.append({"replicate": iteration, "reason": fitted["reason"]})
                        else:
                            coefficients[iteration] = [fitted[n] for n in COEFFICIENTS]
                raw_scaled_intervals[key] = {n: interval(v, points["raw"][n] - points["temperature"][n]) for n, v in samples.items()}
                if family == "exact":
                    coefficient_intervals[key] = {"nonestimable_replicates": invalid, "raw": {}, "temperature": {}, "raw_minus_temperature": {}}
                    for j, name in enumerate(COEFFICIENTS):
                        multiplier = 1 if j == 0 else temperature
                        point = fits["raw"][name]
                        for label, factor in (("raw", 1), ("temperature", multiplier), ("raw_minus_temperature", 1-multiplier)):
                            coefficient_intervals[key][label][name] = interval(coefficients[:, j] * factor, None if point is None else point * factor)
                    np.savez_compressed(OUTPUT / f"exact_coefficients_{domain}_{role}.npz", raw=coefficients)
                print(json.dumps({"stage": "calibration_summary_complete", "key": key, "iterations": ITERATIONS,
                                  "primary_coefficient_nonestimable": len(invalid) if family == "exact" else None}), flush=True)
    pd.DataFrame(cal_rows).to_csv(OUTPUT / "calibration_metrics.csv", index=False)
    thresholds = [row for domain in ROLES for row in _threshold_rows(ensembles["exact", domain, "validation_select"],
        ensembles["exact", domain, "validation_select"].probability_1.to_numpy())]
    pd.DataFrame(thresholds).to_csv(OUTPUT / "exact_validation_threshold_sensitivity.csv", index=False)
    write_json(OUTPUT / "calibration_regression_diagnostics.json", {"point_fits": diagnostics,
        "primary_exact_coefficient_intervals": coefficient_intervals, "no_js_coefficient_intervals": "secondary point estimates only",
        "regression": "unpenalized joint intercept/slope, not fixed-slope calibration-in-the-large; never applied to predictions"})
    verify_freeze()
    for name, digest in sources["files"].items():
        assert sha256_file(PRED / name) == digest
    payload = {"status": "complete_pending_independent_QA_and_author_pause_rule", "completed_at": now(),
        "families": ["exact", "no-js"], "seeds": list(ALLOWED_SEEDS), "threshold": .5,
        "primary_contrast": contrasts, "temperatures": temperatures,
        "raw_minus_temperature_probability_intervals": raw_scaled_intervals,
        "bootstrap": {"iterations": ITERATIONS, "seed": BOOTSTRAP_SEED, "cluster": "patient", "confidence": .95,
            "primary_streams": "independent SeedSequence children per database; both arms paired; same draws for equal mean",
            "scope": "conditional on frozen models/temperatures; no seed-population uncertainty, equivalence or multiplicity adjustment"},
        "primary_archived_ensemble_reproduced": primary_replay,
        "patient_overlap_audit": _patient_overlap_audit(manifest),
        "freeze_sha256": sha256_file(FREEZE), "prediction_manifest_sha256": sha256_file(PRED / "prediction_manifest.json"),
        "input_prediction_hashes": hashes, "selection_after_locked_access": False,
        "output_hashes": {p.name: sha256_file(p) for p in OUTPUT.iterdir() if p.is_file()}}
    write_json(OUTPUT / "analysis_b_final.json", payload)
    print(json.dumps({"stage": "analysis_complete_pending_QA", "sha256": sha256_file(OUTPUT / "analysis_b_final.json")}), flush=True)


def launch():
    verify_freeze()
    assert not PRED.exists() and not OUTPUT.exists()
    JOB.mkdir(exist_ok=False)
    with (JOB / "supervisor.log").open("xb") as log:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "worker"], cwd=ROOT,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps({"supervisor_pid": process.pid, "job": str(JOB)}))


def worker():
    status = {"status": "starting", "started_at": now(), "stages": []}
    try:
        with (BASE / "revision_training.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for stage in ("predict", "analyze"):
                command = ["/usr/bin/caffeinate", "-i", sys.executable, "-u", str(Path(__file__).resolve()), stage]
                with (JOB / f"{stage}.log").open("xb") as log:
                    process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                    status.update(status="running", current_stage=stage, process_pid=process.pid, command=command)
                    write_json(JOB / "status.json", status, replace=True)
                    code = process.wait()
                status["stages"].append({"stage": stage, "exit_code": code, "ended_at": now(),
                                         "log_sha256": sha256_file(JOB / f"{stage}.log")})
                if code != 0:
                    raise RuntimeError(f"{stage} exited {code}; no automatic retry")
            status.update(status="completed_pending_independent_audit", ended_at=now(),
                          result_sha256=sha256_file(OUTPUT / "analysis_b_final.json"))
    except Exception as exc:
        status.update(status="failed_needs_review", ended_at=now(), error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        write_json(JOB / "status.json", status, replace=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "predict", "analyze", "launch", "worker"))
    arguments = parser.parse_args()
    globals()[arguments.stage]()
