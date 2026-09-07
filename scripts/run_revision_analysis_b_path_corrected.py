#!/usr/bin/env python3
"""Prospective, author-approved path correction; original analysis code intact.

Validation uses the frozen training/selection predictions after twelve same-path
audits. Calibration/tests use historical archival inference: batch8, workers0,
CPU12 threads and MPS. Statistics delegate to the frozen Analysis B implementation
with only additive input/output directory bindings; no estimator changes.
"""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

import run_revision_analysis_b as original
from audit_analysis_b_all_validation_paths import AUDIT_FREEZE, OUT as VALIDATION_AUDIT, verify_audit_freeze
from compute_feature_stats import feature_config_from_yaml
from predict_revision_ablation import _load_model, _run_dir
from respiratory_sound.gate11a import ALLOWED_SEEDS, sha256_file
from respiratory_sound.post_gate11a import build_waveform_dataset, infer_probabilities, probability_ensemble

ROOT, BASE = original.ROOT, original.BASE
AMENDMENT = BASE / "analysis_b_path_correction_freeze_2026_09_05.json"
PRED = BASE / "predictions/no_js_path_corrected_2026_09_05"
OUTPUT = BASE / "analysis_b_path_corrected_2026_09_05"
JOB = BASE / "jobs_2026_09_05/analysis_b_path_corrected"
SPEC = ROOT.parent / "experiment_plans/2026-09-03/13_ANALYSIS_B_PATH_AMENDMENT_2026-09-05.md"
BASE_FREEZE_SHA = "16c9dc34ad7aae4c2c055fff35f6c4452f4905c1e8d501a7a6a5f5df7f778f66"
write_json, now = original.write_json, original.now


def require_validation_audit(audit):
    assert audit["status"] == "complete_all_same_path_checks_passed"
    assert not audit["calibration_accessed"] and not audit["heldout_accessed"] and audit["no_training_or_selection"]
    keys = {f"{f}/{s}/{d}" for f in ("exact", "no-js") for s in ALLOWED_SEEDS for d in original.ROLES}
    assert set(audit["records"]) == keys
    for key, record in audit["records"].items():
        for mode in ("training_evaluate_workers4", "infer_workers0_threads1"):
            checked = record["modes"][mode]
            assert checked["maximum_absolute_probability_difference"] <= 1e-6, (key, mode)
            assert checked["changed_0_5_decisions"] == 0, (key, mode)
        assert record["threads1_vs_training_path"]["maximum_absolute_probability_difference"] <= 1e-6
    return audit


def verify_authorized_inputs():
    assert sha256_file(original.FREEZE) == BASE_FREEZE_SHA
    verify_audit_freeze()
    audit = require_validation_audit(json.loads((VALIDATION_AUDIT / "audit_final.json").read_text()))
    for name, digest in audit["files"].items():
        assert sha256_file(ROOT / name) == digest, name
    return audit


def freeze():
    audit = verify_authorized_inputs()
    assert not AMENDMENT.exists() and not PRED.exists() and not OUTPUT.exists() and not JOB.exists()
    paths = [Path(__file__).resolve(), ROOT / "scripts/audit_analysis_b_all_validation_paths.py",
             ROOT / "tests/test_revision_analysis_b_path_correction.py", SPEC, AUDIT_FREEZE,
             VALIDATION_AUDIT / "audit_final.json", original.FREEZE,
             ROOT / "scripts/audit_analysis_b_final.py", ROOT / "tests/test_analysis_b_independent_audit.py"]
    write_json(AMENDMENT, {"status": "prospective_author_approved_implementation_path_correction", "frozen_at": now(),
        "author_authorization": "2026-09-05 approved report11 recommended thread-path audit and correction",
        "base_execution_freeze_sha256": BASE_FREEZE_SHA,
        "validated_case_count": len(audit["records"]), "files": {str(p): sha256_file(p) for p in paths},
        "changes": ["training-path validation replay proven by extended audit", "reuse original selection-validation CSVs in both arms",
                    "new additive output paths; original failed entry/artifacts retained"],
        "calibration_and_test_path": {"batch_size": 8, "workers": 0, "cpu_threads": 12, "device": "mps"},
        "statistical_implementation": "unchanged original.run_revision_analysis_b.analyze, with additive PRED/OUTPUT paths only",
        "preserved": ["selected models", "seeds", "patient roles", "threshold0.5", "primary predictions/results/temperatures",
                      "all estimators/metrics/bootstrap", "no held-out selection"],
        "no_js_prediction_root": str(PRED), "output_root": str(OUTPUT)})
    print(json.dumps({"freeze": str(AMENDMENT), "sha256": sha256_file(AMENDMENT)}), flush=True)


def verify():
    verify_authorized_inputs()
    value = json.loads(AMENDMENT.read_text())
    assert value["status"] == "prospective_author_approved_implementation_path_correction"
    assert value["base_execution_freeze_sha256"] == BASE_FREEZE_SHA
    for path, digest in value["files"].items():
        assert sha256_file(Path(path)) == digest, path


def predict():
    verify()
    assert torch.get_num_threads() == 12 and torch.backends.mps.is_available()
    device = torch.device("mps")
    PRED.mkdir(parents=True, exist_ok=False)
    manifest_path = ROOT / "data/manifests/cross_domain_binary.csv"
    manifest = pd.read_csv(manifest_path, dtype={"patient_id": str})
    files, runs, copied = {}, {}, {}
    for seed in ALLOWED_SEEDS:
        for domain in original.ROLES:
            source = _run_dir(ROOT, "no-js", seed) / f"best_validation_predictions_{domain}.csv"
            frame = original.checked_frame(pd.read_csv(source, dtype={"patient_id": str}), manifest, domain, "validation_select")
            target = PRED / f"seed{seed}_{domain}_validation_select.csv"
            shutil.copy2(source, target)
            assert sha256_file(target) == sha256_file(source)
            files[target.name] = sha256_file(target)
            copied[f"{seed}/{domain}"] = {"source": str(source.relative_to(ROOT)), "sha256": files[target.name],
                                         "events": len(frame), "same_path_replay_passed_in_extended_audit": True}
    write_json(PRED / "validation_reuse_audit.json", copied)
    feature = feature_config_from_yaml(ROOT / "configs/data/gate9a_beats.yaml")
    for seed in ALLOWED_SEEDS:
        model, model_audit = _load_model(ROOT, family="no-js", seed=seed, device=device)
        runs[str(seed)] = model_audit
        for domain, roles in original.ROLES.items():
            for role in roles:
                if role == "validation_select":
                    continue
                dataset = build_waveform_dataset(manifest_path=manifest_path, project_root=ROOT, feature_config=feature,
                                                  role=role, domain=domain)
                frame = original.checked_frame(infer_probabilities(model, dataset, device=device, batch_size=8), manifest, domain, role)
                target = PRED / f"seed{seed}_{domain}_{role}.csv"
                frame.to_csv(target, index=False)
                files[target.name] = sha256_file(target)
                print(json.dumps({"seed": seed, "domain": domain, "role": role, "events": len(frame),
                                  "sha256": files[target.name]}), flush=True)
        del model
        torch.mps.empty_cache()
    for domain, roles in original.ROLES.items():
        for role in roles:
            frames = [pd.read_csv(PRED / f"seed{s}_{domain}_{role}.csv", dtype={"patient_id": str}) for s in ALLOWED_SEEDS]
            frame = original.checked_frame(probability_ensemble(frames), manifest, domain, role)
            target = PRED / f"ensemble_{domain}_{role}.csv"
            frame.to_csv(target, index=False)
            files[target.name] = sha256_file(target)
    verify()
    write_json(PRED / "prediction_manifest.json", {"status": "complete_pending_analysis_audit", "family": "no-js",
        "seeds": list(ALLOWED_SEEDS), "roles": original.ROLES, "device": "mps", "batch_size": 8,
        "cpu_threads": 12, "calibration_test_workers": 0, "runs": runs, "files": files,
        "freeze_sha256": BASE_FREEZE_SHA, "path_amendment_sha256": sha256_file(AMENDMENT),
        "validation_reused_from_frozen_selection": copied, "calibration_accessed": True,
        "locked_tests_accessed": True, "selection_after_locked_access": False})


def analyze():
    verify()
    source = json.loads((PRED / "prediction_manifest.json").read_text())
    assert source["path_amendment_sha256"] == sha256_file(AMENDMENT)
    # The statistics, protected inputs, bootstrap and model selection are unchanged.
    original.PRED = PRED
    original.OUTPUT = OUTPUT
    original.analyze()
    verify()
    write_json(OUTPUT / "path_correction_execution_receipt.json", {"status": "complete_pending_independent_QA",
        "completed_at": now(), "base_execution_freeze_sha256": BASE_FREEZE_SHA,
        "path_amendment_sha256": sha256_file(AMENDMENT),
        "validation_path_audit_sha256": sha256_file(VALIDATION_AUDIT / "audit_final.json"),
        "prediction_manifest_sha256": sha256_file(PRED / "prediction_manifest.json"),
        "analysis_b_final_sha256": sha256_file(OUTPUT / "analysis_b_final.json"),
        "original_statistics_code_unchanged": True, "same_role_paths_matched_between_arms": True})


def launch():
    verify()
    assert not PRED.exists() and not OUTPUT.exists()
    JOB.mkdir(exist_ok=False)
    with (JOB / "supervisor.log").open("xb") as log:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "worker"], cwd=ROOT,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps({"supervisor_pid": process.pid, "job": str(JOB)}), flush=True)


def worker():
    status = {"status": "starting", "started_at": now(), "stages": [], "path_amendment_sha256": sha256_file(AMENDMENT)}
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
    args = parser.parse_args()
    globals()[args.stage]()
