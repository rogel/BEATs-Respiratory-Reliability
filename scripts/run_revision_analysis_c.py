#!/usr/bin/env python3
"""Separately frozen matched-budget LoRA analysis; no fitting/selection of models.

Reuse audited pure helpers, never the historical all-family analysis main.
Validation must replay the training path before any new calibration/test inference.
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
from torch.utils.data import DataLoader

import run_revision_analysis_b as b
import run_revision_analysis_b_path_corrected as corrected_b
from audit_analysis_b_all_validation_paths import compare, MODES
from compute_feature_stats import feature_config_from_yaml
from predict_revision_ablation import _load_model, _run_dir
from train_gate11a_exactmask_fullft import MemoryTracker, _evaluate, _prediction_frame
from respiratory_sound.calibration import TemperatureScaler, fit_temperature, probability_metrics
from respiratory_sound.gate11a import ALLOWED_SEEDS, sha256_file
from respiratory_sound.post_gate11a import build_waveform_dataset, infer_probabilities, probability_ensemble

ROOT, BASE = b.ROOT, b.BASE
FAMILY = "matched-lora"
FAMILIES = ("exact", FAMILY)
PREFIXES = {20260729: 7, 20260730: 9, 20260731: 6}
FREEZE = BASE / "analysis_c_execution_freeze_2026_09_06.json"
VALIDATION = BASE / "analysis_c_validation_replay_2026_09_06"
PRED = BASE / "predictions/matched_lora_2026_09_06"
OUTPUT = BASE / "analysis_c_2026_09_06"
JOB = BASE / "jobs_2026_09_06/analysis_c"
REPORTS = ROOT.parent / "experiment_plans/2026-09-03"
SPEC = REPORTS / "19_ANALYSIS_C_EXECUTION_SPEC_2026-09-06.md"
write_json, read_json, now = b.write_json, b.read_json, b.now


def expected_cases():
    return {f"{seed}/{domain}" for seed in ALLOWED_SEEDS for domain in b.ROLES}


def require_replay(records, *, require_cross_decisions=True):
    assert set(records) == expected_cases(), "Incomplete validation replay"
    for key, record in records.items():
        for mode in (MODES[0], MODES[2]):
            item = record["modes"][mode]
            assert item["maximum_absolute_probability_difference"] <= 1e-6, (key, mode)
            assert item["changed_0_5_decisions"] == 0, (key, mode)
        assert record["threads1_vs_training_path"]["maximum_absolute_probability_difference"] <= 1e-6
        assert record["threads1_vs_training_path"]["changed_0_5_decisions"] == 0
        if require_cross_decisions:
            assert record["modes"][MODES[1]]["changed_0_5_decisions"] == 0, (
                key, "Cross-path decision change requires author review, not tolerance relaxation")


def freeze():
    assert all(not p.exists() for p in (FREEZE, VALIDATION, PRED, OUTPUT, JOB))
    corrected_b.verify()
    assert sha256_file(ROOT / "scripts/train_gate11a_exactmask_fullft.py") == (
        "db20043fa02e6711dcf808ddc445059275a9b75d4cc73f19c8bcdb6498e6f29d")
    files = dict(read_json(b.FREEZE)["files"])
    paths = set(ROOT / p for p in files)
    paths.update((b.FREEZE, corrected_b.AMENDMENT, corrected_b.SPEC,
                  corrected_b.AUDIT_FREEZE, corrected_b.VALIDATION_AUDIT / "audit_final.json"))
    for source in (corrected_b.AMENDMENT, corrected_b.AUDIT_FREEZE,
                   corrected_b.VALIDATION_AUDIT / "audit_final.json"):
        for name, digest in read_json(source)["files"].items():
            assert sha256_file(ROOT / name) == digest, name
            paths.add(ROOT / name)
    paths.update(ROOT / "scripts" / name for name in (
        "run_revision_analysis_c.py", "audit_analysis_c_final.py", "audit_analysis_b_final.py",
        "train_gate11a_exactmask_fullft.py", "run_revision_analysis_b_path_corrected.py",
        "audit_analysis_b_all_validation_paths.py", "launch_revision_training.py"))
    paths.update((ROOT / "tests/test_revision_analysis_c.py", SPEC,
                  REPORTS / "18_MATCHED_LORA_TRAINING_COMPLETE_REPORT_2026-09-06.md"))
    runs = {}
    for seed in ALLOWED_SEEDS:
        run = _run_dir(ROOT, FAMILY, seed)
        audit_path = BASE / "post_training_audits" / f"matched_lora_seed{seed}.json"
        audit = read_json(audit_path)
        assert audit["status"] == "post_training_audit_passed" and audit["family"] == FAMILY
        assert audit["seed"] == seed and audit["epochs_completed"] == PREFIXES[seed]
        assert audit["optimizer_steps"] == 940 * PREFIXES[seed]
        assert not audit["calibration_accessed"] and not audit["locked_tests_accessed"]
        for name, digest in audit["run_files_sha256"].items():
            assert sha256_file(run / name) == digest, (seed, name)
            paths.add(run / name)
        config = read_json(run / "configuration.json")
        assert config["trainable_parameters"] == 99842 and config["specification"]["fixed_prefix"]
        assert config["specification"]["js_max_weight"] == .4
        paths.add(audit_path)
        paths.add(BASE / f"jobs_2026_09_05/matched_lora_seed{seed}/status.json")
        paths.add(BASE / f"smoke/matched_lora_seed{seed}.json")
        runs[str(seed)] = {"checkpoint_sha256": audit["run_files_sha256"]["best.pt"],
                          "selected_epoch": audit["selected_epoch"], "completed_epochs": PREFIXES[seed],
                          "audit_sha256": sha256_file(audit_path)}
    payload = {"status": "frozen_before_analysis_c_validation_and_new_outcome_inference", "frozen_at": now(),
        "families": list(FAMILIES), "seeds": list(ALLOWED_SEEDS), "matched_lora_runs": runs,
        "analysis_plan_sha256": b.PLAN_SHA256, "primary_exact_checkpoints": b.EXACT_SHA,
        "files": {str(p): sha256_file(p) for p in sorted(paths)}, "threshold": .5,
        "primary_temperatures": b.PROTECTED_T, "bootstrap_seed": b.BOOTSTRAP_SEED,
        "bootstrap_iterations": b.ITERATIONS, "selection_after_locked_access": False,
        "runtime": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__},
        "validation_path": "saved workers4 selection CSV after same-path replay; exact reference already audited",
        "calibration_test_path": {"workers": 0, "cpu_threads": 12, "batch_size": 8, "device": "mps"},
        "calibration_scope": "secondary ensemble points and raw-minus-temperature proper-score/ECE intervals; no new primary coefficient or threshold analysis"}
    write_json(FREEZE, payload)
    print(json.dumps({"freeze": str(FREEZE), "sha256": sha256_file(FREEZE), "files": len(payload["files"])}), flush=True)


def verify():
    frozen = read_json(FREEZE)
    assert frozen["status"] == "frozen_before_analysis_c_validation_and_new_outcome_inference"
    assert frozen["families"] == list(FAMILIES) and frozen["seeds"] == list(ALLOWED_SEEDS)
    assert frozen["threshold"] == .5 and frozen["primary_temperatures"] == b.PROTECTED_T
    for name, digest in frozen["files"].items():
        assert sha256_file(Path(name)) == digest, name
    return frozen


def checked_read(path, manifest, domain, role):
    return b.checked_frame(pd.read_csv(path, dtype={"patient_id": str}), manifest, domain, role)


def validate():
    verify()
    assert torch.backends.mps.is_available() and torch.get_num_threads() == 12
    VALIDATION.mkdir(exist_ok=False)
    device = torch.device("mps")
    manifest_path = ROOT / "data/manifests/cross_domain_binary.csv"
    manifest = pd.read_csv(manifest_path, dtype={"patient_id": str})
    feature = feature_config_from_yaml(ROOT / "configs/data/gate9a_beats.yaml")
    records, files = {}, {}
    for seed in ALLOWED_SEEDS:
        torch.set_num_threads(12)
        model, model_audit = _load_model(ROOT, family=FAMILY, seed=seed, device=device)
        for domain in b.ROLES:
            torch.set_num_threads(12)
            key = f"{seed}/{domain}"
            dataset = build_waveform_dataset(manifest_path=manifest_path, project_root=ROOT,
                feature_config=feature, role="validation_select", domain=domain)
            reference = checked_read(_run_dir(ROOT, FAMILY, seed) / f"best_validation_predictions_{domain}.csv",
                                     manifest, domain, "validation_select")
            frames = {}
            for mode in MODES:
                if mode == MODES[0]:
                    prediction = _evaluate(model, DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4),
                                           device, memory_tracker=MemoryTracker())
                    frame = _prediction_frame(prediction, dataset.rows, domain=domain)
                else:
                    if mode == MODES[2]:
                        torch.set_num_threads(1)
                    frame = infer_probabilities(model, dataset, device=device, batch_size=8)
                frame = b.checked_frame(frame, manifest, domain, "validation_select")
                frames[mode] = frame
                path = VALIDATION / f"seed{seed}_{domain}_{mode}.csv"
                frame.to_csv(path, index=False)
                files[path.name] = sha256_file(path)
            records[key] = {"events": len(reference), "model": model_audit,
                "modes": {mode: compare(frame, reference) for mode, frame in frames.items()},
                "threads1_vs_training_path": compare(frames[MODES[2]], frames[MODES[0]])}
            write_json(VALIDATION / "progress.json", {"status": "running_pending_gate", "updated_at": now(),
                "completed_cases": len(records), "total_cases": 6, "records": records}, replace=True)
            print(json.dumps({"case": key, "completed_cases": len(records),
                "same_path_max_difference": records[key]["modes"][MODES[0]]["maximum_absolute_probability_difference"],
                "cross_path_changed_decisions": records[key]["modes"][MODES[1]]["changed_0_5_decisions"]}), flush=True)
        del model
        torch.mps.empty_cache()
    torch.set_num_threads(12)
    require_replay(records, require_cross_decisions=False)
    cross_flips = sum(r["modes"][MODES[1]]["changed_0_5_decisions"] for r in records.values())
    write_json(VALIDATION / "audit_final.json", {"status": "complete_same_path_replay_passed", "completed_at": now(),
        "records": records, "files": files, "freeze_sha256": sha256_file(FREEZE), "threshold": .5,
        "calibration_accessed": False, "locked_tests_accessed": False, "training_or_selection": False,
        "cross_path_changed_decisions": cross_flips, "automatic_predict_gate_passed": cross_flips == 0})
    require_replay(records)
    verify()


def verify_replay():
    audit = read_json(VALIDATION / "audit_final.json")
    assert audit["status"] == "complete_same_path_replay_passed"
    assert audit["freeze_sha256"] == sha256_file(FREEZE) and audit["automatic_predict_gate_passed"]
    assert not audit["calibration_accessed"] and not audit["locked_tests_accessed"]
    require_replay(audit["records"])
    for name, digest in audit["files"].items():
        assert sha256_file(VALIDATION / name) == digest, name
    return audit


def predict():
    verify()
    verify_replay()
    assert torch.backends.mps.is_available() and torch.get_num_threads() == 12
    device = torch.device("mps")
    PRED.mkdir(parents=True, exist_ok=False)
    manifest_path = ROOT / "data/manifests/cross_domain_binary.csv"
    manifest = pd.read_csv(manifest_path, dtype={"patient_id": str})
    feature = feature_config_from_yaml(ROOT / "configs/data/gate9a_beats.yaml")
    files, runs = {}, {}
    for seed in ALLOWED_SEEDS:
        for domain in b.ROLES:
            source = _run_dir(ROOT, FAMILY, seed) / f"best_validation_predictions_{domain}.csv"
            checked_read(source, manifest, domain, "validation_select")
            path = PRED / f"seed{seed}_{domain}_validation_select.csv"
            shutil.copy2(source, path)
            assert sha256_file(source) == sha256_file(path)
            files[path.name] = sha256_file(path)
        model, runs[str(seed)] = _load_model(ROOT, family=FAMILY, seed=seed, device=device)
        for domain, roles in b.ROLES.items():
            for role in roles:
                if role == "validation_select":
                    continue
                dataset = build_waveform_dataset(manifest_path=manifest_path, project_root=ROOT,
                    feature_config=feature, role=role, domain=domain)
                frame = b.checked_frame(infer_probabilities(model, dataset, device=device, batch_size=8), manifest, domain, role)
                path = PRED / f"seed{seed}_{domain}_{role}.csv"
                frame.to_csv(path, index=False)
                files[path.name] = sha256_file(path)
                print(json.dumps({"seed": seed, "dataset": domain, "role": role, "events": len(frame),
                                  "sha256": files[path.name]}), flush=True)
        del model
        torch.mps.empty_cache()
    for domain, roles in b.ROLES.items():
        for role in roles:
            frames = [checked_read(PRED / f"seed{s}_{domain}_{role}.csv", manifest, domain, role) for s in ALLOWED_SEEDS]
            frame = b.checked_frame(probability_ensemble(frames), manifest, domain, role)
            path = PRED / f"ensemble_{domain}_{role}.csv"
            frame.to_csv(path, index=False)
            files[path.name] = sha256_file(path)
    verify()
    write_json(PRED / "prediction_manifest.json", {"status": "complete_pending_analysis_audit", "family": FAMILY,
        "seeds": list(ALLOWED_SEEDS), "roles": b.ROLES, "runs": runs, "files": files,
        "freeze_sha256": sha256_file(FREEZE), "validation_replay_sha256": sha256_file(VALIDATION / "audit_final.json"),
        "device": "mps", "batch_size": 8, "cpu_threads": 12, "calibration_test_workers": 0,
        "validation_reused_from_frozen_selection": True, "calibration_accessed": True,
        "locked_tests_accessed": True, "selection_after_locked_access": False})


def paired_contrasts(ensembles):
    contrasts, draws = {}, {}
    for (domain, role), sequence in zip(b.PRIMARY.items(), np.random.SeedSequence(b.BOOTSTRAP_SEED).spawn(2), strict=True):
        left, right = ensembles["exact", domain, role], ensembles[FAMILY, domain, role]
        assert left[["sample_id", "patient_id", "target"]].equals(right[["sample_id", "patient_id", "target"]])
        y, p, q = left.target.to_numpy(), left.probability_1.to_numpy(), right.probability_1.to_numpy()
        a, c = b._as(y, p), b._as(y, q)
        values = np.full(b.ITERATIONS, np.nan)
        for i, ix in enumerate(b._patient_resamples(left.patient_id.to_numpy(), generator=np.random.default_rng(sequence))):
            if len(np.unique(y[ix])) == 2:
                values[i] = b._as(y[ix], p[ix]) - b._as(y[ix], q[ix])
        draws[domain] = values
        contrasts[domain] = {"role": role, "fullft_average_score": a, "matched_lora_average_score": c,
                             "fullft_minus_matched_lora": b.interval(values, a-c)}
    draws["equal_database_mean"] = np.mean(np.stack(list(draws.values())), axis=0)
    contrasts["equal_database_mean"] = b.interval(draws["equal_database_mean"],
        np.mean([contrasts[d]["fullft_minus_matched_lora"]["estimate"] for d in b.PRIMARY]))
    return contrasts, draws


def analyze():
    verify()
    verify_replay()
    sources = read_json(PRED / "prediction_manifest.json")
    assert sources["family"] == FAMILY and sources["seeds"] == list(ALLOWED_SEEDS)
    assert sources["freeze_sha256"] == sha256_file(FREEZE)
    for name, digest in sources["files"].items():
        assert sha256_file(PRED / name) == digest, name
    OUTPUT.mkdir(exist_ok=False)
    manifest = pd.read_csv(ROOT / "data/manifests/cross_domain_binary.csv", dtype={"patient_id": str})
    ensembles, rows, hashes, primary_replay = {}, [], {}, {}
    for family in FAMILIES:
        for domain, roles in b.ROLES.items():
            for role in roles:
                frames = []
                for seed in ALLOWED_SEEDS:
                    path = b._exact_path(ROOT, seed, domain, role) if family == "exact" else PRED / f"seed{seed}_{domain}_{role}.csv"
                    frame = checked_read(path, manifest, domain, role)
                    frames.append(frame)
                    hashes[str(path)] = sha256_file(path)
                    rows.append({"family": family, "member": f"seed{seed}", "dataset": domain, "role": role,
                        "samples": len(frame), "patients": frame.patient_id.nunique(),
                        **{k:v for k,v in b._classification(frame, frame.probability_1.to_numpy()).items() if isinstance(v,float)}})
                ensemble = b.checked_frame(probability_ensemble(frames), manifest, domain, role)
                ensembles[family,domain,role] = ensemble
                rows.append({"family":family,"member":"ensemble","dataset":domain,"role":role,
                    "samples":len(ensemble),"patients":ensemble.patient_id.nunique(),
                    **{k:v for k,v in b._classification(ensemble,ensemble.probability_1.to_numpy()).items() if isinstance(v,float)}})
                archived_path = None
                if family == FAMILY:
                    archived_path = PRED / f"ensemble_{domain}_{role}.csv"
                elif role != "validation_select":
                    subdir = "calibration_predictions" if role == "calibration" else "locked_predictions"
                    archived_path = ROOT / "artifacts/post_gate11a" / subdir / f"fullft_ensemble_{domain}_{role}.csv"
                if archived_path is not None:
                    archived = checked_read(archived_path,manifest,domain,role)
                    np.testing.assert_allclose(ensemble.probability_1,archived.probability_1,rtol=0,atol=1e-14)
                    assert np.array_equal(ensemble.prediction,archived.prediction)
                    if family == "exact":
                        primary_replay[f"{domain}/{role}"] = True
    pd.DataFrame(rows).to_csv(OUTPUT / "classification_metrics.csv",index=False)
    contrasts,draws = paired_contrasts(ensembles)
    np.savez_compressed(OUTPUT / "classification_bootstrap_draws.npz",**draws)
    print(json.dumps({"stage":"primary_contrast_computed_pending_QA","contrast":contrasts}),flush=True)
    temperatures,cal_rows,diagnostics,changes = {},[],{},{}
    for family in FAMILIES:
        for domain,roles in b.ROLES.items():
            cal = ensembles[family,domain,"calibration"]
            temperature = b.PROTECTED_T[domain] if family == "exact" else float(fit_temperature(cal.target.to_numpy(),cal.probability_1.to_numpy()).temperature)
            assert np.isfinite(temperature) and temperature > 0
            temperatures[f"{family}/{domain}"] = temperature
            for role in roles:
                frame = ensembles[family,domain,role]
                y,p = frame.target.to_numpy(),frame.probability_1.to_numpy()
                scaled = TemperatureScaler(temperature).transform(p)
                assert np.isfinite(scaled).all() and np.array_equal(p>=.5,scaled>=.5)
                key = f"{family}/{domain}/{role}"
                points = {}
                for label,values in (("raw",p),("temperature",scaled)):
                    points[label] = probability_metrics(y,values,bins=15)
                    fit = b.regression(y,values,independent=True)
                    diagnostics[f"{key}/{label}"] = fit
                    cal_rows.append({"family":family,"dataset":domain,"role":role,"scale":label,
                        "temperature":temperature,"samples":len(frame),"patients":frame.patient_id.nunique(),
                        **points[label],**{name:fit[name] for name in b.COEFFICIENTS},
                        "regression_status":fit.get("status","finite_independently_verified")})
                values = {name:np.full(b.ITERATIONS,np.nan) for name in points["raw"]}
                for i,ix in enumerate(b._patient_resamples(frame.patient_id.to_numpy())):
                    if len(np.unique(y[ix])) == 2:
                        a,c = probability_metrics(y[ix],p[ix],bins=15),probability_metrics(y[ix],scaled[ix],bins=15)
                        for name in values:
                            values[name][i] = a[name]-c[name]
                changes[key] = {name:b.interval(v,points["raw"][name]-points["temperature"][name]) for name,v in values.items()}
                print(json.dumps({"stage":"secondary_calibration_complete","key":key}),flush=True)
    pd.DataFrame(cal_rows).to_csv(OUTPUT / "calibration_metrics.csv",index=False)
    write_json(OUTPUT / "calibration_regression_diagnostics.json",{"point_fits":diagnostics,
        "coefficient_intervals":"no new intervals here; protected primary expanded calibration already completed in Analysis B",
        "regression":"unpenalized joint intercept/slope; not used to transform model predictions"})
    verify()
    payload = {"status":"complete_pending_independent_QA_and_scientific_review","completed_at":now(),
        "families":list(FAMILIES),"seeds":list(ALLOWED_SEEDS),"threshold":.5,"primary_contrast":contrasts,
        "temperatures":temperatures,"raw_minus_temperature_probability_intervals":changes,
        "bootstrap":{"iterations":b.ITERATIONS,"seed":b.BOOTSTRAP_SEED,"cluster":"patient","confidence":.95,
            "primary_streams":"independent SeedSequence children per database, paired arms, same draws for equal mean",
            "scope":"conditional on frozen models/temperatures; no seed-population inference, equivalence or multiplicity adjustment"},
        "primary_archived_ensemble_reproduced":primary_replay,"patient_overlap_audit":b._patient_overlap_audit(manifest),
        "freeze_sha256":sha256_file(FREEZE),"prediction_manifest_sha256":sha256_file(PRED/"prediction_manifest.json"),
        "validation_replay_sha256":sha256_file(VALIDATION/"audit_final.json"),"input_prediction_hashes":hashes,
        "selection_after_locked_access":False,"comparison_scope":"fixed rank/layers and matched update/LR schedule, not LoRA-optimal or universally causal",
        "output_hashes":{p.name:sha256_file(p) for p in OUTPUT.iterdir() if p.is_file()}}
    write_json(OUTPUT / "analysis_c_final.json",payload)


def launch():
    verify()
    assert all(not p.exists() for p in (VALIDATION,PRED,OUTPUT,JOB))
    JOB.mkdir(parents=True,exist_ok=False)
    with (JOB/"supervisor.log").open("xb") as log:
        process = subprocess.Popen([sys.executable,str(Path(__file__).resolve()),"worker"],cwd=ROOT,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps({"supervisor_pid":process.pid,"job":str(JOB)}),flush=True)


def worker():
    status = {"status":"starting","started_at":now(),"stages":[],"freeze_sha256":sha256_file(FREEZE)}
    try:
        with (BASE/"revision_training.lock").open("a+") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            for stage in ("validate","predict","analyze"):
                command = ["/usr/bin/caffeinate","-i",sys.executable,"-u",str(Path(__file__).resolve()),stage]
                with (JOB/f"{stage}.log").open("xb") as log:
                    process = subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                    status.update(status="running",current_stage=stage,process_pid=process.pid,command=command)
                    write_json(JOB/"status.json",status,replace=True)
                    code = process.wait()
                status["stages"].append({"stage":stage,"exit_code":code,"ended_at":now(),"log_sha256":sha256_file(JOB/f"{stage}.log")})
                if code != 0:
                    raise RuntimeError(f"{stage} exited {code}; no automatic retry")
            status.update(status="completed_pending_independent_audit",ended_at=now(),result_sha256=sha256_file(OUTPUT/"analysis_c_final.json"))
    except Exception as exc:
        status.update(status="failed_needs_review",ended_at=now(),error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        write_json(JOB/"status.json",status,replace=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage",choices=("freeze","validate","predict","analyze","launch","worker"))
    args = parser.parse_args()
    globals()[args.stage]()
