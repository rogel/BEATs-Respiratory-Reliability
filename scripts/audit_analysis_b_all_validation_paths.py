#!/usr/bin/env python3
"""Author-approved six-checkpoint/two-database validation path audit.

No training or calibration/test inference. Retains the completed first-case
diagnostic and computes the other eleven cases under a prospective source freeze.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from compute_feature_stats import feature_config_from_yaml
from predict_revision_ablation import _load_model, _run_dir
from run_revision_analysis_b import (
    ROOT, BASE, FREEZE, checked_frame, verify_freeze, write_json, now,
)
from train_gate11a_exactmask_fullft import MemoryTracker, _evaluate, _prediction_frame
from respiratory_sound.calibration import probability_metrics
from respiratory_sound.gate11a import ALLOWED_SEEDS, sha256_file
from respiratory_sound.post_gate11a import build_waveform_dataset, infer_probabilities, load_frozen_model
from analyze_revision_ablation import _classification

OUT = BASE / "analysis_b_all_validation_paths_2026_09_05"
AUDIT_FREEZE = BASE / "analysis_b_validation_path_audit_freeze_2026_09_05.json"
FIRST = BASE / "analysis_b_replay_diagnostic_2026_09_05"
MODES = ("training_evaluate_workers4", "archival_infer_workers0", "infer_workers0_threads1")


def metrics(frame):
    y, p = frame.target.to_numpy(), frame.probability_1.to_numpy()
    return {**{k: v for k, v in _classification(frame, p).items() if isinstance(v, float)},
            **probability_metrics(y, p, bins=15)}


def compare(actual, reference):
    p, q = actual.probability_1.to_numpy(), reference.probability_1.to_numpy()
    delta = np.abs(p-q)
    left, right = metrics(actual), metrics(reference)
    return {"maximum_absolute_probability_difference": float(delta.max()),
            "mean_absolute_probability_difference": float(delta.mean()),
            "events_above_1e_6": int(np.sum(delta > 1e-6)),
            "changed_0_5_decisions": int(np.sum(actual.prediction != reference.prediction)),
            "metrics_actual": left, "metrics_reference": right,
            "metric_differences_actual_minus_reference": {k: left[k]-right[k] for k in left}}


def make_freeze():
    verify_freeze()
    assert not OUT.exists() and not AUDIT_FREEZE.exists()
    first = json.loads((FIRST / "diagnostic.json").read_text())
    assert sha256_file(FIRST / "diagnostic.json") == "35a79d65d78ededca3cc2e03fdb203484dca7b23f7f621d8f8930c647ca14d2e"
    files = {str(Path(__file__).resolve()): sha256_file(Path(__file__).resolve()),
             str(FIRST / "diagnostic.json"): sha256_file(FIRST / "diagnostic.json")}
    for name, digest in first["files"].items():
        assert sha256_file(FIRST / name) == digest
        files[str(FIRST / name)] = digest
    write_json(AUDIT_FREEZE, {"status": "frozen_before_extended_validation_only_audit", "frozen_at": now(),
        "author_authorization": "2026-09-05: CPU-thread numerical difference accepted; proceed with recommended validation audit and prospective implementation correction",
        "base_execution_freeze_sha256": sha256_file(FREEZE), "files": files,
        "families": ["exact", "no-js"], "seeds": list(ALLOWED_SEEDS), "domains": ["icbhi2017", "sprsound2022"],
        "role": "validation_select", "modes": list(MODES), "same_path_tolerance": 1e-6,
        "same_path_decisions_must_match": True, "classification_threshold": .5,
        "inference_device": "mps", "batch_size": 8, "default_cpu_threads": 12,
        "calibration_or_heldout_inference_allowed": False, "training_allowed": False,
        "reuse_first_case": "no-js/20260729/icbhi2017; verified original diagnostic and all CSV hashes"})
    print(json.dumps({"freeze": str(AUDIT_FREEZE), "sha256": sha256_file(AUDIT_FREEZE)}), flush=True)


def verify_audit_freeze():
    verify_freeze()
    freeze = json.loads(AUDIT_FREEZE.read_text())
    assert freeze["base_execution_freeze_sha256"] == sha256_file(FREEZE)
    for name, digest in freeze["files"].items():
        assert sha256_file(Path(name)) == digest, name


def run():
    verify_audit_freeze()
    OUT.mkdir(exist_ok=False)
    device = torch.device("mps")
    assert torch.backends.mps.is_available() and torch.get_num_threads() == 12
    manifest_path = ROOT / "data/manifests/cross_domain_binary.csv"
    manifest = pd.read_csv(manifest_path, dtype={"patient_id": str})
    datasets = {domain: build_waveform_dataset(manifest_path=manifest_path, project_root=ROOT,
        feature_config=feature_config_from_yaml(ROOT / "configs/data/gate9a_beats.yaml"),
        role="validation_select", domain=domain) for domain in ("icbhi2017", "sprsound2022")}
    records, all_hashes = {}, {}
    for family in ("exact", "no-js"):
        for seed in ALLOWED_SEEDS:
            torch.set_num_threads(12)
            model = (load_frozen_model(ROOT, family="fullft", seed=seed, device=device) if family == "exact"
                     else _load_model(ROOT, family="no-js", seed=seed, device=device)[0])
            run_dir = ROOT / f"runs/gate11a_exactmask_fullft_seed{seed}" if family == "exact" else _run_dir(ROOT, family, seed)
            for domain, dataset in datasets.items():
                torch.set_num_threads(12)
                key = f"{family}/{seed}/{domain}"
                old = checked_frame(pd.read_csv(run_dir / f"best_validation_predictions_{domain}.csv",
                    dtype={"patient_id": str}), manifest, domain, "validation_select")
                results, frames = {}, {}
                reuse = family == "no-js" and seed == 20260729 and domain == "icbhi2017"
                for mode in MODES:
                    if reuse:
                        path = FIRST / f"{mode}.csv"
                        frame = pd.read_csv(path, dtype={"patient_id": str})
                    else:
                        if mode == "training_evaluate_workers4":
                            prediction = _evaluate(model, DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4),
                                device, memory_tracker=MemoryTracker())
                            frame = _prediction_frame(prediction, dataset.rows, domain=domain)
                        else:
                            if mode == "infer_workers0_threads1":
                                torch.set_num_threads(1)
                            frame = infer_probabilities(model, dataset, device=device, batch_size=8)
                        path = OUT / f"{family.replace('-', '_')}_seed{seed}_{domain}_{mode}.csv"
                        frame.to_csv(path, index=False)
                    frame = checked_frame(frame, manifest, domain, "validation_select")
                    all_hashes[str(path.relative_to(ROOT))] = sha256_file(path)
                    results[mode] = compare(frame, old)
                    frames[mode] = frame
                    if mode != "archival_infer_workers0":
                        assert results[mode]["maximum_absolute_probability_difference"] <= 1e-6, (key, mode, results[mode])
                        assert results[mode]["changed_0_5_decisions"] == 0, (key, mode, results[mode])
                records[key] = {"events": len(old), "patients": old.patient_id.nunique(), "modes": results,
                    "threads1_vs_training_path": compare(frames["infer_workers0_threads1"], frames["training_evaluate_workers4"]),
                    "reused_first_diagnostic": reuse}
                assert records[key]["threads1_vs_training_path"]["maximum_absolute_probability_difference"] <= 1e-6
                write_json(OUT / "progress.json", {"status": "running", "updated_at": now(), "completed_cases": len(records),
                    "total_cases": 12, "records": records}, replace=True)
                print(json.dumps({"case": key, "completed_cases": len(records), "events": len(old),
                    "same_path_max_difference": results[MODES[0]]["maximum_absolute_probability_difference"],
                    "cross_path_max_difference": results[MODES[1]]["maximum_absolute_probability_difference"],
                    "cross_path_changed_decisions": results[MODES[1]]["changed_0_5_decisions"]}), flush=True)
            del model
            torch.mps.empty_cache()
    verify_audit_freeze()
    assert len(records) == 12
    for name, digest in all_hashes.items():
        assert sha256_file(ROOT / name) == digest
    write_json(OUT / "audit_final.json", {"status": "complete_all_same_path_checks_passed", "completed_at": now(),
        "records": records, "files": all_hashes, "audit_freeze_sha256": sha256_file(AUDIT_FREEZE),
        "base_execution_freeze_sha256": sha256_file(FREEZE), "no_training_or_selection": True,
        "calibration_accessed": False, "heldout_accessed": False, "threshold": .5})
    print(json.dumps({"status": "complete", "result": str(OUT / "audit_final.json"),
                      "sha256": sha256_file(OUT / "audit_final.json")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "run"))
    args = parser.parse_args()
    make_freeze() if args.stage == "freeze" else run()
