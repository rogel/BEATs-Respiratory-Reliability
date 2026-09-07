#!/usr/bin/env python3
"""Validation-only diagnostics after the frozen replay gate stopped Analysis B.

No retraining, calibration/test access, threshold change or formal output update.
"""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from compute_feature_stats import feature_config_from_yaml
from predict_revision_ablation import _load_model, _run_dir
from run_revision_analysis_b import checked_frame, verify_freeze, write_json, now
from train_gate11a_exactmask_fullft import MemoryTracker, _evaluate, _prediction_frame
from respiratory_sound.gate11a import sha256_file
from respiratory_sound.post_gate11a import build_waveform_dataset, infer_probabilities

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/revision_2026_09_03/analysis_b_replay_diagnostic_2026_09_05"


def main():
    verify_freeze()
    OUT.mkdir(exist_ok=False)
    seed, domain, role = 20260729, "icbhi2017", "validation_select"
    manifest_path = ROOT / "data/manifests/cross_domain_binary.csv"
    manifest = pd.read_csv(manifest_path, dtype={"patient_id": str})
    dataset = build_waveform_dataset(manifest_path=manifest_path, project_root=ROOT,
        feature_config=feature_config_from_yaml(ROOT / "configs/data/gate9a_beats.yaml"), role=role, domain=domain)
    device = torch.device("mps")
    assert torch.backends.mps.is_available()
    model, model_audit = _load_model(ROOT, family="no-js", seed=seed, device=device)
    old = checked_frame(pd.read_csv(_run_dir(ROOT, "no-js", seed) / f"best_validation_predictions_{domain}.csv",
                                   dtype={"patient_id": str}), manifest, domain, role)
    results, frames = {}, {}
    default_threads = torch.get_num_threads()
    for mode in ("training_evaluate_workers4", "archival_infer_workers0", "infer_workers0_threads1"):
        if mode == "training_evaluate_workers4":
            prediction = _evaluate(model, DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4),
                                   device, memory_tracker=MemoryTracker())
            frame = _prediction_frame(prediction, dataset.rows, domain=domain)
        else:
            if mode == "infer_workers0_threads1":
                torch.set_num_threads(1)
            frame = infer_probabilities(model, dataset, device=device, batch_size=8)
        frame = checked_frame(frame, manifest, domain, role)
        difference = np.abs(frame.probability_1.to_numpy() - old.probability_1.to_numpy())
        results[mode] = {"maximum_absolute_difference_from_saved": float(difference.max()),
                         "mean_absolute_difference": float(difference.mean()),
                         "changed_classification_decisions": int(np.sum(frame.prediction != old.prediction)),
                         "above_1e_6": int(np.sum(difference > 1e-6)),
                         "process_threads": torch.get_num_threads()}
        frame.to_csv(OUT / f"{mode}.csv", index=False)
        frames[mode] = frame
        print(json.dumps({"mode": mode, **results[mode]}), flush=True)
    pairs = {}
    for left in frames:
        for right in frames:
            if left < right:
                pairs[f"{left}__{right}"] = float(np.max(np.abs(frames[left].probability_1 - frames[right].probability_1)))
    # Inspect waveform numerical differences for every validation event at CPU thread settings.
    waveform_differences = []
    for index in range(len(dataset)):
        torch.set_num_threads(default_threads)
        a = dataset[index]
        torch.set_num_threads(1)
        b = dataset[index]
        assert a[2:] == b[2:] and torch.equal(a[1], b[1])
        diff = float(torch.max(torch.abs(a[0] - b[0])))
        if diff:
            waveform_differences.append({"sample_id": a[3], "max_absolute_waveform_difference": diff})
    verify_freeze()
    write_json(OUT / "diagnostic.json", {"created_at": now(), "seed": seed, "domain": domain,
        "role": role, "model_audit": model_audit, "runtime_python": sys.version,
        "default_cpu_threads": default_threads, "probability_results": results, "pairwise_max_difference": pairs,
        "waveform_difference_count": len(waveform_differences), "waveform_differences": waveform_differences,
        "calibration_accessed": False, "locked_tests_accessed": False, "training_performed": False,
        "source_sha256": sha256_file(Path(__file__).resolve()),
        "files": {p.name: sha256_file(p) for p in OUT.iterdir() if p.is_file()}})


if __name__ == "__main__":
    main()
