#!/usr/bin/env python3
"""Audit one completed frozen revision run before advancing to another seed.

No audio inference, calibration/test prediction access, model changes, or new
selection occurs. Stored validation metrics are independently reconstructed.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
import yaml

from analyze_revision_analysis_a import _patient_overlap_audit
from respiratory_sound.gate11a import CLASS_NAMES, sha256_file
from respiratory_sound.metrics import respiratory_metrics

PREFIXES = {20260729: 7, 20260730: 9, 20260731: 6}


def finite_tensors(value):
    if isinstance(value, torch.Tensor):
        assert torch.isfinite(value).all().item(), "nonfinite saved tensor"
        return 1
    if isinstance(value, dict):
        return sum(finite_tensors(x) for x in value.values())
    if isinstance(value, (tuple, list)):
        return sum(finite_tensors(x) for x in value)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("no-js", "matched-lora"), required=True)
    parser.add_argument("--seed", type=int, choices=PREFIXES, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    base = root / "artifacts/revision_2026_09_03"
    name = f"{args.family.replace('-', '_')}_seed{args.seed}"
    run = root / "runs/revision_2026_09_03" / f"revision_{name}"
    job = base / "jobs_2026_09_05" / name
    status = json.loads((job / "status.json").read_text())
    assert status["status"] == "completed_pending_audit" and status["exit_code"] == 0
    assert sha256_file(job / "training.log") == status["training_log_sha256"]
    assert sha256_file(run / "summary.json") == status["summary_sha256"]
    configuration = json.loads((run / "configuration.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    history = pd.read_csv(run / "history.csv")
    frozen = json.loads((base / "implementation_freeze.json").read_text())
    for relative, expected in frozen["files"].items():
        assert sha256_file(root / relative) == expected, relative
    assert sha256_file(base / "implementation_freeze.json") == configuration["implementation_freeze_sha256"]
    assert sha256_file(Path(configuration["analysis_plan"])) == configuration["analysis_plan_sha256"]
    config_path = root / "configs/revision/beats_ablation.yaml"
    config = yaml.safe_load(config_path.read_text())
    assert sha256_file(config_path) == configuration["config_sha256"]
    manifest_path = root / "data/manifests/cross_domain_binary.csv"
    assert sha256_file(manifest_path) == frozen["manifest_sha256"] == configuration["manifest_sha256"]
    manifest = pd.read_csv(manifest_path, dtype={"patient_id": str})
    overlap = _patient_overlap_audit(manifest)
    specification = {**config["common"], **config["families"][args.family],
                     "family": args.family, "seed": args.seed}
    specification["epochs_to_run"] = PREFIXES[args.seed] if specification["fixed_prefix"] else specification["max_epochs"]
    assert configuration["specification"] == specification
    assert configuration["batches_per_epoch"] == 940
    assert configuration["device"] == "mps"
    for record in (configuration, summary):
        assert record["family"] == args.family and record["seed"] == args.seed
        assert not record["calibration_accessed"] and not record["locked_tests_accessed"]
    assert configuration["protocol_audit"]["train_validation_patient_overlap"] == 0
    assert not configuration["protocol_audit"]["calibration_selected"]
    assert not configuration["protocol_audit"]["locked_selected"]
    assert np.isfinite(history.to_numpy(dtype=float)).all()
    assert np.array_equal(history.epoch, np.arange(1, len(history) + 1))
    assert len(history) == summary["epochs_completed"]
    assert len(history) <= specification["epochs_to_run"]
    best_epoch, best_worst, best_mean, stale = 0, -1.0, -1.0, 0
    max_lr_difference = 0.0
    for index, row in history.iterrows():
        if index < specification["warmup_epochs"]:
            factor = (index + 1) / specification["warmup_epochs"]
        else:
            progress = (index - specification["warmup_epochs"]) / (
                specification["schedule_horizon_epochs"] - specification["warmup_epochs"])
            factor = .5 * (1 + math.cos(math.pi * min(1, progress)))
        first_group = "backbone" if args.family == "no-js" else "adaptation"
        for name_col, rate in ((f"{first_group}_learning_rate", specification["backbone_or_adapter_learning_rate"]),
                               ("head_learning_rate", specification["head_learning_rate"])):
            difference = abs(row[name_col] - rate * factor)
            max_lr_difference = max(max_lr_difference, difference)
            assert difference < 1e-15
        expected_js = specification["js_max_weight"] * min(1, (index + 1) / 20)
        assert abs(row.consistency_weight - expected_js) < 1e-14
        np.testing.assert_allclose(row.train_loss, row.train_classification_loss + expected_js * row.train_consistency_loss,
                                   rtol=1e-6, atol=1e-7)
        scores = [row[f"validation_{domain}_average_score"] for domain in ("icbhi2017", "sprsound2022")]
        assert abs(min(scores) - row.validation_worst_average_score) < 1e-12
        assert abs(np.mean(scores) - row.validation_mean_average_score) < 1e-12
        worst, mean = row.validation_worst_average_score, row.validation_mean_average_score
        improved = worst > best_worst + 1e-12 or (abs(worst - best_worst) <= 1e-12 and mean > best_mean + 1e-12)
        if improved:
            best_epoch, best_worst, best_mean, stale = index + 1, worst, mean, 0
        else:
            stale += 1
        if not specification["fixed_prefix"] and stale >= specification["early_stopping_patience"]:
            assert index == len(history) - 1, "continued past frozen early stopping"
    assert best_epoch == summary["best_epoch"]
    assert abs(best_worst - summary["best_worst_domain_average_score"]) < 1e-12
    assert abs(best_mean - summary["best_mean_domain_average_score"]) < 1e-12
    assert len(history) == specification["epochs_to_run"] or (
        not specification["fixed_prefix"] and stale >= specification["early_stopping_patience"])
    assert abs(history.epoch_seconds.sum() - summary["training_seconds_total"]) < 1e-7
    assert sha256_file(run / "configuration.json") == summary["configuration_sha256"]
    assert sha256_file(run / "best.pt") == summary["best_checkpoint_sha256"]
    selected = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    assert selected["family"] == args.family and selected["seed"] == args.seed
    assert selected["specification"] == specification
    assert selected["selection"]["epoch"] == best_epoch
    assert selected["configuration_sha256"] == summary["configuration_sha256"]
    assert selected["analysis_plan_sha256"] == configuration["analysis_plan_sha256"]
    assert not selected["calibration_accessed"] and not selected["locked_tests_accessed"]
    state = selected["trainable_state_dict"]
    assert set(state) == set(configuration["trainable_parameter_names"]) == set(selected["trainable_parameter_names"])
    expected_parameters = 90313330 if args.family == "no-js" else 99842
    assert sum(t.numel() for t in state.values()) == configuration["trainable_parameters"] == expected_parameters
    selected_tensors_checked = finite_tensors(state)
    del state, selected
    recovery = torch.load(run / "resume_state.pt", map_location="cpu", weights_only=False)
    assert recovery["training_complete"] and recovery["next_epoch"] == len(history)
    assert recovery["best_epoch"] == best_epoch and recovery["stale_epochs"] == stale
    assert recovery["configuration_sha256"] == summary["configuration_sha256"]
    assert recovery["family"] == args.family and recovery["seed"] == args.seed
    assert not recovery["calibration_accessed"] and not recovery["locked_tests_accessed"]
    pd.testing.assert_frame_equal(pd.DataFrame(recovery["history"]), history, check_exact=False, rtol=1e-12, atol=1e-12)
    recovery_tensors_checked = finite_tensors(recovery)
    del recovery
    predictions_audit = {}
    for domain in ("icbhi2017", "sprsound2022"):
        path = run / f"best_validation_predictions_{domain}.csv"
        prediction = pd.read_csv(path, dtype={"patient_id": str}).sort_values("sample_id").reset_index(drop=True)
        expected = manifest.loc[(manifest.dataset == domain) & (manifest.protocol_role == "validation_select")]
        expected = expected.sort_values("sample_id").reset_index(drop=True)
        assert not prediction.sample_id.duplicated().any()
        for column in ("sample_id", "dataset", "patient_id", "protocol_role", "locked", "binary_label_id", "fine_label_name"):
            assert prediction[column].astype(str).equals(expected[column].astype(str)), (domain, column)
        np.testing.assert_allclose(prediction.event_duration_seconds, expected.event_duration_seconds, rtol=1e-12)
        y, p = prediction.target.to_numpy(dtype=int), prediction.probability_1.to_numpy(dtype=float)
        assert np.array_equal(y, expected.binary_label_id)
        assert np.isfinite(p).all() and np.all((p >= 0) & (p <= 1))
        assert np.array_equal(prediction.prediction, (p >= .5).astype(int))
        np.testing.assert_allclose(prediction.probability_0 + p, 1, atol=2e-7)
        metrics = respiratory_metrics(y, (p >= .5).astype(int), CLASS_NAMES)
        metrics["auroc"] = float(roc_auc_score(y, p))
        stored = summary["best_validation_metrics"][domain]
        for key, value in metrics.items():
            if isinstance(value, float):
                np.testing.assert_allclose(value, stored[key], atol=1e-12, rtol=1e-12)
                np.testing.assert_allclose(value, history.iloc[best_epoch-1][f"validation_{domain}_{key}"], atol=1e-12, rtol=1e-12)
            else:
                assert value == stored[key]
        predictions_audit[domain] = {"events": len(prediction), "patients": prediction.patient_id.nunique(),
                                     "metrics_recomputed": metrics, "sha256": sha256_file(path)}
    output = base / "post_training_audits" / f"{name}.json"
    output.parent.mkdir(exist_ok=True)
    hashes = {p.name: sha256_file(p) for p in run.iterdir() if p.is_file()}
    result = {"status": "post_training_audit_passed", "family": args.family, "seed": args.seed,
              "audited_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
              "epochs_completed": len(history), "selected_epoch": best_epoch, "final_stale_epochs": stale,
              "optimizer_steps": len(history) * 940,
              "selected_checkpoint_finite_tensors": selected_tensors_checked,
              "recovery_finite_tensors": recovery_tensors_checked,
              "maximum_learning_rate_difference": max_lr_difference,
              "patient_overlap_audit": overlap, "validation_predictions": predictions_audit,
              "training_seconds_total": summary["training_seconds_total"], "memory": summary["memory"],
              "calibration_accessed": False, "locked_tests_accessed": False,
              "claim_scope": "single-seed development QA only; no JS or adaptation superiority conclusion",
              "audit_script_sha256": sha256_file(Path(__file__).resolve()), "run_files_sha256": hashes}
    if output.exists():
        prior = json.loads(output.read_text())
        assert prior["run_files_sha256"] == hashes
        print(json.dumps({"status": "previous_audit_reverified", "output": str(output), "sha256": sha256_file(output)}))
        return
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    output.with_suffix(".sha256").write_text(f"{sha256_file(output)}  {output.name}\n")
    print(json.dumps({"output": str(output), "sha256": sha256_file(output),
                      "epochs_completed": len(history), "selected_epoch": best_epoch,
                      "validation_AS": {d: a["metrics_recomputed"]["average_score"] for d, a in predictions_audit.items()},
                      "training_hours": summary["training_seconds_total"] / 3600}, indent=2))


if __name__ == "__main__":
    main()
