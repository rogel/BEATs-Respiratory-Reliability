#!/usr/bin/env python3
"""Seal post-Gate-11A calibration/selective rules before calibration inference."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from respiratory_sound.gate11a import sha256_file
from respiratory_sound.post_gate11a import FULLFT_RUNS, LORA_RUNS, SEEDS

CODE_FILES = (
    "src/respiratory_sound/calibration.py",
    "src/respiratory_sound/selective_prediction.py",
    "src/respiratory_sound/post_gate11a.py",
    "src/respiratory_sound/models/pretrained_audio.py",
    "src/respiratory_sound/models/beats_adaptation.py",
    "src/respiratory_sound/data/audio.py",
    "scripts/predict_post_gate11a.py",
    "scripts/analyze_post_gate11a_calibration.py",
)


def main() -> None:
    root = Path.cwd().resolve()
    output = root / "artifacts/post_gate11a_development_freeze.json"
    calibration_output = root / "artifacts/post_gate11a/calibration_predictions"
    locked_access = root / "artifacts/post_gate11a_locked_access.json"
    if output.exists():
        raise FileExistsError(f"Development freeze already exists: {output}")
    if calibration_output.exists() or locked_access.exists():
        raise RuntimeError("Calibration or locked inference was already materialized")
    gate_final_path = root / "artifacts/gate11a_exactmask_fullft_final.json"
    gate_final = json.loads(gate_final_path.read_text(encoding="utf-8"))
    if gate_final.get("status") != "gate11a_passed":
        raise RuntimeError("Gate 11A did not pass")

    model_files = {
        f"fullft_seed{seed}": root / FULLFT_RUNS[seed] / "best.pt"
        for seed in SEEDS
    }
    model_files.update(
        {
            f"lora_seed{seed}": root / LORA_RUNS[seed] / "best.pt"
            for seed in SEEDS
        }
    )
    hash_files = {
        "manifest": root / "data/manifests/cross_domain_binary.csv",
        "data_config": root / "configs/data/gate9a_beats.yaml",
        "upstream_checkpoint": root
        / "checkpoints/pretrained/beats_as2m_cpt2/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt",
        "gate11a_final": gate_final_path,
        **{f"code:{name}": root / name for name in CODE_FILES},
        **{f"checkpoint:{name}": path for name, path in model_files.items()},
    }
    missing = [str(path) for path in hash_files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing development-freeze files: {missing}")

    payload = {
        "schema_version": 1,
        "stage": "post_gate11a_development",
        "status": "sealed_before_calibration",
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "gate11a_status": "passed",
        "classifier": {
            "primary": "fullft_equal_probability_ensemble",
            "seeds": list(SEEDS),
            "weights": [1 / 3, 1 / 3, 1 / 3],
            "classification_threshold": 0.5,
            "seed_selection_forbidden": True,
            "fullft_runs": FULLFT_RUNS,
            "matched_efficiency_reference": "lora_equal_probability_ensemble",
            "lora_runs": LORA_RUNS,
            "fullft_superiority_claim_allowed": False,
            "reason": "paired validation bootstrap 95% intervals cross zero",
        },
        "calibration_protocol": {
            "method": "per-domain scalar temperature scaling of ensemble logits",
            "objective": "binary negative log likelihood",
            "classification_threshold_remains": 0.5,
            "ece": "15-bin equal-frequency ECE",
            "also_report": ["NLL", "Brier", "reliability_table"],
            "fit_roles": {
                "icbhi2017": "calibration",
                "sprsound2022": "calibration",
            },
            "sanity_role": "validation_select using already frozen predictions",
            "enable_rule": {
                "mean_validation_nll_not_worse": True,
                "maximum_allowed_per_domain_nll_worsening": 0.02,
                "fixed_0_5_predictions_must_be_identical": True,
            },
            "failure_action": "temperature=1; no alternative calibrator search",
        },
        "selective_protocol": {
            "uncertainty": "normalized binary predictive entropy after temperature scaling",
            "target_coverages": [0.8, 0.9],
            "cutoff_fit": "calibration uncertainty order statistic retaining ceil(target*n)",
            "classification_threshold": 0.5,
            "metrics": [
                "risk-coverage curve",
                "AURC",
                "error risk at fixed cutoffs",
                "balanced error risk at fixed cutoffs",
                "normal/adventitious coverage",
            ],
            "safety": {
                "minimum_each_class_coverage": "target minus 0.15",
                "maximum_class_coverage_gap": 0.20,
                "maximum_error_risk_worsening_vs_full_coverage": 0.02,
                "must_pass_on": ["calibration", "validation_select"],
            },
            "failure_action": (
                "disable selective locked claim; do not search another score or cutoff"
            ),
        },
        "efficiency_protocol": {
            "families": ["fullft", "lora"],
            "objects": ["single_seed20260729", "three_seed_ensemble"],
            "input": "8.0-second 16-kHz mono waveform with exact all-valid mask",
            "batch_sizes": [1, 8],
            "warmup_iterations": 5,
            "timed_iterations": 20,
            "report": [
                "total/trainable parameters",
                "checkpoint bytes",
                "median and p95 latency",
                "events per second",
                "MPS current and driver allocated memory",
                "training wall time and peak memory from run summaries",
            ],
        },
        "locked_protocol": {
            "requires_second_freeze": True,
            "one_access_episode": True,
            "primary_independent_roles": {
                "icbhi2017": "locked_test",
                "sprsound2022": "locked_inter_test",
            },
            "descriptive_only": {"sprsound2022": "locked_intra_test"},
            "models_in_same_prediction_run": [
                "three fullft seeds and equal ensemble",
                "three matched LoRA seeds and equal ensemble",
            ],
            "bootstrap": {
                "patient_cluster_iterations": 4000,
                "seed": 20260729,
                "confidence": 0.95,
            },
            "locked_result_may_not_change": [
                "model",
                "threshold",
                "temperature",
                "rejection cutoffs",
                "method",
            ],
        },
        "data_history_disclosure": (
            "Calibration patients were used by earlier abandoned routes and are tuning data, "
            "not independent evidence. Locked labels/metadata counts were audited, while locked "
            "model predictions and performance remain unevaluated."
        ),
        "smoke_audit": {
            "role": "validation_select",
            "samples": 554,
            "sample_ids_targets_and_predictions_identical": True,
            "maximum_probability_absolute_difference": 0.002014700000000036,
            "calibration_accessed": False,
            "locked_tests_accessed": False,
        },
        "hashes": {name: sha256_file(path) for name, path in hash_files.items()},
        "calibration_accessed": False,
        "locked_tests_accessed": False,
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
