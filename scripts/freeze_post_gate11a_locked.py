#!/usr/bin/env python3
"""Seal fitted postprocessing and all final locked-evaluation rules."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from respiratory_sound.gate11a import sha256_file


def main() -> None:
    root = Path.cwd().resolve()
    output = root / "artifacts/post_gate11a_locked_freeze.json"
    locked_predictions = root / "artifacts/post_gate11a/locked_predictions"
    locked_access = root / "artifacts/post_gate11a_locked_access.json"
    if output.exists() or locked_predictions.exists() or locked_access.exists():
        raise RuntimeError("Final locked freeze/access already exists")

    development_path = root / "artifacts/post_gate11a_development_freeze.json"
    calibration_path = (
        root
        / "artifacts/post_gate11a/calibration_analysis/post_gate11a_calibration_final.json"
    )
    efficiency_path = root / "artifacts/post_gate11a/efficiency_final.json"
    calibration_predictions = (
        root / "artifacts/post_gate11a/calibration_predictions/prediction_manifest.json"
    )
    development = json.loads(development_path.read_text(encoding="utf-8"))
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    efficiency = json.loads(efficiency_path.read_text(encoding="utf-8"))
    if development.get("status") != "sealed_before_calibration":
        raise ValueError("Invalid development freeze")
    if not calibration["decision"]["temperature_enabled_for_locked"]:
        raise ValueError("Frozen temperature rule did not pass")
    if calibration["decision"]["selective_enabled_for_locked"]:
        raise ValueError("Selective prediction unexpectedly passed")
    if efficiency["locked_tests_accessed"]:
        raise ValueError("Efficiency artifact crossed locked boundary")

    for name, frozen_hash in development["hashes"].items():
        if name.startswith("code:"):
            path = root / name.removeprefix("code:")
        elif name.startswith("checkpoint:"):
            continue
        elif name == "manifest":
            path = root / "data/manifests/cross_domain_binary.csv"
        elif name == "data_config":
            path = root / "configs/data/gate9a_beats.yaml"
        elif name == "upstream_checkpoint":
            path = (
                root
                / "checkpoints/pretrained/beats_as2m_cpt2/"
                "BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"
            )
        elif name == "gate11a_final":
            path = root / "artifacts/gate11a_exactmask_fullft_final.json"
        else:
            continue
        if sha256_file(path) != frozen_hash:
            raise ValueError(f"Development-frozen file changed: {name}")

    final_code = {
        "predict_locked": root / "scripts/predict_post_gate11a.py",
        "analyze_locked": root / "scripts/analyze_post_gate11a_locked.py",
        "post_gate_module": root / "src/respiratory_sound/post_gate11a.py",
        "calibration_module": root / "src/respiratory_sound/calibration.py",
        "selective_module": root / "src/respiratory_sound/selective_prediction.py",
    }
    evidence = {
        "development_freeze": development_path,
        "calibration_prediction_manifest": calibration_predictions,
        "calibration_final": calibration_path,
        "efficiency_final": efficiency_path,
        "gate11a_final": root / "artifacts/gate11a_exactmask_fullft_final.json",
        "gate9d_lora_reference": root / "artifacts/gate9d_multiseed_final.json",
    }
    payload = {
        "schema_version": 1,
        "stage": "post_gate11a_locked",
        "status": "sealed_before_locked_access",
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "primary_model": {
            "object": "equal-weight probability ensemble of all three full-FT seeds",
            "weights": [1 / 3, 1 / 3, 1 / 3],
            "classification_threshold": 0.5,
        },
        "matched_reference": {
            "object": "equal-weight probability ensemble of all three matched LoRA seeds",
            "purpose": "performance-efficiency trade-off; no preauthorized superiority claim",
        },
        "fitted_postprocessing": calibration["locked_parameters"],
        "calibration_claim_rule_on_locked": {
            "classification_predictions_must_remain_identical": True,
            "mean_nll_must_not_worsen": True,
            "maximum_per_domain_nll_worsening": 0.02,
            "failed_rule_action": "report calibration as unsupported; no refit or rerun",
        },
        "selective_prediction": {
            "enabled": False,
            "reason": (
                "SPRSound validation Adventitious coverage at the calibration-fitted "
                "80% target cutoff was 0.6035, below the frozen 0.65 floor"
            ),
            "allowed_locked_output": "descriptive risk-coverage curve and AURC only",
            "alternative_search_forbidden": True,
        },
        "locked_roles": {
            "primary_independent": {
                "icbhi2017": "locked_test",
                "sprsound2022": "locked_inter_test",
            },
            "descriptive_same_patient_only": {
                "sprsound2022": "locked_intra_test"
            },
        },
        "locked_robustness_thresholds": {
            "icbhi_average_score": 0.64,
            "sprsound_inter_average_score": 0.82,
            "two_domain_equal_mean_average_score": 0.75,
            "sensitivity_floor_each_primary_domain": 0.50,
            "specificity_floor_each_primary_domain": 0.50,
            "basis": "no more than about 0.08 absolute AS degradation from development ensemble",
        },
        "statistics": {
            "patient_cluster_bootstrap_iterations": 4000,
            "seed": 20260729,
            "confidence": 0.95,
            "report": [
                "three seed metrics and mean plus sample SD",
                "full-FT ensemble absolute intervals",
                "paired full-FT minus LoRA intervals",
                "per-domain and equal-domain-mean results",
            ],
        },
        "one_time_execution": {
            "prediction_command": (
                ".venv/bin/python scripts/predict_post_gate11a.py --project-root . "
                "--stage locked --device mps --batch-size 8 --output-root "
                "artifacts/post_gate11a/locked_predictions"
            ),
            "analysis_command": ".venv/bin/python scripts/analyze_post_gate11a_locked.py",
            "same_run_models": [
                "full-FT seeds 20260729/20260730/20260731 and ensemble",
                "LoRA seeds 20260729/20260730/20260731 and ensemble",
            ],
            "resume_rule": (
                "only the same freeze-bound access episode may reuse hash-verified completed "
                "prediction files after interruption"
            ),
            "after_result": (
                "never alter or rerun model, threshold, temperature, rejection rule, or method"
            ),
        },
        "hashes": {
            **{f"code:{name}": sha256_file(path) for name, path in final_code.items()},
            **{f"evidence:{name}": sha256_file(path) for name, path in evidence.items()},
        },
        "calibration_accessed": True,
        "locked_tests_accessed": False,
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
