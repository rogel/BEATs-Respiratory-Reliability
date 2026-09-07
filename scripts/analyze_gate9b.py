#!/usr/bin/env python3
"""Apply the frozen Gate 9B rules to the four corrected single-seed runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from respiratory_sound.gate9b import (
    epoch_noninferiority,
    noninferiority_decision,
    proposed_screen_decision,
)

EXPECTED_CANDIDATES = {
    "head": "beats_head_only",
    "last_block": "beats_last_block",
    "plain_lora": "beats_lora_qv_r8",
    "anchored_lora": "beats_drift_anchored_lora_qv_r8",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--head-run", type=Path, required=True)
    parser.add_argument("--last-block-run", type=Path, required=True)
    parser.add_argument("--plain-lora-run", type=Path, required=True)
    parser.add_argument("--anchored-lora-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_run(root: Path, run_dir: Path, expected: str) -> dict[str, Any]:
    resolved = root / run_dir
    summary = json.loads(
        (resolved / "summary.json").read_text(encoding="utf-8")
    )
    if summary["candidate"] != expected:
        raise ValueError(
            f"Expected {expected}, received {summary['candidate']} from {run_dir}"
        )
    if summary.get("locked_test_accessed") is not False:
        raise ValueError("Gate 9B summary does not certify locked-test isolation")
    history = pd.read_csv(resolved / "history.csv").to_dict(orient="records")
    if len(history) != int(summary["epochs_completed"]):
        raise ValueError(f"History length mismatch in {run_dir}")
    return {
        "run_dir": str(run_dir),
        "summary": summary,
        "history": history,
    }


def _candidate_payload(run: dict[str, Any]) -> dict[str, Any]:
    history = run["history"]
    summary = run["summary"]
    return {
        "run_dir": run["run_dir"],
        "best_epoch": int(summary["best_epoch"]),
        "epochs_completed": int(summary["epochs_completed"]),
        "trainable_parameters": int(summary["training_parameters"]),
        "trainable_fraction": float(summary["trainable_fraction"]),
        "noninferiority": noninferiority_decision(summary),
        "historical_noninferior_epochs": [
            int(row["epoch"])
            for row in history
            if epoch_noninferiority(row)
        ],
        "maximum_epoch_mean_projection_drift": max(
            float(row["train_projection_drift"]) for row in history
        ),
        "last_epoch_mean_projection_drift": float(
            history[-1]["train_projection_drift"]
        ),
    }


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    run_paths = {
        "head": args.head_run,
        "last_block": args.last_block_run,
        "plain_lora": args.plain_lora_run,
        "anchored_lora": args.anchored_lora_run,
    }
    runs = {
        name: _load_run(root, run_paths[name], EXPECTED_CANDIDATES[name])
        for name in run_paths
    }
    candidate_results = {
        name: _candidate_payload(run)
        for name, run in runs.items()
    }
    screen = proposed_screen_decision(
        runs["anchored_lora"]["summary"],
        runs["plain_lora"]["summary"],
    )
    plain_selected_drift = float(
        runs["plain_lora"]["summary"]["best_train_projection_drift"]
    )
    anchored_selected_drift = float(
        runs["anchored_lora"]["summary"]["best_train_projection_drift"]
    )
    payload = {
        "gate": "9B",
        "stage": "corrected_single_seed_four_candidate_screen",
        "exact_token_mask_amendment": (
            "artifacts/gate9b_exact_token_mask_amendment.json"
        ),
        "development_only": True,
        "calibration_role_accessed": False,
        "locked_tests_accessed": False,
        "candidates": candidate_results,
        "proposed_candidate_screen": screen,
        "mechanism": {
            "plain_lora_selected_epoch_mean_projection_drift": (
                plain_selected_drift
            ),
            "anchored_lora_selected_epoch_mean_projection_drift": (
                anchored_selected_drift
            ),
            "selected_epoch_drift_reduction_fraction": (
                1.0 - anchored_selected_drift / plain_selected_drift
            ),
            "drift_reduced": anchored_selected_drift < plain_selected_drift,
            "task_gain_supported": screen["checks"]["directional_gain"],
        },
        "decision": {
            "plain_lora_parameter_efficient_route_feasible": (
                candidate_results["plain_lora"]["noninferiority"]["passed"]
            ),
            "drift_anchor_single_seed_supported": screen["passed"],
            "extend_plain_and_anchored_lora_to_three_seeds": screen["passed"],
            "gate9b_innovation_supported": screen["passed"],
            "claim_boundary": (
                "Ordinary LoRA is a viable engineering result, but the fixed "
                "drift anchor is not supported and must not be presented as a "
                "successful Q2-level contribution."
                if not screen["passed"]
                else (
                    "Proceed to the frozen three-seed paired confirmation "
                    "before making a Q2-level contribution claim."
                )
            ),
        },
    }
    output = (root / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
