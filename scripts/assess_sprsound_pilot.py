#!/usr/bin/env python3
"""Assess the SPRSound one-seed pilot without touching official test splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from respiratory_sound.metrics import paired_patient_bootstrap_difference

CLASS_NAMES = (
    "normal",
    "rhonchi",
    "wheeze",
    "stridor",
    "coarse_crackle",
    "fine_crackle",
    "wheeze_crackle",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--iterations", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=20_260_727)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/tables/sprsound_pilot_gate.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    run_names = {
        "plain": "sprsound_pilot_plain_7class_seed20260727",
        "repconv": "sprsound_pilot_repconv_7class_seed20260727",
        "tfdcr": "sprsound_pilot_tfdcr_7class_seed20260727",
    }
    summaries = {
        name: json.loads((root / "runs" / run / "summary.json").read_text(encoding="utf-8"))
        for name, run in run_names.items()
    }
    predictions = {
        name: pd.read_csv(
            root / "runs" / run / "best_validation_predictions.csv",
            dtype={"patient_id": str},
        ).sort_values("sample_id")
        for name, run in run_names.items()
    }
    reference = predictions["plain"]
    for name, table in predictions.items():
        if table["sample_id"].tolist() != reference["sample_id"].tolist():
            raise ValueError(f"Prediction sample order differs for {name}")
        if table["target"].tolist() != reference["target"].tolist():
            raise ValueError(f"Prediction targets differ for {name}")

    targets = reference["target"].to_numpy()
    patient_ids = reference["patient_id"].to_numpy()
    paired: dict[str, object] = {}
    for baseline in ("plain", "repconv"):
        for metric in ("score", "macro_f1"):
            paired[f"tfdcr_vs_{baseline}_{metric}"] = paired_patient_bootstrap_difference(
                targets,
                predictions["tfdcr"]["prediction"].to_numpy(),
                predictions[baseline]["prediction"].to_numpy(),
                patient_ids,
                metric=metric,
                iterations=args.iterations,
                seed=args.seed,
                class_names=CLASS_NAMES,
            )

    metrics = {
        name: summary["best_validation_metrics"] for name, summary in summaries.items()
    }
    checks = {
        "tfdcr_score_above_plain": metrics["tfdcr"]["score"] > metrics["plain"]["score"],
        "tfdcr_score_above_repconv": (
            metrics["tfdcr"]["score"] > metrics["repconv"]["score"]
        ),
        "tfdcr_macro_f1_not_below_plain": (
            metrics["tfdcr"]["macro_f1"] >= metrics["plain"]["macro_f1"]
        ),
    }
    payload = {
        "split": "development_split=validation",
        "official_test_accessed": False,
        "sample_count": len(reference),
        "patient_count": int(len(np.unique(patient_ids))),
        "metrics": metrics,
        "differences": {
            "tfdcr_minus_plain_score": (
                metrics["tfdcr"]["score"] - metrics["plain"]["score"]
            ),
            "tfdcr_minus_repconv_score": (
                metrics["tfdcr"]["score"] - metrics["repconv"]["score"]
            ),
            "tfdcr_minus_plain_macro_f1": (
                metrics["tfdcr"]["macro_f1"] - metrics["plain"]["macro_f1"]
            ),
            "tfdcr_minus_repconv_macro_f1": (
                metrics["tfdcr"]["macro_f1"] - metrics["repconv"]["macro_f1"]
            ),
        },
        "paired_patient_bootstrap": paired,
        "checks": checks,
        "pilot_gate_passed": all(checks.values()),
        "decision": (
            "proceed_to_three_seeds"
            if all(checks.values())
            else "stop_before_three_seeds_and_official_testing"
        ),
    }
    output = (root / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
