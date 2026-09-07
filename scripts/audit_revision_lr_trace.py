#!/usr/bin/env python3
"""Verify revision LR traces against the completed Gate 11A histories."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
import yaml

from respiratory_sound.gate11a import sha256_file
from respiratory_sound.training import warmup_cosine_scheduler

PLAN_SHA256 = "18b6e39bba441e5be1ec870591b9dbf2846b7e6ab6293be9d211aab12809be39"
PREFIXES = {20_260_729: 7, 20_260_730: 9, 20_260_731: 6}


def main() -> None:
    root = Path.cwd().resolve()
    config_path = root / "configs/revision/beats_ablation.yaml"
    plan_path = (
        root
        / "../experiment_plans/2026-09-03/"
        "01_REVISION_ANALYSIS_PLAN_FROZEN_2026-09-03.md"
    ).resolve()
    if sha256_file(plan_path) != PLAN_SHA256:
        raise ValueError("Frozen analysis plan changed")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    common = config["common"]
    traces: dict[str, object] = {}
    for seed, epochs in PREFIXES.items():
        adaptation = torch.nn.Parameter(torch.zeros(()))
        head = torch.nn.Parameter(torch.zeros(()))
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": [adaptation],
                    "lr": float(common["backbone_or_adapter_learning_rate"]),
                },
                {"params": [head], "lr": float(common["head_learning_rate"])},
            ]
        )
        scheduler = warmup_cosine_scheduler(
            optimizer,
            warmup_epochs=int(common["warmup_epochs"]),
            max_epochs=int(common["schedule_horizon_epochs"]),
        )
        expected = []
        for epoch in range(epochs):
            expected.append(
                {
                    "epoch": epoch + 1,
                    "backbone_or_adapter_learning_rate": float(
                        optimizer.param_groups[0]["lr"]
                    ),
                    "head_learning_rate": float(optimizer.param_groups[1]["lr"]),
                }
            )
            optimizer.step()
            scheduler.step()
        history = pd.read_csv(
            root / f"runs/gate11a_exactmask_fullft_seed{seed}/history.csv"
        ).iloc[:epochs]
        differences = []
        for expected_row, observed in zip(expected, history.itertuples(), strict=True):
            differences.extend(
                (
                    abs(
                        expected_row["backbone_or_adapter_learning_rate"]
                        - float(observed.backbone_learning_rate)
                    ),
                    abs(
                        expected_row["head_learning_rate"]
                        - float(observed.head_learning_rate)
                    ),
                )
            )
        maximum_difference = max(differences)
        traces[str(seed)] = {
            "epochs": epochs,
            "trace": expected,
            "maximum_absolute_difference_from_gate11a_history": maximum_difference,
            "passed_tolerance_1e-15": maximum_difference <= 1.0e-15,
        }
    result = {
        "schema_version": 1,
        "stage": "BEATS_revision",
        "analysis": "matched_budget_learning_rate_trace",
        "analysis_plan_sha256": PLAN_SHA256,
        "config_sha256": sha256_file(config_path),
        "traces": traces,
        "all_passed": all(
            bool(value["passed_tolerance_1e-15"])
            for value in traces.values()
            if isinstance(value, dict)
        ),
    }
    output = root / "artifacts/revision_2026_09_03/lr_trace_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
