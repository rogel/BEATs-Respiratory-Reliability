#!/usr/bin/env python3
"""Compute patient-cluster bootstrap intervals for one prediction table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from respiratory_sound.metrics import patient_bootstrap_intervals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--iterations", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=20_260_727)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table = pd.read_csv(args.predictions, dtype={"patient_id": str})
    if "patient_id" not in table.columns:
        table["patient_id"] = table["sample_id"].str.split("_", n=1).str[0]
    intervals = patient_bootstrap_intervals(
        targets=table["target"].to_numpy(),
        predictions=table["prediction"].to_numpy(),
        patient_ids=table["patient_id"].to_numpy(),
        iterations=args.iterations,
        seed=args.seed,
    )
    payload = {
        "model_name": args.model_name,
        "predictions": str(args.predictions),
        "cluster_unit": "patient",
        "patient_count": int(table["patient_id"].nunique()),
        "iterations": args.iterations,
        "intervals": intervals,
    }
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
