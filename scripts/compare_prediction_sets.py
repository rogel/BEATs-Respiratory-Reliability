#!/usr/bin/env python3
"""Compare fixed seed sets using a paired patient-cluster bootstrap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from respiratory_sound.metrics import paired_patient_bootstrap_mean_seed_difference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-a", type=Path, nargs="+", required=True)
    parser.add_argument("--predictions-b", type=Path, nargs="+", required=True)
    parser.add_argument("--name-a", required=True)
    parser.add_argument("--name-b", required=True)
    parser.add_argument("--metric", default="icbhi_score")
    parser.add_argument("--iterations", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=20_260_727)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_prediction(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, dtype={"patient_id": str}).sort_values("sample_id")
    if "patient_id" not in table.columns:
        table["patient_id"] = table["sample_id"].str.split("_", n=1).str[0]
    return table


def main() -> None:
    args = parse_args()
    if len(args.predictions_a) != len(args.predictions_b):
        raise SystemExit("Both model sets must contain the same number of seeds")
    tables_a = [load_prediction(path) for path in args.predictions_a]
    tables_b = [load_prediction(path) for path in args.predictions_b]
    reference = tables_a[0]
    for table in [*tables_a[1:], *tables_b]:
        if table["sample_id"].tolist() != reference["sample_id"].tolist():
            raise SystemExit("Prediction files do not contain the same ordered samples")
        if table["target"].tolist() != reference["target"].tolist():
            raise SystemExit("Prediction files disagree on targets")
        if table["patient_id"].tolist() != reference["patient_id"].tolist():
            raise SystemExit("Prediction files disagree on patient IDs")

    difference = paired_patient_bootstrap_mean_seed_difference(
        targets=reference["target"].to_numpy(),
        predictions_a=np.stack([table["prediction"].to_numpy() for table in tables_a]),
        predictions_b=np.stack([table["prediction"].to_numpy() for table in tables_b]),
        patient_ids=reference["patient_id"].to_numpy(),
        metric=args.metric,
        iterations=args.iterations,
        seed=args.seed,
    )
    payload = {
        "model_a": args.name_a,
        "model_b": args.name_b,
        "metric": args.metric,
        "seed_count": len(tables_a),
        "difference_definition": "mean_over_fixed_seeds_of_model_a_minus_model_b",
        "paired_patient_bootstrap": difference,
    }
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
