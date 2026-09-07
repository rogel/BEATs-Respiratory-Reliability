#!/usr/bin/env python3
"""Compare two frozen models with a paired patient-cluster bootstrap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from respiratory_sound.metrics import paired_patient_bootstrap_difference


def ensure_patient_ids(table: pd.DataFrame) -> pd.DataFrame:
    """Support older prediction exports by deriving the audited ICBHI patient ID."""
    if "patient_id" not in table.columns:
        table = table.copy()
        table["patient_id"] = table["sample_id"].str.split("_", n=1).str[0]
    return table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-a", type=Path, required=True)
    parser.add_argument("--predictions-b", type=Path, required=True)
    parser.add_argument("--name-a", required=True)
    parser.add_argument("--name-b", required=True)
    parser.add_argument("--metric", default="icbhi_score")
    parser.add_argument(
        "--class-names",
        nargs="+",
        help=(
            "Optional ordered class names. Required when comparing respiratory_metrics "
            "fields such as the composite 'score' rather than legacy ICBHI metrics."
        ),
    )
    parser.add_argument("--iterations", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=20_260_727)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table_a = ensure_patient_ids(
        pd.read_csv(args.predictions_a, dtype={"patient_id": str})
    ).sort_values("sample_id")
    table_b = ensure_patient_ids(
        pd.read_csv(args.predictions_b, dtype={"patient_id": str})
    ).sort_values("sample_id")
    if table_a["sample_id"].tolist() != table_b["sample_id"].tolist():
        raise SystemExit("Prediction files do not contain the same ordered samples")
    if table_a["target"].tolist() != table_b["target"].tolist():
        raise SystemExit("Prediction files disagree on targets")
    if table_a["patient_id"].tolist() != table_b["patient_id"].tolist():
        raise SystemExit("Prediction files disagree on patient IDs")

    difference = paired_patient_bootstrap_difference(
        targets=table_a["target"].to_numpy(),
        predictions_a=table_a["prediction"].to_numpy(),
        predictions_b=table_b["prediction"].to_numpy(),
        patient_ids=table_a["patient_id"].to_numpy(),
        metric=args.metric,
        iterations=args.iterations,
        seed=args.seed,
        class_names=(tuple(args.class_names) if args.class_names is not None else None),
    )
    payload = {
        "model_a": args.name_a,
        "model_b": args.name_b,
        "metric": args.metric,
        "class_names": args.class_names,
        "difference_definition": "model_a_minus_model_b",
        "paired_patient_bootstrap": difference,
    }
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
