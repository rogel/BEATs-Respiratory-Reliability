#!/usr/bin/env python3
"""Aggregate repeated-seed validation summaries without selecting a favorable seed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _metric_value(metrics: dict[str, object], metric: str) -> float:
    if metric == "icbhi_score":
        if "icbhi_score" in metrics:
            return float(metrics["icbhi_score"])
        return float(metrics["average_score"])
    return float(metrics[metric])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--summaries", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in args.summaries]
    metric_names = ("sensitivity", "specificity", "icbhi_score", "macro_f1", "uar")
    aggregated = {}
    for metric in metric_names:
        values = np.asarray(
            [
                _metric_value(summary["best_validation_metrics"], metric)
                for summary in summaries
            ]
        )
        aggregated[metric] = {
            "values": values.tolist(),
            "mean": float(values.mean()),
            "sample_standard_deviation": (float(values.std(ddof=1)) if len(values) > 1 else 0.0),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
        }
    payload = {
        "model_name": args.model_name,
        "run_count": len(summaries),
        "summaries": [str(path) for path in args.summaries],
        "metrics": aggregated,
    }
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
