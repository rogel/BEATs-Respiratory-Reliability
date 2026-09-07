#!/usr/bin/env python3
"""Combine per-domain feature moments with equal database contribution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payloads = [
        json.loads(path.resolve().read_text(encoding="utf-8"))
        for path in (args.first, args.second)
    ]
    means = np.stack(
        [np.asarray(payload["mean"], dtype=np.float64) for payload in payloads]
    )
    variances = np.stack(
        [np.asarray(payload["std"], dtype=np.float64) ** 2 for payload in payloads]
    )
    if means.shape[0] != 2 or means.shape != variances.shape:
        raise ValueError("Expected two compatible feature-statistic files")
    combined_mean = means.mean(axis=0)
    combined_variance = (variances + (means - combined_mean) ** 2).mean(axis=0)
    output = {
        "method": "equal_domain_moment_mixture",
        "domain_weights": [0.5, 0.5],
        "source_files": [str(args.first), str(args.second)],
        "split_column": "protocol_role",
        "split_value": "train_fit",
        "label_column": "binary_label_id",
        "sample_count": int(sum(payload["sample_count"] for payload in payloads)),
        "mean": combined_mean.tolist(),
        "std": np.sqrt(combined_variance).tolist(),
    }
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
