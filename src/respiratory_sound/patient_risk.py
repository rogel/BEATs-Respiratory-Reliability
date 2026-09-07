"""Patient-balanced risk summaries for respiratory-sound predictions."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from respiratory_sound.calibration import clip_probabilities


def patient_risk_table(
    frame: pd.DataFrame,
    probability_column: str = "probability_1",
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Summarize class-balanced NLL and fixed-threshold errors per patient."""
    required = {"patient_id", "target", probability_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"frame is missing columns: {sorted(missing)}")
    working = frame[["patient_id", "target", probability_column]].copy()
    working["patient_id"] = working["patient_id"].astype(str)
    targets = working["target"].to_numpy(dtype=np.int64)
    if not np.isin(targets, (0, 1)).all():
        raise ValueError("target must be binary")
    probabilities = clip_probabilities(
        working[probability_column].to_numpy(dtype=np.float64)
    )
    working["event_nll"] = -(
        targets * np.log(probabilities)
        + (1 - targets) * np.log1p(-probabilities)
    )
    working["error"] = (probabilities >= threshold).astype(np.int64) != targets
    class_cells = (
        working.groupby(["patient_id", "target"], sort=True, as_index=False)
        .agg(class_mean_nll=("event_nll", "mean"))
    )
    class_balanced = (
        class_cells.groupby("patient_id", sort=True, as_index=False)
        .agg(
            patient_risk=("class_mean_nll", "mean"),
            classes_present=("target", "nunique"),
        )
    )
    event_counts = (
        working.groupby("patient_id", sort=True, as_index=False)
        .agg(
            event_count=("target", "size"),
            error_count=("error", "sum"),
        )
    )
    return class_balanced.merge(
        event_counts,
        on="patient_id",
        how="inner",
        validate="one_to_one",
    )


def top_risk_patients(
    table: pd.DataFrame,
    fraction: float = 0.25,
) -> tuple[str, ...]:
    """Return the deterministic highest-risk patient tail."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    required = {"patient_id", "patient_risk"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"table is missing columns: {sorted(missing)}")
    count = max(1, math.ceil(len(table) * fraction))
    ordered = table.sort_values(
        ["patient_risk", "patient_id"],
        ascending=[False, True],
        kind="stable",
    )
    return tuple(ordered.head(count)["patient_id"].astype(str))


def jaccard_index(first: set[str], second: set[str]) -> float:
    """Compute set Jaccard overlap, including the empty-set identity."""
    union = first | second
    return float(len(first & second) / len(union)) if union else 1.0
