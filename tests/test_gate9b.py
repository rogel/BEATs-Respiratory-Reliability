from __future__ import annotations

from typing import Any

import pytest

from respiratory_sound.gate9b import (
    noninferiority_decision,
    proposed_screen_decision,
)


def _summary(
    icbhi: float,
    sprsound: float,
    *,
    drift: float,
    trainable_fraction: float = 0.001,
) -> dict[str, Any]:
    return {
        "best_mean_domain_average_score": (icbhi + sprsound) / 2.0,
        "best_train_projection_drift": drift,
        "trainable_fraction": trainable_fraction,
        "best_validation_metrics": {
            "icbhi2017": {
                "average_score": icbhi,
                "sensitivity": 0.7,
                "specificity": 0.7,
            },
            "sprsound2022": {
                "average_score": sprsound,
                "sensitivity": 0.8,
                "specificity": 0.9,
            },
        },
    }


def test_noninferiority_requires_all_bounds() -> None:
    assert noninferiority_decision(_summary(0.70, 0.89, drift=0.4))["passed"]
    assert not noninferiority_decision(_summary(0.70, 0.87, drift=0.4))["passed"]


def test_proposed_screen_accepts_small_worst_domain_gain() -> None:
    plain = _summary(0.700, 0.890, drift=0.4)
    anchored = _summary(0.703, 0.887, drift=0.1)
    decision = proposed_screen_decision(anchored, plain)
    assert decision["directional_gain_branches"]["worst_domain_branch"]
    assert decision["passed"]


def test_proposed_screen_rejects_mechanism_without_task_gain() -> None:
    plain = _summary(0.7102010988, 0.8918322853, drift=0.428)
    anchored = _summary(0.6997938117, 0.8712294448, drift=0.097)
    decision = proposed_screen_decision(anchored, plain)
    assert not decision["checks"]["anchored_noninferiority"]
    assert not decision["checks"]["directional_gain"]
    assert not decision["checks"]["maximum_domain_drop"]
    assert not decision["passed"]


def test_selected_scores_rejects_inconsistent_recorded_mean() -> None:
    summary = _summary(0.70, 0.89, drift=0.4)
    summary["best_mean_domain_average_score"] = 0.1
    with pytest.raises(ValueError, match="selected mean"):
        noninferiority_decision(summary)
