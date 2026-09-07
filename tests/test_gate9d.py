from __future__ import annotations

from typing import Any

import pytest

from respiratory_sound.gate9d import (
    multiseed_sampling_decision,
    sampling_ablation_decision,
)


def _summary(icbhi: float, sprsound: float) -> dict[str, Any]:
    return {
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
        }
    }


def _bootstrap(icbhi_probability: float, mean_probability: float) -> dict[str, Any]:
    return {
        "per_domain": {
            "icbhi2017": {
                "probability_gain_above_zero": icbhi_probability,
            },
            "sprsound2022": {
                "probability_gain_above_zero": 0.9,
            },
        },
        "mean_domain": {
            "probability_gain_above_zero": mean_probability,
        },
    }


def test_sampling_ablation_accepts_small_worst_domain_gain() -> None:
    decision = sampling_ablation_decision(
        _summary(0.704, 0.89),
        _summary(0.700, 0.89),
        _bootstrap(0.7, 0.7),
    )
    assert decision["branches"]["worst_database"]["passed"]
    assert decision["passed"]


def test_sampling_ablation_rejects_unreliable_gain() -> None:
    decision = sampling_ablation_decision(
        _summary(0.704, 0.90),
        _summary(0.700, 0.895),
        _bootstrap(0.55, 0.55),
    )
    assert not decision["branches"]["worst_database"]["bootstrap_passed"]
    assert not decision["passed"]


def _multiseed_decision(
    *,
    worst_gain: float,
    mean_gain: float,
    worst_wins: int,
    mean_wins: int,
    weakest_probability: float,
    mean_probability: float,
    icbhi_gain: float = 0.0,
    sprsound_gain: float = 0.0,
    safety: bool = True,
) -> dict[str, Any]:
    return multiseed_sampling_decision(
        paired_seed_results=[{"seed": seed} for seed in (1, 2, 3)],
        mean_paired_gains={
            "icbhi2017": icbhi_gain,
            "sprsound2022": sprsound_gain,
            "worst_database": worst_gain,
            "mean_database": mean_gain,
        },
        seed_wins={
            "worst_database": worst_wins,
            "mean_database": mean_wins,
        },
        balanced_safety_every_seed=safety,
        weakest_database="icbhi2017",
        bootstrap={
            "per_domain": {
                "icbhi2017": {
                    "probability_gain_above_zero": weakest_probability
                },
                "sprsound2022": {"probability_gain_above_zero": 0.9},
            },
            "mean_domain": {
                "probability_gain_above_zero": mean_probability
            },
        },
    )


def test_multiseed_gate_accepts_frozen_mean_database_branch() -> None:
    decision = _multiseed_decision(
        worst_gain=0.001,
        mean_gain=0.004,
        worst_wins=1,
        mean_wins=2,
        weakest_probability=0.5,
        mean_probability=0.81,
        icbhi_gain=0.001,
        sprsound_gain=0.007,
    )
    assert decision["branches"]["mean_database"]["passed"]
    assert decision["passed"]


def test_multiseed_gate_rejects_database_drop_despite_direction() -> None:
    decision = _multiseed_decision(
        worst_gain=0.004,
        mean_gain=0.004,
        worst_wins=2,
        mean_wins=2,
        weakest_probability=0.85,
        mean_probability=0.85,
        icbhi_gain=-0.006,
        sprsound_gain=0.014,
    )
    assert not decision["checks"][
        "no_database_cross_seed_mean_drop_greater_than_0_005"
    ]
    assert not decision["passed"]


def test_multiseed_gate_requires_exactly_three_seed_pairs() -> None:
    with pytest.raises(ValueError, match="three seed pairs"):
        multiseed_sampling_decision(
            paired_seed_results=[{"seed": 1}],
            mean_paired_gains={
                "icbhi2017": 0.01,
                "sprsound2022": 0.01,
                "worst_database": 0.01,
                "mean_database": 0.01,
            },
            seed_wins={"worst_database": 1, "mean_database": 1},
            balanced_safety_every_seed=True,
            weakest_database="icbhi2017",
            bootstrap={
                "per_domain": {
                    "icbhi2017": {"probability_gain_above_zero": 1.0}
                },
                "mean_domain": {"probability_gain_above_zero": 1.0},
            },
        )
