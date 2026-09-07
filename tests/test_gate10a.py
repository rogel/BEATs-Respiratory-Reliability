from __future__ import annotations

from typing import Any

from respiratory_sound.gate10a import sampling_diverse_ensemble_decision


def _metrics(icbhi: float, sprsound: float) -> dict[str, dict[str, float]]:
    return {
        "icbhi2017": {
            "average_score": icbhi,
            "sensitivity": 0.70,
            "specificity": 0.70,
        },
        "sprsound2022": {
            "average_score": sprsound,
            "sensitivity": 0.80,
            "specificity": 0.90,
        },
    }


def _bootstrap(
    icbhi_probability: float,
    mean_probability: float,
) -> dict[str, Any]:
    return {
        "per_domain": {
            "icbhi2017": {
                "probability_gain_above_zero": icbhi_probability
            },
            "sprsound2022": {"probability_gain_above_zero": 0.9},
        },
        "mean_domain": {
            "probability_gain_above_zero": mean_probability
        },
    }


def _diversity() -> dict[str, float]:
    return {
        "disagreement_fraction": 0.05,
        "balanced_unique_correct_fraction": 0.02,
        "event_random_unique_correct_fraction": 0.03,
    }


def test_gate10a_accepts_small_worst_database_gain() -> None:
    decision = sampling_diverse_ensemble_decision(
        candidate_metrics=_metrics(0.704, 0.899),
        reference_metrics=_metrics(0.700, 0.900),
        parent_diversity=_diversity(),
        bootstrap=_bootstrap(0.75, 0.8),
        reference_name="event_random_parent",
    )
    assert decision["branches"]["worst_database"]["passed"]
    assert decision["passed"]


def test_gate10a_accepts_mean_branch_with_small_worst_tolerance() -> None:
    decision = sampling_diverse_ensemble_decision(
        candidate_metrics=_metrics(0.699, 0.908),
        reference_metrics=_metrics(0.700, 0.900),
        parent_diversity=_diversity(),
        bootstrap=_bootstrap(0.4, 0.75),
        reference_name="event_random_parent",
    )
    assert decision["branches"]["mean_database"]["passed"]
    assert decision["passed"]


def test_gate10a_rejects_noncomplementary_parents() -> None:
    diversity = _diversity()
    diversity["disagreement_fraction"] = 0.02
    decision = sampling_diverse_ensemble_decision(
        candidate_metrics=_metrics(0.704, 0.904),
        reference_metrics=_metrics(0.700, 0.900),
        parent_diversity=diversity,
        bootstrap=_bootstrap(0.8, 0.8),
        reference_name="event_random_parent",
    )
    assert not decision["checks"]["parent_prediction_disagreement"]
    assert not decision["passed"]
