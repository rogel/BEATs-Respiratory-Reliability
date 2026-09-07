from __future__ import annotations

from typing import Any

from respiratory_sound.gate10b import distillation_decision


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
                "probability_gain_above_zero": icbhi_probability
            },
            "sprsound2022": {"probability_gain_above_zero": 0.9},
        },
        "mean_domain": {
            "probability_gain_above_zero": mean_probability
        },
    }


def test_gate10b_accepts_small_distilled_gain_and_teacher_retention() -> None:
    decision = distillation_decision(
        student=_summary(0.704, 0.884),
        control=_summary(0.700, 0.880),
        teacher_scores={
            "worst_database": 0.710,
            "mean_database": 0.800,
        },
        bootstrap=_bootstrap(0.7, 0.8),
        teacher_alignment={"relative_reduction": 0.10},
    )
    assert decision["branches"]["worst_database"]["passed"]
    assert decision["passed"]


def test_gate10b_rejects_performance_loss_despite_alignment() -> None:
    decision = distillation_decision(
        student=_summary(0.685, 0.885),
        control=_summary(0.700, 0.880),
        teacher_scores={
            "worst_database": 0.703,
            "mean_database": 0.791,
        },
        bootstrap=_bootstrap(0.1, 0.4),
        teacher_alignment={"relative_reduction": 0.50},
    )
    assert decision["checks"]["teacher_alignment_relative_kl_reduction"]
    assert not decision["checks"]["maximum_single_database_drop_from_control"]
    assert not decision["passed"]
