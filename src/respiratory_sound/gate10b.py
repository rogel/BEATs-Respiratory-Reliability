"""Decision utility for Gate 10B sampling-diverse LoRA distillation."""

from __future__ import annotations

from typing import Any


def distillation_decision(
    *,
    student: dict[str, Any],
    control: dict[str, Any],
    teacher_scores: dict[str, float],
    bootstrap: dict[str, Any],
    teacher_alignment: dict[str, float],
) -> dict[str, Any]:
    """Apply the frozen Gate 10B single-seed lower-bound rule."""
    domains = {"icbhi2017", "sprsound2022"}
    student_metrics = student["best_validation_metrics"]
    control_metrics = control["best_validation_metrics"]
    if set(student_metrics) != domains or set(control_metrics) != domains:
        raise ValueError("Gate 10B requires ICBHI and SPRSound metrics")
    student_scores = {
        domain: float(student_metrics[domain]["average_score"])
        for domain in domains
    }
    control_scores = {
        domain: float(control_metrics[domain]["average_score"])
        for domain in domains
    }
    gains = {
        domain: student_scores[domain] - control_scores[domain]
        for domain in domains
    }
    student_worst = min(student_scores.values())
    control_worst = min(control_scores.values())
    student_mean = sum(student_scores.values()) / len(domains)
    control_mean = sum(control_scores.values()) / len(domains)
    worst_gain = student_worst - control_worst
    mean_gain = student_mean - control_mean
    weakest_control_database = min(control_scores, key=control_scores.get)

    branch_a_direction = worst_gain >= 0.003 and mean_gain >= 0.0
    branch_a_bootstrap = (
        float(
            bootstrap["per_domain"][weakest_control_database][
                "probability_gain_above_zero"
            ]
        )
        >= 0.60
    )
    branch_a = branch_a_direction and branch_a_bootstrap
    branch_b_direction = mean_gain >= 0.003 and worst_gain >= 0.0
    branch_b_bootstrap = (
        float(
            bootstrap["mean_domain"]["probability_gain_above_zero"]
        )
        >= 0.60
    )
    branch_b = branch_b_direction and branch_b_bootstrap

    safety = all(
        float(metrics[metric]) >= 0.5
        for metrics in student_metrics.values()
        for metric in ("sensitivity", "specificity")
    )
    teacher_worst = float(teacher_scores["worst_database"])
    teacher_mean = float(teacher_scores["mean_database"])
    checks = {
        "at_least_one_directional_branch": branch_a or branch_b,
        "maximum_single_database_drop_from_control": (
            min(gains.values()) >= -0.005
        ),
        "student_sensitivity_specificity_safety": safety,
        "teacher_worst_database_retention": (
            student_worst >= teacher_worst - 0.01
        ),
        "teacher_mean_database_retention": (
            student_mean >= teacher_mean - 0.01
        ),
        "teacher_alignment_relative_kl_reduction": (
            float(teacher_alignment["relative_reduction"]) >= 0.05
        ),
    }
    return {
        "student_scores": student_scores,
        "control_scores": control_scores,
        "teacher_scores": teacher_scores,
        "gains": gains
        | {
            "worst_database": worst_gain,
            "mean_database": mean_gain,
        },
        "student_worst_database": student_worst,
        "student_mean_database": student_mean,
        "weakest_control_database": weakest_control_database,
        "branches": {
            "worst_database": {
                "direction_passed": branch_a_direction,
                "bootstrap_passed": branch_a_bootstrap,
                "passed": branch_a,
            },
            "mean_database": {
                "direction_passed": branch_b_direction,
                "bootstrap_passed": branch_b_bootstrap,
                "passed": branch_b,
            },
        },
        "checks": checks,
        "passed": all(checks.values()),
    }
