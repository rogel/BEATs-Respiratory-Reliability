"""Decision utility for the frozen Gate 10A sampling-diverse ensemble."""

from __future__ import annotations

from typing import Any


def sampling_diverse_ensemble_decision(
    *,
    candidate_metrics: dict[str, dict[str, float]],
    reference_metrics: dict[str, dict[str, float]],
    parent_diversity: dict[str, float],
    bootstrap: dict[str, Any],
    reference_name: str,
) -> dict[str, Any]:
    """Apply the single deterministic Gate 10A lower-bound rule."""
    domains = {"icbhi2017", "sprsound2022"}
    if set(candidate_metrics) != domains or set(reference_metrics) != domains:
        raise ValueError("Gate 10A requires ICBHI and SPRSound metrics")
    candidate_scores = {
        domain: float(candidate_metrics[domain]["average_score"])
        for domain in domains
    }
    reference_scores = {
        domain: float(reference_metrics[domain]["average_score"])
        for domain in domains
    }
    gains = {
        domain: candidate_scores[domain] - reference_scores[domain]
        for domain in domains
    }
    candidate_worst = min(candidate_scores.values())
    reference_worst = min(reference_scores.values())
    worst_gain = candidate_worst - reference_worst
    mean_gain = (
        sum(candidate_scores.values()) - sum(reference_scores.values())
    ) / len(domains)
    weakest_reference_database = min(
        reference_scores,
        key=reference_scores.get,
    )

    branch_a_direction = worst_gain >= 0.003 and mean_gain >= -0.002
    branch_a_bootstrap = (
        float(
            bootstrap["per_domain"][weakest_reference_database][
                "probability_gain_above_zero"
            ]
        )
        >= 0.70
    )
    branch_a = branch_a_direction and branch_a_bootstrap

    branch_b_direction = mean_gain >= 0.003 and worst_gain >= -0.002
    branch_b_bootstrap = (
        float(
            bootstrap["mean_domain"]["probability_gain_above_zero"]
        )
        >= 0.70
    )
    branch_b = branch_b_direction and branch_b_bootstrap

    safety = all(
        float(metrics[metric]) >= 0.5
        for metrics in candidate_metrics.values()
        for metric in ("sensitivity", "specificity")
    )
    checks = {
        "at_least_one_directional_branch": branch_a or branch_b,
        "maximum_single_database_drop": min(gains.values()) >= -0.005,
        "candidate_sensitivity_specificity_safety": safety,
        "parent_prediction_disagreement": (
            float(parent_diversity["disagreement_fraction"]) >= 0.03
        ),
        "each_parent_has_unique_correct_events": min(
            float(parent_diversity["balanced_unique_correct_fraction"]),
            float(parent_diversity["event_random_unique_correct_fraction"]),
        )
        >= 0.005,
    }
    return {
        "reference": reference_name,
        "candidate_scores": candidate_scores,
        "reference_scores": reference_scores,
        "gains": gains
        | {
            "worst_database": worst_gain,
            "mean_database": mean_gain,
        },
        "weakest_reference_database": weakest_reference_database,
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
