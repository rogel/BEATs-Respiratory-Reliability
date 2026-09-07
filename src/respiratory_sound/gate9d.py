"""Decision utilities for the matched Gate 9D sampling ablation."""

from __future__ import annotations

from typing import Any


def _domain_scores(summary: dict[str, Any]) -> dict[str, float]:
    return {
        domain: float(metrics["average_score"])
        for domain, metrics in summary["best_validation_metrics"].items()
    }


def sampling_ablation_decision(
    balanced: dict[str, Any],
    event_random: dict[str, Any],
    bootstrap: dict[str, Any],
) -> dict[str, Any]:
    """Apply the frozen low directional threshold without post-hoc selection."""
    balanced_scores = _domain_scores(balanced)
    random_scores = _domain_scores(event_random)
    if set(balanced_scores) != {"icbhi2017", "sprsound2022"}:
        raise ValueError("Gate 9D requires ICBHI and SPRSound metrics")
    gains = {
        domain: balanced_scores[domain] - random_scores[domain]
        for domain in balanced_scores
    }
    balanced_worst = min(balanced_scores.values())
    random_worst = min(random_scores.values())
    worst_gain = balanced_worst - random_worst
    mean_gain = (
        sum(balanced_scores.values()) - sum(random_scores.values())
    ) / len(balanced_scores)
    safety = all(
        float(metrics[metric]) >= 0.5
        for metrics in balanced["best_validation_metrics"].values()
        for metric in ("sensitivity", "specificity")
    )
    branch_a_direction = worst_gain >= 0.003 and mean_gain >= 0.0
    branch_b_direction = mean_gain >= 0.003 and worst_gain >= 0.0
    weakest_domain = min(balanced_scores, key=balanced_scores.get)
    branch_a_bootstrap = (
        float(
            bootstrap["per_domain"][weakest_domain][
                "probability_gain_above_zero"
            ]
        )
        >= 0.60
    )
    branch_b_bootstrap = (
        float(
            bootstrap["mean_domain"]["probability_gain_above_zero"]
        )
        >= 0.60
    )
    branch_a = branch_a_direction and branch_a_bootstrap
    branch_b = branch_b_direction and branch_b_bootstrap
    checks = {
        "at_least_one_directional_branch": branch_a or branch_b,
        "maximum_single_database_drop": min(gains.values()) >= -0.005,
        "balanced_sensitivity_specificity_safety": safety,
    }
    return {
        "balanced_scores": balanced_scores,
        "event_random_scores": random_scores,
        "gains": gains
        | {
            "worst_database": worst_gain,
            "mean_database": mean_gain,
        },
        "weakest_database": weakest_domain,
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


def multiseed_sampling_decision(
    *,
    paired_seed_results: list[dict[str, Any]],
    mean_paired_gains: dict[str, float],
    seed_wins: dict[str, int],
    balanced_safety_every_seed: bool,
    weakest_database: str,
    bootstrap: dict[str, Any],
) -> dict[str, Any]:
    """Apply the Gate 9D three-seed lower-bound rule frozen in advance."""
    if len(paired_seed_results) != 3:
        raise ValueError("Gate 9D multi-seed confirmation requires three seed pairs")
    domains = {"icbhi2017", "sprsound2022"}
    if weakest_database not in domains:
        raise ValueError("Weakest database must be ICBHI or SPRSound")
    if not domains.issubset(mean_paired_gains):
        raise ValueError("Mean paired gains are missing a required database")

    branch_a_direction = (
        float(mean_paired_gains["worst_database"]) >= 0.003
        and float(mean_paired_gains["mean_database"]) >= 0.0
    )
    branch_a_wins = int(seed_wins["worst_database"]) >= 2
    branch_a_bootstrap = (
        float(
            bootstrap["per_domain"][weakest_database][
                "probability_gain_above_zero"
            ]
        )
        >= 0.80
    )
    branch_a = branch_a_direction and branch_a_wins and branch_a_bootstrap

    branch_b_direction = (
        float(mean_paired_gains["mean_database"]) >= 0.003
        and float(mean_paired_gains["worst_database"]) >= 0.0
    )
    branch_b_wins = int(seed_wins["mean_database"]) >= 2
    branch_b_bootstrap = (
        float(
            bootstrap["mean_domain"]["probability_gain_above_zero"]
        )
        >= 0.80
    )
    branch_b = branch_b_direction and branch_b_wins and branch_b_bootstrap

    checks = {
        "at_least_one_directional_branch": branch_a or branch_b,
        "no_database_cross_seed_mean_drop_greater_than_0_005": all(
            float(mean_paired_gains[domain]) >= -0.005
            for domain in domains
        ),
        "balanced_sensitivity_specificity_safety_every_seed": bool(
            balanced_safety_every_seed
        ),
    }
    return {
        "thresholds": {
            "directional_gain": 0.003,
            "paired_wins": 2,
            "maximum_database_mean_drop": 0.005,
            "sensitivity_specificity_floor": 0.5,
            "bootstrap_probability": 0.80,
        },
        "weakest_database": weakest_database,
        "branches": {
            "worst_database": {
                "direction_passed": branch_a_direction,
                "wins_passed": branch_a_wins,
                "bootstrap_passed": branch_a_bootstrap,
                "passed": branch_a,
            },
            "mean_database": {
                "direction_passed": branch_b_direction,
                "wins_passed": branch_b_wins,
                "bootstrap_passed": branch_b_bootstrap,
                "passed": branch_b,
            },
        },
        "checks": checks,
        "passed": all(checks.values()),
    }
