"""Pre-registered decision rules for Gate 9A."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

DOMAINS = ("icbhi2017", "sprsound2022")
GATE5A_MEAN_ANCHOR = 0.7740732347541495


def single_seed_decision(summary: dict[str, Any]) -> dict[str, Any]:
    metrics = summary["best_validation_metrics"]
    icbhi = metrics["icbhi2017"]
    sprsound = metrics["sprsound2022"]
    mean_score = float(
        np.mean([icbhi["average_score"], sprsound["average_score"]])
    )
    checks = {
        "icbhi_average_score_at_least_0_70": (
            float(icbhi["average_score"]) >= 0.70
        ),
        "sprsound_average_score_at_least_0_88": (
            float(sprsound["average_score"]) >= 0.88
        ),
        "two_domain_mean_average_score_at_least_0_79": mean_score >= 0.79,
        "mean_gain_over_gate5a_at_least_0_01": (
            mean_score - GATE5A_MEAN_ANCHOR >= 0.01
        ),
        "sensitivity_each_domain_at_least_0_50": all(
            float(metrics[domain]["sensitivity"]) >= 0.50
            for domain in DOMAINS
        ),
        "specificity_each_domain_at_least_0_50": all(
            float(metrics[domain]["specificity"]) >= 0.50
            for domain in DOMAINS
        ),
    }
    return {
        "seed": int(summary["seed"]),
        "candidate": str(summary["candidate"]),
        "scores": {
            "icbhi2017": float(icbhi["average_score"]),
            "sprsound2022": float(sprsound["average_score"]),
            "two_domain_mean": mean_score,
            "mean_gain_over_gate5a": mean_score - GATE5A_MEAN_ANCHOR,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def three_seed_decision(summaries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if len(summaries) != 3:
        raise ValueError("Gate 9A extension requires exactly three seed summaries")
    candidates = {str(summary["candidate"]) for summary in summaries}
    seeds = {int(summary["seed"]) for summary in summaries}
    if len(candidates) != 1:
        raise ValueError("Three-seed summaries must use one candidate")
    if len(seeds) != 3:
        raise ValueError("Three-seed summaries must use distinct seeds")
    single = [single_seed_decision(summary) for summary in summaries]
    mean_scores = {
        domain: float(
            np.mean(
                [
                    summary["best_validation_metrics"][domain]["average_score"]
                    for summary in summaries
                ]
            )
        )
        for domain in DOMAINS
    }
    two_domain_mean = float(np.mean(tuple(mean_scores.values())))
    checks = {
        "mean_icbhi_average_score_at_least_0_70": (
            mean_scores["icbhi2017"] >= 0.70
        ),
        "mean_sprsound_average_score_at_least_0_88": (
            mean_scores["sprsound2022"] >= 0.88
        ),
        "mean_two_domain_average_score_at_least_0_79": two_domain_mean >= 0.79,
        "single_seed_gate_passes_at_least_2": (
            sum(decision["passed"] for decision in single) >= 2
        ),
        "all_seed_domain_sensitivity_and_specificity_at_least_0_50": all(
            float(summary["best_validation_metrics"][domain][metric]) >= 0.50
            for summary in summaries
            for domain in DOMAINS
            for metric in ("sensitivity", "specificity")
        ),
    }
    return {
        "candidate": next(iter(candidates)),
        "seeds": sorted(seeds),
        "single_seed_decisions": single,
        "mean_scores": {
            **mean_scores,
            "two_domain_mean": two_domain_mean,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def resolve_gate_status(
    candidate_results: dict[str, dict[str, Any]],
    *,
    both_single_seed_screens_complete: bool,
) -> str:
    """Resolve the overall Gate 9A state without changing frozen thresholds."""
    if not both_single_seed_screens_complete:
        return "await_remaining_single_seed_screen"
    single_seed_passers = [
        candidate
        for candidate, result in candidate_results.items()
        if result["single_seed_screen_passed"]
    ]
    if not single_seed_passers:
        return "stop_pretrained_route"
    confirmations = [
        candidate_results[candidate]["three_seed_decision"]
        for candidate in single_seed_passers
    ]
    if any(confirmation is None for confirmation in confirmations):
        return "extend_only_single_seed_passers"
    if any(confirmation["passed"] for confirmation in confirmations):
        return "gate9a_passed_three_seed_confirmation"
    return "gate9a_failed_three_seed_confirmation"
