"""Frozen Gate 9B decision rules."""

from __future__ import annotations

from typing import Any

DOMAIN_THRESHOLDS = {
    "icbhi2017": 0.69,
    "sprsound2022": 0.86,
}
MEAN_THRESHOLD = 0.79
BALANCE_THRESHOLD = 0.5
MAX_TRAINABLE_FRACTION = 0.01
MIN_WORST_DOMAIN_GAIN = 0.003
MIN_MEAN_WITH_WORST_GAIN = -0.002
MIN_MEAN_GAIN = 0.002
MIN_WORST_WITH_MEAN_GAIN = -0.003
MAX_DOMAIN_DROP = 0.01


def selected_scores(summary: dict[str, Any]) -> dict[str, float]:
    """Extract the selected-checkpoint Gate 9B scores."""
    metrics = summary["best_validation_metrics"]
    domain_scores = {
        domain: float(metrics[domain]["average_score"])
        for domain in DOMAIN_THRESHOLDS
    }
    mean_score = sum(domain_scores.values()) / len(domain_scores)
    recorded_mean = float(summary["best_mean_domain_average_score"])
    if abs(mean_score - recorded_mean) > 1.0e-12:
        raise ValueError("Gate 9B selected mean does not match domain metrics")
    return {
        **domain_scores,
        "worst_domain": min(domain_scores.values()),
        "two_domain_mean": mean_score,
    }


def noninferiority_decision(summary: dict[str, Any]) -> dict[str, Any]:
    """Apply the deliberately permissive Gate 9B retention bounds."""
    scores = selected_scores(summary)
    metrics = summary["best_validation_metrics"]
    domain_checks = {
        domain: scores[domain] >= threshold
        for domain, threshold in DOMAIN_THRESHOLDS.items()
    }
    balance_checks = {
        f"{domain}_{metric}": float(metrics[domain][metric]) >= BALANCE_THRESHOLD
        for domain in DOMAIN_THRESHOLDS
        for metric in ("sensitivity", "specificity")
    }
    checks = {
        **{f"{domain}_average_score": passed for domain, passed in domain_checks.items()},
        "two_domain_mean_average_score": (
            scores["two_domain_mean"] >= MEAN_THRESHOLD
        ),
        "all_domain_sensitivity_and_specificity": all(balance_checks.values()),
    }
    return {
        "scores": scores,
        "checks": checks,
        "balance_checks": balance_checks,
        "passed": all(checks.values()),
    }


def proposed_screen_decision(
    anchored_summary: dict[str, Any],
    plain_summary: dict[str, Any],
) -> dict[str, Any]:
    """Compare the proposed anchored LoRA with its matched plain-LoRA ablation."""
    anchored_noninferiority = noninferiority_decision(anchored_summary)
    plain_noninferiority = noninferiority_decision(plain_summary)
    anchored = anchored_noninferiority["scores"]
    plain = plain_noninferiority["scores"]
    gains = {
        key: anchored[key] - plain[key]
        for key in ("icbhi2017", "sprsound2022", "worst_domain", "two_domain_mean")
    }
    directional_branch_worst = (
        gains["worst_domain"] >= MIN_WORST_DOMAIN_GAIN
        and gains["two_domain_mean"] >= MIN_MEAN_WITH_WORST_GAIN
    )
    directional_branch_mean = (
        gains["two_domain_mean"] >= MIN_MEAN_GAIN
        and gains["worst_domain"] >= MIN_WORST_WITH_MEAN_GAIN
    )
    domain_drop_checks = {
        domain: gains[domain] >= -MAX_DOMAIN_DROP
        for domain in DOMAIN_THRESHOLDS
    }
    trainable_fraction = float(anchored_summary["trainable_fraction"])
    checks = {
        "anchored_noninferiority": anchored_noninferiority["passed"],
        "plain_lora_noninferiority": plain_noninferiority["passed"],
        "trainable_fraction_at_most_one_percent": (
            trainable_fraction <= MAX_TRAINABLE_FRACTION
        ),
        "directional_gain": directional_branch_worst or directional_branch_mean,
        "maximum_domain_drop": all(domain_drop_checks.values()),
    }
    return {
        "anchored_noninferiority": anchored_noninferiority,
        "plain_lora_noninferiority": plain_noninferiority,
        "gains_over_plain_lora": gains,
        "directional_gain_branches": {
            "worst_domain_branch": directional_branch_worst,
            "two_domain_mean_branch": directional_branch_mean,
        },
        "domain_drop_checks": domain_drop_checks,
        "trainable_fraction": trainable_fraction,
        "checks": checks,
        "passed": all(checks.values()),
    }


def epoch_noninferiority(row: dict[str, Any]) -> bool:
    """Return whether one historical epoch meets the same retention bounds."""
    domain_checks = all(
        float(row[f"validation_{domain}_average_score"]) >= threshold
        for domain, threshold in DOMAIN_THRESHOLDS.items()
    )
    balance_checks = all(
        float(row[f"validation_{domain}_{metric}"]) >= BALANCE_THRESHOLD
        for domain in DOMAIN_THRESHOLDS
        for metric in ("sensitivity", "specificity")
    )
    return (
        domain_checks
        and float(row["validation_mean_average_score"]) >= MEAN_THRESHOLD
        and balance_checks
    )
