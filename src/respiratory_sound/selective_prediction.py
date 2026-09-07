"""Frozen selective-prediction utilities for the post-Gate-11A study."""

from __future__ import annotations

import numpy as np

from respiratory_sound.calibration import clip_probabilities


def normalized_binary_entropy(probabilities: np.ndarray) -> np.ndarray:
    """Return binary predictive entropy on [0, 1]."""
    values = clip_probabilities(probabilities)
    entropy = -(values * np.log(values) + (1.0 - values) * np.log1p(-values))
    return entropy / np.log(2.0)


def fit_uncertainty_cutoff(
    uncertainties: np.ndarray,
    target_coverage: float,
) -> float:
    """Fit a deterministic cutoff retaining at least the requested calibration share."""
    values = np.asarray(uncertainties, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("uncertainties must be a non-empty finite vector")
    if not 0.0 < target_coverage <= 1.0:
        raise ValueError("target_coverage must be in (0, 1]")
    keep = max(1, int(np.ceil(target_coverage * len(values))))
    return float(np.sort(values, kind="stable")[keep - 1])


def selective_operating_point(
    targets: np.ndarray,
    probabilities: np.ndarray,
    uncertainty_cutoff: float,
) -> dict[str, float | int]:
    """Evaluate a fixed uncertainty cutoff without changing the classifier threshold."""
    labels = np.asarray(targets, dtype=np.int64)
    values = clip_probabilities(probabilities)
    if labels.shape != values.shape or not np.isin(labels, (0, 1)).all():
        raise ValueError("targets/probabilities must be aligned binary vectors")
    uncertainty = normalized_binary_entropy(values)
    accepted = uncertainty <= float(uncertainty_cutoff)
    if not accepted.any():
        raise ValueError("uncertainty cutoff rejected every sample")
    predictions = values >= 0.5
    errors = predictions != labels
    class_coverages = []
    for class_id in (0, 1):
        class_mask = labels == class_id
        if not class_mask.any():
            raise ValueError("both classes are required for class-coverage safety")
        class_coverages.append(float(accepted[class_mask].mean()))
    accepted_labels = labels[accepted]
    accepted_predictions = predictions[accepted]
    sensitivity = (
        float(accepted_predictions[accepted_labels == 1].mean())
        if bool((accepted_labels == 1).any())
        else float("nan")
    )
    specificity = (
        float((~accepted_predictions[accepted_labels == 0]).mean())
        if bool((accepted_labels == 0).any())
        else float("nan")
    )
    average_score = (
        (sensitivity + specificity) / 2.0
        if np.isfinite(sensitivity) and np.isfinite(specificity)
        else float("nan")
    )
    return {
        "accepted": int(accepted.sum()),
        "total": int(len(labels)),
        "coverage": float(accepted.mean()),
        "normal_coverage": class_coverages[0],
        "adventitious_coverage": class_coverages[1],
        "class_coverage_gap": abs(class_coverages[0] - class_coverages[1]),
        "selective_error_risk": float(errors[accepted].mean()),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "average_score": average_score,
        "balanced_error_risk": 1.0 - average_score,
        "uncertainty_cutoff": float(uncertainty_cutoff),
    }


def risk_coverage_curve(
    targets: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[dict[str, float], np.ndarray]:
    """Return discrete error-risk/coverage curve and AURC."""
    labels = np.asarray(targets, dtype=np.int64)
    values = clip_probabilities(probabilities)
    if labels.shape != values.shape or not np.isin(labels, (0, 1)).all():
        raise ValueError("targets/probabilities must be aligned binary vectors")
    uncertainties = normalized_binary_entropy(values)
    order = np.argsort(uncertainties, kind="stable")
    ordered_errors = ((values >= 0.5) != labels)[order].astype(np.float64)
    accepted = np.arange(1, len(labels) + 1, dtype=np.float64)
    coverages = accepted / len(labels)
    risks = np.cumsum(ordered_errors) / accepted
    curve = np.column_stack((coverages, risks, uncertainties[order]))
    return {
        "aurc": float(risks.mean()),
        "full_coverage_error_risk": float(ordered_errors.mean()),
    }, curve


def equal_frequency_reliability_table(
    targets: np.ndarray,
    probabilities: np.ndarray,
    *,
    bins: int = 15,
) -> list[dict[str, float | int]]:
    labels = np.asarray(targets, dtype=np.int64)
    values = clip_probabilities(probabilities)
    if labels.shape != values.shape or not np.isin(labels, (0, 1)).all():
        raise ValueError("targets/probabilities must be aligned binary vectors")
    if bins < 2:
        raise ValueError("bins must be at least two")
    order = np.argsort(values, kind="stable")
    groups = np.array_split(order, min(bins, len(values)))
    return [
        {
            "bin": index + 1,
            "count": int(len(group)),
            "mean_probability": float(values[group].mean()),
            "observed_frequency": float(labels[group].mean()),
            "absolute_gap": abs(
                float(values[group].mean()) - float(labels[group].mean())
            ),
        }
        for index, group in enumerate(groups)
        if len(group)
    ]
