"""Lightweight probability calibration for patient-separated deployment studies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize, minimize_scalar


def _as_binary_targets(targets: np.ndarray) -> np.ndarray:
    values = np.asarray(targets, dtype=np.int64)
    if values.ndim != 1 or not np.isin(values, (0, 1)).all():
        raise ValueError("targets must be a one-dimensional binary array")
    if np.unique(values).size != 2:
        raise ValueError("targets must contain both classes")
    return values


def clip_probabilities(probabilities: np.ndarray, eps: float = 1.0e-6) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("probabilities must be a finite one-dimensional array")
    if not 0.0 < eps < 0.5:
        raise ValueError("eps must be in (0, 0.5)")
    return np.clip(values, eps, 1.0 - eps)


def probabilities_to_logits(probabilities: np.ndarray) -> np.ndarray:
    values = clip_probabilities(probabilities)
    return np.log(values) - np.log1p(-values)


def sigmoid(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    positive = values >= 0
    outputs = np.empty_like(values)
    outputs[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    outputs[~positive] = exponent / (1.0 + exponent)
    return outputs


def binary_negative_log_likelihood(
    targets: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    labels = _as_binary_targets(targets)
    values = clip_probabilities(probabilities)
    return float(
        -np.mean(labels * np.log(values) + (1 - labels) * np.log1p(-values))
    )


def brier_score(targets: np.ndarray, probabilities: np.ndarray) -> float:
    labels = _as_binary_targets(targets)
    values = clip_probabilities(probabilities)
    return float(np.mean((values - labels) ** 2))


def equal_frequency_ece(
    targets: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 15,
) -> float:
    labels = _as_binary_targets(targets)
    values = clip_probabilities(probabilities)
    if bins < 2:
        raise ValueError("bins must be at least two")
    ordered_indices = np.argsort(values, kind="stable")
    groups = np.array_split(ordered_indices, min(bins, len(values)))
    return float(
        sum(
            len(group)
            / len(values)
            * abs(float(values[group].mean()) - float(labels[group].mean()))
            for group in groups
            if len(group)
        )
    )


def probability_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 15,
) -> dict[str, float]:
    return {
        "negative_log_likelihood": binary_negative_log_likelihood(
            targets,
            probabilities,
        ),
        "brier_score": brier_score(targets, probabilities),
        "equal_frequency_ece": equal_frequency_ece(
            targets,
            probabilities,
            bins=bins,
        ),
    }


@dataclass(frozen=True)
class TemperatureScaler:
    temperature: float

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        return sigmoid(probabilities_to_logits(probabilities) / self.temperature)


@dataclass(frozen=True)
class PositivePlattScaler:
    slope: float
    intercept: float

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        logits = probabilities_to_logits(probabilities)
        return sigmoid(self.slope * logits + self.intercept)


def fit_temperature(
    targets: np.ndarray,
    probabilities: np.ndarray,
) -> TemperatureScaler:
    labels = _as_binary_targets(targets)
    logits = probabilities_to_logits(probabilities)

    def objective(log_temperature: float) -> float:
        calibrated = sigmoid(logits / np.exp(log_temperature))
        return binary_negative_log_likelihood(labels, calibrated)

    result = minimize_scalar(
        objective,
        bounds=(-5.0, 5.0),
        method="bounded",
        options={"xatol": 1.0e-10},
    )
    if not result.success:
        raise RuntimeError(f"Temperature fitting failed: {result.message}")
    return TemperatureScaler(temperature=float(np.exp(result.x)))


def fit_positive_platt(
    targets: np.ndarray,
    probabilities: np.ndarray,
) -> PositivePlattScaler:
    labels = _as_binary_targets(targets)
    logits = probabilities_to_logits(probabilities)
    prevalence = np.clip(labels.mean(), 1.0e-4, 1.0 - 1.0e-4)
    initial_intercept = float(
        np.log(prevalence) - np.log1p(-prevalence) - logits.mean()
    )

    def objective(parameters: np.ndarray) -> float:
        slope = np.exp(parameters[0])
        calibrated = sigmoid(slope * logits + parameters[1])
        return binary_negative_log_likelihood(labels, calibrated)

    result = minimize(
        objective,
        x0=np.asarray((0.0, initial_intercept)),
        method="L-BFGS-B",
        bounds=((-5.0, 5.0), (-20.0, 20.0)),
        options={"ftol": 1.0e-12, "gtol": 1.0e-9, "maxiter": 1_000},
    )
    if not result.success:
        raise RuntimeError(f"Platt fitting failed: {result.message}")
    return PositivePlattScaler(
        slope=float(np.exp(result.x[0])),
        intercept=float(result.x[1]),
    )


def balanced_accuracy(
    targets: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> float:
    labels = _as_binary_targets(targets)
    predictions = clip_probabilities(probabilities) >= threshold
    sensitivity = float(predictions[labels == 1].mean())
    specificity = float((~predictions[labels == 0]).mean())
    return (sensitivity + specificity) / 2.0


def select_balanced_threshold(
    targets: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float, float]:
    labels = _as_binary_targets(targets)
    values = clip_probabilities(probabilities)
    candidates = np.concatenate(
        (
            np.unique(values),
            np.asarray((np.nextafter(values.max(), np.inf),)),
        )
    )
    scores = np.asarray(
        [balanced_accuracy(labels, values, threshold) for threshold in candidates]
    )
    best_score = float(scores.max())
    best_candidates = candidates[np.isclose(scores, best_score, atol=1.0e-12)]
    distances = np.abs(best_candidates - 0.5)
    closest = best_candidates[np.isclose(distances, distances.min(), atol=1.0e-12)]
    return float(closest.min()), best_score
