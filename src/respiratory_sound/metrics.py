"""Metrics for respiratory-sound classification tasks."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

CLASS_NAMES = ("normal", "crackle", "wheeze", "both")


def respiratory_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    class_names: tuple[str, ...],
) -> dict[str, Any]:
    """Compute official normal/abnormal scores and class-balanced metrics."""
    matrix = confusion_matrix(targets, predictions, labels=np.arange(len(class_names)))
    supports = matrix.sum(axis=1)
    true_positives = np.diag(matrix)
    recalls = np.divide(
        true_positives,
        supports,
        out=np.zeros_like(true_positives, dtype=np.float64),
        where=supports > 0,
    )
    specificity = float(recalls[0])
    abnormal_support = int(supports[1:].sum())
    sensitivity = (
        float(true_positives[1:].sum() / abnormal_support) if abnormal_support > 0 else 0.0
    )
    average_score = (sensitivity + specificity) / 2.0
    harmonic_score = (
        2.0 * sensitivity * specificity / (sensitivity + specificity)
        if sensitivity + specificity > 0
        else 0.0
    )
    score = (average_score + harmonic_score) / 2.0
    return {
        "accuracy": float(true_positives.sum() / matrix.sum()),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "average_score": average_score,
        "harmonic_score": harmonic_score,
        "score": score,
        "macro_f1": float(
            f1_score(
                targets,
                predictions,
                labels=np.arange(len(class_names)),
                average="macro",
                zero_division=0,
            )
        ),
        "uar": float(recalls.mean()),
        "per_class_recall": {
            class_name: float(recall)
            for class_name, recall in zip(class_names, recalls, strict=True)
        },
        "confusion_matrix": matrix.tolist(),
        "support": {
            class_name: int(support)
            for class_name, support in zip(class_names, supports, strict=True)
        },
    }


def icbhi_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, Any]:
    """Compute the standard ICBHI score and complementary class-balanced metrics."""
    metrics = respiratory_metrics(targets, predictions, CLASS_NAMES)
    metrics["icbhi_score"] = metrics["average_score"]
    return metrics


def patient_bootstrap_intervals(
    targets: np.ndarray,
    predictions: np.ndarray,
    patient_ids: np.ndarray,
    iterations: int = 2_000,
    seed: int = 20_260_727,
    confidence: float = 0.95,
    class_names: tuple[str, ...] | None = None,
) -> dict[str, dict[str, float]]:
    """Cluster bootstrap metrics by patient, preserving cycles within a patient."""
    if not (len(targets) == len(predictions) == len(patient_ids)):
        raise ValueError("targets, predictions, and patient_ids must have equal length")
    unique_patients = np.unique(patient_ids)
    patient_indices = {
        patient: np.flatnonzero(patient_ids == patient) for patient in unique_patients
    }
    random_generator = np.random.default_rng(seed)
    metric_function: Callable[[np.ndarray, np.ndarray], dict[str, Any]]
    if class_names is None:
        metric_function = icbhi_metrics
        tracked_metrics = ("sensitivity", "specificity", "icbhi_score", "macro_f1", "uar")
    else:
        def metric_function(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
            return respiratory_metrics(y, p, class_names)

        tracked_metrics = (
            "sensitivity",
            "specificity",
            "average_score",
            "harmonic_score",
            "score",
            "macro_f1",
            "uar",
        )
    samples = {metric: [] for metric in tracked_metrics}
    for _ in range(iterations):
        sampled_patients = random_generator.choice(
            unique_patients,
            size=len(unique_patients),
            replace=True,
        )
        indices = np.concatenate([patient_indices[patient] for patient in sampled_patients])
        metrics = metric_function(targets[indices], predictions[indices])
        for metric in tracked_metrics:
            samples[metric].append(float(metrics[metric]))
    alpha = (1.0 - confidence) / 2.0
    point_metrics = metric_function(targets, predictions)
    return {
        metric: {
            "estimate": float(point_metrics[metric]),
            "lower": float(np.quantile(values, alpha)),
            "upper": float(np.quantile(values, 1.0 - alpha)),
        }
        for metric, values in samples.items()
    }


def paired_patient_bootstrap_difference(
    targets: np.ndarray,
    predictions_a: np.ndarray,
    predictions_b: np.ndarray,
    patient_ids: np.ndarray,
    metric: str = "icbhi_score",
    iterations: int = 5_000,
    seed: int = 20_260_727,
    confidence: float = 0.95,
    class_names: tuple[str, ...] | None = None,
) -> dict[str, float]:
    """Paired patient bootstrap for the metric difference A minus B."""
    if not (len(targets) == len(predictions_a) == len(predictions_b) == len(patient_ids)):
        raise ValueError("All paired arrays must have equal length")
    metric_function: Callable[[np.ndarray, np.ndarray], dict[str, Any]]
    if class_names is None:
        metric_function = icbhi_metrics
    else:
        def metric_function(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
            return respiratory_metrics(y, p, class_names)

    unique_patients = np.unique(patient_ids)
    patient_indices = {
        patient: np.flatnonzero(patient_ids == patient) for patient in unique_patients
    }
    random_generator = np.random.default_rng(seed)
    differences: list[float] = []
    for _ in range(iterations):
        sampled_patients = random_generator.choice(
            unique_patients,
            size=len(unique_patients),
            replace=True,
        )
        indices = np.concatenate([patient_indices[patient] for patient in sampled_patients])
        metrics_a = metric_function(targets[indices], predictions_a[indices])
        metrics_b = metric_function(targets[indices], predictions_b[indices])
        differences.append(float(metrics_a[metric]) - float(metrics_b[metric]))
    alpha = (1.0 - confidence) / 2.0
    point_difference = float(
        metric_function(targets, predictions_a)[metric]
        - metric_function(targets, predictions_b)[metric]
    )
    return {
        "estimate": point_difference,
        "lower": float(np.quantile(differences, alpha)),
        "upper": float(np.quantile(differences, 1.0 - alpha)),
        "probability_a_greater_than_b": float(np.mean(np.asarray(differences) > 0)),
    }


def paired_patient_bootstrap_mean_seed_difference(
    targets: np.ndarray,
    predictions_a: np.ndarray,
    predictions_b: np.ndarray,
    patient_ids: np.ndarray,
    metric: str = "icbhi_score",
    iterations: int = 5_000,
    seed: int = 20_260_727,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Bootstrap patients once, then average paired differences across fixed seeds."""
    if predictions_a.ndim != 2 or predictions_b.ndim != 2:
        raise ValueError("Prediction arrays must have shape [seeds, samples]")
    if predictions_a.shape != predictions_b.shape:
        raise ValueError("Paired prediction arrays must have identical shapes")
    if predictions_a.shape[1] != len(targets) or len(targets) != len(patient_ids):
        raise ValueError("Every seed must contain one prediction per target and patient")

    unique_patients = np.unique(patient_ids)
    patient_indices = {
        patient: np.flatnonzero(patient_ids == patient) for patient in unique_patients
    }
    random_generator = np.random.default_rng(seed)

    def mean_seed_difference(indices: np.ndarray) -> float:
        seed_differences = [
            float(
                icbhi_metrics(targets[indices], seed_a[indices])[metric]
                - icbhi_metrics(targets[indices], seed_b[indices])[metric]
            )
            for seed_a, seed_b in zip(predictions_a, predictions_b, strict=True)
        ]
        return float(np.mean(seed_differences))

    differences: list[float] = []
    for _ in range(iterations):
        sampled_patients = random_generator.choice(
            unique_patients,
            size=len(unique_patients),
            replace=True,
        )
        indices = np.concatenate([patient_indices[patient] for patient in sampled_patients])
        differences.append(mean_seed_difference(indices))

    alpha = (1.0 - confidence) / 2.0
    point_difference = mean_seed_difference(np.arange(len(targets)))
    return {
        "estimate": point_difference,
        "lower": float(np.quantile(differences, alpha)),
        "upper": float(np.quantile(differences, 1.0 - alpha)),
        "probability_a_greater_than_b": float(np.mean(np.asarray(differences) > 0)),
    }
