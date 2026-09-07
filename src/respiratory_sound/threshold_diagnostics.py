"""Patient-balanced threshold selection and patient-level cross-fitting."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def _binary_arrays(
    targets: np.ndarray,
    probabilities: np.ndarray,
    patient_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray(targets, dtype=np.int64)
    values = np.asarray(probabilities, dtype=np.float64)
    patients = np.asarray(patient_ids).astype(str)
    if not (labels.ndim == values.ndim == patients.ndim == 1):
        raise ValueError("targets, probabilities, and patient_ids must be 1D")
    if not (len(labels) == len(values) == len(patients)):
        raise ValueError("targets, probabilities, and patient_ids must align")
    if not np.isin(labels, (0, 1)).all() or np.unique(labels).size != 2:
        raise ValueError("targets must contain both binary classes")
    if not np.isfinite(values).all():
        raise ValueError("probabilities must be finite")
    return labels, values, patients


def patient_class_balanced_accuracy(
    targets: np.ndarray,
    predictions: np.ndarray,
    patient_ids: np.ndarray,
) -> float:
    """Average recall equally over patients within class, then over classes."""
    labels = np.asarray(targets, dtype=np.int64)
    predicted = np.asarray(predictions, dtype=np.int64)
    patients = np.asarray(patient_ids).astype(str)
    if not (len(labels) == len(predicted) == len(patients)):
        raise ValueError("targets, predictions, and patient_ids must align")
    if not np.isin(labels, (0, 1)).all() or np.unique(labels).size != 2:
        raise ValueError("targets must contain both binary classes")
    class_recalls: list[float] = []
    for class_id in (0, 1):
        patient_recalls = []
        for patient_id in np.unique(patients[labels == class_id]):
            mask = (patients == patient_id) & (labels == class_id)
            patient_recalls.append(float(np.mean(predicted[mask] == class_id)))
        if not patient_recalls:
            raise ValueError(f"class {class_id} has no patient observations")
        class_recalls.append(float(np.mean(patient_recalls)))
    return float(np.mean(class_recalls))


def _threshold_candidates(probabilities: np.ndarray) -> np.ndarray:
    unique = np.unique(np.asarray(probabilities, dtype=np.float64))
    if unique.size == 1:
        return np.asarray(
            (
                np.nextafter(unique[0], -np.inf),
                np.nextafter(unique[0], np.inf),
            )
        )
    midpoints = unique[:-1] + (unique[1:] - unique[:-1]) / 2.0
    return np.concatenate(
        (
            np.asarray((np.nextafter(unique[0], -np.inf),)),
            midpoints,
            np.asarray((np.nextafter(unique[-1], np.inf),)),
        )
    )


def select_patient_balanced_threshold(
    targets: np.ndarray,
    probabilities: np.ndarray,
    patient_ids: np.ndarray,
    domains: np.ndarray | None = None,
) -> tuple[float, float]:
    """Select a threshold with patient/class balance and optional domain balance."""
    labels, values, patients = _binary_arrays(
        targets,
        probabilities,
        patient_ids,
    )
    domain_values = None if domains is None else np.asarray(domains).astype(str)
    if domain_values is not None and len(domain_values) != len(labels):
        raise ValueError("domains must align with targets")

    def objective(predictions: np.ndarray) -> float:
        if domain_values is None:
            return patient_class_balanced_accuracy(labels, predictions, patients)
        domain_scores = [
            patient_class_balanced_accuracy(
                labels[domain_values == domain],
                predictions[domain_values == domain],
                patients[domain_values == domain],
            )
            for domain in np.unique(domain_values)
        ]
        return float(np.mean(domain_scores))

    candidates = _threshold_candidates(values)
    scores = np.asarray(
        [objective((values >= threshold).astype(np.int64)) for threshold in candidates]
    )
    best_score = float(scores.max())
    best = candidates[np.isclose(scores, best_score, atol=1.0e-12)]
    distances = np.abs(best - 0.5)
    closest = best[np.isclose(distances, distances.min(), atol=1.0e-12)]
    return float(closest.min()), best_score


def leave_one_patient_out_thresholds(
    frame: pd.DataFrame,
    strategy: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Create OOF predictions with fixed, global, or domain-conditioned thresholds."""
    required = {"target", "probability_1", "patient_id", "dataset"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"frame is missing columns: {sorted(missing)}")
    if strategy not in {"fixed", "global", "domain_conditioned"}:
        raise ValueError(f"unsupported threshold strategy: {strategy}")
    labels = frame["target"].to_numpy(dtype=np.int64)
    probabilities = frame["probability_1"].to_numpy(dtype=np.float64)
    patient_ids = frame["patient_id"].astype(str).to_numpy()
    domains = frame["dataset"].astype(str).to_numpy()
    predictions = np.empty(len(frame), dtype=np.int64)
    sample_thresholds = np.empty(len(frame), dtype=np.float64)
    patient_thresholds: dict[str, float] = {}

    for patient_id in np.unique(patient_ids):
        held_out = patient_ids == patient_id
        if strategy == "fixed":
            threshold = 0.5
        else:
            fitting = ~held_out
            fitting_domains = None
            if strategy == "domain_conditioned":
                fitting &= domains == domains[held_out][0]
            else:
                fitting_domains = domains[fitting]
            threshold, _ = select_patient_balanced_threshold(
                labels[fitting],
                probabilities[fitting],
                patient_ids[fitting],
                domains=fitting_domains,
            )
        sample_thresholds[held_out] = threshold
        predictions[held_out] = (probabilities[held_out] >= threshold).astype(
            np.int64
        )
        patient_thresholds[patient_id] = float(threshold)
    return predictions, sample_thresholds, patient_thresholds


def threshold_summary(
    patient_thresholds: dict[str, float],
    patient_filter: Callable[[str], bool] | None = None,
) -> dict[str, float | int]:
    """Summarize fitted thresholds across held-out patients."""
    values = np.asarray(
        [
            threshold
            for patient, threshold in patient_thresholds.items()
            if patient_filter is None or patient_filter(patient)
        ],
        dtype=np.float64,
    )
    if values.size == 0:
        raise ValueError("no thresholds selected by patient_filter")
    q25, median, q75 = np.quantile(values, (0.25, 0.5, 0.75))
    return {
        "patients": int(values.size),
        "minimum": float(values.min()),
        "q25": float(q25),
        "median": float(median),
        "q75": float(q75),
        "maximum": float(values.max()),
        "iqr": float(q75 - q25),
        "standard_deviation": float(values.std(ddof=0)),
    }
