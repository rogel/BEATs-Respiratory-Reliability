#!/usr/bin/env python3
"""Class-conditional audit of the frozen, disabled entropy-rejection rule."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from respiratory_sound.calibration import TemperatureScaler
from respiratory_sound.gate11a import ALLOWED_SEEDS, sha256_file
from respiratory_sound.selective_prediction import normalized_binary_entropy

DOMAINS = ("icbhi2017", "sprsound2022")
ROLES = ("calibration", "validation_select")
CLASS_LABELS = {0: "Normal", 1: "Adventitious"}


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _load_exact_ensemble(
    root: Path,
    manifest: pd.DataFrame,
    domain: str,
    role: str,
) -> tuple[pd.DataFrame, dict[str, str]]:
    if role == "calibration":
        path = (
            root
            / "artifacts/post_gate11a/calibration_predictions"
            / f"fullft_ensemble_{domain}_calibration.csv"
        )
        return pd.read_csv(path, dtype={"patient_id": str}), {
            str(path.relative_to(root)): sha256_file(path)
        }
    tables = []
    hashes = {}
    for seed in ALLOWED_SEEDS:
        path = (
            root
            / f"runs/gate11a_exactmask_fullft_seed{seed}"
            / f"best_validation_predictions_{domain}.csv"
        )
        table = pd.read_csv(path)
        tables.append(
            table[["sample_id", "target", "probability_1"]].rename(
                columns={"probability_1": f"probability_1_seed{seed}"}
            )
        )
        hashes[str(path.relative_to(root))] = sha256_file(path)
    metadata = manifest.loc[
        manifest["dataset"].astype(str).eq(domain)
        & manifest["protocol_role"].astype(str).eq(role),
        [
            "sample_id",
            "dataset",
            "patient_id",
            "protocol_role",
            "locked",
            "binary_label_id",
            "fine_label_name",
            "event_duration_seconds",
        ],
    ].copy()
    merged = metadata
    for table in tables:
        merged = merged.merge(table, on="sample_id", how="inner", validate="one_to_one")
    probability_columns = [f"probability_1_seed{seed}" for seed in ALLOWED_SEEDS]
    merged["probability_1"] = merged[probability_columns].mean(axis=1)
    merged["target"] = merged["binary_label_id"].astype(int)
    if len(merged) != len(metadata):
        raise ValueError(f"Validation ensemble coverage mismatch: {domain}")
    return merged, hashes


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "sample_sd": float(np.std(values, ddof=1)),
        "median": float(np.median(values)),
        "q05": float(np.quantile(values, 0.05)),
        "q10": float(np.quantile(values, 0.10)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "q90": float(np.quantile(values, 0.90)),
        "q95": float(np.quantile(values, 0.95)),
        "iqr": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
    }


def _operating_point(
    targets: np.ndarray,
    probabilities: np.ndarray,
    entropy: np.ndarray,
    cutoff: float,
) -> dict[str, Any]:
    accepted = entropy <= cutoff
    predictions = (probabilities >= 0.5).astype(int)
    errors = predictions != targets
    result: dict[str, Any] = {
        "cutoff": cutoff,
        "total": int(len(targets)),
        "accepted": int(accepted.sum()),
        "rejected": int((~accepted).sum()),
        "overall_coverage": float(accepted.mean()),
        "accepted_set_error": float(errors[accepted].mean()),
        "classes": {},
    }
    for label, name in CLASS_LABELS.items():
        selected = targets == label
        class_accepted = selected & accepted
        class_rejected = selected & ~accepted
        result["classes"][name] = {
            "total": int(selected.sum()),
            "accepted": int(class_accepted.sum()),
            "rejected": int(class_rejected.sum()),
            "coverage": float(class_accepted.sum() / selected.sum()),
            "accepted_set_error": (
                float(errors[class_accepted].mean())
                if class_accepted.any()
                else None
            ),
            "accepted_composition_fraction": float(
                class_accepted.sum() / max(int(accepted.sum()), 1)
            ),
            "rejected_composition_fraction": float(
                class_rejected.sum() / max(int((~accepted).sum()), 1)
            ),
        }
    return result


def _risk_coverage_rows(
    targets: np.ndarray,
    probabilities: np.ndarray,
    entropy: np.ndarray,
    *,
    domain: str,
    role: str,
    class_label: int,
) -> list[dict[str, Any]]:
    selected = np.flatnonzero(targets == class_label)
    ordered = selected[np.argsort(entropy[selected], kind="stable")]
    errors = ((probabilities >= 0.5).astype(int) != targets)[ordered]
    cumulative = np.cumsum(errors)
    return [
        {
            "dataset": domain,
            "role": role,
            "class": CLASS_LABELS[class_label],
            "accepted": rank,
            "total_class": len(ordered),
            "coverage": rank / len(ordered),
            "error_risk": float(cumulative[rank - 1] / rank),
            "entropy_cutoff": float(entropy[ordered[rank - 1]]),
        }
        for rank in range(1, len(ordered) + 1)
    ]


def main() -> None:
    root = Path.cwd().resolve()
    output_root = root / "artifacts/revision_2026_09_03/entropy_analysis"
    output_root.mkdir(parents=True, exist_ok=False)
    manifest_path = root / "data/manifests/cross_domain_binary.csv"
    manifest = pd.read_csv(manifest_path, dtype={"patient_id": str})
    calibration_path = (
        root
        / "artifacts/post_gate11a/calibration_analysis/post_gate11a_calibration_final.json"
    )
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    outputs: dict[str, Any] = {}
    source_hashes = {
        str(manifest_path.relative_to(root)): sha256_file(manifest_path),
        str(calibration_path.relative_to(root)): sha256_file(calibration_path),
    }
    curve_rows = []
    for domain in DOMAINS:
        temperature = float(calibration["domains"][domain]["temperature"])
        cutoffs = {
            key: float(value["uncertainty_cutoff"])
            for key, value in calibration["domains"][domain]["selective"].items()
        }
        for role in ROLES:
            frame, hashes = _load_exact_ensemble(root, manifest, domain, role)
            source_hashes.update(hashes)
            targets = frame["target"].to_numpy(dtype=int)
            raw = frame["probability_1"].to_numpy(dtype=float)
            probabilities = TemperatureScaler(temperature).transform(raw)
            entropy = normalized_binary_entropy(probabilities)
            if not np.isfinite(entropy).all():
                raise FloatingPointError("Entropy became non-finite")
            class_summaries = {
                name: _summary(entropy[targets == label])
                for label, name in CLASS_LABELS.items()
            }
            operating_points = {
                target: _operating_point(
                    targets,
                    probabilities,
                    entropy,
                    cutoff,
                )
                for target, cutoff in cutoffs.items()
            }
            outputs[f"{domain}/{role}"] = {
                "samples": len(frame),
                "patients": int(frame["patient_id"].nunique()),
                "temperature": temperature,
                "entropy_by_class": class_summaries,
                "frozen_operating_points_descriptive_only": operating_points,
            }
            for label in CLASS_LABELS:
                curve_rows.extend(
                    _risk_coverage_rows(
                        targets,
                        probabilities,
                        entropy,
                        domain=domain,
                        role=role,
                        class_label=label,
                    )
                )
    pd.DataFrame(curve_rows).to_csv(
        output_root / "class_conditional_risk_coverage.csv", index=False
    )
    payload = {
        "schema_version": 1,
        "stage": "BEATS_class_conditional_entropy",
        "primary_model": "protected exact-mask Full-FT three-seed ensemble",
        "uncertainty": "normalized binary predictive entropy after frozen temperature scaling",
        "classification_threshold": 0.5,
        "selective_prediction_enabled": False,
        "failure_action": "fail closed; no abstention is applied to held-out predictions",
        "results": outputs,
        "source_hashes": source_hashes,
        "finite_value_audit_passed": True,
        "calibration_accessed": True,
        "locked_tests_accessed": False,
    }
    _atomic_json(payload, output_root / "class_conditional_entropy_final.json")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
