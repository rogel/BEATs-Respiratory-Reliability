#!/usr/bin/env python3
"""Read-only verification of the supplied manifest, predictions and results."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_SHA256 = "2dd168129fc0fc159db201f2990c4d7074cb746fc9bc2c13bd5031e47dbd1419"
EXPECTED_ROLE_COUNTS = {
    ("icbhi2017", "train_fit"): 2910,
    ("icbhi2017", "validation_select"): 554,
    ("icbhi2017", "calibration"): 561,
    ("icbhi2017", "locked_test"): 2873,
    ("sprsound2022", "train_fit"): 4605,
    ("sprsound2022", "validation_select"): 1013,
    ("sprsound2022", "calibration"): 1038,
    ("sprsound2022", "locked_inter_test"): 1429,
    ("sprsound2022", "locked_intra_test"): 1004,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def assert_close(found: float, expected: float, label: str, tol: float = 1e-12) -> None:
    if not math.isclose(found, expected, rel_tol=0.0, abs_tol=tol):
        raise AssertionError(f"{label}: found {found}, expected {expected}")


def classification_metrics(path: Path) -> dict[str, float]:
    rows = read_csv(path)
    target = [int(row["target"]) for row in rows]
    prediction = [int(row["prediction"]) for row in rows]
    if len(target) != len(prediction):
        raise AssertionError(f"Target/prediction length mismatch: {path}")
    pairs = list(zip(target, prediction))
    tn = sum(y == 0 and p == 0 for y, p in pairs)
    fp = sum(y == 0 and p == 1 for y, p in pairs)
    fn = sum(y == 1 and p == 0 for y, p in pairs)
    tp = sum(y == 1 and p == 1 for y, p in pairs)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return {
        "samples": float(len(rows)),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "average_score": (sensitivity + specificity) / 2.0,
    }


def verify_manifest() -> None:
    path = ROOT / "data/manifests/cross_domain_binary.csv"
    if sha256_file(path) != MANIFEST_SHA256:
        raise AssertionError("Unified manifest SHA-256 mismatch")
    rows = read_csv(path)
    if len(rows) != 15_987:
        raise AssertionError(f"Unexpected manifest row count: {len(rows)}")
    patients = {row["patient_id"] for row in rows}
    if len(patients) != 414:
        raise AssertionError(f"Unexpected database-prefixed patient count: {len(patients)}")
    counts = Counter((row["dataset"], row["protocol_role"]) for row in rows)
    if counts != Counter(EXPECTED_ROLE_COUNTS):
        raise AssertionError(f"Unexpected role counts: {dict(counts)}")

    patient_roles: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        patient_roles[(row["dataset"], row["protocol_role"])].add(row["patient_id"])
    primary_roles = {
        "icbhi2017": ("train_fit", "validation_select", "calibration", "locked_test"),
        "sprsound2022": (
            "train_fit",
            "validation_select",
            "calibration",
            "locked_inter_test",
        ),
    }
    for dataset, roles in primary_roles.items():
        for index, left in enumerate(roles):
            for right in roles[index + 1 :]:
                overlap = patient_roles[(dataset, left)] & patient_roles[(dataset, right)]
                if overlap:
                    raise AssertionError(f"Unexpected patient overlap: {dataset} {left}/{right}")


def verify_prediction_manifest(directory: Path) -> None:
    registry = read_json(directory / "prediction_manifest.json")
    for filename, expected_hash in registry["files"].items():
        path = directory / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing registered prediction file: {path}")
        if sha256_file(path) != expected_hash:
            raise AssertionError(f"Prediction SHA-256 mismatch: {path}")


def verify_primary_results() -> dict[str, float]:
    prediction_root = ROOT / "artifacts/post_gate11a/locked_predictions"
    locked = read_json(ROOT / "artifacts/post_gate11a/locked_final.json")
    roles = {
        "icbhi2017": "locked_test",
        "sprsound2022": "locked_inter_test",
    }
    results: dict[str, float] = {}
    domain_scores: dict[str, dict[str, float]] = defaultdict(dict)
    for family in ("fullft", "lora"):
        for domain, role in roles.items():
            path = prediction_root / f"{family}_ensemble_{domain}_{role}.csv"
            metrics = classification_metrics(path)
            recorded = locked["families"][family][domain]["ensemble_metrics"]
            for name in ("sensitivity", "specificity", "average_score"):
                assert_close(metrics[name], float(recorded[name]), f"{family} {domain} {name}")
            domain_scores[family][domain] = metrics["average_score"]

    fullft_equal = sum(domain_scores["fullft"].values()) / 2.0
    lora_equal = sum(domain_scores["lora"].values()) / 2.0
    paired = locked["paired_fullft_minus_lora_bootstrap"]["two_domain_mean"]
    difference = fullft_equal - lora_equal
    assert_close(difference, float(paired["estimate"]), "equal-database full-FT minus LoRA")
    results.update(
        {
            "icbhi_fullft_as": domain_scores["fullft"]["icbhi2017"],
            "sprsound_fullft_as": domain_scores["fullft"]["sprsound2022"],
            "equal_database_fullft_as": fullft_equal,
            "equal_database_difference": difference,
            "difference_lower": float(paired["lower"]),
            "difference_upper": float(paired["upper"]),
        }
    )
    return results


def verify_integrity_calibration_and_resources() -> dict[str, float]:
    amendment = read_json(ROOT / "artifacts/gate9b_exact_token_mask_amendment.json")
    replay = amendment["failure_batch_replay"]
    if replay["old_valid_token_counts_for_two_views"] != [7, 0]:
        raise AssertionError("Legacy-mask replay count mismatch")
    if replay["corrected_valid_token_counts_for_two_views"] != [16, 16]:
        raise AssertionError("Exact-mask replay count mismatch")
    required_finite = (
        "all_12_encoder_layers_finite",
        "binary_logits_finite",
        "all_99842_trainable_gradients_finite",
    )
    if not all(bool(replay[name]) for name in required_finite):
        raise AssertionError("Exact-mask finite-value audit failed")

    locked = read_json(ROOT / "artifacts/post_gate11a/locked_final.json")
    for domain in ("icbhi2017", "sprsound2022"):
        reliability = locked["primary_fullft_reliability"][domain]
        raw = reliability["raw_probability_metrics"]
        scaled = reliability["temperature_probability_metrics"]
        if not float(scaled["negative_log_likelihood"]) < float(raw["negative_log_likelihood"]):
            raise AssertionError(f"NLL did not improve for {domain}")
        if not float(scaled["brier_score"]) < float(raw["brier_score"]):
            raise AssertionError(f"Brier score did not improve for {domain}")

    calibration = read_json(
        ROOT / "artifacts/post_gate11a/calibration_analysis/post_gate11a_calibration_final.json"
    )
    coverage = calibration["domains"]["sprsound2022"]["selective"]["coverage_80"]
    validation = coverage["validation_select"]
    assert_close(float(validation["coverage"]), 0.7413622902270484, "SPRSound coverage")
    assert_close(
        float(validation["adventitious_coverage"]),
        0.6035242290748899,
        "SPRSound Adventitious coverage",
    )
    if bool(coverage["passed"]):
        raise AssertionError("The prespecified SPRSound coverage audit should be disabled")

    efficiency = read_json(ROOT / "artifacts/post_gate11a/efficiency_final.json")
    fullft = efficiency["families"]["fullft"]["single"]
    lora = efficiency["families"]["lora"]["single"]
    if int(fullft["training_parameters"]) != 90_313_330:
        raise AssertionError("Full-FT trainable-parameter count mismatch")
    if int(lora["training_parameters"]) != 99_842:
        raise AssertionError("LoRA trainable-parameter count mismatch")
    return {
        "overall_coverage": float(validation["coverage"]),
        "adventitious_coverage": float(validation["adventitious_coverage"]),
        "parameter_ratio": int(fullft["training_parameters"])
        / int(lora["training_parameters"]),
        "storage_ratio": int(fullft["checkpoint_bytes"]) / int(lora["checkpoint_bytes"]),
    }


def verify_exclusions() -> None:
    forbidden_suffixes = {".wav", ".flac", ".mp3", ".pt", ".pth", ".safetensors"}
    violations = [
        path.relative_to(ROOT)
        for path in ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in forbidden_suffixes
    ]
    if violations:
        raise AssertionError(f"Excluded audio/model binaries found: {violations}")


def main() -> None:
    verify_exclusions()
    verify_manifest()
    verify_prediction_manifest(ROOT / "artifacts/post_gate11a/calibration_predictions")
    verify_prediction_manifest(ROOT / "artifacts/post_gate11a/locked_predictions")
    primary = verify_primary_results()
    audits = verify_integrity_calibration_and_resources()
    print("Reference result verification PASSED")
    print("Manifest: 15,987 events; 414 database-prefixed patient identifiers")
    print(
        "Full-FT AS: "
        f"ICBHI={primary['icbhi_fullft_as']:.4f}; "
        f"SPRSound={primary['sprsound_fullft_as']:.4f}; "
        f"equal-database={primary['equal_database_fullft_as']:.4f}"
    )
    print(
        "Full-FT minus LoRA equal-database AS: "
        f"{primary['equal_database_difference']:.4f} "
        f"(stored 95% interval {primary['difference_lower']:.4f}-"
        f"{primary['difference_upper']:.4f})"
    )
    print(
        "SPRSound validation coverage: "
        f"overall={audits['overall_coverage']:.2%}; "
        f"Adventitious={audits['adventitious_coverage']:.2%}"
    )
    print(
        "Resource ratios (full-FT/LoRA): "
        f"trainable parameters={audits['parameter_ratio']:.1f}x; "
        f"task-specific storage={audits['storage_ratio']:.1f}x"
    )


if __name__ == "__main__":
    main()
