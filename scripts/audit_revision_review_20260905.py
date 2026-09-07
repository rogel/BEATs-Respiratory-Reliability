#!/usr/bin/env python3
"""Read-only source audit and solver diagnosis; never replace formal results.

The only writes are a new, dated diagnostic JSON and its checksum.  Logistic
regressions describe calibration of frozen predictions, not a new calibrator.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize
from scipy.special import expit

from analyze_revision_ablation import (
    PRIMARY, ROLES, _as, _calibration_regression, _exact_path, _load_seed,
    _revision_path,
)
from analyze_revision_analysis_a import (
    FAMILIES, MANIFEST_SHA256, MATCHED_SEEDS, _matched_completer_ensemble,
    _patient_overlap_audit, _verify_freeze, _verify_source_manifests,
)
from respiratory_sound.calibration import (
    TemperatureScaler, probabilities_to_logits, probability_metrics,
)
from respiratory_sound.gate11a import sha256_file


def diagnose(y: np.ndarray, p: np.ndarray) -> dict:
    x = probabilities_to_logits(p)
    design = np.column_stack((np.ones(len(x)), x))

    def objective(beta):
        z = design @ beta
        return float(np.mean(np.logaddexp(0.0, z) - y * z))

    def gradient(beta):
        return design.T @ (expit(design @ beta) - y) / len(y)

    def hessian(beta):
        fitted = expit(design @ beta)
        return (design.T * (fitted * (1 - fitted))) @ design / len(y)

    records = []
    for method in ("BFGS", "trust-exact"):
        for start in ([0.0, 0.0], [0.0, 1.0], [-3.0, 1.0]):
            kwargs = {"hess": hessian} if method == "trust-exact" else {}
            result = minimize(
                objective, start, jac=gradient, method=method,
                options={"gtol": 1e-10, "maxiter": 1000}, **kwargs,
            )
            grad = float(np.max(np.abs(gradient(result.x))))
            eigenvalues = np.linalg.eigvalsh(hessian(result.x))
            assert np.isfinite(result.x).all()
            assert grad < 1e-8, (method, start, grad)
            assert np.min(eigenvalues) > 0
            records.append({
                "method": method, "start": start,
                "optimizer_success": bool(result.success),
                "optimizer_message": str(result.message),
                "intercept": float(result.x[0]), "slope": float(result.x[1]),
                "gradient_infinity_norm": grad,
                "hessian_eigenvalues": eigenvalues.tolist(),
            })
    coefficients = np.array([[r["intercept"], r["slope"]] for r in records])
    spread = float(np.max(np.ptp(coefficients, axis=0)))
    assert spread < 1e-6
    assert any(r["optimizer_success"] for r in records)
    best = min(records, key=lambda r: r["gradient_infinity_norm"])
    return {
        "model": "unpenalized joint logistic regression: logit Pr(Y=1) = a + b logit(p)",
        "interpretation": "diagnostic only; never apply fitted probabilities to evaluation",
        "stored_solver_result": _calibration_regression(y, p),
        "independently_verified_intercept": best["intercept"],
        "independently_verified_slope": best["slope"],
        "maximum_coefficient_spread": spread,
        "class_logit_ranges": {
            str(label): [float(x[y == label].min()), float(x[y == label].max())]
            for label in (0, 1)
        },
        "fits": records,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    _verify_freeze(root)
    _verify_source_manifests(root)
    manifest_path = root / "data/manifests/cross_domain_binary.csv"
    assert sha256_file(manifest_path) == MANIFEST_SHA256
    manifest = pd.read_csv(manifest_path, dtype={"patient_id": str})
    final_path = root / "artifacts/revision_2026_09_03/analysis_a_final/analysis_a_final.json"
    previous = json.loads(final_path.read_text())
    source_hashes, audit_rows, solver = {}, [], {}
    as_points = {}
    for family in FAMILIES:
        for domain, roles in ROLES.items():
            for role in roles:
                frames = []
                for seed in MATCHED_SEEDS:
                    frame, digest = _load_seed(
                        root, manifest, family=family, seed=seed, domain=domain, role=role,
                    )
                    path = (_exact_path(root, seed, domain, role) if family == "exact"
                            else _revision_path(root, family, seed, domain, role))
                    original = pd.read_csv(path, dtype={"patient_id": str}).set_index("sample_id")
                    aligned = original.loc[frame["sample_id"]].reset_index()
                    for column in ("dataset", "patient_id", "protocol_role", "locked",
                                   "binary_label_id", "fine_label_name"):
                        assert aligned[column].astype(str).equals(frame[column].astype(str)), (path, column)
                    np.testing.assert_allclose(aligned["event_duration_seconds"], frame["event_duration_seconds"], rtol=1e-12)
                    assert np.array_equal(aligned["prediction"], frame["prediction"])
                    np.testing.assert_allclose(aligned["probability_0"] + aligned["probability_1"], 1, atol=2e-7)
                    source_hashes[str(path.relative_to(root))] = digest
                    audit_rows.append({"family": family, "seed": seed, "dataset": domain,
                                       "role": role, "events": len(frame), "metadata_and_values_passed": True})
                    frames.append(frame)
                ensemble = _matched_completer_ensemble(frames)
                y = ensemble["target"].to_numpy(dtype=int)
                p = ensemble["probability_1"].to_numpy(dtype=float)
                if role == PRIMARY[domain]:
                    point = _as(y, p)
                    expected = previous["exact_minus_legacy"][domain][
                        "exact_average_score" if family == "exact" else "legacy_average_score"]
                    assert abs(point - expected) < 1e-14
                    as_points[f"{family}/{domain}"] = point
                if domain == "sprsound2022" and role == "locked_inter_test":
                    raw = diagnose(y, p)
                    temperature = previous["temperatures"][family][domain]
                    scaled_p = TemperatureScaler(temperature).transform(p)
                    scaled = diagnose(y, scaled_p)
                    np.testing.assert_allclose(raw["independently_verified_intercept"],
                                               scaled["independently_verified_intercept"], atol=1e-6)
                    np.testing.assert_allclose(raw["independently_verified_slope"] * temperature,
                                               scaled["independently_verified_slope"], atol=1e-6)
                    solver[family] = {"raw": raw, "temperature_scaled": scaled,
                                      "rescaling_invariance_passed": True,
                                      "raw_probability_metrics": probability_metrics(y, p, bins=15)}
    payload = {
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "scope": "review-only diagnosis; no training, model selection, new calibrator, or formal output replacement",
        "analysis_readiness": "Analysis A calibration regression requires audited implementation correction; AS and probability predictions unaffected",
        "source_manifest_sha256": MANIFEST_SHA256,
        "formal_analysis_a_sha256": sha256_file(final_path),
        "audit_script_sha256": sha256_file(Path(__file__).resolve()),
        "scipy_version": scipy.__version__,
        "patient_overlap_audit": _patient_overlap_audit(manifest),
        "prediction_audits": audit_rows, "prediction_sha256": source_hashes,
        "reproduced_classification_points": as_points,
        "calibration_regression_diagnosis": solver,
        "confidence_intervals_recomputed_in_this_diagnostic": False,
        "warning": "Do not interpret solver failure as separation or use finite-value checks as a convergence certificate.",
    }
    output = root / "artifacts/revision_2026_09_03/review_2026_09_05"
    output.mkdir(exist_ok=False)
    path = output / "independent_review_diagnostic.json"
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    digest = sha256_file(path)
    (output / "independent_review_diagnostic.sha256").write_text(f"{digest}  {path.name}\n")
    print(json.dumps({"output": str(path), "sha256": digest, "prediction_files_verified": len(audit_rows),
                      "as_points": as_points,
                      "raw_regression": {k: [v["raw"]["independently_verified_intercept"],
                                               v["raw"]["independently_verified_slope"]]
                                         for k, v in solver.items()}}, indent=2))


if __name__ == "__main__":
    main()
