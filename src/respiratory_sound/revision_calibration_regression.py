"""Audited, unpenalized joint calibration regression for revision diagnostics.

This module does not fit or apply the study's temperature calibrators. Historical
analysis modules remain unchanged for provenance; new revision analyses use this
implementation to estimate logit Pr(Y=1) = intercept + slope * logit(p).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from respiratory_sound.calibration import probabilities_to_logits


class CalibrationRegressionError(RuntimeError):
    """An explicitly classified non-estimability or numerical failure."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


def fit_calibration_regression(
    targets: np.ndarray,
    probabilities: np.ndarray,
    *,
    independent_check: bool = False,
) -> dict:
    """Return a finite MLE only after identifiability and convergence checks.

    Centering/scaling the single predictor is an exact reparameterization, not
    regularization. Complete/quasi separation is checked from ordered class
    supports. No weights, penalty, outcome exclusion, or replacement is applied.
    """
    labels = np.asarray(targets, dtype=float)
    values = np.asarray(probabilities, dtype=float)
    if (labels.ndim != 1 or values.shape != labels.shape or len(labels) < 3
            or not np.isfinite(labels).all() or not np.isfinite(values).all()
            or not np.isin(labels, (0.0, 1.0)).all()
            or np.any((values < 0) | (values > 1))):
        raise ValueError("Invalid binary labels or probability array")
    if len(np.unique(labels)) != 2:
        raise CalibrationRegressionError("single_class")
    x = probabilities_to_logits(values)
    location, scale = float(x.mean()), float(x.std())
    if scale <= 1e-12:
        raise CalibrationRegressionError("rank_deficient_predictor")
    negative, positive = x[labels == 0], x[labels == 1]
    gaps = (float(positive.min() - negative.max()),
            float(negative.min() - positive.max()))
    if max(gaps) >= 0:
        raise CalibrationRegressionError(
            "complete_separation" if max(gaps) > 0 else "quasi_complete_separation"
        )
    design = np.column_stack((np.ones(len(x)), (x - location) / scale))

    def objective(beta):
        z = design @ beta
        return float(np.mean(np.logaddexp(0.0, z) - labels * z))

    def gradient(beta):
        return design.T @ (expit(design @ beta) - labels) / len(labels)

    def hessian(beta):
        fitted = expit(design @ beta)
        return (design.T * (fitted * (1 - fitted))) @ design / len(labels)

    def checked_fit(method):
        result = minimize(
            objective, np.zeros(2), method=method, jac=gradient,
            **({"hess": hessian} if method == "trust-exact" else {}),
            options={"gtol": 1e-10, "maxiter": 500},
        )
        if not np.isfinite(result.x).all():
            return None
        grad = gradient(result.x)
        information = hessian(result.x)
        eigenvalues = np.linalg.eigvalsh(information)
        if not np.isfinite(information).all() or eigenvalues.min() <= 1e-12:
            return None
        decrement = float(grad @ np.linalg.solve(information, grad))
        if (float(np.max(np.abs(grad))) > 1e-8 or decrement > 1e-12
                or not np.isfinite(objective(result.x))
                or objective(result.x) > objective(np.zeros(2)) + 1e-12):
            return None
        slope = float(result.x[1] / scale)
        intercept = float(result.x[0] - slope * location)
        return {
            "calibration_intercept": intercept, "calibration_slope": slope,
            "converged_by_verified_criteria": True,
            "optimizer_success": bool(result.success),
            "optimizer_message": str(result.message), "method": method,
            "iterations": int(result.nit),
            "standardized_gradient_infinity_norm": float(np.max(np.abs(grad))),
            "newton_decrement_squared": decrement,
            "standardized_hessian_min_eigenvalue": float(eigenvalues.min()),
        }

    primary = checked_fit("trust-exact")
    alternate = checked_fit("BFGS") if independent_check or primary is None else None
    if primary is None:
        if alternate is None:
            raise CalibrationRegressionError("numerical_nonconvergence")
        primary = alternate
    if independent_check:
        if alternate is None:
            raise CalibrationRegressionError("independent_solver_check_failed")
        for key in ("calibration_intercept", "calibration_slope"):
            if not np.isclose(primary[key], alternate[key], rtol=1e-6, atol=1e-6):
                raise CalibrationRegressionError("independent_solver_disagreement")
        primary["independent_BFGS_check_passed"] = True
    return primary
