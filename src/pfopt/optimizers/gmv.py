"""Global minimum variance (GMV) portfolios.

Long-only GMV solves the convex quadratic program

    minimize   w' Sigma w
    subject to sum(w) = 1,  w >= 0,  (optional)  w <= max_weight,

with cvxpy (CLARABEL, tightened tolerances) and a logged SLSQP fallback. The unconstrained GMV has
the closed form ``Sigma^{-1} 1 / (1' Sigma^{-1} 1)`` (Markowitz, 1952; Merton, 1972), computed with
a linear solve and never an explicit inverse.
"""

from __future__ import annotations

import logging

import cvxpy as cp
import numpy as np
from scipy.optimize import minimize

from pfopt.weights import check_weights

logger = logging.getLogger(__name__)

_CAP_FEASIBILITY_SLACK = 1e-12


def _prepare(sigma: np.ndarray) -> np.ndarray:
    """Validate a covariance matrix and return it symmetrized as a float array."""
    s = np.asarray(sigma, dtype=float)
    if s.ndim != 2 or s.shape[0] != s.shape[1] or s.shape[0] < 1:
        raise ValueError("Sigma must be a non-empty square matrix")
    if not np.isfinite(s).all():
        raise ValueError("Sigma must be finite")
    return (s + s.T) / 2.0


def _mean_variance(s: np.ndarray) -> float:
    scale = float(np.trace(s)) / s.shape[0]
    if scale <= 0:
        raise ValueError("Sigma must have a positive trace")
    return scale


def _solve_cvxpy(sigma_scaled: np.ndarray, cap: float | None, tol: float) -> np.ndarray:
    """Solve the long-only GMV QP with CLARABEL; raise if the solution is not optimal."""
    n = sigma_scaled.shape[0]
    w = cp.Variable(n)
    constraints = [cp.sum(w) == 1, w >= 0]
    if cap is not None:
        constraints.append(w <= cap)
    objective = cp.Minimize(cp.quad_form(w, cp.psd_wrap(sigma_scaled)))
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.CLARABEL, tol_gap_abs=tol, tol_gap_rel=tol, tol_feas=tol)
    if problem.status != cp.OPTIMAL or w.value is None:
        raise RuntimeError(f"CLARABEL returned status {problem.status!r}")
    return np.asarray(w.value, dtype=float)


def _solve_slsqp(sigma_scaled: np.ndarray, cap: float | None, tol: float) -> np.ndarray:
    """Fallback solver: SLSQP from the equal-weight starting point."""
    n = sigma_scaled.shape[0]
    upper = 1.0 if cap is None else cap
    result = minimize(
        lambda w: float(w @ sigma_scaled @ w),
        np.full(n, 1.0 / n),
        jac=lambda w: 2.0 * sigma_scaled @ w,
        method="SLSQP",
        bounds=[(0.0, upper)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(n)}],
        options={"ftol": tol, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"SLSQP failed: {result.message}")
    return np.asarray(result.x, dtype=float)


def gmv_long_only(
    sigma: np.ndarray,
    max_weight: float | None = None,
    tol: float = 1e-10,
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    """Long-only, fully invested global minimum variance portfolio.

    Parameters
    ----------
    sigma : numpy.ndarray
        ``N x N`` symmetric positive semi-definite covariance matrix. It is rescaled internally by
        its mean variance (weights are unaffected). A tiny negative eigenvalue caused by round-off
        (for example from a rank-deficient sample covariance) is removed by shifting the diagonal.
    max_weight : float, optional
        Per-asset cap in ``(0, 1]``. Must satisfy ``max_weight >= 1 / N``.
    tol : float, default 1e-10
        Solver tolerance (gap and feasibility for CLARABEL, ``ftol`` for SLSQP).
    return_info : bool, default False
        If True, also return a dict with ``solver`` ("CLARABEL", "SLSQP" or "closed-form"),
        ``fallback`` (True if SLSQP replaced a failed CLARABEL run) and ``objective``
        (``w' Sigma w`` in the original units).

    Returns
    -------
    numpy.ndarray or (numpy.ndarray, dict)
        Non-negative weights summing to 1 (and not above ``max_weight``).

    Raises
    ------
    ValueError
        For an invalid or clearly non-PSD matrix, an invalid cap, or a cap below ``1 / N`` (the
        problem is then infeasible).
    RuntimeError
        If both the primary and the fallback solver fail.
    """
    s = _prepare(sigma)
    n = s.shape[0]
    if max_weight is not None:
        if not 0.0 < max_weight <= 1.0:
            raise ValueError(f"max_weight must be in (0, 1], got {max_weight}")
        if max_weight * n < 1.0 - _CAP_FEASIBILITY_SLACK:
            raise ValueError(
                f"max_weight={max_weight} is infeasible for N={n} assets: need max_weight >= 1/N"
                f" = {1.0 / n:.6f}"
            )
    scale = _mean_variance(s)
    scaled = s / scale
    eigenvalues = np.linalg.eigvalsh(scaled)
    if eigenvalues[0] < -1e-8 * max(float(eigenvalues[-1]), 1.0):
        raise ValueError(f"Sigma is not PSD (smallest scaled eigenvalue {eigenvalues[0]:.3e})")
    if eigenvalues[0] < 0:
        logger.debug("shifting the diagonal of Sigma by %.3e to remove round-off", -eigenvalues[0])
        scaled = scaled + (-eigenvalues[0]) * np.eye(n)

    solver, fallback = "CLARABEL", False
    if n == 1:
        w, solver = np.ones(1), "closed-form"
    elif max_weight is not None and abs(max_weight * n - 1.0) <= _CAP_FEASIBILITY_SLACK:
        w, solver = np.full(n, 1.0 / n), "closed-form"
    else:
        try:
            w = _solve_cvxpy(scaled, max_weight, tol)
        except Exception as exc:  # solver and cvxpy errors are heterogeneous
            logger.warning("CLARABEL failed (%s); falling back to SLSQP", exc)
            w, solver, fallback = _solve_slsqp(scaled, max_weight, tol), "SLSQP", True

    w = np.clip(w, 0.0, 1.0 if max_weight is None else max_weight)
    w = w / w.sum()
    check_weights(w, max_weight=max_weight)
    if not return_info:
        return w
    return w, {"solver": solver, "fallback": fallback, "objective": float(w @ s @ w)}


def gmv_unconstrained(
    sigma: np.ndarray,
    ridge_frac: float = 1e-8,
    rank_tol: float = 1e-12,
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    """Unconstrained (shorting allowed) GMV weights ``Sigma^{-1} 1 / (1' Sigma^{-1} 1)``.

    Computed with :func:`numpy.linalg.solve`. If ``Sigma`` is numerically singular (smallest
    eigenvalue at most ``rank_tol`` times the largest, as for a sample covariance with ``T <= N``),
    a ridge of ``ridge_frac * trace(Sigma) / N`` is added to the diagonal and a warning is logged.

    Parameters
    ----------
    sigma : numpy.ndarray
        ``N x N`` symmetric covariance matrix.
    ridge_frac : float, default 1e-8
        Ridge as a fraction of the mean variance, used only for singular matrices.
    rank_tol : float, default 1e-12
        Relative eigenvalue threshold below which the matrix is treated as singular.
    return_info : bool, default False
        If True, also return a dict with ``ridge`` (0.0 if none was added) and ``leverage``
        (``sum |w|``).

    Returns
    -------
    numpy.ndarray or (numpy.ndarray, dict)
        Weights summing to 1; entries may be negative.

    Raises
    ------
    ValueError
        For an invalid matrix or if the weights are undefined (``1' Sigma^{-1} 1 = 0``).
    """
    s = _prepare(sigma)
    n = s.shape[0]
    scale = _mean_variance(s)
    eigenvalues = np.linalg.eigvalsh(s / scale)
    ridge = 0.0
    if eigenvalues[0] <= rank_tol * eigenvalues[-1]:
        ridge = ridge_frac * scale
        logger.warning("Sigma is numerically singular; adding ridge %.3e to the diagonal", ridge)
        s = s + ridge * np.eye(n)
    x = np.linalg.solve(s, np.ones(n))
    total = float(x.sum())
    if abs(total) < np.finfo(float).tiny:
        raise ValueError("unconstrained GMV weights are undefined (1' Sigma^-1 1 is zero)")
    w = x / total
    check_weights(w, long_only=False)
    if not return_info:
        return w
    return w, {"ridge": ridge, "leverage": float(np.abs(w).sum())}
