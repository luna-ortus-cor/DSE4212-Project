"""Risk-based portfolios: inverse volatility and equal risk contribution (PLAN.md step 8.7).

Inverse volatility (naive risk parity) uses only the variances, ``w_i proportional to 1 / sigma_i``.
Equal risk contribution (ERC, Maillard, Roncalli and Teiletche, 2010) chooses weights whose risk
contributions ``RC_i = w_i (Sigma w)_i / (w' Sigma w)`` are all equal to ``1 / N``. With the
long-only budget constraint the problem is the strictly convex program of Spinu (2013),

    minimize   0.5 y' Sigma y - (1 / N) sum_i log y_i      over   y > 0,

whose optimality condition ``y_i (Sigma y)_i = 1 / N`` is exactly equal risk contributions; the
weights are ``w = y / sum(y)``. It is solved here by a damped Newton method (no solver needed).
Neither portfolio has a weight-cap mechanism, so ``backtest.max_weight`` does not apply to them.
"""

from __future__ import annotations

import numpy as np


def _prepare(cov: np.ndarray) -> np.ndarray:
    """Validate a covariance matrix with a positive diagonal and return it symmetrized."""
    s = np.asarray(cov, dtype=float)
    if s.ndim != 2 or s.shape[0] != s.shape[1] or s.shape[0] < 1:
        raise ValueError("cov must be a non-empty square matrix")
    if not np.isfinite(s).all():
        raise ValueError("cov must be finite")
    if (np.diag(s) <= 0).any():
        raise ValueError("cov must have a positive diagonal")
    return (s + s.T) / 2.0


def inverse_vol_weights(cov: np.ndarray) -> np.ndarray:
    """Inverse-volatility weights ``(1 / sigma_i) / sum_j (1 / sigma_j)`` from the diagonal."""
    inv = 1.0 / np.sqrt(np.diag(_prepare(cov)))
    return inv / inv.sum()


def risk_contributions(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Fractional risk contributions ``w_i (Sigma w)_i / (w' Sigma w)`` (they sum to 1)."""
    s = _prepare(cov)
    w = np.asarray(weights, dtype=float)
    marginal = w * (s @ w)
    return marginal / marginal.sum()


def erc_weights(cov: np.ndarray, tol: float = 1e-12, max_iter: int = 100) -> tuple[np.ndarray, int]:
    """Long-only equal-risk-contribution weights (Spinu 2013 formulation, damped Newton).

    Parameters
    ----------
    cov : numpy.ndarray
        Symmetric positive semi-definite ``N x N`` covariance with a positive diagonal.
    tol : float, default 1e-12
        Convergence: ``max_i |y_i (Sigma y)_i - 1 / N| <= tol / N`` on the scaled problem.
    max_iter : int, default 100
        Maximum Newton iterations.

    Returns
    -------
    (numpy.ndarray, int)
        Weights (positive, summing to 1) and the number of Newton iterations used.

    Raises
    ------
    RuntimeError
        If the iteration does not converge.
    """
    s = _prepare(cov)
    n = s.shape[0]
    s = s / (np.trace(s) / n)
    b = 1.0 / n
    y = 1.0 / np.sqrt(np.diag(s) * n)

    def objective(v: np.ndarray) -> float:
        return float(0.5 * v @ s @ v - b * np.log(v).sum())

    for iteration in range(max_iter + 1):
        sy = s @ y
        gradient = sy - b / y
        if np.max(np.abs(y * sy - b)) <= tol * b:
            return y / y.sum(), iteration
        hessian = s + np.diag(b / y**2)
        step = -np.linalg.solve(hessian, gradient)
        alpha, f0 = 1.0, objective(y)
        while np.any(y + alpha * step <= 0) or objective(y + alpha * step) > f0 + 1e-14:
            alpha *= 0.5
            if alpha < 1e-12:
                raise RuntimeError("ERC line search failed")
        y = y + alpha * step
    raise RuntimeError(f"ERC did not converge in {max_iter} iterations")
