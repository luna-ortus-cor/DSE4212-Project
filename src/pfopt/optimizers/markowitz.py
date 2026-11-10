"""Long-only Markowitz portfolios and the efficient frontier.

Maximum-Sharpe portfolio (long-only, fully invested, optional per-asset cap): the problem
``max (mu - rf)'w / sqrt(w'Sigma w)`` is quasi-concave but has a convex reformulation (Cornuejols
and Tutuncu, 2006): minimize ``y'Sigma y`` subject to ``(mu - rf)'y = 1`` and ``y >= 0``
(and ``y <= cap * sum(y)`` with a cap), then ``w = y / sum(y)``.

Efficient frontier: for each target return between the GMV portfolio's return and the largest
attainable return, minimize ``w'Sigma w`` subject to ``1'w = 1``, ``w >= 0``, ``mu'w >= target``.

All inputs are per-period quantities (the same period for ``mu`` and ``Sigma``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cvxpy as cp
import numpy as np
from scipy.optimize import minimize

from pfopt.optimizers.gmv import _mean_variance, _prepare, gmv_long_only
from pfopt.weights import check_weights

logger = logging.getLogger(__name__)

_SLACK = 1e-12


@dataclass(frozen=True)
class Frontier:
    """Long-only efficient frontier.

    Attributes
    ----------
    returns : numpy.ndarray
        Expected return of each frontier portfolio (per period), increasing.
    volatilities : numpy.ndarray
        Standard deviation of each portfolio (per period), non-decreasing.
    weights : numpy.ndarray
        ``n_points x N`` weights, from the GMV portfolio to the maximum-return portfolio.
    """

    returns: np.ndarray
    volatilities: np.ndarray
    weights: np.ndarray


def _validate(
    mu: np.ndarray, sigma: np.ndarray, max_weight: float | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate inputs; return ``mu``, the symmetrized ``Sigma`` and its rescaled PSD version."""
    s = _prepare(sigma)
    n = s.shape[0]
    m = np.asarray(mu, dtype=float).ravel()
    if m.size != n:
        raise ValueError(f"mu has {m.size} entries but Sigma is {n} x {n}")
    if not np.isfinite(m).all():
        raise ValueError("mu must be finite")
    if max_weight is not None:
        if not 0.0 < max_weight <= 1.0:
            raise ValueError(f"max_weight must be in (0, 1], got {max_weight}")
        if max_weight * n < 1.0 - _SLACK:
            raise ValueError(
                f"max_weight={max_weight} is infeasible for N={n} assets: need max_weight >= 1/N"
            )
    scaled = s / _mean_variance(s)
    eigenvalues = np.linalg.eigvalsh(scaled)
    if eigenvalues[0] < -1e-8 * max(float(eigenvalues[-1]), 1.0):
        raise ValueError(f"Sigma is not PSD (smallest scaled eigenvalue {eigenvalues[0]:.3e})")
    if eigenvalues[0] < 0:
        scaled = scaled + (-eigenvalues[0]) * np.eye(n)
    return m, s, scaled


def _max_return_weights(mu: np.ndarray, cap: float | None) -> tuple[np.ndarray, float]:
    """Weights maximizing ``mu'w`` on the (capped) simplex: fill the best assets up to the cap."""
    limit = 1.0 if cap is None else cap
    weights = np.zeros(mu.size)
    remaining = 1.0
    for i in np.argsort(-mu, kind="stable"):
        take = min(limit, remaining)
        weights[i] = take
        remaining -= take
        if remaining <= _SLACK:
            break
    weights /= weights.sum()
    return weights, float(mu @ weights)


def _cvx_tangency(
    sigma_scaled: np.ndarray, mu_scaled: np.ndarray, cap: float | None, tol: float
) -> np.ndarray:
    n = sigma_scaled.shape[0]
    y = cp.Variable(n)
    constraints = [mu_scaled @ y == 1, y >= 0]
    if cap is not None:
        constraints.append(y <= cap * cp.sum(y))
    problem = cp.Problem(cp.Minimize(cp.quad_form(y, cp.psd_wrap(sigma_scaled))), constraints)
    problem.solve(solver=cp.CLARABEL, tol_gap_abs=tol, tol_gap_rel=tol, tol_feas=tol)
    if problem.status != cp.OPTIMAL or y.value is None:
        raise RuntimeError(f"CLARABEL returned status {problem.status!r}")
    return np.asarray(y.value, dtype=float)


def _slsqp_max_sharpe(
    sigma_scaled: np.ndarray, mu_scaled: np.ndarray, cap: float | None, tol: float
) -> np.ndarray:
    n = sigma_scaled.shape[0]
    upper = 1.0 if cap is None else cap

    def neg_sharpe(w: np.ndarray) -> float:
        return -float(mu_scaled @ w) / np.sqrt(max(float(w @ sigma_scaled @ w), 1e-300))

    def neg_sharpe_grad(w: np.ndarray) -> np.ndarray:
        variance = max(float(w @ sigma_scaled @ w), 1e-300)
        root = np.sqrt(variance)
        return -(mu_scaled / root - float(mu_scaled @ w) * (sigma_scaled @ w) / (variance * root))

    result = minimize(
        neg_sharpe,
        np.full(n, 1.0 / n),
        jac=neg_sharpe_grad,
        method="SLSQP",
        bounds=[(0.0, upper)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"ftol": tol, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"SLSQP failed: {result.message}")
    return np.asarray(result.x, dtype=float)


def _cvx_min_variance(
    sigma_scaled: np.ndarray, mu_scaled: np.ndarray, target: float, cap: float | None, tol: float
) -> np.ndarray:
    n = sigma_scaled.shape[0]
    w = cp.Variable(n)
    constraints = [cp.sum(w) == 1, w >= 0, mu_scaled @ w >= target]
    if cap is not None:
        constraints.append(w <= cap)
    problem = cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(sigma_scaled))), constraints)
    problem.solve(solver=cp.CLARABEL, tol_gap_abs=tol, tol_gap_rel=tol, tol_feas=tol)
    if problem.status != cp.OPTIMAL or w.value is None:
        raise RuntimeError(f"CLARABEL returned status {problem.status!r}")
    return np.asarray(w.value, dtype=float)


def _slsqp_min_variance(
    sigma_scaled: np.ndarray, mu_scaled: np.ndarray, target: float, cap: float | None, tol: float
) -> np.ndarray:
    n = sigma_scaled.shape[0]
    upper = 1.0 if cap is None else cap
    result = minimize(
        lambda w: float(w @ sigma_scaled @ w),
        np.full(n, 1.0 / n),
        jac=lambda w: 2.0 * sigma_scaled @ w,
        method="SLSQP",
        bounds=[(0.0, upper)] * n,
        constraints=[
            {"type": "eq", "fun": lambda w: w.sum() - 1.0},
            {"type": "ineq", "fun": lambda w: float(mu_scaled @ w) - target},
        ],
        options={"ftol": tol, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"SLSQP failed: {result.message}")
    return np.asarray(result.x, dtype=float)


def _finalize(w: np.ndarray, max_weight: float | None) -> np.ndarray:
    w = np.clip(w, 0.0, 1.0 if max_weight is None else max_weight)
    w = w / w.sum()
    check_weights(w, max_weight=max_weight)
    return w


def max_sharpe_long_only(
    mu: np.ndarray,
    sigma: np.ndarray,
    rf: float = 0.0,
    max_weight: float | None = None,
    tol: float = 1e-10,
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    """Long-only, fully invested maximum-Sharpe (tangency) portfolio.

    If no asset has an expected return above ``rf`` the Sharpe ratio cannot be positive, and the
    long-only GMV portfolio is returned instead (a warning is logged).

    Parameters
    ----------
    mu : numpy.ndarray
        Length-``N`` expected returns per period.
    sigma : numpy.ndarray
        ``N x N`` covariance matrix for the same period.
    rf : float, default 0.0
        Risk-free rate per period.
    max_weight : float, optional
        Per-asset cap in ``(0, 1]`` with ``max_weight >= 1 / N``.
    tol : float, default 1e-10
        Solver tolerance.
    return_info : bool, default False
        If True, also return a dict with ``solver`` ("CLARABEL", "SLSQP", "closed-form" or
        "gmv-fallback"), ``fallback_gmv`` (True if no asset beat ``rf``), ``fallback_slsqp`` and
        ``sharpe`` (per-period Sharpe ratio of the result).

    Returns
    -------
    numpy.ndarray or (numpy.ndarray, dict)
        Non-negative weights summing to 1 (and not above ``max_weight``).

    Raises
    ------
    ValueError
        For inconsistent or invalid inputs, or an infeasible cap.
    RuntimeError
        If both the primary and the fallback solver fail.
    """
    m, s, scaled = _validate(mu, sigma, max_weight)
    n = m.size
    excess = m - rf
    solver, fallback_gmv, fallback_slsqp = "CLARABEL", False, False
    if not (excess > 0).any():
        logger.warning("no asset has expected return above rf; falling back to GMV")
        w = gmv_long_only(s, max_weight=max_weight, tol=tol)
        solver, fallback_gmv = "gmv-fallback", True
    elif n == 1:
        w, solver = np.ones(1), "closed-form"
    elif max_weight is not None and abs(max_weight * n - 1.0) <= _SLACK:
        w, solver = np.full(n, 1.0 / n), "closed-form"
    else:
        mu_scaled = excess / excess.max()
        try:
            y = _cvx_tangency(scaled, mu_scaled, max_weight, tol)
            w = np.clip(y, 0.0, None)
            w = w / w.sum()
        except Exception as exc:  # solver and cvxpy errors are heterogeneous
            logger.warning("CLARABEL failed (%s); falling back to SLSQP", exc)
            w = _slsqp_max_sharpe(scaled, mu_scaled, max_weight, tol)
            solver, fallback_slsqp = "SLSQP", True
    w = _finalize(w, max_weight)
    if not return_info:
        return w
    variance = float(w @ s @ w)
    sharpe = float((m @ w - rf) / np.sqrt(variance)) if variance > 0 else float("nan")
    info = {
        "solver": solver,
        "fallback_gmv": fallback_gmv,
        "fallback_slsqp": fallback_slsqp,
        "sharpe": sharpe,
    }
    return w, info


def min_variance_for_return(
    mu: np.ndarray,
    sigma: np.ndarray,
    target: float,
    max_weight: float | None = None,
    tol: float = 1e-10,
) -> np.ndarray:
    """Long-only minimum-variance portfolio with expected return at least ``target``.

    Parameters
    ----------
    mu, sigma : numpy.ndarray
        Expected returns and covariance matrix (same period).
    target : float
        Required expected return; it must lie between the GMV portfolio's return and the largest
        attainable return (so the result is on the efficient frontier).
    max_weight : float, optional
        Per-asset cap.
    tol : float, default 1e-10
        Solver tolerance.

    Raises
    ------
    ValueError
        If ``target`` is outside the attainable efficient range, or inputs are invalid.
    """
    m, s, scaled = _validate(mu, sigma, max_weight)
    w_gmv = gmv_long_only(s, max_weight=max_weight, tol=tol)
    w_max, r_max = _max_return_weights(m, max_weight)
    r_gmv = float(m @ w_gmv)
    span = max(abs(r_max), abs(r_gmv), 1e-300)
    if target > r_max + _SLACK * span:
        raise ValueError(f"target {target:.6g} exceeds the maximum attainable return {r_max:.6g}")
    if target < r_gmv - _SLACK * span:
        raise ValueError(f"target {target:.6g} is below the GMV return {r_gmv:.6g}")
    if target <= r_gmv + _SLACK * span:
        return w_gmv
    if target >= r_max - _SLACK * span:
        return w_max
    scale = float(np.abs(m).max())
    mu_scaled, target_scaled = m / scale, target / scale
    try:
        w = _cvx_min_variance(scaled, mu_scaled, target_scaled, max_weight, tol)
    except Exception as exc:  # solver and cvxpy errors are heterogeneous
        logger.warning("CLARABEL failed (%s); falling back to SLSQP", exc)
        w = _slsqp_min_variance(scaled, mu_scaled, target_scaled, max_weight, tol)
    return _finalize(w, max_weight)


def efficient_frontier(
    mu: np.ndarray,
    sigma: np.ndarray,
    n_points: int = 50,
    max_weight: float | None = None,
    tol: float = 1e-10,
) -> Frontier:
    """Long-only efficient frontier on an equally spaced grid of target returns.

    The first portfolio is the GMV portfolio and the last is the maximum-return portfolio (all
    weight on the highest-``mu`` asset, or the best assets up to the cap); the others minimize
    variance for their target return.

    Parameters
    ----------
    mu, sigma : numpy.ndarray
        Expected returns and covariance matrix (same period).
    n_points : int, default 50
        Number of frontier portfolios (at least 2).
    max_weight : float, optional
        Per-asset cap.
    tol : float, default 1e-10
        Solver tolerance.

    Returns
    -------
    Frontier
        Returns, volatilities and weights per portfolio. If every asset has the same expected
        return the frontier collapses to the GMV portfolio repeated ``n_points`` times.
    """
    if n_points < 2:
        raise ValueError(f"n_points must be >= 2, got {n_points}")
    m, s, _ = _validate(mu, sigma, max_weight)
    w_gmv = gmv_long_only(s, max_weight=max_weight, tol=tol)
    w_max, r_max = _max_return_weights(m, max_weight)
    r_gmv = float(m @ w_gmv)
    weights = np.tile(w_gmv, (n_points, 1))
    if r_max - r_gmv > _SLACK * max(abs(r_max), abs(r_gmv), 1e-300):
        targets = np.linspace(r_gmv, r_max, n_points)
        weights[-1] = w_max
        for k in range(1, n_points - 1):
            weights[k] = min_variance_for_return(m, s, float(targets[k]), max_weight, tol)
    returns = weights @ m
    volatilities = np.sqrt(np.einsum("ki,ij,kj->k", weights, s, weights))
    return Frontier(returns=returns, volatilities=volatilities, weights=weights)
