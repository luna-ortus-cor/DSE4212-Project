"""Regularized minimum-variance weights.

Ridge: ``min w'(Sigma + lambda m I) w`` with ``m = tr(Sigma)/N`` (long-only); ``lambda -> infinity``
gives 1/N. Shrinkage: ``(1 - gamma) w + gamma / N``. L1 bound (DeMiguel, Garlappi, Nogales and
Uppal, 2009): ``min w'Sigma w`` subject to ``sum(w) = 1`` and ``||w||_1 <= c`` with shorting
allowed; ``c = 1`` is the long-only portfolio, and a large ``c`` is the unconstrained one. Under
long-only and ``sum(w) = 1`` the L1 norm is constant, so the bound only matters with shorting.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np

from pfopt.optimizers.gmv import gmv_long_only
from pfopt.weights import check_weights


def ridge_cov(sigma: np.ndarray, multiple: float) -> np.ndarray:
    """Return ``Sigma + multiple * tr(Sigma)/N * I``.

    Parameters
    ----------
    sigma : numpy.ndarray
        ``N x N`` covariance matrix.
    multiple : float
        Ridge as a multiple of the mean variance, at least 0.
    """
    if multiple < 0:
        raise ValueError(f"multiple must be >= 0, got {multiple}")
    s = np.asarray(sigma, dtype=float)
    n = s.shape[0]
    return (s + s.T) / 2.0 + multiple * float(np.trace(s)) / n * np.eye(n)


def gmv_ridge_long_only(sigma: np.ndarray, multiple: float) -> np.ndarray:
    """Long-only GMV weights of the ridge-regularized covariance."""
    return np.asarray(gmv_long_only(ridge_cov(sigma, multiple)))


def shrink_to_equal(weights: np.ndarray, gamma: float) -> np.ndarray:
    """Return ``(1 - gamma) w + gamma / N`` for ``gamma`` in ``[0, 1]``."""
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must be in [0, 1], got {gamma}")
    w = np.asarray(weights, dtype=float)
    return (1.0 - gamma) * w + gamma / w.size


def gmv_l1(sigma: np.ndarray, bound: float, ridge: float = 0.0) -> tuple[np.ndarray, float]:
    """GMV with ``sum(w) = 1`` and ``||w||_1 <= bound`` (shorting allowed).

    Parameters
    ----------
    sigma : numpy.ndarray
        ``N x N`` covariance matrix.
    bound : float
        L1 bound, at least 1.
    ridge : float, default 0.0
        Optional ridge multiple (elastic net when positive).

    Returns
    -------
    (numpy.ndarray, float)
        Weights (sum 1, possibly negative) and their leverage ``||w||_1``.
    """
    if bound < 1.0:
        raise ValueError(f"bound must be >= 1, got {bound}")
    s = ridge_cov(sigma, ridge)
    scale = float(np.trace(s)) / s.shape[0]
    n = s.shape[0]
    w = cp.Variable(n)
    objective = cp.Minimize(cp.quad_form(w, cp.psd_wrap(s / scale)))
    problem = cp.Problem(objective, [cp.sum(w) == 1, cp.norm1(w) <= bound])
    problem.solve(solver=cp.CLARABEL, tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10)
    if problem.status != cp.OPTIMAL or w.value is None:
        raise RuntimeError(f"CLARABEL returned status {problem.status!r}")
    x = np.asarray(w.value, dtype=float)
    x = x / x.sum()
    check_weights(x, long_only=False)
    return x, float(np.abs(x).sum())
