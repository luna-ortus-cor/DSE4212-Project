"""PCA factor covariance with a sector-block residual (PLAN.md step 8.24).

The PCA-factor estimator keeps the top ``K`` principal components and a diagonal residual. Step 8.23
found sector correlation in the discarded part, so this estimator keeps the residual of the sample
covariance for pairs in the same group:

    S = sample covariance,   Sigma_sys = V_K Lambda_K V_K',   R = S - Sigma_sys,
    Sigma = Sigma_sys + R o M,   M_ij = 1 if i = j or i and j share a known group, else 0.

``R`` is positive semi-definite (the retained part of a spectral decomposition is removed) and a
principal submatrix of a positive semi-definite matrix is positive semi-definite, so ``R o M`` (a
direct sum of principal blocks of ``R``) is too. The diagonal equals the sample variances. Special
cases: with no known group the result is the PCA-factor covariance; with one group for all assets
it is the sample covariance.

References: Fan, Liao and Mincheva (2013), "Large covariance estimation by thresholding principal
orthogonal complements", JRSS-B 75(4) (block-sparse residual of a factor model); Bai and Ng (2002)
and Marchenko and Pastur (1967) for the choice of ``K`` (see :mod:`pfopt.estimators.pca_factor`).
"""

from __future__ import annotations

import numpy as np

from pfopt.estimators.pca_factor import n_factors
from pfopt.estimators.sample import sample_cov
from pfopt.estimators.utils import validate_returns

UNKNOWN_GROUP = -1


def block_mask(groups: np.ndarray) -> np.ndarray:
    """Boolean ``N x N`` mask: diagonal, and pairs sharing a known (non-negative) group."""
    g = np.asarray(groups)
    if g.ndim != 1:
        raise ValueError("groups must be a 1-D array")
    same = (g[:, None] == g[None, :]) & (g[:, None] >= 0)
    return same | np.eye(g.size, dtype=bool)


def sector_block_cov(
    returns: np.ndarray,
    groups: np.ndarray,
    K: int | None = None,
    rule: str = "mp",
    kmax: int = 10,
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    """PCA-factor covariance plus the same-group part of the residual.

    Parameters
    ----------
    returns : numpy.ndarray
        ``T x N`` array of returns without NaN.
    groups : numpy.ndarray
        Integer group (sector) of each of the ``N`` assets; negative values mean unknown, and an
        unknown asset keeps only its own residual variance.
    K, rule, kmax
        Number of factors and the rule that chooses it, as in
        :func:`pfopt.estimators.pca_factor.pca_factor_cov`.
    return_info : bool, default False
        If True, also return a dict with ``K``, ``explained`` (share of total variance carried
        by the systematic part) and ``block_share`` (share of the residual's squared off-diagonal
        mass that lies inside the kept blocks).

    Returns
    -------
    numpy.ndarray or (numpy.ndarray, dict)
        ``N x N`` symmetric positive semi-definite covariance matrix.

    Raises
    ------
    ValueError
        For invalid returns, a ``groups`` array that does not match the number of assets, or an
        invalid ``K`` / ``rule``.
    """
    x = validate_returns(returns)
    n = x.shape[1]
    g = np.asarray(groups)
    if g.shape != (n,):
        raise ValueError(f"groups must have length {n}, got shape {g.shape}")
    k = n_factors(x, rule=rule, K=K, kmax=kmax)
    sample = sample_cov(x)
    values, vectors = np.linalg.eigh(sample)
    if k > 0:
        top_values = np.maximum(values[n - k :], 0.0)
        top_vectors = vectors[:, n - k :]
        systematic = (top_vectors * top_values) @ top_vectors.T
    else:
        systematic = np.zeros((n, n))
    residual = sample - systematic
    mask = block_mask(g)
    cov = systematic + np.where(mask, residual, 0.0)
    cov = (cov + cov.T) / 2.0
    if not return_info:
        return cov
    off = ~np.eye(n, dtype=bool)
    mass = float((residual[off] ** 2).sum())
    kept = float((residual[mask & off] ** 2).sum())
    total = float(np.trace(sample))
    info: dict[str, object] = {
        "K": k,
        "explained": float(np.trace(systematic) / total) if total > 0 else 0.0,
        "block_share": kept / mass if mass > 0 else 0.0,
    }
    return cov, info
