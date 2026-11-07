"""PCA statistical-factor covariance estimator (PLAN.md step 2.3).

Model: returns follow a ``K``-factor structure estimated by principal components, with a diagonal
(idiosyncratic) residual covariance. The systematic part is the rank-``K`` truncation of the
sample covariance and the diagonal residual restores each asset's own variance:

    Sigma_sys = V_K Lambda_K V_K',   D = diag(S - Sigma_sys),   Sigma_PCA = Sigma_sys + D.

References: Bai and Ng (2002), "Determining the number of factors in approximate factor models",
Econometrica 70(1), 191-221; Fan, Liao and Mincheva (2013), "Large covariance estimation by
thresholding principal orthogonal complements", JRSS-B 75(4), for the "low-rank plus sparse
(here diagonal) residual" structure.
"""

from __future__ import annotations

import numpy as np

from pfopt.estimators.sample import sample_cov
from pfopt.estimators.utils import corr_matrix, n_above_mp_edge, validate_returns

K_RULES = ("mp", "fixed", "baing")


def _bai_ng_ic2(eigenvalues: np.ndarray, t: int, n: int, kmax: int) -> int:
    """Number of factors minimizing Bai-Ng's IC_p2 over ``k = 1..kmax``.

    ``IC(k) = ln V(k) + k (N + T) / (N T) ln(min(N, T))`` where ``V(k)`` is the mean squared
    residual of the rank-``k`` principal-components fit of the standardized returns, which equals
    ``(N - sum of the k largest correlation eigenvalues) / N``.
    """
    k_eff = int(min(kmax, n - 1, t - 1))
    if k_eff < 1:
        return 1
    ks = np.arange(1, k_eff + 1)
    v = (n - np.cumsum(eigenvalues[:k_eff])) / n
    penalty = ks * (n + t) / (n * t) * np.log(min(n, t))
    valid = v > 1e-12
    ic = np.full(k_eff, np.inf)
    ic[valid] = np.log(v[valid]) + penalty[valid]
    return int(np.argmin(ic)) + 1


def n_factors(
    returns: np.ndarray, rule: str = "mp", K: int | None = None, kmax: int = 10
) -> int:
    """Number of factors ``K`` chosen by the given rule.

    Parameters
    ----------
    returns : numpy.ndarray
        ``T x N`` array of returns without NaN.
    rule : {"mp", "fixed", "baing"}, default "mp"
        ``"mp"``: number of correlation-matrix eigenvalues above the Marchenko-Pastur upper edge
        (as in :func:`pfopt.estimators.rmt.rmt_cov`), at least 1. ``"fixed"``: the given ``K``
        (``0 <= K <= N``). ``"baing"``: Bai-Ng (2002) IC_p2 on standardized returns, searched over
        ``1..kmax``.
    K : int, optional
        Required for ``rule="fixed"``, must be None otherwise.
    kmax : int, default 10
        Maximum number of factors searched by the Bai-Ng rule.

    Raises
    ------
    ValueError
        For an unknown rule, an invalid or missing ``K``, or invalid returns.
    """
    if rule not in K_RULES:
        raise ValueError(f"rule must be one of {K_RULES}, got {rule!r}")
    x = validate_returns(returns)
    t, n = x.shape
    if rule == "fixed":
        if K is None or not isinstance(K, int | np.integer) or not 0 <= K <= n:
            raise ValueError(f"K must be an integer in [0, {n}] for rule 'fixed', got {K!r}")
        return int(K)
    if K is not None:
        raise ValueError(f"K must be None for rule {rule!r}")
    eigenvalues = np.linalg.eigh(corr_matrix(x))[0][::-1]
    if rule == "mp":
        return max(1, n_above_mp_edge(eigenvalues, n / t))
    if kmax < 1:
        raise ValueError(f"kmax must be >= 1, got {kmax}")
    return _bai_ng_ic2(eigenvalues, t, n, kmax)


def pca_factor_cov(
    returns: np.ndarray,
    K: int | None = None,
    rule: str = "mp",
    kmax: int = 10,
    min_resid_frac: float = 1e-10,
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    """PCA-factor covariance: top-``K`` principal components plus a diagonal residual.

    Parameters
    ----------
    returns : numpy.ndarray
        ``T x N`` array of returns without NaN.
    K : int, optional
        Number of factors for ``rule="fixed"`` (``0`` gives a diagonal matrix, ``N`` reproduces the
        sample covariance). Must be None for the other rules.
    rule : {"mp", "fixed", "baing"}, default "mp"
        How ``K`` is chosen; see :func:`n_factors`.
    kmax : int, default 10
        Maximum factors for the Bai-Ng rule.
    min_resid_frac : float, default 1e-10
        Floor for each residual variance as a fraction of the asset's sample variance. It only
        binds when the rank-``K`` fit explains an asset's variance (almost) exactly, for example
        ``K = N``; otherwise the diagonal of the result equals the sample variances.
    return_info : bool, default False
        If True, also return a dict with ``K``, ``rule`` and ``explained`` (the share of total
        sample variance carried by the systematic part).

    Returns
    -------
    numpy.ndarray or (numpy.ndarray, dict)
        ``N x N`` symmetric positive semi-definite covariance matrix (positive definite when every
        sample variance is positive).
    """
    x = validate_returns(returns)
    n = x.shape[1]
    k = n_factors(x, rule=rule, K=K, kmax=kmax)
    sample = sample_cov(x)
    eigenvalues, eigenvectors = np.linalg.eigh(sample)
    if k > 0:
        top_values = np.maximum(eigenvalues[n - k :], 0.0)
        top_vectors = eigenvectors[:, n - k :]
        systematic = (top_vectors * top_values) @ top_vectors.T
    else:
        systematic = np.zeros((n, n))
    variances = np.diag(sample)
    residual = np.maximum(variances - np.diag(systematic), min_resid_frac * variances)
    cov = systematic + np.diag(residual)
    cov = (cov + cov.T) / 2.0
    if not return_info:
        return cov
    total = float(np.trace(sample))
    info: dict[str, object] = {
        "K": k,
        "rule": rule,
        "explained": float(np.trace(systematic) / total) if total > 0 else 0.0,
    }
    return cov, info
