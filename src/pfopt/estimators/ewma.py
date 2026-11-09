"""Exponentially weighted covariance and its RMT-clipped version (PLAN.md steps 8.3 and 8.22).

Within an estimation window of ``T`` observations (oldest first), observation ``i`` gets the weight
``w_i proportional to 0.5 ** ((T - 1 - i) / halflife)``, normalised to sum to one, so the most
recent day counts most. The weighted covariance uses the reliability-weights correction

    Sigma = sum_i w_i (x_i - m)(x_i - m)' / (1 - sum_i w_i^2),   m = sum_i w_i x_i,

which is unbiased for independent observations and equals the usual sample covariance
(``ddof=1``) for equal weights (``halflife = inf``). The effective number of observations is
``T_eff = 1 / sum_i w_i^2`` (Kish, 1965).

Time-weighting is the classical answer to a covariance that moves (RiskMetrics, 1996); it trades
a smaller effective sample for less bias from stale data, in contrast to eigenvalue clipping, which
only reduces the noise of a fixed-size sample.
"""

from __future__ import annotations

import numpy as np

from pfopt.estimators.utils import mp_edges, validate_returns


def ewma_weights(n_obs: int, halflife: float) -> np.ndarray:
    """Normalised exponential weights for ``n_obs`` observations, oldest first.

    Parameters
    ----------
    n_obs : int
        Number of observations (at least 2).
    halflife : float
        Half-life in observations (positive); ``inf`` gives equal weights.

    Returns
    -------
    numpy.ndarray
        Length-``n_obs`` weights that sum to one and increase with time.
    """
    if n_obs < 2:
        raise ValueError(f"n_obs must be >= 2, got {n_obs}")
    if not halflife > 0:
        raise ValueError(f"halflife must be positive, got {halflife}")
    if np.isinf(halflife):
        return np.full(n_obs, 1.0 / n_obs)
    raw = 0.5 ** ((n_obs - 1 - np.arange(n_obs)) / halflife)
    return raw / raw.sum()


def effective_sample_size(weights: np.ndarray) -> float:
    """Kish effective sample size ``(sum w)^2 / sum w^2`` of a weight vector."""
    w = np.asarray(weights, dtype=float)
    return float(w.sum() ** 2 / np.sum(w**2))


def ewma_cov(returns: np.ndarray, halflife: float) -> np.ndarray:
    """Exponentially weighted covariance matrix of a ``T x N`` array (oldest row first).

    Parameters
    ----------
    returns : numpy.ndarray
        ``T x N`` returns without NaN.
    halflife : float
        Half-life in days; ``inf`` recovers :func:`pfopt.estimators.sample.sample_cov`.

    Returns
    -------
    numpy.ndarray
        ``N x N`` symmetric positive semi-definite covariance matrix.
    """
    x = validate_returns(returns)
    w = ewma_weights(x.shape[0], halflife)
    centered = x - w @ x
    cov = (centered * w[:, None]).T @ centered / (1.0 - np.sum(w**2))
    return (cov + cov.T) / 2.0


def ewma_rmt_cov(
    returns: np.ndarray, halflife: float, return_info: bool = False
) -> np.ndarray | tuple[np.ndarray, dict[str, float]]:
    """RMT eigenvalue clipping applied to the exponentially weighted covariance.

    The weighted correlation matrix is eigendecomposed, eigenvalues below the Marchenko-Pastur
    edge ``(1 + sqrt(N / T_eff))^2`` (with the effective sample size ``T_eff`` in place of ``T``)
    are replaced by their mean, and the matrix is rescaled to a unit diagonal and to the weighted
    standard deviations. With ``halflife = inf`` this is :func:`pfopt.estimators.rmt.rmt_cov`.

    Parameters
    ----------
    returns : numpy.ndarray
        ``T x N`` returns without NaN.
    halflife : float
        Half-life in days.
    return_info : bool, default False
        If True, also return ``n_signal``, ``lam_plus`` and ``t_eff``.

    Returns
    -------
    numpy.ndarray or (numpy.ndarray, dict)
        ``N x N`` symmetric positive semi-definite covariance matrix.
    """
    x = validate_returns(returns)
    n_obs, n = x.shape
    cov = ewma_cov(x, halflife)
    sd = np.sqrt(np.diag(cov))
    if np.any(sd <= 0):
        raise ValueError("returns contain a column with zero variance")
    corr = cov / np.outer(sd, sd)
    t_eff = effective_sample_size(ewma_weights(n_obs, halflife))
    lam_plus = mp_edges(n / t_eff)[1]
    eigenvalues, eigenvectors = np.linalg.eigh((corr + corr.T) / 2.0)
    signal = eigenvalues > lam_plus
    clipped = eigenvalues.copy()
    if (~signal).any():
        clipped[~signal] = eigenvalues[~signal].mean()
    tilde = (eigenvectors * clipped) @ eigenvectors.T
    scale = 1.0 / np.sqrt(np.diag(tilde))
    clean = tilde * np.outer(scale, scale)
    out = clean * np.outer(sd, sd)
    out = (out + out.T) / 2.0
    if return_info:
        return out, {"n_signal": float(signal.sum()), "lam_plus": float(lam_plus), "t_eff": t_eff}
    return out
