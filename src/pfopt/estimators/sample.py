"""Sample and Ledoit-Wolf covariance estimators (PLAN.md step 2.1).

Both take a ``T x N`` array of log returns and return an ``N x N`` symmetric covariance matrix.
"""

from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf

from pfopt.estimators.utils import validate_returns


def sample_cov(returns: np.ndarray) -> np.ndarray:
    """Sample covariance matrix with demeaned columns and ``ddof=1``.

    Formula: ``S = (X - mean)' (X - mean) / (T - 1)``. The matrix is explicitly symmetrized.
    It is positive semi-definite up to floating-point error, and has rank at most ``T - 1``,
    so it is singular whenever ``T <= N`` (the situation that motivates denoising and shrinkage).

    Parameters
    ----------
    returns : numpy.ndarray
        ``T x N`` array of returns without NaN.

    Returns
    -------
    numpy.ndarray
        ``N x N`` symmetric covariance matrix.

    Raises
    ------
    ValueError
        If ``returns`` is not 2-D, has fewer than two rows, or contains non-finite values.
    """
    x = validate_returns(returns)
    centered = x - x.mean(axis=0)
    cov = centered.T @ centered / (x.shape[0] - 1)
    return (cov + cov.T) / 2.0


def lw_cov(
    returns: np.ndarray, return_shrinkage: bool = False
) -> np.ndarray | tuple[np.ndarray, float]:
    """Ledoit-Wolf linear shrinkage toward a scaled identity.

    Formula: ``Sigma = (1 - delta) S + delta mu I`` with ``S`` the maximum-likelihood sample
    covariance (divisor ``T``), ``mu = trace(S) / N`` and ``delta`` in ``[0, 1]`` the optimal
    intensity of Ledoit and Wolf (2004), "A well-conditioned estimator for large-dimensional
    covariance matrices", Journal of Multivariate Analysis 88, 365-411. Implemented with
    :class:`sklearn.covariance.LedoitWolf`. The trace is preserved and the result is positive
    definite even when ``T <= N``.

    Parameters
    ----------
    returns : numpy.ndarray
        ``T x N`` array of returns without NaN.
    return_shrinkage : bool, default False
        If True, also return the estimated shrinkage intensity ``delta``.

    Returns
    -------
    numpy.ndarray or (numpy.ndarray, float)
        ``N x N`` symmetric covariance matrix, optionally with ``delta``.
    """
    x = validate_returns(returns)
    fit = LedoitWolf().fit(x)
    cov = (fit.covariance_ + fit.covariance_.T) / 2.0
    if return_shrinkage:
        return cov, float(fit.shrinkage_)
    return cov
