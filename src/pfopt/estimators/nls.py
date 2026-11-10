"""Quadratic-inverse shrinkage (QIS), a nonlinear shrinkage covariance estimator.

Reference: O. Ledoit and M. Wolf (2022), "Quadratic shrinkage for large covariance matrices",
Bernoulli 28(3), 1519-1547 (their closed-form MATLAB routine ``QIS``, re-implemented here).

The sample eigenvectors are kept and the sample eigenvalues ``lambda_i`` are replaced by nonlinearly
shrunk values ``delta_i``. With ``c = N / n`` (``n = T - 1``) and the kernel bandwidth
``h = min(c^2, 1 / c^2)^0.35 / N^0.35``, the shrunk values come from a quadratic-inverse estimate of
the Hilbert transform of the sample spectral density, evaluated on the inverse eigenvalues. For
``N <= n``

    delta_i = 1 / ( (1 - c)^2 / lambda_i + 2 c (1 - c) theta_i / lambda_i
                    + c^2 (theta_i^2 + H_i^2) / lambda_i ),

and for ``N > n`` the ``N - n`` zero eigenvalues share the common value
``1 / ((c - 1) mean(1 / lambda))``. The result is rescaled so that the trace equals the trace of the
sample covariance. Like RMT clipping, only the eigenvalues are modified; unlike the hard threshold
at the Marchenko-Pastur edge, the shrinkage is smooth and applies to every eigenvalue.
"""

from __future__ import annotations

import numpy as np

from pfopt.estimators.sample import sample_cov
from pfopt.estimators.utils import validate_returns

_EIGENVALUE_FLOOR = 1e-12


def nls_cov(
    returns: np.ndarray, return_info: bool = False
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    """Quadratic-inverse shrinkage (QIS) covariance estimate.

    Parameters
    ----------
    returns : numpy.ndarray
        ``T x N`` array of returns without NaN (demeaned internally, divisor ``T - 1``), with
        ``T >= 3``.
    return_info : bool, default False
        Also return a dict with the bandwidth ``h``, the sample eigenvalues ``sample_eigenvalues``
        and the shrunk eigenvalues ``eigenvalues`` (both ascending, length ``N``).

    Returns
    -------
    numpy.ndarray or (numpy.ndarray, dict)
        Symmetric positive definite ``N x N`` covariance with the same trace as the sample
        covariance.

    Raises
    ------
    ValueError
        If ``returns`` is invalid or has fewer than three rows.
    """
    x = validate_returns(returns)
    t, p = x.shape
    if t < 3:
        raise ValueError(f"need at least 3 observations, got {t}")
    n = t - 1
    c = p / n
    values, vectors = np.linalg.eigh(sample_cov(x))
    keep = min(p, n)
    lam = np.maximum(values[p - keep :], _EIGENVALUE_FLOOR * max(float(values[-1]), 1e-300))
    inverse = 1.0 / lam
    h = min(c**2, 1.0 / c**2) ** 0.35 / p**0.35
    lj = np.tile(inverse, (keep, 1))
    diff = lj - lj.T
    denominator = diff**2 + h**2 * lj**2
    theta = np.mean(lj * diff / denominator, axis=1)
    h_theta = np.mean(lj * (h * lj) / denominator, axis=1)
    a_theta2 = theta**2 + h_theta**2
    if p <= n:
        delta = 1.0 / (
            (1.0 - c) ** 2 * inverse
            + 2.0 * c * (1.0 - c) * inverse * theta
            + c**2 * inverse * a_theta2
        )
    else:
        delta0 = 1.0 / ((c - 1.0) * float(np.mean(inverse)))
        delta = np.concatenate([np.full(p - n, delta0), 1.0 / (inverse * a_theta2)])
    delta = delta * (float(lam.sum()) / float(delta.sum()))
    cov = (vectors * delta) @ vectors.T
    cov = (cov + cov.T) / 2.0
    if not return_info:
        return cov
    return cov, {"h": h, "sample_eigenvalues": values, "eigenvalues": delta}
