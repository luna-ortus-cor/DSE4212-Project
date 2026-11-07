"""Shared estimator utilities: correlation matrix and Marchenko-Pastur (MP) helpers.

Reference: Marchenko and Pastur (1967); Laloux, Cizeau, Bouchaud and Potters (1999) for the use of
the MP law to separate signal from noise eigenvalues of financial correlation matrices.
"""

from __future__ import annotations

import numpy as np


def validate_returns(returns: np.ndarray) -> np.ndarray:
    """Return ``returns`` as a finite float 2-D array with at least two rows.

    Parameters
    ----------
    returns : numpy.ndarray
        Candidate ``T x N`` array of returns.

    Raises
    ------
    ValueError
        If the array is not 2-D, has fewer than two rows, or contains NaN or inf.
    """
    x = np.asarray(returns, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError("returns must be a 2-D array with at least 2 rows")
    if not np.isfinite(x).all():
        raise ValueError("returns must be finite (no NaN or inf)")
    return x


def corr_matrix(returns: np.ndarray) -> np.ndarray:
    """Sample correlation matrix of a ``T x N`` array.

    Columns are demeaned and scaled by their sample standard deviation (``ddof=1``);
    ``C = Z'Z / (T - 1)``. The result is explicitly symmetrized with a unit diagonal.

    Parameters
    ----------
    returns : numpy.ndarray
        ``T x N`` array of returns (no NaN).

    Returns
    -------
    numpy.ndarray
        ``N x N`` symmetric correlation matrix.

    Raises
    ------
    ValueError
        If the input is not 2-D, has fewer than 2 rows, or a column has zero variance.
    """
    x = np.asarray(returns, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError("returns must be a 2-D array with at least 2 rows")
    x = x - x.mean(axis=0)
    sd = x.std(axis=0, ddof=1)
    if np.any(sd <= 0):
        raise ValueError("returns contain a column with zero variance")
    z = x / sd
    c = z.T @ z / (x.shape[0] - 1)
    c = (c + c.T) / 2.0
    np.fill_diagonal(c, 1.0)
    return c


def cov_to_corr(cov: np.ndarray) -> np.ndarray:
    """Convert a covariance matrix to a correlation matrix, ``D^{-1/2} Sigma D^{-1/2}``.

    Parameters
    ----------
    cov : numpy.ndarray
        ``N x N`` covariance matrix with strictly positive diagonal.

    Returns
    -------
    numpy.ndarray
        Symmetric correlation matrix with unit diagonal, clipped to ``[-1, 1]``.

    Raises
    ------
    ValueError
        If the matrix is not square or has a non-positive diagonal entry.
    """
    a = np.asarray(cov, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("cov must be a square matrix")
    sd = np.sqrt(np.diag(a))
    if np.any(~np.isfinite(sd)) or np.any(sd <= 0):
        raise ValueError("cov must have a strictly positive diagonal")
    corr = a / np.outer(sd, sd)
    corr = np.clip((corr + corr.T) / 2.0, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def corr_eigenvalues(returns: np.ndarray) -> np.ndarray:
    """Eigenvalues of the sample correlation matrix, sorted in descending order.

    Parameters
    ----------
    returns : numpy.ndarray
        ``T x N`` array of returns.
    """
    return np.linalg.eigh(corr_matrix(returns))[0][::-1]


def mp_edges(q: float, sigma2: float = 1.0) -> tuple[float, float]:
    """Marchenko-Pastur support edges ``sigma2 * (1 -/+ sqrt(q))**2`` for ``q = N / T``.

    Parameters
    ----------
    q : float
        Aspect ratio ``N / T`` (positive).
    sigma2 : float, default 1.0
        Variance of the underlying noise; 1 for a correlation matrix.

    Returns
    -------
    (float, float)
        Lower edge ``lambda_-`` and upper edge ``lambda_+``.
    """
    if q <= 0:
        raise ValueError(f"q must be > 0, got {q}")
    root = np.sqrt(q)
    return float(sigma2 * (1.0 - root) ** 2), float(sigma2 * (1.0 + root) ** 2)


def mp_density(x: np.ndarray, q: float, sigma2: float = 1.0) -> np.ndarray:
    """Marchenko-Pastur density ``sqrt((l+ - x)(x - l-)) / (2 pi sigma2 q x)`` on ``(l-, l+)``.

    Zero outside the support. Only ``0 < q <= 1`` is supported (for ``q > 1`` the law has a
    point mass at zero).

    Parameters
    ----------
    x : numpy.ndarray
        Points at which to evaluate the density.
    q : float
        Aspect ratio ``N / T`` in ``(0, 1]``.
    sigma2 : float, default 1.0
        Noise variance.
    """
    if not 0 < q <= 1:
        raise ValueError(f"mp_density supports 0 < q <= 1, got {q}")
    lo, hi = mp_edges(q, sigma2)
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    inside = (x > lo) & (x < hi)
    xi = x[inside]
    out[inside] = np.sqrt((hi - xi) * (xi - lo)) / (2.0 * np.pi * sigma2 * q * xi)
    return out


def n_above_mp_edge(eigenvalues: np.ndarray, q: float, sigma2: float = 1.0) -> int:
    """Number of eigenvalues strictly above the MP upper edge (the "signal" count).

    Parameters
    ----------
    eigenvalues : numpy.ndarray
        Eigenvalues of a correlation matrix.
    q : float
        Aspect ratio ``N / T``.
    sigma2 : float, default 1.0
        Noise variance.
    """
    return int(np.sum(np.asarray(eigenvalues) > mp_edges(q, sigma2)[1]))


def winsorize_cols(x: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """Clip each column of ``x`` at its own ``lower`` and ``upper`` quantiles.

    Meant to be applied to one estimation window at a time, so that no information from outside
    the window (in particular from the future) enters the clipping bounds.

    Parameters
    ----------
    x : numpy.ndarray
        ``T x N`` array.
    lower, upper : float
        Quantile levels with ``0 <= lower < upper <= 1`` (for example 0.001 and 0.999).

    Returns
    -------
    numpy.ndarray
        A new array of the same shape; the input is not modified.
    """
    if not 0.0 <= lower < upper <= 1.0:
        raise ValueError(f"need 0 <= lower < upper <= 1, got lower={lower}, upper={upper}")
    a = np.asarray(x, dtype=float)
    if a.ndim != 2:
        raise ValueError("x must be a 2-D array")
    lo = np.quantile(a, lower, axis=0)
    hi = np.quantile(a, upper, axis=0)
    return np.clip(a, lo, hi)


def condition_number(cov: np.ndarray) -> float:
    """Ratio of the largest to the smallest eigenvalue of a symmetric matrix.

    Returns ``inf`` if the smallest eigenvalue is not positive (singular or indefinite matrix).

    Parameters
    ----------
    cov : numpy.ndarray
        ``N x N`` symmetric matrix.
    """
    eigenvalues = np.linalg.eigh((np.asarray(cov, dtype=float) + np.asarray(cov).T) / 2.0)[0]
    if eigenvalues[0] <= 0:
        return float("inf")
    return float(eigenvalues[-1] / eigenvalues[0])


def check_psd(cov: np.ndarray, tol: float = 1e-10) -> None:
    """Raise if ``cov`` is not a symmetric positive semi-definite matrix.

    Symmetry is checked to ``tol`` times the largest absolute entry, and the smallest eigenvalue
    must be at least ``-tol`` times the largest eigenvalue (so rank-deficient but valid matrices
    with tiny negative round-off eigenvalues pass).

    Parameters
    ----------
    cov : numpy.ndarray
        Candidate covariance matrix.
    tol : float, default 1e-10
        Relative tolerance.

    Raises
    ------
    ValueError
        If the matrix is not square, not symmetric, or has a clearly negative eigenvalue.
    """
    a = np.asarray(cov, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("matrix must be square")
    scale = float(np.abs(a).max()) if a.size else 0.0
    if not np.allclose(a, a.T, rtol=0.0, atol=tol * max(scale, np.finfo(float).tiny)):
        raise ValueError("matrix is not symmetric")
    eig = np.linalg.eigh((a + a.T) / 2.0)[0]
    if eig[0] < -tol * max(float(eig[-1]), np.finfo(float).tiny):
        raise ValueError(f"matrix is not PSD: smallest eigenvalue {eig[0]:.3e}")
