"""Random-matrix-theory (RMT) eigenvalue clipping (PLAN.md step 2.2).

Idea: for ``T`` observations of ``N`` independent assets, the eigenvalues of the sample
correlation matrix follow the Marchenko-Pastur (MP) law with upper edge
``lambda_+ = sigma2 * (1 + sqrt(q))**2`` and ``q = N / T``. Eigenvalues below the edge are
indistinguishable from noise; those above it carry signal (the market mode, sectors). Clipping
keeps the signal eigenvalues and flattens the noise ones.

References: Marchenko and Pastur (1967); Laloux, Cizeau, Bouchaud and Potters (1999), "Noise
dressing of financial correlation matrices", Physical Review Letters 83, 1467; Bouchaud and
Potters (2009), "Financial applications of random matrix theory: a short review".

An optional market-adjusted edge uses ``sigma2 = 1 - lambda_max / N`` (the noise variance left
after removing the market mode). It is available through the ``sigma2`` argument but is not the
headline setting, which is ``sigma2 = 1``.
"""

from __future__ import annotations

import numpy as np

from pfopt.estimators.utils import corr_matrix, mp_edges, validate_returns


def rmt_corr(
    returns: np.ndarray,
    q: float | None = None,
    sigma2: float = 1.0,
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    """RMT-clipped correlation matrix.

    Steps: take the sample correlation ``C`` of the ``T x N`` returns; eigendecompose it;
    keep eigenvalues above ``lambda_+ = sigma2 * (1 + sqrt(q))**2``; replace all the others by
    their mean (so the trace stays ``N``); rebuild the matrix; rescale it to a unit diagonal,
    ``C_clean = D^{-1/2} C_tilde D^{-1/2}`` with ``D = diag(C_tilde)``.

    Parameters
    ----------
    returns : numpy.ndarray
        ``T x N`` array of returns without NaN.
    q : float, optional
        Aspect ratio used for the MP edge. Defaults to ``N / T``. A different value can be passed
        to account for an effective sample size.
    sigma2 : float, default 1.0
        Noise variance of the correlation matrix.
    return_info : bool, default False
        If True, also return a dict with ``n_signal``, ``lam_plus``, ``q``, ``eigenvalues``
        (sample, descending), ``eigenvalues_clipped`` (descending, trace ``N``) and
        ``noise_mean`` (the common value given to noise eigenvalues, NaN if there are none).

    Returns
    -------
    numpy.ndarray or (numpy.ndarray, dict)
        ``N x N`` symmetric positive semi-definite correlation matrix with unit diagonal.

    Raises
    ------
    ValueError
        For invalid input, a zero-variance column, or a degenerate clipped matrix.
    """
    x = validate_returns(returns)
    t, n = x.shape
    q_eff = n / t if q is None else float(q)
    corr = corr_matrix(x)
    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    lam_plus = mp_edges(q_eff, sigma2)[1]
    signal = eigenvalues > lam_plus
    clipped = eigenvalues.copy()
    noise_mean = float("nan")
    if (~signal).any():
        noise_mean = float(eigenvalues[~signal].mean())
        clipped[~signal] = noise_mean
    tilde = (eigenvectors * clipped) @ eigenvectors.T
    diag = np.diag(tilde)
    if np.any(diag <= 0):
        raise ValueError("clipped correlation matrix has a non-positive diagonal")
    scale = 1.0 / np.sqrt(diag)
    clean = tilde * np.outer(scale, scale)
    clean = (clean + clean.T) / 2.0
    np.fill_diagonal(clean, 1.0)
    if not return_info:
        return clean
    info: dict[str, object] = {
        "n_signal": int(signal.sum()),
        "lam_plus": float(lam_plus),
        "q": float(q_eff),
        "eigenvalues": eigenvalues[::-1].copy(),
        "eigenvalues_clipped": clipped[::-1].copy(),
        "noise_mean": noise_mean,
    }
    return clean, info


def rmt_cov(
    returns: np.ndarray,
    q: float | None = None,
    sigma2: float = 1.0,
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    """RMT-denoised covariance matrix.

    Formula: ``Sigma = diag(s) C_clean diag(s)`` where ``s`` are the sample standard deviations
    (``ddof=1``) and ``C_clean`` is the clipped, unit-diagonal correlation matrix from
    :func:`rmt_corr`. The diagonal of ``Sigma`` equals the sample variances.

    Parameters
    ----------
    returns : numpy.ndarray
        ``T x N`` array of returns without NaN.
    q : float, optional
        MP aspect ratio; defaults to ``N / T``.
    sigma2 : float, default 1.0
        Noise variance of the correlation matrix (1 for the headline version).
    return_info : bool, default False
        If True, also return the info dict described in :func:`rmt_corr`.

    Returns
    -------
    numpy.ndarray or (numpy.ndarray, dict)
        ``N x N`` symmetric positive semi-definite covariance matrix.
    """
    clean, info = rmt_corr(returns, q=q, sigma2=sigma2, return_info=True)
    s = np.asarray(returns, dtype=float).std(axis=0, ddof=1)
    cov = clean * np.outer(s, s)
    cov = (cov + cov.T) / 2.0
    if return_info:
        return cov, info
    return cov
