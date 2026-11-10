"""Hierarchical Risk Parity (HRP) allocation.

Algorithm (Lopez de Prado, 2016, "Building diversified portfolios that outperform out of sample",
Journal of Portfolio Management 42(4), 59-69):

1. Distance ``d_ij = sqrt(0.5 * (1 - rho_ij))`` from the correlation matrix.
2. Hierarchical clustering of the condensed distance matrix (``scipy.cluster.hierarchy``).
3. Quasi-diagonalization: order the assets by the dendrogram so that similar assets are adjacent.
4. Recursive bisection: split the ordered list in half; with cluster variance
   ``V = w' Sigma_c w`` under inverse-variance weights ``w`` inside the cluster, give the first half
   the share ``alpha = 1 - V_1 / (V_1 + V_2)`` and the second half ``1 - alpha``; recurse down to
   single assets.

The covariance matrix is never inverted: only its diagonal is.

Leaf order. The bisection cuts the ordered list by position, so the weights depend on the order of
the leaves. scipy's order depends on the input order of the assets, which makes the result vary
when the same assets are shuffled. By default this module therefore uses a canonical order that
depends only on the data: at every merge the larger cluster comes first, ties broken by the mean
variance of the cluster's assets. Pass ``canonical_order=False`` for the scipy order.
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from pfopt.estimators.utils import cov_to_corr
from pfopt.weights import check_weights

LINKAGE_METHODS = ("single", "complete", "average", "ward")


def _validate_cov(cov: np.ndarray) -> np.ndarray:
    a = np.asarray(cov, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] < 1:
        raise ValueError("cov must be a non-empty square matrix")
    if not np.isfinite(a).all():
        raise ValueError("cov must be finite")
    if not np.allclose(a, a.T, rtol=1e-8, atol=1e-12 * float(np.abs(a).max())):
        raise ValueError("cov must be symmetric")
    if np.any(np.diag(a) <= 0):
        raise ValueError("cov must have strictly positive variances")
    return (a + a.T) / 2.0


def _canonical_leaf_order(z: np.ndarray, variances: np.ndarray) -> list[int]:
    """Leaf order from a linkage matrix with children ordered by (size desc, mean variance)."""
    n = z.shape[0] + 1
    order: dict[int, list[int]] = {i: [i] for i in range(n)}
    var_sum: dict[int, float] = {i: float(variances[i]) for i in range(n)}
    for step, row in enumerate(z):
        a, b = int(row[0]), int(row[1])
        key_a = (-len(order[a]), var_sum[a] / len(order[a]))
        key_b = (-len(order[b]), var_sum[b] / len(order[b]))
        first, second = (a, b) if key_a <= key_b else (b, a)
        node = n + step
        order[node] = order.pop(first) + order.pop(second)
        var_sum[node] = var_sum.pop(first) + var_sum.pop(second)
    return order[2 * n - 2]


def _cluster_variance(cov: np.ndarray, idx: np.ndarray) -> float:
    """Variance of a cluster under inverse-variance weights."""
    sub = cov[np.ix_(idx, idx)]
    ivp = 1.0 / np.diag(sub)
    ivp /= ivp.sum()
    return float(ivp @ sub @ ivp)


def hrp_weights(
    cov: np.ndarray,
    corr: np.ndarray | None = None,
    linkage_method: str = "single",
    canonical_order: bool = True,
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    """Hierarchical Risk Parity weights.

    Parameters
    ----------
    cov : numpy.ndarray
        ``N x N`` symmetric covariance matrix with positive variances. It is used for the cluster
        variances in the recursive bisection (and need not be positive definite).
    corr : numpy.ndarray, optional
        ``N x N`` correlation matrix used for the clustering distance. Defaults to the correlation
        implied by ``cov``. Pass an RMT-cleaned correlation (with an RMT covariance) for HRP-RMT.
    linkage_method : {"single", "complete", "average", "ward"}, default "single"
        Linkage rule for :func:`scipy.cluster.hierarchy.linkage`.
    canonical_order : bool, default True
        Use the data-only leaf order described in the module docstring. If False, use scipy's
        ``leaves_list`` order (which can depend on the input order of the assets).
    return_info : bool, default False
        If True, also return a dict with ``order`` (leaf order) and ``linkage`` (scipy matrix).

    Returns
    -------
    numpy.ndarray or (numpy.ndarray, dict)
        Weights aligned with the rows of ``cov``: non-negative and summing to 1.

    Raises
    ------
    ValueError
        For an invalid covariance or correlation matrix, or an unknown linkage method.
    """
    if linkage_method not in LINKAGE_METHODS:
        raise ValueError(f"linkage_method must be in {LINKAGE_METHODS}, got {linkage_method!r}")
    sigma = _validate_cov(cov)
    n = sigma.shape[0]
    if corr is None:
        rho = cov_to_corr(sigma)
    else:
        rho = np.asarray(corr, dtype=float)
        if rho.shape != sigma.shape:
            raise ValueError("corr must have the same shape as cov")
        if not np.isfinite(rho).all():
            raise ValueError("corr must be finite")
        rho = np.clip((rho + rho.T) / 2.0, -1.0, 1.0)
        np.fill_diagonal(rho, 1.0)

    if n == 1:
        weights = np.ones(1)
        info: dict[str, object] = {"order": [0], "linkage": np.zeros((0, 4))}
        return (weights, info) if return_info else weights

    dist = np.sqrt(0.5 * (1.0 - rho))
    np.fill_diagonal(dist, 0.0)
    z = linkage(squareform(dist, checks=False), method=linkage_method)
    if canonical_order:
        order = _canonical_leaf_order(z, np.diag(sigma))
    else:
        order = [int(i) for i in leaves_list(z)]

    weights = np.ones(n)
    clusters = [np.asarray(order)]
    while clusters:
        next_clusters = []
        for cluster in clusters:
            if cluster.size < 2:
                continue
            half = cluster.size // 2
            left, right = cluster[:half], cluster[half:]
            v_left = _cluster_variance(sigma, left)
            v_right = _cluster_variance(sigma, right)
            alpha = 1.0 - v_left / (v_left + v_right)
            weights[left] *= alpha
            weights[right] *= 1.0 - alpha
            next_clusters += [left, right]
        clusters = next_clusters

    check_weights(weights)
    if return_info:
        return weights, {"order": order, "linkage": z}
    return weights
