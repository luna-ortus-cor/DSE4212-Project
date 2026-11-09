"""Residual-structure diagnostic: sector blocks left by the PCA factors (PLAN.md step 8.23).

The PCA-factor and RMT estimators keep ``K`` components (the eigenvalues above the Marchenko-Pastur
edge) and replace the rest of the sample correlation matrix by a diagonal (PCA) or by its average
(RMT). If sectors still drive the correlation of the discarded part, that part is not noise.

For an estimation window with sample correlation matrix ``C`` and eigenpairs ``(l_i, v_i)``:

    S = sum_{i <= K} l_i v_i v_i',       R = C - S,       E = D^{-1/2} R D^{-1/2},  D = diag(R),

so ``E`` is the discarded part rescaled to a unit diagonal (``R`` is exactly orthogonal to the
retained eigenvectors; the rescaling only changes the scale of each asset). With static sector
labels the diagnostics are

* the mean correlation of same-sector pairs minus that of different-sector pairs (the *gap*), for
  ``C`` and for ``E``;
* a one-sided permutation p-value of the gap of ``E`` (sector labels shuffled among the members,
  which keeps the group sizes);
* the number of eigenvalues of ``E`` above the Marchenko-Pastur upper edge ``(1 + sqrt(N / T))^2``
  (the edge uses the full ``N``; ``E`` has rank ``N - K``, so this is approximate), and the share
  of those eigenvectors that lies in the span of the sector indicators, to be compared with the
  value ``G / n`` of a random direction (``G`` sectors, ``n`` members with a known sector).

Sector labels are the static snapshot of ``sp500_companies.csv``, used as labels only (never to
select assets). Members with an unknown sector are left out of the sector quantities. Everything
uses data dated ``<= t`` only.

References: Marchenko and Pastur (1967); Laloux, Cizeau, Bouchaud and Potters (1999), "Noise
dressing of financial correlation matrices"; Plerou et al. (2002), for sector structure in the
bulk and the edge of the empirical spectrum.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pfopt.backtest.structure import UNKNOWN_SECTOR
from pfopt.data.eda import get_window
from pfopt.estimators.utils import corr_matrix, mp_edges, n_above_mp_edge, winsorize_cols

UNKNOWN_CODE = -1
_MIN_RESIDUAL_VARIANCE = 1e-12
_SEED_TAG = 23


def sector_codes(symbols: list[str], sectors: pd.Series, categories: list[str]) -> np.ndarray:
    """Integer sector code of every symbol (``-1`` if the sector is missing or unknown).

    Parameters
    ----------
    symbols : list of str
        Symbols, in the order of the correlation matrix.
    sectors : pandas.Series
        Sector label indexed by symbol.
    categories : list of str
        All sector names; the code of a symbol is the position of its sector in this list.
    """
    position = {name: i for i, name in enumerate(categories)}
    labels = sectors.reindex(symbols)
    return np.array(
        [position.get(s, UNKNOWN_CODE) if isinstance(s, str) else UNKNOWN_CODE for s in labels],
        dtype=int,
    )


def pca_residual_corr(corr: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Part of a correlation matrix left after removing its top ``k`` principal components.

    Parameters
    ----------
    corr : numpy.ndarray
        ``N x N`` sample correlation matrix.
    k : int
        Number of retained components, ``1 <= k < N``.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        ``E`` (the residual ``C - V_k L_k V_k'`` rescaled to a unit diagonal) and the unscaled
        residual ``R``.

    Raises
    ------
    ValueError
        If ``k`` is out of range or a residual variance is not positive.
    """
    n = corr.shape[0]
    if not 1 <= k < n:
        raise ValueError(f"k must be in [1, {n - 1}], got {k}")
    values, vectors = np.linalg.eigh(corr)
    top = vectors[:, ::-1][:, :k]
    systematic = (top * values[::-1][:k]) @ top.T
    resid = corr - systematic
    resid = (resid + resid.T) / 2.0
    d = np.diag(resid).copy()
    if np.any(d <= _MIN_RESIDUAL_VARIANCE):
        raise ValueError("a residual variance is not positive; reduce k")
    scale = 1.0 / np.sqrt(d)
    e = resid * np.outer(scale, scale)
    np.fill_diagonal(e, 1.0)
    return (e + e.T) / 2.0, resid


def _offdiagonal_known(matrix: np.ndarray, codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    known = codes != UNKNOWN_CODE
    sub = matrix[np.ix_(known, known)].copy()
    np.fill_diagonal(sub, 0.0)
    return sub, codes[known]


def _group_sums(sub: np.ndarray, codes: np.ndarray, n_groups: int) -> tuple[np.ndarray, np.ndarray]:
    indicator = np.eye(n_groups)[codes]
    return indicator.T @ sub @ indicator, indicator.sum(axis=0)


def _gap_from_sums(sums: np.ndarray, sizes: np.ndarray) -> tuple[float, float]:
    pairs_within = float((sizes * (sizes - 1.0)).sum())
    pairs_between = float(sizes.sum() ** 2 - (sizes**2).sum())
    if pairs_within <= 0 or pairs_between <= 0:
        raise ValueError("need at least one same-sector pair and one cross-sector pair")
    within_sum = float(np.trace(sums))
    return within_sum / pairs_within, (float(sums.sum()) - within_sum) / pairs_between


def sector_gap(matrix: np.ndarray, codes: np.ndarray, n_groups: int) -> tuple[float, float]:
    """Mean off-diagonal entry for same-sector pairs and for different-sector pairs.

    Members with code ``-1`` are ignored.

    Returns
    -------
    (float, float)
        ``(within, between)``.
    """
    sub, known = _offdiagonal_known(matrix, codes)
    sums, sizes = _group_sums(sub, known, n_groups)
    return _gap_from_sums(sums, sizes)


def permutation_gap_test(
    matrix: np.ndarray, codes: np.ndarray, n_groups: int, n_perm: int, rng: np.random.Generator
) -> dict[str, float]:
    """One-sided permutation test of the within-minus-between sector gap.

    The labels of the members with a known sector are shuffled ``n_perm`` times (group sizes are
    kept). The p-value is ``(1 + #{null gap >= observed}) / (n_perm + 1)``.

    Returns
    -------
    dict
        ``gap``, ``p_value``, ``null_mean`` and ``null_sd`` of the permuted gaps.
    """
    sub, known = _offdiagonal_known(matrix, codes)
    sums, sizes = _group_sums(sub, known, n_groups)
    within, between = _gap_from_sums(sums, sizes)
    gap = within - between
    null = np.empty(n_perm)
    for i in range(n_perm):
        s, z = _group_sums(sub, rng.permutation(known), n_groups)
        w, b = _gap_from_sums(s, z)
        null[i] = w - b
    return {
        "gap": gap,
        "p_value": float((1 + np.sum(null >= gap)) / (n_perm + 1)),
        "null_mean": float(null.mean()),
        "null_sd": float(null.std(ddof=1)),
    }


def sector_share(vectors: np.ndarray, codes: np.ndarray, n_groups: int) -> np.ndarray:
    """Share of each column's squared norm lying in the span of the sector indicators.

    Members with code ``-1`` are dropped and each column is restricted to the rest. A column that
    is constant within every sector has share 1; a random direction has about ``G / n``.

    Parameters
    ----------
    vectors : numpy.ndarray
        ``N x m`` array, one eigenvector per column.
    codes : numpy.ndarray
        Sector code of each of the ``N`` members.
    n_groups : int
        Number of sector categories.
    """
    known = codes != UNKNOWN_CODE
    v = vectors[known]
    indicator = np.eye(n_groups)[codes[known]]
    sizes = indicator.sum(axis=0)
    present = sizes > 0
    projected = (indicator.T @ v)[present] ** 2 / sizes[present, None]
    return projected.sum(axis=0) / (v**2).sum(axis=0)


def block_matrix(matrix: np.ndarray, codes: np.ndarray, n_groups: int) -> np.ndarray:
    """Mean off-diagonal entry for every pair of sectors (``NaN`` for empty cells)."""
    sub, known = _offdiagonal_known(matrix, codes)
    sums, sizes = _group_sums(sub, known, n_groups)
    pairs = np.outer(sizes, sizes)
    np.fill_diagonal(pairs, sizes * (sizes - 1.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(pairs > 0, sums / pairs, np.nan)
    return out


def window_residual_diagnostics(
    x: np.ndarray, codes: np.ndarray, n_groups: int, n_perm: int, rng: np.random.Generator
) -> tuple[dict[str, float], np.ndarray]:
    """All diagnostics of one estimation window.

    Parameters
    ----------
    x : numpy.ndarray
        ``T x N`` returns of the window (already winsorized).
    codes : numpy.ndarray
        Sector code per column (``-1`` = unknown).
    n_groups : int
        Number of sector categories.
    n_perm : int
        Permutations for the residual-gap test.
    rng : numpy.random.Generator
        Source of the permutations.

    Returns
    -------
    (dict, numpy.ndarray)
        The scalar diagnostics and the ``n_groups x n_groups`` sector-block matrix of ``E``.
    """
    t, n = x.shape
    corr = corr_matrix(x)
    values = np.linalg.eigvalsh(corr)[::-1]
    k = max(1, n_above_mp_edge(values, n / t))
    e, _ = pca_residual_corr(corr, k)
    lam_plus = mp_edges(n / t)[1]
    e_values, e_vectors = np.linalg.eigh(e)
    signal = e_values > lam_plus
    raw_within, raw_between = sector_gap(corr, codes, n_groups)
    test = permutation_gap_test(e, codes, n_groups, n_perm, rng)
    res_within, res_between = sector_gap(e, codes, n_groups)
    known = codes != UNKNOWN_CODE
    sizes = np.bincount(codes[known], minlength=n_groups)
    share = (
        float(sector_share(e_vectors[:, signal], codes, n_groups).mean())
        if signal.any()
        else np.nan
    )
    row = {
        "n_members": float(n),
        "n_known_sector": float(known.sum()),
        "k_factors": float(k),
        "raw_within": raw_within,
        "raw_between": raw_between,
        "raw_gap": raw_within - raw_between,
        "resid_within": res_within,
        "resid_between": res_between,
        "resid_gap": test["gap"],
        "perm_null_mean": test["null_mean"],
        "perm_null_sd": test["null_sd"],
        "perm_z": (test["gap"] - test["null_mean"]) / test["null_sd"],
        "perm_p": test["p_value"],
        "resid_n_signal": float(signal.sum()),
        "resid_top_eig": float(e_values[-1]),
        "resid_sector_share": share,
        "sector_share_null": float((sizes > 0).sum() / known.sum()),
    }
    return row, block_matrix(e, codes, n_groups)


def residual_diagnostics(
    log_returns: pd.DataFrame,
    universe: pd.DataFrame,
    sectors: pd.Series,
    lookback_days: int,
    winsor_lower: float,
    winsor_upper: float,
    n_perm: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Residual-structure diagnostics for the estimation window of every rebalance date.

    Parameters
    ----------
    log_returns : pandas.DataFrame
        Date x Symbol log returns.
    universe : pandas.DataFrame
        Boolean rebalance date x symbol membership (as in the backtest).
    sectors : pandas.Series
        Static sector label indexed by symbol.
    lookback_days, winsor_lower, winsor_upper
        Window length and per-window winsorization quantiles used for the weights.
    n_perm : int
        Permutations per window.
    seed : int
        Global seed; window ``i`` uses ``default_rng([seed, 23, i])``.

    Returns
    -------
    (pandas.DataFrame, pandas.DataFrame)
        One row of scalar diagnostics per rebalance date, and the sector-by-sector residual
        correlation averaged over the windows.
    """
    members = sorted({s for s in sectors.dropna().unique() if s != UNKNOWN_SECTOR})
    rows, blocks = [], []
    for i, (date, flags) in enumerate(universe.iterrows()):
        symbols = list(universe.columns[flags.to_numpy(dtype=bool)])
        window = get_window(log_returns, date, lookback_days, symbols)
        x = winsorize_cols(window.to_numpy(), winsor_lower, winsor_upper)
        codes = sector_codes(symbols, sectors, members)
        rng = np.random.default_rng([seed, _SEED_TAG, i])
        row, block = window_residual_diagnostics(x, codes, len(members), n_perm, rng)
        rows.append(row)
        blocks.append(block.ravel())
    frame = pd.DataFrame(rows, index=universe.index)
    mean_block = pd.DataFrame(blocks).mean(axis=0, skipna=True).to_numpy()
    average = pd.DataFrame(
        mean_block.reshape(len(members), len(members)), index=members, columns=members
    )
    return frame, average
