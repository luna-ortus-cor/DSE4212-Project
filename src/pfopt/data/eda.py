"""Exploratory data analysis computations.

Pure functions that turn the processed return matrices into the quantities shown in the EDA
figures and tables. Plotting lives in :mod:`pfopt.viz.plots`. All statistics use raw (not
winsorized) log returns so that fat tails remain visible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from pfopt.data.universe import select_universe
from pfopt.estimators.utils import corr_matrix, mp_edges, n_above_mp_edge


def get_window(
    log_returns: pd.DataFrame,
    asof_date: str | pd.Timestamp,
    lookback_days: int,
    symbols: list[str],
) -> pd.DataFrame:
    """Estimation window: the last ``lookback_days`` log returns up to ``asof_date`` (inclusive).

    Parameters
    ----------
    log_returns : pandas.DataFrame
        Date x Symbol log returns.
    asof_date : str or Timestamp
        A date in the index.
    lookback_days : int
        Window length ``T``.
    symbols : list of str
        Columns to keep (the universe at ``asof_date``).

    Returns
    -------
    pandas.DataFrame
        ``T x len(symbols)`` frame without NaN.

    Raises
    ------
    ValueError
        If the date is missing, history is too short, or the window contains NaN.
    """
    ts = pd.Timestamp(asof_date)
    if ts not in log_returns.index:
        raise ValueError(f"asof_date {ts.date()} is not a date in the returns index")
    pos = int(log_returns.index.get_loc(ts))
    if pos + 1 < lookback_days:
        raise ValueError(f"fewer than {lookback_days} return observations up to {ts.date()}")
    window = log_returns.iloc[pos - lookback_days + 1 : pos + 1][symbols]
    if window.isna().any().any():
        raise ValueError(f"window ending {ts.date()} contains NaN for the requested symbols")
    return window


def return_summary(window: pd.DataFrame, annualization: int) -> pd.DataFrame:
    """Per-asset return statistics over a window.

    Parameters
    ----------
    window : pandas.DataFrame
        ``T x N`` log returns.
    annualization : int
        Trading days per year.

    Returns
    -------
    pandas.DataFrame
        Index = symbols; columns ``ann_mean`` (mean * annualization), ``ann_vol`` (std *
        sqrt(annualization)), ``skew`` and ``excess_kurtosis`` (bias-corrected, from pandas),
        and daily ``min`` and ``max``.
    """
    return pd.DataFrame(
        {
            "ann_mean": window.mean() * annualization,
            "ann_vol": window.std(ddof=1) * np.sqrt(annualization),
            "skew": window.skew(),
            "excess_kurtosis": window.kurt(),
            "min": window.min(),
            "max": window.max(),
        }
    )


def pooled_standardized(window: pd.DataFrame) -> np.ndarray:
    """Pool all returns after standardizing each column to zero mean and unit variance."""
    x = window.to_numpy(dtype=float)
    z = (x - x.mean(axis=0)) / x.std(axis=0, ddof=1)
    return z.ravel()


def pooled_summary(window: pd.DataFrame, tail_sd: float = 4.0) -> dict[str, float]:
    """Summary of all returns in a window pooled across assets, with a fat-tail measure.

    Parameters
    ----------
    window : pandas.DataFrame
        ``T x N`` log returns.
    tail_sd : float, default 4.0
        Threshold (in standard deviations) for the tail-frequency comparison with the normal law.

    Returns
    -------
    dict
        ``n_obs, mean, std, skew, excess_kurtosis, min, max`` of the pooled raw returns, plus
        ``tail_sd``, ``frac_beyond_tail`` (share of standardized returns with ``|z| > tail_sd``) and
        ``normal_frac_beyond_tail`` (the same share under a normal distribution).
    """
    pooled = window.to_numpy(dtype=float).ravel()
    z = pooled_standardized(window)
    return {
        "n_obs": float(pooled.size),
        "mean": float(pooled.mean()),
        "std": float(pooled.std(ddof=1)),
        "skew": float(stats.skew(pooled, bias=False)),
        "excess_kurtosis": float(stats.kurtosis(pooled, fisher=True, bias=False)),
        "min": float(pooled.min()),
        "max": float(pooled.max()),
        "tail_sd": float(tail_sd),
        "frac_beyond_tail": float(np.mean(np.abs(z) > tail_sd)),
        "normal_frac_beyond_tail": float(2.0 * stats.norm.sf(tail_sd)),
    }


def leaf_order(corr: np.ndarray, method: str = "average") -> np.ndarray:
    """Leaf order of a hierarchical clustering on the distance ``sqrt(0.5 * (1 - rho))``.

    Used only to order the correlation heatmap so that correlated stocks sit together.

    Parameters
    ----------
    corr : numpy.ndarray
        ``N x N`` correlation matrix.
    method : str, default "average"
        Linkage method passed to :func:`scipy.cluster.hierarchy.linkage`.

    Returns
    -------
    numpy.ndarray
        Permutation of ``range(N)``.
    """
    dist = np.sqrt(0.5 * np.clip(1.0 - corr, 0.0, None))
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist, 0.0)
    return leaves_list(linkage(squareform(dist, checks=False), method=method))


def spectrum_stats(values: np.ndarray) -> dict[str, object]:
    """Correlation-matrix spectrum of a ``T x N`` window against the Marchenko-Pastur law.

    Parameters
    ----------
    values : numpy.ndarray
        ``T x N`` returns.

    Returns
    -------
    dict
        ``corr`` (matrix), ``eigenvalues`` (descending), ``q`` (= N / T), ``lambda_plus`` (MP
        upper edge), ``n_signal`` (eigenvalues above it), ``lambda_max``, ``top_share``
        (``lambda_max / N``, the market-mode share of total variance) and ``avg_corr`` (mean
        off-diagonal correlation).
    """
    t, n = values.shape
    corr = corr_matrix(values)
    eigenvalues = np.linalg.eigh(corr)[0][::-1]
    q = n / t
    return {
        "corr": corr,
        "eigenvalues": eigenvalues,
        "q": q,
        "lambda_plus": mp_edges(q)[1],
        "n_signal": n_above_mp_edge(eigenvalues, q),
        "lambda_max": float(eigenvalues[0]),
        "top_share": float(eigenvalues[0] / n),
        "avg_corr": float((corr.sum() - n) / (n * (n - 1))),
    }


def spectrum_timeseries(
    log_returns: pd.DataFrame,
    dollar_volume: pd.DataFrame,
    dates: pd.DatetimeIndex,
    lookback_days: int,
    n_assets: int,
    dollar_volume_window: int,
) -> pd.DataFrame:
    """Spectrum diagnostics at each date, using only data up to that date.

    For every date the point-in-time universe is selected, and the correlation spectrum of its
    window is compared with the Marchenko-Pastur upper edge.

    Returns
    -------
    pandas.DataFrame
        Index = ``dates``; columns ``n_signal``, ``lambda_max``, ``lambda_plus``, ``top_share``,
        ``avg_corr``.
    """
    rows = []
    for date in dates:
        universe = select_universe(
            log_returns, dollar_volume, date, lookback_days, n_assets, dollar_volume_window
        )
        window = get_window(log_returns, date, lookback_days, universe)
        s = spectrum_stats(window.to_numpy(dtype=float))
        keys = ("n_signal", "lambda_max", "lambda_plus", "top_share", "avg_corr")
        rows.append({k: s[k] for k in keys})
    return pd.DataFrame(rows, index=pd.DatetimeIndex(dates, name="Date"))


def rolling_avg_correlation(log_returns: pd.DataFrame, window: int) -> pd.Series:
    """Cross-sectional average pairwise correlation over a rolling window.

    At each date the symbols with a complete window (no NaN, non-zero variance) are used, so only
    data up to that date enter. With ``z`` the standardized window returns and ``n`` symbols,
    the mean off-diagonal correlation is ``(|sum_i z_i|^2 / (window - 1) - n) / (n (n - 1))``.

    Parameters
    ----------
    log_returns : pandas.DataFrame
        Date x Symbol log returns.
    window : int
        Rolling window length in trading days (at least 2).

    Returns
    -------
    pandas.Series
        Indexed like ``log_returns``; NaN before the first full window or if fewer than two
        symbols qualify.
    """
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    x = log_returns.to_numpy(dtype=float)
    out = np.full(len(x), np.nan)
    for t in range(window - 1, len(x)):
        block = x[t - window + 1 : t + 1]
        block = block[:, ~np.isnan(block).any(axis=0)]
        sd = block.std(axis=0, ddof=1)
        keep = sd > 0
        n = int(keep.sum())
        if n < 2:
            continue
        z = (block[:, keep] - block[:, keep].mean(axis=0)) / sd[keep]
        s = z.sum(axis=1)
        out[t] = (s @ s / (window - 1) - n) / (n * (n - 1))
    return pd.Series(out, index=log_returns.index, name="avg_corr")


def index_realized_vol(index: pd.DataFrame, window: int, annualization: int) -> pd.Series:
    """Annualized rolling realized volatility of the index from its level series.

    Parameters
    ----------
    index : pandas.DataFrame
        ``Date`` plus one index-level column (output of the index loader).
    window : int
        Number of daily log returns in each window.
    annualization : int
        Trading days per year.

    Returns
    -------
    pandas.Series
        Named ``index_realized_vol``, indexed by date; NaN until a full window of returns exists.
    """
    levels = [c for c in index.columns if c != "Date"]
    if len(levels) != 1:
        raise ValueError(f"index must have exactly one level column besides Date, got {levels}")
    series = index.set_index("Date")[levels[0]].sort_index()
    returns = np.log(series).diff()
    vol = returns.rolling(window=window, min_periods=window).std(ddof=1) * np.sqrt(annualization)
    return vol.rename("index_realized_vol")
