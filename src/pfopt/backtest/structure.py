"""Weight and portfolio structure analysis (PLAN.md step 5.4).

Concentration (Herfindahl index, effective number of holdings, pooled weight distribution),
sector exposure from a static sector label, and the link between the correlation-spectrum
diagnostics of each estimation window (market-mode share, mean correlation, RMT signal count) and
the realized performance of the denoised / shrunk strategies relative to sample GMV.

The sector file is a *current* snapshot: it is used only as a static label, never to select
assets. Window diagnostics are explanatory and use the same data as the weights (``<= t``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from pfopt.backtest.covquality import weight_stats
from pfopt.data.eda import get_window
from pfopt.estimators.utils import corr_matrix, winsorize_cols

UNKNOWN_SECTOR = "Unknown"


def herfindahl_series(weights: pd.DataFrame) -> pd.Series:
    """Herfindahl-Hirschman index ``sum(w^2)`` of every rebalance row."""
    return (weights**2).sum(axis=1).rename("hhi")


def effective_n_table(weights: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Rebalance date x strategy table of the effective number of holdings ``1 / HHI``."""
    return pd.DataFrame({s: 1.0 / herfindahl_series(w) for s, w in weights.items()})


def pooled_member_weights(weights: pd.DataFrame, universe: pd.DataFrame) -> np.ndarray:
    """Weights of universe members only, pooled over the rebalance dates."""
    mask = universe.reindex(index=weights.index, columns=weights.columns).to_numpy(dtype=bool)
    return weights.to_numpy(dtype=float)[mask]


def concentration_table(
    weights: Mapping[str, pd.DataFrame],
    universe: pd.DataFrame,
    position_threshold: float = 1e-3,
) -> pd.DataFrame:
    """Strategy x concentration statistics, averaged over rebalance dates.

    Columns: ``mean_hhi``, ``mean_effective_n``, ``mean_max_weight``, ``mean_top10_share``,
    ``mean_n_positions`` (weights above ``position_threshold``), ``share_zero`` (fraction of
    universe-member holdings at or below the threshold) and the pooled weight quantiles ``p50``,
    ``p90``, ``p99`` and ``max`` over member holdings.
    """
    rows = {}
    for strategy, frame in weights.items():
        per_date = pd.DataFrame(
            [weight_stats(row, position_threshold) for row in frame.to_numpy(dtype=float)]
        )
        pooled = pooled_member_weights(frame, universe)
        rows[strategy] = {
            "mean_hhi": float(herfindahl_series(frame).mean()),
            "mean_effective_n": float(per_date["effective_n"].mean()),
            "mean_max_weight": float(per_date["max_weight"].mean()),
            "mean_top10_share": float(per_date["top_k_share"].mean()),
            "mean_n_positions": float(per_date["n_positions"].mean()),
            "share_zero": float(np.mean(pooled <= position_threshold)),
            "p50": float(np.quantile(pooled, 0.50)),
            "p90": float(np.quantile(pooled, 0.90)),
            "p99": float(np.quantile(pooled, 0.99)),
            "max": float(pooled.max()),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def sector_exposure(weights: pd.DataFrame, sectors: pd.Series) -> pd.DataFrame:
    """Rebalance date x sector total weight; symbols without a sector go to ``Unknown``.

    Parameters
    ----------
    weights : pandas.DataFrame
        Rebalance date x symbol weights.
    sectors : pandas.Series
        Sector label indexed by symbol (a static snapshot).
    """
    labels = sectors.reindex(weights.columns).fillna(UNKNOWN_SECTOR)
    return weights.T.groupby(labels.to_numpy()).sum().T


def mean_sector_table(weights: Mapping[str, pd.DataFrame], sectors: pd.Series) -> pd.DataFrame:
    """Strategy x sector weight averaged over rebalance dates (rows sum to 1)."""
    rows = {s: sector_exposure(w, sectors).mean(axis=0) for s, w in weights.items()}
    return pd.DataFrame(rows).T.fillna(0.0)


def window_eigen_diagnostics(
    log_returns: pd.DataFrame,
    universe: pd.DataFrame,
    lookback_days: int,
    winsor_lower: float,
    winsor_upper: float,
) -> pd.DataFrame:
    """Spectrum summary of the estimation window behind each rebalance.

    Parameters
    ----------
    log_returns : pandas.DataFrame
        Date x Symbol log returns.
    universe : pandas.DataFrame
        Boolean rebalance date x symbol membership.
    lookback_days : int
        Window length ``T``.
    winsor_lower, winsor_upper : float
        Per-window winsorization quantiles (as used for the weights).

    Returns
    -------
    pandas.DataFrame
        One row per rebalance date: ``top_eig_share`` (largest correlation eigenvalue divided by
        ``N``, the market-mode share) and ``mean_corr`` (average off-diagonal correlation).
    """
    rows = []
    for date, flags in universe.iterrows():
        symbols = list(universe.columns[flags.to_numpy(dtype=bool)])
        window = get_window(log_returns, date, lookback_days, symbols)
        corr = corr_matrix(winsorize_cols(window.to_numpy(), winsor_lower, winsor_upper))
        n = corr.shape[0]
        rows.append(
            {
                "top_eig_share": float(np.linalg.eigvalsh(corr)[-1] / n),
                "mean_corr": float((corr.sum() - n) / (n * (n - 1))),
            }
        )
    return pd.DataFrame(rows, index=universe.index)


def relative_performance(
    net: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    baseline: str,
    others: Sequence[str],
    annualization: int = 252,
) -> pd.DataFrame:
    """Realized volatility and return of each strategy relative to a baseline, per holding period.

    Holding period ``k`` covers the net returns dated in ``(d_k, d_{k+1}]``; the final period
    (no later rebalance) is dropped. For each ``other`` strategy the columns are
    ``logvol_<other>`` (``log`` of its annualized realized volatility over the baseline's;
    negative means lower risk) and ``ret_<other>`` (annualized mean return difference).

    Parameters
    ----------
    net : pandas.DataFrame
        Daily net returns, one column per strategy.
    rebalance_dates : pandas.DatetimeIndex
        Rebalance dates ``d_k``.
    baseline : str
        Reference strategy column.
    others : sequence of str
        Strategy columns compared with the baseline.
    annualization : int, default 252
        Trading days per year.
    """
    rows = {}
    for start, end in zip(rebalance_dates[:-1], rebalance_dates[1:], strict=True):
        block = net.loc[(net.index > start) & (net.index <= end)]
        if len(block) < 2:
            continue
        vol = block.std(ddof=1) * np.sqrt(annualization)
        mean = block.mean() * annualization
        row = {}
        for other in others:
            row[f"logvol_{other}"] = float(np.log(vol[other] / vol[baseline]))
            row[f"ret_{other}"] = float(mean[other] - mean[baseline])
        rows[start] = row
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("Date")


def diagnostic_correlations(diagnostics: pd.DataFrame, performance: pd.DataFrame) -> pd.DataFrame:
    """Spearman rank correlations between window diagnostics and relative performance.

    Rows are (diagnostic, target) pairs over the dates both frames share. The windows overlap
    heavily, so the diagnostics are strongly autocorrelated and the p-values are optimistic;
    treat them as descriptive.
    """
    index = diagnostics.index.intersection(performance.index)
    rows = []
    for diag in diagnostics.columns:
        for target in performance.columns:
            result = stats.spearmanr(diagnostics.loc[index, diag], performance.loc[index, target])
            rows.append(
                {
                    "diagnostic": diag,
                    "target": target,
                    "spearman_rho": float(result.statistic),
                    "p_value": float(result.pvalue),
                    "n": len(index),
                }
            )
    return pd.DataFrame(rows)
