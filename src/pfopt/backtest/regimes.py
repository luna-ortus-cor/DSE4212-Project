"""Subperiod and volatility-regime analysis (PLAN.md step 5.2).

Subperiods are fixed calendar windows (COVID stress, one stress year, and the remaining "calm"
days, which are concatenated across their non-contiguous stretches; drawdowns on that series
ignore the gaps).

Volatility regimes use the S&P 500's trailing realized volatility ``vol_t`` and its *expanding*
median ``m_t`` (the median of all ``vol_s`` with ``s <= t``, never the full-sample median):

- raw label at ``t``: "high" if ``vol_t > m_t`` else "low", undefined until ``min_history``
  volatility observations exist;
- the label of day ``d`` is the raw label of the previous trading day, so a day is never classified
  with information that includes its own return.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from pfopt.backtest.metrics import performance_metrics


def period_masks(
    index: pd.DatetimeIndex, covid_start: str, covid_end: str, year: int
) -> dict[str, np.ndarray]:
    """Boolean masks over ``index`` for the full sample and the subperiods.

    Parameters
    ----------
    index : pandas.DatetimeIndex
        Return dates.
    covid_start, covid_end : str
        Inclusive ISO bounds of the COVID window.
    year : int
        Calendar year of the second stress period.

    Returns
    -------
    dict
        Keys ``full``, ``covid``, ``year_<year>`` and ``calm`` (neither COVID nor the stress
        year). The last three are disjoint and together cover ``full``.

    Raises
    ------
    ValueError
        If the COVID window and the stress year share a date in ``index``.
    """
    covid = np.asarray((index >= pd.Timestamp(covid_start)) & (index <= pd.Timestamp(covid_end)))
    in_year = np.asarray(index.year == year)
    if (covid & in_year).any():
        raise ValueError("the COVID window and the stress year overlap")
    return {
        "full": np.ones(len(index), dtype=bool),
        "covid": covid,
        f"year_{year}": in_year,
        "calm": ~covid & ~in_year,
    }


def expanding_median_regime(vol: pd.Series, min_history: int) -> pd.Series:
    """Raw regime label at each date from the expanding median of ``vol``.

    Parameters
    ----------
    vol : pandas.Series
        Trailing realized volatility (NaN where unavailable).
    min_history : int
        Minimum number of non-NaN volatility observations before labelling starts.

    Returns
    -------
    pandas.Series
        "high" if ``vol_t`` exceeds the median of ``vol_s`` for ``s <= t``, "low" otherwise, NaN
        where the median or the volatility is unavailable. Uses only data up to ``t``.
    """
    median = vol.expanding(min_periods=min_history).median()
    labels = pd.Series(np.where(vol > median, "high", "low"), index=vol.index, dtype=object)
    labels[median.isna() | vol.isna()] = np.nan
    return labels


def regime_labels_for_returns(
    vol: pd.Series, return_index: pd.DatetimeIndex, min_history: int
) -> pd.Series:
    """Regime label for each return date: the raw label of the previous trading day.

    Parameters
    ----------
    vol : pandas.Series
        Trailing realized volatility of the index, indexed by date.
    return_index : pandas.DatetimeIndex
        Trading calendar of the strategy returns.
    min_history : int
        See :func:`expanding_median_regime`.
    """
    raw = expanding_median_regime(vol, min_history)
    return raw.reindex(return_index).shift(1)


def regime_masks(labels: pd.Series) -> dict[str, np.ndarray]:
    """Boolean masks ``high_vol`` and ``low_vol`` from labels (unclassified days are in neither)."""
    return {
        "high_vol": (labels == "high").to_numpy(),
        "low_vol": (labels == "low").to_numpy(),
    }


def metrics_by_period(
    returns: pd.DataFrame,
    masks: Mapping[str, np.ndarray],
    annualization: int = 252,
    rf: float = 0.0,
) -> pd.DataFrame:
    """Performance metrics of every strategy within every period.

    Parameters
    ----------
    returns : pandas.DataFrame
        Date x strategy daily returns.
    masks : mapping of str to numpy.ndarray
        Boolean masks over ``returns.index``, one per period.
    annualization : int, default 252
        Trading days per year.
    rf : float, default 0.0
        Annual risk-free rate for the Sharpe ratio.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``(period, strategy)``; columns as in
        :func:`pfopt.backtest.metrics.performance_metrics` (``annual_turnover`` is NaN).

    Raises
    ------
    ValueError
        If a period contains fewer than two days.
    """
    rows = {}
    for period, mask in masks.items():
        sub = returns.loc[np.asarray(mask)]
        if len(sub) < 2:
            raise ValueError(f"period {period!r} has fewer than 2 observations")
        for strategy in returns.columns:
            rows[(period, strategy)] = performance_metrics(
                sub[strategy], None, annualization, rf
            )
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index = pd.MultiIndex.from_tuples(table.index, names=["period", "strategy"])
    return table
