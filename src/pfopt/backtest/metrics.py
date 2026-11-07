"""Out-of-sample performance metrics (PLAN.md step 4.3).

All metrics are computed from daily simple returns. Definitions (``n`` days, ``A`` = annualization
factor, 252 by default; ratios with a zero denominator are NaN):

- annualized return (geometric): ``(prod(1 + r)) ** (A / n) - 1``
- annualized volatility: ``std(r, ddof=1) * sqrt(A)``
- Sharpe ratio: ``mean(r - rf / A) / std(r, ddof=1) * sqrt(A)``
- Sortino ratio: ``mean(r - MAR) / DD * sqrt(A)`` with ``DD = sqrt(mean(min(r - MAR, 0) ** 2))``
  (target semideviation over all ``n`` days) and ``MAR = 0``
- maximum drawdown: ``min(W_t / max(1, W_s, s <= t) - 1)`` with wealth ``W_t = prod(1 + r)``
  starting from 1 before the first return (a negative number)
- Calmar ratio: annualized return divided by ``|maximum drawdown|``
- skewness and excess kurtosis: bias-corrected (pandas)
- worst day: ``min(r)``; hit rate: share of days with ``r > 0``
- 95% historical CVaR: minus the mean of the daily returns at or below the 5% quantile (a positive
  loss when the tail is negative)
- annual turnover: sum of ``sum |dw|`` over rebalances (excluding the initial investment unless
  requested) divided by the number of years ``n / A``
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pfopt.backtest.engine import SimulationResult

_ZERO_TOL = 1e-14  # daily standard deviations below this are treated as zero (constant series)

METRIC_NAMES = (
    "n_days",
    "ann_return",
    "ann_vol",
    "sharpe",
    "sortino",
    "max_drawdown",
    "calmar",
    "skewness",
    "excess_kurtosis",
    "worst_day",
    "hit_rate",
    "cvar_95",
    "annual_turnover",
)


@dataclass(frozen=True)
class DrawdownInfo:
    """Maximum drawdown and its dates.

    ``peak`` is NaT if the drawdown starts from the initial wealth (before the first return),
    ``recovery`` is NaT if wealth never regains the peak, and all three dates are NaT when there
    is no drawdown.
    """

    max_drawdown: float
    peak: pd.Timestamp
    trough: pd.Timestamp
    recovery: pd.Timestamp


def _validated(returns: pd.Series) -> pd.Series:
    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series")
    if len(returns) < 2:
        raise ValueError("at least 2 daily returns are required")
    if not np.isfinite(returns.to_numpy(dtype=float)).all():
        raise ValueError("returns must be finite")
    if (returns <= -1.0).any():
        raise ValueError("returns must be greater than -1")
    return returns.astype(float)


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Drawdown ``W_t / max(1, W_s, s <= t) - 1`` for each day (zero or negative)."""
    r = _validated(returns)
    wealth = (1.0 + r).cumprod().to_numpy()
    running_max = np.maximum.accumulate(np.concatenate([[1.0], wealth]))[1:]
    return pd.Series(wealth / running_max - 1.0, index=r.index, name="drawdown")


def max_drawdown(returns: pd.Series) -> DrawdownInfo:
    """Maximum drawdown with the dates of its peak, trough and recovery."""
    r = _validated(returns)
    wealth = (1.0 + r).cumprod().to_numpy()
    drawdown = drawdown_series(r).to_numpy()
    trough = int(np.argmin(drawdown))
    depth = float(drawdown[trough])
    if depth == 0.0:
        return DrawdownInfo(0.0, pd.NaT, pd.NaT, pd.NaT)
    peak_value = max(1.0, float(wealth[: trough + 1].max()))
    peaks = np.flatnonzero(wealth[: trough + 1] == peak_value)
    recovered = np.flatnonzero(wealth[trough + 1 :] >= peak_value)
    return DrawdownInfo(
        max_drawdown=depth,
        peak=r.index[peaks[-1]] if peaks.size else pd.NaT,
        trough=r.index[trough],
        recovery=r.index[trough + 1 + recovered[0]] if recovered.size else pd.NaT,
    )


def annual_turnover(
    turnover: pd.Series, n_days: int, annualization: int = 252, include_initial: bool = False
) -> float:
    """Turnover (``sum |dw|`` per rebalance) per year.

    Parameters
    ----------
    turnover : pandas.Series
        Turnover at each rebalance; the first entry is the initial investment.
    n_days : int
        Number of days in the return sample.
    annualization : int, default 252
        Trading days per year.
    include_initial : bool, default False
        Whether to count the initial investment from cash.
    """
    values = turnover.to_numpy(dtype=float)
    total = values.sum() if include_initial else values[1:].sum()
    return float(total / (n_days / annualization))


def performance_metrics(
    returns: pd.Series,
    turnover: pd.Series | None = None,
    annualization: int = 252,
    rf: float = 0.0,
    mar: float = 0.0,
    cvar_level: float = 0.95,
) -> dict[str, float]:
    """Performance metrics of a daily simple-return series.

    Parameters
    ----------
    returns : pandas.Series
        Daily simple returns (finite, above -1), at least two.
    turnover : pandas.Series, optional
        Turnover per rebalance (see :func:`annual_turnover`); ``annual_turnover`` is NaN if omitted.
    annualization : int, default 252
        Trading days per year.
    rf : float, default 0.0
        Annual risk-free rate used in the Sharpe ratio (divided by ``annualization`` per day).
    mar : float, default 0.0
        Daily minimum acceptable return for the Sortino ratio.
    cvar_level : float, default 0.95
        Confidence level of the historical CVaR.

    Returns
    -------
    dict
        Keys in :data:`METRIC_NAMES`.
    """
    series = _validated(returns)
    r = series.to_numpy()
    n = r.size
    mean, std = float(r.mean()), float(r.std(ddof=1))
    ann_return = float(np.prod(1.0 + r) ** (annualization / n) - 1.0)
    sharpe = np.nan
    if std > _ZERO_TOL:
        sharpe = (mean - rf / annualization) / std * np.sqrt(annualization)
    downside = float(np.sqrt(np.mean(np.minimum(r - mar, 0.0) ** 2)))
    sortino = np.nan
    if downside > _ZERO_TOL:
        sortino = (mean - mar) / downside * np.sqrt(annualization)
    depth = max_drawdown(series).max_drawdown
    calmar = ann_return / abs(depth) if depth < 0 else np.nan
    threshold = np.quantile(r, 1.0 - cvar_level)
    cvar = -float(r[r <= threshold].mean())
    return {
        "n_days": float(n),
        "ann_return": ann_return,
        "ann_vol": std * float(np.sqrt(annualization)),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": depth,
        "calmar": float(calmar),
        "skewness": float(series.skew()),
        "excess_kurtosis": float(series.kurt()),
        "worst_day": float(r.min()),
        "hit_rate": float(np.mean(r > 0)),
        "cvar_95": cvar,
        "annual_turnover": (
            annual_turnover(turnover, n, annualization) if turnover is not None else np.nan
        ),
    }


def metrics_table(
    simulations: Mapping[str, SimulationResult],
    which: str = "net",
    annualization: int = 252,
    rf: float = 0.0,
    mar: float = 0.0,
    cvar_level: float = 0.95,
) -> pd.DataFrame:
    """Strategies x metrics table from simulation results.

    Parameters
    ----------
    simulations : mapping of str to SimulationResult
        Output of :func:`pfopt.backtest.engine.simulate_all`.
    which : {"net", "gross"}, default "net"
        Which return series to evaluate (turnover is the same for both).
    annualization, rf, mar, cvar_level
        See :func:`performance_metrics`.

    Returns
    -------
    pandas.DataFrame
        One row per strategy with the columns of :data:`METRIC_NAMES` plus ``mdd_peak``,
        ``mdd_trough`` and ``mdd_recovery`` (ISO dates, empty if undefined).
    """
    if which not in ("net", "gross"):
        raise ValueError(f"which must be 'net' or 'gross', got {which!r}")
    rows = {}
    for strategy, sim in simulations.items():
        series = getattr(sim, which)
        row: dict[str, object] = dict(
            performance_metrics(series, sim.turnover, annualization, rf, mar, cvar_level)
        )
        info = max_drawdown(series)
        for key, value in (
            ("mdd_peak", info.peak),
            ("mdd_trough", info.trough),
            ("mdd_recovery", info.recovery),
        ):
            row[key] = "" if pd.isna(value) else f"{value:%Y-%m-%d}"
        rows[strategy] = row
    return pd.DataFrame.from_dict(rows, orient="index")
