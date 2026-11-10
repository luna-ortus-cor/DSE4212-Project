"""Covariance-quality and weight-stability metrics.

Risk-forecast quality asks whether the volatility a strategy's covariance matrix *predicts* at a
rebalance date matches the volatility the strategy *realizes* afterwards. For strategy ``s`` and
the holding period ``k`` (the trading days after rebalance date ``t_k`` up to and including
``t_{k+1}``, or to the end of the data for the last period):

- predicted volatility ``P_k = sqrt(w_k' Sigma_k w_k) * sqrt(A)``, with ``Sigma_k`` the covariance
  matrix that produced the weights (stored by the engine as ``pred_vol_<strategy>``);
- realized volatility ``V_k = std(gross daily returns in period k, ddof=1) * sqrt(A)``;
- period ratio ``V_k / P_k`` and its log. Over ``n_k`` daily returns the standard error of
  ``V_k`` is about ``V_k / sqrt(2 (n_k - 1))`` (roughly 16% for 21 days), so single periods are
  noisy and averages over periods are the meaningful numbers.

Two aggregates do not depend on the period length:

- pooled variance ratio: variance of all included daily returns divided by their day-weighted mean
  predicted variance (A times the squared daily forecast);
- bias statistic: standard deviation (``ddof=1``) of the daily returns divided by their period's
  predicted daily volatility. It is 1 for an unbiased risk forecast and above 1 when risk is
  under-predicted. A rolling version over ``bias_window`` periods is also returned.

The weights drift between rebalances while the forecast uses the target weights; that small
difference is part of what is measured. Periods with fewer than ``min_days`` returns are skipped.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pfopt.backtest.engine import SimulationResult


def weight_stats(
    weights: np.ndarray, position_threshold: float = 1e-3, top_k: int = 10
) -> dict[str, float]:
    """Concentration statistics of one weight vector.

    Parameters
    ----------
    weights : numpy.ndarray
        Weights (long-only, summing to 1).
    position_threshold : float, default 1e-3
        A holding counts as a position if its weight exceeds this (0.1%).
    top_k : int, default 10
        Number of largest weights summed in ``top_k_share``.

    Returns
    -------
    dict
        ``n_positions`` (weights above the threshold), ``effective_n`` (``1 / sum(w^2)``),
        ``max_weight`` and ``top_k_share`` (sum of the ``top_k`` largest weights).
    """
    w = np.asarray(weights, dtype=float)
    ordered = np.sort(w)[::-1]
    return {
        "n_positions": float(np.sum(w > position_threshold)),
        "effective_n": float(1.0 / np.sum(w**2)),
        "max_weight": float(ordered[0]),
        "top_k_share": float(ordered[:top_k].sum()),
    }


def weight_stability(weights: pd.DataFrame, turnover: pd.Series | None = None) -> dict[str, float]:
    """Stability and concentration of a strategy's weights over the rebalances.

    Parameters
    ----------
    weights : pandas.DataFrame
        Rebalance date x symbol target weights.
    turnover : pandas.Series, optional
        Drift-adjusted turnover ``sum |target - drifted|`` per rebalance (first entry is the
        initial investment and is excluded from ``mean_turnover``).

    Returns
    -------
    dict
        ``mean_l1_target_change`` (mean ``sum |W_k - W_{k-1}|`` between consecutive target
        vectors), ``mean_turnover`` (NaN without ``turnover``), ``mean_effective_n``,
        ``median_effective_n``, ``mean_max_weight``, ``mean_n_positions`` (weights above 0.1%) and
        ``n_rebalances``.
    """
    w = weights.to_numpy(dtype=float)
    stats = pd.DataFrame([weight_stats(row) for row in w])
    l1_change = np.abs(np.diff(w, axis=0)).sum(axis=1)
    mean_turnover = np.nan
    if turnover is not None and len(turnover) > 1:
        mean_turnover = float(turnover.to_numpy(dtype=float)[1:].mean())
    return {
        "mean_l1_target_change": float(l1_change.mean()) if l1_change.size else np.nan,
        "mean_turnover": mean_turnover,
        "mean_effective_n": float(stats["effective_n"].mean()),
        "median_effective_n": float(stats["effective_n"].median()),
        "mean_max_weight": float(stats["max_weight"].mean()),
        "mean_n_positions": float(stats["n_positions"].mean()),
        "n_rebalances": float(len(w)),
    }


def stability_table(
    weights: Mapping[str, pd.DataFrame],
    simulations: Mapping[str, SimulationResult] | None = None,
) -> pd.DataFrame:
    """Strategies x stability metrics (see :func:`weight_stability`)."""
    rows = {
        strategy: weight_stability(
            frame, None if simulations is None else simulations[strategy].turnover
        )
        for strategy, frame in weights.items()
    }
    return pd.DataFrame.from_dict(rows, orient="index")


@dataclass(frozen=True)
class ForecastQuality:
    """Risk-forecast quality of one strategy.

    Attributes
    ----------
    periods : pandas.DataFrame
        One row per included holding period, indexed by the rebalance date, with ``n_days``,
        ``predicted_vol``, ``realized_vol`` (annualized), ``ratio`` and ``log_ratio``.
    summary : dict
        ``n_periods``, ``mean_ratio``, ``median_ratio``, ``mean_log_ratio``,
        ``pooled_variance_ratio``, ``bias_statistic``, ``realized_vol_pooled`` and
        ``mean_predicted_vol`` (annualized).
    rolling_bias : pandas.Series
        Bias statistic over a rolling window of periods, indexed by the last rebalance date in
        the window (empty if there are fewer periods than the window).
    """

    periods: pd.DataFrame
    summary: dict[str, float]
    rolling_bias: pd.Series


def risk_forecast_quality(
    diagnostics: pd.DataFrame,
    simulation: SimulationResult,
    strategy: str,
    annualization: int = 252,
    min_days: int = 10,
    bias_window: int = 12,
) -> ForecastQuality:
    """Compare a strategy's predicted and realized volatility (see the module docstring).

    Parameters
    ----------
    diagnostics : pandas.DataFrame
        Engine diagnostics with a ``pred_vol_<strategy>`` column, indexed by rebalance date.
    simulation : SimulationResult
        Simulation of the same strategy; its gross returns are the realized returns.
    strategy : str
        Strategy id.
    annualization : int, default 252
        Trading days per year.
    min_days : int, default 10
        Holding periods with fewer daily returns are skipped (at least 2 are always required).
    bias_window : int, default 12
        Number of periods in the rolling bias statistic.

    Raises
    ------
    ValueError
        If the forecast column is missing, the rebalance dates of the diagnostics and the
        simulation differ, or no holding period has enough observations.
    """
    column = f"pred_vol_{strategy}"
    if column not in diagnostics.columns:
        raise ValueError(f"diagnostics has no column {column!r}; regenerate the weights")
    dates = diagnostics.index
    if not dates.equals(simulation.turnover.index):
        raise ValueError("diagnostics and simulation must share the same rebalance dates")
    predicted = diagnostics[column].to_numpy(dtype=float)
    returns = simulation.gross.to_numpy(dtype=float)
    days = simulation.gross.index.to_numpy()
    starts = np.searchsorted(days, dates.to_numpy(), side="right")
    ends = np.append(np.searchsorted(days, dates.to_numpy()[1:], side="right"), len(returns))
    root = float(np.sqrt(annualization))

    rows, kept_dates, z_blocks, return_blocks, variance_blocks = [], [], [], [], []
    for k in range(len(dates)):
        block = returns[starts[k] : ends[k]]
        if block.size < max(min_days, 2) or not predicted[k] > 0:
            continue
        realized = float(block.std(ddof=1)) * root
        ratio = realized / predicted[k]
        rows.append(
            {
                "n_days": block.size,
                "predicted_vol": predicted[k],
                "realized_vol": realized,
                "ratio": ratio,
                "log_ratio": float(np.log(ratio)) if ratio > 0 else -np.inf,
            }
        )
        kept_dates.append(dates[k])
        z_blocks.append(block / (predicted[k] / root))
        return_blocks.append(block)
        variance_blocks.append(np.full(block.size, predicted[k] ** 2))
    if not rows:
        raise ValueError("no holding period has enough observations")

    periods = pd.DataFrame(rows, index=pd.DatetimeIndex(kept_dates, name="Date"))
    all_returns = np.concatenate(return_blocks)
    mean_predicted_variance = float(np.concatenate(variance_blocks).mean())
    realized_variance = float(all_returns.var(ddof=1)) * annualization
    summary = {
        "n_periods": float(len(periods)),
        "mean_ratio": float(periods["ratio"].mean()),
        "median_ratio": float(periods["ratio"].median()),
        "mean_log_ratio": float(periods["log_ratio"].mean()),
        "pooled_variance_ratio": realized_variance / mean_predicted_variance,
        "bias_statistic": float(np.concatenate(z_blocks).std(ddof=1)),
        "realized_vol_pooled": float(np.sqrt(realized_variance)),
        "mean_predicted_vol": float(np.sqrt(mean_predicted_variance)),
    }
    rolling = [
        float(np.concatenate(z_blocks[k - bias_window + 1 : k + 1]).std(ddof=1))
        for k in range(bias_window - 1, len(z_blocks))
    ]
    rolling_bias = pd.Series(
        rolling, index=periods.index[bias_window - 1 :], name="rolling_bias_statistic"
    )
    return ForecastQuality(periods=periods, summary=summary, rolling_bias=rolling_bias)


def forecast_quality_table(
    diagnostics: pd.DataFrame,
    simulations: Mapping[str, SimulationResult],
    strategies: Sequence[str] | None = None,
    annualization: int = 252,
    min_days: int = 10,
    bias_window: int = 12,
) -> pd.DataFrame:
    """Strategies x risk-forecast summary (see :func:`risk_forecast_quality`).

    By default every strategy that has a ``pred_vol_`` column in ``diagnostics`` is included.
    """
    if strategies is None:
        prefix = "pred_vol_"
        strategies = [c.removeprefix(prefix) for c in diagnostics.columns if c.startswith(prefix)]
    rows = {
        s: risk_forecast_quality(
            diagnostics, simulations[s], s, annualization, min_days, bias_window
        ).summary
        for s in strategies
    }
    return pd.DataFrame.from_dict(rows, orient="index")
