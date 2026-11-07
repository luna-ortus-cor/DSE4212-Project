"""Backtest engine: leakage-free weight generation (PLAN.md step 4.1).

At every rebalance date ``t`` the engine uses only data dated ``<= t``: the point-in-time universe
(:func:`pfopt.data.universe.select_universe`), the last ``T`` log returns up to and including ``t``,
and per-column winsorization computed inside that window. The weights are meant to be applied to
returns from ``t + 1`` onward (that alignment is handled by the simulator, step 4.2).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pfopt.backtest.strategies import (
    CORE_STRATEGIES,
    ESTIMATORS,
    MARKOWITZ_STRATEGIES,
    PREDICTION_COVARIANCE,
    core_strategies,
    markowitz_strategies,
)
from pfopt.config import Config
from pfopt.data.eda import get_window
from pfopt.data.universe import rebalance_dates, select_universe
from pfopt.estimators.utils import condition_number, winsorize_cols
from pfopt.weights import check_weights

logger = logging.getLogger(__name__)

ALL_STRATEGIES = CORE_STRATEGIES + MARKOWITZ_STRATEGIES
_LOG_EVERY = 25


@dataclass(frozen=True)
class WeightsResult:
    """Output of :func:`generate_weights`.

    Attributes
    ----------
    weights : dict of str to pandas.DataFrame
        Per strategy id: rebalance date x symbol weights over *all* symbols (zero outside the
        universe at that date). Every row is non-negative and sums to 1.
    diagnostics : pandas.DataFrame
        One row per rebalance date: universe size, covariance condition numbers (``cond_<name>``),
        Ledoit-Wolf shrinkage, RMT signal count and edge, PCA ``K``, solver used per optimized
        strategy (``solver_<id>``), ``n_solver_fallbacks`` and the annualized volatility forecast
        ``pred_vol_<id>`` = ``sqrt(w' Sigma w) * sqrt(A)`` of each strategy except EW, with the
        covariance ``Sigma`` that produced its weights.
    universe : pandas.DataFrame
        Boolean rebalance date x symbol membership of the point-in-time universe.
    """

    weights: dict[str, pd.DataFrame]
    diagnostics: pd.DataFrame
    universe: pd.DataFrame


def generate_weights(
    log_returns: pd.DataFrame,
    dollar_volume: pd.DataFrame,
    cfg: Config,
    strategies: Sequence[str] | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> WeightsResult:
    """Generate weights for every rebalance date without using future data.

    Calendar: the first rebalance date is the first date with at least ``N`` eligible symbols
    (that is, the first date with a full ``T``-day window for ``N`` names); later dates follow
    every ``backtest.rebalance_days`` trading days.

    Parameters
    ----------
    log_returns, dollar_volume : pandas.DataFrame
        Date x Symbol frames with identical index and columns (from the cleaning step).
    cfg : Config
        Uses ``universe``, ``backtest``, ``data`` (winsorization quantiles), ``pca``, ``hrp``,
        ``run``.
    strategies : sequence of str, optional
        Strategy ids to store (default: all of :data:`ALL_STRATEGIES`).
    end_date : str or Timestamp, optional
        If given, only rebalance dates up to and including it are generated.

    Returns
    -------
    WeightsResult

    Raises
    ------
    ValueError
        For unknown strategy ids, an ``end_date`` before the first rebalance date, or if fewer
        than ``N`` symbols are eligible at some rebalance date.
    """
    ids = tuple(ALL_STRATEGIES if strategies is None else strategies)
    unknown = sorted(set(ids) - set(ALL_STRATEGIES))
    if unknown:
        raise ValueError(f"unknown strategy id(s) {unknown}; available: {ALL_STRATEGIES}")
    n, t, w = cfg.universe.n_assets, cfg.universe.lookback_days, cfg.universe.dollar_volume_window
    dates = rebalance_dates(log_returns, t, n, cfg.backtest.rebalance_days)
    if end_date is not None:
        dates = dates[dates <= pd.Timestamp(end_date)]
    if len(dates) == 0:
        raise ValueError("no rebalance dates on or before end_date")

    columns = list(log_returns.columns)
    position = {symbol: i for i, symbol in enumerate(columns)}
    arrays = {s: np.zeros((len(dates), len(columns))) for s in ids}
    members = np.zeros((len(dates), len(columns)), dtype=bool)
    rows: list[dict[str, object]] = []
    need_markowitz = any(s in MARKOWITZ_STRATEGIES for s in ids)
    annual = float(np.sqrt(cfg.run.annualization))

    for k, date in enumerate(dates):
        universe = select_universe(log_returns, dollar_volume, date, t, n, w)
        window = get_window(log_returns, date, t, universe)
        x = winsorize_cols(window.to_numpy(), cfg.data.winsor_lower, cfg.data.winsor_upper)
        result = core_strategies(x, cfg)
        weights = dict(result.weights)
        row: dict[str, object] = {
            "n_universe": len(universe),
            "lw_shrinkage": result.info["lw_shrinkage"],
            "rmt_n_signal": result.info["rmt_n_signal"],
            "rmt_lam_plus": result.info["rmt_lam_plus"],
            "pca_K": result.info["pca_K"],
        }
        for name in ESTIMATORS:
            row[f"cond_{name}"] = condition_number(result.covariances[name])
        n_fallbacks = 0
        for strategy, info in result.info["gmv"].items():
            row[f"solver_{strategy}"] = info["solver"]
            n_fallbacks += int(bool(info["fallback"]))
        if need_markowitz:
            mv_weights, mv_infos = markowitz_strategies(x, cfg, result.covariances)
            weights.update(mv_weights)
            for strategy, info in mv_infos.items():
                row[f"solver_{strategy}"] = info["solver"]
                n_fallbacks += int(bool(info["fallback_slsqp"]) or bool(info["fallback_gmv"]))
        row["n_solver_fallbacks"] = n_fallbacks
        for strategy in ids:
            estimator = PREDICTION_COVARIANCE.get(strategy)
            if estimator is not None:
                cov, wv = result.covariances[estimator], weights[strategy]
                row[f"pred_vol_{strategy}"] = float(np.sqrt(max(wv @ cov @ wv, 0.0))) * annual
        rows.append(row)

        cols = [position[symbol] for symbol in universe]
        members[k, cols] = True
        for strategy in ids:
            arrays[strategy][k, cols] = weights[strategy]
        if (k + 1) % _LOG_EVERY == 0 or k + 1 == len(dates):
            logger.info("weights generated for %d / %d rebalance dates", k + 1, len(dates))

    index = pd.DatetimeIndex(dates, name="Date")
    frames = {s: pd.DataFrame(arrays[s], index=index, columns=columns) for s in ids}
    for strategy, frame in frames.items():
        for row_weights in frame.to_numpy():
            check_weights(row_weights, max_weight=_cap_for(strategy, cfg))
    diagnostics = pd.DataFrame(rows, index=index)
    universe_frame = pd.DataFrame(members, index=index, columns=columns)
    return WeightsResult(weights=frames, diagnostics=diagnostics, universe=universe_frame)


@dataclass(frozen=True)
class SimulationResult:
    """Output of :func:`simulate` for one strategy.

    Attributes
    ----------
    gross, net : pandas.Series
        Daily portfolio simple returns before and after transaction costs, indexed by the days on
        which the portfolio is invested (the first rebalance date is not included: weights chosen
        at its close are held from the next day).
    turnover : pandas.Series
        ``sum |target - drifted|`` at each rebalance date (the first entry is 1: the initial
        investment from cash).
    weights_pre : pandas.DataFrame
        Drifted weights just before each rebalance trade (zeros for the initial investment).
    weights_post : pandas.DataFrame
        Target weights after each rebalance (the input weights).
    daily_weights : pandas.DataFrame
        Weights held during each day of ``gross`` (start of day, after drift and rebalancing).
    n_missing_returns : int
        Number of (day, asset) pairs where a held asset had no return and was assumed flat (0).
    """

    gross: pd.Series
    net: pd.Series
    turnover: pd.Series
    weights_pre: pd.DataFrame
    weights_post: pd.DataFrame
    daily_weights: pd.DataFrame
    n_missing_returns: int


def simulate(
    weights: pd.DataFrame, simple_returns: pd.DataFrame, cost_bps: float
) -> SimulationResult:
    """Simulate a strategy's daily P&L with weight drift and proportional transaction costs.

    Timeline. Weights in row ``t`` of ``weights`` are decided at the close of ``t`` and held
    from the next trading day. Between rebalances the weights drift,
    ``w_i <- w_i (1 + R_i) / (1 + R_p)`` with ``R_p = sum_i w_i R_i``. On the next rebalance
    date ``t'`` the old weights earn that day's return, and at the close of ``t'`` the
    portfolio is traded to the new target. Turnover is ``sum_i |target_i - drifted_i|`` over
    all symbols (assets entering or leaving the universe included), and the cost
    ``cost_bps / 1e4 * turnover`` is deducted from the portfolio return of ``t'``. The initial
    investment from cash has turnover 1; since the first rebalance date has no return of its
    own, its cost is deducted on the first holding day. A held asset with a missing return is
    assumed flat (return 0) and counted.

    Parameters
    ----------
    weights : pandas.DataFrame
        Rebalance date x symbol target weights; rows non-negative and summing to 1.
    simple_returns : pandas.DataFrame
        Date x symbol simple returns; must contain every symbol of ``weights`` and every rebalance
        date in its index.
    cost_bps : float
        Proportional cost in basis points per unit of turnover (``sum |dw|``), at least 0.

    Returns
    -------
    SimulationResult

    Raises
    ------
    ValueError
        For inconsistent inputs, a negative cost, or if there is no holding day after the first
        rebalance date.
    """
    if cost_bps < 0:
        raise ValueError(f"cost_bps must be >= 0, got {cost_bps}")
    missing_cols = [c for c in weights.columns if c not in simple_returns.columns]
    if missing_cols:
        raise ValueError(f"simple_returns is missing {len(missing_cols)} symbol(s) of weights")
    dates = simple_returns.index
    position = {d: i for i, d in enumerate(dates)}
    missing_dates = [d for d in weights.index if d not in position]
    if missing_dates:
        raise ValueError(f"{len(missing_dates)} rebalance date(s) are not in simple_returns")
    rebalance_pos = np.array([position[d] for d in weights.index])
    if np.any(np.diff(rebalance_pos) <= 0):
        raise ValueError("rebalance dates must be strictly increasing")
    targets = weights.to_numpy(dtype=float)
    for row in targets:
        check_weights(row)
    first, n_days = int(rebalance_pos[0]), len(dates)
    if first >= n_days - 1:
        raise ValueError("no holding day after the first rebalance date")

    returns = simple_returns[list(weights.columns)].to_numpy(dtype=float)
    cost_rate = cost_bps / 1e4
    n_hold = n_days - first - 1
    gross, cost = np.zeros(n_hold), np.zeros(n_hold)
    daily = np.zeros((n_hold, targets.shape[1]))
    turnover = np.zeros(len(targets))
    pre = np.zeros_like(targets)
    turnover[0] = np.abs(targets[0]).sum()
    cost[0] = cost_rate * turnover[0]
    w, k, n_missing = targets[0].copy(), 1, 0

    for i in range(n_hold):
        day = first + 1 + i
        r = returns[day]
        nan = np.isnan(r)
        n_missing += int(np.sum(nan & (w > 0)))
        r = np.where(nan, 0.0, r)
        daily[i] = w
        portfolio = float(w @ r)
        if portfolio <= -1.0:
            raise ValueError("portfolio lost its entire value; simulation is undefined")
        gross[i] = portfolio
        drifted = w * (1.0 + r) / (1.0 + portfolio)
        if k < len(targets) and day == rebalance_pos[k]:
            pre[k] = drifted
            turnover[k] = np.abs(targets[k] - drifted).sum()
            cost[i] += cost_rate * turnover[k]
            w = targets[k].copy()
            k += 1
        else:
            w = drifted

    if n_missing:
        logger.warning("%d held (day, asset) returns were missing and set to 0", n_missing)
    index = dates[first + 1 :]
    rebalance_index = weights.index
    return SimulationResult(
        gross=pd.Series(gross, index=index, name="gross"),
        net=pd.Series(gross - cost, index=index, name="net"),
        turnover=pd.Series(turnover, index=rebalance_index, name="turnover"),
        weights_pre=pd.DataFrame(pre, index=rebalance_index, columns=weights.columns),
        weights_post=weights.copy(),
        daily_weights=pd.DataFrame(daily, index=index, columns=weights.columns),
        n_missing_returns=n_missing,
    )


def simulate_all(
    result: WeightsResult, simple_returns: pd.DataFrame, cost_bps: float
) -> dict[str, SimulationResult]:
    """Run :func:`simulate` for every strategy in a :class:`WeightsResult`."""
    return {
        strategy: simulate(frame, simple_returns, cost_bps)
        for strategy, frame in result.weights.items()
    }


def _cap_for(strategy: str, cfg: Config) -> float | None:
    """Weight cap that applies to a strategy (HRP variants have no cap mechanism)."""
    if strategy in ("HRP", "HRP-RMT"):
        return None
    return cfg.backtest.max_weight


def write_weights_result(result: WeightsResult, out_dir: str | Path) -> None:
    """Write a :class:`WeightsResult` to parquet files in ``out_dir`` (created if needed)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for strategy, frame in result.weights.items():
        frame.to_parquet(out / f"weights_{strategy}.parquet")
    result.diagnostics.to_parquet(out / "diagnostics.parquet")
    result.universe.to_parquet(out / "universe.parquet")


def read_weights_result(in_dir: str | Path) -> WeightsResult:
    """Read a result written by :func:`write_weights_result`.

    Raises
    ------
    FileNotFoundError
        If ``in_dir`` has no diagnostics or universe file.
    """
    base = Path(in_dir)
    for name in ("diagnostics.parquet", "universe.parquet"):
        if not (base / name).exists():
            raise FileNotFoundError(f"{base / name} not found")
    weights = {
        path.stem.removeprefix("weights_"): pd.read_parquet(path)
        for path in sorted(base.glob("weights_*.parquet"))
    }
    return WeightsResult(
        weights=weights,
        diagnostics=pd.read_parquet(base / "diagnostics.parquet"),
        universe=pd.read_parquet(base / "universe.parquet"),
    )
