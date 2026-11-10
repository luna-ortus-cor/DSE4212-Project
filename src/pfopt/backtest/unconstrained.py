"""Unconstrained (shorting allowed) GMV strategies for hypothesis H3.

H3: the advantage of denoised / shrunk covariances shrinks under the long-only constraint
(Jagannathan and Ma, 2003). The unconstrained strategies use exactly the same universe, rebalance
calendar, estimation windows, winsorization and covariance estimators as the long-only ones
(:func:`pfopt.backtest.engine.generate_weights`); only the optimizer differs, in closed form:
``w = Sigma^{-1} 1 / (1' Sigma^{-1} 1)``. Weights sum to 1 and may be negative. Short positions
are simulated without borrow or financing costs, which flatters these strategies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pfopt.backtest.strategies import build_covariances
from pfopt.config import Config
from pfopt.data.eda import get_window
from pfopt.data.universe import rebalance_dates, select_universe
from pfopt.estimators.utils import validate_returns, winsorize_cols
from pfopt.optimizers.gmv import gmv_unconstrained
from pfopt.weights import check_weights

logger = logging.getLogger(__name__)

# Unconstrained strategy id -> (covariance estimator, long-only counterpart).
UNCONSTRAINED = {
    "GMV-SAMPLE-UNC": ("Sample", "GMV-SAMPLE"),
    "GMV-LW-UNC": ("Ledoit-Wolf", "GMV-LW"),
    "GMV-RMT-UNC": ("RMT", "GMV-RMT"),
    "GMV-PCA-UNC": ("PCA", "GMV-PCA"),
}
UNCONSTRAINED_STRATEGIES = tuple(UNCONSTRAINED)


@dataclass(frozen=True)
class UnconstrainedResult:
    """Output of :func:`generate_unconstrained_weights`.

    Attributes
    ----------
    weights : dict of str to pandas.DataFrame
        Per strategy id: rebalance date x symbol weights over all symbols (zero outside the
        universe). Rows sum to 1 and may contain negative entries.
    diagnostics : pandas.DataFrame
        One row per rebalance date: ``n_universe``, and for each strategy the leverage
        ``leverage_<id>`` (``sum |w|``), ``min_weight_<id>``, ``max_weight_<id>``, the ridge
        ``ridge_<id>`` added for numerical singularity (0 if none) and the annualized volatility
        forecast ``pred_vol_<id>``.
    """

    weights: dict[str, pd.DataFrame]
    diagnostics: pd.DataFrame


def generate_unconstrained_weights(
    log_returns: pd.DataFrame,
    dollar_volume: pd.DataFrame,
    cfg: Config,
    end_date: str | pd.Timestamp | None = None,
) -> UnconstrainedResult:
    """Unconstrained GMV weights for every rebalance date, using only data up to that date.

    Parameters
    ----------
    log_returns, dollar_volume : pandas.DataFrame
        Date x Symbol frames with identical index and columns.
    cfg : Config
        Uses ``universe``, ``backtest.rebalance_days``, ``data`` (winsorization), ``pca``, ``run``.
    end_date : str or Timestamp, optional
        If given, only rebalance dates up to and including it are generated.

    Returns
    -------
    UnconstrainedResult

    Raises
    ------
    ValueError
        If there is no rebalance date on or before ``end_date`` or too few eligible symbols.
    """
    n, t, w = cfg.universe.n_assets, cfg.universe.lookback_days, cfg.universe.dollar_volume_window
    dates = rebalance_dates(log_returns, t, n, cfg.backtest.rebalance_days)
    if end_date is not None:
        dates = dates[dates <= pd.Timestamp(end_date)]
    if len(dates) == 0:
        raise ValueError("no rebalance dates on or before end_date")
    columns = list(log_returns.columns)
    position = {symbol: i for i, symbol in enumerate(columns)}
    arrays = {s: np.zeros((len(dates), len(columns))) for s in UNCONSTRAINED}
    rows: list[dict[str, float]] = []
    annual = float(np.sqrt(cfg.run.annualization))

    for k, date in enumerate(dates):
        universe = select_universe(log_returns, dollar_volume, date, t, n, w)
        window = get_window(log_returns, date, t, universe)
        clipped = winsorize_cols(window.to_numpy(), cfg.data.winsor_lower, cfg.data.winsor_upper)
        x = validate_returns(clipped)
        covariances, _ = build_covariances(x, cfg)
        row: dict[str, float] = {"n_universe": float(len(universe))}
        cols = [position[symbol] for symbol in universe]
        for strategy, (estimator, _) in UNCONSTRAINED.items():
            cov = covariances[estimator]
            weights, info = gmv_unconstrained(cov, return_info=True)
            row[f"leverage_{strategy}"] = float(info["leverage"])
            row[f"ridge_{strategy}"] = float(info["ridge"])
            row[f"min_weight_{strategy}"] = float(weights.min())
            row[f"max_weight_{strategy}"] = float(weights.max())
            row[f"pred_vol_{strategy}"] = float(np.sqrt(max(weights @ cov @ weights, 0.0))) * annual
            arrays[strategy][k, cols] = weights
        rows.append(row)
        if (k + 1) % 25 == 0 or k + 1 == len(dates):
            logger.info("unconstrained weights for %d / %d rebalance dates", k + 1, len(dates))

    index = pd.DatetimeIndex(dates, name="Date")
    frames = {s: pd.DataFrame(arrays[s], index=index, columns=columns) for s in UNCONSTRAINED}
    for frame in frames.values():
        for row_weights in frame.to_numpy():
            check_weights(row_weights, long_only=False)
    return UnconstrainedResult(weights=frames, diagnostics=pd.DataFrame(rows, index=index))
