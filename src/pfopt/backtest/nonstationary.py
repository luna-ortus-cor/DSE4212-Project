"""EWMA strategies on the real data for the non-stationarity test (PLAN.md step 8.22).

Same universe, calendar, estimation windows, winsorization, long-only constraint and cost model as
the core strategies (:func:`pfopt.backtest.engine.generate_weights`); only the covariance differs:
the exponentially weighted covariance of the window (``GMV-EWMA<h>``) and its RMT-clipped version
with the effective sample size in the Marchenko-Pastur edge (``GMV-EWMA<h>-RMT``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pfopt.config import Config
from pfopt.data.eda import get_window
from pfopt.data.universe import rebalance_dates, select_universe
from pfopt.estimators.ewma import effective_sample_size, ewma_cov, ewma_rmt_cov, ewma_weights
from pfopt.estimators.utils import validate_returns, winsorize_cols
from pfopt.optimizers.gmv import gmv_long_only
from pfopt.weights import check_weights

logger = logging.getLogger(__name__)


def ewma_strategy_ids(halflives: tuple[int, ...]) -> tuple[str, ...]:
    """Strategy ids for the given half-lives: ``GMV-EWMA<h>`` then ``GMV-EWMA<h>-RMT``."""
    return tuple(i for h in halflives for i in (f"GMV-EWMA{h}", f"GMV-EWMA{h}-RMT"))


@dataclass(frozen=True)
class EwmaResult:
    """Output of :func:`generate_ewma_weights`.

    Attributes
    ----------
    weights : dict of str to pandas.DataFrame
        Per strategy id: rebalance date x symbol long-only weights over all symbols (zero outside
        the universe); rows sum to 1.
    diagnostics : pandas.DataFrame
        One row per rebalance date: ``n_universe``, per half-life ``t_eff_<h>`` and
        ``rmt_n_signal_<h>``, and per strategy ``pred_vol_<id>`` (annualized volatility forecast).
    """

    weights: dict[str, pd.DataFrame]
    diagnostics: pd.DataFrame


def generate_ewma_weights(
    log_returns: pd.DataFrame,
    dollar_volume: pd.DataFrame,
    cfg: Config,
    end_date: str | pd.Timestamp | None = None,
) -> EwmaResult:
    """Long-only GMV weights from EWMA covariances at every rebalance date (data up to that date).

    Parameters
    ----------
    log_returns, dollar_volume : pandas.DataFrame
        Date x Symbol frames with identical index and columns.
    cfg : Config
        Uses ``universe``, ``backtest`` (rebalance days, weight cap), ``data`` (winsorization),
        ``nonstationarity.halflives`` and ``run``.
    end_date : str or Timestamp, optional
        If given, only rebalance dates up to and including it are generated.

    Returns
    -------
    EwmaResult
    """
    n, t, w = cfg.universe.n_assets, cfg.universe.lookback_days, cfg.universe.dollar_volume_window
    halflives = cfg.nonstationarity.halflives
    ids = ewma_strategy_ids(halflives)
    dates = rebalance_dates(log_returns, t, n, cfg.backtest.rebalance_days)
    if end_date is not None:
        dates = dates[dates <= pd.Timestamp(end_date)]
    if len(dates) == 0:
        raise ValueError("no rebalance dates on or before end_date")
    columns = list(log_returns.columns)
    position = {symbol: i for i, symbol in enumerate(columns)}
    arrays = {s: np.zeros((len(dates), len(columns))) for s in ids}
    rows: list[dict[str, float]] = []
    annual = float(np.sqrt(cfg.run.annualization))
    cap = cfg.backtest.max_weight

    for k, date in enumerate(dates):
        universe = select_universe(log_returns, dollar_volume, date, t, n, w)
        window = get_window(log_returns, date, t, universe)
        clipped = winsorize_cols(window.to_numpy(), cfg.data.winsor_lower, cfg.data.winsor_upper)
        x = validate_returns(clipped)
        row: dict[str, float] = {"n_universe": float(len(universe))}
        cols = [position[symbol] for symbol in universe]
        for h in halflives:
            row[f"t_eff_{h}"] = effective_sample_size(ewma_weights(x.shape[0], h))
            rmt, info = ewma_rmt_cov(x, h, return_info=True)
            row[f"rmt_n_signal_{h}"] = info["n_signal"]
            for strategy, cov in ((f"GMV-EWMA{h}", ewma_cov(x, h)), (f"GMV-EWMA{h}-RMT", rmt)):
                weights = gmv_long_only(cov, max_weight=cap)
                variance = max(float(weights @ cov @ weights), 0.0)
                row[f"pred_vol_{strategy}"] = float(np.sqrt(variance)) * annual
                arrays[strategy][k, cols] = weights
        rows.append(row)
        if (k + 1) % 25 == 0 or k + 1 == len(dates):
            logger.info("EWMA weights for %d / %d rebalance dates", k + 1, len(dates))

    index = pd.DatetimeIndex(dates, name="Date")
    frames = {s: pd.DataFrame(arrays[s], index=index, columns=columns) for s in ids}
    for frame in frames.values():
        for row_weights in frame.to_numpy():
            check_weights(row_weights, max_weight=cap)
    return EwmaResult(weights=frames, diagnostics=pd.DataFrame(rows, index=index))
