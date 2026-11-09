"""Nonlinear-shrinkage GMV strategy on the real data (PLAN.md step 8.2).

``GMV-NLS``: long-only GMV with the quadratic-inverse-shrinkage covariance
(:func:`pfopt.estimators.nls.nls_cov`) on exactly the dates, universes, windows, winsorization,
constraint and costs of the core strategies; only the covariance differs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pfopt.config import Config
from pfopt.data.eda import get_window
from pfopt.data.universe import rebalance_dates, select_universe
from pfopt.estimators.nls import nls_cov
from pfopt.estimators.utils import condition_number, validate_returns, winsorize_cols
from pfopt.optimizers.gmv import gmv_long_only
from pfopt.weights import check_weights

logger = logging.getLogger(__name__)

STRATEGY = "GMV-NLS"


@dataclass(frozen=True)
class NlsResult:
    """Output of :func:`generate_nls_weights`.

    Attributes
    ----------
    weights : pandas.DataFrame
        Rebalance date x symbol long-only weights over all symbols (zero outside the universe).
    diagnostics : pandas.DataFrame
        One row per rebalance date: ``n_universe``, ``cond`` (condition number of the covariance),
        ``h`` (kernel bandwidth) and ``pred_vol`` (annualized volatility forecast).
    """

    weights: pd.DataFrame
    diagnostics: pd.DataFrame


def generate_nls_weights(
    log_returns: pd.DataFrame,
    dollar_volume: pd.DataFrame,
    cfg: Config,
    end_date: str | pd.Timestamp | None = None,
) -> NlsResult:
    """Long-only GMV weights from the nonlinear-shrinkage covariance at every rebalance date.

    Parameters
    ----------
    log_returns, dollar_volume : pandas.DataFrame
        Date x Symbol frames with identical index and columns.
    cfg : Config
        Uses ``universe``, ``backtest``, ``data`` (winsorization) and ``run``.
    end_date : str or Timestamp, optional
        If given, only rebalance dates up to and including it are generated.

    Returns
    -------
    NlsResult
    """
    n, t, w = cfg.universe.n_assets, cfg.universe.lookback_days, cfg.universe.dollar_volume_window
    dates = rebalance_dates(log_returns, t, n, cfg.backtest.rebalance_days)
    if end_date is not None:
        dates = dates[dates <= pd.Timestamp(end_date)]
    if len(dates) == 0:
        raise ValueError("no rebalance dates on or before end_date")
    columns = list(log_returns.columns)
    position = {symbol: i for i, symbol in enumerate(columns)}
    weights = np.zeros((len(dates), len(columns)))
    rows: list[dict[str, float]] = []
    annual = float(np.sqrt(cfg.run.annualization))
    cap = cfg.backtest.max_weight

    for k, date in enumerate(dates):
        universe = select_universe(log_returns, dollar_volume, date, t, n, w)
        window = get_window(log_returns, date, t, universe)
        clipped = winsorize_cols(window.to_numpy(), cfg.data.winsor_lower, cfg.data.winsor_upper)
        cov, info = nls_cov(validate_returns(clipped), return_info=True)
        row_weights = gmv_long_only(cov, max_weight=cap)
        variance = max(float(row_weights @ cov @ row_weights), 0.0)
        weights[k, [position[s] for s in universe]] = row_weights
        rows.append(
            {
                "n_universe": float(len(universe)),
                "cond": condition_number(cov),
                "h": float(info["h"]),
                "pred_vol": float(np.sqrt(variance)) * annual,
            }
        )
        if (k + 1) % 25 == 0 or k + 1 == len(dates):
            logger.info("NLS weights for %d / %d rebalance dates", k + 1, len(dates))

    index = pd.DatetimeIndex(dates, name="Date")
    frame = pd.DataFrame(weights, index=index, columns=columns)
    for row_weights in frame.to_numpy():
        check_weights(row_weights, max_weight=cap)
    return NlsResult(weights=frame, diagnostics=pd.DataFrame(rows, index=index))
