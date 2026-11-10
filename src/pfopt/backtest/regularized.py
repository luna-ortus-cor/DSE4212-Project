"""Regularized weights on the real data.

All strategies use the sample covariance of exactly the same universes, windows, dates and
winsorization as the core strategies; only the weight rule differs. Ridge and shrinkage versions are
long-only. The L1-bounded and elastic-net versions allow shorting (no borrow cost, as in 5.5).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pfopt.config import Config
from pfopt.data.eda import get_window
from pfopt.data.universe import rebalance_dates, select_universe
from pfopt.estimators.sample import sample_cov
from pfopt.estimators.utils import validate_returns, winsorize_cols
from pfopt.optimizers.gmv import gmv_long_only
from pfopt.optimizers.regularized import gmv_l1, gmv_ridge_long_only, shrink_to_equal
from pfopt.weights import check_weights

logger = logging.getLogger(__name__)


def strategy_ids(cfg: Config) -> dict[str, bool]:
    """Strategy id -> whether it is long-only, in a fixed order."""
    r = cfg.regularized
    ids = {f"GMV-RIDGE-{v:g}": True for v in r.ridge_lambdas}
    ids.update({f"GMV-SHRINK-{g:g}": True for g in r.shrink_gammas})
    ids.update({f"GMV-L1-{c:g}": False for c in r.l1_bounds})
    ids[f"GMV-ENET-{r.enet_bound:g}-{r.enet_lambda:g}"] = False
    return ids


@dataclass(frozen=True)
class RegularizedResult:
    """Output of :func:`generate_regularized_weights`.

    Attributes
    ----------
    weights : dict of str to pandas.DataFrame
        Per strategy id: rebalance date x symbol weights over all symbols.
    diagnostics : pandas.DataFrame
        One row per rebalance date: ``n_universe`` and, per strategy, ``pred_vol_<id>``
        (annualized, ``w' Sigma_sample w``) and ``leverage_<id>``.
    """

    weights: dict[str, pd.DataFrame]
    diagnostics: pd.DataFrame


def _weights_for(sigma: np.ndarray, cfg: Config) -> dict[str, np.ndarray]:
    r = cfg.regularized
    gmv = np.asarray(gmv_long_only(sigma))
    out: dict[str, np.ndarray] = {}
    for v in r.ridge_lambdas:
        out[f"GMV-RIDGE-{v:g}"] = gmv_ridge_long_only(sigma, v)
    for g in r.shrink_gammas:
        out[f"GMV-SHRINK-{g:g}"] = shrink_to_equal(gmv, g)
    for c in r.l1_bounds:
        out[f"GMV-L1-{c:g}"] = gmv_l1(sigma, c)[0]
    out[f"GMV-ENET-{r.enet_bound:g}-{r.enet_lambda:g}"] = gmv_l1(
        sigma, r.enet_bound, r.enet_lambda
    )[0]
    return out


def generate_regularized_weights(
    log_returns: pd.DataFrame,
    dollar_volume: pd.DataFrame,
    cfg: Config,
    end_date: str | pd.Timestamp | None = None,
) -> RegularizedResult:
    """Regularized weights at every rebalance date, using only data up to that date.

    Parameters
    ----------
    log_returns, dollar_volume : pandas.DataFrame
        Date x Symbol frames with identical index and columns.
    cfg : Config
        Uses ``universe``, ``backtest``, ``data``, ``run`` and ``regularized``.
    end_date : str or Timestamp, optional
        If given, only rebalance dates up to and including it are generated.
    """
    n, t, w = cfg.universe.n_assets, cfg.universe.lookback_days, cfg.universe.dollar_volume_window
    dates = rebalance_dates(log_returns, t, n, cfg.backtest.rebalance_days)
    if end_date is not None:
        dates = dates[dates <= pd.Timestamp(end_date)]
    if len(dates) == 0:
        raise ValueError("no rebalance dates on or before end_date")
    ids = strategy_ids(cfg)
    columns = list(log_returns.columns)
    position = {symbol: i for i, symbol in enumerate(columns)}
    arrays = {s: np.zeros((len(dates), len(columns))) for s in ids}
    rows: list[dict[str, float]] = []
    annual = float(np.sqrt(cfg.run.annualization))

    for k, date in enumerate(dates):
        universe = select_universe(log_returns, dollar_volume, date, t, n, w)
        window = get_window(log_returns, date, t, universe)
        clipped = winsorize_cols(window.to_numpy(), cfg.data.winsor_lower, cfg.data.winsor_upper)
        sigma = sample_cov(validate_returns(clipped))
        cols = [position[s] for s in universe]
        row: dict[str, float] = {"n_universe": float(len(universe))}
        for strategy, weights in _weights_for(sigma, cfg).items():
            variance = max(float(weights @ sigma @ weights), 0.0)
            row[f"pred_vol_{strategy}"] = float(np.sqrt(variance)) * annual
            row[f"leverage_{strategy}"] = float(np.abs(weights).sum())
            arrays[strategy][k, cols] = weights
        rows.append(row)
        if (k + 1) % 25 == 0 or k + 1 == len(dates):
            logger.info("regularized weights for %d / %d rebalance dates", k + 1, len(dates))

    index = pd.DatetimeIndex(dates, name="Date")
    frames = {s: pd.DataFrame(arrays[s], index=index, columns=columns) for s in ids}
    for strategy, frame in frames.items():
        for row_weights in frame.to_numpy():
            check_weights(row_weights, long_only=ids[strategy])
    return RegularizedResult(weights=frames, diagnostics=pd.DataFrame(rows, index=index))
