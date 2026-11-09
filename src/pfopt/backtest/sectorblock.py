"""PCA plus sector-block strategy on the real data (PLAN.md step 8.24).

Same universe, calendar, estimation windows, winsorization, long-only constraint and cost model as
the core strategies (:func:`pfopt.backtest.engine.generate_weights`); only the covariance differs
(:func:`pfopt.estimators.sector_block.sector_block_cov`). Sector labels are the static snapshot of
``sp500_companies.csv``, used as labels only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pfopt.backtest.residual import UNKNOWN_CODE, sector_codes
from pfopt.backtest.structure import UNKNOWN_SECTOR
from pfopt.config import Config
from pfopt.data.eda import get_window
from pfopt.data.universe import rebalance_dates, select_universe
from pfopt.estimators.sector_block import sector_block_cov
from pfopt.estimators.utils import validate_returns, winsorize_cols
from pfopt.optimizers.gmv import gmv_long_only
from pfopt.weights import check_weights

logger = logging.getLogger(__name__)

STRATEGY = "GMV-PCA-SECTOR"
PLACEBO_TAG = 27


def sector_categories(sectors: pd.Series) -> list[str]:
    """Sorted list of the known sector names (the label ``Unknown`` and missing values dropped)."""
    return sorted({s for s in sectors.dropna().unique() if s != UNKNOWN_SECTOR})


def placebo_codes(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shuffle the known group labels among the members (step 8.25 placebo).

    Group sizes and the set of unknown members are unchanged; only which asset carries which label
    is randomized, so any sector information is destroyed and the block structure is kept.
    """
    out = groups.copy()
    known = np.flatnonzero(groups != UNKNOWN_CODE)
    out[known] = groups[rng.permutation(known)]
    return out


def covariance_from_config(
    x: np.ndarray, groups: np.ndarray, cfg: Config
) -> tuple[np.ndarray, dict[str, object]]:
    """Sector-block covariance with the K rule of ``cfg.pca`` (same K as the PCA strategy)."""
    k_fixed = cfg.pca.k_fixed if cfg.pca.k_rule == "fixed" else None
    return sector_block_cov(
        x, groups, K=k_fixed, rule=cfg.pca.k_rule, kmax=cfg.pca.baing_kmax, return_info=True
    )


@dataclass(frozen=True)
class SectorBlockResult:
    """Output of :func:`generate_sector_block_weights`.

    Attributes
    ----------
    weights : pandas.DataFrame
        Rebalance date x symbol long-only weights over all symbols (zero outside the universe).
    diagnostics : pandas.DataFrame
        One row per rebalance date: ``n_universe``, ``n_known_sector``, ``K``, ``block_share`` and
        ``pred_vol`` (annualized volatility forecast of the portfolio).
    """

    weights: pd.DataFrame
    diagnostics: pd.DataFrame


def generate_sector_block_weights(
    log_returns: pd.DataFrame,
    dollar_volume: pd.DataFrame,
    sectors: pd.Series,
    cfg: Config,
    end_date: str | pd.Timestamp | None = None,
    placebo_draw: int | None = None,
) -> SectorBlockResult:
    """Long-only GMV weights from the sector-block covariance at every rebalance date.

    Parameters
    ----------
    log_returns, dollar_volume : pandas.DataFrame
        Date x Symbol frames with identical index and columns.
    sectors : pandas.Series
        Static sector label indexed by symbol.
    cfg : Config
        Uses ``universe``, ``backtest``, ``data`` (winsorization), ``pca`` and ``run``.
    end_date : str or Timestamp, optional
        If given, only rebalance dates up to and including it are generated.
    placebo_draw : int, optional
        If given, the known labels are shuffled among the members at every rebalance (step 8.25),
        with the generator seeded ``[seed, 27, placebo_draw, rebalance index]``.

    Returns
    -------
    SectorBlockResult
    """
    n, t, w = cfg.universe.n_assets, cfg.universe.lookback_days, cfg.universe.dollar_volume_window
    dates = rebalance_dates(log_returns, t, n, cfg.backtest.rebalance_days)
    if end_date is not None:
        dates = dates[dates <= pd.Timestamp(end_date)]
    if len(dates) == 0:
        raise ValueError("no rebalance dates on or before end_date")
    categories = sector_categories(sectors)
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
        x = validate_returns(clipped)
        groups = sector_codes(universe, sectors, categories)
        if placebo_draw is not None:
            rng = np.random.default_rng([cfg.run.seed, PLACEBO_TAG, placebo_draw, k])
            groups = placebo_codes(groups, rng)
        cov, info = covariance_from_config(x, groups, cfg)
        row_weights = gmv_long_only(cov, max_weight=cap)
        variance = max(float(row_weights @ cov @ row_weights), 0.0)
        weights[k, [position[s] for s in universe]] = row_weights
        rows.append(
            {
                "n_universe": float(len(universe)),
                "n_known_sector": float((groups != UNKNOWN_CODE).sum()),
                "K": float(info["K"]),
                "block_share": float(info["block_share"]),
                "pred_vol": float(np.sqrt(variance)) * annual,
            }
        )
        if (k + 1) % 25 == 0 or k + 1 == len(dates):
            logger.info("sector-block weights for %d / %d rebalance dates", k + 1, len(dates))

    index = pd.DatetimeIndex(dates, name="Date")
    frame = pd.DataFrame(weights, index=index, columns=columns)
    for row_weights in frame.to_numpy():
        check_weights(row_weights, max_weight=cap)
    return SectorBlockResult(weights=frame, diagnostics=pd.DataFrame(rows, index=index))
