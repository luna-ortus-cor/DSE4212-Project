"""Risk-based strategies on the real data.

``IVOL`` (inverse volatility) and ``ERC-<estimator>`` (equal risk contribution with the sample,
Ledoit-Wolf, RMT and PCA covariances) on exactly the dates, universes, windows and winsorization of
the core strategies; only the weight rule differs. Both rules are long-only and fully invested and
have no weight cap.
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
from pfopt.optimizers.risk_based import erc_weights, inverse_vol_weights
from pfopt.weights import check_weights

logger = logging.getLogger(__name__)

ERC_ESTIMATORS = {
    "ERC-SAMPLE": "Sample",
    "ERC-LW": "Ledoit-Wolf",
    "ERC-RMT": "RMT",
    "ERC-PCA": "PCA",
}
STRATEGIES = ("IVOL", *ERC_ESTIMATORS)


@dataclass(frozen=True)
class RiskBasedResult:
    """Output of :func:`generate_risk_based_weights`.

    Attributes
    ----------
    weights : dict of str to pandas.DataFrame
        Per strategy id (:data:`STRATEGIES`): rebalance date x symbol weights over all symbols.
    diagnostics : pandas.DataFrame
        One row per rebalance date: ``n_universe``, ``erc_iterations_<id>`` and the annualized
        volatility forecast ``pred_vol_<id>`` of every strategy.
    """

    weights: dict[str, pd.DataFrame]
    diagnostics: pd.DataFrame


def generate_risk_based_weights(
    log_returns: pd.DataFrame,
    dollar_volume: pd.DataFrame,
    cfg: Config,
    end_date: str | pd.Timestamp | None = None,
) -> RiskBasedResult:
    """Inverse-volatility and ERC weights at every rebalance date, without using future data.

    Parameters
    ----------
    log_returns, dollar_volume : pandas.DataFrame
        Date x Symbol frames with identical index and columns.
    cfg : Config
        Uses ``universe``, ``backtest``, ``data`` (winsorization), ``pca`` and ``run``.
    end_date : str or Timestamp, optional
        If given, only rebalance dates up to and including it are generated.

    Returns
    -------
    RiskBasedResult
    """
    n, t, w = cfg.universe.n_assets, cfg.universe.lookback_days, cfg.universe.dollar_volume_window
    dates = rebalance_dates(log_returns, t, n, cfg.backtest.rebalance_days)
    if end_date is not None:
        dates = dates[dates <= pd.Timestamp(end_date)]
    if len(dates) == 0:
        raise ValueError("no rebalance dates on or before end_date")
    columns = list(log_returns.columns)
    position = {symbol: i for i, symbol in enumerate(columns)}
    arrays = {s: np.zeros((len(dates), len(columns))) for s in STRATEGIES}
    rows: list[dict[str, float]] = []
    annual = float(np.sqrt(cfg.run.annualization))

    for k, date in enumerate(dates):
        universe = select_universe(log_returns, dollar_volume, date, t, n, w)
        window = get_window(log_returns, date, t, universe)
        clipped = winsorize_cols(window.to_numpy(), cfg.data.winsor_lower, cfg.data.winsor_upper)
        x = validate_returns(clipped)
        covariances, _ = build_covariances(x, cfg)
        cols = [position[s] for s in universe]
        row: dict[str, float] = {"n_universe": float(len(universe))}
        chosen = {"IVOL": inverse_vol_weights(covariances["Sample"])}
        for strategy, estimator in ERC_ESTIMATORS.items():
            chosen[strategy], row[f"erc_iterations_{strategy}"] = erc_weights(
                covariances[estimator]
            )
        for strategy, weights in chosen.items():
            estimator = ERC_ESTIMATORS.get(strategy, "Sample")
            variance = max(float(weights @ covariances[estimator] @ weights), 0.0)
            row[f"pred_vol_{strategy}"] = float(np.sqrt(variance)) * annual
            arrays[strategy][k, cols] = weights
        rows.append(row)
        if (k + 1) % 25 == 0 or k + 1 == len(dates):
            logger.info("risk-based weights for %d / %d rebalance dates", k + 1, len(dates))

    index = pd.DatetimeIndex(dates, name="Date")
    frames = {s: pd.DataFrame(arrays[s], index=index, columns=columns) for s in STRATEGIES}
    for frame in frames.values():
        for row_weights in frame.to_numpy():
            check_weights(row_weights)
    return RiskBasedResult(weights=frames, diagnostics=pd.DataFrame(rows, index=index))
