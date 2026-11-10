"""Long-only maximum-Sharpe portfolios fed by ML return signals.

The signal is a cross-sectional rank score, not a return forecast. It enters the optimizer as the
alpha ``a_i = score_i - mean(score)`` over the universe, with ``mu_i = rf + a_i``. The tangency
portfolio depends on the excess return only up to a positive factor, so the unit of the score does
not matter, and centering means the portfolio bets on the ranking, not on a market premium. The
covariance is the RMT-denoised covariance of the same winsorized window as the core ``MV-RMT``;
constraints, dates and costs are those of the core strategies.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pfopt.config import Config
from pfopt.data.eda import get_window
from pfopt.estimators.rmt import rmt_cov
from pfopt.estimators.utils import validate_returns, winsorize_cols
from pfopt.optimizers.markowitz import max_sharpe_long_only
from pfopt.weights import check_weights

logger = logging.getLogger(__name__)

ML_STRATEGIES = {"MV-ML-RIDGE-RMT": "RIDGE", "MV-ML-RF-RMT": "RF"}


def alpha_vector(scores: np.ndarray) -> np.ndarray:
    """Cross-sectionally centered scores (zero sum), the excess return passed to the optimizer."""
    s = np.asarray(scores, dtype=float).ravel()
    if s.size == 0 or not np.isfinite(s).all():
        raise ValueError("scores must be a non-empty finite vector")
    return s - s.mean()


@dataclass(frozen=True)
class MLPortfolioResult:
    """Output of :func:`generate_ml_weights`.

    Attributes
    ----------
    weights : dict of str to pandas.DataFrame
        Per strategy id: rebalance date x symbol weights over all symbols (zero outside the
        universe), non-negative and summing to 1.
    diagnostics : pandas.DataFrame
        One row per date: ``n_universe`` and, per strategy, ``pred_vol_<id>`` (annualized
        ``sqrt(w' Sigma_RMT w)``), ``solver_<id>`` and ``effective_n_<id>``.
    """

    weights: dict[str, pd.DataFrame]
    diagnostics: pd.DataFrame


def generate_ml_weights(
    log_returns: pd.DataFrame,
    predictions: dict[str, pd.DataFrame],
    cfg: Config,
    strategies: Sequence[str] = tuple(ML_STRATEGIES),
) -> MLPortfolioResult:
    """Max-Sharpe weights with ML alphas and RMT covariance at every prediction date.

    Parameters
    ----------
    log_returns : pandas.DataFrame
        Date x Symbol log returns.
    predictions : dict of str to pandas.DataFrame
        Model name (``"RIDGE"``, ``"RF"``) -> date x symbol scores from
        :func:`pfopt.backtest.mlsignal.walk_forward_signals`; finite exactly on the universe of
        the date, and the same universe for every model.
    cfg : Config
        Uses ``universe``, ``data`` (winsorization), ``backtest.max_weight`` and ``run``.
    strategies : sequence of str, default all of :data:`ML_STRATEGIES`
        Strategy ids to build.

    Raises
    ------
    ValueError
        For unknown strategies, missing models, or models whose universes differ.
    """
    unknown = sorted(set(strategies) - set(ML_STRATEGIES))
    if unknown:
        raise ValueError(f"unknown strategy id(s) {unknown}; available: {tuple(ML_STRATEGIES)}")
    models = {s: ML_STRATEGIES[s] for s in strategies}
    missing = sorted(set(models.values()) - set(predictions))
    if missing:
        raise ValueError(f"no predictions for model(s) {missing}")
    reference = predictions[next(iter(models.values()))]
    for model in models.values():
        frame = predictions[model]
        if not frame.index.equals(reference.index) or not frame.columns.equals(reference.columns):
            raise ValueError("prediction frames must share the same dates and symbols")
        if not np.array_equal(np.isfinite(frame.to_numpy()), np.isfinite(reference.to_numpy())):
            raise ValueError("prediction frames must have the same universe at every date")

    t = cfg.universe.lookback_days
    columns = list(reference.columns)
    position = {s: i for i, s in enumerate(columns)}
    arrays = {s: np.zeros((len(reference), len(columns))) for s in models}
    rf_per_day = cfg.run.rf / cfg.run.annualization
    annual = float(np.sqrt(cfg.run.annualization))
    rows: list[dict[str, object]] = []

    for k, date in enumerate(reference.index):
        finite = np.isfinite(reference.iloc[k].to_numpy())
        universe = [c for c, ok in zip(columns, finite, strict=True) if ok]
        if not universe:
            raise ValueError(f"no predictions on {date.date()}")
        window = get_window(log_returns, date, t, universe)
        clipped = winsorize_cols(window.to_numpy(), cfg.data.winsor_lower, cfg.data.winsor_upper)
        sigma = rmt_cov(validate_returns(clipped))
        cols = [position[s] for s in universe]
        row: dict[str, object] = {"n_universe": len(universe)}
        for strategy, model in models.items():
            scores = predictions[model].iloc[k].to_numpy()[finite]
            w, info = max_sharpe_long_only(
                rf_per_day + alpha_vector(scores),
                sigma,
                rf=rf_per_day,
                max_weight=cfg.backtest.max_weight,
                return_info=True,
            )
            arrays[strategy][k, cols] = w
            row[f"pred_vol_{strategy}"] = float(np.sqrt(max(w @ sigma @ w, 0.0))) * annual
            row[f"solver_{strategy}"] = info["solver"]
            row[f"effective_n_{strategy}"] = float(1.0 / np.sum(w**2))
        rows.append(row)
        if (k + 1) % 25 == 0 or k + 1 == len(reference):
            logger.info("ML weights for %d / %d dates", k + 1, len(reference))

    index = pd.DatetimeIndex(reference.index, name="Date")
    frames = {s: pd.DataFrame(arrays[s], index=index, columns=columns) for s in models}
    for frame in frames.values():
        for row_weights in frame.to_numpy():
            check_weights(row_weights, max_weight=cfg.backtest.max_weight)
    return MLPortfolioResult(weights=frames, diagnostics=pd.DataFrame(rows, index=index))
