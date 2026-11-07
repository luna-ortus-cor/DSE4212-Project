"""Estimated versus realized efficient frontiers (PLAN.md step 3.2).

Markowitz optimization is an "error maximizer": it loads on the assets whose estimated mean or
covariance is most favourable, which are partly the ones with the largest estimation errors. The
experiment below builds long-only efficient frontiers from one estimation window and measures what
the same portfolios (constant weights, rebalanced daily) actually delivered over the following days.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pfopt.config import Config
from pfopt.estimators.mean import expected_returns
from pfopt.estimators.rmt import rmt_cov
from pfopt.estimators.sample import sample_cov
from pfopt.estimators.utils import validate_returns
from pfopt.optimizers.markowitz import efficient_frontier, max_sharpe_long_only


@dataclass(frozen=True)
class FrontierCurve:
    """Estimated and realized risk-return of one covariance estimator's frontier.

    All returns and volatilities are annualized. ``est_*`` are what the optimizer expected from the
    estimation window; ``real_*`` are measured on the future days. ``ms_est`` and ``ms_real`` are
    ``(volatility, return)`` of the maximum-Sharpe portfolio.
    """

    est_vol: np.ndarray
    est_ret: np.ndarray
    real_vol: np.ndarray
    real_ret: np.ndarray
    weights: np.ndarray
    max_sharpe_weights: np.ndarray
    ms_est: tuple[float, float]
    ms_real: tuple[float, float]


def realized_risk_return(
    weights: np.ndarray, future: np.ndarray, annualization: int
) -> tuple[np.ndarray, np.ndarray]:
    """Annualized mean return and volatility of constant-weight portfolios on future returns.

    Parameters
    ----------
    weights : numpy.ndarray
        ``N`` or ``k x N`` weights, held fixed (rebalanced daily to these weights).
    future : numpy.ndarray
        ``H x N`` simple returns after the estimation date (``H >= 2``).
    annualization : int
        Trading days per year.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        Annualized arithmetic mean return (daily mean times ``annualization``) and annualized
        volatility (daily standard deviation, ``ddof=1``, times ``sqrt(annualization)``), one
        entry per portfolio.
    """
    w = np.atleast_2d(np.asarray(weights, dtype=float))
    r = np.asarray(future, dtype=float)
    if r.ndim != 2 or r.shape[0] < 2 or r.shape[1] != w.shape[1]:
        raise ValueError("future must be H x N with H >= 2 and N matching the weights")
    if not np.isfinite(r).all():
        raise ValueError("future returns must be finite")
    daily = r @ w.T
    return daily.mean(axis=0) * annualization, daily.std(axis=0, ddof=1) * np.sqrt(annualization)


def frontier_experiment(
    window: np.ndarray, future: np.ndarray, cfg: Config, n_points: int | None = None
) -> dict[str, FrontierCurve]:
    """Build sample- and RMT-covariance frontiers from a window and evaluate them out of sample.

    Expected returns are ``expected_returns(window, cfg.backtest.mu_convention)`` for both
    frontiers; only the covariance differs. The optional weight cap and the risk-free rate come
    from the config.

    Parameters
    ----------
    window : numpy.ndarray
        ``T x N`` winsorized log returns of the point-in-time universe.
    future : numpy.ndarray
        ``H x N`` simple returns of the same assets after the window (no NaN).
    cfg : Config
        Uses ``backtest.mu_convention``, ``backtest.max_weight``, ``run`` and
        ``eda.frontier_points``.
    n_points : int, optional
        Portfolios per frontier; defaults to ``cfg.eda.frontier_points``.

    Returns
    -------
    dict of str to FrontierCurve
        Keys ``"Sample"`` and ``"RMT"``.
    """
    x = validate_returns(window)
    ann = cfg.run.annualization
    points = cfg.eda.frontier_points if n_points is None else n_points
    cap = cfg.backtest.max_weight
    rf_period = cfg.run.rf / ann
    mu = expected_returns(x, cfg.backtest.mu_convention)
    curves: dict[str, FrontierCurve] = {}
    for name, cov in (("Sample", sample_cov(x)), ("RMT", rmt_cov(x))):
        frontier = efficient_frontier(mu, cov, n_points=points, max_weight=cap)
        real_ret, real_vol = realized_risk_return(frontier.weights, future, ann)
        w_ms = max_sharpe_long_only(mu, cov, rf=rf_period, max_weight=cap)
        ms_ret, ms_vol = realized_risk_return(w_ms, future, ann)
        curves[name] = FrontierCurve(
            est_vol=frontier.volatilities * np.sqrt(ann),
            est_ret=frontier.returns * ann,
            real_vol=real_vol,
            real_ret=real_ret,
            weights=frontier.weights,
            max_sharpe_weights=w_ms,
            ms_est=(float(np.sqrt(w_ms @ cov @ w_ms) * np.sqrt(ann)), float(mu @ w_ms * ann)),
            ms_real=(float(ms_vol[0]), float(ms_ret[0])),
        )
    return curves
