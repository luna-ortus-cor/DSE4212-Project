"""Covariance forecast loss on random portfolios.

The GMV backtest scores a covariance estimator only through the portfolio that optimizes against
it, where selection bias is strongest. Here every estimator forecasts the variance of the *same*
set of random long-only portfolios (Dirichlet weights, plus the equal-weight portfolio), and the
forecast ``w' Sigma_hat w`` is compared with the realized variance of the following holding period,
``RV = mean_d (w' R_d)^2`` over the next ``backtest.rebalance_days`` trading days (simple returns,
constant weights, zero mean). The loss is QLIKE,

    QLIKE = RV / F - log(RV / F) - 1  >= 0,

which is zero when the forecast equals the realized variance; ``log(F / RV)`` is also reported as a
bias (negative = the risk is under-forecast). Differences against the sample covariance are
averaged over rebalance dates and tested with a Newey-West (Bartlett) standard error, Holm-adjusted
within each portfolio family. The weights and forecasts use data up to the rebalance date only.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy import stats

from pfopt.backtest.inference import adjust_pvalues
from pfopt.backtest.residual import sector_codes
from pfopt.backtest.sectorblock import covariance_from_config, sector_categories
from pfopt.backtest.strategies import build_covariances
from pfopt.config import Config
from pfopt.data.eda import get_window
from pfopt.data.universe import rebalance_dates, select_universe
from pfopt.estimators.nls import nls_cov
from pfopt.estimators.utils import validate_returns, winsorize_cols

logger = logging.getLogger(__name__)

REFERENCE = "Sample"
ESTIMATORS = ("Sample", "Ledoit-Wolf", "RMT", "PCA", "NLS", "PCA-SECTOR")
EQUAL_WEIGHT = "equal weight"
_SEED_TAG = 28
_MIN_VARIANCE = 1e-18


def family_name(alpha: float) -> str:
    """Label of the Dirichlet family with concentration ``alpha``."""
    return f"Dirichlet({alpha:g})"


def random_portfolios(
    n_assets: int, n_portfolios: int, alpha: float, rng: np.random.Generator
) -> np.ndarray:
    """``n_portfolios x n_assets`` long-only, fully invested Dirichlet(``alpha``) weights."""
    if n_assets < 1 or n_portfolios < 1 or alpha <= 0:
        raise ValueError("need n_assets >= 1, n_portfolios >= 1 and alpha > 0")
    return rng.dirichlet(np.full(n_assets, alpha), size=n_portfolios)


def forecast_variance(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Forecast variance ``w' Sigma w`` of every row of ``weights``."""
    return np.einsum("pi,ij,pj->p", weights, cov, weights)


def realized_variance(weights: np.ndarray, future_returns: np.ndarray) -> np.ndarray:
    """Mean squared daily portfolio return of every row of ``weights`` over ``future_returns``."""
    if future_returns.ndim != 2 or future_returns.shape[0] < 1:
        raise ValueError("future_returns must be a non-empty days x assets array")
    return np.mean((future_returns @ weights.T) ** 2, axis=0)


def qlike(realized: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    """QLIKE loss ``RV / F - log(RV / F) - 1`` (zero when ``RV == F``)."""
    ratio = np.maximum(np.asarray(realized, dtype=float), _MIN_VARIANCE) / np.asarray(forecast)
    return ratio - np.log(ratio) - 1.0


def forecast_loss_dates(
    log_returns: pd.DataFrame,
    simple_returns: pd.DataFrame,
    dollar_volume: pd.DataFrame,
    sectors: pd.Series,
    cfg: Config,
) -> pd.DataFrame:
    """QLIKE loss and bias of every estimator for every rebalance date and portfolio family.

    Rebalance dates without a full holding period of ``backtest.rebalance_days`` later trading days
    are dropped. The portfolios of date ``k`` are drawn from ``[seed, 28, k]``, identical for every
    estimator. Missing returns of held assets count as 0, as in the backtest.

    Returns
    -------
    pandas.DataFrame
        Columns ``date``, ``family``, ``estimator``, ``qlike`` and ``log_bias`` (both averaged over
        the portfolios of the family on that date).
    """
    n, t, w = cfg.universe.n_assets, cfg.universe.lookback_days, cfg.universe.dollar_volume_window
    horizon = cfg.backtest.rebalance_days
    fc = cfg.forecast
    dates = rebalance_dates(log_returns, t, n, horizon)
    categories = sector_categories(sectors)
    rows: list[dict[str, object]] = []
    for k, date in enumerate(dates):
        pos = simple_returns.index.get_loc(date)
        if pos + horizon >= len(simple_returns):
            continue
        universe = select_universe(log_returns, dollar_volume, date, t, n, w)
        window = get_window(log_returns, date, t, universe)
        clipped = winsorize_cols(window.to_numpy(), cfg.data.winsor_lower, cfg.data.winsor_upper)
        x = validate_returns(clipped)
        covariances, _ = build_covariances(x, cfg)
        groups = sector_codes(universe, sectors, categories)
        estimates: Mapping[str, np.ndarray] = {
            **covariances,
            "NLS": nls_cov(x),
            "PCA-SECTOR": covariance_from_config(x, groups, cfg)[0],
        }
        future = simple_returns.iloc[pos + 1 : pos + 1 + horizon][list(universe)].to_numpy()
        future = np.nan_to_num(future, nan=0.0)
        rng = np.random.default_rng([cfg.run.seed, _SEED_TAG, k])
        families = {EQUAL_WEIGHT: np.full((1, len(universe)), 1.0 / len(universe))}
        for alpha in fc.dirichlet_alphas:
            families[family_name(alpha)] = random_portfolios(
                len(universe), fc.n_portfolios, alpha, rng
            )
        for family, weights in families.items():
            realized = realized_variance(weights, future)
            for name in ESTIMATORS:
                forecast = forecast_variance(weights, estimates[name])
                rows.append(
                    {
                        "date": date,
                        "family": family,
                        "estimator": name,
                        "qlike": float(qlike(realized, forecast).mean()),
                        "log_bias": float(
                            np.log(forecast / np.maximum(realized, _MIN_VARIANCE)).mean()
                        ),
                    }
                )
    if not rows:
        raise ValueError("no rebalance date has a full holding period")
    return pd.DataFrame(rows)


def newey_west_mean(values: np.ndarray, lag: int) -> tuple[float, float]:
    """Mean of a series and its Newey-West (Bartlett kernel, ``lag``) standard error."""
    x = np.asarray(values, dtype=float)
    n = x.size
    if n < 2 or lag < 0:
        raise ValueError("need at least two values and lag >= 0")
    centered = x - x.mean()
    variance = float(centered @ centered) / n
    for j in range(1, min(lag, n - 1) + 1):
        weight = 1.0 - j / (lag + 1.0)
        variance += 2.0 * weight * float(centered[j:] @ centered[:-j]) / n
    return float(x.mean()), float(np.sqrt(max(variance, 0.0) / n))


def summarize_loss(dates: pd.DataFrame, lag: int, alpha: float = 0.05) -> pd.DataFrame:
    """Mean loss and paired difference against the sample covariance, per family and estimator.

    Returns
    -------
    pandas.DataFrame
        Columns ``family``, ``estimator``, ``n_dates``, ``mean_qlike``, ``mean_log_bias``,
        ``diff_qlike`` (estimator minus sample; negative = better), ``se``, ``ci_low``,
        ``ci_high``, ``p_value`` (normal, two-sided) and ``p_holm`` (within the family; NaN for
        the reference).
    """
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    rows = []
    for family, block in dates.groupby("family", sort=False):
        wide = block.pivot(index="date", columns="estimator", values="qlike")
        bias = block.pivot(index="date", columns="estimator", values="log_bias")
        family_rows = []
        for name in wide.columns:
            if name == REFERENCE:
                mean, se, p = 0.0, float("nan"), float("nan")
            else:
                mean, se = newey_west_mean((wide[name] - wide[REFERENCE]).to_numpy(), lag)
                p = float(2.0 * stats.norm.sf(abs(mean / se))) if se > 0 else float("nan")
            family_rows.append(
                {
                    "family": family,
                    "estimator": name,
                    "n_dates": len(wide),
                    "mean_qlike": float(wide[name].mean()),
                    "mean_log_bias": float(bias[name].mean()),
                    "diff_qlike": mean,
                    "se": se,
                    "ci_low": mean - z * se,
                    "ci_high": mean + z * se,
                    "p_value": p,
                }
            )
        tested = [r for r in family_rows if r["estimator"] != REFERENCE]
        adjusted = adjust_pvalues([r["p_value"] for r in tested], "holm")
        for r, p_adj in zip(tested, adjusted, strict=True):
            r["p_holm"] = float(p_adj)
        for r in family_rows:
            r.setdefault("p_holm", float("nan"))
        rows.extend(family_rows)
    return pd.DataFrame(rows)
