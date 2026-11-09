"""Calibration and positive control for the non-stationarity test (PLAN.md step 8.22).

**Calibration.** In the time-varying design of step 6.3 each factor has a stationary AR(1) log
volatility ``x_t`` with standard deviation ``sd`` and daily autocorrelation ``phi``; the variance
scale is ``s_t^2 = exp(2 x_t - 2 v)``. For non-overlapping blocks of ``b`` days the log
realized variance of the equal-weight market return is, to a good approximation,
``const + 2 x_block + noise`` with the noise variance of a log chi-square, ``trigamma(b / 2)`` for
Gaussian returns. The moment estimators are therefore

    sd^2 = (Var[y] - trigamma(b / 2)) / 4,
    rho_x = Corr(y_j, y_{j+1}) * Var[y] / (Var[y] - trigamma(b / 2)),
    phi = rho_x ** (1 / b),

with ``y`` the block log realized variances. The approximation treats a block as one draw of the
daily process: block means are a little more persistent and less dispersed than the daily
process, so for ``sd = 0.5`` and ``phi = 0.98`` the estimates are about 7% too small and 0.007
too large. Heavy tails make the Gaussian noise correction too small (``sd`` then biased upwards).

**Positive control.** The estimators of the non-stationarity test, fitted on simulated samples and
scored by the true variance of their long-only GMV portfolio relative to the oracle, in three
designs: stationary, time-varying with the a priori parameters, time-varying with the calibrated
ones. If time-weighting helps only in the time-varying designs, the test can detect
non-stationarity.
"""

from __future__ import annotations

import dataclasses
import logging

import numpy as np
import pandas as pd
from scipy import special

from pfopt.backtest.strategies import build_covariances
from pfopt.config import Config, SimulationConfig
from pfopt.estimators.ewma import ewma_cov, ewma_rmt_cov
from pfopt.estimators.utils import validate_returns
from pfopt.optimizers.gmv import gmv_long_only
from pfopt.simulation.dgp import FactorModel, simulate_returns, simulate_time_varying
from pfopt.simulation.run import LONG_ONLY, oracle_portfolios, paired_vs_sample, true_model

logger = logging.getLogger(__name__)

DESIGNS = ("stationary", "time-varying (a priori)", "time-varying (calibrated)")
BASE_METHODS = ("GMV-SAMPLE", "GMV-RMT", "GMV-PCA")


def calibrate_market_volatility(market: np.ndarray, block_days: int = 21) -> dict[str, float]:
    """Moment estimates of the log-volatility dynamics from a daily market return series.

    Parameters
    ----------
    market : numpy.ndarray
        Daily returns of the market proxy (finite, zero mean assumed).
    block_days : int, default 21
        Block length ``b``.

    Returns
    -------
    dict
        ``vol_log_sd``, ``vol_persistence``, ``n_blocks``, ``block_var`` (variance of the block log
        realized variance) and ``noise_var`` (its sampling-noise part).

    Raises
    ------
    ValueError
        If there are fewer than 10 blocks or the signal variance is not positive.
    """
    r = np.asarray(market, dtype=float)
    if r.ndim != 1 or not np.isfinite(r).all():
        raise ValueError("market must be a finite 1-D array")
    n_blocks = r.size // block_days
    if n_blocks < 10:
        raise ValueError(f"need at least 10 blocks of {block_days} days, got {n_blocks}")
    blocks = r[: n_blocks * block_days].reshape(n_blocks, block_days)
    y = np.log((blocks**2).sum(axis=1))
    var_y = float(y.var(ddof=1))
    noise = float(special.polygamma(1, block_days / 2.0))
    signal = var_y - noise
    if signal <= 0:
        raise ValueError("block variance does not exceed the sampling noise; cannot calibrate")
    yc = y - y.mean()
    rho_y = float((yc[:-1] * yc[1:]).sum() / (yc**2).sum())
    rho_x = min(max(rho_y * var_y / signal, 0.0), 0.999)
    return {
        "vol_log_sd": float(np.sqrt(signal) / 2.0),
        "vol_persistence": float(rho_x ** (1.0 / block_days)),
        "n_blocks": float(n_blocks),
        "block_var": var_y,
        "noise_var": noise,
    }


def methods(halflives: tuple[int, ...]) -> tuple[str, ...]:
    """Method names of the positive control, in display order."""
    ewma = [m for h in halflives for m in (f"GMV-EWMA{h}", f"GMV-EWMA{h}-RMT")]
    return (*BASE_METHODS, *ewma)


def design_config(
    sim: SimulationConfig, design: str, calibrated: dict[str, float]
) -> SimulationConfig:
    """Simulation settings of one design.

    ``calibrated`` holds ``vol_log_sd`` and ``vol_persistence`` (used by the calibrated design).
    """
    if design == DESIGNS[0]:
        return dataclasses.replace(sim, time_varying=False)
    if design == DESIGNS[1]:
        return dataclasses.replace(sim, time_varying=True)
    if design == DESIGNS[2]:
        return dataclasses.replace(
            sim,
            time_varying=True,
            vol_log_sd=calibrated["vol_log_sd"],
            vol_persistence=calibrated["vol_persistence"],
        )
    raise ValueError(f"unknown design {design!r}")


def evaluate_sample_ewma(
    x: np.ndarray, sigma: np.ndarray, oracle_variance: float, cfg: Config
) -> dict[str, float]:
    """True-variance ratio (to the oracle) of the long-only GMV weights of every method."""
    x = validate_returns(x)
    covariances, _ = build_covariances(x, cfg)
    cov = {
        "GMV-SAMPLE": covariances["Sample"],
        "GMV-RMT": covariances["RMT"],
        "GMV-PCA": covariances["PCA"],
    }
    for h in cfg.nonstationarity.halflives:
        cov[f"GMV-EWMA{h}"] = ewma_cov(x, h)
        cov[f"GMV-EWMA{h}-RMT"] = ewma_rmt_cov(x, h)
    out = {}
    for name, matrix in cov.items():
        w = gmv_long_only(matrix)
        out[name] = float(w @ sigma @ w) / oracle_variance
    return out


def run_design(
    cfg: Config, design: str, calibrated: dict[str, float], model: FactorModel | None = None
) -> pd.DataFrame:
    """Run one design for all sample sizes and replications.

    The generator of replication ``r`` at sample size ``T`` is seeded with ``[seed, 5, T, r]``, so
    all designs and methods see the same underlying draws.

    Returns
    -------
    pandas.DataFrame
        Columns ``distribution`` (the design name, so the frame can be passed to
        :func:`pfopt.simulation.run.paired_vs_sample`), ``T``, ``rep``, ``constraint``, ``method``
        and ``var_ratio``.
    """
    ns = cfg.nonstationarity
    sim = design_config(cfg.simulation, design, calibrated)
    model = true_model(cfg) if model is None else model
    static = None if sim.time_varying else oracle_portfolios(model.sigma)
    rows = []
    for n_obs in ns.sim_sample_sizes:
        for rep in range(ns.sim_n_reps):
            rng = np.random.default_rng([cfg.run.seed, 5, n_obs, rep])
            if sim.time_varying:
                x, forward = simulate_time_varying(model, n_obs, sim.horizon_days, rng, sim)
                oracle = oracle_portfolios(forward)
            else:
                x, oracle = simulate_returns(model, n_obs, rng), static
            ratios = evaluate_sample_ewma(x, oracle.sigma, oracle.variance[LONG_ONLY], cfg)
            rows.extend(
                {
                    "distribution": design,
                    "T": n_obs,
                    "rep": rep,
                    "constraint": LONG_ONLY,
                    "method": method,
                    "var_ratio": ratio,
                }
                for method, ratio in ratios.items()
            )
        logger.info("%s: T=%d done (%d replications)", design, n_obs, ns.sim_n_reps)
    return pd.DataFrame(rows)


def paired_control(scores: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Paired log true-variance ratio against ``GMV-SAMPLE`` (see ``paired_vs_sample``)."""
    return paired_vs_sample(scores, alpha).rename(columns={"distribution": "design"})
