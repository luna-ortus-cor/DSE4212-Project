"""Simulation check of nonlinear shrinkage against the other estimators (PLAN.md step 8.2).

The factor-model study of step 6.1 (same true model, same generators ``[seed, 1, T, r]``, so the
draws are identical to the baseline study) is rerun with the quadratic-inverse-shrinkage covariance
added to the sample, Ledoit-Wolf, RMT and PCA estimators. Two scenarios: ``stationary`` (the 6.1
baseline) and ``time-varying`` (the 6.3 design). Scores are the relative Frobenius error of the
covariance and the true variance of the long-only and unconstrained GMV weights relative to the
oracle. The frame layout matches :func:`pfopt.simulation.run.run_grid`, so
:func:`pfopt.simulation.run.paired_vs_sample` applies unchanged.
"""

from __future__ import annotations

import dataclasses
import logging

import numpy as np
import pandas as pd

from pfopt.backtest.strategies import build_covariances
from pfopt.config import Config
from pfopt.estimators.nls import nls_cov
from pfopt.estimators.utils import validate_returns
from pfopt.optimizers.gmv import gmv_long_only, gmv_unconstrained
from pfopt.simulation.dgp import simulate_returns, simulate_time_varying
from pfopt.simulation.run import (
    COVARIANCE,
    COVARIANCE_METHODS,
    LONG_ONLY,
    UNCONSTRAINED,
    Oracle,
    oracle_portfolios,
    true_model,
)

logger = logging.getLogger(__name__)

NLS_METHOD = "GMV-NLS"
METHODS = (*COVARIANCE_METHODS, NLS_METHOD)
SCENARIOS = ("stationary", "time-varying")


def scenario_config(cfg: Config, scenario: str) -> Config:
    """Configuration of one scenario (``time_varying`` switched off or on)."""
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; available: {SCENARIOS}")
    sim = dataclasses.replace(cfg.simulation, time_varying=scenario == "time-varying")
    return dataclasses.replace(cfg, simulation=sim)


def evaluate_sample(x: np.ndarray, oracle: Oracle, cfg: Config) -> list[dict[str, object]]:
    """Fit every estimator (including NLS) on one sample and score it against the truth."""
    x = validate_returns(x)
    sigma = oracle.sigma
    covariances, _ = build_covariances(x, cfg)
    cov = {m: covariances[e] for m, e in COVARIANCE_METHODS.items()}
    cov[NLS_METHOD] = nls_cov(x)
    norm = float(np.linalg.norm(sigma))
    nan = float("nan")
    rows: list[dict[str, object]] = []
    for method, matrix in cov.items():
        rows.append(
            {
                "method": method,
                "constraint": COVARIANCE,
                "frob_error": float(np.linalg.norm(matrix - sigma)) / norm,
                "var_ratio": nan,
            }
        )
        for constraint, solver in ((LONG_ONLY, gmv_long_only), (UNCONSTRAINED, gmv_unconstrained)):
            w = solver(matrix)
            ratio = float(w @ sigma @ w) / oracle.variance[constraint]
            rows.append(
                {"method": method, "constraint": constraint, "frob_error": nan, "var_ratio": ratio}
            )
    return rows


def run_nls_grid(cfg: Config, scenario: str) -> pd.DataFrame:
    """Run all sample sizes and replications of one scenario.

    Returns
    -------
    pandas.DataFrame
        Columns ``distribution`` (the scenario name), ``T``, ``rep``, ``method``, ``constraint``,
        ``frob_error`` and ``var_ratio``; Gaussian returns.
    """
    ctrl = scenario_config(cfg, scenario)
    sim = ctrl.simulation
    model = true_model(ctrl)
    oracle = None if sim.time_varying else oracle_portfolios(model.sigma)
    rows: list[dict[str, object]] = []
    for n_obs in sim.sample_sizes:
        for rep in range(sim.n_reps):
            rng = np.random.default_rng([cfg.run.seed, 1, n_obs, rep])
            if sim.time_varying:
                x, forward = simulate_time_varying(model, n_obs, sim.horizon_days, rng, sim)
                scored = evaluate_sample(x, oracle_portfolios(forward), ctrl)
            else:
                x = simulate_returns(model, n_obs, rng)
                scored = evaluate_sample(x, oracle, ctrl)
            key = {"distribution": scenario, "T": n_obs, "rep": rep}
            rows.extend({**key, **row} for row in scored)
        logger.info("%s: T=%d done (%d replications)", scenario, n_obs, sim.n_reps)
    return pd.DataFrame(rows)
