"""Simulation study runner and summaries (PLAN.md step 6.1).

For every grid cell ``(distribution, T)`` and replication, returns are simulated from a fixed true
factor model, every estimator is fitted on the sample, and three accuracy measures are recorded:

- ``frob_error``: relative Frobenius error ``||Sigma_hat - Sigma||_F / ||Sigma||_F`` (covariance
  estimators only);
- ``var_ratio``: the *true* variance ``w' Sigma w`` of the estimated GMV weights divided by the
  oracle GMV variance (same constraint, true ``Sigma``); it is at least 1 by construction;
- ``l1_dist``: ``sum |w - w_oracle|`` (``0`` = oracle weights, at most 2 for long-only).

Methods are the four GMV strategies (sample, Ledoit-Wolf, RMT, PCA covariance), HRP (sample
covariance) and the equal-weight portfolio. Unconstrained weights exist for the four GMV methods;
HRP and EW are long-only by construction.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from pfopt.backtest.strategies import build_covariances
from pfopt.config import Config
from pfopt.estimators.hrp import hrp_weights
from pfopt.estimators.utils import validate_returns
from pfopt.optimizers.gmv import gmv_long_only, gmv_unconstrained
from pfopt.simulation.dgp import FactorModel, make_factor_model, simulate_returns

logger = logging.getLogger(__name__)

COVARIANCE_METHODS = {
    "GMV-SAMPLE": "Sample",
    "GMV-LW": "Ledoit-Wolf",
    "GMV-RMT": "RMT",
    "GMV-PCA": "PCA",
}
METHODS = (*COVARIANCE_METHODS, "HRP", "EW")
COVARIANCE, LONG_ONLY, UNCONSTRAINED = "covariance", "long-only", "unconstrained"
METRICS = ("frob_error", "var_ratio", "l1_dist")
SAMPLE = "GMV-SAMPLE"


@dataclass(frozen=True)
class Oracle:
    """Optimal GMV portfolios under the true covariance.

    Attributes
    ----------
    sigma : numpy.ndarray
        True covariance.
    weights : dict of str to numpy.ndarray
        Oracle weights per constraint (``"long-only"``, ``"unconstrained"``).
    variance : dict of str to float
        Oracle variance ``w' Sigma w`` per constraint.
    """

    sigma: np.ndarray
    weights: dict[str, np.ndarray]
    variance: dict[str, float]


def oracle_portfolios(sigma: np.ndarray) -> Oracle:
    """Long-only and unconstrained GMV portfolios under the true covariance ``sigma``."""
    weights = {LONG_ONLY: gmv_long_only(sigma), UNCONSTRAINED: gmv_unconstrained(sigma)}
    variance = {name: float(w @ sigma @ w) for name, w in weights.items()}
    return Oracle(sigma=sigma, weights=weights, variance=variance)


def evaluate_sample(
    x: np.ndarray, oracle: Oracle, cfg: Config
) -> tuple[list[dict[str, object]], dict[str, float]]:
    """Fit every estimator on one simulated sample and score it against the truth.

    Parameters
    ----------
    x : numpy.ndarray
        ``T x N`` simulated returns.
    oracle : Oracle
        Truth for this model.
    cfg : Config
        Uses ``pca`` and ``hrp``.

    Returns
    -------
    (list of dict, dict)
        One row per (method, constraint) with the keys ``method``, ``constraint`` and the
        metrics in :data:`METRICS` (NaN where not defined), and the detection diagnostics
        ``rmt_n_signal``, ``pca_K`` and ``lw_shrinkage``.
    """
    x = validate_returns(x)
    n = x.shape[1]
    sigma = oracle.sigma
    covariances, cov_info = build_covariances(x, cfg)
    norm = float(np.linalg.norm(sigma))
    nan = float("nan")
    rows: list[dict[str, object]] = []

    def add(method: str, constraint: str, **values: float) -> None:
        rows.append({"method": method, "constraint": constraint, **dict.fromkeys(METRICS, nan)})
        rows[-1].update(values)

    def weight_row(method: str, constraint: str, w: np.ndarray) -> None:
        reference = oracle.weights[constraint]
        add(
            method,
            constraint,
            var_ratio=float(w @ sigma @ w) / oracle.variance[constraint],
            l1_dist=float(np.abs(w - reference).sum()),
        )

    for method, estimator in COVARIANCE_METHODS.items():
        cov = covariances[estimator]
        add(method, COVARIANCE, frob_error=float(np.linalg.norm(cov - sigma)) / norm)
        weight_row(method, LONG_ONLY, gmv_long_only(cov))
        weight_row(method, UNCONSTRAINED, gmv_unconstrained(cov))
    weight_row("HRP", LONG_ONLY, hrp_weights(covariances["Sample"], linkage_method=cfg.hrp.linkage))
    weight_row("EW", LONG_ONLY, np.full(n, 1.0 / n))
    info = {
        "rmt_n_signal": float(cov_info["rmt_n_signal"]),
        "pca_K": float(cov_info["pca_K"]),
        "lw_shrinkage": float(cov_info["lw_shrinkage"]),
    }
    return rows, info


def true_model(cfg: Config) -> FactorModel:
    """The true factor model of the study, drawn once from the global seed."""
    return make_factor_model(
        cfg.simulation, cfg.run.annualization, np.random.default_rng([cfg.run.seed, 0])
    )


def run_grid(
    cfg: Config,
    distribution: str,
    model: FactorModel | None = None,
    on_cell: Callable[[int], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all sample sizes and replications for one return distribution.

    The generator of replication ``r`` at sample size ``T`` is seeded with ``[seed, 1, T, r]``,
    independent of the grid and the number of replications, and gives the same underlying normal
    draws for both distributions.

    Parameters
    ----------
    cfg : Config
        Uses ``simulation``, ``run``, ``pca`` and ``hrp``.
    distribution : {"gaussian", "student_t"}
        Return distribution.
    model : FactorModel, optional
        True model; by default :func:`true_model`.
    on_cell : callable, optional
        Called with the sample size after each sample size is finished.

    Returns
    -------
    (pandas.DataFrame, pandas.DataFrame)
        Long-format scores (``distribution``, ``T``, ``rep``, ``method``, ``constraint`` and the
        metrics) and per-replication detection diagnostics (``distribution``, ``T``, ``rep``,
        ``rmt_n_signal``, ``pca_K``, ``lw_shrinkage``).
    """
    sim = cfg.simulation
    model = true_model(cfg) if model is None else model
    oracle = oracle_portfolios(model.sigma)
    score_rows: list[dict[str, object]] = []
    info_rows: list[dict[str, object]] = []
    for n_obs in sim.sample_sizes:
        for rep in range(sim.n_reps):
            rng = np.random.default_rng([cfg.run.seed, 1, n_obs, rep])
            x = simulate_returns(model, n_obs, rng, distribution, sim.t_df)
            rows, info = evaluate_sample(x, oracle, cfg)
            key = {"distribution": distribution, "T": n_obs, "rep": rep}
            score_rows.extend({**key, **row} for row in rows)
            info_rows.append({**key, **info})
        logger.info("%s: T=%d done (%d replications)", distribution, n_obs, sim.n_reps)
        if on_cell is not None:
            on_cell(n_obs)
    return pd.DataFrame(score_rows), pd.DataFrame(info_rows)


def summarize(scores: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Mean, standard error, normal confidence band and median of every metric across replications.

    Returns
    -------
    pandas.DataFrame
        Columns ``distribution``, ``T``, ``method``, ``constraint``, ``metric``, ``n``, ``mean``,
        ``se``, ``ci_low``, ``ci_high``, ``median``.
    """
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    keys = ["distribution", "T", "method", "constraint"]
    long = scores.melt(id_vars=[*keys, "rep"], value_vars=list(METRICS), var_name="metric")
    long = long.dropna(subset=["value"])
    grouped = long.groupby([*keys, "metric"])["value"]
    out = grouped.agg(n="count", mean="mean", sd="std", median="median").reset_index()
    out["se"] = out["sd"] / np.sqrt(out["n"])
    out["ci_low"] = out["mean"] - z * out["se"]
    out["ci_high"] = out["mean"] + z * out["se"]
    columns = [*keys, "metric", "n", "mean", "se", "ci_low", "ci_high", "median"]
    return out[columns]


def paired_vs_sample(scores: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Paired comparison of true GMV variance against the sample-covariance GMV.

    For every (distribution, T, constraint) cell and method, the paired per-replication
    ``log(var_ratio_method) - log(var_ratio_sample)`` (negative = lower true variance than the
    sample covariance), its normal confidence interval, the share of replications in which the
    method beats the sample covariance, and the share in which it has the lowest true variance of
    all methods available under that constraint.

    Returns
    -------
    pandas.DataFrame
        Columns ``distribution``, ``T``, ``constraint``, ``method``, ``mean_log_ratio``,
        ``ci_low``, ``ci_high``, ``beats_sample``, ``best_rate``.
    """
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    weights = scores[scores["constraint"].isin([LONG_ONLY, UNCONSTRAINED])]
    rows = []
    for (dist, n_obs, constraint), cell in weights.groupby(["distribution", "T", "constraint"]):
        wide = cell.pivot(index="rep", columns="method", values="var_ratio")
        best = wide.idxmin(axis=1).value_counts(normalize=True)
        for method in wide.columns:
            diff = np.log(wide[method]) - np.log(wide[SAMPLE])
            se = float(diff.std(ddof=1)) / np.sqrt(len(diff))
            rows.append(
                {
                    "distribution": dist,
                    "T": n_obs,
                    "constraint": constraint,
                    "method": method,
                    "mean_log_ratio": float(diff.mean()),
                    "ci_low": float(diff.mean()) - z * se,
                    "ci_high": float(diff.mean()) + z * se,
                    "beats_sample": float((wide[method] < wide[SAMPLE]).mean()),
                    "best_rate": float(best.get(method, 0.0)),
                }
            )
    return pd.DataFrame(rows)


def winner_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Method with the lowest mean true-variance ratio per (distribution, T, constraint)."""
    ratio = summary[summary["metric"] == "var_ratio"]
    best = ratio.loc[ratio.groupby(["distribution", "T", "constraint"])["mean"].idxmin()]
    return best[["distribution", "T", "constraint", "method", "mean"]].reset_index(drop=True)
