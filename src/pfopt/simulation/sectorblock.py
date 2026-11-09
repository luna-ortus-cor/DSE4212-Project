"""Positive and negative control for the sector-block estimator (PLAN.md step 8.24).

Positive control: a factor model with ``residual.sim_n_groups`` weak group factors, each loading on
its own block of assets, whose volatility (``sim_group_vol``) is set so that the within-group
residual correlation is close to the 0.06 found on the real data in step 8.23. The group factors
are weak enough to stay below the Marchenko-Pastur edge, so the PCA factors do not absorb them.
The sector-block covariance is fitted with the true groups and with ``sim_label_error`` of the
labels wrong (relabelled among themselves), which mimics a sector snapshot that is out of date.

Negative control: the same model without group factors and with arbitrary labels; it shows what
the extra blocks cost when there is no sector structure.

Estimators are scored by the true variance of their long-only GMV portfolio relative to the
oracle, as in step 6.1 (stationary, Gaussian).
"""

from __future__ import annotations

import dataclasses
import logging

import numpy as np
import pandas as pd
from scipy import stats

from pfopt.backtest.sectorblock import covariance_from_config
from pfopt.backtest.strategies import build_covariances
from pfopt.config import Config
from pfopt.estimators.utils import validate_returns
from pfopt.optimizers.gmv import gmv_long_only
from pfopt.simulation.dgp import simulate_returns
from pfopt.simulation.run import (
    LONG_ONLY,
    oracle_portfolios,
    paired_vs_sample,
    true_model,
)

logger = logging.getLogger(__name__)

DESIGNS = ("weak sector factors", "no sector factors")
METHODS = ("GMV-SAMPLE", "GMV-PCA", "GMV-PCA-SECTOR", "GMV-PCA-SECTOR-NOISY")
_SEED_TAGS = (24, 25, 26)


def control_config(cfg: Config, structured: bool) -> Config:
    """Configuration of one control design (group factors present or absent)."""
    rc = cfg.residual
    sim = dataclasses.replace(
        cfg.simulation,
        n_groups=rc.sim_n_groups if structured else 0,
        group_vols=(rc.sim_group_vol,),
        sample_sizes=rc.sim_sample_sizes,
        time_varying=False,
    )
    return dataclasses.replace(cfg, simulation=sim)


def mislabel(groups: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    """Copy of ``groups`` with a random ``fraction`` of the entries shuffled among themselves."""
    out = np.array(groups, dtype=int, copy=True)
    count = int(round(fraction * out.size))
    if count >= 2:
        chosen = rng.choice(out.size, size=count, replace=False)
        out[chosen] = rng.permutation(out[chosen])
    return out


def arbitrary_labels(n_assets: int, n_groups: int, rng: np.random.Generator) -> np.ndarray:
    """Near-equal groups assigned to the assets in random order (labels unrelated to the model)."""
    return rng.permutation(np.arange(n_assets) * n_groups // n_assets)


def evaluate_sample(
    x: np.ndarray,
    sigma: np.ndarray,
    oracle_variance: float,
    groups: np.ndarray,
    noisy_groups: np.ndarray,
    cfg: Config,
) -> dict[str, float]:
    """True-variance ratio (to the oracle) of the long-only GMV weights of every method."""
    x = validate_returns(x)
    covariances, _ = build_covariances(x, cfg)
    cov = {
        "GMV-SAMPLE": covariances["Sample"],
        "GMV-PCA": covariances["PCA"],
        "GMV-PCA-SECTOR": covariance_from_config(x, groups, cfg)[0],
        "GMV-PCA-SECTOR-NOISY": covariance_from_config(x, noisy_groups, cfg)[0],
    }
    out = {}
    for name, matrix in cov.items():
        w = gmv_long_only(matrix)
        out[name] = float(w @ sigma @ w) / oracle_variance
    return out


def run_control(cfg: Config, structured: bool) -> pd.DataFrame:
    """Run one control design for all sample sizes and replications.

    Replication ``r`` at sample size ``T`` draws its returns from ``[seed, 24, T, r]`` and its
    label errors from ``[seed, 25, T, r]``; the labels of the unstructured design come from
    ``[seed, 26]``.

    Returns
    -------
    pandas.DataFrame
        Columns ``distribution`` (the design name, so the frame can be passed to
        :func:`pfopt.simulation.run.paired_vs_sample`), ``T``, ``rep``, ``constraint``,
        ``method`` and ``var_ratio``.
    """
    rc = cfg.residual
    ctrl = control_config(cfg, structured)
    model = true_model(ctrl)
    oracle = oracle_portfolios(model.sigma)
    n = ctrl.simulation.n_assets
    if structured:
        groups = model.groups
    else:
        groups = arbitrary_labels(n, rc.sim_n_groups, np.random.default_rng([cfg.run.seed, 26]))
    design = DESIGNS[0] if structured else DESIGNS[1]
    rows = []
    for n_obs in rc.sim_sample_sizes:
        for rep in range(rc.sim_n_reps):
            x = simulate_returns(
                model, n_obs, np.random.default_rng([cfg.run.seed, 24, n_obs, rep])
            )
            noise_rng = np.random.default_rng([cfg.run.seed, 25, n_obs, rep])
            noisy = mislabel(groups, rc.sim_label_error, noise_rng)
            ratios = evaluate_sample(
                x, oracle.sigma, oracle.variance[LONG_ONLY], groups, noisy, ctrl
            )
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
        logger.info("%s: T=%d done (%d replications)", design, n_obs, rc.sim_n_reps)
    return pd.DataFrame(rows)


def paired_control(scores: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Paired log true-variance ratio against ``GMV-SAMPLE`` (see ``paired_vs_sample``)."""
    return paired_vs_sample(scores, alpha).rename(columns={"distribution": "design"})


def paired_vs_pca(scores: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Paired log true-variance ratio of the sector-block methods against ``GMV-PCA``.

    For every design and sample size, ``log(ratio of the method) - log(ratio of GMV-PCA)`` is
    formed per replication (same draws), then averaged with a normal-approximation interval across
    replications. Negative means the method has lower true variance than PCA.

    Returns
    -------
    pandas.DataFrame
        Columns ``design``, ``T``, ``method``, ``mean_log_ratio``, ``ci_low``, ``ci_high`` and
        ``n``.
    """
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    rows = []
    for (design, n_obs), cell in scores.groupby(["distribution", "T"]):
        wide = cell.pivot(index="rep", columns="method", values="var_ratio")
        for method in ("GMV-PCA-SECTOR", "GMV-PCA-SECTOR-NOISY"):
            d = np.log(wide[method]) - np.log(wide["GMV-PCA"])
            mean, se = float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))
            rows.append(
                {
                    "design": design,
                    "T": int(n_obs),
                    "method": method,
                    "mean_log_ratio": mean,
                    "ci_low": mean - z * se,
                    "ci_high": mean + z * se,
                    "n": len(d),
                }
            )
    return pd.DataFrame(rows)
