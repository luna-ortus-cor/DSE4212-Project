"""Core strategies built from one estimation window.

One function builds every covariance estimator and every core strategy from a single, already
winsorized ``T x N`` window of log returns, so that the estimator comparison (2.5) and the
backtest engine (4.1) share exactly the same logic and every strategy faces identical conditions.

Constraint note: the optional per-asset cap (``backtest.max_weight``) is enforced for the GMV
strategies. HRP has no mechanism for a cap and ignores it; this must be handled explicitly when the
cap sensitivity is run.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pfopt.config import Config
from pfopt.estimators.hrp import hrp_weights
from pfopt.estimators.mean import expected_returns
from pfopt.estimators.pca_factor import pca_factor_cov
from pfopt.estimators.rmt import rmt_cov
from pfopt.estimators.sample import lw_cov, sample_cov
from pfopt.estimators.utils import cov_to_corr, validate_returns
from pfopt.optimizers.gmv import gmv_long_only
from pfopt.optimizers.markowitz import max_sharpe_long_only

CORE_STRATEGIES = ("EW", "GMV-SAMPLE", "GMV-LW", "GMV-RMT", "GMV-PCA", "HRP", "HRP-RMT")
MARKOWITZ_STRATEGIES = ("MV-SAMPLE", "MV-RMT")
ESTIMATORS = ("Sample", "Ledoit-Wolf", "RMT", "PCA")

# Covariance estimator behind each strategy's weights, used to forecast the strategy's volatility.
# EW has no estimator and is therefore absent.
PREDICTION_COVARIANCE = {
    "GMV-SAMPLE": "Sample",
    "GMV-LW": "Ledoit-Wolf",
    "GMV-RMT": "RMT",
    "GMV-PCA": "PCA",
    "HRP": "Sample",
    "HRP-RMT": "RMT",
    "MV-SAMPLE": "Sample",
    "MV-RMT": "RMT",
}


@dataclass(frozen=True)
class StrategyResult:
    """Output of :func:`core_strategies`.

    Attributes
    ----------
    weights : dict of str to numpy.ndarray
        Weights per strategy id (keys in :data:`CORE_STRATEGIES` order), aligned with the columns
        of the window.
    covariances : dict of str to numpy.ndarray
        Covariance matrix per estimator (keys in :data:`ESTIMATORS` order).
    info : dict
        Diagnostics: ``lw_shrinkage``, ``rmt_n_signal``, ``rmt_lam_plus``, ``pca_K``,
        ``pca_explained`` and ``gmv`` (solver info per GMV strategy).
    """

    weights: dict[str, np.ndarray]
    covariances: dict[str, np.ndarray]
    info: dict[str, object]


def build_covariances(
    x: np.ndarray, cfg: Config
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Sample, Ledoit-Wolf, RMT and PCA-factor covariances of one winsorized window.

    Parameters
    ----------
    x : numpy.ndarray
        Validated ``T x N`` log-return window.
    cfg : Config
        Uses ``pca`` (K rule).

    Returns
    -------
    (dict, dict)
        Covariance per estimator (keys in :data:`ESTIMATORS` order) and the diagnostics
        ``lw_shrinkage``, ``rmt_n_signal``, ``rmt_lam_plus``, ``pca_K`` and ``pca_explained``.
    """
    sample = sample_cov(x)
    lw, delta = lw_cov(x, return_shrinkage=True)
    rmt, rmt_info = rmt_cov(x, return_info=True)
    k_fixed = cfg.pca.k_fixed if cfg.pca.k_rule == "fixed" else None
    pca, pca_info = pca_factor_cov(
        x, K=k_fixed, rule=cfg.pca.k_rule, kmax=cfg.pca.baing_kmax, return_info=True
    )
    covariances = {"Sample": sample, "Ledoit-Wolf": lw, "RMT": rmt, "PCA": pca}
    info = {
        "lw_shrinkage": delta,
        "rmt_n_signal": rmt_info["n_signal"],
        "rmt_lam_plus": rmt_info["lam_plus"],
        "pca_K": pca_info["K"],
        "pca_explained": pca_info["explained"],
    }
    return covariances, info


def core_strategies(window: np.ndarray, cfg: Config) -> StrategyResult:
    """Build all covariance estimators and core strategy weights from one window.

    Parameters
    ----------
    window : numpy.ndarray
        ``T x N`` log returns of the point-in-time universe, already winsorized.
    cfg : Config
        Uses ``pca`` (K rule), ``hrp.linkage`` and ``backtest.max_weight``.

    Returns
    -------
    StrategyResult
        Long-only, fully invested weights for ``EW``, ``GMV-SAMPLE``, ``GMV-LW``, ``GMV-RMT``,
        ``GMV-PCA``, ``HRP`` (sample covariance) and ``HRP-RMT`` (RMT covariance and correlation).
    """
    x = validate_returns(window)
    n = x.shape[1]
    covariances, cov_info = build_covariances(x, cfg)
    sample, lw = covariances["Sample"], covariances["Ledoit-Wolf"]
    rmt, pca = covariances["RMT"], covariances["PCA"]

    cap = cfg.backtest.max_weight
    weights: dict[str, np.ndarray] = {"EW": np.full(n, 1.0 / n)}
    gmv_info: dict[str, object] = {}
    for strategy, cov in (
        ("GMV-SAMPLE", sample),
        ("GMV-LW", lw),
        ("GMV-RMT", rmt),
        ("GMV-PCA", pca),
    ):
        weights[strategy], gmv_info[strategy] = gmv_long_only(cov, max_weight=cap, return_info=True)
    weights["HRP"] = hrp_weights(sample, linkage_method=cfg.hrp.linkage)
    weights["HRP-RMT"] = hrp_weights(rmt, corr=cov_to_corr(rmt), linkage_method=cfg.hrp.linkage)

    info = {**cov_info, "gmv": gmv_info}
    return StrategyResult(weights=weights, covariances=covariances, info=info)


def markowitz_strategies(
    window: np.ndarray, cfg: Config, covariances: dict[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, object]]]:
    """Track-B long-only maximum-Sharpe strategies from one window.

    ``MV-SAMPLE`` uses the sample covariance and ``MV-RMT`` the RMT covariance; both use the same
    expected returns, ``expected_returns(window, cfg.backtest.mu_convention)``, the risk-free rate
    ``cfg.run.rf / cfg.run.annualization`` per day, and the optional weight cap.

    Parameters
    ----------
    window : numpy.ndarray
        ``T x N`` winsorized log returns.
    cfg : Config
        Uses ``backtest.mu_convention``, ``backtest.max_weight`` and ``run``.
    covariances : dict of str to numpy.ndarray
        Must contain ``"Sample"`` and ``"RMT"`` (as in :attr:`StrategyResult.covariances`).

    Returns
    -------
    (dict, dict)
        Weights and optimizer info (see :func:`max_sharpe_long_only`) per strategy id.
    """
    x = validate_returns(window)
    mu = expected_returns(x, cfg.backtest.mu_convention)
    rf_per_day = cfg.run.rf / cfg.run.annualization
    weights: dict[str, np.ndarray] = {}
    infos: dict[str, dict[str, object]] = {}
    for strategy, estimator in (("MV-SAMPLE", "Sample"), ("MV-RMT", "RMT")):
        weights[strategy], infos[strategy] = max_sharpe_long_only(
            mu,
            covariances[estimator],
            rf=rf_per_day,
            max_weight=cfg.backtest.max_weight,
            return_info=True,
        )
    return weights, infos
