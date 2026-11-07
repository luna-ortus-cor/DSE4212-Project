"""Tests for the core strategy builder."""

import dataclasses

import numpy as np
import pytest

from pfopt.backtest.strategies import CORE_STRATEGIES, ESTIMATORS, core_strategies
from pfopt.config import BacktestConfig, Config, HRPConfig, PCAConfig
from pfopt.estimators.utils import check_psd
from pfopt.weights import check_weights


def window(t: int = 300, n: int = 30, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    loadings = rng.choice([-1.0, 1.0], size=(n, 2))
    return 0.01 * (rng.normal(size=(t, n)) + rng.normal(size=(t, 2)) @ loadings.T)


@pytest.fixture(scope="module")
def result():
    return core_strategies(window(), Config())


def test_keys_and_validity(result) -> None:
    assert tuple(result.weights) == CORE_STRATEGIES
    assert tuple(result.covariances) == ESTIMATORS
    for name, w in result.weights.items():
        assert w.shape == (30,), name
        check_weights(w)
    for cov in result.covariances.values():
        assert cov.shape == (30, 30)
        check_psd(cov)
    np.testing.assert_allclose(result.weights["EW"], np.full(30, 1 / 30))


def test_gmv_is_optimal_under_its_own_covariance(result) -> None:
    pairs = [
        ("GMV-SAMPLE", "Sample", ["EW", "HRP"]),
        ("GMV-LW", "Ledoit-Wolf", ["EW"]),
        ("GMV-RMT", "RMT", ["EW", "HRP-RMT"]),
        ("GMV-PCA", "PCA", ["EW"]),
    ]
    for strategy, estimator, others in pairs:
        cov = result.covariances[estimator]
        own = result.weights[strategy] @ cov @ result.weights[strategy]
        for other in others:
            w = result.weights[other]
            assert own <= w @ cov @ w + 1e-12, (strategy, other)


def test_diagnostics(result) -> None:
    info = result.info
    assert 0.0 <= info["lw_shrinkage"] <= 1.0
    assert info["rmt_n_signal"] >= 0 and info["rmt_lam_plus"] > 1.0
    assert info["pca_K"] >= 1 and 0.0 < info["pca_explained"] <= 1.0
    assert set(info["gmv"]) == {"GMV-SAMPLE", "GMV-LW", "GMV-RMT", "GMV-PCA"}
    assert all("solver" in entry for entry in info["gmv"].values())


def test_cap_applies_to_gmv_strategies() -> None:
    cfg = dataclasses.replace(Config(), backtest=BacktestConfig(max_weight=0.1))
    res = core_strategies(window(), cfg)
    for strategy in ("GMV-SAMPLE", "GMV-LW", "GMV-RMT", "GMV-PCA"):
        check_weights(res.weights[strategy], max_weight=0.1)


def test_fixed_pca_rule_and_hrp_linkage_come_from_config() -> None:
    cfg = dataclasses.replace(
        Config(), pca=PCAConfig(k_rule="fixed", k_fixed=2), hrp=HRPConfig(linkage="average")
    )
    res = core_strategies(window(), cfg)
    assert res.info["pca_K"] == 2
    check_weights(res.weights["HRP"])
    check_weights(res.weights["HRP-RMT"])


def test_deterministic_and_input_unchanged() -> None:
    x = window(seed=1)
    before = x.copy()
    first, second = core_strategies(x, Config()), core_strategies(x, Config())
    for name in CORE_STRATEGIES:
        np.testing.assert_array_equal(first.weights[name], second.weights[name])
    np.testing.assert_array_equal(x, before)


def test_invalid_window_raises() -> None:
    with pytest.raises(ValueError, match="2-D"):
        core_strategies(np.arange(5.0), Config())
    bad = window()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        core_strategies(bad, Config())
