"""Tests for the estimated-versus-realized frontier experiment."""

import dataclasses

import numpy as np
import pytest

from pfopt.backtest.frontier import frontier_experiment, realized_risk_return
from pfopt.config import BacktestConfig, Config, EdaConfig
from pfopt.weights import check_weights


def data(t: int = 300, h: int = 60, n: int = 12, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    loadings = rng.choice([-1.0, 1.0], size=(n, 2))
    window = 0.01 * (rng.normal(size=(t, n)) + rng.normal(size=(t, 2)) @ loadings.T) + 0.0004
    future = 0.0004 + 0.01 * rng.normal(size=(h, n))
    return window, future


def test_realized_risk_return_hand_computed() -> None:
    future = np.array([[0.01, 0.02], [0.03, 0.00], [0.02, 0.04]])
    ret, vol = realized_risk_return(np.array([0.5, 0.5]), future, annualization=252)
    daily = future @ np.array([0.5, 0.5])
    assert ret[0] == pytest.approx(daily.mean() * 252)
    assert vol[0] == pytest.approx(daily.std(ddof=1) * np.sqrt(252))
    ret2, vol2 = realized_risk_return(np.array([[1.0, 0.0], [0.0, 1.0]]), future, 252)
    np.testing.assert_allclose(ret2, future.mean(axis=0) * 252)
    np.testing.assert_allclose(vol2, future.std(axis=0, ddof=1) * np.sqrt(252))


def test_realized_risk_return_validates_inputs() -> None:
    with pytest.raises(ValueError, match="H x N"):
        realized_risk_return(np.array([0.5, 0.5]), np.ones((1, 2)), 252)
    with pytest.raises(ValueError, match="H x N"):
        realized_risk_return(np.array([0.5, 0.5]), np.ones((5, 3)), 252)
    bad = np.ones((5, 2))
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        realized_risk_return(np.array([0.5, 0.5]), bad, 252)


def test_experiment_returns_consistent_curves() -> None:
    window, future = data()
    cfg = Config()
    curves = frontier_experiment(window, future, cfg, n_points=8)
    assert set(curves) == {"Sample", "RMT"}
    for curve in curves.values():
        assert curve.weights.shape == (8, 12)
        assert len(curve.est_vol) == len(curve.est_ret) == len(curve.real_vol) == 8
        assert np.all(np.diff(curve.est_ret) > 0)
        assert np.all(np.diff(curve.est_vol) >= -1e-9)
        for w in curve.weights:
            check_weights(w)
        check_weights(curve.max_sharpe_weights)
        # Realized numbers are exactly the evaluation of the stored weights on the future days.
        ret, vol = realized_risk_return(curve.weights, future, cfg.run.annualization)
        np.testing.assert_allclose(curve.real_ret, ret)
        np.testing.assert_allclose(curve.real_vol, vol)
        # Estimated coordinates of the max-Sharpe portfolio lie on the estimated frontier's range.
        assert curve.est_vol.min() - 1e-9 <= curve.ms_est[0] <= curve.est_vol.max() + 1e-9
        assert curve.est_ret.min() - 1e-9 <= curve.ms_est[1] <= curve.est_ret.max() + 1e-9


def test_estimated_frontier_is_optimistic_relative_to_realized_in_noisy_data() -> None:
    # Pure noise with identical true means: whatever the optimizer "finds" is estimation error,
    # so on average the estimated return and Sharpe ratio exceed what the same portfolios deliver.
    return_gaps, sharpe_gaps = [], []
    for seed in range(12):
        rng = np.random.default_rng(seed)
        window = 0.01 * rng.normal(size=(250, 40))
        future = 0.01 * rng.normal(size=(250, 40))
        curve = frontier_experiment(window, future, Config(), n_points=6)["Sample"]
        return_gaps.append(curve.est_ret[-1] - curve.real_ret[-1])
        sharpe_gaps.append(curve.ms_est[1] / curve.ms_est[0] - curve.ms_real[1] / curve.ms_real[0])
    assert np.mean(return_gaps) > 0.1
    assert np.mean(sharpe_gaps) > 0.5


def test_config_cap_and_mu_convention_are_used() -> None:
    window, future = data(seed=1)
    cfg = dataclasses.replace(
        Config(),
        backtest=BacktestConfig(max_weight=0.2, mu_convention="log"),
        eda=EdaConfig(frontier_points=6),
    )
    curves = frontier_experiment(window, future, cfg)
    for curve in curves.values():
        assert curve.weights.shape[0] == 6
        assert curve.weights.max() <= 0.2 + 1e-8
        check_weights(curve.max_sharpe_weights, max_weight=0.2)


def test_inputs_not_mutated_and_deterministic() -> None:
    window, future = data(seed=2)
    w0, f0 = window.copy(), future.copy()
    first = frontier_experiment(window, future, Config(), n_points=5)
    second = frontier_experiment(window, future, Config(), n_points=5)
    np.testing.assert_array_equal(window, w0)
    np.testing.assert_array_equal(future, f0)
    np.testing.assert_array_equal(first["Sample"].weights, second["Sample"].weights)
