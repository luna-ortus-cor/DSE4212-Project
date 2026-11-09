"""Tests for the exponentially weighted covariance estimators."""

import numpy as np
import pytest

from pfopt.estimators.ewma import effective_sample_size, ewma_cov, ewma_rmt_cov, ewma_weights
from pfopt.estimators.rmt import rmt_cov
from pfopt.estimators.sample import sample_cov


def make_returns(n_obs: int = 200, n: int = 12, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    market = rng.normal(0.0, 0.01, size=(n_obs, 1))
    return market * rng.uniform(0.5, 1.5, size=(1, n)) + rng.normal(0.0, 0.01, size=(n_obs, n))


def test_weights_sum_to_one_increase_and_halve() -> None:
    w = ewma_weights(100, 10)
    assert w.sum() == pytest.approx(1.0)
    assert (np.diff(w) > 0).all()
    assert w[-1] / w[-11] == pytest.approx(2.0)
    np.testing.assert_allclose(ewma_weights(7, np.inf), np.full(7, 1 / 7))


def test_effective_sample_size_known_values() -> None:
    assert effective_sample_size(np.full(50, 1 / 50)) == pytest.approx(50.0)
    assert effective_sample_size(np.array([1.0, 0.0, 0.0])) == pytest.approx(1.0)
    assert effective_sample_size(np.array([2.0, 2.0])) == pytest.approx(2.0)
    w = ewma_weights(500, 20)
    assert effective_sample_size(w) == pytest.approx(1.0 / np.sum(w**2))
    assert 2 < effective_sample_size(w) < 500
    # long window: T_eff -> (1 + r) / (1 - r) with r = 0.5 ** (1 / h)
    r = 0.5 ** (1 / 20)
    limit = (1 + r) / (1 - r)
    assert effective_sample_size(ewma_weights(5000, 20)) == pytest.approx(limit, rel=1e-3)


def test_infinite_halflife_recovers_the_sample_covariance() -> None:
    x = make_returns()
    np.testing.assert_allclose(ewma_cov(x, np.inf), sample_cov(x), atol=1e-14)
    np.testing.assert_allclose(ewma_rmt_cov(x, np.inf), rmt_cov(x), atol=1e-12)


def test_ewma_cov_is_symmetric_psd_and_weights_recent_data() -> None:
    x = make_returns(300)
    cov = ewma_cov(x, 30)
    np.testing.assert_allclose(cov, cov.T)
    assert np.linalg.eigvalsh(cov).min() > 0
    wild_then_calm = np.vstack([3.0 * x[:150], 0.1 * x[150:]])
    assert np.trace(ewma_cov(wild_then_calm, 20)) < 0.2 * np.trace(sample_cov(wild_then_calm))


def test_ewma_cov_matches_a_loop() -> None:
    x = make_returns(40, 4, seed=3)
    w = ewma_weights(40, 8)
    mean = (w[:, None] * x).sum(axis=0)
    expected = sum(w[i] * np.outer(x[i] - mean, x[i] - mean) for i in range(40))
    expected = expected / (1.0 - np.sum(w**2))
    np.testing.assert_allclose(ewma_cov(x, 8), expected)


def test_ewma_rmt_keeps_the_variances_and_uses_the_effective_sample() -> None:
    x = make_returns(250, 20, seed=5)
    cov, info = ewma_rmt_cov(x, 40, return_info=True)
    np.testing.assert_allclose(np.diag(cov), np.diag(ewma_cov(x, 40)), rtol=1e-10)
    np.testing.assert_allclose(cov, cov.T)
    assert np.linalg.eigvalsh(cov).min() > 0
    t_eff = effective_sample_size(ewma_weights(250, 40))
    assert info["t_eff"] == pytest.approx(t_eff)
    assert info["lam_plus"] == pytest.approx((1 + np.sqrt(20 / t_eff)) ** 2)
    assert 1 <= info["n_signal"] < 20


def test_input_validation() -> None:
    with pytest.raises(ValueError):
        ewma_weights(1, 10)
    with pytest.raises(ValueError):
        ewma_weights(10, 0)
    with pytest.raises(ValueError):
        ewma_cov(np.array([[1.0, np.nan], [0.0, 1.0]]), 5)
