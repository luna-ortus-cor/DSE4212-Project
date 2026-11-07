"""Tests for the factor-model data-generating process."""

import numpy as np
import pytest
from scipy import stats

from pfopt.config import SimulationConfig
from pfopt.simulation.dgp import make_factor_model, simulate_returns

CFG = SimulationConfig(n_assets=12, n_groups=2, sample_sizes=(30,), n_reps=2)


def model(cfg: SimulationConfig = CFG, seed: int = 0):
    return make_factor_model(cfg, 252, np.random.default_rng(seed))


def test_covariance_structure_is_symmetric_positive_definite() -> None:
    m = model()
    sigma = m.sigma
    assert sigma.shape == (12, 12) and m.n_factors == 3 and m.n_assets == 12
    np.testing.assert_allclose(sigma, sigma.T)
    assert np.linalg.eigvalsh(sigma).min() > 0
    expected = m.loadings @ np.diag(m.factor_var) @ m.loadings.T + np.diag(m.idio_var)
    np.testing.assert_allclose(sigma, expected)


def test_group_blocks_are_disjoint_balanced_and_loadings_in_range() -> None:
    m = model(SimulationConfig(n_assets=13, n_groups=3, sample_sizes=(30,), n_reps=2))
    assert m.n_factors == 4
    sizes = np.bincount(m.groups)
    assert sizes.sum() == 13 and sizes.max() - sizes.min() <= 1
    market = m.loadings[:, 0]
    assert market.min() >= 0.5 and market.max() <= 1.5
    for block in range(3):
        column = m.loadings[:, 1 + block]
        members = m.groups == block
        assert (column[members] >= 0.5).all() and (column[members] <= 1.5).all()
        assert (column[~members] == 0.0).all()


def test_factor_and_idiosyncratic_variances_are_daily_and_cycle_group_vols() -> None:
    cfg = SimulationConfig(
        n_assets=12, n_groups=3, sample_sizes=(30,), n_reps=2, group_vols=(0.08, 0.06)
    )
    m = model(cfg)
    expected = np.array([0.15, 0.08, 0.06, 0.08]) ** 2 / 252
    np.testing.assert_allclose(m.factor_var, expected)
    assert (m.idio_var >= 0.15**2 / 252).all() and (m.idio_var <= 0.45**2 / 252).all()


def test_zero_groups_gives_a_single_factor_model() -> None:
    m = model(SimulationConfig(n_assets=6, n_groups=0, sample_sizes=(30,), n_reps=2))
    assert m.n_factors == 1 and (m.groups == -1).all()


def test_same_seed_same_model_and_returns() -> None:
    a, b = model(seed=3), model(seed=3)
    np.testing.assert_array_equal(a.loadings, b.loadings)
    x = simulate_returns(a, 50, np.random.default_rng(1))
    y = simulate_returns(b, 50, np.random.default_rng(1))
    np.testing.assert_array_equal(x, y)
    assert not np.array_equal(x, simulate_returns(a, 50, np.random.default_rng(2)))


@pytest.mark.parametrize("distribution", ["gaussian", "student_t"])
def test_sample_covariance_converges_to_the_true_covariance(distribution: str) -> None:
    m = model()
    x = simulate_returns(m, 300_000, np.random.default_rng(5), distribution, df=5.0)
    error = np.linalg.norm(np.cov(x, rowvar=False) - m.sigma) / np.linalg.norm(m.sigma)
    assert error < 0.05
    np.testing.assert_allclose(x.mean(axis=0), 0.0, atol=2e-4)


def test_student_t_has_heavy_tails_and_gaussian_does_not() -> None:
    m = model()
    gaussian = simulate_returns(m, 300_000, np.random.default_rng(5), "gaussian")
    heavy = simulate_returns(m, 300_000, np.random.default_rng(5), "student_t", df=5.0)
    assert abs(stats.kurtosis(gaussian[:, 0])) < 0.2
    assert stats.kurtosis(heavy[:, 0]) > 1.5


def test_distributions_share_the_normals_and_differ_by_a_daily_scale() -> None:
    m = model()
    gaussian = simulate_returns(m, 200, np.random.default_rng(9), "gaussian")
    heavy = simulate_returns(m, 200, np.random.default_rng(9), "student_t", df=5.0)
    ratio = heavy / gaussian
    np.testing.assert_allclose(ratio, np.repeat(ratio[:, :1], 12, axis=1))
    assert (ratio[:, 0] > 0).all()


def test_invalid_arguments_rejected() -> None:
    m = model()
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="unknown distribution"):
        simulate_returns(m, 10, rng, "laplace")
    with pytest.raises(ValueError, match="df"):
        simulate_returns(m, 10, rng, "student_t", df=2.0)
    with pytest.raises(ValueError, match="n_obs"):
        simulate_returns(m, 0, rng)
