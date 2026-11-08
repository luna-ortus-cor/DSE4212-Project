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


# ---- time-varying design (step 6.3) --------------------------------------------------------
TV = SimulationConfig(
    n_assets=12, n_groups=2, sample_sizes=(30,), n_reps=2, time_varying=True, horizon_days=5
)


def test_ar1_paths_have_the_requested_variance_and_autocorrelation() -> None:
    from pfopt.simulation.dgp import ar1_paths

    paths = ar1_paths(20_000, 3, 0.9, 0.4, np.random.default_rng(1))
    assert paths.shape == (20_000, 3)
    np.testing.assert_allclose(paths.std(axis=0), 0.4, rtol=0.1)
    for j in range(3):
        assert np.corrcoef(paths[1:, j], paths[:-1, j])[0, 1] == pytest.approx(0.9, abs=0.03)


def test_loading_drift_keeps_zeros_and_has_the_requested_spread() -> None:
    from pfopt.simulation.dgp import loading_paths

    m = model()
    paths = loading_paths(m, 5_000, 0.95, 0.2, np.random.default_rng(2))
    assert paths.shape == (5_000, 12, 3)
    zero = m.loadings == 0.0
    assert (paths[:, zero] == 0.0).all()
    deviation = (paths - m.loadings)[:, ~zero]
    assert deviation.std() == pytest.approx(0.2, rel=0.15)


def test_forward_covariance_equals_the_static_truth_without_time_variation() -> None:
    from pfopt.simulation.dgp import simulate_time_varying

    flat = SimulationConfig(
        n_assets=12, n_groups=2, sample_sizes=(30,), n_reps=2, time_varying=True,
        vol_log_sd=0.0, loading_drift_sd=0.0,
    )  # fmt: skip
    m = model()
    x, forward = simulate_time_varying(m, 40, 7, np.random.default_rng(3), flat)
    assert x.shape == (40, 12)
    np.testing.assert_allclose(forward, m.sigma, atol=1e-15)


def test_forward_covariance_is_valid_and_averages_to_the_unconditional_one() -> None:
    from pfopt.simulation.dgp import simulate_time_varying

    m = model()
    draws = [simulate_time_varying(m, 10, 5, np.random.default_rng(i), TV)[1] for i in range(1500)]
    for forward in draws[:5]:
        np.testing.assert_allclose(forward, forward.T)
        assert np.linalg.eigvalsh(forward).min() > 0
    assert not np.allclose(draws[0], m.sigma)
    mean = np.mean(draws, axis=0)
    # the loading drift adds tau^2 * factor variance to the diagonal, so compare off the diagonal
    off = ~np.eye(12, dtype=bool)
    assert np.linalg.norm((mean - m.sigma)[off]) / np.linalg.norm(m.sigma[off]) < 0.1


def test_time_varying_returns_cluster_in_volatility_and_static_ones_do_not() -> None:
    from pfopt.simulation.dgp import simulate_time_varying

    cfg = SimulationConfig(
        n_assets=12, n_groups=2, sample_sizes=(30,), n_reps=2, time_varying=True, horizon_days=1
    )
    m = model()
    tv, _ = simulate_time_varying(m, 200_000, 1, np.random.default_rng(4), cfg)
    static = simulate_returns(m, 200_000, np.random.default_rng(4))

    def acf1(x: np.ndarray) -> float:
        market = (x.mean(axis=1)) ** 2
        return float(np.corrcoef(market[1:], market[:-1])[0, 1])

    assert acf1(tv) > 0.05 and abs(acf1(static)) < 0.03


def test_time_varying_is_reproducible_and_distributions_share_the_normals() -> None:
    from pfopt.simulation.dgp import simulate_time_varying

    m = model()
    a = simulate_time_varying(m, 30, 5, np.random.default_rng(5), TV)
    b = simulate_time_varying(m, 30, 5, np.random.default_rng(5), TV)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    heavy = simulate_time_varying(m, 30, 5, np.random.default_rng(5), TV, "student_t")
    np.testing.assert_array_equal(a[1], heavy[1])  # the target ignores the day scale
    ratio = heavy[0] / a[0]
    np.testing.assert_allclose(ratio, np.repeat(ratio[:, :1], 12, axis=1))
    with pytest.raises(ValueError, match="unknown distribution"):
        simulate_time_varying(m, 30, 5, np.random.default_rng(5), TV, "laplace")
    with pytest.raises(ValueError, match="horizon"):
        simulate_time_varying(m, 30, 0, np.random.default_rng(5), TV)
