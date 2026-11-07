"""Tests for the PCA-factor covariance estimator."""

import numpy as np
import pytest

from pfopt.estimators.pca_factor import n_factors, pca_factor_cov
from pfopt.estimators.sample import sample_cov
from pfopt.estimators.utils import check_psd


def noise(t: int, n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(t, n))


def factor_data(t: int, n: int, k: int, loading: float, seed: int) -> np.ndarray:
    """Unit idiosyncratic noise plus ``k`` independent factors with loadings ``+-loading``.

    Random signs make every asset's total loading identical, so all assets have the same
    idiosyncratic variance share. (With Gaussian loadings the shares differ across assets and
    Bai-Ng's criterion on standardized returns can add spurious factors.)
    """
    rng = np.random.default_rng(seed)
    loadings = loading * rng.choice([-1.0, 1.0], size=(n, k))
    return rng.normal(size=(t, n)) + rng.normal(size=(t, k)) @ loadings.T


def uncorrelated_data(t: int, n: int, seed: int) -> np.ndarray:
    """Zero-mean data whose sample correlation matrix is exactly the identity."""
    raw = noise(t, n, seed)
    raw -= raw.mean(axis=0)
    return np.linalg.qr(raw)[0]


def one_factor_equal_noise_data() -> tuple[np.ndarray, float, float, np.ndarray]:
    """X with S = c * b b' + s2 * I exactly (T = 30, N = 4); returns X, c, s2, b."""
    t, a, sigma = 30, 3.0, 1.0
    b = np.array([1.0, 2.0, 3.0, 4.0])
    raw = noise(t, 1 + b.size, 0)
    raw -= raw.mean(axis=0)
    q = np.linalg.qr(raw)[0]  # orthonormal, zero-mean columns
    x = a * np.outer(q[:, 0], b) + sigma * q[:, 1:]
    return x, a**2 / (t - 1), sigma**2 / (t - 1), b


def test_known_answer_one_factor_with_equal_noise() -> None:
    x, c, s2, b = one_factor_equal_noise_data()
    u = b / np.linalg.norm(b)
    expected = c * np.outer(b, b) + s2 * (np.outer(u, u) + np.diag(1.0 - u**2))
    np.testing.assert_allclose(pca_factor_cov(x, K=1, rule="fixed"), expected, atol=1e-12)
    assert n_factors(x, rule="mp") == 1


@pytest.mark.parametrize(("rule", "k"), [("mp", None), ("fixed", 3), ("baing", None)])
def test_psd_symmetric_and_diagonal_equals_sample_variances(rule: str, k: int | None) -> None:
    x = factor_data(504, 100, 3, 1.0, seed=0)
    cov = pca_factor_cov(x, K=k, rule=rule)
    assert cov.shape == (100, 100)
    assert np.array_equal(cov, cov.T)
    check_psd(cov)
    np.testing.assert_allclose(np.diag(cov), np.diag(sample_cov(x)), rtol=1e-10)


def test_all_factors_reproduce_the_sample_covariance() -> None:
    x = noise(100, 8, 1)
    np.testing.assert_allclose(
        pca_factor_cov(x, K=8, rule="fixed"), sample_cov(x), rtol=0, atol=1e-8
    )


def test_zero_factors_give_a_diagonal_matrix() -> None:
    x = noise(100, 8, 2)
    cov = pca_factor_cov(x, K=0, rule="fixed")
    assert np.array_equal(cov, np.diag(np.diag(sample_cov(x))))


def test_residual_floor_binds_only_when_the_fit_is_exact() -> None:
    x = noise(100, 6, 3)
    variances = np.diag(sample_cov(x))
    cov = pca_factor_cov(x, K=6, rule="fixed", min_resid_frac=1e-3)
    np.testing.assert_allclose(np.diag(cov), variances * (1.0 + 1e-3), rtol=1e-6)


def test_mp_rule_counts_planted_factors() -> None:
    x = factor_data(1000, 60, 3, 1.0, seed=4)
    assert n_factors(x, rule="mp") == 3


def test_mp_rule_is_at_least_one_and_baing_finds_one_on_uncorrelated_data() -> None:
    x = uncorrelated_data(300, 20, 5)
    assert n_factors(x, rule="mp") == 1
    assert n_factors(x, rule="baing") == 1


def test_baing_finds_planted_factors_within_one() -> None:
    assert abs(n_factors(factor_data(1000, 60, 3, 1.0, seed=6), rule="baing") - 3) <= 1
    strong = factor_data(1000, 60, 6, 2.0, seed=7)
    assert abs(n_factors(strong, rule="baing") - 6) <= 1


def test_baing_respects_kmax() -> None:
    strong = factor_data(1000, 60, 6, 2.0, seed=7)
    assert n_factors(strong, rule="baing", kmax=3) == 3


def test_invalid_rule_and_k_combinations_raise() -> None:
    x = noise(50, 5, 8)
    with pytest.raises(ValueError, match="rule"):
        n_factors(x, rule="magic")
    with pytest.raises(ValueError, match="fixed"):
        n_factors(x, rule="fixed")
    with pytest.raises(ValueError, match="fixed"):
        n_factors(x, rule="fixed", K=-1)
    with pytest.raises(ValueError, match="fixed"):
        n_factors(x, rule="fixed", K=6)
    with pytest.raises(ValueError, match="fixed"):
        n_factors(x, rule="fixed", K=2.5)
    with pytest.raises(ValueError, match="None"):
        n_factors(x, rule="mp", K=2)
    with pytest.raises(ValueError, match="None"):
        n_factors(x, rule="baing", K=2)
    with pytest.raises(ValueError, match="kmax"):
        n_factors(x, rule="baing", kmax=0)


def test_info_reports_k_rule_and_explained_variance() -> None:
    x = factor_data(300, 20, 2, 1.0, seed=9)
    _, info = pca_factor_cov(x, K=3, rule="fixed", return_info=True)
    assert info["K"] == 3 and info["rule"] == "fixed"
    assert 0.0 < info["explained"] < 1.0
    assert pca_factor_cov(x, K=0, rule="fixed", return_info=True)[1]["explained"] == 0.0
    assert pca_factor_cov(x, K=20, rule="fixed", return_info=True)[1]["explained"] == pytest.approx(
        1.0
    )


def test_more_assets_than_observations_still_gives_a_positive_definite_matrix() -> None:
    x = noise(20, 50, 10)
    cov = pca_factor_cov(x)
    check_psd(cov)
    assert np.linalg.eigvalsh(cov)[0] > 0.0
    np.testing.assert_allclose(np.diag(cov), x.var(axis=0, ddof=1), rtol=1e-9)


def test_conditioning_is_better_than_the_sample_covariance() -> None:
    x = noise(504, 100, 11)
    assert np.linalg.cond(pca_factor_cov(x)) < np.linalg.cond(sample_cov(x))


def test_permutation_equivariance_and_input_unchanged() -> None:
    x = factor_data(300, 12, 2, 1.0, seed=12)
    before = x.copy()
    perm = np.random.default_rng(0).permutation(12)
    base = pca_factor_cov(x, K=2, rule="fixed")
    np.testing.assert_allclose(
        pca_factor_cov(x[:, perm], K=2, rule="fixed"), base[np.ix_(perm, perm)], atol=1e-10
    )
    np.testing.assert_array_equal(x, before)


def test_invalid_returns_raise() -> None:
    with pytest.raises(ValueError, match="2-D"):
        pca_factor_cov(np.arange(5.0))
    bad = noise(30, 4, 0)
    bad[3, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        pca_factor_cov(bad)
