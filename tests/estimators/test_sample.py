"""Tests for the sample and Ledoit-Wolf covariance estimators."""

import numpy as np
import pytest

from pfopt.estimators.sample import lw_cov, sample_cov
from pfopt.estimators.utils import check_psd


def draw(t: int, n: int, seed: int = 0, scale: float = 1.0) -> np.ndarray:
    return scale * np.random.default_rng(seed).normal(size=(t, n))


def test_sample_cov_matches_numpy() -> None:
    for scale in (1.0, 0.01):
        x = draw(200, 10, seed=1, scale=scale)
        np.testing.assert_allclose(sample_cov(x), np.cov(x, rowvar=False), rtol=0, atol=1e-12)


def test_sample_cov_hand_computed_example() -> None:
    x = np.array([[1.0, 2.0], [3.0, 6.0], [5.0, 4.0]])
    np.testing.assert_allclose(sample_cov(x), [[4.0, 2.0], [2.0, 4.0]], atol=1e-14)


def test_sample_cov_symmetric_psd_and_input_unchanged() -> None:
    x = draw(120, 15, seed=2)
    before = x.copy()
    cov = sample_cov(x)
    assert cov.shape == (15, 15)
    assert np.array_equal(cov, cov.T)
    check_psd(cov)
    np.testing.assert_array_equal(x, before)


def test_sample_cov_is_rank_deficient_when_t_below_n() -> None:
    t, n = 20, 50
    cov = sample_cov(draw(t, n, seed=3))
    eig = np.linalg.eigvalsh(cov)
    assert np.linalg.matrix_rank(cov) == t - 1
    assert (eig > 1e-10).sum() == t - 1
    assert abs(eig[0]) < 1e-12
    check_psd(cov)


def test_lw_cov_symmetric_psd_trace_preserved_and_shrinkage_in_unit_interval() -> None:
    x = draw(100, 30, seed=4)
    cov, delta = lw_cov(x, return_shrinkage=True)
    assert 0.0 <= delta <= 1.0
    assert np.array_equal(cov, cov.T)
    check_psd(cov)
    mle = np.cov(x, rowvar=False, ddof=0)
    assert np.trace(cov) == pytest.approx(np.trace(mle), rel=1e-10)


def test_lw_cov_equals_shrinkage_formula() -> None:
    x = draw(80, 12, seed=5)
    cov, delta = lw_cov(x, return_shrinkage=True)
    mle = np.cov(x, rowvar=False, ddof=0)
    mu = np.trace(mle) / mle.shape[0]
    expected = (1.0 - delta) * mle + delta * mu * np.eye(mle.shape[0])
    np.testing.assert_allclose(cov, expected, rtol=0, atol=1e-12)


def test_lw_cov_default_returns_matrix_only() -> None:
    x = draw(60, 8, seed=6)
    out = lw_cov(x)
    assert isinstance(out, np.ndarray)
    np.testing.assert_array_equal(out, lw_cov(x, return_shrinkage=True)[0])


def test_lw_cov_is_full_rank_when_t_below_n() -> None:
    t, n = 20, 50
    x = draw(t, n, seed=7)
    cov, delta = lw_cov(x, return_shrinkage=True)
    eig = np.linalg.eigvalsh(cov)
    assert eig[0] > 0.0
    assert delta > 0.0
    assert np.linalg.cond(cov) < 1e3


def test_lw_improves_conditioning_and_shrinks_more_with_less_data() -> None:
    n = 50
    x = draw(100, n, seed=8)
    assert np.linalg.cond(lw_cov(x)) < np.linalg.cond(sample_cov(x))
    # Heterogeneous volatilities: the truth is not a scaled identity (the shrinkage target), so the
    # optimal intensity falls as T grows. With an identity truth it would stay at 1.
    scales = np.linspace(0.5, 3.0, n)
    _, delta_small_t = lw_cov(draw(20, n, seed=9) * scales, return_shrinkage=True)
    _, delta_large_t = lw_cov(draw(2000, n, seed=9) * scales, return_shrinkage=True)
    assert delta_small_t > delta_large_t
    assert delta_large_t < 0.5


@pytest.mark.parametrize("estimator", [sample_cov, lw_cov])
def test_invalid_inputs_raise(estimator) -> None:
    with pytest.raises(ValueError, match="2-D"):
        estimator(np.arange(5.0))
    with pytest.raises(ValueError, match="2-D"):
        estimator(np.ones((1, 4)))
    bad = draw(10, 3)
    bad[2, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        estimator(bad)
