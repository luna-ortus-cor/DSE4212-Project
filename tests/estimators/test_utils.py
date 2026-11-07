"""Tests for correlation and Marchenko-Pastur utilities."""

import numpy as np
import pytest
from scipy.integrate import trapezoid

from pfopt.estimators.utils import (
    check_psd,
    condition_number,
    corr_eigenvalues,
    corr_matrix,
    cov_to_corr,
    mp_density,
    mp_edges,
    n_above_mp_edge,
    winsorize_cols,
)


def test_corr_matrix_matches_numpy_and_is_symmetric() -> None:
    x = np.random.default_rng(0).normal(size=(60, 7))
    c = corr_matrix(x)
    np.testing.assert_allclose(c, np.corrcoef(x, rowvar=False), atol=1e-12)
    assert np.array_equal(c, c.T)
    np.testing.assert_array_equal(np.diag(c), np.ones(7))


def test_corr_matrix_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="zero variance"):
        corr_matrix(np.column_stack([np.ones(10), np.arange(10.0)]))
    with pytest.raises(ValueError, match="2-D"):
        corr_matrix(np.arange(5.0))
    with pytest.raises(ValueError, match="2-D"):
        corr_matrix(np.ones((1, 3)))


def test_corr_eigenvalues_descending_and_trace() -> None:
    x = np.random.default_rng(1).normal(size=(80, 6))
    eig = corr_eigenvalues(x)
    assert np.all(np.diff(eig) <= 0)
    assert eig.sum() == pytest.approx(6.0, abs=1e-10)


def test_mp_edges_known_values() -> None:
    lo, hi = mp_edges(0.25)
    assert lo == pytest.approx(0.25)
    assert hi == pytest.approx(2.25)
    lo2, hi2 = mp_edges(0.25, sigma2=2.0)
    assert (lo2, hi2) == (pytest.approx(0.5), pytest.approx(4.5))
    assert mp_edges(100 / 504)[1] == pytest.approx(2.0893, abs=1e-3)
    with pytest.raises(ValueError):
        mp_edges(0.0)


@pytest.mark.parametrize("q", [0.05, 0.2, 0.5, 0.9])
def test_mp_density_integrates_to_one(q: float) -> None:
    lo, hi = mp_edges(q)
    grid = np.linspace(lo, hi, 400_001)
    assert trapezoid(mp_density(grid, q), grid) == pytest.approx(1.0, abs=2e-3)


def test_mp_density_zero_outside_support_and_rejects_q_above_one() -> None:
    lo, hi = mp_edges(0.2)
    dens = mp_density(np.array([lo - 0.1, lo, hi, hi + 0.1, 10.0]), 0.2)
    np.testing.assert_array_equal(dens, np.zeros(5))
    assert mp_density(np.array([1.0]), 0.2)[0] > 0
    with pytest.raises(ValueError, match="q <= 1"):
        mp_density(np.array([1.0]), 1.5)


def test_pure_noise_spectrum_sits_inside_mp_support() -> None:
    n, t = 100, 504
    eig = corr_eigenvalues(np.random.default_rng(0).normal(size=(t, n)))
    lo, hi = mp_edges(n / t)
    assert eig.mean() == pytest.approx(1.0, abs=1e-10)
    assert eig.max() < 1.15 * hi
    assert eig.min() > 0.7 * lo
    assert n_above_mp_edge(eig, n / t) <= 2


def test_planted_factor_creates_signal_eigenvalue() -> None:
    rng = np.random.default_rng(0)
    n, t = 100, 504
    factor = rng.normal(size=(t, 1))
    x = rng.normal(size=(t, n)) + 1.0 * factor
    eig = corr_eigenvalues(x)
    assert n_above_mp_edge(eig, n / t) >= 1
    assert eig[0] > 10 * mp_edges(n / t)[1]


def test_cov_to_corr_matches_numpy_and_validates() -> None:
    x = np.random.default_rng(3).normal(size=(80, 6)) * np.arange(1, 7)
    cov = np.cov(x, rowvar=False)
    corr = cov_to_corr(cov)
    np.testing.assert_allclose(corr, np.corrcoef(x, rowvar=False), atol=1e-12)
    np.testing.assert_array_equal(np.diag(corr), np.ones(6))
    assert np.array_equal(corr, corr.T)
    with pytest.raises(ValueError, match="square"):
        cov_to_corr(np.ones((2, 3)))
    with pytest.raises(ValueError, match="positive diagonal"):
        cov_to_corr(np.diag([1.0, 0.0]))


def test_winsorize_cols_clips_each_column_at_its_own_quantiles() -> None:
    x = np.column_stack([np.arange(101.0), 10.0 * np.arange(101.0)])
    before = x.copy()
    out = winsorize_cols(x, 0.01, 0.99)
    assert out[:, 0].min() == pytest.approx(1.0) and out[:, 0].max() == pytest.approx(99.0)
    assert out[:, 1].min() == pytest.approx(10.0) and out[:, 1].max() == pytest.approx(990.0)
    np.testing.assert_array_equal(out[2:99], x[2:99])
    np.testing.assert_array_equal(x, before)
    assert out.shape == x.shape


def test_winsorize_cols_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError, match="lower < upper"):
        winsorize_cols(np.ones((5, 2)), 0.9, 0.1)
    with pytest.raises(ValueError, match="2-D"):
        winsorize_cols(np.ones(5), 0.1, 0.9)


def test_condition_number_known_values() -> None:
    assert condition_number(np.diag([1.0, 4.0])) == pytest.approx(4.0)
    assert condition_number(np.eye(3)) == pytest.approx(1.0)
    assert condition_number(np.zeros((2, 2))) == float("inf")
    assert condition_number(np.array([[1.0, 2.0], [2.0, 1.0]])) == float("inf")


def test_check_psd_accepts_valid_and_rank_deficient_matrices() -> None:
    x = np.random.default_rng(0).normal(size=(5, 20))  # rank-deficient 20 x 20 Gram matrix
    check_psd(x.T @ x)
    check_psd(np.eye(3))
    check_psd(np.zeros((2, 2)))


def test_check_psd_rejects_bad_matrices() -> None:
    with pytest.raises(ValueError, match="symmetric"):
        check_psd(np.array([[1.0, 0.5], [0.0, 1.0]]))
    with pytest.raises(ValueError, match="not PSD"):
        check_psd(np.array([[1.0, 2.0], [2.0, 1.0]]))
    with pytest.raises(ValueError, match="square"):
        check_psd(np.ones((2, 3)))
