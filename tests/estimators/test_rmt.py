"""Tests for RMT eigenvalue clipping."""

import numpy as np
import pytest

from pfopt.estimators.rmt import rmt_corr, rmt_cov
from pfopt.estimators.sample import sample_cov
from pfopt.estimators.utils import check_psd, mp_edges


def noise(t: int, n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(t, n))


def factor_data(t: int, n: int, seed: int, loading: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(t, n)) + loading * rng.normal(size=(t, 1))


def avg_offdiag(c: np.ndarray) -> float:
    n = c.shape[0]
    return float((c.sum() - n) / (n * (n - 1)))


def test_cleaned_correlation_has_unit_diagonal_and_is_symmetric_psd() -> None:
    clean = rmt_corr(factor_data(504, 100, 0, 1.0))
    np.testing.assert_array_equal(np.diag(clean), np.ones(100))
    assert np.array_equal(clean, clean.T)
    check_psd(clean)


def test_eigenvalue_trace_is_preserved_by_clipping() -> None:
    x = factor_data(504, 100, 1, 1.0)
    _, info = rmt_corr(x, return_info=True)
    eig, clipped = info["eigenvalues"], info["eigenvalues_clipped"]
    assert eig.sum() == pytest.approx(100.0, abs=1e-8)
    assert clipped.sum() == pytest.approx(100.0, abs=1e-8)
    k = info["n_signal"]
    np.testing.assert_allclose(clipped[:k], eig[:k])
    np.testing.assert_allclose(clipped[k:], info["noise_mean"])
    assert np.all(eig[:k] > info["lam_plus"]) and np.all(eig[k:] <= info["lam_plus"])


def test_covariance_diagonal_equals_sample_variances_and_is_psd() -> None:
    x = 0.01 * factor_data(504, 100, 2, 1.0)
    cov = rmt_cov(x)
    np.testing.assert_allclose(np.diag(cov), x.var(axis=0, ddof=1), rtol=1e-12)
    assert np.array_equal(cov, cov.T)
    check_psd(cov)


def test_no_signal_gives_identity_correlation() -> None:
    raw = noise(300, 20, 0)
    raw -= raw.mean(axis=0)
    orthonormal, _ = np.linalg.qr(raw)  # zero-mean, exactly uncorrelated columns
    x = 0.1 * orthonormal
    clean, info = rmt_corr(x, return_info=True)
    assert info["n_signal"] == 0
    np.testing.assert_allclose(clean, np.eye(20), atol=1e-10)
    np.testing.assert_allclose(rmt_cov(x), np.diag(x.var(axis=0, ddof=1)), atol=1e-12)


def test_pure_noise_has_at_most_a_few_signal_eigenvalues() -> None:
    _, info = rmt_cov(noise(504, 100, 1), return_info=True)
    assert info["n_signal"] <= 2
    assert info["lam_plus"] == pytest.approx(mp_edges(100 / 504)[1])


def test_planted_factor_is_detected_and_kept() -> None:
    x = factor_data(2000, 50, 3, 1.0)
    clean, info = rmt_corr(x, return_info=True)
    assert info["n_signal"] == 1
    assert info["eigenvalues"][0] > 20.0
    assert avg_offdiag(clean) == pytest.approx(avg_offdiag(np.corrcoef(x, rowvar=False)), abs=0.02)


def test_small_q_output_approaches_sample_correlation() -> None:
    n, t = 5, 200_000
    true_corr = np.full((n, n), 0.5)
    np.fill_diagonal(true_corr, 1.0)
    x = noise(t, n, 4) @ np.linalg.cholesky(true_corr).T
    clean, info = rmt_corr(x, return_info=True)
    assert info["n_signal"] == 1
    np.testing.assert_allclose(clean, np.corrcoef(x, rowvar=False), atol=0.01)


def test_clipping_nothing_returns_the_sample_covariance() -> None:
    x = noise(100, 5, 5)
    cov, info = rmt_cov(x, sigma2=1e-6, return_info=True)
    assert info["n_signal"] == 5
    assert np.isnan(info["noise_mean"])
    np.testing.assert_allclose(cov, sample_cov(x), rtol=1e-9, atol=1e-12)


def test_q_and_sigma2_set_the_edge() -> None:
    x = noise(200, 20, 6)
    _, default = rmt_cov(x, return_info=True)
    _, custom_q = rmt_cov(x, q=0.5, return_info=True)
    _, doubled = rmt_cov(x, sigma2=2.0, return_info=True)
    assert default["q"] == pytest.approx(20 / 200)
    assert default["lam_plus"] == pytest.approx(mp_edges(0.1)[1])
    assert custom_q["lam_plus"] == pytest.approx(mp_edges(0.5)[1])
    assert doubled["lam_plus"] == pytest.approx(2.0 * default["lam_plus"])


def test_more_observations_than_assets_not_required() -> None:
    x = noise(20, 50, 7)  # T < N: the sample covariance is singular
    cov = rmt_cov(x)
    assert np.isfinite(cov).all()
    check_psd(cov)
    assert np.linalg.eigvalsh(cov)[0] > 0.0
    np.testing.assert_allclose(np.diag(cov), x.var(axis=0, ddof=1), rtol=1e-12)


def test_denoising_improves_conditioning() -> None:
    x = noise(504, 100, 8)
    assert np.linalg.cond(rmt_cov(x)) < np.linalg.cond(sample_cov(x))


def test_permutation_equivariance_and_input_unchanged() -> None:
    x = factor_data(300, 12, 9, 1.0)
    before = x.copy()
    perm = np.random.default_rng(0).permutation(12)
    base = rmt_cov(x)
    np.testing.assert_allclose(rmt_cov(x[:, perm]), base[np.ix_(perm, perm)], atol=1e-10)
    np.testing.assert_array_equal(x, before)


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError, match="2-D"):
        rmt_cov(np.arange(5.0))
    bad = noise(30, 4, 0)
    bad[3, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        rmt_cov(bad)
    constant = noise(30, 4, 0)
    constant[:, 2] = 1.0
    with pytest.raises(ValueError, match="zero variance"):
        rmt_cov(constant)
