"""Tests for the quadratic-inverse-shrinkage covariance (step 8.2)."""

import numpy as np
import pytest

from pfopt.estimators.nls import nls_cov
from pfopt.estimators.sample import sample_cov


def spiked_truth(n: int = 50) -> np.ndarray:
    values = np.concatenate([[8.0, 4.0, 2.0], np.ones(n - 3)])
    q, _ = np.linalg.qr(np.random.default_rng(99).standard_normal((n, n)))
    return (q * values) @ q.T


def draw(truth: np.ndarray, t: int, seed: int) -> np.ndarray:
    chol = np.linalg.cholesky(truth)
    return np.random.default_rng(seed).standard_normal((t, truth.shape[0])) @ chol.T


def test_output_is_symmetric_positive_definite_with_the_sample_trace() -> None:
    x = draw(spiked_truth(), 200, 0)
    cov = nls_cov(x)
    np.testing.assert_allclose(cov, cov.T)
    assert np.linalg.eigvalsh(cov).min() > 0
    assert np.trace(cov) == pytest.approx(np.trace(sample_cov(x)), rel=1e-10)


def test_spectrum_is_compressed_signal_stays_on_top_and_eigenvectors_are_the_sample_ones() -> None:
    # QIS is a smooth kernel-based approximation: it does not give a monotone map in the noise
    # bulk (the kernel bandwidth is narrow for N = 50), so only the guaranteed properties are tested
    x = draw(spiked_truth(), 200, 1)
    cov, info = nls_cov(x, return_info=True)
    shrunk, sample = info["eigenvalues"], info["sample_eigenvalues"]
    assert np.all(np.diff(shrunk[-3:]) > 0) and shrunk.argmax() == shrunk.size - 1
    assert shrunk.max() < sample.max() and shrunk.min() > sample.min()
    assert shrunk.std() < sample.std()
    values, vectors = np.linalg.eigh(sample_cov(x))
    rebuilt = (vectors * info["eigenvalues"]) @ vectors.T
    np.testing.assert_allclose(cov, (rebuilt + rebuilt.T) / 2.0, atol=1e-12)
    assert info["sample_eigenvalues"][-1] == pytest.approx(values[-1])


def test_identity_truth_is_recovered_much_better_than_by_the_sample_covariance() -> None:
    x = np.random.default_rng(2).standard_normal((1000, 50))
    err_nls = np.linalg.norm(nls_cov(x) - np.eye(50))
    err_sample = np.linalg.norm(sample_cov(x) - np.eye(50))
    assert err_nls < 0.5 * err_sample


def test_beats_the_sample_covariance_in_frobenius_error_for_large_concentration() -> None:
    truth = spiked_truth()
    nls, sample = [], []
    for seed in range(6):
        x = draw(truth, 150, seed)
        nls.append(np.linalg.norm(nls_cov(x) - truth))
        sample.append(np.linalg.norm(sample_cov(x) - truth))
    assert np.mean(nls) < np.mean(sample)


def test_scale_equivariance() -> None:
    x = draw(spiked_truth(30), 120, 3)
    np.testing.assert_allclose(nls_cov(3.0 * x), 9.0 * nls_cov(x), rtol=1e-8)


def test_more_variables_than_observations_gives_a_valid_estimate() -> None:
    x = np.random.default_rng(4).standard_normal((40, 60))
    cov = nls_cov(x)
    assert cov.shape == (60, 60) and np.linalg.eigvalsh(cov).min() > 0
    assert np.trace(cov) == pytest.approx(np.trace(sample_cov(x)), rel=1e-10)


def test_invalid_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        nls_cov(np.ones((2, 5)))
    with pytest.raises(ValueError):
        nls_cov(np.array([[1.0, np.nan], [2.0, 3.0], [4.0, 5.0]]))
