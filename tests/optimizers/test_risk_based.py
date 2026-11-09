"""Tests for inverse-volatility and equal-risk-contribution weights (step 8.7)."""

import numpy as np
import pytest

from pfopt.optimizers.risk_based import erc_weights, inverse_vol_weights, risk_contributions


def random_cov(rng: np.random.Generator, n: int = 12) -> np.ndarray:
    a = rng.standard_normal((n, 3 * n))
    return a @ a.T / (3 * n) * 1e-4


def test_inverse_vol_known_values() -> None:
    cov = np.diag([1.0, 4.0, 16.0])
    np.testing.assert_allclose(inverse_vol_weights(cov), np.array([4.0, 2.0, 1.0]) / 7.0)


def test_erc_risk_contributions_are_equal() -> None:
    cov = random_cov(np.random.default_rng(0))
    w, iterations = erc_weights(cov)
    np.testing.assert_allclose(risk_contributions(w, cov), 1.0 / 12, atol=1e-8)
    assert (w > 0).all() and w.sum() == pytest.approx(1.0) and iterations < 50


def test_erc_equals_inverse_vol_for_two_assets_and_for_uncorrelated_assets() -> None:
    cov2 = np.array([[0.04, 0.01], [0.01, 0.09]])
    np.testing.assert_allclose(erc_weights(cov2)[0], inverse_vol_weights(cov2), atol=1e-9)
    diag = np.diag([0.01, 0.02, 0.05, 0.08])
    np.testing.assert_allclose(erc_weights(diag)[0], inverse_vol_weights(diag), atol=1e-9)


def test_erc_is_scale_and_permutation_invariant() -> None:
    cov = random_cov(np.random.default_rng(1))
    w = erc_weights(cov)[0]
    np.testing.assert_allclose(erc_weights(1e3 * cov)[0], w, atol=1e-9)
    order = np.random.default_rng(2).permutation(12)
    np.testing.assert_allclose(erc_weights(cov[np.ix_(order, order)])[0], w[order], atol=1e-9)


def test_erc_with_equal_vols_and_correlations_is_equal_weight() -> None:
    cov = 0.04 * (0.3 * np.ones((5, 5)) + 0.7 * np.eye(5))
    np.testing.assert_allclose(erc_weights(cov)[0], 0.2, atol=1e-9)


def test_invalid_covariances_are_rejected() -> None:
    with pytest.raises(ValueError):
        erc_weights(np.ones((2, 3)))
    with pytest.raises(ValueError, match="positive diagonal"):
        inverse_vol_weights(np.diag([1.0, 0.0]))
    with pytest.raises(ValueError, match="finite"):
        erc_weights(np.array([[1.0, np.nan], [np.nan, 1.0]]))
