"""Tests for the regularized weight rules."""

import numpy as np
import pytest

from pfopt.optimizers.gmv import gmv_long_only, gmv_unconstrained
from pfopt.optimizers.regularized import (
    gmv_l1,
    gmv_ridge_long_only,
    ridge_cov,
    shrink_to_equal,
)


def random_cov(rng: np.random.Generator, n: int = 10) -> np.ndarray:
    a = rng.standard_normal((n, 3 * n))
    scales = rng.uniform(0.5, 2.0, size=n)
    return (a @ a.T / (3 * n)) * np.outer(scales, scales) * 1e-4


def test_ridge_known_value_and_limits() -> None:
    cov = np.diag([1.0, 3.0])
    np.testing.assert_allclose(ridge_cov(cov, 0.5), np.diag([2.0, 4.0]))
    cov = random_cov(np.random.default_rng(0))
    np.testing.assert_allclose(gmv_ridge_long_only(cov, 0.0), gmv_long_only(cov), atol=1e-7)
    np.testing.assert_allclose(gmv_ridge_long_only(cov, 1e8), np.full(10, 0.1), atol=1e-6)
    with pytest.raises(ValueError):
        ridge_cov(cov, -1.0)


def test_shrink_to_equal_limits_and_validity() -> None:
    w = np.array([0.7, 0.3, 0.0])
    np.testing.assert_allclose(shrink_to_equal(w, 0.0), w)
    np.testing.assert_allclose(shrink_to_equal(w, 1.0), np.full(3, 1 / 3))
    out = shrink_to_equal(w, 0.4)
    assert out.sum() == pytest.approx(1.0) and out.min() > 0
    with pytest.raises(ValueError):
        shrink_to_equal(w, 1.5)


def test_l1_bound_is_active_and_interpolates_between_long_only_and_unconstrained() -> None:
    cov = random_cov(np.random.default_rng(1))
    unconstrained = gmv_unconstrained(cov)
    leverage = float(np.abs(unconstrained).sum())
    assert leverage > 1.05
    bound = 1.0 + 0.5 * (leverage - 1.0)
    w, lev = gmv_l1(cov, bound)
    assert w.sum() == pytest.approx(1.0) and lev == pytest.approx(bound, abs=1e-6)
    assert w.min() < -1e-6
    w1, lev1 = gmv_l1(cov, 1.0)
    assert lev1 == pytest.approx(1.0, abs=1e-6) and w1.min() > -1e-6
    np.testing.assert_allclose(w1, gmv_long_only(cov), atol=1e-5)
    loose, _ = gmv_l1(cov, leverage + 1.0)
    np.testing.assert_allclose(loose, unconstrained, atol=1e-5)
    variances = [float(gmv_l1(cov, c)[0] @ cov @ gmv_l1(cov, c)[0]) for c in (1.0, bound, leverage)]
    assert variances[0] >= variances[1] >= variances[2] - 1e-12


def test_l1_with_large_ridge_is_equal_weight_and_bound_below_one_is_rejected() -> None:
    cov = random_cov(np.random.default_rng(2))
    w, _ = gmv_l1(cov, 1.5, ridge=1e8)
    np.testing.assert_allclose(w, np.full(10, 0.1), atol=1e-5)
    with pytest.raises(ValueError):
        gmv_l1(cov, 0.9)
