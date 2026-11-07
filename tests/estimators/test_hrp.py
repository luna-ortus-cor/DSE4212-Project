"""Tests for Hierarchical Risk Parity."""

import numpy as np
import pytest
from scipy.linalg import block_diag

from pfopt.estimators.hrp import LINKAGE_METHODS, hrp_weights
from pfopt.estimators.rmt import rmt_corr, rmt_cov
from pfopt.estimators.sample import sample_cov
from pfopt.estimators.utils import cov_to_corr
from pfopt.weights import check_weights

BLOCK_A = np.array([[1.0, 0.5], [0.5, 1.0]])
BLOCK_B = np.array([[4.0, 2.0], [2.0, 4.0]])
BLOCK_COV = block_diag(BLOCK_A, BLOCK_B)  # two clear clusters; weights worked out by hand
BLOCK_WEIGHTS = np.array([0.4, 0.4, 0.1, 0.1])


def random_cov(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    loadings = rng.normal(size=(n, 3))
    scales = rng.uniform(0.5, 2.0, size=n)
    returns = rng.normal(size=(400, n)) * scales + rng.normal(size=(400, 3)) @ loadings.T
    return sample_cov(returns)


def test_diagonal_covariance_gives_inverse_variance_weights() -> None:
    variances = np.array([1.0, 2.0, 4.0, 8.0, 3.0])
    expected = (1.0 / variances) / (1.0 / variances).sum()
    for method in ("single", "average", "complete", "ward"):
        w = hrp_weights(np.diag(variances), linkage_method=method)
        np.testing.assert_allclose(w, expected, atol=1e-12)


def test_two_assets_are_weighted_by_inverse_variance() -> None:
    w = hrp_weights(np.array([[1.0, 0.3], [0.3, 4.0]]))
    np.testing.assert_allclose(w, [0.8, 0.2], atol=1e-12)


def test_block_covariance_known_answer_for_all_linkages() -> None:
    for method in ("single", "average", "complete"):
        np.testing.assert_allclose(
            hrp_weights(BLOCK_COV, linkage_method=method), BLOCK_WEIGHTS, atol=1e-12
        )
    perm = np.array([2, 0, 3, 1])
    w = hrp_weights(BLOCK_COV[np.ix_(perm, perm)])
    np.testing.assert_allclose(w, BLOCK_WEIGHTS[perm], atol=1e-12)


def test_original_scipy_order_gives_same_answer_on_the_block_example() -> None:
    w = hrp_weights(BLOCK_COV, canonical_order=False)
    np.testing.assert_allclose(w, BLOCK_WEIGHTS, atol=1e-12)


@pytest.mark.parametrize("method", LINKAGE_METHODS)
def test_weights_are_valid_for_every_linkage(method: str) -> None:
    w = hrp_weights(random_cov(25, 0), linkage_method=method)
    assert w.shape == (25,)
    check_weights(w)
    assert (w > 0).all()


@pytest.mark.parametrize("method", LINKAGE_METHODS)
def test_permutation_invariance(method: str) -> None:
    cov = random_cov(30, 1)
    perm = np.random.default_rng(2).permutation(30)
    w = hrp_weights(cov, linkage_method=method)
    w_perm = hrp_weights(cov[np.ix_(perm, perm)], linkage_method=method)
    np.testing.assert_allclose(w_perm, w[perm], atol=1e-12)


def test_works_for_a_singular_covariance_without_inversion() -> None:
    x = np.random.default_rng(3).normal(size=(20, 50))
    cov = sample_cov(x)  # rank 19 < 50
    assert np.linalg.matrix_rank(cov) < 50
    w = hrp_weights(cov)
    check_weights(w)
    assert np.isfinite(w).all()


def test_single_asset() -> None:
    np.testing.assert_array_equal(hrp_weights(np.array([[2.0]])), [1.0])


def test_return_info_gives_leaf_order_and_linkage() -> None:
    w, info = hrp_weights(BLOCK_COV, return_info=True)
    assert sorted(info["order"]) == [0, 1, 2, 3]
    assert info["linkage"].shape == (3, 4)
    np.testing.assert_allclose(w, BLOCK_WEIGHTS, atol=1e-12)


def test_explicit_correlation_is_used_for_clustering() -> None:
    cov = random_cov(20, 4)
    np.testing.assert_allclose(hrp_weights(cov, corr=cov_to_corr(cov)), hrp_weights(cov))
    other = cov_to_corr(random_cov(20, 5))
    assert not np.allclose(hrp_weights(cov, corr=other), hrp_weights(cov))


def test_hrp_rmt_combination_runs_and_is_valid() -> None:
    returns = np.random.default_rng(6).normal(size=(504, 40)) + np.random.default_rng(7).normal(
        size=(504, 1)
    )
    w = hrp_weights(rmt_cov(returns), corr=rmt_corr(returns))
    check_weights(w)
    assert w.shape == (40,)


def test_input_not_mutated() -> None:
    cov = random_cov(10, 8)
    before = cov.copy()
    hrp_weights(cov)
    np.testing.assert_array_equal(cov, before)


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError, match="square"):
        hrp_weights(np.ones((2, 3)))
    with pytest.raises(ValueError, match="symmetric"):
        hrp_weights(np.array([[1.0, 0.5], [0.0, 1.0]]))
    with pytest.raises(ValueError, match="positive variances"):
        hrp_weights(np.diag([1.0, 0.0]))
    with pytest.raises(ValueError, match="finite"):
        hrp_weights(np.array([[1.0, np.nan], [np.nan, 1.0]]))
    with pytest.raises(ValueError, match="linkage_method"):
        hrp_weights(BLOCK_COV, linkage_method="nope")
    with pytest.raises(ValueError, match="same shape"):
        hrp_weights(BLOCK_COV, corr=np.eye(3))
