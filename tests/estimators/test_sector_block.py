"""Tests for the PCA plus sector-block covariance estimator."""

import numpy as np
import pytest

from pfopt.estimators.pca_factor import pca_factor_cov
from pfopt.estimators.sample import sample_cov
from pfopt.estimators.sector_block import block_mask, sector_block_cov


def make_returns(
    seed: int = 0, n_obs: int = 400, n_groups: int = 4, size: int = 6, loading: float = 0.5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Market factor, one factor per group, unit noise; returns, groups and the true covariance."""
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_groups), size)
    f = rng.standard_normal(n_obs)
    g = rng.standard_normal((n_obs, n_groups))
    x = 0.5 * f[:, None] + loading * g[:, groups] + rng.standard_normal((n_obs, groups.size))
    same = groups[:, None] == groups[None, :]
    true = 0.25 * np.ones((groups.size, groups.size)) + loading**2 * same + np.eye(groups.size)
    return x, groups, true


def test_block_mask() -> None:
    mask = block_mask(np.array([0, 0, 1, -1, -1]))
    expected = np.array(
        [
            [1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ],
        dtype=bool,
    )
    np.testing.assert_array_equal(mask, expected)
    with pytest.raises(ValueError):
        block_mask(np.zeros((2, 2), dtype=int))


def test_no_known_group_gives_the_pca_covariance() -> None:
    x, groups, _ = make_returns()
    got = sector_block_cov(x, np.full(groups.size, -1), K=2, rule="fixed")
    np.testing.assert_allclose(got, pca_factor_cov(x, K=2, rule="fixed"), atol=1e-10)
    got_mp = sector_block_cov(x, np.full(groups.size, -1))
    np.testing.assert_allclose(got_mp, pca_factor_cov(x), atol=1e-10)


def test_a_single_group_gives_the_sample_covariance() -> None:
    x, groups, _ = make_returns()
    got = sector_block_cov(x, np.zeros(groups.size, dtype=int), K=2, rule="fixed")
    np.testing.assert_allclose(got, sample_cov(x), atol=1e-10)


def test_blocks_keep_the_sample_residual_and_other_entries_the_systematic_part() -> None:
    x, groups, _ = make_returns()
    sample = sample_cov(x)
    cov = sector_block_cov(x, groups, K=2, rule="fixed")
    pca_part = pca_factor_cov(x, K=2, rule="fixed")
    same = block_mask(groups)
    np.testing.assert_allclose(cov[same], sample[same], atol=1e-10)
    off = ~same
    np.testing.assert_allclose(cov[off], pca_part[off], atol=1e-10)
    np.testing.assert_allclose(np.diag(cov), np.diag(sample), atol=1e-12)


@pytest.mark.parametrize("rule", ["mp", "baing"])
def test_symmetric_psd_with_sample_variances_on_the_diagonal(rule) -> None:
    x, groups, _ = make_returns(seed=3)
    groups = groups.copy()
    groups[:3] = -1
    cov = sector_block_cov(x, groups, rule=rule)
    np.testing.assert_allclose(cov, cov.T)
    assert np.linalg.eigvalsh(cov).min() > -1e-10
    np.testing.assert_allclose(np.diag(cov), np.diag(sample_cov(x)), atol=1e-12)


def test_info_and_validation() -> None:
    x, groups, _ = make_returns()
    _, info = sector_block_cov(x, groups, return_info=True)
    assert info["K"] >= 1 and 0.0 < info["explained"] < 1.0 and 0.0 < info["block_share"] < 1.0
    _, one = sector_block_cov(x, np.zeros(groups.size, dtype=int), return_info=True)
    _, none = sector_block_cov(x, np.full(groups.size, -1), return_info=True)
    assert one["block_share"] == pytest.approx(1.0) and none["block_share"] == 0.0
    with pytest.raises(ValueError, match="length"):
        sector_block_cov(x, groups[:-1])
    with pytest.raises(ValueError):
        sector_block_cov(x, groups, K=3, rule="mp")


def test_blocks_reduce_the_error_when_the_groups_are_real() -> None:
    x, groups, true = make_returns(seed=5, n_obs=1000, n_groups=8, size=8, loading=0.5)
    sector = sector_block_cov(x, groups, K=1, rule="fixed")
    pca = pca_factor_cov(x, K=1, rule="fixed")
    assert np.linalg.norm(sector - true) < 0.8 * np.linalg.norm(pca - true)
    shuffled = np.random.default_rng(0).permutation(groups)
    wrong = sector_block_cov(x, shuffled, K=1, rule="fixed")
    assert np.linalg.norm(wrong - true) > np.linalg.norm(sector - true)
