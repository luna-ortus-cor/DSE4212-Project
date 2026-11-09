"""Tests for the residual-structure diagnostic (sector blocks left by the PCA factors)."""

import numpy as np
import pandas as pd
import pytest

from pfopt.backtest.residual import (
    block_matrix,
    pca_residual_corr,
    permutation_gap_test,
    residual_diagnostics,
    sector_codes,
    sector_gap,
    sector_share,
    window_residual_diagnostics,
)
from pfopt.estimators.utils import corr_matrix


def make_returns(
    rng: np.random.Generator,
    n_obs: int = 300,
    n_sectors: int = 8,
    size: int = 8,
    market: float = 0.5,
    sector: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """Market factor plus one factor per sector plus noise; returns and the sector codes."""
    codes = np.repeat(np.arange(n_sectors), size)
    f = rng.standard_normal(n_obs)
    g = rng.standard_normal((n_obs, n_sectors))
    x = market * f[:, None] + sector * g[:, codes] + rng.standard_normal((n_obs, codes.size))
    return x, codes


BLOCK = np.array(
    [
        [1.0, 0.5, 0.1, 0.1, 0.9],
        [0.5, 1.0, 0.1, 0.1, 0.9],
        [0.1, 0.1, 1.0, 0.5, 0.9],
        [0.1, 0.1, 0.5, 1.0, 0.9],
        [0.9, 0.9, 0.9, 0.9, 1.0],
    ]
)


def test_sector_gap_and_block_matrix_known_values() -> None:
    codes = np.array([0, 0, 1, 1, -1])
    within, between = sector_gap(BLOCK, codes, 2)
    assert within == pytest.approx(0.5) and between == pytest.approx(0.1)
    np.testing.assert_allclose(block_matrix(BLOCK, codes, 2), [[0.5, 0.1], [0.1, 0.5]])
    with pytest.raises(ValueError, match="same-sector"):
        sector_gap(BLOCK, np.array([0, 1, 2, 3, -1]), 4)


def test_block_matrix_marks_empty_cells_as_nan() -> None:
    out = block_matrix(BLOCK, np.array([0, 0, 1, 1, -1]), 3)
    assert out.shape == (3, 3) and np.isnan(out[2]).all() and np.isnan(out[:, 2]).all()


def test_sector_codes_handle_unknown_and_missing_symbols() -> None:
    sectors = pd.Series({"A": "Tech", "B": "Energy", "C": np.nan})
    codes = sector_codes(["A", "B", "C", "D"], sectors, ["Energy", "Tech"])
    assert codes.tolist() == [1, 0, -1, -1]


def test_pca_residual_has_unit_diagonal_and_is_orthogonal_to_the_factors() -> None:
    rng = np.random.default_rng(0)
    x, _ = make_returns(rng)
    corr = corr_matrix(x)
    e, resid = pca_residual_corr(corr, 3)
    np.testing.assert_allclose(np.diag(e), 1.0)
    np.testing.assert_allclose(e, e.T)
    values, vectors = np.linalg.eigh(corr)
    top = vectors[:, ::-1][:, :3]
    np.testing.assert_allclose(resid @ top, 0.0, atol=1e-10)
    explained = ((top * values[::-1][:3]) @ top.T).diagonal()
    np.testing.assert_allclose(np.diag(resid), 1.0 - explained)
    np.testing.assert_allclose(e, resid / np.sqrt(np.outer(np.diag(resid), np.diag(resid))))
    with pytest.raises(ValueError):
        pca_residual_corr(corr, 0)
    with pytest.raises(ValueError):
        pca_residual_corr(corr, corr.shape[0])


def test_sector_share_is_one_for_indicators_and_g_over_n_for_random_vectors() -> None:
    codes = np.repeat(np.arange(5), 60)
    indicator = np.eye(5)[codes][:, :2] * np.array([1.0, -3.0])
    np.testing.assert_allclose(sector_share(indicator, codes, 5), 1.0)
    random = np.random.default_rng(1).standard_normal((codes.size, 400))
    assert sector_share(random, codes, 5).mean() == pytest.approx(5 / 300, abs=0.004)
    with_unknown = np.concatenate([codes, [-1, -1]])
    padded = np.vstack([indicator, np.ones((2, 2))])
    np.testing.assert_allclose(sector_share(padded, with_unknown, 5), 1.0)


def test_permutation_p_values_are_uniform_under_the_null() -> None:
    rng = np.random.default_rng(2)
    codes = np.repeat(np.arange(6), 10)
    p = []
    for _ in range(60):
        corr = corr_matrix(rng.standard_normal((200, codes.size)))
        p.append(permutation_gap_test(corr, codes, 6, 49, rng)["p_value"])
    assert np.mean(p) == pytest.approx(0.5, abs=0.15)
    assert min(p) >= 1 / 50


def test_permutation_test_detects_sector_blocks() -> None:
    rng = np.random.default_rng(3)
    x, codes = make_returns(rng, sector=0.5)
    out = permutation_gap_test(corr_matrix(x), codes, 8, 99, rng)
    assert out["gap"] > 0 and out["p_value"] == pytest.approx(1 / 100)


def test_weak_sector_factors_survive_in_the_residual() -> None:
    rng = np.random.default_rng(4)
    x, codes = make_returns(rng, sector=0.3)
    row, block = window_residual_diagnostics(x, codes, 8, 99, rng)
    assert row["resid_gap"] > 0.03 and row["perm_p"] <= 0.02 and row["perm_z"] > 4
    assert row["k_factors"] >= 1
    assert block.shape == (8, 8) and np.nanmean(np.diag(block)) > np.nanmean(
        block[~np.eye(8, dtype=bool)]
    )


def test_strong_sector_factors_are_absorbed_by_the_retained_components() -> None:
    rng = np.random.default_rng(5)
    x, codes = make_returns(rng, sector=0.9)
    row, _ = window_residual_diagnostics(x, codes, 8, 49, rng)
    assert row["raw_gap"] > 0.3 and row["resid_gap"] < 0.5 * row["raw_gap"]
    assert row["k_factors"] >= 8


def test_no_sector_structure_gives_an_insignificant_gap() -> None:
    rng = np.random.default_rng(6)
    x, codes = make_returns(rng, sector=0.0)
    row, _ = window_residual_diagnostics(x, codes, 8, 99, rng)
    assert abs(row["perm_z"]) < 4.0
    assert row["sector_share_null"] == pytest.approx(8 / 64)


def make_panel(seed: int = 7, n_days: int = 700) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    x, _ = make_returns(rng, n_obs=n_days, n_sectors=4, size=8, sector=0.3)
    x = 0.01 * x
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    symbols = [f"S{i:02d}" for i in range(x.shape[1])]
    log_returns = pd.DataFrame(x, index=dates, columns=symbols)
    universe = pd.DataFrame(True, index=dates[[300, 400]], columns=symbols)
    labels = [["Tech", "Energy", "Health", "Utilities"][i // 8] for i in range(len(symbols))]
    sectors = pd.Series(labels, index=symbols)
    sectors.iloc[0] = "Unknown"
    sectors.iloc[1] = np.nan
    return log_returns, universe, sectors


def test_driver_shapes_reproducibility_and_no_leakage() -> None:
    log_returns, universe, sectors = make_panel()
    args = (250, 0.001, 0.999, 29, 11)
    frame, blocks = residual_diagnostics(log_returns, universe, sectors, *args)
    assert frame.index.equals(universe.index) and len(frame) == 2
    assert blocks.shape == (4, 4) and list(blocks.index) == sorted(blocks.index)
    assert (frame["n_known_sector"] == 30).all() and (frame["n_members"] == 32).all()
    again, _ = residual_diagnostics(log_returns, universe, sectors, *args)
    pd.testing.assert_frame_equal(frame, again)
    changed = log_returns.copy()
    changed.iloc[401:] = 5.0 * np.random.default_rng(0).standard_normal(changed.iloc[401:].shape)
    perturbed, _ = residual_diagnostics(changed, universe, sectors, *args)
    pd.testing.assert_frame_equal(frame, perturbed)
