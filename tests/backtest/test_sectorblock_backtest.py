"""Tests for the PCA plus sector-block strategy on the backtest calendar (step 8.24)."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from pfopt.backtest.engine import generate_weights
from pfopt.backtest.residual import sector_codes
from pfopt.backtest.sectorblock import (
    covariance_from_config,
    generate_sector_block_weights,
    sector_categories,
)
from pfopt.config import BacktestConfig, Config, UniverseConfig
from pfopt.data.eda import get_window
from pfopt.data.universe import select_universe
from pfopt.estimators.utils import winsorize_cols
from pfopt.optimizers.gmv import gmv_long_only

N_DATES, N_SYMBOLS = 130, 14
CFG = dataclasses.replace(
    Config(),
    universe=UniverseConfig(n_assets=8, lookback_days=40, dollar_volume_window=20),
    backtest=BacktestConfig(rebalance_days=5),
)


def make_data(seed: int = 1) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=N_DATES)
    symbols = [f"S{i:02d}" for i in range(N_SYMBOLS)]
    market = rng.normal(0.0, 0.008, size=(N_DATES, 1))
    betas = rng.uniform(0.3, 1.5, size=(1, N_SYMBOLS))
    returns = market * betas + rng.normal(0.0, 0.01, size=(N_DATES, N_SYMBOLS))
    volume = rng.uniform(1.0, 3.0, size=(N_DATES, N_SYMBOLS))
    labels = ["Tech", "Energy", "Health"] * 4 + ["Unknown", np.nan]
    return (
        pd.DataFrame(returns, index=dates, columns=symbols),
        pd.DataFrame(volume, index=dates, columns=symbols),
        pd.Series(labels, index=symbols),
    )


@pytest.fixture(scope="module")
def data():
    return make_data()


@pytest.fixture(scope="module")
def result(data):
    return generate_sector_block_weights(*data, CFG)


def test_sector_categories_drop_unknown_and_missing(data) -> None:
    assert sector_categories(data[2]) == ["Energy", "Health", "Tech"]


def test_calendar_matches_the_core_and_weights_are_valid(data, result) -> None:
    core = generate_weights(data[0], data[1], CFG, strategies=("GMV-PCA",))
    pd.testing.assert_index_equal(result.weights.index, core.diagnostics.index)
    np.testing.assert_allclose(result.weights.sum(axis=1), 1.0, atol=1e-10)
    assert (result.weights.to_numpy() >= -1e-12).all()
    assert (result.weights.to_numpy()[~core.universe.to_numpy()] == 0.0).all()
    assert {"n_known_sector", "K", "block_share", "pred_vol"} <= set(result.diagnostics.columns)


def test_first_row_matches_a_direct_computation(data, result) -> None:
    log_returns, volume, sectors = data
    date = result.weights.index[0]
    universe = select_universe(log_returns, volume, date, 40, 8, 20)
    window = get_window(log_returns, date, 40, universe)
    x = winsorize_cols(window.to_numpy(), CFG.data.winsor_lower, CFG.data.winsor_upper)
    groups = sector_codes(universe, sectors, sector_categories(sectors))
    expected = gmv_long_only(covariance_from_config(x, groups, CFG)[0])
    np.testing.assert_allclose(result.weights.loc[date, universe].to_numpy(), expected, atol=1e-8)


def test_without_sector_labels_the_strategy_is_the_pca_strategy(data) -> None:
    log_returns, volume, sectors = data
    blank = pd.Series(np.nan, index=sectors.index)
    got = generate_sector_block_weights(log_returns, volume, blank, CFG)
    pca = generate_weights(log_returns, volume, CFG, strategies=("GMV-PCA",))
    np.testing.assert_allclose(got.weights.to_numpy(), pca.weights["GMV-PCA"].to_numpy(), atol=1e-6)


def test_no_leakage_and_truncation_prefix(data, result) -> None:
    log_returns, volume, sectors = data
    cutoff = result.weights.index[len(result.weights) // 2]
    rng = np.random.default_rng(7)
    future = log_returns.index > cutoff
    bad_returns, bad_volume = log_returns.copy(), volume.copy()
    bad_returns.loc[future] = 0.03 * rng.normal(size=bad_returns.loc[future].shape)
    bad_volume.loc[future] = rng.uniform(size=bad_volume.loc[future].shape)
    corrupted = generate_sector_block_weights(bad_returns, bad_volume, sectors, CFG)
    short = generate_sector_block_weights(log_returns, volume, sectors, CFG, end_date=cutoff)
    past = result.weights.loc[:cutoff].to_numpy()
    np.testing.assert_array_equal(past, corrupted.weights.loc[:cutoff].to_numpy())
    np.testing.assert_array_equal(past, short.weights.to_numpy())
