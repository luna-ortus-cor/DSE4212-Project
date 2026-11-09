"""Tests for the EWMA strategies on the backtest calendar (step 8.22)."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from pfopt.backtest.engine import generate_weights
from pfopt.backtest.nonstationary import ewma_strategy_ids, generate_ewma_weights
from pfopt.config import BacktestConfig, Config, NonStationarityConfig, UniverseConfig
from pfopt.data.eda import get_window
from pfopt.data.universe import select_universe
from pfopt.estimators.ewma import ewma_cov
from pfopt.estimators.utils import winsorize_cols
from pfopt.optimizers.gmv import gmv_long_only

N_DATES, N_SYMBOLS = 130, 14
CFG = dataclasses.replace(
    Config(),
    universe=UniverseConfig(n_assets=8, lookback_days=40, dollar_volume_window=20),
    backtest=BacktestConfig(rebalance_days=5),
    nonstationarity=NonStationarityConfig(halflives=(10, 20)),
)


def make_data(seed: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=N_DATES)
    symbols = [f"S{i:02d}" for i in range(N_SYMBOLS)]
    market = rng.normal(0.0, 0.008, size=(N_DATES, 1))
    betas = rng.uniform(0.3, 1.5, size=(1, N_SYMBOLS))
    returns = market * betas + rng.normal(0.0, 0.01, size=(N_DATES, N_SYMBOLS))
    volume = rng.uniform(1.0, 3.0, size=(N_DATES, N_SYMBOLS))
    return (
        pd.DataFrame(returns, index=dates, columns=symbols),
        pd.DataFrame(volume, index=dates, columns=symbols),
    )


@pytest.fixture(scope="module")
def data():
    return make_data()


@pytest.fixture(scope="module")
def result(data):
    return generate_ewma_weights(*data, CFG)


def test_ids_and_calendar(data, result) -> None:
    assert ewma_strategy_ids((10, 20)) == (
        "GMV-EWMA10",
        "GMV-EWMA10-RMT",
        "GMV-EWMA20",
        "GMV-EWMA20-RMT",
    )
    assert list(result.weights) == list(ewma_strategy_ids((10, 20)))
    core = generate_weights(*data, CFG, strategies=("GMV-SAMPLE",))
    pd.testing.assert_index_equal(result.diagnostics.index, core.diagnostics.index)
    for frame in result.weights.values():
        np.testing.assert_allclose(frame.sum(axis=1), 1.0, atol=1e-10)
        assert (frame.to_numpy() >= -1e-12).all()
        assert (frame.to_numpy()[~core.universe.to_numpy()] == 0.0).all()
    assert (result.diagnostics["t_eff_10"] < result.diagnostics["t_eff_20"]).all()


def test_first_row_matches_a_direct_computation(data, result) -> None:
    log_returns, volume = data
    date = result.diagnostics.index[0]
    universe = select_universe(log_returns, volume, date, 40, 8, 20)
    window = get_window(log_returns, date, 40, universe)
    x = winsorize_cols(window.to_numpy(), CFG.data.winsor_lower, CFG.data.winsor_upper)
    expected = gmv_long_only(ewma_cov(x, 10))
    got = result.weights["GMV-EWMA10"].loc[date, universe].to_numpy()
    np.testing.assert_allclose(got, expected, atol=1e-8)


def test_no_leakage_and_truncation_prefix(data, result) -> None:
    log_returns, volume = data
    cutoff = result.diagnostics.index[len(result.diagnostics) // 2]
    rng = np.random.default_rng(7)
    future = log_returns.index > cutoff
    bad_returns, bad_volume = log_returns.copy(), volume.copy()
    bad_returns.loc[future] = 0.03 * rng.normal(size=bad_returns.loc[future].shape)
    bad_volume.loc[future] = rng.uniform(size=bad_volume.loc[future].shape)
    corrupted = generate_ewma_weights(bad_returns, bad_volume, CFG)
    short = generate_ewma_weights(log_returns, volume, CFG, end_date=cutoff)
    for strategy in result.weights:
        past = result.weights[strategy].loc[:cutoff]
        np.testing.assert_array_equal(past.to_numpy(), corrupted.weights[strategy].loc[:cutoff])
        np.testing.assert_array_equal(past.to_numpy(), short.weights[strategy].to_numpy())


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        NonStationarityConfig(halflives=())
    with pytest.raises(ValueError):
        NonStationarityConfig(calibration_block_days=2)
    with pytest.raises(ValueError):
        NonStationarityConfig(sim_n_reps=1)
