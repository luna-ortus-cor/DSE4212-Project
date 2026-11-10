"""Tests for the nonlinear-shrinkage strategy on the backtest calendar."""

import dataclasses

import numpy as np
import pandas as pd

from pfopt.backtest.engine import generate_weights
from pfopt.backtest.nonlinear import generate_nls_weights
from pfopt.config import BacktestConfig, Config, UniverseConfig
from pfopt.data.eda import get_window
from pfopt.data.universe import select_universe
from pfopt.estimators.nls import nls_cov
from pfopt.estimators.utils import winsorize_cols
from pfopt.optimizers.gmv import gmv_long_only

CFG = dataclasses.replace(
    Config(),
    universe=UniverseConfig(n_assets=8, lookback_days=40, dollar_volume_window=20),
    backtest=BacktestConfig(rebalance_days=5),
)


def make_data(seed: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=130)
    symbols = [f"S{i:02d}" for i in range(14)]
    market = rng.normal(0.0, 0.008, size=(130, 1))
    returns = market * rng.uniform(0.3, 1.5, size=(1, 14)) + rng.normal(0.0, 0.01, (130, 14))
    volume = rng.uniform(1.0, 3.0, size=(130, 14))
    return (
        pd.DataFrame(returns, index=dates, columns=symbols),
        pd.DataFrame(volume, index=dates, columns=symbols),
    )


def test_calendar_matches_the_core_and_weights_are_valid() -> None:
    data = make_data()
    result = generate_nls_weights(*data, CFG)
    core = generate_weights(*data, CFG, strategies=("GMV-PCA",))
    pd.testing.assert_index_equal(result.weights.index, core.diagnostics.index)
    np.testing.assert_allclose(result.weights.sum(axis=1), 1.0, atol=1e-10)
    assert (result.weights.to_numpy() >= -1e-12).all()
    assert (result.weights.to_numpy()[~core.universe.to_numpy()] == 0.0).all()
    assert {"n_universe", "cond", "h", "pred_vol"} <= set(result.diagnostics.columns)


def test_first_row_matches_a_direct_computation() -> None:
    log_returns, volume = make_data()
    result = generate_nls_weights(log_returns, volume, CFG)
    date = result.weights.index[0]
    universe = select_universe(log_returns, volume, date, 40, 8, 20)
    window = get_window(log_returns, date, 40, universe)
    x = winsorize_cols(window.to_numpy(), CFG.data.winsor_lower, CFG.data.winsor_upper)
    expected = gmv_long_only(nls_cov(x))
    np.testing.assert_allclose(result.weights.loc[date, universe].to_numpy(), expected, atol=1e-8)


def test_no_leakage_and_truncation_prefix() -> None:
    log_returns, volume = make_data()
    full = generate_nls_weights(log_returns, volume, CFG).weights
    cut = full.index[6]
    changed = log_returns.copy()
    later = changed.index > cut
    changed[later] = 5.0 * np.random.default_rng(0).standard_normal(changed[later].shape)
    perturbed = generate_nls_weights(changed, volume, CFG).weights
    truncated = generate_nls_weights(log_returns, volume, CFG, end_date=cut).weights
    pd.testing.assert_frame_equal(full.loc[:cut], perturbed.loc[:cut])
    pd.testing.assert_frame_equal(full.loc[:cut], truncated)
