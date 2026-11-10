"""Tests for the regularized strategies on the backtest calendar."""

import dataclasses

import numpy as np
import pandas as pd

from pfopt.backtest.engine import generate_weights
from pfopt.backtest.regularized import generate_regularized_weights, strategy_ids
from pfopt.config import BacktestConfig, Config, RegularizedConfig, UniverseConfig
from pfopt.data.eda import get_window
from pfopt.data.universe import select_universe
from pfopt.estimators.sample import sample_cov
from pfopt.estimators.utils import winsorize_cols
from pfopt.optimizers.gmv import gmv_long_only
from pfopt.optimizers.regularized import gmv_l1, shrink_to_equal

CFG = dataclasses.replace(
    Config(),
    universe=UniverseConfig(n_assets=8, lookback_days=40, dollar_volume_window=20),
    backtest=BacktestConfig(rebalance_days=5),
    regularized=RegularizedConfig(
        ridge_lambdas=(1.0,),
        shrink_gammas=(0.5,),
        l1_bounds=(1.05,),
        enet_bound=1.05,
        enet_lambda=1.0,
    ),
)


def make_data(seed: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=130)
    symbols = [f"S{i:02d}" for i in range(14)]
    market = rng.normal(0.0, 0.008, size=(130, 1))
    returns = market * rng.uniform(0.3, 1.05, size=(1, 14)) + rng.normal(0.0, 0.01, (130, 14))
    volume = rng.uniform(1.0, 3.0, size=(130, 14))
    return (
        pd.DataFrame(returns, index=dates, columns=symbols),
        pd.DataFrame(volume, index=dates, columns=symbols),
    )


def test_calendar_matches_the_core_and_weights_are_valid() -> None:
    data = make_data()
    result = generate_regularized_weights(*data, CFG)
    core = generate_weights(*data, CFG, strategies=("GMV-SAMPLE",))
    ids = strategy_ids(CFG)
    assert set(result.weights) == set(ids)
    for strategy, frame in result.weights.items():
        pd.testing.assert_index_equal(frame.index, core.diagnostics.index)
        np.testing.assert_allclose(frame.sum(axis=1), 1.0, atol=1e-8)
        assert (frame.to_numpy()[~core.universe.to_numpy()] == 0.0).all()
        if ids[strategy]:
            assert frame.to_numpy().min() >= -1e-9
    assert result.weights["GMV-L1-1.05"].to_numpy().min() < 0
    leverage = result.diagnostics["leverage_GMV-L1-1.05"]
    assert leverage.max() <= 1.05 + 1e-5 and leverage.max() > 1.05 - 1e-5
    assert (leverage > 1.0 + 1e-6).any()


def test_first_row_matches_a_direct_computation() -> None:
    data = make_data()
    result = generate_regularized_weights(*data, CFG)
    date = result.weights["GMV-L1-1.05"].index[0]
    universe = select_universe(data[0], data[1], date, 40, 8, 20)
    window = get_window(data[0], date, 40, universe)
    x = winsorize_cols(window.to_numpy(), CFG.data.winsor_lower, CFG.data.winsor_upper)
    sigma = sample_cov(x)
    np.testing.assert_allclose(
        result.weights["GMV-L1-1.05"].loc[date, universe].to_numpy(),
        gmv_l1(sigma, 1.05)[0],
        atol=1e-9,
    )
    np.testing.assert_allclose(
        result.weights["GMV-SHRINK-0.5"].loc[date, universe].to_numpy(),
        shrink_to_equal(gmv_long_only(sigma), 0.5),
        atol=1e-9,
    )


def test_no_leakage_and_truncation_prefix() -> None:
    log_returns, volume = make_data()
    full = generate_regularized_weights(log_returns, volume, CFG)
    cut = full.weights["GMV-L1-1.05"].index[6]
    changed = log_returns.copy()
    changed[changed.index > cut] = 5.0 * np.random.default_rng(0).standard_normal(
        changed[changed.index > cut].shape
    )
    perturbed = generate_regularized_weights(changed, volume, CFG)
    truncated = generate_regularized_weights(log_returns, volume, CFG, end_date=cut)
    for strategy in strategy_ids(CFG):
        before = full.weights[strategy].loc[:cut]
        pd.testing.assert_frame_equal(before, perturbed.weights[strategy].loc[:cut])
        pd.testing.assert_frame_equal(before, truncated.weights[strategy])
