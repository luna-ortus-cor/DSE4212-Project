"""Tests for the risk-based strategies on the backtest calendar (step 8.7)."""

import dataclasses

import numpy as np
import pandas as pd

from pfopt.backtest.engine import generate_weights
from pfopt.backtest.riskbased import ERC_ESTIMATORS, STRATEGIES, generate_risk_based_weights
from pfopt.backtest.strategies import build_covariances
from pfopt.config import BacktestConfig, Config, UniverseConfig
from pfopt.data.eda import get_window
from pfopt.data.universe import select_universe
from pfopt.estimators.utils import winsorize_cols
from pfopt.optimizers.risk_based import erc_weights, inverse_vol_weights, risk_contributions

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
    result = generate_risk_based_weights(*data, CFG)
    core = generate_weights(*data, CFG, strategies=("GMV-PCA",))
    assert set(result.weights) == set(STRATEGIES)
    for frame in result.weights.values():
        pd.testing.assert_index_equal(frame.index, core.diagnostics.index)
        np.testing.assert_allclose(frame.sum(axis=1), 1.0, atol=1e-10)
        assert (frame.to_numpy()[~core.universe.to_numpy()] == 0.0).all()
    assert {"n_universe", "pred_vol_IVOL", "pred_vol_ERC-PCA"} <= set(result.diagnostics.columns)


def test_first_row_matches_a_direct_computation_and_has_equal_risk_contributions() -> None:
    data = make_data()
    result = generate_risk_based_weights(*data, CFG)
    date = result.weights["IVOL"].index[0]
    universe = select_universe(data[0], data[1], date, 40, 8, 20)
    window = get_window(data[0], date, 40, universe)
    x = winsorize_cols(window.to_numpy(), CFG.data.winsor_lower, CFG.data.winsor_upper)
    covariances, _ = build_covariances(x, CFG)
    np.testing.assert_allclose(
        result.weights["IVOL"].loc[date, universe].to_numpy(),
        inverse_vol_weights(covariances["Sample"]),
        atol=1e-12,
    )
    for strategy, estimator in ERC_ESTIMATORS.items():
        w = result.weights[strategy].loc[date, universe].to_numpy()
        np.testing.assert_allclose(w, erc_weights(covariances[estimator])[0], atol=1e-10)
        np.testing.assert_allclose(risk_contributions(w, covariances[estimator]), 1 / 8, atol=1e-8)


def test_no_leakage_and_truncation_prefix() -> None:
    log_returns, volume = make_data()
    full = generate_risk_based_weights(log_returns, volume, CFG)
    cut = full.weights["IVOL"].index[6]
    changed = log_returns.copy()
    changed[changed.index > cut] = 5.0 * np.random.default_rng(0).standard_normal(
        changed[changed.index > cut].shape
    )
    perturbed = generate_risk_based_weights(changed, volume, CFG)
    truncated = generate_risk_based_weights(log_returns, volume, CFG, end_date=cut)
    for strategy in STRATEGIES:
        before = full.weights[strategy].loc[:cut]
        pd.testing.assert_frame_equal(before, perturbed.weights[strategy].loc[:cut])
        pd.testing.assert_frame_equal(before, truncated.weights[strategy])
