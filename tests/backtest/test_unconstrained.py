"""Tests for the unconstrained GMV strategies, their simulation and the H3 bootstrap statistic."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from pfopt.backtest.engine import generate_weights, simulate
from pfopt.backtest.stats import log_var_double_difference
from pfopt.backtest.strategies import build_covariances, core_strategies
from pfopt.backtest.unconstrained import UNCONSTRAINED_STRATEGIES, generate_unconstrained_weights
from pfopt.config import BacktestConfig, Config, UniverseConfig
from pfopt.data.eda import get_window
from pfopt.data.universe import select_universe
from pfopt.estimators.utils import validate_returns, winsorize_cols

N_DATES, N_SYMBOLS = 130, 14
CFG = dataclasses.replace(
    Config(),
    universe=UniverseConfig(n_assets=8, lookback_days=40, dollar_volume_window=20),
    backtest=BacktestConfig(rebalance_days=5),
)


def make_data(seed: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=N_DATES)
    symbols = [f"S{i:02d}" for i in range(N_SYMBOLS)]
    market = rng.normal(0.0, 0.008, size=(N_DATES, 1))
    betas = rng.uniform(0.3, 1.5, size=(1, N_SYMBOLS))
    returns = 0.0004 + market * betas + rng.normal(0.0, 0.01, size=(N_DATES, N_SYMBOLS))
    log_returns = pd.DataFrame(returns, index=dates, columns=symbols)
    volume = pd.DataFrame(
        rng.uniform(1.0, 3.0, size=(N_DATES, N_SYMBOLS)), index=dates, columns=symbols
    )
    return log_returns, volume


@pytest.fixture(scope="module")
def data() -> tuple[pd.DataFrame, pd.DataFrame]:
    return make_data()


@pytest.fixture(scope="module")
def result(data):
    return generate_unconstrained_weights(*data, CFG)


def test_build_covariances_matches_core_strategies() -> None:
    rng = np.random.default_rng(0)
    x = validate_returns(rng.normal(0.0, 0.01, size=(200, 10)))
    covariances, info = build_covariances(x, CFG)
    core = core_strategies(x, CFG)
    for name, matrix in covariances.items():
        np.testing.assert_array_equal(matrix, core.covariances[name])
    assert info["pca_K"] == core.info["pca_K"] and info["rmt_n_signal"] == core.info["rmt_n_signal"]


def test_rows_sum_to_one_have_shorts_and_match_the_long_only_calendar(data, result) -> None:
    long_only = generate_weights(*data, CFG, strategies=("GMV-SAMPLE",))
    assert list(result.weights) == list(UNCONSTRAINED_STRATEGIES)
    pd.testing.assert_index_equal(result.diagnostics.index, long_only.diagnostics.index)
    for frame in result.weights.values():
        np.testing.assert_allclose(frame.sum(axis=1), 1.0, atol=1e-10)
        outside = ~long_only.universe.to_numpy()
        assert (frame.to_numpy()[outside] == 0.0).all()
    sample = result.weights["GMV-SAMPLE-UNC"]
    assert (sample.to_numpy() < -1e-6).any()
    leverage = sample.abs().sum(axis=1)
    np.testing.assert_allclose(result.diagnostics["leverage_GMV-SAMPLE-UNC"], leverage)
    assert (leverage >= 1.0 - 1e-12).all()


def test_first_row_matches_the_closed_form_on_the_same_window(data, result) -> None:
    log_returns, volume = data
    date = result.diagnostics.index[0]
    universe = select_universe(log_returns, volume, date, 40, 8, 20)
    window = get_window(log_returns, date, 40, universe)
    x = winsorize_cols(window.to_numpy(), CFG.data.winsor_lower, CFG.data.winsor_upper)
    sigma = np.cov(x, rowvar=False)
    solved = np.linalg.solve(sigma, np.ones(8))
    expected = solved / solved.sum()
    got = result.weights["GMV-SAMPLE-UNC"].loc[date, universe].to_numpy()
    np.testing.assert_allclose(got, expected, atol=1e-8)


def test_no_leakage_and_truncation_prefix(data, result) -> None:
    log_returns, volume = data
    cutoff = result.diagnostics.index[len(result.diagnostics) // 2]
    rng = np.random.default_rng(7)
    future = log_returns.index > cutoff
    bad_returns, bad_volume = log_returns.copy(), volume.copy()
    bad_returns.loc[future] = 0.03 * rng.normal(size=bad_returns.loc[future].shape)
    bad_volume.loc[future] = rng.uniform(size=bad_volume.loc[future].shape)
    corrupted = generate_unconstrained_weights(bad_returns, bad_volume, CFG)
    short = generate_unconstrained_weights(log_returns, volume, CFG, end_date=cutoff)
    for strategy in UNCONSTRAINED_STRATEGIES:
        past = result.weights[strategy].loc[:cutoff]
        np.testing.assert_array_equal(past.to_numpy(), corrupted.weights[strategy].loc[:cutoff])
        np.testing.assert_array_equal(past.to_numpy(), short.weights[strategy].to_numpy())


def test_simulate_allows_shorts_only_when_asked() -> None:
    days = pd.bdate_range("2021-01-01", periods=6)
    returns = pd.DataFrame({"A": [0.0, 0.01, 0.01, 0.01, 0.01, 0.01], "B": 0.0}, index=days)
    weights = pd.DataFrame({"A": [1.5], "B": [-0.5]}, index=days[[0]])
    with pytest.raises(ValueError):
        simulate(weights, returns, 0.0)
    sim = simulate(weights, returns, 0.0, long_only=False)
    assert sim.gross.iloc[0] == pytest.approx(0.015)
    assert sim.turnover.iloc[0] == pytest.approx(2.0)
    assert sim.n_missing_returns == 0
    drifted = 1.5 * 1.01 / 1.015
    assert sim.daily_weights["A"].iloc[1] == pytest.approx(drifted)


def test_double_difference_known_scaling_and_null() -> None:
    rng = np.random.default_rng(3)
    base = rng.normal(0.0, 0.01, size=1500)
    other = rng.normal(0.0, 0.01, size=1500)
    frame = pd.DataFrame({"a": 0.5 * base, "a_ref": base, "b": other, "b_ref": other})
    out = log_var_double_difference(
        frame, {"scaled": ("a", "a_ref", "b", "b_ref")}, block_length=5, n_boot=300, seed=1
    )
    row = out.iloc[0]
    assert row["gap_a"] == pytest.approx(np.log(0.25)) and row["gap_b"] == pytest.approx(0.0)
    assert row["estimate"] == pytest.approx(np.log(0.25))
    assert row["ci_high"] < 0 and row["significant"]

    null = pd.DataFrame(rng.normal(0.0, 0.01, size=(1500, 4)), columns=["a", "a_ref", "b", "b_ref"])
    out = log_var_double_difference(
        null, {"n": ("a", "a_ref", "b", "b_ref")}, block_length=5, n_boot=300, seed=1, alpha=0.01
    )
    row = out.iloc[0]
    # Coverage of 0 under the null is probabilistic (the estimate itself is noisy), so check the
    # deterministic properties: the interval brackets the point estimate and has positive width.
    assert row["ci_low"] < row["estimate"] < row["ci_high"]
    assert row["gap_a"] - row["gap_b"] == pytest.approx(row["estimate"])
    assert 0.0 < row["p_value"] <= 1.0

    identical = null[["a", "a_ref"]].rename(columns={"a": "b", "a_ref": "b_ref"})
    identical = pd.concat([null[["a", "a_ref"]], identical], axis=1)
    out = log_var_double_difference(
        identical, {"same": ("a", "a_ref", "b", "b_ref")}, block_length=5, n_boot=300, seed=1
    )
    assert out.iloc[0]["estimate"] == 0.0 and out.iloc[0]["ci_low"] == out.iloc[0]["ci_high"] == 0.0
    assert out.iloc[0]["p_value"] == 1.0 and not out.iloc[0]["significant"]
    with pytest.raises(ValueError, match="not found"):
        log_var_double_difference(null, {"x": ("a", "a_ref", "b", "zzz")}, n_boot=10)
