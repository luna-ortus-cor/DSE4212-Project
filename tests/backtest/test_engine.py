"""Tests for leakage-free weight generation on a small synthetic panel.

Panel: 130 business days, 14 symbols. One symbol ("LATE") has no data for the first 50 days.
Settings: N = 8 names, T = 40 days of history, 20-day dollar-volume window, rebalance every 5 days.
"""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from pfopt.backtest.engine import (
    ALL_STRATEGIES,
    generate_weights,
    read_weights_result,
    write_weights_result,
)
from pfopt.backtest.strategies import PREDICTION_COVARIANCE
from pfopt.config import BacktestConfig, Config, UniverseConfig
from pfopt.data.universe import first_feasible_date
from pfopt.estimators.sample import sample_cov
from pfopt.estimators.utils import winsorize_cols

N_DATES, N_SYMBOLS, T, N, W, STEP = 130, 14, 40, 8, 20, 5
CFG = dataclasses.replace(
    Config(),
    universe=UniverseConfig(n_assets=N, lookback_days=T, dollar_volume_window=W),
    backtest=BacktestConfig(rebalance_days=STEP),
)


def make_data(seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=N_DATES)
    symbols = [f"S{i:02d}" for i in range(N_SYMBOLS - 1)] + ["LATE"]
    market = rng.normal(0.0, 0.008, size=(N_DATES, 1))
    returns = 0.0004 + market + rng.normal(0.0, 0.01, size=(N_DATES, N_SYMBOLS))
    log_returns = pd.DataFrame(returns, index=dates, columns=symbols)
    log_returns.loc[dates[:50], "LATE"] = np.nan
    scale = rng.uniform(1.0, 3.0, size=N_SYMBOLS)
    volume = scale * rng.uniform(1.0, 2.0, size=(N_DATES, N_SYMBOLS))
    dollar_volume = pd.DataFrame(volume, index=dates, columns=symbols).where(log_returns.notna())
    return log_returns, dollar_volume


@pytest.fixture(scope="module")
def data() -> tuple[pd.DataFrame, pd.DataFrame]:
    return make_data()


@pytest.fixture(scope="module")
def result(data):
    return generate_weights(*data, CFG)


def test_calendar_starts_at_first_feasible_date_and_steps(data, result) -> None:
    log_returns, _ = data
    dates = result.diagnostics.index
    assert dates[0] == first_feasible_date(log_returns, T, N)
    positions = [log_returns.index.get_loc(d) for d in dates]
    assert np.all(np.diff(positions) == STEP)
    assert dates.equals(result.universe.index)
    for frame in result.weights.values():
        assert frame.index.equals(dates)


def test_all_strategies_present_with_valid_rows(data, result) -> None:
    log_returns, _ = data
    assert set(result.weights) == set(ALL_STRATEGIES)
    for strategy, frame in result.weights.items():
        assert list(frame.columns) == list(log_returns.columns)
        assert (frame.to_numpy() >= 0).all(), strategy
        np.testing.assert_allclose(frame.sum(axis=1), 1.0, atol=1e-8, err_msg=strategy)
        outside = frame.to_numpy()[~result.universe.to_numpy()]
        assert (outside == 0).all(), strategy


def test_universe_membership_matches_the_point_in_time_rule(data, result) -> None:
    log_returns, _ = data
    assert (result.universe.sum(axis=1) == N).all()
    late_first_full_window = log_returns.index[50 + T - 1]
    early = result.universe.index < late_first_full_window
    assert not result.universe.loc[early, "LATE"].any()
    assert result.diagnostics["n_universe"].eq(N).all()


def test_diagnostics_columns(result) -> None:
    columns = set(result.diagnostics.columns)
    expected = {
        "n_universe",
        "lw_shrinkage",
        "rmt_n_signal",
        "rmt_lam_plus",
        "pca_K",
        "n_solver_fallbacks",
        "cond_Sample",
        "cond_Ledoit-Wolf",
        "cond_RMT",
        "cond_PCA",
        "solver_GMV-SAMPLE",
        "solver_GMV-LW",
        "solver_GMV-RMT",
        "solver_GMV-PCA",
        "solver_MV-SAMPLE",
        "solver_MV-RMT",
    }
    assert expected <= columns
    assert (result.diagnostics["pca_K"] >= 1).all()
    assert (result.diagnostics["cond_Sample"] >= 1.0).all()


def test_predicted_volatility_columns(data, result) -> None:
    log_returns, _ = data
    columns = result.diagnostics.columns
    assert "pred_vol_EW" not in columns
    for strategy in PREDICTION_COVARIANCE:
        assert (result.diagnostics[f"pred_vol_{strategy}"] > 0).all()
    date = result.diagnostics.index[4]
    universe = list(result.universe.columns[result.universe.loc[date].to_numpy()])
    pos = log_returns.index.get_loc(date)
    window = log_returns.iloc[pos - T + 1 : pos + 1][universe].to_numpy()
    x = winsorize_cols(window, CFG.data.winsor_lower, CFG.data.winsor_upper)
    w = result.weights["GMV-SAMPLE"].loc[date, universe].to_numpy()
    expected = np.sqrt(w @ sample_cov(x) @ w) * np.sqrt(252)
    assert result.diagnostics.loc[date, "pred_vol_GMV-SAMPLE"] == pytest.approx(expected, rel=1e-9)


def test_no_leakage_corrupting_the_future_leaves_past_weights_bit_identical(data, result) -> None:
    log_returns, dollar_volume = data
    cutoff = log_returns.index[95]
    rng = np.random.default_rng(123)
    future = log_returns.index > cutoff
    bad_returns, bad_volume = log_returns.copy(), dollar_volume.copy()
    bad_returns.loc[future] = 5.0 * rng.normal(size=bad_returns.loc[future].shape)
    bad_volume.loc[future] = rng.uniform(0.1, 9.0, size=bad_volume.loc[future].shape)
    corrupted = generate_weights(bad_returns, bad_volume, CFG)

    past = result.diagnostics.index <= cutoff
    assert past.sum() >= 5
    for strategy in ALL_STRATEGIES:
        before = result.weights[strategy].loc[past].to_numpy()
        after = corrupted.weights[strategy].loc[corrupted.diagnostics.index <= cutoff].to_numpy()
        assert np.array_equal(before, after), strategy
    assert np.array_equal(
        result.universe.loc[past].to_numpy(),
        corrupted.universe.loc[corrupted.diagnostics.index <= cutoff].to_numpy(),
    )
    # The corruption does change later weights, so the check above is not vacuous.
    later = result.diagnostics.index > cutoff
    assert not np.array_equal(
        result.weights["GMV-SAMPLE"].loc[later].to_numpy(),
        corrupted.weights["GMV-SAMPLE"].loc[corrupted.diagnostics.index > cutoff].to_numpy(),
    )


def test_truncation_gives_an_identical_prefix(data, result) -> None:
    end = result.diagnostics.index[6]
    short = generate_weights(*data, CFG, end_date=end)
    assert short.diagnostics.index[-1] == end
    for strategy in ALL_STRATEGIES:
        assert np.array_equal(
            short.weights[strategy].to_numpy(), result.weights[strategy].iloc[:7].to_numpy()
        )


def test_strategy_subset_matches_the_full_run(data, result) -> None:
    subset = generate_weights(*data, CFG, strategies=["EW", "GMV-SAMPLE"])
    assert set(subset.weights) == {"EW", "GMV-SAMPLE"}
    for strategy in subset.weights:
        assert np.array_equal(
            subset.weights[strategy].to_numpy(), result.weights[strategy].to_numpy()
        )
    assert "solver_MV-SAMPLE" not in subset.diagnostics.columns


def test_cap_applies_to_optimized_strategies(data) -> None:
    cfg = dataclasses.replace(CFG, backtest=BacktestConfig(rebalance_days=STEP, max_weight=0.25))
    res = generate_weights(
        *data, cfg, strategies=["EW", "GMV-SAMPLE", "GMV-RMT", "MV-SAMPLE", "MV-RMT"]
    )
    for strategy in res.weights:
        assert res.weights[strategy].to_numpy().max() <= 0.25 + 1e-8, strategy


def test_deterministic(data, result) -> None:
    again = generate_weights(*data, CFG)
    for strategy in ALL_STRATEGIES:
        assert np.array_equal(
            again.weights[strategy].to_numpy(), result.weights[strategy].to_numpy()
        )


def test_write_and_read_roundtrip(result, tmp_path) -> None:
    write_weights_result(result, tmp_path / "cache")
    back = read_weights_result(tmp_path / "cache")
    assert set(back.weights) == set(result.weights)
    for strategy in result.weights:
        pd.testing.assert_frame_equal(
            back.weights[strategy], result.weights[strategy], check_freq=False
        )
    pd.testing.assert_frame_equal(back.universe, result.universe, check_freq=False)
    pd.testing.assert_frame_equal(back.diagnostics, result.diagnostics, check_freq=False)
    with pytest.raises(FileNotFoundError):
        read_weights_result(tmp_path / "missing")


def test_invalid_requests_raise(data) -> None:
    with pytest.raises(ValueError, match="unknown strategy"):
        generate_weights(*data, CFG, strategies=["EW", "NOPE"])
    with pytest.raises(ValueError, match="no rebalance dates"):
        generate_weights(*data, CFG, end_date="2019-01-01")


def test_error_when_too_few_names_are_eligible_later(data) -> None:
    log_returns, dollar_volume = data
    broken = log_returns.copy()
    broken.iloc[100, :8] = np.nan  # leaves only 6 names with a full window around day 100
    with pytest.raises(ValueError, match="need 8"):
        generate_weights(broken, dollar_volume.where(broken.notna()), CFG)
