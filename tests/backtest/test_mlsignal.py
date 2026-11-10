"""Tests for the walk-forward ML signals."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from pfopt.backtest.mlsignal import (
    FEATURES,
    MODELS,
    build_panel,
    forward_returns,
    ic_gate,
    ic_summary,
    rank_ic,
    rank_scores,
    raw_features,
    training_indices,
    walk_forward_signals,
)
from pfopt.config import BacktestConfig, Config, MLConfig, UniverseConfig
from pfopt.data.universe import rebalance_dates

ML = MLConfig(
    momentum_days=20,
    skip_days=5,
    reversal_days=5,
    vol_days=10,
    horizon_days=5,
    purge_days=5,
    min_train_dates=10,
    retrain_every=4,
    ridge_alpha=0.1,
    rf_trees=20,
    rf_max_depth=3,
    rf_min_leaf=5,
    rf_max_features=1.0,
)
CFG = dataclasses.replace(
    Config(),
    universe=UniverseConfig(n_assets=8, lookback_days=40, dollar_volume_window=20),
    backtest=BacktestConfig(rebalance_days=5),
    ml=ML,
)


def make_data(seed: int = 3, n_days: int = 400) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Panel where each symbol has a persistent drift, so momentum predicts the next return."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    symbols = [f"S{i:02d}" for i in range(14)]
    drift = np.linspace(-0.004, 0.004, 14)[rng.permutation(14)]
    market = rng.normal(0.0, 0.006, size=(n_days, 1))
    returns = drift + market + rng.normal(0.0, 0.01, (n_days, 14))
    volume = rng.uniform(1.0, 3.0, size=(n_days, 14))
    return (
        pd.DataFrame(returns, index=dates, columns=symbols),
        pd.DataFrame(volume, index=dates, columns=symbols),
    )


def test_rank_scores_known_answers_and_errors() -> None:
    np.testing.assert_allclose(rank_scores(np.array([10.0, 30.0, 20.0])), [-0.5, 0.5, 0.0])
    np.testing.assert_allclose(rank_scores(np.array([1.0, 1.0, 2.0])), [-0.25, -0.25, 0.5])
    with pytest.raises(ValueError):
        rank_scores(np.array([1.0]))
    with pytest.raises(ValueError):
        rank_scores(np.array([1.0, np.nan, 2.0]))


def test_rank_ic_known_answers() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert rank_ic(x, 10 * x + 1) == pytest.approx(1.0)
    assert rank_ic(x, -x) == pytest.approx(-1.0)
    assert np.isnan(rank_ic(x, np.ones(4)))
    assert np.isnan(rank_ic(x[:2], x[:2]))
    with_nan = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
    assert rank_ic(with_nan, np.array([1.0, 2.0, 3.0, 99.0, 5.0])) == pytest.approx(1.0)


def tiny_cfg() -> Config:
    return dataclasses.replace(
        Config(),
        universe=UniverseConfig(n_assets=2, lookback_days=6, dollar_volume_window=3),
        ml=MLConfig(
            momentum_days=6,
            skip_days=2,
            reversal_days=2,
            vol_days=3,
            horizon_days=2,
            purge_days=2,
            min_train_dates=2,
        ),
    )


def tiny_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2021-01-04", periods=10)
    i = np.arange(10)
    returns = pd.DataFrame({"A": 0.01 * i, "B": 0.01 * (i % 3)}, index=dates)
    volume = pd.DataFrame({"A": i + 1.0, "B": np.full(10, 5.0)}, index=dates)
    return returns, volume


def test_raw_features_known_answer() -> None:
    returns, volume = tiny_frames()
    out = raw_features(returns, volume, returns.index[9], ["A", "B"], tiny_cfg())
    # momentum: rows 4..7; reversal: rows 8..9; vol: std of rows 7..9; dollar volume: rows 7..9
    expected = [[0.22, 0.17, 0.01, 9.0], [0.04, 0.02, 0.01, 5.0]]
    np.testing.assert_allclose(out, expected, atol=1e-12)


def test_raw_features_ignore_future_rows() -> None:
    returns, volume = tiny_frames()
    base = raw_features(returns, volume, returns.index[7], ["A", "B"], tiny_cfg())
    shocked_returns, shocked_volume = returns.copy(), volume.copy()
    shocked_returns.iloc[8:] = 9.0
    shocked_volume.iloc[8:] = 1e6
    again = raw_features(shocked_returns, shocked_volume, returns.index[7], ["A", "B"], tiny_cfg())
    np.testing.assert_array_equal(base, again)


def test_raw_features_reject_short_history() -> None:
    returns, volume = tiny_frames()
    with pytest.raises(ValueError):
        raw_features(returns, volume, returns.index[3], ["A", "B"], tiny_cfg())


def test_forward_returns_known_answer() -> None:
    returns, _ = tiny_frames()
    out = forward_returns(returns, returns.index[7], ["A", "B"], 2)
    np.testing.assert_allclose(out, [0.08 + 0.09, 0.02 + 0.0], atol=1e-12)
    assert np.isnan(forward_returns(returns, returns.index[8], ["A", "B"], 2)).all()


def test_training_indices_purge() -> None:
    positions = np.array([10, 15, 20, 25, 30])
    np.testing.assert_array_equal(training_indices(positions, 4, 5), [0, 1, 2, 3])
    np.testing.assert_array_equal(training_indices(positions, 4, 10), [0, 1, 2])
    assert len(training_indices(positions, 0, 5)) == 0


def test_panel_calendar_ranks_and_no_overlap_of_training_labels() -> None:
    returns, volume = make_data()
    panel = build_panel(returns, volume, CFG)
    pd.testing.assert_index_equal(
        panel.dates, rebalance_dates(returns, 40, 8, 5).rename("Date"), check_names=False
    )
    for features in panel.features:
        assert features.shape == (8, len(FEATURES))
        np.testing.assert_allclose(features.min(axis=0), -0.5)
        np.testing.assert_allclose(features.max(axis=0), 0.5)
    for k in range(len(panel.dates)):
        selected = training_indices(panel.positions, k, CFG.ml.purge_days)
        assert (panel.positions[selected] + CFG.ml.horizon_days <= panel.positions[k]).all()


def test_panel_rejects_windows_longer_than_the_lookback() -> None:
    returns, volume = make_data()
    cfg = dataclasses.replace(CFG, ml=dataclasses.replace(ML, momentum_days=60))
    with pytest.raises(ValueError):
        build_panel(returns, volume, cfg)


def test_walk_forward_schedule_universe_and_signal() -> None:
    returns, volume = make_data()
    panel = build_panel(returns, volume, CFG)
    result = walk_forward_signals(panel, CFG)
    first = panel.dates.get_loc(result.ic.index[0])
    assert first > 0 and result.fits["date"].iloc[0] == panel.dates[first]
    assert result.fits["n_dates"].iloc[0] == CFG.ml.min_train_dates
    assert result.fits["n_dates"].is_monotonic_increasing
    steps = np.diff(panel.dates.get_indexer(pd.DatetimeIndex(result.fits["date"])))
    assert (steps == CFG.ml.retrain_every).all()
    assert list(result.ic.columns) == [*MODELS, *FEATURES]
    for model in MODELS:
        frame = result.predictions[model]
        pd.testing.assert_index_equal(frame.index, panel.dates[first:])
        for k, date in enumerate(frame.index):
            finite = frame.loc[date].dropna().index
            assert set(finite) == set(panel.symbols[first + k])
    summary = ic_summary(result.ic)
    assert summary.loc["RIDGE", "mean_ic"] > 0.1
    assert summary.loc["mom_12_1", "mean_ic"] > 0.0
    assert ic_gate(summary, MODELS, 2.0)["RIDGE"]


def test_predictions_do_not_use_future_data() -> None:
    returns, volume = make_data()
    base = walk_forward_signals(build_panel(returns, volume, CFG), CFG)
    cut = 250
    last_kept = returns.index[cut]
    rng = np.random.default_rng(99)
    shocked_returns, shocked_volume = returns.copy(), volume.copy()
    shocked_returns.iloc[cut + 1 :] += rng.normal(0.0, 0.05, shocked_returns.iloc[cut + 1 :].shape)
    shocked_volume.iloc[cut + 1 :] = rng.uniform(1.0, 3.0, shocked_volume.iloc[cut + 1 :].shape)
    again = walk_forward_signals(build_panel(shocked_returns, shocked_volume, CFG), CFG)
    for model in MODELS:
        kept = base.predictions[model].loc[:last_kept]
        assert len(kept) > CFG.ml.retrain_every
        pd.testing.assert_frame_equal(kept, again.predictions[model].loc[:last_kept])
        assert (
            not base.predictions[model]
            .loc[last_kept:]
            .equals(again.predictions[model].loc[last_kept:])
        )


def test_walk_forward_needs_enough_dates() -> None:
    returns, volume = make_data(n_days=60)
    panel = build_panel(returns, volume, CFG)
    with pytest.raises(ValueError):
        walk_forward_signals(panel, CFG)


def test_ic_summary_and_gate_known_answers() -> None:
    ic = pd.DataFrame(
        {"a": [0.1, 0.2, 0.3], "b": [-0.1, -0.2, -0.3], "c": [0.1, np.nan, 0.1]},
        index=pd.RangeIndex(3),
    )
    summary = ic_summary(ic)
    assert summary.loc["a", "mean_ic"] == pytest.approx(0.2)
    assert summary.loc["a", "std_ic"] == pytest.approx(0.1)
    assert summary.loc["a", "t_stat"] == pytest.approx(0.2 / (0.1 / np.sqrt(3)))
    assert summary.loc["a", "hit_rate"] == 1.0 and summary.loc["a", "n_dates"] == 3
    assert summary.loc["c", "n_dates"] == 2 and np.isnan(summary.loc["c", "t_stat"])
    assert ic_gate(summary, ("a", "b", "c"), 2.0) == {"a": True, "b": False, "c": False}
    assert ic_gate(summary, ("a",), 4.0) == {"a": False}
