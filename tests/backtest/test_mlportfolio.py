"""Tests for the ML-fed max-Sharpe portfolios."""

import dataclasses
import functools

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from pfopt.backtest.mlportfolio import ML_STRATEGIES, alpha_vector, generate_ml_weights
from pfopt.backtest.mlsignal import build_panel, walk_forward_signals
from pfopt.config import BacktestConfig, Config, MLConfig, UniverseConfig

CFG = dataclasses.replace(
    Config(),
    universe=UniverseConfig(n_assets=8, lookback_days=40, dollar_volume_window=20),
    backtest=BacktestConfig(rebalance_days=5),
    ml=MLConfig(
        momentum_days=20,
        skip_days=5,
        reversal_days=5,
        vol_days=10,
        horizon_days=5,
        purge_days=5,
        min_train_dates=10,
        retrain_every=4,
        rf_trees=20,
        rf_max_depth=3,
        rf_min_leaf=5,
        rf_max_features=1.0,
    ),
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


@functools.lru_cache(maxsize=1)
def _predictions() -> dict[str, pd.DataFrame]:
    returns, volume = make_data()
    return walk_forward_signals(build_panel(returns, volume, CFG), CFG).predictions


def signals() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    returns, volume = make_data()
    return returns, volume, _predictions()


def test_alpha_vector_is_centered() -> None:
    out = alpha_vector(np.array([0.5, -0.5, 0.2, 0.1]))
    assert out.sum() == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(out - out[0], np.array([0.5, -0.5, 0.2, 0.1]) - 0.5)
    with pytest.raises(ValueError):
        alpha_vector(np.array([0.1, np.nan]))


def test_weights_valid_aligned_and_follow_the_scores() -> None:
    returns, _, predictions = signals()
    result = generate_ml_weights(returns, predictions, CFG)
    assert set(result.weights) == set(ML_STRATEGIES)
    for strategy, model in ML_STRATEGIES.items():
        frame = result.weights[strategy]
        pd.testing.assert_index_equal(frame.index, predictions[model].index, check_names=False)
        np.testing.assert_allclose(frame.sum(axis=1), 1.0, atol=1e-8)
        assert frame.to_numpy().min() >= -1e-9
        outside = ~np.isfinite(predictions[model].to_numpy())
        assert (frame.to_numpy()[outside] == 0.0).all()
        correlations = []
        for date in frame.index:
            scores = predictions[model].loc[date].dropna()
            correlations.append(spearmanr(frame.loc[date, scores.index], scores)[0])
        assert np.nanmean(correlations) > 0.0
    assert (result.diagnostics["n_universe"] == 8).all()


def test_weights_respect_the_cap() -> None:
    returns, _, predictions = signals()
    capped = dataclasses.replace(CFG, backtest=BacktestConfig(rebalance_days=5, max_weight=0.25))
    result = generate_ml_weights(returns, predictions, capped, ("MV-ML-RIDGE-RMT",))
    assert result.weights["MV-ML-RIDGE-RMT"].to_numpy().max() <= 0.25 + 1e-8


def test_weights_do_not_use_future_returns() -> None:
    returns, _, predictions = signals()
    base = generate_ml_weights(returns, predictions, CFG, ("MV-ML-RF-RMT",))
    cut = 250
    last_kept = returns.index[cut]
    shocked = returns.copy()
    future = shocked.iloc[cut + 1 :]
    shocked.iloc[cut + 1 :] = future + np.random.default_rng(5).normal(0.0, 0.05, future.shape)
    again = generate_ml_weights(shocked, predictions, CFG, ("MV-ML-RF-RMT",))
    kept = base.weights["MV-ML-RF-RMT"].loc[:last_kept]
    assert len(kept) > 5
    pd.testing.assert_frame_equal(kept, again.weights["MV-ML-RF-RMT"].loc[:last_kept])
    assert not base.weights["MV-ML-RF-RMT"].equals(again.weights["MV-ML-RF-RMT"])


def test_constant_scores_fall_back_to_gmv() -> None:
    returns, _, predictions = signals()
    flat = {"RIDGE": predictions["RIDGE"].where(predictions["RIDGE"].isna(), 0.1)}
    result = generate_ml_weights(returns, flat, CFG, ("MV-ML-RIDGE-RMT",))
    assert (result.diagnostics["solver_MV-ML-RIDGE-RMT"] == "gmv-fallback").all()
    np.testing.assert_allclose(result.weights["MV-ML-RIDGE-RMT"].sum(axis=1), 1.0, atol=1e-8)


def test_inconsistent_inputs_rejected() -> None:
    returns, _, predictions = signals()
    with pytest.raises(ValueError):
        generate_ml_weights(returns, predictions, CFG, ("MV-ML-XGB-RMT",))
    with pytest.raises(ValueError):
        generate_ml_weights(returns, {"RIDGE": predictions["RIDGE"]}, CFG)
    broken = predictions["RF"].copy()
    first = broken.index[0]
    column = broken.loc[first].dropna().index[0]
    broken.loc[first, column] = np.nan
    with pytest.raises(ValueError):
        generate_ml_weights(returns, {"RIDGE": predictions["RIDGE"], "RF": broken}, CFG)
