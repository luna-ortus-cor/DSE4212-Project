"""Tests for the covariance forecast loss on random portfolios."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from pfopt.backtest.forecastloss import (
    EQUAL_WEIGHT,
    ESTIMATORS,
    forecast_loss_dates,
    forecast_variance,
    newey_west_mean,
    qlike,
    random_portfolios,
    realized_variance,
    summarize_loss,
)
from pfopt.config import BacktestConfig, Config, ForecastConfig, UniverseConfig

CFG = dataclasses.replace(
    Config(),
    universe=UniverseConfig(n_assets=8, lookback_days=40, dollar_volume_window=20),
    backtest=BacktestConfig(rebalance_days=5),
    forecast=ForecastConfig(n_portfolios=20, dirichlet_alphas=(1.0, 0.1), hac_lag=2),
)


def test_qlike_is_zero_at_the_truth_and_positive_otherwise() -> None:
    f = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(qlike(f, f), 0.0, atol=1e-15)
    assert (qlike(2.0 * f, f) > 0).all() and (qlike(0.5 * f, f) > 0).all()
    assert qlike(np.array([2.0]), np.array([1.0]))[0] == pytest.approx(1.0 - np.log(2.0))


def test_random_portfolios_are_valid_and_seeded() -> None:
    a = random_portfolios(10, 30, 0.5, np.random.default_rng(0))
    b = random_portfolios(10, 30, 0.5, np.random.default_rng(0))
    np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(a.sum(axis=1), 1.0)
    assert (a >= 0).all()
    spread = random_portfolios(50, 200, 0.1, np.random.default_rng(1)).max(axis=1).mean()
    assert spread > random_portfolios(50, 200, 5.0, np.random.default_rng(1)).max(axis=1).mean()
    with pytest.raises(ValueError):
        random_portfolios(5, 3, 0.0, np.random.default_rng(0))


def test_forecast_and_realized_variance_known_values() -> None:
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    w = np.array([[0.5, 0.5], [1.0, 0.0]])
    np.testing.assert_allclose(forecast_variance(w, cov), [0.0375, 0.04])
    future = np.array([[0.1, 0.1], [-0.1, 0.1]])
    np.testing.assert_allclose(realized_variance(w, future), [0.005, 0.01])
    with pytest.raises(ValueError):
        realized_variance(w, np.zeros((0, 2)))


def test_newey_west_matches_the_iid_standard_error_without_lags() -> None:
    x = np.random.default_rng(0).standard_normal(400)
    mean, se = newey_west_mean(x, 0)
    assert mean == pytest.approx(x.mean()) and se == pytest.approx(x.std(ddof=0) / np.sqrt(400))
    _, se_lag = newey_west_mean(x, 3)
    assert 0.7 * se < se_lag < 1.3 * se
    with pytest.raises(ValueError):
        newey_west_mean(np.array([1.0]), 0)


def test_summary_detects_a_planted_loss_difference_and_adjusts_p_values() -> None:
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2020-01-01", periods=60)
    rows = []
    for name, shift in (("Sample", 0.0), ("A", -0.5), ("B", 0.0)):
        for d in dates:
            q = 1.0 + shift + 0.1 * rng.standard_normal()
            rows.append({"date": d, "family": "F", "estimator": name, "qlike": q, "log_bias": -0.1})
    out = summarize_loss(pd.DataFrame(rows), lag=2).set_index("estimator")
    assert out.loc["A", "diff_qlike"] == pytest.approx(-0.5, abs=0.1)
    assert out.loc["A", "p_holm"] < 0.01 and out.loc["B", "p_holm"] > 0.05
    assert out.loc["Sample", "diff_qlike"] == 0.0 and np.isnan(out.loc["Sample", "p_holm"])


def make_panel(n_days: int = 130) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    symbols = [f"S{i:02d}" for i in range(14)]
    group = rng.normal(0.0, 0.006, size=(n_days, 3))
    returns = group[:, np.arange(14) % 3] + rng.normal(0.0, 0.01, (n_days, 14))
    volume = rng.uniform(1.0, 3.0, size=(n_days, 14))
    labels = ["Tech", "Energy", "Health"] * 4 + ["Unknown", np.nan]
    log_returns = pd.DataFrame(returns, index=dates, columns=symbols)
    return (
        log_returns,
        np.expm1(log_returns),
        pd.DataFrame(volume, index=dates, columns=symbols),
        pd.Series(labels, index=symbols),
    )


def test_driver_shape_losses_and_reproducibility() -> None:
    log_returns, simple, volume, sectors = make_panel()
    out = forecast_loss_dates(log_returns, simple, volume, sectors, CFG)
    assert set(out["estimator"]) == set(ESTIMATORS)
    assert set(out["family"]) == {EQUAL_WEIGHT, "Dirichlet(1)", "Dirichlet(0.1)"}
    assert (out["qlike"] >= 0).all() and np.isfinite(out["log_bias"]).all()
    counts = out.groupby(["date", "family"]).size()
    assert (counts == len(ESTIMATORS)).all()
    pd.testing.assert_frame_equal(
        out, forecast_loss_dates(log_returns, simple, volume, sectors, CFG)
    )


def test_forecasts_use_the_past_and_results_ignore_data_beyond_the_holding_period() -> None:
    log_returns, simple, volume, sectors = make_panel()
    out = forecast_loss_dates(log_returns, simple, volume, sectors, CFG)
    last = out["date"].max()
    pos = simple.index.get_loc(last) + CFG.backtest.rebalance_days
    changed_log, changed_simple = log_returns.copy(), simple.copy()
    noise = 5.0 * np.random.default_rng(0).standard_normal(changed_log.iloc[pos + 1 :].shape)
    changed_log.iloc[pos + 1 :] = noise
    changed_simple.iloc[pos + 1 :] = np.expm1(noise)
    again = forecast_loss_dates(changed_log, changed_simple, volume, sectors, CFG)
    pd.testing.assert_frame_equal(out[out["date"] <= last], again[again["date"] <= last])
