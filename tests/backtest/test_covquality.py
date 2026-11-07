"""Tests for concentration, weight-stability and risk-forecast-quality metrics."""

import numpy as np
import pandas as pd
import pytest

from pfopt.backtest.covquality import (
    forecast_quality_table,
    risk_forecast_quality,
    stability_table,
    weight_stability,
    weight_stats,
)
from pfopt.backtest.engine import SimulationResult

ROOT252 = np.sqrt(252)


def fake_simulation(gross: pd.Series, rebalance_dates: pd.DatetimeIndex) -> SimulationResult:
    empty = pd.DataFrame()
    return SimulationResult(
        gross=gross,
        net=gross.copy(),
        turnover=pd.Series(1.0, index=rebalance_dates),
        weights_pre=empty,
        weights_post=empty,
        daily_weights=empty,
        n_missing_returns=0,
    )


# ---------------------------------------------------------------- concentration statistics


def test_weight_stats_hand_computed() -> None:
    stats = weight_stats(np.array([0.5, 0.3, 0.2]))
    assert stats["n_positions"] == 3
    assert stats["effective_n"] == pytest.approx(1.0 / (0.25 + 0.09 + 0.04))
    assert stats["max_weight"] == 0.5
    assert stats["top_k_share"] == pytest.approx(1.0)


def test_equal_weights_have_effective_n_equal_to_n() -> None:
    stats = weight_stats(np.full(4, 0.25))
    assert stats["effective_n"] == pytest.approx(4.0)
    assert stats["max_weight"] == 0.25


def test_threshold_and_top_k_are_respected() -> None:
    w = np.array([0.9995, 0.0005])
    stats = weight_stats(w, position_threshold=1e-3, top_k=1)
    assert stats["n_positions"] == 1
    assert stats["top_k_share"] == pytest.approx(0.9995)
    assert weight_stats(w, position_threshold=1e-4)["n_positions"] == 2


def test_weight_stats_input_not_mutated_and_order_irrelevant() -> None:
    w = np.array([0.1, 0.6, 0.3])
    before = w.copy()
    assert weight_stats(w) == weight_stats(w[::-1])
    np.testing.assert_array_equal(w, before)


# ---------------------------------------------------------------- weight stability


def test_weight_stability_hand_computed() -> None:
    dates = pd.bdate_range("2020-01-01", periods=3)
    weights = pd.DataFrame(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.0]], index=dates, columns=list("ABC")
    )
    turnover = pd.Series([1.0, 1.8, 0.7], index=dates)
    stats = weight_stability(weights, turnover)
    assert stats["mean_l1_target_change"] == pytest.approx((2.0 + 1.0) / 2)
    assert stats["mean_turnover"] == pytest.approx((1.8 + 0.7) / 2)  # initial investment excluded
    assert stats["mean_effective_n"] == pytest.approx((1.0 + 1.0 + 2.0) / 3)
    assert stats["median_effective_n"] == pytest.approx(1.0)
    assert stats["mean_max_weight"] == pytest.approx((1.0 + 1.0 + 0.5) / 3)
    assert stats["mean_n_positions"] == pytest.approx((1 + 1 + 2) / 3)
    assert stats["n_rebalances"] == 3
    assert np.isnan(weight_stability(weights)["mean_turnover"])
    assert np.isnan(weight_stability(weights.iloc[:1])["mean_l1_target_change"])


def test_stability_table_collects_strategies() -> None:
    dates = pd.bdate_range("2020-01-01", periods=3)
    a = pd.DataFrame([[1.0, 0.0]] * 3, index=dates, columns=["X", "Y"])
    b = pd.DataFrame([[0.5, 0.5], [0.4, 0.6], [0.5, 0.5]], index=dates, columns=["X", "Y"])
    sims = {"A": fake_simulation(pd.Series([0.0], index=dates[1:2]), dates),
            "B": fake_simulation(pd.Series([0.0], index=dates[1:2]), dates)}
    table = stability_table({"A": a, "B": b}, sims)
    assert list(table.index) == ["A", "B"]
    assert table.loc["A", "mean_l1_target_change"] == 0.0
    assert table.loc["B", "mean_l1_target_change"] == pytest.approx(0.2)
    assert table.loc["B", "mean_effective_n"] == pytest.approx((2.0 + 1 / 0.52 + 2.0) / 3)
    assert "mean_turnover" in stability_table({"A": a}).columns


# ---------------------------------------------------------------- risk-forecast quality


def hand_example() -> tuple[pd.DataFrame, SimulationResult, np.ndarray]:
    days = pd.bdate_range("2020-01-01", periods=10)  # d0..d9
    rebalance = days[[0, 3, 6]]
    gross_values = np.array([0.01, -0.01, 0.01, 0.02, -0.02, 0.0, 0.01, 0.03, -0.01])
    gross = pd.Series(gross_values, index=days[1:])  # d1..d9: periods (d1-d3), (d4-d6), (d7-d9)
    diagnostics = pd.DataFrame({"pred_vol_X": [0.20, 0.25, 0.30]}, index=rebalance)
    return diagnostics, fake_simulation(gross, rebalance), gross_values


def test_forecast_quality_hand_computed_periods() -> None:
    diagnostics, sim, values = hand_example()
    quality = risk_forecast_quality(diagnostics, sim, "X", min_days=3)
    periods = quality.periods
    assert list(periods["n_days"]) == [3, 3, 3]
    # Period 0: returns (1%, -1%, 1%) have sample std 0.011547, i.e. 18.33% annualized.
    assert periods["realized_vol"].iloc[0] == pytest.approx(0.011547005 * ROOT252, abs=1e-6)
    assert periods["realized_vol"].iloc[0] == pytest.approx(0.18330, abs=1e-4)
    assert periods["ratio"].iloc[0] == pytest.approx(0.18330 / 0.20, abs=1e-4)
    # Period 1: (2%, -2%, 0%) has std 2%, i.e. 31.75% annualized.
    assert periods["realized_vol"].iloc[1] == pytest.approx(0.02 * ROOT252)
    np.testing.assert_allclose(periods["predicted_vol"], [0.20, 0.25, 0.30])
    np.testing.assert_allclose(np.log(periods["ratio"]), periods["log_ratio"])


def test_forecast_quality_hand_computed_aggregates() -> None:
    diagnostics, sim, values = hand_example()
    summary = risk_forecast_quality(diagnostics, sim, "X", min_days=3).summary
    blocks = [values[0:3], values[3:6], values[6:9]]
    predicted = np.array([0.20, 0.25, 0.30])
    ratios = np.array([b.std(ddof=1) * ROOT252 / p for b, p in zip(blocks, predicted, strict=True)])
    assert summary["n_periods"] == 3
    assert summary["mean_ratio"] == pytest.approx(ratios.mean())
    assert summary["median_ratio"] == pytest.approx(np.median(ratios))
    assert summary["mean_log_ratio"] == pytest.approx(np.log(ratios).mean())
    mean_predicted_variance = np.mean(predicted**2)  # equal block lengths
    assert summary["pooled_variance_ratio"] == pytest.approx(
        values.var(ddof=1) * 252 / mean_predicted_variance
    )
    z = np.concatenate([b / (p / ROOT252) for b, p in zip(blocks, predicted, strict=True)])
    assert summary["bias_statistic"] == pytest.approx(z.std(ddof=1))
    assert summary["mean_predicted_vol"] == pytest.approx(np.sqrt(mean_predicted_variance))
    assert summary["realized_vol_pooled"] == pytest.approx(values.std(ddof=1) * ROOT252)


def calibrated_example(scale: float, seed: int = 0) -> tuple[pd.DataFrame, SimulationResult]:
    """20 periods of 500 days (plus a 7-day partial period); forecast = scale * true vol."""
    true_daily = 0.01
    days = pd.bdate_range("2000-01-03", periods=10_008)
    rebalance = days[0:10_001:500]
    gross = pd.Series(np.random.default_rng(seed).normal(0.0, true_daily, 10_007), index=days[1:])
    diagnostics = pd.DataFrame(
        {"pred_vol_X": np.full(len(rebalance), scale * true_daily * ROOT252)}, index=rebalance
    )
    return diagnostics, fake_simulation(gross, rebalance)


def test_calibrated_forecast_has_ratios_and_bias_near_one() -> None:
    diagnostics, sim = calibrated_example(1.0)
    quality = risk_forecast_quality(diagnostics, sim, "X")
    summary = quality.summary
    assert summary["n_periods"] == 20  # the 7-day partial period is skipped (min_days = 10)
    assert summary["mean_ratio"] == pytest.approx(1.0, abs=0.02)
    assert summary["pooled_variance_ratio"] == pytest.approx(1.0, abs=0.05)
    assert summary["bias_statistic"] == pytest.approx(1.0, abs=0.03)
    assert abs(summary["mean_log_ratio"]) < 0.02
    assert len(quality.rolling_bias) == 20 - 12 + 1
    assert quality.rolling_bias.index.equals(quality.periods.index[11:])
    assert np.allclose(quality.rolling_bias, 1.0, atol=0.05)


def test_underpredicted_risk_gives_ratios_above_one() -> None:
    diagnostics, sim = calibrated_example(0.5)
    summary = risk_forecast_quality(diagnostics, sim, "X").summary
    assert summary["mean_ratio"] == pytest.approx(2.0, abs=0.06)
    assert summary["pooled_variance_ratio"] == pytest.approx(4.0, abs=0.2)
    assert summary["bias_statistic"] == pytest.approx(2.0, abs=0.06)
    assert summary["mean_log_ratio"] == pytest.approx(np.log(2.0), abs=0.02)


def test_short_periods_are_skipped_and_empty_selection_raises() -> None:
    diagnostics, sim, _ = hand_example()
    with pytest.raises(ValueError, match="enough observations"):
        risk_forecast_quality(diagnostics, sim, "X")  # default min_days = 10 > 3
    quality = risk_forecast_quality(diagnostics, sim, "X", min_days=3, bias_window=2)
    assert len(quality.rolling_bias) == 2


def test_forecast_quality_table_defaults_to_strategies_with_forecasts() -> None:
    diagnostics, sim = calibrated_example(1.0)
    diagnostics["pred_vol_Y"] = diagnostics["pred_vol_X"] * 0.5
    diagnostics["unrelated"] = 1.0
    sims = {"X": sim, "Y": sim}
    table = forecast_quality_table(diagnostics, sims)
    assert list(table.index) == ["X", "Y"]
    assert table.loc["X", "mean_ratio"] == pytest.approx(1.0, abs=0.02)
    assert table.loc["Y", "mean_ratio"] == pytest.approx(2.0, abs=0.06)
    assert list(forecast_quality_table(diagnostics, sims, strategies=["Y"]).index) == ["Y"]


def test_forecast_quality_input_validation() -> None:
    diagnostics, sim, _ = hand_example()
    with pytest.raises(ValueError, match="pred_vol_Z"):
        risk_forecast_quality(diagnostics, sim, "Z", min_days=3)
    shifted = diagnostics.copy()
    shifted.index = shifted.index + pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="same rebalance dates"):
        risk_forecast_quality(shifted, sim, "X", min_days=3)
