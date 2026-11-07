"""Tests for performance metrics with known answers."""

import numpy as np
import pandas as pd
import pytest

from pfopt.backtest.engine import simulate
from pfopt.backtest.metrics import (
    METRIC_NAMES,
    annual_turnover,
    drawdown_series,
    max_drawdown,
    metrics_table,
    performance_metrics,
)


def series(values: list[float] | np.ndarray, start: str = "2020-01-01") -> pd.Series:
    values = np.asarray(values, dtype=float)
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)))


def test_constant_return_series() -> None:
    m = performance_metrics(series(np.full(252, 0.001)))
    assert m["ann_return"] == pytest.approx(1.001**252 - 1.0)
    assert m["ann_vol"] == pytest.approx(0.0, abs=1e-12)
    assert np.isnan(m["sharpe"]) and np.isnan(m["sortino"]) and np.isnan(m["calmar"])
    assert m["max_drawdown"] == 0.0
    assert m["hit_rate"] == 1.0
    assert m["worst_day"] == pytest.approx(0.001)
    assert m["cvar_95"] == pytest.approx(-0.001)
    assert m["n_days"] == 252


def test_known_drawdown_and_dates() -> None:
    r = series([0.10, -0.20, 0.05, 0.10, -0.10, 0.30])
    wealth = [1.10, 0.88, 0.924, 1.0164, 0.91476, 1.189188]
    np.testing.assert_allclose((1.0 + r).cumprod(), wealth)
    np.testing.assert_allclose(
        drawdown_series(r), [0.0, 0.88 / 1.1 - 1, 0.924 / 1.1 - 1, 1.0164 / 1.1 - 1, 0.91476 / 1.1 - 1, 0.0]
    )
    info = max_drawdown(r)
    assert info.max_drawdown == pytest.approx(-0.2)
    assert info.peak == r.index[0] and info.trough == r.index[1] and info.recovery == r.index[5]
    m = performance_metrics(r)
    assert m["max_drawdown"] == pytest.approx(-0.2)
    assert m["calmar"] == pytest.approx(m["ann_return"] / 0.2)


def test_drawdown_from_the_initial_wealth_has_no_peak_date_and_no_recovery() -> None:
    info = max_drawdown(series([-0.10, 0.05]))
    assert info.max_drawdown == pytest.approx(-0.10)
    assert pd.isna(info.peak) and pd.isna(info.recovery)
    assert info.trough == pd.bdate_range("2020-01-01", periods=2)[0]


def test_no_drawdown_gives_nat_dates() -> None:
    info = max_drawdown(series([0.01, 0.02, 0.0]))
    assert info.max_drawdown == 0.0 and pd.isna(info.peak) and pd.isna(info.trough)


def test_sharpe_matches_manual_computation() -> None:
    r = series(np.random.default_rng(0).normal(0.0005, 0.01, 500))
    mean, std = r.mean(), r.std(ddof=1)
    assert performance_metrics(r)["sharpe"] == pytest.approx(mean / std * np.sqrt(252))
    with_rf = performance_metrics(r, rf=0.04)["sharpe"]
    assert with_rf == pytest.approx((mean - 0.04 / 252) / std * np.sqrt(252))
    assert performance_metrics(r)["ann_vol"] == pytest.approx(std * np.sqrt(252))


def test_sortino_hand_computed() -> None:
    r = series([0.02, -0.01, 0.03, -0.02])
    downside = np.sqrt((0.01**2 + 0.02**2) / 4)
    assert performance_metrics(r)["sortino"] == pytest.approx(0.005 / downside * np.sqrt(252))
    no_losses = performance_metrics(series([0.01, 0.02, 0.03]))
    assert np.isnan(no_losses["sortino"])


def test_geometric_annualized_return() -> None:
    r = series([0.10, -0.10])
    assert performance_metrics(r, annualization=2)["ann_return"] == pytest.approx(1.1 * 0.9 - 1.0)


def test_cvar_is_the_mean_loss_beyond_the_quantile() -> None:
    r = series(np.linspace(-0.10, 0.09, 20))
    assert performance_metrics(r)["cvar_95"] == pytest.approx(0.10)
    assert performance_metrics(r, cvar_level=0.90)["cvar_95"] == pytest.approx(0.095)


def test_hit_rate_counts_only_positive_days() -> None:
    assert performance_metrics(series([0.01, -0.01, 0.0, 0.02]))["hit_rate"] == 0.5


def test_higher_moments_match_pandas() -> None:
    r = series(np.random.default_rng(1).standard_t(df=5, size=400) * 0.01)
    m = performance_metrics(r)
    assert m["skewness"] == pytest.approx(r.skew())
    assert m["excess_kurtosis"] == pytest.approx(r.kurt())
    assert m["worst_day"] == r.min()


def test_annual_turnover_excludes_the_initial_investment_by_default() -> None:
    turnover = pd.Series([1.0, 0.2, 0.3, 0.1])
    assert annual_turnover(turnover, 504) == pytest.approx(0.6 / 2.0)
    assert annual_turnover(turnover, 504, include_initial=True) == pytest.approx(1.6 / 2.0)
    r = series(np.random.default_rng(2).normal(0, 0.01, 504))
    assert performance_metrics(r, turnover)["annual_turnover"] == pytest.approx(0.3)
    assert np.isnan(performance_metrics(r)["annual_turnover"])


def test_invalid_returns_raise() -> None:
    with pytest.raises(TypeError, match="Series"):
        performance_metrics(np.array([0.01, 0.02]))
    with pytest.raises(ValueError, match="at least 2"):
        performance_metrics(series([0.01]))
    with pytest.raises(ValueError, match="finite"):
        performance_metrics(series([0.01, np.nan, 0.02]))
    with pytest.raises(ValueError, match="greater than -1"):
        performance_metrics(series([0.01, -1.0]))


def test_metrics_table_from_simulations() -> None:
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2020-01-01", periods=61)
    returns = pd.DataFrame(rng.normal(0.0005, 0.01, size=(61, 2)), index=dates, columns=["A", "B"])
    weights = {
        "S": pd.DataFrame(rng.dirichlet(np.ones(2), size=3), index=dates[[0, 20, 40]], columns=["A", "B"]),
        "T": pd.DataFrame([[0.5, 0.5]] * 3, index=dates[[0, 20, 40]], columns=["A", "B"]),
    }
    sims = {name: simulate(frame, returns, cost_bps=50.0) for name, frame in weights.items()}
    net, gross = metrics_table(sims, "net"), metrics_table(sims, "gross")
    assert list(net.index) == ["S", "T"]
    assert set(METRIC_NAMES) <= set(net.columns)
    assert {"mdd_peak", "mdd_trough", "mdd_recovery"} <= set(net.columns)
    assert (net["ann_return"] < gross["ann_return"]).all()
    assert net.loc["S", "n_days"] == len(sims["S"].net)
    expected = annual_turnover(sims["S"].turnover, len(sims["S"].net))
    assert net.loc["S", "annual_turnover"] == pytest.approx(expected)
    with pytest.raises(ValueError, match="which"):
        metrics_table(sims, "after-tax")
