"""Tests for subperiod masks, expanding-median volatility regimes and per-period metrics."""

import numpy as np
import pandas as pd
import pytest

from pfopt.backtest.metrics import performance_metrics
from pfopt.backtest.regimes import (
    expanding_median_regime,
    metrics_by_period,
    period_masks,
    regime_labels_for_returns,
    regime_masks,
)

DAYS = pd.bdate_range("2019-06-03", "2023-06-30")


def test_period_masks_partition_the_sample() -> None:
    masks = period_masks(DAYS, "2020-02-19", "2020-12-31", 2022)
    assert set(masks) == {"full", "covid", "year_2022", "calm"}
    covid, year, calm, full = (masks[k] for k in ("covid", "year_2022", "calm", "full"))
    assert not (covid & year).any() and not (covid & calm).any() and not (year & calm).any()
    np.testing.assert_array_equal(covid | year | calm, full)
    assert full.all()
    assert DAYS[covid][0] == pd.Timestamp("2020-02-19") and DAYS[covid][-1] == pd.Timestamp("2020-12-31")
    assert year.sum() == (DAYS.year == 2022).sum()
    assert not calm[DAYS.get_loc(pd.Timestamp("2020-06-01"))]


def test_overlapping_windows_raise() -> None:
    with pytest.raises(ValueError, match="overlap"):
        period_masks(DAYS, "2022-03-01", "2022-04-01", 2022)


def vol_series(values: list[float], start: int = 0) -> pd.Series:
    return pd.Series(values, index=DAYS[start : start + len(values)], dtype=float)


def test_expanding_median_regime_hand_computed() -> None:
    labels = expanding_median_regime(vol_series([1, 3, 2, 5, 4, 6]), min_history=3)
    assert labels.iloc[:2].isna().all()
    assert list(labels.iloc[2:]) == ["low", "high", "high", "high"]  # medians 2, 2.5, 3, 3.5


def test_expanding_median_ignores_leading_nan_in_the_history_count() -> None:
    labels = expanding_median_regime(vol_series([np.nan, np.nan, 1, 3, 2, 5]), min_history=3)
    assert labels.iloc[:4].isna().all()
    assert list(labels.iloc[4:]) == ["low", "high"]  # medians of [1,3,2] = 2 and [1,3,2,5] = 2.5


def test_expanding_median_uses_only_past_data() -> None:
    base = vol_series([1, 3, 2, 5, 4, 6, 2, 8])
    changed = base.copy()
    changed.iloc[5:] = 100.0
    a = expanding_median_regime(base, 3)
    b = expanding_median_regime(changed, 3)
    pd.testing.assert_series_equal(a.iloc[:5], b.iloc[:5])


def test_labels_for_returns_are_lagged_by_one_trading_day() -> None:
    vol = vol_series([1, 3, 2, 5, 4, 6], start=2)  # DAYS[2..7]; raw labels start at DAYS[4]
    return_index = DAYS[0:10]
    labels = regime_labels_for_returns(vol, return_index, min_history=3)
    assert labels.index.equals(return_index)
    assert labels.iloc[:5].isna().all()  # DAYS[0..4]: the first label is for DAYS[5]
    assert labels.iloc[5] == "low"  # raw label of DAYS[4]
    assert list(labels.iloc[6:9]) == ["high", "high", "high"]
    assert pd.isna(labels.iloc[9])  # raw label of DAYS[8] does not exist


def test_regime_masks_are_disjoint_and_exclude_unclassified_days() -> None:
    labels = pd.Series(["high", "low", np.nan, "high", "low", "low"], dtype=object)
    masks = regime_masks(labels)
    np.testing.assert_array_equal(masks["high_vol"], [True, False, False, True, False, False])
    np.testing.assert_array_equal(masks["low_vol"], [False, True, False, False, True, True])


def test_metrics_by_period_matches_direct_computation_and_partitions_days() -> None:
    rng = np.random.default_rng(0)
    returns = pd.DataFrame(rng.normal(0.0004, 0.01, (len(DAYS), 2)), index=DAYS, columns=["A", "B"])
    masks = period_masks(DAYS, "2020-02-19", "2020-12-31", 2022)
    table = metrics_by_period(returns, masks)
    assert list(table.index.names) == ["period", "strategy"]
    assert len(table) == 4 * 2
    direct = performance_metrics(returns.loc[masks["calm"], "B"], None)
    assert table.loc[("calm", "B"), "sharpe"] == pytest.approx(direct["sharpe"])
    assert table.loc[("covid", "A"), "n_days"] == masks["covid"].sum()
    parts = sum(table.loc[(p, "A"), "n_days"] for p in ("covid", "year_2022", "calm"))
    assert parts == table.loc[("full", "A"), "n_days"]


def test_metrics_by_period_rejects_tiny_periods() -> None:
    returns = pd.DataFrame(np.zeros((5, 1)) + 0.001, index=DAYS[:5], columns=["A"])
    with pytest.raises(ValueError, match="fewer than 2"):
        metrics_by_period(returns, {"one_day": np.array([True, False, False, False, False])})
