"""Tests for point-in-time universe selection on a hand-computable toy panel.

Ten business days d0..d9, five symbols with constant dollar volume
A=100, B=500, C=300, D=200, E=400 (dollar volume is NaN wherever the return is NaN).
NaN returns: B at d6; E at d0..d4 (late listing); C at d9. With T=4 and W=3:
- asof d5: window d2..d5 -> E ineligible; eligible A, B, C, D.
- asof d8: window d5..d8 -> B ineligible; eligible A, C, D, E.
- asof d9: window d6..d9 -> B and C ineligible; eligible A, D, E.
"""

import numpy as np
import pandas as pd
import pytest

from pfopt.data.universe import (
    eligible_counts,
    eligible_symbols,
    first_feasible_date,
    rebalance_dates,
    select_universe,
    snapshot_dates,
)

DATES = pd.bdate_range("2020-01-01", periods=10)
SYMS = ["A", "B", "C", "D", "E"]
ADV = {"A": 100.0, "B": 500.0, "C": 300.0, "D": 200.0, "E": 400.0}
T, W = 4, 3


def make_frames(adv: dict[str, float] = ADV) -> tuple[pd.DataFrame, pd.DataFrame]:
    ret = pd.DataFrame(0.01, index=DATES, columns=SYMS)
    ret.loc[DATES[6], "B"] = np.nan
    ret.loc[DATES[:5], "E"] = np.nan
    ret.loc[DATES[9], "C"] = np.nan
    dv = pd.DataFrame({s: [adv[s]] * len(DATES) for s in SYMS}, index=DATES)
    return ret, dv.where(ret.notna())


def select(ret, dv, day: int, n: int) -> list[str]:
    return select_universe(ret, dv, DATES[day], T, n, W)


def test_eligible_symbols_window_rule() -> None:
    ret, _ = make_frames()
    assert eligible_symbols(ret, DATES[5], T) == ["A", "B", "C", "D"]
    assert eligible_symbols(ret, DATES[8], T) == ["A", "C", "D", "E"]
    assert eligible_symbols(ret, DATES[9], T) == ["A", "D", "E"]


def test_window_includes_asof_date_and_excludes_earlier_nan() -> None:
    ret, _ = make_frames()
    assert "B" in eligible_symbols(ret, DATES[5], T)  # NaN at d6 is in the future
    assert "B" not in eligible_symbols(ret, DATES[6], T)  # NaN on the as-of date itself
    assert "B" not in eligible_symbols(ret, DATES[9], T)  # NaN inside the window
    assert "E" in eligible_symbols(ret, DATES[8], T)  # NaN at d4 has left the window


def test_select_universe_ranks_by_trailing_dollar_volume() -> None:
    ret, dv = make_frames()
    assert select(ret, dv, 5, 2) == ["B", "C"]
    assert select(ret, dv, 8, 3) == ["E", "C", "D"]
    assert select(ret, dv, 9, 2) == ["E", "D"]
    assert select(ret, dv, 9, 3) == ["E", "D", "A"]


def test_size_equals_n_or_clear_error() -> None:
    ret, dv = make_frames()
    for n in (1, 2, 3):
        assert len(select(ret, dv, 9, n)) == n
    with pytest.raises(ValueError, match="need 4"):
        select(ret, dv, 9, 4)


def test_ties_broken_alphabetically() -> None:
    ret, dv = make_frames({s: 100.0 for s in SYMS})
    assert select(ret, dv, 5, 3) == ["A", "B", "C"]
    assert select(ret, dv, 5, 4) == ["A", "B", "C", "D"]


def test_dollar_volume_uses_trailing_window_only() -> None:
    ret, dv = make_frames()
    dv = dv.copy()
    dv.loc[DATES[:3], "A"] = 10_000.0  # outside the W=3 window ending at d5 (d3..d5)
    assert select(ret, dv, 5, 4) == ["B", "C", "D", "A"]
    dv.loc[DATES[5], "A"] = 10_000.0  # inside the window: mean of d3..d5 = (100+100+10000)/3
    assert select(ret, dv, 5, 1) == ["A"]


def test_no_leakage_future_data_does_not_change_selection() -> None:
    ret, dv = make_frames()
    before = {(day, n): select(ret, dv, day, n) for day, n in [(5, 2), (5, 4), (8, 3)]}
    rng = np.random.default_rng(0)
    ret2, dv2 = ret.copy(), dv.copy()
    ret2.iloc[9:] = rng.normal(size=ret2.iloc[9:].shape)
    dv2.iloc[9:] = 1e9
    assert select(ret2, dv2, 8, 3) == before[(8, 3)]
    ret3, dv3 = ret.copy(), dv.copy()
    ret3.iloc[6:] = rng.normal(size=ret3.iloc[6:].shape)
    dv3.iloc[6:] = 1e9
    assert select(ret3, dv3, 5, 2) == before[(5, 2)]
    assert select(ret3, dv3, 5, 4) == before[(5, 4)]


def test_insufficient_history_and_bad_inputs() -> None:
    ret, dv = make_frames()
    with pytest.raises(ValueError, match="fewer than"):
        select(ret, dv, 2, 2)
    with pytest.raises(ValueError, match="fewer than"):
        select_universe(ret, dv, DATES[5], T, 2, 7)
    with pytest.raises(ValueError, match="not a date"):
        select_universe(ret, dv, "2019-12-31", T, 2, W)
    with pytest.raises(ValueError, match="same index and columns"):
        select_universe(ret, dv.drop(columns=["A"]), DATES[5], T, 2, W)
    with pytest.raises(ValueError, match="sorted"):
        select_universe(ret.iloc[::-1], dv.iloc[::-1], DATES[5], T, 2, W)


def test_inputs_not_mutated() -> None:
    ret, dv = make_frames()
    ret0, dv0 = ret.copy(deep=True), dv.copy(deep=True)
    select(ret, dv, 9, 2)
    pd.testing.assert_frame_equal(ret, ret0)
    pd.testing.assert_frame_equal(dv, dv0)


def test_all_nan_dollar_volume_symbol_is_excluded() -> None:
    ret, dv = make_frames()
    dv = dv.copy()
    dv["B"] = np.nan
    assert select(ret, dv, 5, 3) == ["C", "D", "A"]
    with pytest.raises(ValueError, match="need 4"):
        select(ret, dv, 5, 4)


def test_eligible_counts_match_pointwise_rule() -> None:
    ret, _ = make_frames()
    counts = eligible_counts(ret, T)
    assert list(counts.index) == list(DATES[T - 1 :])
    assert list(counts) == [4, 4, 4, 3, 3, 4, 3]
    for date, n in counts.items():
        assert n == len(eligible_symbols(ret, date, T))


def test_first_feasible_date() -> None:
    ret, _ = make_frames()
    assert first_feasible_date(ret, T, 4) == DATES[3]
    assert first_feasible_date(ret, T, 3) == DATES[3]
    with pytest.raises(ValueError, match="never 5"):
        first_feasible_date(ret, T, 5)


def test_snapshot_dates_first_stress_last() -> None:
    ret, _ = make_frames()
    snaps = snapshot_dates(ret, T, 4, "2020-01-08")  # DATES[5]
    assert snaps == {"first": DATES[3], "stress": DATES[5], "last": DATES[9]}
    assert list(snaps) == ["first", "stress", "last"]
    with pytest.raises(ValueError, match="stress_date"):
        snapshot_dates(ret, T, 4, "2020-01-03")  # DATES[2]: before the first feasible date
    with pytest.raises(ValueError, match="stress_date"):
        snapshot_dates(ret, T, 4, "2019-06-03")  # not in the index


def test_rebalance_dates_start_at_first_feasible_and_step() -> None:
    ret, _ = make_frames()
    assert list(rebalance_dates(ret, T, 4, 3)) == [DATES[3], DATES[6], DATES[9]]
    assert list(rebalance_dates(ret, T, 4, 1)) == list(DATES[3:])
    with pytest.raises(ValueError, match="rebalance_days"):
        rebalance_dates(ret, T, 4, 0)
