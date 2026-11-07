"""Tests for the audit functions on a small hand-countable panel.

Panel (6 business days d0..d5, three symbols, plus one duplicated row):
- AAA: valid on all days; duplicate of its d0 row; Close=0 on d1; Volume=0 on d2;
  High<Low (and Close>High) on d3; Adj Close above Close on d4.
- BBB: all-NaN on d0, d1, d3, d4; valid on d2 and d5 (late listing, internal gap of length 2).
- CCC: valid d0..d3, all-NaN on d4, d5 (early end).
"""

import numpy as np
import pandas as pd
import pytest

from pfopt.data.audit import (
    _longest_na_run,
    audit_companies,
    audit_index,
    audit_stocks,
    run_audit,
)

DATES = pd.bdate_range("2020-01-01", periods=6)
COLS = ["Date", "Symbol", "Adj Close", "Close", "High", "Low", "Open", "Volume"]


def _row(day: int, symbol: str, valid: bool = True, **overrides: float) -> dict:
    if not valid:
        return {
            "Date": DATES[day],
            "Symbol": symbol,
            **{c: np.nan for c in COLS[2:]},
        }
    base = {
        "Date": DATES[day],
        "Symbol": symbol,
        "Adj Close": 10.0,
        "Close": 10.0,
        "High": 11.0,
        "Low": 9.0,
        "Open": 10.0,
        "Volume": 1000.0,
    }
    base.update(overrides)
    return base


@pytest.fixture
def stocks() -> pd.DataFrame:
    rows = [
        _row(0, "AAA"),
        _row(1, "AAA", Close=0.0),
        _row(2, "AAA", Volume=0.0),
        _row(3, "AAA", High=8.0, Low=9.0),
        _row(4, "AAA", **{"Adj Close": 10.5}),
        _row(5, "AAA"),
        _row(0, "AAA"),  # duplicate (Date, Symbol) and full row
        _row(0, "BBB", valid=False),
        _row(1, "BBB", valid=False),
        _row(2, "BBB"),
        _row(3, "BBB", valid=False),
        _row(4, "BBB", valid=False),
        _row(5, "BBB"),
        *[_row(d, "CCC") for d in range(4)],
        _row(4, "CCC", valid=False),
        _row(5, "CCC", valid=False),
    ]
    return pd.DataFrame(rows, columns=COLS).reset_index(drop=True)


def get(table: pd.DataFrame, section: str, metric: str, detail: str = "") -> object:
    sel = table[
        (table["section"] == section) & (table["metric"] == metric) & (table["detail"] == detail)
    ]
    assert len(sel) == 1, f"expected one row for {section}/{metric}/{detail!r}, got {len(sel)}"
    return sel.iloc[0]["value"]


def test_longest_na_run() -> None:
    assert _longest_na_run(np.array([], dtype=bool)) == 0
    assert _longest_na_run(np.array([False, True, True, False, True])) == 2
    assert _longest_na_run(np.array([True, True, True])) == 3
    assert _longest_na_run(np.array([False, False])) == 0


def test_audit_stocks_overview_and_missing(stocks: pd.DataFrame) -> None:
    t = audit_stocks(stocks, ffill_limit=1, top_k=1)
    assert list(t.columns) == ["section", "metric", "detail", "value"]
    assert get(t, "overview", "n_rows") == 19
    assert get(t, "overview", "n_symbols") == 3
    assert get(t, "overview", "n_unique_dates") == 6
    assert get(t, "overview", "date_min_all_rows") == "2020-01-01"
    assert get(t, "overview", "date_max_all_rows") == "2020-01-08"
    assert get(t, "overview", "first_valid_adj_close_date") == "2020-01-01"
    assert get(t, "missing", "n_rows_all_prices_nan") == 6
    assert get(t, "missing", "pct_rows_all_prices_nan") == pytest.approx(6 / 19 * 100)
    assert get(t, "missing", "pct_missing", "Adj Close") == pytest.approx(6 / 19 * 100)
    assert get(t, "missing", "n_rows_volume_without_price") == 0


def test_audit_stocks_anomalies(stocks: pd.DataFrame) -> None:
    t = audit_stocks(stocks, ffill_limit=1, top_k=1)
    assert get(t, "anomalies", "n_duplicate_date_symbol_rows") == 1
    assert get(t, "anomalies", "n_duplicate_full_rows") == 1
    assert get(t, "anomalies", "n_nonpositive", "Close") == 1
    assert get(t, "anomalies", "n_nonpositive", "Adj Close") == 0
    assert get(t, "anomalies", "n_zero_volume") == 1
    assert get(t, "anomalies", "n_zero_volume_with_price") == 1
    assert get(t, "anomalies", "n_negative_volume") == 0
    assert get(t, "anomalies", "n_high_lt_low") == 1
    assert get(t, "anomalies", "n_close_outside_high_low") == 2


def test_audit_stocks_symbol_history(stocks: pd.DataFrame) -> None:
    t = audit_stocks(stocks, ffill_limit=1, top_k=1)
    assert get(t, "symbol_history", "rows_per_symbol_min") == 6
    assert get(t, "symbol_history", "rows_per_symbol_max") == 7
    assert get(t, "symbol_history", "valid_adj_close_rows_per_symbol_min") == 2
    assert get(t, "symbol_history", "valid_adj_close_rows_per_symbol_max") == 7
    assert get(t, "symbol_history", "n_symbols_without_valid_price") == 0
    assert get(t, "symbol_history", "n_late_listing", "first valid > global first") == 1
    assert get(t, "symbol_history", "n_early_end", "last valid < global last") == 1
    shortest = t[t["metric"] == "shortest_history_valid_rows"]
    assert len(shortest) == 1
    assert shortest.iloc[0]["detail"] == "BBB first=2020-01-03 last=2020-01-08"
    assert shortest.iloc[0]["value"] == 2
    assert get(t, "symbol_history", "n_symbols_internal_gap") == 1
    assert get(t, "symbol_history", "n_internal_gap_gt_ffill_limit", "limit=1") == 1
    assert get(t, "symbol_history", "max_internal_gap_run") == 2
    assert get(t, "symbol_history", "longest_internal_gap_run", "BBB") == 2


def test_internal_gap_limit_respected(stocks: pd.DataFrame) -> None:
    t = audit_stocks(stocks, ffill_limit=2, top_k=1)
    assert get(t, "symbol_history", "n_internal_gap_gt_ffill_limit", "limit=2") == 0


def test_audit_stocks_coverage_and_adjustment(stocks: pd.DataFrame) -> None:
    t = audit_stocks(stocks, ffill_limit=1, top_k=1)
    assert get(t, "coverage", "valid_symbols_per_date_min") == 1
    assert get(t, "coverage", "valid_symbols_per_date_max") == 3
    assert get(t, "coverage", "n_dates_with_no_valid_price") == 0
    assert get(t, "coverage", "n_dates_coverage_below_50pct") == 1
    assert get(t, "coverage", "n_dates_coverage_below_10pct") == 0
    assert get(t, "adjustment", "n_adj_close_gt_close") == 1
    assert get(t, "adjustment", "adj_close_over_close_max") == pytest.approx(1.05)
    assert get(t, "adjustment", "adj_close_over_close_max_at") == "AAA 2020-01-07"


def test_audit_does_not_mutate_input(stocks: pd.DataFrame) -> None:
    before = stocks.copy(deep=True)
    audit_stocks(stocks, ffill_limit=1)
    pd.testing.assert_frame_equal(stocks, before)


def test_symbol_without_any_price() -> None:
    rows = [_row(d, "AAA") for d in range(3)] + [_row(d, "ZZZ", valid=False) for d in range(3)]
    t = audit_stocks(pd.DataFrame(rows, columns=COLS), ffill_limit=3, top_k=1)
    assert get(t, "symbol_history", "n_symbols_without_valid_price") == 1
    assert get(t, "symbol_history", "symbols_without_valid_price") == "ZZZ"


def test_audit_companies() -> None:
    companies = pd.DataFrame(
        {
            "Symbol": pd.array(["AAA", "BBB", "DDD"], dtype="string"),
            "Sector": pd.array(["Tech", None, "Tech"], dtype="string"),
        }
    )
    t = audit_companies(companies, {"AAA", "BBB", "CCC"})
    assert get(t, "companies", "n_rows") == 3
    assert get(t, "companies", "n_duplicate_symbols") == 0
    assert get(t, "companies", "n_missing_sector") == 1
    assert get(t, "companies", "n_symbols_only_in_stocks") == 1
    assert get(t, "companies", "symbols_only_in_stocks") == "CCC"
    assert get(t, "companies", "n_symbols_only_in_companies") == 1
    assert get(t, "companies", "symbols_only_in_companies") == "DDD"
    assert get(t, "companies", "n_per_sector", "Tech") == 2


def test_audit_index() -> None:
    days = ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07", "2020-01-09"]
    index = pd.DataFrame(
        {
            "Date": pd.to_datetime(days),
            "S&P500": [100.0, 101.0, 102.0, 103.0, np.nan, 105.0],
        }
    )
    t = audit_index(index, pd.DatetimeIndex(DATES), top_k=1)
    assert get(t, "index", "n_rows") == 6
    assert get(t, "index", "date_min") == "2020-01-01"
    assert get(t, "index", "date_max") == "2020-01-09"
    assert get(t, "index", "n_duplicate_dates") == 0
    assert get(t, "index", "is_sorted_ascending")
    assert get(t, "index", "n_missing_values", "S&P500") == 1
    assert get(t, "index", "n_nonpositive_values", "S&P500") == 0
    assert get(t, "index", "max_gap_days") == 3
    assert get(t, "index", "n_gaps_gt_threshold", "days>4") == 0
    assert get(t, "index", "n_dates_in_index_not_in_stocks") == 1
    assert get(t, "index", "n_stock_dates_in_index_range_not_in_index") == 1
    assert get(t, "index", "largest_gap_days", "2020-01-03->2020-01-06") == 3


def test_audit_index_requires_single_level_column() -> None:
    index = pd.DataFrame({"Date": DATES[:2], "a": [1.0, 2.0], "b": [1.0, 2.0]})
    with pytest.raises(ValueError, match="exactly one level column"):
        audit_index(index, pd.DatetimeIndex(DATES))


def test_run_audit_combines_sections(stocks: pd.DataFrame) -> None:
    companies = pd.DataFrame(
        {
            "Symbol": pd.array(["AAA", "BBB", "CCC"], dtype="string"),
            "Sector": pd.array(["Tech", "Tech", "Energy"], dtype="string"),
        }
    )
    index = pd.DataFrame({"Date": DATES, "S&P500": np.linspace(100, 105, 6)})
    t = run_audit(stocks, companies, index, ffill_limit=3, top_k=2)
    assert {"overview", "missing", "anomalies", "companies", "index"} <= set(t["section"])
    assert get(t, "index", "n_dates_in_index_not_in_stocks") == 0
