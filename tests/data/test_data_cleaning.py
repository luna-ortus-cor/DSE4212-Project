"""Tests for cleaning and return construction on a hand-computable toy panel.

Eight business days d0..d7 (2020-01-01 ... 2020-01-10) and seven symbols:
- AAA: steady prices.
- BBB: not listed on d0, d1 (late listing).
- CCC: 2-day gap (d2, d3): within the forward-fill limit of 3.
- DDD: 4-day gap (d2..d5): 3 days filled, 1 left missing -> 12.5% of span, dropped at 5%.
- EEE: never has a price.
- FFF: no price on d6, d7 (after its last price): must not be filled.
- GGG: price doubles on d3 (suspect return).
Close = 2 * Adj Close and Volume = 100 whenever a price exists.
"""

import numpy as np
import pandas as pd
import pytest

from pfopt.data.cleaning import PROCESSED_FILES, clean_prices, write_processed

NAN = np.nan
DATES = pd.bdate_range("2020-01-01", periods=8)
PRICES = {
    "AAA": [10, 11, 12, 13, 14, 15, 16, 17],
    "BBB": [NAN, NAN, 20, 21, 22, 23, 24, 25],
    "CCC": [30, 31, NAN, NAN, 34, 35, 36, 37],
    "DDD": [40, 41, NAN, NAN, NAN, NAN, 46, 47],
    "EEE": [NAN] * 8,
    "FFF": [50, 51, 52, 53, 54, 55, NAN, NAN],
    "GGG": [60, 61, 62, 124, 125, 126, 127, 128],
}
KWARGS = {
    "start_date": None,
    "end_date": None,
    "ffill_limit": 3,
    "max_missing_frac": 0.05,
    "suspect_abs_logret": 0.40,
}


def make_stocks(prices: dict[str, list[float]], dates: pd.DatetimeIndex = DATES) -> pd.DataFrame:
    rows = []
    for symbol, series in prices.items():
        for date, price in zip(dates, series, strict=True):
            valid = not np.isnan(price)
            rows.append(
                {
                    "Date": date,
                    "Symbol": symbol,
                    "Adj Close": price,
                    "Close": 2.0 * price,
                    "Volume": 100.0 if valid else NAN,
                }
            )
    return pd.DataFrame(rows)


def metric(result_log: pd.DataFrame, name: str) -> object:
    sel = result_log[result_log["metric"] == name]
    assert len(sel) == 1, name
    return sel.iloc[0]["value"]


@pytest.fixture
def result():
    return clean_prices(make_stocks(PRICES), **KWARGS)


def test_symbols_kept_and_dropped(result) -> None:
    assert list(result.log_returns.columns) == ["AAA", "BBB", "CCC", "FFF", "GGG"]
    reasons = dict(zip(result.dropped["Symbol"], result.dropped["reason"], strict=True))
    assert reasons == {"DDD": "missing_gt_threshold", "EEE": "no_price"}
    ddd = result.dropped.set_index("Symbol").loc["DDD"]
    assert ddd["missing_frac"] == pytest.approx(1 / 8)
    assert ddd["first_valid"] == "2020-01-01"
    assert metric(result.log, "n_symbols_dropped_no_price") == 1
    assert metric(result.log, "n_symbols_dropped_missing_gt_threshold") == 1
    assert metric(result.log, "n_symbols_kept") == 5


def test_late_listing_is_not_missing() -> None:
    kwargs = {**KWARGS, "max_missing_frac": 0.0}
    res = clean_prices(make_stocks({"AAA": PRICES["AAA"], "BBB": PRICES["BBB"]}), **kwargs)
    assert "BBB" in res.log_returns.columns
    assert np.isnan(res.log_returns.loc[DATES[1:3], "BBB"]).all()
    assert res.log_returns.loc[DATES[3], "BBB"] == pytest.approx(np.log(21 / 20))


def test_forward_fill_within_limit_gives_zero_return(result) -> None:
    lr = result.log_returns["CCC"]
    assert lr.loc[DATES[2]] == 0.0
    assert lr.loc[DATES[3]] == 0.0
    assert lr.loc[DATES[4]] == pytest.approx(np.log(34 / 31))
    assert metric(result.log, "n_prices_forward_filled") == 2
    assert metric(result.log, "pct_prices_forward_filled") == pytest.approx(100 * 2 / 36)


def test_gap_longer_than_limit_stays_missing_when_symbol_kept() -> None:
    kwargs = {**KWARGS, "max_missing_frac": 0.2}
    res = clean_prices(make_stocks({"AAA": PRICES["AAA"], "DDD": PRICES["DDD"]}), **kwargs)
    lr = res.log_returns["DDD"]
    assert lr.loc[DATES[2]] == 0.0 and lr.loc[DATES[3]] == 0.0 and lr.loc[DATES[4]] == 0.0
    assert np.isnan(lr.loc[DATES[5]])
    assert np.isnan(lr.loc[DATES[6]])
    assert lr.loc[DATES[7]] == pytest.approx(np.log(47 / 46))


def test_no_fill_after_last_price(result) -> None:
    lr = result.log_returns["FFF"]
    assert np.isnan(lr.loc[DATES[6]]) and np.isnan(lr.loc[DATES[7]])
    assert lr.loc[DATES[5]] == pytest.approx(np.log(55 / 54))


def test_return_identities_and_index(result) -> None:
    lr, sr = result.log_returns, result.simple_returns
    np.testing.assert_allclose(np.expm1(lr.to_numpy()), sr.to_numpy(), atol=1e-12, equal_nan=True)
    assert not np.isinf(lr.to_numpy()).any()
    assert lr.index.is_monotonic_increasing
    assert list(lr.index) == list(DATES[1:])
    assert lr.loc[DATES[1], "AAA"] == pytest.approx(np.log(11 / 10))
    assert sr.loc[DATES[1], "AAA"] == pytest.approx(0.1)
    for frame in (result.volume, result.dollar_volume):
        assert frame.shape == lr.shape
        assert frame.index.equals(lr.index)
        assert list(frame.columns) == list(lr.columns)


def test_dollar_volume_is_volume_times_close_and_unfilled(result) -> None:
    assert result.volume.loc[DATES[1], "AAA"] == 100.0
    assert result.dollar_volume.loc[DATES[1], "AAA"] == 100.0 * 22.0
    assert np.isnan(result.dollar_volume.loc[DATES[2], "CCC"])
    assert np.isnan(result.volume.loc[DATES[2], "CCC"])


def test_suspect_returns_flagged(result) -> None:
    sus = result.suspect
    assert len(sus) == 1
    row = sus.iloc[0]
    assert row["Symbol"] == "GGG"
    assert row["Date"] == DATES[3]
    assert row["log_return"] == pytest.approx(np.log(2.0))
    assert row["simple_return"] == pytest.approx(1.0)
    assert row["close_log_return"] == pytest.approx(np.log(2.0))
    assert metric(result.log, "n_suspect_returns") == 1


def test_date_window_applied() -> None:
    kwargs = {**KWARGS, "start_date": "2020-01-02", "end_date": "2020-01-09"}
    res = clean_prices(make_stocks(PRICES), **kwargs)
    assert res.log_returns.index[0] == DATES[2]
    assert res.log_returns.index[-1] == DATES[6]
    assert len(res.log_returns) == 5


def test_empty_date_dropped_without_threshold() -> None:
    dates = DATES[:4]
    prices = {"AAA": [10, NAN, 12, 13], "BBB": [20, NAN, 22, 23]}
    res = clean_prices(make_stocks(prices, dates), **KWARGS)
    assert metric(res.log, "n_dates_dropped_no_price") == 1
    assert list(res.log_returns.index) == [dates[2], dates[3]]
    assert res.log_returns.loc[dates[2], "AAA"] == pytest.approx(np.log(12 / 10))


def test_nonpositive_price_masked_and_no_inf() -> None:
    prices = {"AAA": [10, 10, 0, 10, 10, 10, 10, 10], "BBB": PRICES["AAA"]}
    res = clean_prices(make_stocks(prices), **KWARGS)
    assert metric(res.log, "n_nonpositive_prices_masked") == 1
    assert not np.isinf(res.log_returns.to_numpy()).any()
    assert res.log_returns.loc[DATES[2], "AAA"] == 0.0


def test_duplicate_rows_raise() -> None:
    stocks = make_stocks({"AAA": PRICES["AAA"]})
    with pytest.raises(ValueError, match="duplicate"):
        clean_prices(pd.concat([stocks, stocks.iloc[:1]], ignore_index=True), **KWARGS)


def test_missing_column_raises() -> None:
    with pytest.raises(ValueError, match="missing column"):
        clean_prices(make_stocks({"AAA": PRICES["AAA"]}).drop(columns=["Volume"]), **KWARGS)


def test_all_symbols_dropped_raises() -> None:
    with pytest.raises(ValueError, match="no symbols left"):
        clean_prices(make_stocks({"EEE": PRICES["EEE"], "AAA": PRICES["EEE"]}), **KWARGS)


def test_input_not_mutated() -> None:
    stocks = make_stocks(PRICES)
    before = stocks.copy(deep=True)
    clean_prices(stocks, **KWARGS)
    pd.testing.assert_frame_equal(stocks, before)


def test_no_future_information(result) -> None:
    """Changing prices after date d must not change any return up to and including d."""
    cutoff = 4
    changed = {k: list(v) for k, v in PRICES.items()}
    for series in changed.values():
        for i in range(cutoff + 1, 8):
            series[i] = 999.0
    res2 = clean_prices(make_stocks(changed), **KWARGS)
    upto = DATES[1 : cutoff + 1]
    pd.testing.assert_frame_equal(
        result.log_returns.loc[upto, ["AAA", "BBB", "CCC", "GGG"]],
        res2.log_returns.loc[upto, ["AAA", "BBB", "CCC", "GGG"]],
    )


def test_write_processed_roundtrip(result, tmp_path) -> None:
    paths = write_processed(result, tmp_path / "processed")
    assert [p.name for p in paths] == [f"{n}.parquet" for n in PROCESSED_FILES]
    for name in PROCESSED_FILES:
        back = pd.read_parquet(tmp_path / "processed" / f"{name}.parquet")
        pd.testing.assert_frame_equal(
            back, getattr(result, name), check_freq=False, check_names=False
        )
