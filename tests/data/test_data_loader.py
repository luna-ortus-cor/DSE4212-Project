"""Tests for the raw CSV loaders."""

import logging
from pathlib import Path

import pandas as pd
import pytest

from pfopt.data.loader import load_companies, load_index, load_stocks

STOCK_COLUMNS = ("Date", "Symbol", "Adj Close", "Close", "High", "Low", "Open", "Volume")

STOCKS_CSV = """Date,Symbol,Adj Close,Close,High,Low,Open,Volume
2020-01-02,AAA,,,,,,
2020-01-03,AAA,10.5,11.0,11.5,10.0,10.8,1000
2020-01-03,BBB,5.0,5.0,5.5,4.5,5.1,0
"""


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_load_stocks_types_and_nan(tmp_path: Path) -> None:
    df = load_stocks(_write(tmp_path / "s.csv", STOCKS_CSV), STOCK_COLUMNS)
    assert list(df.columns) == list(STOCK_COLUMNS)
    assert len(df) == 3
    assert pd.api.types.is_datetime64_any_dtype(df["Date"])
    assert str(df["Symbol"].dtype) == "string"
    assert all(df[c].dtype == "float64" for c in STOCK_COLUMNS[2:])
    assert df.loc[0, ["Adj Close", "Close", "Volume"]].isna().all()
    assert df.loc[1, "Adj Close"] == 10.5
    assert df.loc[2, "Volume"] == 0.0


def test_load_stocks_does_not_reorder_or_drop(tmp_path: Path) -> None:
    text = "Date,Symbol,Adj Close,Close,High,Low,Open,Volume\n"
    for line in ("2020-01-03,ZZZ", "2020-01-02,AAA", "2020-01-02,AAA"):
        text += f"{line},1,1,1,1,1,1\n"
    df = load_stocks(_write(tmp_path / "s.csv", text), STOCK_COLUMNS)
    assert list(df["Symbol"]) == ["ZZZ", "AAA", "AAA"]


def test_missing_column_raises(tmp_path: Path) -> None:
    text = "Date,Symbol,Close\n2020-01-02,AAA,1\n"
    with pytest.raises(ValueError, match="missing column"):
        load_stocks(_write(tmp_path / "s.csv", text), STOCK_COLUMNS)


def test_extra_column_warns_and_is_dropped(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    text = STOCKS_CSV.replace("Volume", "Volume,Extra").replace(",1000\n", ",1000,7\n")
    with caplog.at_level(logging.WARNING, logger="pfopt.data.loader"):
        df = load_stocks(_write(tmp_path / "s.csv", text), STOCK_COLUMNS)
    assert "Extra" not in df.columns
    assert any("Extra" in rec.message for rec in caplog.records)


def test_bad_date_format_raises(tmp_path: Path) -> None:
    text = STOCKS_CSV.replace("2020-01-03,AAA", "03/01/2020,AAA")
    with pytest.raises(ValueError, match="Date"):
        load_stocks(_write(tmp_path / "s.csv", text), STOCK_COLUMNS)


def test_non_numeric_price_raises(tmp_path: Path) -> None:
    text = STOCKS_CSV.replace("10.5", "abc")
    with pytest.raises(ValueError):
        load_stocks(_write(tmp_path / "s.csv", text), STOCK_COLUMNS)


def test_load_companies_handles_multiline_text(tmp_path: Path) -> None:
    text = (
        "Symbol,Sector,Industry,Marketcap,Weight,Longbusinesssummary\n"
        'AAA,Technology,Software,100.0,0.5,"First line, with comma\nsecond line"\n'
        "BBB,Energy,Oil,50.0,0.25,short\n"
    )
    cols = ("Symbol", "Sector", "Industry", "Marketcap", "Weight")
    df = load_companies(_write(tmp_path / "c.csv", text), cols)
    assert len(df) == 2
    assert list(df.columns) == list(cols)
    assert list(df["Sector"]) == ["Technology", "Energy"]
    assert df["Weight"].dtype == "float64"


def test_load_index(tmp_path: Path) -> None:
    text = "Date,S&P500\n2020-01-02,3257.85\n2020-01-03,3234.85\n"
    df = load_index(_write(tmp_path / "i.csv", text), ("Date", "S&P500"))
    assert list(df.columns) == ["Date", "S&P500"]
    assert pd.api.types.is_datetime64_any_dtype(df["Date"])
    assert df["S&P500"].dtype == "float64"
    assert df.loc[1, "S&P500"] == 3234.85
