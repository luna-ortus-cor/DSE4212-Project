"""Read-only audit of the raw CSVs (PLAN.md step 1.1).

Every function returns a long table with columns ``section, metric, detail, value`` and never
modifies its inputs. The audit reports anomalies; it does not clean anything.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

PRICE_COLS = ("Adj Close", "Close", "High", "Low", "Open")
AUDIT_COLUMNS = ("section", "metric", "detail", "value")

_TOLERANCE = 1e-6  # relative tolerance when comparing Adj Close with Close
_LONG_GAP_DAYS = 4  # calendar days; longer gaps are not explained by a weekend plus one holiday
_MAX_LISTED = 20  # maximum number of ticker names written into a single detail string
_COVERAGE_FRACTIONS = (0.1, 0.5)
_QUANTILES = (("min", 0.0), ("q25", 0.25), ("median", 0.5), ("q75", 0.75), ("max", 1.0))

Row = tuple[str, str, str, Any]


def _fmt_date(ts: object) -> str:
    """Format a timestamp as ``YYYY-MM-DD``."""
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def _to_frame(rows: list[Row]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=list(AUDIT_COLUMNS))


def _distribution(section: str, metric: str, values: pd.Series, detail: str = "") -> list[Row]:
    """Min, quartiles and max of ``values`` as audit rows named ``<metric>_<stat>``."""
    return [
        (section, f"{metric}_{label}", detail, float(values.quantile(q))) for label, q in _QUANTILES
    ]


def _longest_na_run(mask: np.ndarray) -> int:
    """Length of the longest run of ``True`` in a boolean array (0 if empty)."""
    best = current = 0
    for flag in mask:
        current = current + 1 if flag else 0
        best = max(best, current)
    return best


def _stocks_overview(df: pd.DataFrame) -> list[Row]:
    valid = df["Adj Close"].notna()
    valid_dates = df.loc[valid, "Date"]
    rows: list[Row] = [
        ("overview", "n_rows", "", len(df)),
        ("overview", "n_symbols", "", int(df["Symbol"].nunique())),
        ("overview", "n_unique_dates", "", int(df["Date"].nunique())),
        ("overview", "date_min_all_rows", "", _fmt_date(df["Date"].min())),
        ("overview", "date_max_all_rows", "", _fmt_date(df["Date"].max())),
    ]
    if len(valid_dates):
        rows += [
            ("overview", "first_valid_adj_close_date", "", _fmt_date(valid_dates.min())),
            ("overview", "last_valid_adj_close_date", "", _fmt_date(valid_dates.max())),
        ]
    return rows


def _stocks_missing(df: pd.DataFrame) -> list[Row]:
    no_price = ~df[list(PRICE_COLS)].notna().any(axis=1)
    n_empty = int(no_price.sum())
    n_volume_only = int((no_price & df["Volume"].notna()).sum())
    rows: list[Row] = [
        ("missing", "pct_missing", col, float(df[col].isna().mean() * 100)) for col in df.columns
    ]
    rows += [
        ("missing", "n_rows_all_prices_nan", "", n_empty),
        ("missing", "pct_rows_all_prices_nan", "", float(n_empty / len(df) * 100)),
        ("missing", "n_rows_volume_without_price", "", n_volume_only),
    ]
    return rows


def _stocks_anomalies(df: pd.DataFrame) -> list[Row]:
    n_dup_key = int(df.duplicated(["Date", "Symbol"]).sum())
    zero_volume = df["Volume"] == 0
    n_zero_with_price = int((zero_volume & df["Adj Close"].notna()).sum())
    close_outside = (df["Close"] > df["High"]) | (df["Close"] < df["Low"])
    rows: list[Row] = [
        ("anomalies", "n_duplicate_date_symbol_rows", "", n_dup_key),
        ("anomalies", "n_duplicate_full_rows", "", int(df.duplicated().sum())),
    ]
    rows += [("anomalies", "n_nonpositive", col, int((df[col] <= 0).sum())) for col in PRICE_COLS]
    rows += [
        ("anomalies", "n_negative_volume", "", int((df["Volume"] < 0).sum())),
        ("anomalies", "n_zero_volume", "", int(zero_volume.sum())),
        ("anomalies", "n_zero_volume_with_price", "", n_zero_with_price),
        ("anomalies", "n_high_lt_low", "", int((df["High"] < df["Low"]).sum())),
        ("anomalies", "n_close_outside_high_low", "", int(close_outside.sum())),
    ]
    return rows


def _stocks_symbol_history(df: pd.DataFrame, ffill_limit: int, top_k: int) -> list[Row]:
    sym = df["Symbol"]
    valid = df["Adj Close"].notna()
    rows_per = sym.value_counts().sort_index()
    valid_per = valid.groupby(sym).sum().astype(int).sort_index()

    rows: list[Row] = []
    rows += _distribution("symbol_history", "rows_per_symbol", rows_per)
    rows += _distribution("symbol_history", "valid_adj_close_rows_per_symbol", valid_per)

    valid_rows = df.loc[valid, ["Symbol", "Date"]]
    first = valid_rows.groupby("Symbol")["Date"].min()
    last = valid_rows.groupby("Symbol")["Date"].max()
    no_valid = sorted(str(s) for s in valid_per.index[valid_per == 0])
    rows.append(("symbol_history", "n_symbols_without_valid_price", "", len(no_valid)))
    if no_valid:
        listed = ", ".join(no_valid[:_MAX_LISTED])
        rows.append(("symbol_history", "symbols_without_valid_price", "", listed))
    if len(valid_rows):
        n_late = int((first > valid_rows["Date"].min()).sum())
        n_early = int((last < valid_rows["Date"].max()).sum())
        rows += [
            ("symbol_history", "n_late_listing", "first valid > global first", n_late),
            ("symbol_history", "n_early_end", "last valid < global last", n_early),
        ]

    def describe(symbol: str) -> str:
        if symbol not in first.index:
            return f"{symbol} first=none last=none"
        return f"{symbol} first={_fmt_date(first[symbol])} last={_fmt_date(last[symbol])}"

    shortest = valid_per.sort_values(kind="stable").head(top_k)
    longest = valid_per.sort_values(ascending=False, kind="stable").head(top_k)
    rows += [
        ("symbol_history", "shortest_history_valid_rows", describe(s), int(n))
        for s, n in shortest.items()
    ]
    rows += [
        ("symbol_history", "longest_history_valid_rows", describe(s), int(n))
        for s, n in longest.items()
    ]

    # Missing values *inside* each symbol's own listed span (first to last valid date).
    in_span = (df["Date"] >= sym.map(first)) & (df["Date"] <= sym.map(last))
    span = df.loc[in_span, ["Symbol", "Date", "Adj Close"]].sort_values(["Symbol", "Date"])
    runs = pd.Series(
        {
            str(s): _longest_na_run(g["Adj Close"].isna().to_numpy())
            for s, g in span.groupby("Symbol", sort=False)
        },
        dtype="int64",
    )
    n_over_limit = int((runs > ffill_limit).sum())
    limit_note = f"limit={ffill_limit}"
    rows += [
        ("symbol_history", "n_symbols_internal_gap", "", int((runs > 0).sum())),
        ("symbol_history", "n_internal_gap_gt_ffill_limit", limit_note, n_over_limit),
        ("symbol_history", "max_internal_gap_run", "", int(runs.max()) if len(runs) else 0),
    ]
    worst = runs[runs > 0].sort_index().sort_values(ascending=False, kind="stable").head(top_k)
    rows += [
        ("symbol_history", "longest_internal_gap_run", str(s), int(n)) for s, n in worst.items()
    ]
    return rows


def _stocks_coverage(df: pd.DataFrame) -> list[Row]:
    n_symbols = df["Symbol"].nunique()
    coverage = df.loc[df["Adj Close"].notna()].groupby("Date")["Symbol"].nunique()
    n_zero = int(df["Date"].nunique() - len(coverage))
    rows: list[Row] = [("coverage", "n_dates_with_no_valid_price", "", n_zero)]
    if len(coverage):
        rows += _distribution("coverage", "valid_symbols_per_date", coverage)
    for frac in _COVERAGE_FRACTIONS:
        below = int((coverage < frac * n_symbols).sum()) + n_zero
        rows.append(("coverage", f"n_dates_coverage_below_{round(frac * 100)}pct", "", below))
    return rows


def _stocks_adjustment(df: pd.DataFrame) -> list[Row]:
    mask = (df["Adj Close"] > 0) & (df["Close"] > 0)
    if not mask.any():
        return []
    ratio = df.loc[mask, "Adj Close"] / df.loc[mask, "Close"]
    rows: list[Row] = _distribution("adjustment", "adj_close_over_close", ratio)
    rows += [
        ("adjustment", "adj_close_over_close_q0.1pct", "", float(ratio.quantile(0.001))),
        ("adjustment", "adj_close_over_close_q99.9pct", "", float(ratio.quantile(0.999))),
        ("adjustment", "n_adj_close_gt_close", "", int((ratio > 1 + _TOLERANCE).sum())),
    ]
    for label, pos in (("min", ratio.idxmin()), ("max", ratio.idxmax())):
        where = f"{df.at[pos, 'Symbol']} {_fmt_date(df.at[pos, 'Date'])}"
        rows.append(("adjustment", f"adj_close_over_close_{label}_at", "", where))
    return rows


def audit_stocks(df: pd.DataFrame, ffill_limit: int, top_k: int = 5) -> pd.DataFrame:
    """Audit the long stock panel.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of :func:`pfopt.data.loader.load_stocks` (default ``RangeIndex`` required).
    ffill_limit : int
        Maximum forward-fill gap from the config; used to count internal gaps that exceed it.
    top_k : int, default 5
        Number of shortest/longest histories and worst gaps listed.

    Returns
    -------
    pandas.DataFrame
        Columns ``section, metric, detail, value``.
    """
    rows: list[Row] = []
    rows += _stocks_overview(df)
    rows += _stocks_missing(df)
    rows += _stocks_anomalies(df)
    rows += _stocks_symbol_history(df, ffill_limit, top_k)
    rows += _stocks_coverage(df)
    rows += _stocks_adjustment(df)
    return _to_frame(rows)


def audit_companies(companies: pd.DataFrame, stock_symbols: set[str]) -> pd.DataFrame:
    """Audit the company snapshot and its symbol overlap with the stock panel.

    Parameters
    ----------
    companies : pandas.DataFrame
        Output of :func:`pfopt.data.loader.load_companies` (needs ``Symbol`` and ``Sector``).
    stock_symbols : set of str
        Symbols present in the stock panel.
    """
    company_symbols = set(companies["Symbol"].dropna().astype(str))
    only_stocks = sorted(stock_symbols - company_symbols)
    only_companies = sorted(company_symbols - stock_symbols)
    rows: list[Row] = [
        ("companies", "n_rows", "", len(companies)),
        ("companies", "n_unique_symbols", "", len(company_symbols)),
        ("companies", "n_duplicate_symbols", "", int(companies["Symbol"].duplicated().sum())),
        ("companies", "n_missing_sector", "", int(companies["Sector"].isna().sum())),
        ("companies", "n_symbols_only_in_stocks", "", len(only_stocks)),
        ("companies", "symbols_only_in_stocks", "", ", ".join(only_stocks[:_MAX_LISTED])),
        ("companies", "n_symbols_only_in_companies", "", len(only_companies)),
        ("companies", "symbols_only_in_companies", "", ", ".join(only_companies[:_MAX_LISTED])),
    ]
    sectors = companies["Sector"].value_counts().sort_index()
    rows += [("companies", "n_per_sector", str(s), int(n)) for s, n in sectors.items()]
    return _to_frame(rows)


def audit_index(index: pd.DataFrame, stock_dates: pd.DatetimeIndex, top_k: int = 5) -> pd.DataFrame:
    """Audit the index series and its date overlap with the stock panel.

    Parameters
    ----------
    index : pandas.DataFrame
        Output of :func:`pfopt.data.loader.load_index`: ``Date`` plus one level column.
    stock_dates : pandas.DatetimeIndex
        Dates on which at least one stock has a valid ``Adj Close``.
    top_k : int, default 5
        Number of largest calendar gaps listed.
    """
    levels = [c for c in index.columns if c != "Date"]
    if len(levels) != 1:
        raise ValueError(f"index must have exactly one level column besides Date, got {levels}")
    level = index[levels[0]]
    dates = pd.DatetimeIndex(index["Date"]).drop_duplicates().sort_values()
    gap_days = dates.to_series().diff().dt.days
    previous = dates.to_series().shift(1)

    index_set = set(dates)
    stock_set = set(stock_dates)
    in_range = {d for d in stock_set if dates.min() <= d <= dates.max()}
    n_long_gaps = int((gap_days > _LONG_GAP_DAYS).sum())
    rows: list[Row] = [
        ("index", "n_rows", "", len(index)),
        ("index", "date_min", "", _fmt_date(dates.min())),
        ("index", "date_max", "", _fmt_date(dates.max())),
        ("index", "n_duplicate_dates", "", int(index["Date"].duplicated().sum())),
        ("index", "is_sorted_ascending", "", bool(index["Date"].is_monotonic_increasing)),
        ("index", "n_missing_values", levels[0], int(level.isna().sum())),
        ("index", "n_nonpositive_values", levels[0], int((level <= 0).sum())),
        ("index", "max_gap_days", "", int(gap_days.max()) if len(dates) > 1 else 0),
        ("index", "n_gaps_gt_threshold", f"days>{_LONG_GAP_DAYS}", n_long_gaps),
        ("index", "n_dates_in_index_not_in_stocks", "", len(index_set - stock_set)),
        ("index", "n_stock_dates_in_index_range_not_in_index", "", len(in_range - index_set)),
    ]
    for end, days in gap_days.nlargest(top_k).items():
        span = f"{_fmt_date(previous[end])}->{_fmt_date(end)}"
        rows.append(("index", "largest_gap_days", span, int(days)))
    return _to_frame(rows)


def run_audit(
    stocks: pd.DataFrame,
    companies: pd.DataFrame,
    index: pd.DataFrame,
    ffill_limit: int,
    top_k: int = 5,
) -> pd.DataFrame:
    """Run all audits and concatenate them into one table.

    Parameters
    ----------
    stocks, companies, index : pandas.DataFrame
        Outputs of the three loaders.
    ffill_limit : int
        Forward-fill limit from the config (see :func:`audit_stocks`).
    top_k : int, default 5
        Length of the "top" lists.
    """
    valid_dates = pd.DatetimeIndex(stocks.loc[stocks["Adj Close"].notna(), "Date"].unique())
    stock_symbols = set(stocks["Symbol"].dropna().astype(str))
    return pd.concat(
        [
            audit_stocks(stocks, ffill_limit, top_k),
            audit_companies(companies, stock_symbols),
            audit_index(index, valid_dates, top_k),
        ],
        ignore_index=True,
    )
