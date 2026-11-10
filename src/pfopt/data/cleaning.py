"""Cleaning and return matrices.

Turns the raw long stock panel into Date x Symbol matrices of log returns (for estimation),
simple returns (for portfolio P&L), volume and dollar volume. No look-ahead is possible here: every
operation is causal in time (forward fill, one-day differences). Winsorization is *not* applied
here; it is done per estimation window inside the backtest to avoid leakage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROCESSED_FILES = ("log_returns", "simple_returns", "volume", "dollar_volume")
_REQUIRED_COLUMNS = ("Date", "Symbol", "Adj Close", "Close", "Volume")


@dataclass(frozen=True)
class CleaningResult:
    """Output of :func:`clean_prices`.

    Attributes
    ----------
    log_returns, simple_returns : pandas.DataFrame
        Date x Symbol returns from the forward-filled adjusted close. NaN where a price is
        unavailable on either day. The first calendar date is dropped (no previous price).
    volume, dollar_volume : pandas.DataFrame
        Same index and columns as the returns. ``dollar_volume = Volume * Close`` (unfilled).
    suspect : pandas.DataFrame
        One row per return with ``|log return| > threshold`` (``Date, Symbol, log_return,
        simple_return, close_log_return``). ``close_log_return`` uses the unadjusted close, so a
        split artifact shows up as a large value there but not in ``log_return``.
    dropped : pandas.DataFrame
        Symbols removed, with ``reason``, first/last valid date, span length and missing share.
    log : pandas.DataFrame
        Columns ``metric, detail, value`` with counts for the cleaning log.
    """

    log_returns: pd.DataFrame
    simple_returns: pd.DataFrame
    volume: pd.DataFrame
    dollar_volume: pd.DataFrame
    suspect: pd.DataFrame
    dropped: pd.DataFrame
    log: pd.DataFrame


def _pivot(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Pivot the long panel to Date x Symbol with plain string column labels."""
    wide = df.pivot(index="Date", columns="Symbol", values=column).sort_index()
    wide.columns = pd.Index([str(c) for c in wide.columns], name="Symbol")
    return wide.sort_index(axis=1)


def _fmt(ts: object) -> str:
    return "" if ts is None or pd.isna(ts) else pd.Timestamp(ts).strftime("%Y-%m-%d")


def clean_prices(
    stocks: pd.DataFrame,
    *,
    start_date: str | None,
    end_date: str | None,
    ffill_limit: int,
    max_missing_frac: float,
    suspect_abs_logret: float,
) -> CleaningResult:
    """Clean the raw stock panel and build return, volume and dollar-volume matrices.

    Steps (all causal in time):

    1. Restrict to ``[start_date, end_date]``; pivot ``Adj Close``, ``Close`` and ``Volume`` to
       Date x Symbol. Non-positive adjusted prices are masked to NaN. Dates on which no symbol has
       a price are dropped (no coverage threshold is applied).
    2. Forward-fill ``Adj Close`` gaps of at most ``ffill_limit`` consecutive days, **only inside
       each symbol's own listed span** (first to last valid price). Nothing is filled before
       listing or after the last price. Volume is never filled.
    3. Drop symbols with no price at all, and symbols whose share of missing days inside their own
       span (after filling) exceeds ``max_missing_frac``. A late listing is not "missing".
    4. Returns: ``r = ln(P_t / P_{t-1})`` (estimation) and ``R = P_t / P_{t-1} - 1`` (P&L), so that
       ``exp(r) - 1 == R``. Days with a filled price have a return of 0.
    5. Flag returns with ``|r| > suspect_abs_logret``.

    Parameters
    ----------
    stocks : pandas.DataFrame
        Long panel from :func:`pfopt.data.loader.load_stocks`.
    start_date, end_date : str or None
        Inclusive ISO bounds, or ``None`` for no bound.
    ffill_limit : int
        Maximum run of consecutive missing days that is forward-filled.
    max_missing_frac : float
        Maximum tolerated missing share inside a symbol's own span.
    suspect_abs_logret : float
        Absolute log-return threshold for the suspect table.

    Returns
    -------
    CleaningResult

    Raises
    ------
    ValueError
        On missing columns, duplicate ``(Date, Symbol)`` rows, an empty date window, no symbols
        left after cleaning, or non-finite returns.
    """
    missing_cols = [c for c in _REQUIRED_COLUMNS if c not in stocks.columns]
    if missing_cols:
        raise ValueError(f"stocks is missing column(s) {missing_cols}")
    if stocks.duplicated(["Date", "Symbol"]).any():
        raise ValueError("stocks contains duplicate (Date, Symbol) rows")

    df = stocks
    if start_date is not None:
        df = df[df["Date"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        df = df[df["Date"] <= pd.Timestamp(end_date)]
    if df.empty:
        raise ValueError("no rows inside the [start_date, end_date] window")

    adj = _pivot(df, "Adj Close")
    close = _pivot(df, "Close")
    vol = _pivot(df, "Volume")
    n_symbols_input = adj.shape[1]
    n_dates_window = adj.shape[0]

    n_nonpositive = int((adj.notna() & ~(adj > 0)).sum().sum())
    adj = adj.where(adj > 0)
    close = close.where(close > 0)

    has_data = adj.notna().any(axis=1)
    n_empty_dates = int((~has_data).sum())
    adj, close, vol = adj[has_data], close[has_data], vol[has_data]

    valid = adj.notna()
    started = valid.astype(int).cummax().astype(bool)
    not_ended = valid.iloc[::-1].astype(int).cummax().iloc[::-1].astype(bool)
    in_span = started & not_ended
    filled = adj.ffill(limit=ffill_limit).where(in_span)

    span_len = in_span.sum()
    n_missing = (in_span & filled.isna()).sum()
    no_price = ~valid.any()
    frac = n_missing / span_len.where(span_len > 0)
    too_missing = ~no_price & (frac > max_missing_frac)
    drop_mask = no_price | too_missing

    info = pd.DataFrame(
        {
            "reason": np.where(no_price, "no_price", "missing_gt_threshold"),
            "first_valid": adj.apply(pd.Series.first_valid_index).map(_fmt),
            "last_valid": adj.apply(pd.Series.last_valid_index).map(_fmt),
            "span_days": span_len,
            "missing_frac": frac,
        },
        index=adj.columns,
    )
    dropped = info[drop_mask].rename_axis("Symbol").reset_index()

    keep = ~drop_mask
    if not keep.any():
        raise ValueError("no symbols left after cleaning")
    filled, close, vol = filled.loc[:, keep], close.loc[:, keep], vol.loc[:, keep]
    n_filled = int((filled.notna() & adj.loc[:, keep].isna()).sum().sum())
    n_span_cells = int(in_span.loc[:, keep].sum().sum())

    ratio = filled / filled.shift(1)
    log_ret = np.log(ratio).iloc[1:]
    simple_ret = (ratio - 1.0).iloc[1:]
    if np.isinf(log_ret.to_numpy()).any() or np.isinf(simple_ret.to_numpy()).any():
        raise ValueError("non-finite returns produced")
    idx = log_ret.index
    volume = vol.loc[idx]
    dollar_volume = (vol * close).loc[idx]

    close_ret = np.log(close / close.shift(1)).loc[idx]
    rows, cols = np.nonzero((log_ret.abs() > suspect_abs_logret).to_numpy())
    suspect = pd.DataFrame(
        {
            "Date": idx[rows],
            "Symbol": log_ret.columns[cols],
            "log_return": log_ret.to_numpy()[rows, cols],
            "simple_return": simple_ret.to_numpy()[rows, cols],
            "close_log_return": close_ret.to_numpy()[rows, cols],
        }
    )
    suspect = suspect.sort_values(["Date", "Symbol"]).reset_index(drop=True)

    pct_filled = 100.0 * n_filled / n_span_cells
    n_dropped_no_price = int(no_price.sum())
    n_dropped_missing = int(too_missing.sum())
    log = pd.DataFrame(
        [
            ("n_symbols_input", "", n_symbols_input),
            ("n_dates_in_window", "", n_dates_window),
            ("n_dates_dropped_no_price", "", n_empty_dates),
            ("n_dates_kept", "", int(has_data.sum())),
            ("n_return_dates", "", len(idx)),
            ("date_min", "", _fmt(idx.min())),
            ("date_max", "", _fmt(idx.max())),
            ("n_nonpositive_prices_masked", "", n_nonpositive),
            ("n_symbols_dropped_no_price", "", n_dropped_no_price),
            (
                "n_symbols_dropped_missing_gt_threshold",
                f"max_missing_frac={max_missing_frac}",
                n_dropped_missing,
            ),
            ("n_symbols_kept", "", int(keep.sum())),
            ("n_prices_forward_filled", f"limit={ffill_limit}", n_filled),
            ("pct_prices_forward_filled", "of in-span cells, kept symbols", pct_filled),
            ("n_nan_log_returns", "", int(log_ret.isna().sum().sum())),
            ("n_suspect_returns", f"abs_logret>{suspect_abs_logret}", len(suspect)),
            ("n_suspect_symbols", "", int(suspect["Symbol"].nunique())),
        ],
        columns=["metric", "detail", "value"],
    )
    for _, row in log.iterrows():
        logger.info("%s %s: %s", row["metric"], row["detail"], row["value"])

    return CleaningResult(
        log_returns=log_ret,
        simple_returns=simple_ret,
        volume=volume,
        dollar_volume=dollar_volume,
        suspect=suspect,
        dropped=dropped,
        log=log,
    )


def write_processed(result: CleaningResult, out_dir: str | Path) -> list[Path]:
    """Write the four matrices to ``out_dir`` as parquet files.

    Parameters
    ----------
    result : CleaningResult
        Output of :func:`clean_prices`.
    out_dir : str or Path
        Target directory (created if needed). Raw data is never written here.

    Returns
    -------
    list of Path
        Paths of ``log_returns``, ``simple_returns``, ``volume`` and ``dollar_volume`` files.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in PROCESSED_FILES:
        path = out / f"{name}.parquet"
        getattr(result, name).to_parquet(path)
        paths.append(path)
    return paths


def read_processed(in_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Read the four matrices written by :func:`write_processed`.

    Parameters
    ----------
    in_dir : str or Path
        Directory containing ``log_returns.parquet``, ``simple_returns.parquet``,
        ``volume.parquet`` and ``dollar_volume.parquet``.

    Returns
    -------
    dict of str to pandas.DataFrame
        Keys are the names in :data:`PROCESSED_FILES`.

    Raises
    ------
    FileNotFoundError
        If a file is missing (run ``scripts/01b_clean_data.py`` first).
    """
    base = Path(in_dir)
    frames = {}
    for name in PROCESSED_FILES:
        path = base / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run scripts/01b_clean_data.py first")
        frames[name] = pd.read_parquet(path)
    return frames
