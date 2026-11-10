"""Readers for the raw Kaggle CSVs.

The loaders parse types explicitly and fail loudly on schema problems. They never clean,
fill, sort or drop anything: that is the job of ``cleaning.py``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DATE_FORMAT = "%Y-%m-%d"
_STRING_COLUMNS = frozenset({"Date", "Symbol", "Exchange", "Sector", "Industry"})


def _check_columns(found: Sequence[str], required: Sequence[str], name: str) -> None:
    """Raise if a required column is absent; warn about columns that will be ignored."""
    missing = [c for c in required if c not in found]
    if missing:
        raise ValueError(f"{name}: missing column(s) {missing}; file has {list(found)}")
    extra = [c for c in found if c not in required]
    if extra:
        logger.warning("%s: ignoring column(s) not requested in config: %s", name, extra)


def _parse_dates(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Convert the ``Date`` column to datetime64, requiring strict ``YYYY-MM-DD`` strings."""
    try:
        parsed = pd.to_datetime(df["Date"], format=DATE_FORMAT)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name}: 'Date' is not strictly {DATE_FORMAT}: {exc}") from exc
    if parsed.isna().any():
        raise ValueError(f"{name}: {int(parsed.isna().sum())} empty Date value(s)")
    return df.assign(Date=parsed)


def _read(path: str | Path, columns: Sequence[str], name: str) -> pd.DataFrame:
    """Read ``columns`` of a CSV with explicit dtypes (strings for ids, float64 otherwise)."""
    header = list(pd.read_csv(path, nrows=0).columns)
    _check_columns(header, columns, name)
    dtypes = {c: ("string" if c in _STRING_COLUMNS else "float64") for c in columns}
    df = pd.read_csv(path, usecols=list(columns), dtype=dtypes)
    df = df[list(columns)]
    if "Date" in df.columns:
        df = _parse_dates(df, name)
    logger.info("%s: loaded %d rows x %d columns from %s", name, len(df), df.shape[1], path)
    return df


def load_stocks(path: str | Path, columns: Sequence[str]) -> pd.DataFrame:
    """Load the daily stock panel (one row per Date and Symbol, long format).

    Parameters
    ----------
    path : str or Path
        Location of ``sp500_stocks.csv``.
    columns : sequence of str
        Columns to read; must all exist in the file. ``Date`` becomes datetime64, ``Symbol``
        a string, every other column float64. Missing values stay NaN.

    Returns
    -------
    pandas.DataFrame
        Raw rows in file order with a default ``RangeIndex``. No cleaning is applied.
    """
    return _read(path, columns, "stocks")


def load_companies(path: str | Path, columns: Sequence[str]) -> pd.DataFrame:
    """Load the company snapshot (sector, market cap, index weight).

    The file is a *current* snapshot, so it is only suitable for static labels such as sectors.

    Parameters
    ----------
    path : str or Path
        Location of ``sp500_companies.csv``.
    columns : sequence of str
        Columns to read. Id-like columns (``Symbol``, ``Sector``, ``Industry``, ``Exchange``)
        are strings; all others are float64.
    """
    return _read(path, columns, "companies")


def load_index(path: str | Path, columns: Sequence[str]) -> pd.DataFrame:
    """Load the S&P 500 index level series.

    Parameters
    ----------
    path : str or Path
        Location of ``sp500_index.csv``.
    columns : sequence of str
        ``("Date", <level column>)``. The level column is float64 and keeps its raw name.
    """
    return _read(path, columns, "index")
