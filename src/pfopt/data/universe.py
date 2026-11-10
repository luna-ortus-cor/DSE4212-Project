"""Point-in-time universe selection.

At a rebalance date ``t`` the universe uses only data dated ``<= t``:

1. Eligible symbols have no NaN in their last ``T`` log returns (up to and including ``t``).
2. Eligible symbols are ranked by trailing average dollar volume (last ``dollar_volume_window``
   days up to and including ``t``, NaN days skipped), and the top ``N`` are kept.

Limitation: the supplied data contains only symbols that are in the index at the snapshot date
(current constituents, and only those with prices), so survivorship bias remains. Applying the
rule point-in-time reduces look-ahead in the choice of names but cannot restore companies that
left the index earlier.
"""

from __future__ import annotations

import pandas as pd


def _check_frames(log_returns: pd.DataFrame, dollar_volume: pd.DataFrame | None = None) -> None:
    index = log_returns.index
    if not (index.is_unique and index.is_monotonic_increasing):
        raise ValueError("log_returns index must be unique and sorted ascending")
    if dollar_volume is not None and not (
        dollar_volume.index.equals(index) and dollar_volume.columns.equals(log_returns.columns)
    ):
        raise ValueError("dollar_volume must have the same index and columns as log_returns")


def _position(index: pd.Index, asof_date: str | pd.Timestamp) -> int:
    ts = pd.Timestamp(asof_date)
    if ts not in index:
        raise ValueError(f"asof_date {ts.date()} is not a date in the returns index")
    return int(index.get_loc(ts))


def eligible_symbols(
    log_returns: pd.DataFrame, asof_date: str | pd.Timestamp, lookback_days: int
) -> list[str]:
    """Symbols with no NaN in the last ``lookback_days`` log returns up to ``asof_date``.

    Parameters
    ----------
    log_returns : pandas.DataFrame
        Date x Symbol log returns (output of the cleaning step).
    asof_date : str or Timestamp
        A date in ``log_returns.index``; the window includes this date.
    lookback_days : int
        Window length ``T`` in trading days.

    Returns
    -------
    list of str
        Eligible symbols in column order.

    Raises
    ------
    ValueError
        If ``asof_date`` is not in the index or fewer than ``lookback_days`` observations exist.
    """
    _check_frames(log_returns)
    pos = _position(log_returns.index, asof_date)
    if pos + 1 < lookback_days:
        day = pd.Timestamp(asof_date).date()
        raise ValueError(f"fewer than {lookback_days} return observations up to {day}")
    window = log_returns.iloc[pos - lookback_days + 1 : pos + 1]
    ok = window.notna().all(axis=0).to_numpy()
    return [str(c) for c in window.columns[ok]]


def select_universe(
    log_returns: pd.DataFrame,
    dollar_volume: pd.DataFrame,
    asof_date: str | pd.Timestamp,
    lookback_days: int,
    n_assets: int,
    dollar_volume_window: int,
) -> list[str]:
    """Select the top ``n_assets`` eligible symbols by trailing average dollar volume.

    Uses only rows dated ``<= asof_date``. Eligibility is defined in :func:`eligible_symbols`.
    Ties in average dollar volume are broken alphabetically by symbol, so the result is
    deterministic.

    Parameters
    ----------
    log_returns, dollar_volume : pandas.DataFrame
        Date x Symbol frames with identical index and columns.
    asof_date : str or Timestamp
        A date in the index; the rule is evaluated at its close.
    lookback_days : int
        Estimation window ``T``.
    n_assets : int
        Universe size ``N``.
    dollar_volume_window : int
        Number of trailing days (including ``asof_date``) averaged to rank eligible names.

    Returns
    -------
    list of str
        ``n_assets`` symbols ordered by average dollar volume, largest first.

    Raises
    ------
    ValueError
        If frames are inconsistent, history is shorter than ``lookback_days`` or
        ``dollar_volume_window``, or fewer than ``n_assets`` eligible symbols have volume data.
    """
    _check_frames(log_returns, dollar_volume)
    pos = _position(log_returns.index, asof_date)
    need = max(lookback_days, dollar_volume_window)
    if pos + 1 < need:
        raise ValueError(
            f"fewer than {need} observations up to {pd.Timestamp(asof_date).date()} "
            f"(lookback_days={lookback_days}, dollar_volume_window={dollar_volume_window})"
        )
    eligible = eligible_symbols(log_returns, asof_date, lookback_days)
    window = dollar_volume.iloc[pos - dollar_volume_window + 1 : pos + 1]
    adv = window[eligible].mean(skipna=True).dropna()
    if len(adv) < n_assets:
        raise ValueError(
            f"only {len(adv)} eligible symbols with dollar-volume data at "
            f"{pd.Timestamp(asof_date).date()}, need {n_assets}"
        )
    table = adv.rename("adv").rename_axis("symbol").reset_index()
    table = table.sort_values(["adv", "symbol"], ascending=[False, True])
    return [str(s) for s in table["symbol"].head(n_assets)]


def eligible_counts(log_returns: pd.DataFrame, lookback_days: int) -> pd.Series:
    """Number of eligible symbols at every date with a full ``lookback_days`` window.

    Equals ``len(eligible_symbols(...))`` at each date, computed in one rolling pass.

    Parameters
    ----------
    log_returns : pandas.DataFrame
        Date x Symbol log returns.
    lookback_days : int
        Window length ``T``.

    Returns
    -------
    pandas.Series
        Integer counts indexed by date, starting at the ``lookback_days``-th return date.
    """
    _check_frames(log_returns)
    rolling = (
        log_returns.isna().astype(int).rolling(window=lookback_days, min_periods=lookback_days)
    )
    clean = rolling.sum() == 0
    counts = clean.sum(axis=1).iloc[lookback_days - 1 :]
    return counts.astype("int64").rename("n_eligible")


def rebalance_dates(
    log_returns: pd.DataFrame, lookback_days: int, n_assets: int, rebalance_days: int
) -> pd.DatetimeIndex:
    """Rebalance calendar: the first feasible date, then every ``rebalance_days`` trading days.

    Later dates are not checked for feasibility; :func:`select_universe` raises if fewer than
    ``n_assets`` symbols are eligible at one of them.

    Parameters
    ----------
    log_returns : pandas.DataFrame
        Date x Symbol log returns.
    lookback_days, n_assets : int
        Window ``T`` and universe size ``N`` defining feasibility.
    rebalance_days : int
        Trading days between rebalances (must be positive).
    """
    if rebalance_days <= 0:
        raise ValueError(f"rebalance_days must be > 0, got {rebalance_days}")
    first = first_feasible_date(log_returns, lookback_days, n_assets)
    start = int(log_returns.index.get_loc(first))
    return pd.DatetimeIndex(log_returns.index[start::rebalance_days])


def snapshot_dates(
    log_returns: pd.DataFrame, lookback_days: int, n_assets: int, stress_date: str
) -> dict[str, pd.Timestamp]:
    """The three snapshot dates used by the EDA and the estimator comparison.

    Parameters
    ----------
    log_returns : pandas.DataFrame
        Date x Symbol log returns.
    lookback_days, n_assets : int
        Window ``T`` and universe size ``N`` defining feasibility.
    stress_date : str
        ISO date of the stress snapshot; it must be a feasible date (a date with a full window and
        not earlier than the first feasible date).

    Returns
    -------
    dict
        ``{"first": first feasible date, "stress": stress_date, "last": last date}``.

    Raises
    ------
    ValueError
        If the stress date is not a feasible date.
    """
    counts = eligible_counts(log_returns, lookback_days)
    first = first_feasible_date(log_returns, lookback_days, n_assets)
    stress = pd.Timestamp(stress_date)
    if stress not in counts.index or stress < first:
        message = f"stress_date {stress.date()} is not a feasible date (first: {first.date()})"
        raise ValueError(message)
    return {"first": first, "stress": stress, "last": pd.Timestamp(log_returns.index[-1])}


def first_feasible_date(
    log_returns: pd.DataFrame, lookback_days: int, n_assets: int
) -> pd.Timestamp:
    """First date at which at least ``n_assets`` symbols are eligible.

    Raises
    ------
    ValueError
        If the universe size is never reached.
    """
    counts = eligible_counts(log_returns, lookback_days)
    feasible = counts[counts >= n_assets]
    if feasible.empty:
        raise ValueError(
            f"never {n_assets} eligible symbols (maximum {int(counts.max()) if len(counts) else 0})"
        )
    return pd.Timestamp(feasible.index[0])
