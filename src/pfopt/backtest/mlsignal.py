"""Walk-forward machine-learning return signals.

At each rebalance date ``t`` four features of every name in the point-in-time universe are
computed from data dated ``<= t``: 12-1 month momentum, 1-month return (the reversal signal with
its sign left to the model), trailing volatility and trailing average dollar volume. Each feature
is ranked cross-sectionally within the universe (uniform on ``[-0.5, 0.5]``). The label of date
``s`` is the cross-sectional rank of the next ``horizon_days`` log return.

Models (Ridge and a random forest) are fitted on an expanding window of past rebalance dates and
refitted on a fixed schedule. A training date ``s`` enters the fit at cutoff ``c`` only if its label
window has ended ``purge_days >= horizon_days`` rows before ``c``, so no training label uses a
return after the cutoff. The prediction at ``t`` uses the model from the latest refit date
``c <= t``, hence only data dated ``<= t``.

The signal is first judged on its own: the out-of-sample Spearman rank information coefficient (IC)
between the prediction and the realized next-``horizon_days`` return, one value per rebalance date.
The t-statistic ``mean / (std / sqrt(n))`` treats the dates as independent, which holds when
``rebalance_days >= horizon_days`` (the label windows then do not overlap).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from pfopt.config import Config
from pfopt.data.universe import rebalance_dates, select_universe

logger = logging.getLogger(__name__)

FEATURES = ("mom_12_1", "ret_1m", "vol", "dollar_volume")
MODELS = ("RIDGE", "RF")
_MIN_CROSS_SECTION = 3


def rank_scores(values: np.ndarray) -> np.ndarray:
    """Cross-sectional rank of a vector, scaled to ``[-0.5, 0.5]`` (ties get the average rank).

    Parameters
    ----------
    values : numpy.ndarray
        Finite vector with at least two entries.

    Returns
    -------
    numpy.ndarray
        ``(rank - 1) / (n - 1) - 0.5``: the smallest value maps to -0.5 and the largest to 0.5.
    """
    v = np.asarray(values, dtype=float).ravel()
    if v.size < 2:
        raise ValueError("need at least two values to rank")
    if not np.isfinite(v).all():
        raise ValueError("values must be finite")
    return (rankdata(v) - 1.0) / (v.size - 1.0) - 0.5


def rank_ic(scores: np.ndarray, realized: np.ndarray) -> float:
    """Spearman rank correlation between scores and realized returns over the finite pairs.

    Returns NaN if fewer than three finite pairs exist or either side is constant.
    """
    a = np.asarray(scores, dtype=float).ravel()
    b = np.asarray(realized, dtype=float).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < _MIN_CROSS_SECTION:
        return float("nan")
    ra, rb = rankdata(a[ok]), rankdata(b[ok])
    if ra.std() == 0.0 or rb.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def raw_features(
    log_returns: pd.DataFrame,
    dollar_volume: pd.DataFrame,
    date: str | pd.Timestamp,
    symbols: list[str],
    cfg: Config,
) -> np.ndarray:
    """Unranked features of ``symbols`` at ``date``, using only rows dated ``<= date``.

    Parameters
    ----------
    log_returns, dollar_volume : pandas.DataFrame
        Date x Symbol frames with identical index and columns.
    date : str or Timestamp
        A date in the index.
    symbols : list of str
        Universe at ``date`` (no missing returns in the needed windows).
    cfg : Config
        Uses ``ml`` and ``universe.dollar_volume_window``.

    Returns
    -------
    numpy.ndarray
        ``len(symbols) x 4`` array in :data:`FEATURES` order: momentum (sum of log returns over
        ``momentum_days`` ending ``skip_days`` before ``date``), trailing ``reversal_days`` return,
        sample standard deviation of the last ``vol_days`` log returns and the mean dollar volume
        over ``universe.dollar_volume_window`` days.

    Raises
    ------
    ValueError
        If history is too short or a feature is not finite.
    """
    m = cfg.ml
    dv_window = cfg.universe.dollar_volume_window
    pos = int(log_returns.index.get_loc(pd.Timestamp(date)))
    need = max(m.momentum_days, m.reversal_days, m.vol_days)
    if pos + 1 < max(need, dv_window):
        raise ValueError(f"not enough history before {pd.Timestamp(date).date()} for the features")
    r = log_returns[symbols].iloc[pos + 1 - need : pos + 1].to_numpy(dtype=float)
    momentum = r[need - m.momentum_days : need - m.skip_days].sum(axis=0)
    recent = r[need - m.reversal_days :].sum(axis=0)
    vol = r[need - m.vol_days :].std(axis=0, ddof=1)
    volume = dollar_volume[symbols].iloc[pos + 1 - dv_window : pos + 1].mean(skipna=True)
    out = np.column_stack([momentum, recent, vol, volume.to_numpy(dtype=float)])
    if not np.isfinite(out).all():
        raise ValueError(f"non-finite feature at {pd.Timestamp(date).date()}")
    return out


def forward_returns(
    log_returns: pd.DataFrame, date: str | pd.Timestamp, symbols: list[str], horizon: int
) -> np.ndarray:
    """Sum of log returns over the ``horizon`` rows after ``date`` (NaN if not yet observable).

    This uses data *after* ``date``. It is a label or an evaluation target and must never feed a
    feature or a weight at ``date``.
    """
    pos = int(log_returns.index.get_loc(pd.Timestamp(date)))
    if pos + horizon >= len(log_returns.index):
        return np.full(len(symbols), np.nan)
    window = log_returns[symbols].iloc[pos + 1 : pos + 1 + horizon].to_numpy(dtype=float)
    return window.sum(axis=0)


@dataclass(frozen=True)
class MLPanel:
    """Features and labels on the rebalance calendar.

    Attributes
    ----------
    dates : pandas.DatetimeIndex
        Rebalance dates (the calendar of the core strategies).
    positions : numpy.ndarray
        Row position of each date in the returns index.
    columns : tuple of str
        All symbols (column order of the returned frames).
    symbols : tuple of tuple of str
        Universe at each date.
    features : tuple of numpy.ndarray
        Cross-sectionally ranked features per date, ``n_t x 4`` in :data:`FEATURES` order.
    forward : tuple of numpy.ndarray
        Unranked next-``horizon_days`` log return per name (NaN if incomplete). Used only through
        :func:`training_indices` and for evaluation.
    """

    dates: pd.DatetimeIndex
    positions: np.ndarray
    columns: tuple[str, ...]
    symbols: tuple[tuple[str, ...], ...]
    features: tuple[np.ndarray, ...]
    forward: tuple[np.ndarray, ...]


def build_panel(
    log_returns: pd.DataFrame,
    dollar_volume: pd.DataFrame,
    cfg: Config,
    end_date: str | pd.Timestamp | None = None,
) -> MLPanel:
    """Features and forward returns of the point-in-time universe at every rebalance date.

    Parameters
    ----------
    log_returns, dollar_volume : pandas.DataFrame
        Date x Symbol frames with identical index and columns.
    cfg : Config
        Uses ``universe``, ``backtest.rebalance_days`` and ``ml``.
    end_date : str or Timestamp, optional
        If given, only rebalance dates up to and including it are used.

    Raises
    ------
    ValueError
        If a feature window is longer than ``universe.lookback_days`` (names are only guaranteed
        to have returns over that window) or no rebalance date remains.
    """
    n, t, w = cfg.universe.n_assets, cfg.universe.lookback_days, cfg.universe.dollar_volume_window
    m = cfg.ml
    if max(m.momentum_days, m.reversal_days, m.vol_days) > t:
        raise ValueError(f"ml feature windows exceed universe.lookback_days={t}")
    dates = rebalance_dates(log_returns, t, n, cfg.backtest.rebalance_days)
    if end_date is not None:
        dates = dates[dates <= pd.Timestamp(end_date)]
    if len(dates) == 0:
        raise ValueError("no rebalance dates on or before end_date")
    positions = np.array([log_returns.index.get_loc(d) for d in dates])
    symbols, features, forward = [], [], []
    for date in dates:
        universe = select_universe(log_returns, dollar_volume, date, t, n, w)
        raw = raw_features(log_returns, dollar_volume, date, universe, cfg)
        symbols.append(tuple(universe))
        features.append(np.column_stack([rank_scores(raw[:, j]) for j in range(raw.shape[1])]))
        forward.append(forward_returns(log_returns, date, universe, m.horizon_days))
    return MLPanel(
        dates=pd.DatetimeIndex(dates, name="Date"),
        positions=positions,
        columns=tuple(log_returns.columns),
        symbols=tuple(symbols),
        features=tuple(features),
        forward=tuple(forward),
    )


def training_indices(positions: np.ndarray, k: int, purge_days: int) -> np.ndarray:
    """Indices ``j < k`` whose label window ends at least ``purge_days`` rows before date ``k``.

    The label of date ``j`` covers rows ``positions[j] + 1`` to ``positions[j] + horizon``; with
    ``purge_days >= horizon`` it therefore ends at or before row ``positions[k]``.
    """
    earlier = np.arange(k)
    return earlier[positions[earlier] + purge_days <= positions[k]]


def _training_data(panel: MLPanel, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Stacked features, ranked labels and the number of dates used (labels not all missing)."""
    xs, ys, used = [], [], 0
    for j in indices:
        label = panel.forward[j]
        ok = np.isfinite(label)
        if ok.sum() < _MIN_CROSS_SECTION:
            continue
        xs.append(panel.features[j][ok])
        ys.append(rank_scores(label[ok]))
        used += 1
    if not xs:
        raise ValueError("no training dates with labels")
    return np.vstack(xs), np.concatenate(ys), used


def _fit(model: str, x: np.ndarray, y: np.ndarray, cfg: Config) -> Ridge | RandomForestRegressor:
    m = cfg.ml
    if model == "RIDGE":
        return Ridge(alpha=m.ridge_alpha * len(y)).fit(x, y)
    if model == "RF":
        return RandomForestRegressor(
            n_estimators=m.rf_trees,
            max_depth=m.rf_max_depth,
            min_samples_leaf=m.rf_min_leaf,
            max_features=m.rf_max_features,
            random_state=cfg.run.seed,
            n_jobs=1,
        ).fit(x, y)
    raise ValueError(f"unknown model {model!r}; available: {MODELS}")


@dataclass(frozen=True)
class SignalResult:
    """Output of :func:`walk_forward_signals`.

    Attributes
    ----------
    predictions : dict of str to pandas.DataFrame
        Per model: evaluation date x symbol predicted rank score (NaN outside the universe).
    ic : pandas.DataFrame
        Evaluation date x predictor (the models and the four single features) rank IC with the
        realized next-``horizon_days`` return; NaN where the return is not yet observable.
    fits : pandas.DataFrame
        One row per refit: ``date``, ``n_dates`` and ``n_samples`` of the training set.
    """

    predictions: dict[str, pd.DataFrame]
    ic: pd.DataFrame
    fits: pd.DataFrame


def walk_forward_signals(panel: MLPanel, cfg: Config) -> SignalResult:
    """Expanding-window, scheduled-refit predictions of both models and their rank IC.

    The first evaluation date is the first one with ``ml.min_train_dates`` purged training dates;
    models are refitted at that date and every ``ml.retrain_every`` rebalance dates after it.

    Raises
    ------
    ValueError
        If the panel is too short for ``ml.min_train_dates`` purged training dates.
    """
    m = cfg.ml
    n_dates = len(panel.dates)
    usable = np.array([np.isfinite(f).sum() >= _MIN_CROSS_SECTION for f in panel.forward])
    counts = [
        int(usable[training_indices(panel.positions, k, m.purge_days)].sum())
        for k in range(n_dates)
    ]
    first = next((k for k, c in enumerate(counts) if c >= m.min_train_dates), None)
    if first is None:
        raise ValueError(
            f"only {n_dates} rebalance dates: cannot get {m.min_train_dates} purged training dates"
        )
    index = panel.dates[first:]
    column = {s: i for i, s in enumerate(panel.columns)}
    arrays = {name: np.full((len(index), len(panel.columns)), np.nan) for name in MODELS}
    ic_rows: list[dict[str, float]] = []
    fit_rows: list[dict[str, object]] = []
    fitted: dict[str, Ridge | RandomForestRegressor] = {}

    for k in range(first, n_dates):
        if (k - first) % m.retrain_every == 0:
            x, y, used = _training_data(panel, training_indices(panel.positions, k, m.purge_days))
            fitted = {name: _fit(name, x, y, cfg) for name in MODELS}
            fit_rows.append({"date": panel.dates[k], "n_dates": used, "n_samples": len(y)})
            logger.info("refit at %s on %d dates (%d samples)", panel.dates[k].date(), used, len(y))
        cols = [column[s] for s in panel.symbols[k]]
        row: dict[str, float] = {}
        for name in MODELS:
            score = fitted[name].predict(panel.features[k])
            arrays[name][k - first, cols] = score
            row[name] = rank_ic(score, panel.forward[k])
        for j, feature in enumerate(FEATURES):
            row[feature] = rank_ic(panel.features[k][:, j], panel.forward[k])
        ic_rows.append(row)

    predictions = {
        name: pd.DataFrame(arrays[name], index=index, columns=list(panel.columns))
        for name in MODELS
    }
    return SignalResult(
        predictions=predictions,
        ic=pd.DataFrame(ic_rows, index=index),
        fits=pd.DataFrame(fit_rows),
    )


def ic_summary(ic: pd.DataFrame) -> pd.DataFrame:
    """Mean, standard deviation, t-statistic and hit rate of each predictor's rank IC series.

    Parameters
    ----------
    ic : pandas.DataFrame
        Date x predictor rank IC (NaN dates are dropped per predictor).

    Returns
    -------
    pandas.DataFrame
        Columns ``mean_ic``, ``std_ic``, ``t_stat`` (``mean / (std / sqrt(n))``), ``hit_rate``
        (share of dates with positive IC) and ``n_dates``.
    """
    rows = {}
    for name in ic.columns:
        x = ic[name].dropna().to_numpy(dtype=float)
        n = len(x)
        mean = float(x.mean()) if n else float("nan")
        std = float(x.std(ddof=1)) if n > 1 else float("nan")
        t_stat = mean / (std / np.sqrt(n)) if n > 1 and std > 0 else float("nan")
        rows[name] = {
            "mean_ic": mean,
            "std_ic": std,
            "t_stat": t_stat,
            "hit_rate": float((x > 0).mean()) if n else float("nan"),
            "n_dates": n,
        }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("predictor")


def ic_gate(summary: pd.DataFrame, models: tuple[str, ...], t_threshold: float) -> dict[str, bool]:
    """Whether each model's mean rank IC is positive with t-statistic at least the threshold."""
    return {
        name: bool(summary.loc[name, "mean_ic"] > 0 and summary.loc[name, "t_stat"] >= t_threshold)
        for name in models
    }
