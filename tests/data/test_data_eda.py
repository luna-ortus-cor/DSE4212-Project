"""Tests for the EDA computations on small synthetic data."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from pfopt.data.eda import (
    get_window,
    index_realized_vol,
    leaf_order,
    pooled_standardized,
    pooled_summary,
    return_summary,
    rolling_avg_correlation,
    spectrum_stats,
    spectrum_timeseries,
)
from pfopt.data.universe import select_universe
from pfopt.estimators.utils import mp_edges


def factor_frame(n_dates: int, n_syms: int, seed: int, loading: float = 1.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    factor = rng.normal(size=(n_dates, 1))
    values = rng.normal(size=(n_dates, n_syms)) + loading * factor
    cols = [f"S{i}" for i in range(n_syms)]
    return pd.DataFrame(values * 0.01, index=dates, columns=cols)


def test_get_window_shape_and_errors() -> None:
    ret = factor_frame(20, 4, 0)
    w = get_window(ret, ret.index[9], 5, ["S0", "S2"])
    assert w.shape == (5, 2)
    assert w.index[-1] == ret.index[9] and w.index[0] == ret.index[5]
    with pytest.raises(ValueError, match="not a date"):
        get_window(ret, "2019-01-01", 5, ["S0"])
    with pytest.raises(ValueError, match="fewer than"):
        get_window(ret, ret.index[2], 5, ["S0"])
    ret.iloc[7, 1] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        get_window(ret, ret.index[9], 5, ["S1"])


def test_return_summary_matches_manual() -> None:
    window = pd.DataFrame({"A": [0.01, -0.01, 0.02, 0.0], "B": [0.01, 0.01, -0.01, -0.01]})
    out = return_summary(window, annualization=252)
    assert out.loc["A", "ann_mean"] == pytest.approx(0.005 * 252)
    assert out.loc["A", "ann_vol"] == pytest.approx(window["A"].std(ddof=1) * np.sqrt(252))
    assert out.loc["A", "min"] == -0.01 and out.loc["A", "max"] == 0.02
    assert out.loc["B", "skew"] == pytest.approx(0.0, abs=1e-12)
    assert list(out.columns) == ["ann_mean", "ann_vol", "skew", "excess_kurtosis", "min", "max"]


def test_pooled_standardized_columns_are_standard() -> None:
    window = factor_frame(50, 3, 1)
    z = pooled_standardized(window)
    assert z.shape == (150,)
    zz = z.reshape(50, 3)
    np.testing.assert_allclose(zz.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(zz.std(axis=0, ddof=1), 1.0, atol=1e-12)


def test_pooled_summary_detects_fat_tails() -> None:
    rng = np.random.default_rng(0)
    gaussian = pd.DataFrame(rng.normal(size=(20_000, 5)))
    heavy = pd.DataFrame(rng.standard_t(df=3, size=(20_000, 5)))
    g, h = pooled_summary(gaussian), pooled_summary(heavy)
    assert g["excess_kurtosis"] == pytest.approx(0.0, abs=0.2)
    assert h["excess_kurtosis"] > 3.0
    assert g["normal_frac_beyond_tail"] == pytest.approx(2 * stats.norm.sf(4.0))
    assert h["frac_beyond_tail"] > 10 * h["normal_frac_beyond_tail"]
    assert g["n_obs"] == 100_000
    assert g["min"] < 0 < g["max"]


def test_leaf_order_groups_correlated_blocks() -> None:
    block = np.repeat([0, 1], 3)
    corr = np.where(block[:, None] == block[None, :], 0.9, 0.0)
    np.fill_diagonal(corr, 1.0)
    perm = np.array([0, 3, 1, 4, 2, 5])
    order = leaf_order(corr[np.ix_(perm, perm)])
    assert sorted(order) == list(range(6))
    labels = block[perm][order]
    assert np.count_nonzero(np.diff(labels)) == 1


def test_spectrum_stats_planted_factor() -> None:
    window = factor_frame(504, 100, 2, loading=1.0).to_numpy()
    s = spectrum_stats(window)
    eig = s["eigenvalues"]
    assert s["q"] == pytest.approx(100 / 504)
    assert s["lambda_plus"] == pytest.approx(mp_edges(100 / 504)[1])
    assert s["n_signal"] >= 1
    assert eig.sum() == pytest.approx(100.0, abs=1e-8)
    assert s["top_share"] == pytest.approx(eig[0] / 100)
    c = np.corrcoef(window, rowvar=False)
    assert s["avg_corr"] == pytest.approx((c.sum() - 100) / (100 * 99))
    assert s["avg_corr"] > 0.3


def test_spectrum_timeseries_matches_pointwise_and_has_no_lookahead() -> None:
    ret = factor_frame(90, 8, 3)
    rng = np.random.default_rng(4)
    dv = pd.DataFrame(rng.uniform(1.0, 2.0, size=ret.shape), index=ret.index, columns=ret.columns)
    dates = pd.DatetimeIndex([ret.index[40], ret.index[60]])
    ts = spectrum_timeseries(ret, dv, dates, 30, 5, 10)
    assert list(ts.index) == list(dates)
    assert list(ts.columns) == ["n_signal", "lambda_max", "lambda_plus", "top_share", "avg_corr"]
    universe = select_universe(ret, dv, dates[1], 30, 5, 10)
    s = spectrum_stats(get_window(ret, dates[1], 30, universe).to_numpy())
    assert ts.loc[dates[1], "lambda_max"] == pytest.approx(s["lambda_max"])
    assert ts.loc[dates[1], "avg_corr"] == pytest.approx(s["avg_corr"])
    ret2, dv2 = ret.copy(), dv.copy()
    ret2.iloc[41:] = rng.normal(size=ret2.iloc[41:].shape)
    dv2.iloc[41:] = 9.0
    ts2 = spectrum_timeseries(ret2, dv2, dates[:1], 30, 5, 10)
    pd.testing.assert_frame_equal(ts2, ts.iloc[:1])


def test_rolling_avg_correlation_known_cases() -> None:
    x = np.random.default_rng(0).normal(size=40)
    same = pd.DataFrame({"a": x, "b": 2 * x, "c": 3 * x})
    out = rolling_avg_correlation(same, 10)
    assert out.iloc[:9].isna().all()
    np.testing.assert_allclose(out.iloc[9:], 1.0, atol=1e-12)
    opposite = pd.DataFrame({"a": x, "b": -x})
    np.testing.assert_allclose(rolling_avg_correlation(opposite, 10).iloc[9:], -1.0, atol=1e-12)
    with pytest.raises(ValueError, match="window"):
        rolling_avg_correlation(same, 1)


def test_rolling_avg_correlation_handles_nan_columns_and_matches_numpy() -> None:
    rng = np.random.default_rng(1)
    frame = pd.DataFrame(rng.normal(size=(50, 6)), columns=list("abcdef"))
    frame.iloc[30:33, 2] = np.nan
    frame.iloc[:20, 4] = np.nan
    out = rolling_avg_correlation(frame, 15)
    block = frame.iloc[40 - 15 + 1 : 41]
    ok = block.dropna(axis=1)
    c = np.corrcoef(ok.to_numpy(), rowvar=False)
    n = c.shape[0]
    assert out.iloc[40] == pytest.approx((c.sum() - n) / (n * (n - 1)))
    assert out.iloc[30] == pytest.approx(_avg_offdiag(frame.iloc[16:31].dropna(axis=1).to_numpy()))
    single = pd.DataFrame({"a": rng.normal(size=20), "b": np.nan})
    assert rolling_avg_correlation(single, 5).isna().all()


def _avg_offdiag(values: np.ndarray) -> float:
    c = np.corrcoef(values, rowvar=False)
    n = c.shape[0]
    return float((c.sum() - n) / (n * (n - 1)))


def test_rolling_avg_correlation_independent_noise_near_zero() -> None:
    frame = pd.DataFrame(np.random.default_rng(5).normal(size=(200, 100)))
    out = rolling_avg_correlation(frame, 60).dropna()
    assert abs(out.mean()) < 0.02


def test_rolling_avg_correlation_uses_only_past() -> None:
    frame = factor_frame(60, 5, 6)
    base = rolling_avg_correlation(frame, 15)
    changed = frame.copy()
    changed.iloc[41:] = 5.0 * np.random.default_rng(7).normal(size=changed.iloc[41:].shape)
    np.testing.assert_allclose(rolling_avg_correlation(changed, 15).iloc[:41], base.iloc[:41])


def test_index_realized_vol_known_value() -> None:
    returns = np.array([0.01, -0.01] * 5)
    levels = 100.0 * np.exp(np.concatenate([[0.0], np.cumsum(returns)]))
    dates = pd.bdate_range("2020-01-01", periods=len(levels))
    index = pd.DataFrame({"Date": dates, "S&P500": levels})
    vol = index_realized_vol(index, window=4, annualization=252)
    assert vol.name == "index_realized_vol"
    assert vol.iloc[:4].isna().all()
    expected = 0.01 * np.sqrt(4 / 3) * np.sqrt(252)
    np.testing.assert_allclose(vol.iloc[4:], expected, rtol=1e-9)


def test_index_realized_vol_requires_single_level_column() -> None:
    index = pd.DataFrame({"Date": pd.bdate_range("2020-01-01", periods=3), "a": 1.0, "b": 2.0})
    with pytest.raises(ValueError, match="exactly one level column"):
        index_realized_vol(index, 2, 252)
