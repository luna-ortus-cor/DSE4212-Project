"""Tests for the weight-structure and spectrum-diagnostic helpers (step 5.4)."""

import numpy as np
import pandas as pd
import pytest

from pfopt.backtest import structure as st

DATES = pd.to_datetime(["2020-01-03", "2020-01-10", "2020-01-17"])
SYMBOLS = ["A", "B", "C", "D"]


def frame(rows: list[list[float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, index=DATES[: len(rows)], columns=SYMBOLS)


def test_herfindahl_and_effective_n_hand_computed() -> None:
    weights = frame([[0.25, 0.25, 0.25, 0.25], [0.5, 0.5, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    np.testing.assert_allclose(st.herfindahl_series(weights), [0.25, 0.5, 1.0])
    table = st.effective_n_table({"EW": weights})
    np.testing.assert_allclose(table["EW"], [4.0, 2.0, 1.0])


def test_concentration_table_equal_weight_and_concentrated() -> None:
    universe = pd.DataFrame(True, index=DATES[:2], columns=SYMBOLS)
    equal = frame([[0.25] * 4, [0.25] * 4])
    single = frame([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    table = st.concentration_table({"EW": equal, "X": single}, universe)
    assert table.loc["EW", "mean_effective_n"] == pytest.approx(4.0)
    assert table.loc["EW", "share_zero"] == 0.0 and table.loc["EW", "max"] == 0.25
    assert table.loc["X", "mean_hhi"] == pytest.approx(1.0)
    assert table.loc["X", "share_zero"] == pytest.approx(0.75)
    assert table.loc["X", "mean_max_weight"] == 1.0


def test_pooled_member_weights_drops_non_members() -> None:
    weights = frame([[0.5, 0.5, 0.0, 0.0], [0.0, 0.5, 0.5, 0.0]])
    universe = pd.DataFrame(
        [[True, True, True, False], [True, True, True, False]], index=weights.index, columns=SYMBOLS
    )
    pooled = st.pooled_member_weights(weights, universe)
    assert pooled.size == 6 and pooled.sum() == pytest.approx(2.0)


def test_sector_exposure_hand_computed_and_sums_to_one() -> None:
    weights = frame([[0.1, 0.2, 0.3, 0.4], [0.25, 0.25, 0.25, 0.25]])
    sectors = pd.Series({"A": "Tech", "B": "Tech", "C": "Energy"})
    exposure = st.sector_exposure(weights, sectors)
    assert set(exposure.columns) == {"Tech", "Energy", st.UNKNOWN_SECTOR}
    np.testing.assert_allclose(exposure["Tech"], [0.3, 0.5])
    np.testing.assert_allclose(exposure[st.UNKNOWN_SECTOR], [0.4, 0.25])
    np.testing.assert_allclose(exposure.sum(axis=1), 1.0)
    mean = st.mean_sector_table({"S": weights}, sectors)
    assert mean.loc["S", "Tech"] == pytest.approx(0.4)
    assert mean.loc["S"].sum() == pytest.approx(1.0)


def test_window_eigen_diagnostics_perfect_and_independent() -> None:
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2020-01-01", periods=60)
    common = rng.normal(size=(60, 1))
    scale = np.array([[1.0, 2.0, 3.0, 4.0]])
    perfect = pd.DataFrame(common * scale, index=days, columns=SYMBOLS)
    universe = pd.DataFrame(True, index=days[[-1]], columns=SYMBOLS)
    out = st.window_eigen_diagnostics(perfect, universe, 40, 0.0, 1.0)
    assert out.loc[days[-1], "top_eig_share"] == pytest.approx(1.0)
    assert out.loc[days[-1], "mean_corr"] == pytest.approx(1.0)

    independent = pd.DataFrame(rng.normal(size=(2000, 4)), columns=SYMBOLS)
    independent.index = pd.bdate_range("2010-01-01", periods=2000)
    universe = pd.DataFrame(True, index=independent.index[[-1]], columns=SYMBOLS)
    out = st.window_eigen_diagnostics(independent, universe, 2000, 0.0, 1.0)
    assert abs(out["mean_corr"].iloc[0]) < 0.06
    assert out["top_eig_share"].iloc[0] < 0.35


def test_relative_performance_hand_computed() -> None:
    days = pd.bdate_range("2020-01-06", periods=8)
    base = pd.Series([0.01, -0.01] * 4, index=days)
    net = pd.DataFrame({"base": base, "half": 0.5 * base, "same": base})
    dates = pd.DatetimeIndex([days[0] - pd.Timedelta(days=3), days[3], days[7]])
    out = st.relative_performance(net, dates, "base", ["half", "same"], annualization=252)
    assert len(out) == 2
    np.testing.assert_allclose(out["logvol_half"], np.log(0.5))
    np.testing.assert_allclose(out["logvol_same"], 0.0, atol=1e-12)
    np.testing.assert_allclose(out["ret_same"], 0.0, atol=1e-12)
    first = base.iloc[:4].mean() * 252
    assert out["ret_half"].iloc[0] == pytest.approx(0.5 * first - first)


def test_diagnostic_correlations_monotone_relations() -> None:
    index = pd.date_range("2020-01-01", periods=30, freq="W")
    x = np.arange(30.0)
    diagnostics = pd.DataFrame({"d": x}, index=index)
    performance = pd.DataFrame({"up": x**3, "down": -x}, index=index)
    out = st.diagnostic_correlations(diagnostics, performance).set_index("target")
    assert out.loc["up", "spearman_rho"] == pytest.approx(1.0)
    assert out.loc["down", "spearman_rho"] == pytest.approx(-1.0)
    assert (out["n"] == 30).all()
