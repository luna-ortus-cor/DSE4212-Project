"""Smoke tests for the shared style and EDA figures (headless backend)."""

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from pfopt.backtest.strategies import CORE_STRATEGIES, ESTIMATORS  # noqa: E402
from pfopt.data.eda import pooled_standardized, spectrum_stats  # noqa: E402
from pfopt.viz.plots import (  # noqa: E402
    plot_corr_heatmaps,
    plot_cov_spectra,
    plot_drawdowns,
    plot_eigen_vs_performance,
    plot_equity_curves,
    plot_frontiers,
    plot_h3_summary,
    plot_heatmap_panels,
    plot_mp_spectra,
    plot_period_bars,
    plot_return_histograms,
    plot_risk_return,
    plot_rolling_correlation,
    plot_rolling_sharpe_vol,
    plot_simulation_accuracy,
    plot_sorted_weights,
    plot_spectrum_timeseries,
    plot_turnover_effective_n,
    plot_vol_forecast_ratio,
    plot_weight_dynamics,
    plot_weight_survival,
)
from pfopt.viz.style import FIG_WIDTH_IN, METHOD_COLORS, PALETTE, save_figure  # noqa: E402


@pytest.fixture(scope="module")
def window() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-01", periods=200)
    values = rng.normal(size=(200, 20)) + rng.normal(size=(200, 1))
    return pd.DataFrame(values * 0.01, index=dates, columns=[f"S{i}" for i in range(20)])


def _assert_figure(fig: Figure, n_axes_min: int) -> None:
    assert isinstance(fig, Figure)
    assert fig.get_figwidth() <= FIG_WIDTH_IN + 1e-9
    assert len(fig.axes) >= n_axes_min
    plt.close(fig)


def test_palette_and_method_colors_are_valid_and_distinct() -> None:
    for color in list(PALETTE.values()) + list(METHOD_COLORS.values()):
        assert len(color) == 7 and color.startswith("#")
    assert len(set(METHOD_COLORS.values())) == len(METHOD_COLORS)


def test_return_histograms(window: pd.DataFrame) -> None:
    z = pooled_standardized(window)
    _assert_figure(plot_return_histograms([("A", z), ("B", z)]), 2)


def test_corr_heatmaps(window: pd.DataFrame) -> None:
    corr = np.corrcoef(window.to_numpy(), rowvar=False)
    _assert_figure(plot_corr_heatmaps([("A", corr), ("B", corr), ("C", corr)]), 3)


def test_mp_spectra(window: pd.DataFrame) -> None:
    s = spectrum_stats(window.to_numpy())
    _assert_figure(plot_mp_spectra([("A", s), ("B", s)]), 2)


def test_spectrum_timeseries_with_and_without_index_vol() -> None:
    dates = pd.bdate_range("2020-01-01", periods=12, freq="21B")
    ts = pd.DataFrame(
        {"n_signal": [3, 4] * 6, "top_share": 0.3, "avg_corr": 0.2, "lambda_max": 20.0},
        index=dates,
    )
    vol = pd.Series(0.15, index=pd.bdate_range("2019-12-01", periods=400))
    _assert_figure(plot_spectrum_timeseries(ts, vol), 3)
    _assert_figure(plot_spectrum_timeseries(ts, None), 2)


def test_rolling_correlation() -> None:
    idx = pd.bdate_range("2019-01-01", "2023-12-31")
    series = pd.Series(np.linspace(0.1, 0.5, len(idx)), index=idx)
    series.iloc[:20] = np.nan
    _assert_figure(plot_rolling_correlation(series, (2020, 2022), 21), 1)


def test_save_figure_writes_png_and_pdf(tmp_path: Path) -> None:
    fig = plot_rolling_correlation(
        pd.Series(0.3, index=pd.bdate_range("2020-01-01", periods=30)), (2020,), 21
    )
    paths = save_figure(fig, "eda_test", tmp_path / "figs")
    assert [p.name for p in paths] == ["eda_test.png", "eda_test.pdf"]
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)


def test_save_figure_rejects_too_wide_figures(tmp_path: Path) -> None:
    fig = plt.figure(figsize=(FIG_WIDTH_IN + 1.0, 2.0))
    with pytest.raises(ValueError, match="exceeds"):
        save_figure(fig, "too_wide", tmp_path)
    plt.close(fig)


def test_frontier_figure() -> None:
    @dataclass(frozen=True)
    class Curve:
        est_vol: np.ndarray
        est_ret: np.ndarray
        real_vol: np.ndarray
        real_ret: np.ndarray
        ms_est: tuple[float, float]
        ms_real: tuple[float, float]

    vol = np.linspace(0.1, 0.3, 10)
    curve = Curve(vol, 0.2 * vol + 0.05, vol * 1.1, 0.1 * vol, (0.15, 0.1), (0.16, 0.03))
    curves = {"Sample": curve, "RMT": curve}
    _assert_figure(plot_frontiers([("A", curves), ("B", curves), ("C", curves)]), 3)


def _wealth(names: list[str]) -> pd.DataFrame:
    index = pd.bdate_range("2020-01-01", periods=300)
    rng = np.random.default_rng(0)
    returns = pd.DataFrame(
        rng.normal(0.0004, 0.01, size=(300, len(names))), index=index, columns=names
    )
    return (1.0 + returns).cumprod()


ALL_NAMES = list(CORE_STRATEGIES) + ["MV-SAMPLE", "MV-RMT"]


def test_equity_and_drawdown_figures() -> None:
    wealth = _wealth(ALL_NAMES + ["SPX"])
    _assert_figure(plot_equity_curves(wealth, "Wealth (start = 1)"), 1)
    drawdowns = wealth / wealth.cummax() - 1.0
    _assert_figure(plot_drawdowns(drawdowns), 1)


def test_rolling_figure() -> None:
    wealth = _wealth(ALL_NAMES)
    returns = wealth.pct_change().dropna()
    rolling = returns.rolling(60)
    sharpe = rolling.mean() / rolling.std() * np.sqrt(252)
    _assert_figure(plot_rolling_sharpe_vol(sharpe, rolling.std() * np.sqrt(252), 60), 2)


def test_turnover_and_risk_return_figures() -> None:
    names = ALL_NAMES
    values = pd.Series(np.linspace(1.0, 3.0, len(names)), index=names)
    _assert_figure(plot_turnover_effective_n(values, values * 10), 2)
    _assert_figure(plot_risk_return(values / 20, values / 40), 1)


def test_vol_forecast_ratio_figure() -> None:
    index = pd.bdate_range("2012-01-03", periods=40, freq="21B")
    rng = np.random.default_rng(1)
    ratios = {
        s: pd.Series(np.exp(rng.normal(0.1, 0.2, 40)), index=index) for s in CORE_STRATEGIES[1:5]
    }
    _assert_figure(plot_vol_forecast_ratio(ratios), 1)


def test_period_bars_figure() -> None:
    periods = ["Full", "COVID", "2022", "Calm", "High vol", "Low vol"]
    rng = np.random.default_rng(2)
    sharpe = pd.DataFrame(rng.normal(0.8, 0.5, (6, 9)), index=periods, columns=ALL_NAMES)
    drawdown = pd.DataFrame(-rng.uniform(0.05, 0.4, (6, 9)), index=periods, columns=ALL_NAMES)
    _assert_figure(plot_period_bars(sharpe, drawdown), 2)


def test_cov_spectra() -> None:
    spectra = {
        name: np.sort(np.random.default_rng(i).uniform(0.1, 5.0, 20))[::-1]
        for i, name in enumerate(ESTIMATORS)
    }
    _assert_figure(plot_cov_spectra([("A", spectra), ("B", spectra)]), 2)


def test_sorted_weights_grid() -> None:
    rng = np.random.default_rng(0)
    weights = {s: rng.dirichlet(np.ones(15)) for s in CORE_STRATEGIES}
    panels = [("A", weights), ("B", weights), ("C", weights)]
    fig = plot_sorted_weights(panels, CORE_STRATEGIES)
    _assert_figure(fig, len(CORE_STRATEGIES) * 3)


def test_heatmap_panels_figure() -> None:
    rng = np.random.default_rng(4)
    table = pd.DataFrame(rng.uniform(0.1, 1.0, (9, 4)), index=ALL_NAMES, columns=list("abcd"))
    panels = [("vol", table, ".3f"), ("sharpe", table, ".2f"), ("turnover", table, ".1f")]
    _assert_figure(plot_heatmap_panels(panels), 3)


def test_diverging_heatmap_and_structure_figures() -> None:
    rng = np.random.default_rng(5)
    table = pd.DataFrame(rng.normal(0, 0.03, (9, 5)), index=ALL_NAMES, columns=list("abcde"))
    panels = [("mean", table.abs(), ".1f"), ("active", table, ".1f", "RdBu_r")]
    _assert_figure(plot_heatmap_panels(panels, ncols=1, xtick_rotation=30), 2)

    dates = pd.date_range("2013-01-01", periods=30, freq="W")
    eff = pd.DataFrame(rng.uniform(5, 50, (30, 9)), index=dates, columns=ALL_NAMES)
    turnover = pd.DataFrame(rng.uniform(0, 0.5, (30, 9)), index=dates, columns=ALL_NAMES)
    _assert_figure(plot_weight_dynamics(eff, turnover), 2)

    pooled = {name: rng.dirichlet(np.ones(100)) for name in ALL_NAMES}
    _assert_figure(plot_weight_survival(pooled), 1)

    diagnostics = pd.DataFrame(
        {"top_eig_share": rng.uniform(0.2, 0.6, 30), "rmt_n_signal": rng.integers(1, 5, 30)},
        index=dates,
    )
    performance = pd.DataFrame(
        {"logvol_GMV-RMT": rng.normal(0, 0.1, 30), "logvol_GMV-PCA": rng.normal(0, 0.1, 30)},
        index=dates,
    )
    _assert_figure(plot_eigen_vs_performance(diagnostics, performance, ("GMV-RMT", "GMV-PCA")), 2)


def test_h3_summary_figure() -> None:
    rng = np.random.default_rng(6)
    names = ["Sample", "Ledoit-Wolf", "RMT", "PCA"]
    vol = pd.DataFrame(
        {"Long-only": rng.uniform(12, 15, 4), "Unconstrained": rng.uniform(10, 20, 4)}, index=names
    )
    relative = pd.DataFrame(rng.normal(0, 5, (3, 2)), index=names[1:], columns=vol.columns)
    dates = pd.date_range("2013-01-01", periods=20, freq="W")
    leverage = pd.DataFrame(rng.uniform(1.5, 4, (20, 4)), index=dates, columns=names)
    _assert_figure(plot_h3_summary(vol, relative, leverage), 3)


def test_simulation_accuracy_figure() -> None:
    rng = np.random.default_rng(8)
    rows = []
    specs = (
        ("frob_error", "covariance", ["GMV-SAMPLE", "GMV-LW", "GMV-RMT", "GMV-PCA"]),
        ("l1_dist", "long-only", ["GMV-SAMPLE", "GMV-LW", "GMV-RMT", "GMV-PCA", "HRP", "EW"]),
        ("var_ratio", "long-only", ["GMV-SAMPLE", "GMV-LW", "GMV-RMT", "GMV-PCA", "HRP", "EW"]),
        ("var_ratio", "unconstrained", ["GMV-SAMPLE", "GMV-LW", "GMV-RMT", "GMV-PCA"]),
    )
    for metric, constraint, methods in specs:
        for method in methods:
            for t in (126, 252, 504):
                mean = rng.uniform(1.0, 2.0)
                rows.append(
                    {
                        "distribution": "gaussian",
                        "T": t,
                        "method": method,
                        "constraint": constraint,
                        "metric": metric,
                        "mean": mean,
                        "ci_low": 0.9 * mean,
                        "ci_high": 1.1 * mean,
                    }
                )
    _assert_figure(plot_simulation_accuracy(pd.DataFrame(rows), "gaussian"), 4)
