"""EDA figures (PLAN.md step 1.4). Every function returns a matplotlib ``Figure``; saving is done
with :func:`pfopt.viz.style.save_figure`. All figures use the shared style in ``style.py``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy import stats

from pfopt.estimators.utils import mp_density
from pfopt.viz.style import ESTIMATOR_COLORS, FIG_WIDTH_IN, METHOD_COLORS, PALETTE, apply_style

_HIST_BINS = 80
_MP_BINS = 30
_MP_GRID = 2000
_EIGENVALUE_FLOOR = 1e-18
_MARKERS = {
    "EW": "o",
    "GMV-SAMPLE": "s",
    "GMV-LW": "^",
    "GMV-RMT": "v",
    "GMV-PCA": "D",
    "HRP": "P",
    "HRP-RMT": "X",
    "MV-SAMPLE": "<",
    "MV-RMT": ">",
    "SPX": "*",
}


def _strategy_style(name: str) -> dict[str, Any]:
    """Colour and line style of a strategy (SPX dashed grey, others solid)."""
    return {"color": METHOD_COLORS[name], "ls": "--" if name == "SPX" else "-"}


def plot_equity_curves(wealth: pd.DataFrame, ylabel: str = "Wealth (start = 1)") -> Figure:
    """Cumulative wealth of each column on a logarithmic axis.

    Parameters
    ----------
    wealth : pandas.DataFrame
        Date x strategy wealth indices (columns are keys of ``METHOD_COLORS``).
    ylabel : str
        Y-axis label (state whether returns are net of costs).
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, 3.6))
    for name in wealth.columns:
        ax.plot(wealth.index, wealth[name], label=name, lw=1.0, **_strategy_style(name))
    ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", ncol=2, fontsize=6.5)
    return fig


def plot_drawdowns(drawdowns: pd.DataFrame) -> Figure:
    """Drawdown (percent below the running peak) of each column.

    Parameters
    ----------
    drawdowns : pandas.DataFrame
        Date x strategy drawdowns (zero or negative fractions).
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, 3.2))
    for name in drawdowns.columns:
        ax.plot(drawdowns.index, 100 * drawdowns[name], label=name, lw=0.8, **_strategy_style(name))
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("Date")
    ax.legend(loc="lower left", ncol=3, fontsize=6.5)
    return fig


def plot_rolling_sharpe_vol(sharpe: pd.DataFrame, vol: pd.DataFrame, window: int) -> Figure:
    """Rolling Sharpe ratio (top) and annualized volatility (bottom) of each strategy.

    Parameters
    ----------
    sharpe, vol : pandas.DataFrame
        Date x strategy rolling statistics (NaN before the first full window).
    window : int
        Window length in trading days, used for the labels only.
    """
    apply_style()
    fig, axes = plt.subplots(2, 1, figsize=(FIG_WIDTH_IN, 5.0), sharex=True)
    for name in sharpe.columns:
        axes[0].plot(sharpe.index, sharpe[name], label=name, lw=0.8, **_strategy_style(name))
        axes[1].plot(vol.index, 100 * vol[name], lw=0.8, **_strategy_style(name))
    axes[0].axhline(0.0, color=PALETTE["grey"], lw=0.5)
    axes[0].set_ylabel(f"Sharpe ratio\n({window}-day window)")
    axes[1].set_ylabel(f"Volatility (%, annualized)\n({window}-day window)")
    axes[1].set_xlabel("Date")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5, fontsize=6.5)
    return fig


def plot_turnover_effective_n(annual_turnover: pd.Series, effective_n: pd.Series) -> Figure:
    """Bar charts of annual turnover (left) and mean effective number of holdings (right).

    Parameters
    ----------
    annual_turnover : pandas.Series
        ``sum |dw|`` per year by strategy (keys of ``METHOD_COLORS``).
    effective_n : pandas.Series
        Mean ``1 / sum(w^2)`` by strategy, same index.
    """
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH_IN, 3.0))
    for ax, series, label in (
        (axes[0], annual_turnover, "Annual turnover (sum of |dw|)"),
        (axes[1], effective_n, "Mean effective number of holdings"),
    ):
        colors = [METHOD_COLORS[name] for name in series.index]
        ax.bar(range(len(series)), series.to_numpy(), color=colors)
        ax.set_xticks(range(len(series)))
        ax.set_xticklabels(series.index, rotation=60, ha="right", fontsize=7)
        ax.set_ylabel(label)
    return fig


def plot_risk_return(annual_vol: pd.Series, annual_return: pd.Series) -> Figure:
    """Scatter of annualized volatility against annualized return, one marker per strategy.

    Points cluster, so strategies are identified by colour and marker shape in a legend placed
    outside the axes instead of by text labels next to the points.

    Parameters
    ----------
    annual_vol, annual_return : pandas.Series
        Fractions (not percent) by strategy (keys of ``METHOD_COLORS``).
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, 3.4))
    for name in annual_vol.index:
        ax.scatter(
            100 * annual_vol[name],
            100 * annual_return[name],
            color=METHOD_COLORS[name],
            marker=_MARKERS.get(name, "o"),
            s=40,
            zorder=3,
            label=name,
        )
    ax.set_xlabel("Annualized volatility (%)")
    ax.set_ylabel("Annualized return (%)")
    fig.legend(loc="outside right center", fontsize=7)
    return fig


def plot_vol_forecast_ratio(
    ratios: Mapping[str, pd.Series], rolling_window: int = 12
) -> Figure:
    """Realized-to-predicted volatility per holding period, with a rolling mean.

    A ratio above 1 means the strategy's risk was under-predicted. Dots are single 21-day periods
    (noisy); lines are rolling means over ``rolling_window`` periods.

    Parameters
    ----------
    ratios : mapping of str to pandas.Series
        Per-period ratio (indexed by rebalance date) by strategy.
    rolling_window : int, default 12
        Number of periods in the rolling mean.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, 3.4))
    for name, series in ratios.items():
        color = METHOD_COLORS[name]
        ax.plot(series.index, series, ls="none", marker=".", ms=2.5, color=color, alpha=0.35)
        ax.plot(series.index, series.rolling(rolling_window).mean(), color=color, lw=1.2, label=name)
    ax.axhline(1.0, color=PALETTE["black"], lw=0.7)
    ax.set_yscale("log")
    ax.set_ylabel("Realized / predicted volatility")
    ax.set_xlabel("Rebalance date")
    ax.legend(loc="upper left", ncol=2, fontsize=6.5)
    return fig


def plot_period_bars(sharpe: pd.DataFrame, max_drawdown: pd.DataFrame) -> Figure:
    """Grouped bars of the Sharpe ratio (top) and maximum drawdown (bottom) by period.

    Parameters
    ----------
    sharpe, max_drawdown : pandas.DataFrame
        Period x strategy tables (rows are the groups on the x-axis, columns are strategies and
        keys of ``METHOD_COLORS``). Drawdowns are negative fractions.
    """
    apply_style()
    fig, axes = plt.subplots(2, 1, figsize=(FIG_WIDTH_IN, 5.4), sharex=True)
    strategies = list(sharpe.columns)
    x = np.arange(len(sharpe.index))
    width = 0.8 / len(strategies)
    for j, name in enumerate(strategies):
        offset = (j - (len(strategies) - 1) / 2.0) * width
        axes[0].bar(x + offset, sharpe[name], width, color=METHOD_COLORS[name], label=name)
        axes[1].bar(x + offset, 100 * max_drawdown[name], width, color=METHOD_COLORS[name])
    axes[0].axhline(0.0, color=PALETTE["black"], lw=0.6)
    axes[0].set_ylabel("Sharpe ratio (net)")
    axes[1].set_ylabel("Maximum drawdown (%)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(list(sharpe.index), fontsize=7)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5, fontsize=6.5)
    return fig


def plot_frontiers(panels: Sequence[tuple[str, Mapping[str, Any]]]) -> Figure:
    """Estimated (solid) versus realized (dashed) efficient frontiers, one panel per date.

    Stars mark the maximum-Sharpe portfolio: filled for the estimated position, hollow for the
    realized one. Axes are annualized volatility and return in percent.

    Parameters
    ----------
    panels : sequence of (title, curves)
        ``curves`` maps an estimator name (a key of ``ESTIMATOR_COLORS``) to an object with
        attributes ``est_vol``, ``est_ret``, ``real_vol``, ``real_ret`` (arrays) and ``ms_est``,
        ``ms_real`` (``(vol, ret)`` pairs), such as
        :class:`pfopt.backtest.frontier.FrontierCurve`.
    """
    apply_style()
    fig, axes = plt.subplots(1, len(panels), figsize=(FIG_WIDTH_IN, 3.0))
    axes = np.atleast_1d(axes)
    for ax, (title, curves) in zip(axes, panels, strict=True):
        for name, curve in curves.items():
            color = ESTIMATOR_COLORS[name]
            ax.plot(100 * curve.est_vol, 100 * curve.est_ret, color=color, label=f"{name}, estimated")
            ax.plot(
                100 * curve.real_vol,
                100 * curve.real_ret,
                color=color,
                ls="--",
                marker="o",
                ms=2.0,
                lw=0.9,
                label=f"{name}, realized",
            )
            ax.plot(100 * curve.ms_est[0], 100 * curve.ms_est[1], "*", ms=9, color=color)
            ax.plot(
                100 * curve.ms_real[0],
                100 * curve.ms_real[1],
                "*",
                ms=9,
                color=color,
                markerfacecolor="white",
            )
        ax.axhline(0.0, color=PALETTE["grey"], lw=0.6)
        ax.set_title(title, fontsize=8)
    axes[0].set_ylabel("Annualized return (%)")
    axes[0].legend(loc="upper left", fontsize=6)
    fig.supxlabel("Annualized volatility (%)", fontsize=9)
    return fig


def plot_cov_spectra(panels: Sequence[tuple[str, dict[str, np.ndarray]]]) -> Figure:
    """Eigenvalues (descending, log scale) of each covariance estimator, one panel per date.

    Parameters
    ----------
    panels : sequence of (title, spectra)
        ``spectra`` maps an estimator name (a key of ``ESTIMATOR_COLORS``) to its eigenvalues in
        descending order.
    """
    apply_style()
    fig, axes = plt.subplots(1, len(panels), figsize=(FIG_WIDTH_IN, 2.7))
    axes = np.atleast_1d(axes)
    for ax, (title, spectra) in zip(axes, panels, strict=True):
        for name, eigenvalues in spectra.items():
            values = np.maximum(np.asarray(eigenvalues, dtype=float), _EIGENVALUE_FLOOR)
            ranks = np.arange(1, values.size + 1)
            ax.plot(ranks, values, color=ESTIMATOR_COLORS[name], label=name)
        ax.set_yscale("log")
        ax.set_title(title, fontsize=9)
    axes[0].set_ylabel("Eigenvalue of annualized\ncovariance (log scale)")
    axes[0].legend(loc="lower left")
    fig.supxlabel("Eigenvalue rank", fontsize=9)
    return fig


def plot_sorted_weights(
    panels: Sequence[tuple[str, dict[str, np.ndarray]]], strategies: Sequence[str]
) -> Figure:
    """Sorted weights (largest first) of each strategy; rows are strategies, columns are dates.

    All panels share the y-axis so that concentration is directly comparable. Each panel is
    annotated with the effective number of holdings ``1 / sum(w^2)``.

    Parameters
    ----------
    panels : sequence of (title, weights)
        ``weights`` maps a strategy id to its weight vector.
    strategies : sequence of str
        Strategy ids (keys of ``METHOD_COLORS``), one row each.
    """
    apply_style()
    rows, cols = len(strategies), len(panels)
    size = (FIG_WIDTH_IN, 0.95 * rows + 0.9)
    fig, axes = plt.subplots(rows, cols, figsize=size, sharex=True, sharey=True, squeeze=False)
    y_max = 1.05 * max(float(np.max(w)) for _, ws in panels for w in ws.values())
    for j, (title, weights) in enumerate(panels):
        for i, strategy in enumerate(strategies):
            w = np.sort(np.asarray(weights[strategy], dtype=float))[::-1]
            ax = axes[i][j]
            ax.bar(np.arange(1, w.size + 1), w, width=1.0, color=METHOD_COLORS[strategy])
            ax.set_ylim(0.0, y_max)
            label = f"eff. N = {1.0 / np.sum(w**2):.0f}"
            ax.text(0.97, 0.8, label, transform=ax.transAxes, ha="right", fontsize=7)
            if j == 0:
                ax.set_ylabel(strategy, rotation=0, ha="right", va="center", fontsize=7)
            if i == 0:
                ax.set_title(title, fontsize=9)
    fig.supxlabel("Rank of holding (largest first)", fontsize=9)
    return fig


def plot_return_histograms(panels: Sequence[tuple[str, np.ndarray]]) -> Figure:
    """Pooled standardized daily returns against a standard normal density (log y-axis).

    Parameters
    ----------
    panels : sequence of (title, z)
        One panel per snapshot; ``z`` are pooled per-stock standardized returns.
    """
    apply_style()
    fig, axes = plt.subplots(1, len(panels), figsize=(FIG_WIDTH_IN, 2.5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (title, z) in zip(axes, panels, strict=True):
        zmax = float(np.ceil(np.abs(z).max()))
        bins = np.linspace(-zmax, zmax, _HIST_BINS + 1)
        ax.hist(z, bins=bins, density=True, color=PALETTE["blue"], alpha=0.8, label="Returns")
        grid = np.linspace(-zmax, zmax, 400)
        ax.plot(grid, stats.norm.pdf(grid), color=PALETTE["vermillion"], label="Normal")
        ax.set_yscale("log")
        ax.set_ylim(0.5 / (z.size * (bins[1] - bins[0])), 1.0)
        ax.set_title(title, fontsize=9)
    axes[0].set_ylabel("Density (log scale)")
    axes[0].legend(loc="upper right")
    fig.supxlabel("Standardized daily log return (std. dev.)", fontsize=9)
    return fig


def plot_corr_heatmaps(panels: Sequence[tuple[str, np.ndarray]]) -> Figure:
    """Correlation heatmaps; matrices must already be in hierarchical-clustering leaf order.

    Parameters
    ----------
    panels : sequence of (title, corr)
        One ``N x N`` ordered correlation matrix per snapshot.
    """
    apply_style()
    fig, axes = plt.subplots(1, len(panels), figsize=(FIG_WIDTH_IN, 2.6))
    axes = np.atleast_1d(axes)
    vmin = min(0.0, min(float(m.min()) for _, m in panels))
    image = None
    for ax, (title, corr) in zip(axes, panels, strict=True):
        image = ax.imshow(corr, cmap="viridis", vmin=vmin, vmax=1.0, interpolation="nearest")
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    fig.supxlabel("Stocks (hierarchical-cluster order)", fontsize=9)
    fig.colorbar(image, ax=list(axes), shrink=0.8, label="Correlation")
    return fig


def plot_mp_spectra(panels: Sequence[tuple[str, dict]]) -> Figure:
    """Eigenvalue spectrum of the correlation matrix against the Marchenko-Pastur density.

    Both axes are logarithmic so that the noise bulk and the outlying "market mode" are visible.
    The histogram is a density (count / (N * bin width)).

    Parameters
    ----------
    panels : sequence of (title, stats)
        ``stats`` is the dict from :func:`pfopt.data.eda.spectrum_stats`.
    """
    apply_style()
    fig, axes = plt.subplots(1, len(panels), figsize=(FIG_WIDTH_IN, 2.9), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (title, s) in zip(axes, panels, strict=True):
        eig = np.asarray(s["eigenvalues"], dtype=float)
        edges = np.geomspace(eig.min() * 0.9, eig.max() * 1.1, _MP_BINS + 1)
        counts, _ = np.histogram(eig, bins=edges)
        dens = counts / (eig.size * np.diff(edges))
        ax.bar(
            edges[:-1],
            dens,
            width=np.diff(edges),
            align="edge",
            color=PALETTE["blue"],
            alpha=0.8,
            label="Empirical",
        )
        xs = np.geomspace(edges[0], edges[-1], _MP_GRID)
        mp = mp_density(xs, float(s["q"]))
        ax.plot(xs, mp, color=PALETTE["vermillion"], label="Marchenko-Pastur")
        edge = float(s["lambda_plus"])
        ax.axvline(edge, color=PALETTE["black"], ls="--", lw=0.9, label="MP edge")
        ax.set_xscale("log")
        ax.set_yscale("log")
        positive = dens[dens > 0]
        ax.set_ylim(0.3 * positive.min(), 1.5 * max(dens.max(), mp.max()))
        ax.set_title(
            f"{title}\n{s['n_signal']} above edge; max {float(s['lambda_max']):.1f}", fontsize=8
        )
    axes[0].set_ylabel("Density (log scale)")
    axes[0].legend(loc="lower left")
    fig.supxlabel("Eigenvalue (log scale)", fontsize=9)
    return fig


def plot_spectrum_timeseries(ts: pd.DataFrame, index_vol: pd.Series | None = None) -> Figure:
    """Signal count, market-mode share and average correlation over rebalance dates.

    Parameters
    ----------
    ts : pandas.DataFrame
        Output of :func:`pfopt.data.eda.spectrum_timeseries` (``n_signal``, ``top_share``,
        ``avg_corr`` columns).
    index_vol : pandas.Series, optional
        Annualized index realized volatility, drawn in an extra panel (clipped to the range of
        ``ts``). Omit if no index data is available.
    """
    apply_style()
    n_panels = 3 if index_vol is not None else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(FIG_WIDTH_IN, 1.9 * n_panels), sharex=True)
    axes[0].step(ts.index, ts["n_signal"], where="post", color=PALETTE["blue"])
    axes[0].set_ylabel("Eigenvalues above\nMP edge")
    axes[0].set_ylim(bottom=0)
    axes[1].plot(ts.index, ts["top_share"], color=PALETTE["orange"], label="Top eigenvalue share")
    axes[1].plot(ts.index, ts["avg_corr"], color=PALETTE["green"], label="Average correlation")
    axes[1].set_ylabel("Fraction")
    axes[1].legend(loc="upper left", ncol=2)
    if index_vol is not None:
        vol = index_vol.loc[ts.index[0] : ts.index[-1]].dropna()
        axes[2].plot(vol.index, vol.to_numpy(), color=PALETTE["black"])
        axes[2].set_ylabel("S&P 500 realized\nvol. (annualized)")
    axes[-1].set_xlabel("Date")
    return fig


def plot_rolling_correlation(
    avg_corr: pd.Series, highlight_years: Sequence[int], window: int
) -> Figure:
    """Rolling cross-sectional average correlation with highlighted calendar years.

    Parameters
    ----------
    avg_corr : pandas.Series
        Output of :func:`pfopt.data.eda.rolling_avg_correlation`.
    highlight_years : sequence of int
        Calendar years shaded in the background.
    window : int
        Window length, used only for the axis label.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, 2.8))
    series = avg_corr.dropna()
    ax.plot(series.index, series.to_numpy(), color=PALETTE["blue"], lw=0.9)
    shades = (PALETTE["orange"], PALETTE["vermillion"], PALETTE["green"], PALETTE["purple"])
    for i, year in enumerate(highlight_years):
        ax.axvspan(
            pd.Timestamp(f"{year}-01-01"),
            pd.Timestamp(f"{year}-12-31"),
            color=shades[i % len(shades)],
            alpha=0.2,
            label=str(year),
        )
    ax.set_xlabel("Date")
    ax.set_ylabel(f"Average pairwise correlation\n({window}-day window)")
    if highlight_years:
        ax.legend(loc="upper left", ncol=len(highlight_years))
    return fig
