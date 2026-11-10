"""EDA figures. Every function returns a matplotlib ``Figure``; saving is done
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


def plot_vol_forecast_ratio(ratios: Mapping[str, pd.Series], rolling_window: int = 12) -> Figure:
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
        ax.plot(
            series.index, series.rolling(rolling_window).mean(), color=color, lw=1.2, label=name
        )
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
            ax.plot(
                100 * curve.est_vol, 100 * curve.est_ret, color=color, label=f"{name}, estimated"
            )
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
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=7)
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


def plot_heatmap_panels(
    panels: Sequence[tuple[str, pd.DataFrame, str] | tuple[str, pd.DataFrame, str, str]],
    ncols: int = 2,
    panel_height: float = 2.6,
    xtick_rotation: float = 0.0,
) -> Figure:
    """Annotated heatmaps (rows = strategies, columns = scenario variants), one per panel.

    Parameters
    ----------
    panels : sequence of (title, table, number format[, colormap])
        ``table`` is strategy x variant; ``number format`` is a format spec such as ``".2f"``.
        Each panel has its own colour scale (default viridis). For a non-default colormap whose
        values straddle zero (for example ``"RdBu_r"``) the scale is symmetric around zero.
    ncols : int, default 2
        Panels per row.
    panel_height : float, default 2.6
        Height in inches of one row of panels.
    xtick_rotation : float, default 0
        Rotation (degrees) of the column labels.
    """
    apply_style()
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(FIG_WIDTH_IN, panel_height * nrows),
        squeeze=False,
        layout="constrained",
    )
    for ax in axes.ravel()[len(panels) :]:
        ax.set_visible(False)
    for ax, (title, table, fmt, *extra) in zip(axes.ravel(), panels, strict=False):
        cmap = extra[0] if extra else "viridis"
        values = table.to_numpy(dtype=float)
        lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
        limits = {}
        if cmap != "viridis" and lo < 0.0 < hi:
            limits = {"vmin": -max(-lo, hi), "vmax": max(-lo, hi)}
        image = ax.imshow(values, aspect="auto", cmap=cmap, **limits)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                red, green, blue, _ = image.cmap(image.norm(values[i, j]))
                color = "white" if 0.299 * red + 0.587 * green + 0.114 * blue < 0.5 else "black"
                label = format(values[i, j], fmt)
                ax.text(j, i, label, ha="center", va="center", fontsize=6, color=color)
        ax.set_xticks(range(values.shape[1]))
        ha = "right" if xtick_rotation else "center"
        ax.set_xticklabels(list(table.columns), fontsize=6, rotation=xtick_rotation, ha=ha)
        ax.set_yticks(range(values.shape[0]))
        ax.set_yticklabels(list(table.index), fontsize=6.5)
        ax.set_title(title, fontsize=8)
        ax.tick_params(length=0)
        fig.colorbar(image, ax=ax, shrink=0.8, pad=0.02).ax.tick_params(labelsize=6)
    return fig


def plot_weight_dynamics(effective_n: pd.DataFrame, turnover: pd.DataFrame) -> Figure:
    """Effective number of holdings (top) and per-rebalance turnover (bottom) over time.

    Parameters
    ----------
    effective_n : pandas.DataFrame
        Rebalance date x strategy ``1 / sum(w^2)``.
    turnover : pandas.DataFrame
        Rebalance date x strategy ``sum |target - drifted|`` (without the initial investment).
        Columns are keys of ``METHOD_COLORS``.
    """
    apply_style()
    fig, axes = plt.subplots(2, 1, figsize=(FIG_WIDTH_IN, 5.2), sharex=True, layout="constrained")
    for name in effective_n.columns:
        color = METHOD_COLORS[name]
        axes[0].plot(effective_n.index, effective_n[name], color=color, lw=0.9, label=name)
        if name in turnover.columns:
            axes[1].plot(turnover.index, turnover[name], color=color, lw=0.7)
    axes[0].set_ylabel("Effective number of holdings")
    axes[1].set_ylabel("Turnover per rebalance (sum of |dw|)")
    axes[1].set_xlabel("Rebalance date")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5, fontsize=6.5)
    return fig


def plot_weight_survival(pooled: Mapping[str, np.ndarray], threshold: float = 1e-3) -> Figure:
    """Share of holdings with weight above ``x`` (log x-axis), one line per strategy.

    Parameters
    ----------
    pooled : mapping of str to numpy.ndarray
        Pooled member weights per strategy (keys of ``METHOD_COLORS``).
    threshold : float, default 1e-3
        Left edge of the x-axis (weights at or below it count as no position).
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, 3.4), layout="constrained")
    grid = np.logspace(np.log10(threshold), np.log10(0.5), 200)
    for name, weights in pooled.items():
        share = [(weights > x).mean() for x in grid]
        ax.plot(grid, share, color=METHOD_COLORS[name], lw=1.2, label=name)
    ax.set_xscale("log")
    ax.set_xlabel("Weight x")
    ax.set_ylabel("Share of holdings with weight > x")
    ax.legend(fontsize=6.5, ncol=2)
    return fig


def plot_eigen_vs_performance(
    diagnostics: pd.DataFrame, performance: pd.DataFrame, estimators: Sequence[str]
) -> Figure:
    """Relative realized volatility of denoised strategies against window diagnostics.

    Each panel plots, per holding period, ``log(vol_strategy / vol_GMV-SAMPLE)`` (negative means
    lower risk than sample GMV) against one window diagnostic, with a least-squares line.

    Parameters
    ----------
    diagnostics : pandas.DataFrame
        Rebalance date x diagnostic (columns ``top_eig_share``, ``rmt_n_signal``).
    performance : pandas.DataFrame
        Rebalance date x ``logvol_<strategy>`` columns.
    estimators : sequence of str
        Strategies to show (keys of ``METHOD_COLORS``).
    """
    apply_style()
    labels = {
        "top_eig_share": "Market-mode share (largest eigenvalue / N)",
        "rmt_n_signal": "RMT signal eigenvalues",
    }
    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH_IN, 3.2), sharey=True, layout="constrained")
    rng = np.random.default_rng(0)
    index = diagnostics.index.intersection(performance.index)
    for ax, column in zip(axes, labels, strict=True):
        x = diagnostics.loc[index, column].to_numpy(dtype=float)
        jitter = rng.normal(0.0, 0.04, x.size) if column == "rmt_n_signal" else 0.0
        for name in estimators:
            y = performance.loc[index, f"logvol_{name}"].to_numpy(dtype=float)
            ax.scatter(x + jitter, y, s=7, alpha=0.5, color=METHOD_COLORS[name], label=name)
            if np.ptp(x) > 0:
                slope, intercept = np.polyfit(x, y, 1)
                grid = np.linspace(x.min(), x.max(), 20)
                ax.plot(grid, intercept + slope * grid, color=METHOD_COLORS[name], lw=1.2)
        ax.axhline(0.0, color=PALETTE["black"], lw=0.6)
        ax.set_xlabel(labels[column])
    axes[0].set_ylabel("log(vol / vol of GMV-SAMPLE), 21-day periods")
    axes[0].legend(fontsize=6.5)
    return fig


def plot_h3_summary(
    vol_pct: pd.DataFrame, relative_pct: pd.DataFrame, leverage: pd.DataFrame
) -> Figure:
    """Long-only versus unconstrained GMV: volatility, advantage over sample, and leverage.

    Parameters
    ----------
    vol_pct : pandas.DataFrame
        Estimator x (``Long-only``, ``Unconstrained``) annualized volatility in percent. Index
        values are keys of ``ESTIMATOR_COLORS``.
    relative_pct : pandas.DataFrame
        Same columns; rows are the non-sample estimators; volatility relative to the sample
        covariance in percent (negative = lower risk).
    leverage : pandas.DataFrame
        Rebalance date x estimator ``sum |w|`` of the unconstrained portfolios.
    """
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH_IN, 3.1), layout="constrained")
    for ax, table, label in (
        (axes[0], vol_pct, "Realized volatility (%, net)"),
        (axes[1], relative_pct, "Volatility vs sample GMV (%)"),
    ):
        x = np.arange(len(table.index))
        for offset, column, hatch in ((-0.2, "Long-only", None), (0.2, "Unconstrained", "////")):
            colors = [ESTIMATOR_COLORS[name] for name in table.index]
            ax.bar(x + offset, table[column], 0.38, color=colors, hatch=hatch, edgecolor="white")
        ax.axhline(0.0, color=PALETTE["black"], lw=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(list(table.index), rotation=30, ha="right", fontsize=7)
        ax.set_ylabel(label)
    for name in leverage.columns:
        color = ESTIMATOR_COLORS[name]
        axes[2].plot(leverage.index, leverage[name], color=color, lw=0.9, label=name)
    axes[2].set_ylabel("Unconstrained leverage (sum of |w|)")
    axes[2].tick_params(axis="x", rotation=30)
    axes[2].legend(fontsize=6.5)
    axes[0].bar([], [], color="grey", label="Long-only")
    axes[0].bar([], [], color="grey", hatch="////", edgecolor="white", label="Unconstrained")
    axes[0].legend(fontsize=6.5, loc="upper left")
    return fig


def plot_simulation_accuracy(summary: pd.DataFrame, distribution: str) -> Figure:
    """Estimator accuracy against the sample size in the simulation study.

    Four panels with the mean across replications and a confidence band: relative Frobenius error
    of the covariance, true variance of the long-only and of the unconstrained GMV portfolio as a
    ratio to the oracle, and L1 distance of the long-only weights to the oracle weights.

    Parameters
    ----------
    summary : pandas.DataFrame
        Output of :func:`pfopt.simulation.run.summarize` (columns ``distribution``, ``T``,
        ``method``, ``constraint``, ``metric``, ``mean``, ``ci_low``, ``ci_high``). Method names
        are keys of ``METHOD_COLORS``.
    distribution : str
        Distribution to plot.
    """
    apply_style()
    data = summary[summary["distribution"] == distribution]
    panels = (
        ("frob_error", "covariance", "Relative Frobenius error", True),
        ("l1_dist", "long-only", "L1 distance to oracle (long-only)", False),
        ("var_ratio", "long-only", "True variance / oracle (long-only)", True),
        ("var_ratio", "unconstrained", "True variance / oracle (unconstrained)", True),
    )
    fig, axes = plt.subplots(2, 2, figsize=(FIG_WIDTH_IN, 5.0), layout="constrained")
    for ax, (metric, constraint, label, log_y) in zip(axes.ravel(), panels, strict=True):
        cell = data[(data["metric"] == metric) & (data["constraint"] == constraint)]
        for method, group in cell.groupby("method", sort=False):
            group = group.sort_values("T")
            style = _strategy_style(method)
            marker = _MARKERS[method]
            ax.plot(group["T"], group["mean"], marker=marker, ms=3.5, lw=1.0, label=method, **style)
            ax.fill_between(
                group["T"], group["ci_low"], group["ci_high"], alpha=0.2, color=style["color"], lw=0
            )
        ax.set_xscale("log")
        ticks = sorted(cell["T"].unique())
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks], fontsize=7)
        ax.minorticks_off()
        if log_y:
            ax.set_yscale("log")
        ax.set_xlabel("Sample size T (days)")
        ax.set_ylabel(label, fontsize=7.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    extra_handles, extra_labels = axes[0, 1].get_legend_handles_labels()
    seen = dict(zip(labels, handles, strict=True))
    seen.update(zip(extra_labels, extra_handles, strict=True))
    fig.legend(list(seen.values()), list(seen), loc="outside lower center", ncol=6, fontsize=6.5)
    return fig


def plot_residual_timeseries(frame: pd.DataFrame) -> Figure:
    """Sector structure of the PCA residual over the rebalance dates.

    Parameters
    ----------
    frame : pandas.DataFrame
        Output of :func:`pfopt.backtest.residual.residual_diagnostics` (columns ``raw_gap``,
        ``resid_gap``, ``perm_null_mean``, ``perm_null_sd``, ``k_factors``, ``resid_n_signal``,
        ``resid_sector_share``, ``sector_share_null``).
    """
    apply_style()
    fig, axes = plt.subplots(3, 1, figsize=(FIG_WIDTH_IN, 5.6), sharex=True)
    axes[0].plot(frame.index, frame["raw_gap"], color=PALETTE["black"], label="Sample correlation")
    axes[0].plot(frame.index, frame["resid_gap"], color=PALETTE["orange"], label="PCA residual")
    low = frame["perm_null_mean"] - 2.0 * frame["perm_null_sd"]
    high = frame["perm_null_mean"] + 2.0 * frame["perm_null_sd"]
    axes[0].fill_between(frame.index, low, high, color=PALETTE["blue"], alpha=0.25, lw=0)
    axes[0].plot([], [], color=PALETTE["blue"], alpha=0.5, lw=6, label="Shuffled labels (±2 sd)")
    axes[0].set_ylabel("Same-sector minus\ncross-sector correlation")
    axes[0].legend(loc="upper right", ncol=3, fontsize=6.5)
    axes[1].step(frame.index, frame["k_factors"], where="post", color=PALETTE["blue"], label="K")
    axes[1].step(
        frame.index,
        frame["resid_n_signal"],
        where="post",
        color=PALETTE["orange"],
        label="Residual eigenvalues above MP edge",
    )
    axes[1].set_ylabel("Count")
    axes[1].set_ylim(bottom=0)
    axes[1].legend(loc="upper right", ncol=2, fontsize=6.5)
    axes[2].plot(frame.index, frame["resid_sector_share"], color=PALETTE["green"], label="Observed")
    axes[2].plot(
        frame.index, frame["sector_share_null"], color=PALETTE["black"], ls="--", label="Random"
    )
    axes[2].set_ylabel("Sector share of\nresidual eigenvectors")
    axes[2].set_ylim(bottom=0)
    axes[2].legend(loc="upper right", ncol=2, fontsize=6.5)
    axes[-1].set_xlabel("Date")
    return fig
