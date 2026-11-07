"""Step 1.4: exploratory data analysis (tables and figures).

Snapshots: the first feasible date, the stress date from ``config.yaml`` and the last date, each
with its own point-in-time top-N universe and T-day window. Writes tables to ``results/tables``
(including a short ``eda_summary.txt``) and figures (PNG + PDF) to ``results/figures`` with the
prefix ``eda_``. The log stays short; the key numbers are in ``eda_summary.txt``.

Usage: ``python scripts/03_eda.py --config config.yaml``
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from pfopt.config import load_config
from pfopt.data.cleaning import read_processed
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
from pfopt.data.loader import load_index
from pfopt.data.universe import rebalance_dates, select_universe, snapshot_dates
from pfopt.viz.plots import (
    plot_corr_heatmaps,
    plot_mp_spectra,
    plot_return_histograms,
    plot_rolling_correlation,
    plot_spectrum_timeseries,
)
from pfopt.viz.style import save_figure

logger = logging.getLogger("eda")
_LABELS = {"first": "First", "stress": "Stress", "last": "Last"}


def _three(series: pd.Series, fmt: str = "{:.3f}") -> str:
    """Format min / median / max of a series."""
    parts = [fmt.format(v) for v in (series.min(), series.median(), series.max())]
    return "min " + parts[0] + " | median " + parts[1] + " | max " + parts[2]


def main() -> None:
    """Compute EDA quantities, save tables and figures, and write a short summary file."""
    parser = argparse.ArgumentParser(description="Exploratory data analysis.")
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    matplotlib.use("Agg")

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    root = config_path.parent
    tables, figures = root / cfg.paths.tables_dir, root / cfg.paths.figures_dir
    tables.mkdir(parents=True, exist_ok=True)
    n, t, w = cfg.universe.n_assets, cfg.universe.lookback_days, cfg.universe.dollar_volume_window
    ann = cfg.run.annualization

    frames = read_processed(root / cfg.paths.processed_dir)
    log_returns, dollar_volume = frames["log_returns"], frames["dollar_volume"]

    snapshots = snapshot_dates(log_returns, t, n, cfg.eda.stress_date)

    summary: list[str] = []
    pooled_rows, z_panels, heat_panels, mp_panels = {}, [], [], []
    for key, date in snapshots.items():
        universe = select_universe(log_returns, dollar_volume, date, t, n, w)
        window = get_window(log_returns, date, t, universe)
        title = f"{_LABELS[key]}: {date:%Y-%m-%d}"

        per_asset = return_summary(window, ann)
        per_asset.rename_axis("Symbol").to_csv(tables / f"eda_return_summary_{key}.csv")
        pooled = pooled_summary(window)
        pooled_rows[key] = {"date": f"{date:%Y-%m-%d}", **pooled}

        spec = spectrum_stats(window.to_numpy())
        order = leaf_order(spec["corr"])
        z_panels.append((title, pooled_standardized(window)))
        heat_panels.append((title, spec["corr"][np.ix_(order, order)]))
        mp_panels.append((title, spec))
        summary.append(
            f"{title} | N={window.shape[1]} T={window.shape[0]} q={spec['q']:.3f} "
            f"MP_edge={spec['lambda_plus']:.3f} n_signal={spec['n_signal']} "
            f"lambda_max={spec['lambda_max']:.1f} top_share={spec['top_share']:.3f} "
            f"avg_corr={spec['avg_corr']:.3f} | pooled ex.kurt={pooled['excess_kurtosis']:.2f} "
            f"share>|{pooled['tail_sd']:.0f}sd|={pooled['frac_beyond_tail']:.2e} "
            f"(normal {pooled['normal_frac_beyond_tail']:.1e}) | "
            f"median ann.vol={per_asset['ann_vol'].median():.3f} "
            f"mean skew={per_asset['skew'].mean():.2f}"
        )

    pd.DataFrame.from_dict(pooled_rows, orient="index").rename_axis("snapshot").to_csv(
        tables / "eda_pooled_summary.csv"
    )
    save_figure(plot_return_histograms(z_panels), "eda_return_hist", figures)
    save_figure(plot_corr_heatmaps(heat_panels), "eda_corr_heatmap", figures)
    save_figure(plot_mp_spectra(mp_panels), "eda_mp_spectrum", figures)

    dates = rebalance_dates(log_returns, t, n, cfg.backtest.rebalance_days)
    ts = spectrum_timeseries(log_returns, dollar_volume, dates, t, n, w)
    ts.to_csv(tables / "eda_spectrum_timeseries.csv", date_format="%Y-%m-%d")
    index = load_index(root / cfg.paths.raw_dir / cfg.data.index_file, cfg.data.index_columns)
    vol = index_realized_vol(index, cfg.eda.index_vol_window_days, ann)
    save_figure(plot_spectrum_timeseries(ts, vol), "eda_spectrum_timeseries", figures)

    vol_at = vol.reindex(ts.index)
    summary += [
        f"rebalance dates: {len(ts)} from {ts.index[0].date()} to {ts.index[-1].date()}",
        f"n_signal: {_three(ts['n_signal'], '{:.0f}')}",
        f"top_share: {_three(ts['top_share'])} (peak {ts['top_share'].idxmax().date()})",
        f"avg_corr: {_three(ts['avg_corr'])} (peak {ts['avg_corr'].idxmax().date()})",
        f"index vol overlay from {vol.dropna().index[0].date()}; correlation with rebalance-date "
        f"series over {int(vol_at.notna().sum())} dates: top_share "
        f"{ts['top_share'].corr(vol_at):.2f}, avg_corr {ts['avg_corr'].corr(vol_at):.2f}, "
        f"n_signal {ts['n_signal'].corr(vol_at):.2f}",
    ]

    rolling = rolling_avg_correlation(log_returns, cfg.eda.corr_window_days)
    rolling.rename_axis("Date").to_csv(tables / "eda_rolling_corr.csv", date_format="%Y-%m-%d")
    save_figure(
        plot_rolling_correlation(rolling, cfg.eda.highlight_years, cfg.eda.corr_window_days),
        "eda_rolling_corr",
        figures,
    )
    valid = rolling.dropna()
    by_year = ", ".join(
        f"{y} mean {valid[valid.index.year == y].mean():.3f}" for y in cfg.eda.highlight_years
    )
    other = valid[~valid.index.year.isin(list(cfg.eda.highlight_years))]
    summary.append(
        f"rolling {cfg.eda.corr_window_days}d avg corr: mean {valid.mean():.3f}, "
        f"min {valid.min():.3f} ({valid.idxmin().date()}), "
        f"max {valid.max():.3f} ({valid.idxmax().date()}); {by_year}; "
        f"other years mean {other.mean():.3f}"
    )

    (tables / "eda_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    logger.info("EDA done; figures in %s; short summary in %s", figures, tables / "eda_summary.txt")


if __name__ == "__main__":
    main()
