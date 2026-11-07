"""Step 3.2: estimated versus realized long-only efficient frontiers at three dates.

Dates: the first feasible date, the stress date from ``config.yaml``, and the latest date that still
has ``eda.frontier_horizon_days`` trading days after it (the last date of the data has no future).
For each date the window of the point-in-time universe is winsorized, frontiers are built from
the sample and the RMT covariance with the same expected returns, and the frontier portfolios
(constant weights) are evaluated on the following days. Writes ``mv_*`` tables to ``results/tables``
(including a short ``mv_summary.txt``) and ``mv_frontier`` (PNG + PDF) to ``results/figures``.

Usage: ``python scripts/03b_markowitz_frontier.py --config config.yaml``
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
import pandas as pd

from pfopt.backtest.covquality import weight_stats
from pfopt.backtest.frontier import frontier_experiment
from pfopt.config import load_config
from pfopt.data.cleaning import read_processed
from pfopt.data.eda import get_window
from pfopt.data.universe import select_universe, snapshot_dates
from pfopt.estimators.utils import winsorize_cols
from pfopt.viz.plots import plot_frontiers
from pfopt.viz.style import save_figure

logger = logging.getLogger("frontier")


def main() -> None:
    """Run the frontier experiment at the three dates and write tables, figure and summary."""
    parser = argparse.ArgumentParser(description="Estimated vs realized efficient frontiers.")
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
    horizon = cfg.eda.frontier_horizon_days

    frames = read_processed(root / cfg.paths.processed_dir)
    log_returns, simple_returns = frames["log_returns"], frames["simple_returns"]
    dollar_volume = frames["dollar_volume"]
    snaps = snapshot_dates(log_returns, t, n, cfg.eda.stress_date)
    latest = log_returns.index[len(log_returns.index) - 1 - horizon]
    if latest <= snaps["stress"]:
        raise SystemExit("not enough data after the stress date for the frontier horizon")
    dates = {"First": snaps["first"], "Stress": snaps["stress"], "Latest with future": latest}

    panels, point_rows, ms_rows, summary = [], [], [], []
    for label, date in dates.items():
        universe = select_universe(log_returns, dollar_volume, date, t, n, w)
        window = get_window(log_returns, date, t, universe)
        x = winsorize_cols(window.to_numpy(), cfg.data.winsor_lower, cfg.data.winsor_upper)
        pos = int(log_returns.index.get_loc(date))
        future_df = simple_returns.iloc[pos + 1 : pos + 1 + horizon][universe]
        n_missing = int(future_df.isna().sum().sum())
        future = future_df.fillna(0.0).to_numpy()
        if n_missing:
            logger.warning("%s: %d missing future returns set to 0", label, n_missing)

        curves = frontier_experiment(x, future, cfg)
        title = f"{label}: {date:%Y-%m-%d}"
        panels.append((title, curves))
        for name, curve in curves.items():
            for k in range(len(curve.est_vol)):
                point_rows.append(
                    {
                        "date": f"{date:%Y-%m-%d}",
                        "estimator": name,
                        "point": k,
                        "est_vol": curve.est_vol[k],
                        "est_ret": curve.est_ret[k],
                        "real_vol": curve.real_vol[k],
                        "real_ret": curve.real_ret[k],
                    }
                )
            stats = weight_stats(curve.max_sharpe_weights)
            est_sr = curve.ms_est[1] / curve.ms_est[0]
            real_sr = curve.ms_real[1] / curve.ms_real[0]
            ms_rows.append(
                {
                    "date": f"{date:%Y-%m-%d}",
                    "estimator": name,
                    "est_vol": curve.ms_est[0],
                    "est_ret": curve.ms_est[1],
                    "est_sharpe": est_sr,
                    "real_vol": curve.ms_real[0],
                    "real_ret": curve.ms_real[1],
                    "real_sharpe": real_sr,
                    "effective_n": stats["effective_n"],
                    "max_weight": stats["max_weight"],
                    "n_positions": stats["n_positions"],
                }
            )
            summary.append(
                f"{title} | {name} | max-Sharpe est ret {curve.ms_est[1]:.3f} vol "
                f"{curve.ms_est[0]:.3f} SR {est_sr:.2f} | realized ret {curve.ms_real[1]:.3f} vol "
                f"{curve.ms_real[0]:.3f} SR {real_sr:.2f} | eff.N {stats['effective_n']:.1f} "
                f"max w {stats['max_weight']:.3f} | frontier top end est ret "
                f"{curve.est_ret[-1]:.3f} vs realized {curve.real_ret[-1]:.3f}"
            )
        if n_missing:
            summary.append(f"  ({n_missing} missing future returns set to 0)")

    pd.DataFrame(point_rows).to_csv(tables / "mv_frontier_points.csv", index=False)
    pd.DataFrame(ms_rows).to_csv(tables / "mv_max_sharpe.csv", index=False)
    (tables / "mv_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    save_figure(plot_frontiers(panels), "mv_frontier", figures)
    logger.info("done; summary in %s, figure in %s", tables / "mv_summary.txt", figures)


if __name__ == "__main__":
    main()
