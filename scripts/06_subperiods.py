"""Step 5.2: subperiod and volatility-regime analysis of the net-of-cost strategy returns.

Reads ``returns_net.parquet`` from the cache written by ``02_run_backtest.py``. Periods: full
out-of-sample, COVID stress, the stress year, the remaining calm days, and the high/low volatility
regimes defined by the S&P 500's trailing realized volatility against its *expanding* median
(labels lagged one day, no look-ahead). Writes ``subperiod_metrics.csv`` and
``subperiod_summary.txt`` to ``results/tables`` and ``subperiod_sharpe_drawdown`` to
``results/figures``.

Usage: ``python scripts/06_subperiods.py --config config.yaml``
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
import pandas as pd

from pfopt.backtest.regimes import (
    metrics_by_period,
    period_masks,
    regime_labels_for_returns,
    regime_masks,
)
from pfopt.config import load_config
from pfopt.data.eda import index_realized_vol
from pfopt.data.loader import load_index
from pfopt.viz.plots import plot_period_bars
from pfopt.viz.style import save_figure

logger = logging.getLogger("subperiods")


def main() -> None:
    """Compute the per-period metrics, write tables and the figure, and a short summary."""
    parser = argparse.ArgumentParser(description="Subperiod and regime analysis.")
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    matplotlib.use("Agg")

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    root = config_path.parent
    path = root / cfg.paths.cache_dir / cfg.cache_key() / "returns_net.parquet"
    if not path.exists():
        raise SystemExit(f"{path} not found; run scripts/02_run_backtest.py first")
    returns = pd.read_parquet(path)
    ann, rf, r = cfg.run.annualization, cfg.run.rf, cfg.regimes

    index = load_index(root / cfg.paths.raw_dir / cfg.data.index_file, cfg.data.index_columns)
    vol = index_realized_vol(index, r.vol_window_days, ann)
    labels = regime_labels_for_returns(vol, returns.index, r.min_history_days)
    masks = period_masks(returns.index, r.covid_start, r.covid_end, r.stress_year)
    masks.update(regime_masks(labels))
    order = ["full", "covid", f"year_{r.stress_year}", "calm", "high_vol", "low_vol"]
    names = {
        "full": "Full OOS",
        "covid": "COVID stress",
        f"year_{r.stress_year}": str(r.stress_year),
        "calm": "Calm (rest)",
        "high_vol": "High-vol regime",
        "low_vol": "Low-vol regime",
    }
    table = metrics_by_period(returns, {k: masks[k] for k in order}, ann, rf)

    tables, figures = root / cfg.paths.tables_dir, root / cfg.paths.figures_dir
    tables.mkdir(parents=True, exist_ok=True)
    table.to_csv(tables / "subperiod_metrics.csv")

    days = {k: int(masks[k].sum()) for k in order}
    labelled = {k: f"{names[k]}\n({days[k]}d)" for k in order}

    def matrix(metric: str) -> pd.DataFrame:
        wide = table[metric].unstack("strategy").loc[order]
        return wide.rename(index=names)

    sharpe, drawdown, volatility = matrix("sharpe"), matrix("max_drawdown"), matrix("ann_vol")
    figure_sharpe = table["sharpe"].unstack("strategy").loc[order].rename(index=labelled)
    figure_drawdown = table["max_drawdown"].unstack("strategy").loc[order].rename(index=labelled)
    save_figure(
        plot_period_bars(figure_sharpe, figure_drawdown), "subperiod_sharpe_drawdown", figures
    )

    classified = labels.dropna()
    share_high = days["high_vol"] / (days["high_vol"] + days["low_vol"])
    lines = [
        "days per period: " + ", ".join(f"{names[k]} {days[k]}" for k in order),
        f"regime labels available from {classified.index[0].date()} "
        f"to {classified.index[-1].date()} "
        f"({len(classified)} days; {100 * share_high:.0f}% high-vol); unclassified days are in "
        f"neither regime; window {r.vol_window_days}d, expanding median after "
        f"{r.min_history_days} observations, label lagged one day",
        "short periods give noisy ratios: the standard error of an annualized Sharpe ratio is "
        "roughly sqrt(252 / days) (about 1.0 for 250 days)",
        "",
        "Sharpe ratio (net):",
        sharpe.round(2).to_string(),
        "",
        "Maximum drawdown (net):",
        drawdown.round(3).to_string(),
        "",
        "Annualized volatility (net):",
        volatility.round(3).to_string(),
    ]
    (tables / "subperiod_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("summary written to %s", tables / "subperiod_summary.txt")


if __name__ == "__main__":
    main()
