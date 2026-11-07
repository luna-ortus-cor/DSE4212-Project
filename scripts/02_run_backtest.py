"""Step 4.5: run the main backtest (Tracks A and B) and write tables and figures.

1. Load the cached weights (``results/cache/<config hash>/``) or generate them; stale caches
   without the volatility-forecast columns are regenerated.
2. Simulate every strategy with drift and proportional costs (``backtest.cost_bps``).
3. Write the metrics tables to ``results/tables`` and the figures ``bt_*`` to ``results/figures``.
4. Cache daily gross/net returns and turnover (``returns_gross.parquet``, ``returns_net.parquet``,
   ``turnover.parquet``, ``spx_returns.parquet``) next to the weights for the Phase 5 analyses.

The S&P 500 index file starts later than the strategies and is a price index (no dividends), so SPX
only appears in the common-period table and figure, which start at its first return date.

Usage: ``python scripts/02_run_backtest.py --config config.yaml [--force]``
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from pfopt.backtest.covquality import (
    forecast_quality_table,
    risk_forecast_quality,
    stability_table,
)
from pfopt.backtest.engine import (
    ALL_STRATEGIES,
    WeightsResult,
    generate_weights,
    read_weights_result,
    simulate_all,
    write_weights_result,
)
from pfopt.backtest.metrics import drawdown_series, metrics_table, performance_metrics
from pfopt.config import Config, load_config
from pfopt.data.cleaning import read_processed
from pfopt.data.loader import load_index
from pfopt.viz.plots import (
    plot_drawdowns,
    plot_equity_curves,
    plot_risk_return,
    plot_rolling_sharpe_vol,
    plot_turnover_effective_n,
    plot_vol_forecast_ratio,
)
from pfopt.viz.style import save_figure

logger = logging.getLogger("backtest")
_GMV = ("GMV-SAMPLE", "GMV-LW", "GMV-RMT", "GMV-PCA")


def _load_or_generate(
    cfg: Config, cache: Path, log_returns: pd.DataFrame, dollar_volume: pd.DataFrame, force: bool
) -> tuple[WeightsResult, float | None]:
    """Cached weights if they are complete and current, otherwise freshly generated ones."""
    if not force and (cache / "diagnostics.parquet").exists():
        cached = read_weights_result(cache)
        current = "pred_vol_GMV-SAMPLE" in cached.diagnostics.columns
        if current and set(cached.weights) == set(ALL_STRATEGIES):
            logger.info("using cached weights from %s", cache)
            return cached, None
        logger.info("cached weights are stale (missing strategies or forecast columns); regenerating")
    start = time.perf_counter()
    result = generate_weights(log_returns, dollar_volume, cfg)
    elapsed = time.perf_counter() - start
    write_weights_result(result, cache)
    logger.info("weights generated in %.1f s and cached in %s", elapsed, cache)
    return result, elapsed


def _headline(net: pd.DataFrame, gross: pd.DataFrame, cost_bps: float) -> list[str]:
    lines = [
        f"headline metrics, net of {cost_bps:g} bps costs (gross Sharpe in brackets):",
        "strategy     | ann.ret | ann.vol | Sharpe [gross] | Sortino | maxDD  | turnover/yr",
    ]
    for name in net.index:
        n, g = net.loc[name], gross.loc[name]
        lines.append(
            f"{name:<12} | {n['ann_return']:7.3f} | {n['ann_vol']:7.3f} | "
            f"{n['sharpe']:6.2f} [{g['sharpe']:5.2f}] | {n['sortino']:7.2f} | "
            f"{n['max_drawdown']:6.3f} | {n['annual_turnover']:6.2f}"
        )
    return lines


def main() -> None:
    """Run the backtest end to end and write all outputs."""
    parser = argparse.ArgumentParser(description="Run the main backtest.")
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    parser.add_argument("--force", action="store_true", help="regenerate the weights")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    matplotlib.use("Agg")
    started = time.perf_counter()

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    root = config_path.parent
    tables, figures = root / cfg.paths.tables_dir, root / cfg.paths.figures_dir
    tables.mkdir(parents=True, exist_ok=True)
    cache = root / cfg.paths.cache_dir / cfg.cache_key()
    ann, rf, cost = cfg.run.annualization, cfg.run.rf, cfg.backtest.cost_bps

    frames = read_processed(root / cfg.paths.processed_dir)
    log_returns, simple_returns = frames["log_returns"], frames["simple_returns"]
    result, generation_seconds = _load_or_generate(
        cfg, cache, log_returns, frames["dollar_volume"], args.force
    )
    sims = simulate_all(result, simple_returns, cost)

    gross_df = pd.DataFrame({s: sim.gross for s, sim in sims.items()})
    net_df = pd.DataFrame({s: sim.net for s, sim in sims.items()})
    turnover_df = pd.DataFrame({s: sim.turnover for s, sim in sims.items()})
    for name, frame in (("returns_gross", gross_df), ("returns_net", net_df), ("turnover", turnover_df)):
        frame.to_parquet(cache / f"{name}.parquet")

    # ---- metric tables ------------------------------------------------------------------
    net_table = metrics_table(sims, "net", ann, rf)
    gross_table = metrics_table(sims, "gross", ann, rf)
    main = pd.concat(
        [gross_table.assign(basis="gross"), net_table.assign(basis="net")]
    ).rename_axis("strategy").reset_index()
    main = main[["strategy", "basis", *[c for c in main.columns if c not in ("strategy", "basis")]]]
    main.to_csv(tables / "main_metrics.csv", index=False)

    forecast = forecast_quality_table(result.diagnostics, sims, annualization=ann)
    forecast.to_csv(tables / "risk_forecast_quality.csv")
    stability = stability_table(result.weights, sims)
    stability.to_csv(tables / "weight_stability.csv")

    # ---- SPX and the common period ------------------------------------------------------------
    index = load_index(root / cfg.paths.raw_dir / cfg.data.index_file, cfg.data.index_columns)
    level = index.set_index("Date")[cfg.data.index_columns[1]].sort_index()
    spx = level.pct_change().dropna().rename("SPX")
    spx = spx.loc[spx.index.intersection(net_df.index)]
    spx.to_frame().to_parquet(cache / "spx_returns.parquet")
    common = net_df.loc[spx.index]
    common_rows = {s: performance_metrics(common[s], None, ann, rf) for s in common.columns}
    common_rows["SPX"] = performance_metrics(spx, None, ann, rf)
    pd.DataFrame.from_dict(common_rows, orient="index").rename_axis("strategy").to_csv(
        tables / "common_period_metrics.csv"
    )

    # ---- figures ----------------------------------------------------------------------------------
    wealth = (1.0 + net_df).cumprod()
    save_figure(plot_equity_curves(wealth, "Wealth, net of costs (start = 1)"), "bt_equity_curves", figures)
    wealth_common = (1.0 + pd.concat([common, spx], axis=1)).cumprod()
    save_figure(
        plot_equity_curves(wealth_common, "Wealth since the SPX start, net (S&P 500 price index)"),
        "bt_equity_since_spx",
        figures,
    )
    drawdowns = pd.DataFrame({s: drawdown_series(net_df[s]) for s in net_df.columns})
    save_figure(plot_drawdowns(drawdowns), "bt_drawdowns", figures)
    rolling = net_df.rolling(ann)
    rolling_vol = rolling.std(ddof=1) * np.sqrt(ann)
    rolling_sharpe = (rolling.mean() - rf / ann) / rolling.std(ddof=1) * np.sqrt(ann)
    save_figure(plot_rolling_sharpe_vol(rolling_sharpe, rolling_vol, ann), "bt_rolling_sharpe_vol", figures)
    save_figure(
        plot_turnover_effective_n(net_table["annual_turnover"], stability["mean_effective_n"]),
        "bt_turnover_effective_n",
        figures,
    )
    save_figure(plot_risk_return(net_table["ann_vol"], net_table["ann_return"]), "bt_risk_return", figures)
    ratios = {
        s: risk_forecast_quality(result.diagnostics, sims[s], s, ann).periods["ratio"] for s in _GMV
    }
    save_figure(plot_vol_forecast_ratio(ratios), "bt_vol_forecast_ratio", figures)

    # ---- sanity checks and headline summary -----------------------------------------------
    sums = max(float(np.abs(f.sum(axis=1) - 1.0).max()) for f in result.weights.values())
    min_weight = min(float(f.to_numpy().min()) for f in result.weights.values())
    nan_returns = int(gross_df.isna().sum().sum() + net_df.isna().sum().sum())
    fallbacks = int((result.diagnostics["n_solver_fallbacks"] > 0).sum())
    missing_events = sum(sim.n_missing_returns for sim in sims.values())
    elapsed = time.perf_counter() - started
    generation = "cached" if generation_seconds is None else f"{generation_seconds:.0f} s"
    universe_min = int(result.diagnostics["n_universe"].min())
    universe_max = int(result.diagnostics["n_universe"].max())
    checks = [
        f"OOS: {net_df.index[0].date()} to {net_df.index[-1].date()} ({len(net_df)} days), "
        f"{len(result.diagnostics)} rebalances, cost {cost:g} bps",
        f"weights: max |row sum - 1| = {sums:.2e}, min weight = {min_weight:.2e}",
        f"NaN in gross/net returns: {nan_returns}; missing held-asset returns: {missing_events}",
        f"rebalances with a solver fallback: {fallbacks}; universe sizes "
        f"{universe_min} to {universe_max}",
        f"SPX returns: {spx.index[0].date()} to {spx.index[-1].date()} ({len(spx)} days)",
        f"runtime: weights {generation}, total {elapsed:.0f} s",
    ]
    lines = [
        *checks,
        "",
        *_headline(net_table, gross_table, cost),
        "",
        "risk forecast (realized / predicted volatility):",
        "strategy     | mean ratio | pooled variance ratio | bias statistic",
    ]
    for name in forecast.index:
        row = forecast.loc[name]
        lines.append(
            f"{name:<12} | {row['mean_ratio']:.3f} | {row['pooled_variance_ratio']:.3f} | "
            f"{row['bias_statistic']:.3f}"
        )
    (tables / "backtest_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("backtest done in %.0f s; summary in %s", elapsed, tables / "backtest_summary.txt")


if __name__ == "__main__":
    main()
