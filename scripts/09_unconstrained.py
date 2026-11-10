"""Unconstrained GMV strategies and the long-only gap (hypothesis H3).

Generates (or loads) unconstrained GMV weights for the sample, Ledoit-Wolf, RMT and PCA
covariances on exactly the same dates and universes as the long-only strategies, simulates them
with the same cost model (shorts pay no borrow cost), and compares realized volatility,
concentration, leverage and turnover. H3 is tested with a block bootstrap on the difference in
log variance ratios (estimator vs sample covariance) between the unconstrained and long-only
portfolios.

Reads ``returns_net.parquet``, ``turnover.parquet`` and the long-only weights written by
``02_run_backtest.py``. Writes ``unconstrained_metrics.csv``, ``unconstrained_h3.csv`` and
``unconstrained_summary.txt`` to ``results/tables`` and ``unc_h3`` to ``results/figures``.

Usage: ``python scripts/09_unconstrained.py --config config.yaml [--force]``
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import matplotlib
import pandas as pd

from pfopt.backtest.engine import read_weights_result, simulate
from pfopt.backtest.metrics import performance_metrics
from pfopt.backtest.stats import log_var_double_difference
from pfopt.backtest.unconstrained import (
    UNCONSTRAINED,
    UNCONSTRAINED_STRATEGIES,
    UnconstrainedResult,
    generate_unconstrained_weights,
)
from pfopt.config import Config, load_config
from pfopt.data.cleaning import read_processed
from pfopt.viz.plots import plot_h3_summary
from pfopt.viz.style import save_figure

logger = logging.getLogger("unconstrained")
ALTERNATIVES = ("Ledoit-Wolf", "RMT", "PCA")
SHORT = {"Sample": "SAMPLE", "Ledoit-Wolf": "LW", "RMT": "RMT", "PCA": "PCA"}


def _load_or_generate(
    cfg: Config, cache: Path, frames: dict[str, pd.DataFrame], force: bool
) -> UnconstrainedResult:
    marker = cache / "unc_diagnostics.parquet"
    if marker.exists() and not force:
        logger.info("loading cached unconstrained weights from %s", cache)
        weights = {s: pd.read_parquet(cache / f"unc_{s}.parquet") for s in UNCONSTRAINED_STRATEGIES}
        return UnconstrainedResult(weights=weights, diagnostics=pd.read_parquet(marker))
    started = time.perf_counter()
    result = generate_unconstrained_weights(frames["log_returns"], frames["dollar_volume"], cfg)
    logger.info("unconstrained weights generated in %.0f s", time.perf_counter() - started)
    for strategy, frame in result.weights.items():
        frame.to_parquet(cache / f"unc_{strategy}.parquet")
    result.diagnostics.to_parquet(marker)
    return result


def main() -> None:
    """Run the unconstrained strategies, the H3 test and write tables, figure and summary."""
    parser = argparse.ArgumentParser(description="Unconstrained GMV and H3.")
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    parser.add_argument("--force", action="store_true", help="regenerate the weights")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    matplotlib.use("Agg")

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    root = config_path.parent
    cache = root / cfg.paths.cache_dir / cfg.cache_key()
    for name in ("returns_net.parquet", "turnover.parquet", "diagnostics.parquet"):
        if not (cache / name).exists():
            raise SystemExit(f"{cache / name} not found; run scripts/02_run_backtest.py first")
    ann, rf = cfg.run.annualization, cfg.run.rf
    frames = read_processed(root / cfg.paths.processed_dir)
    unc = _load_or_generate(cfg, cache, frames, args.force)
    long_only = read_weights_result(cache)

    sims = {
        s: simulate(w, frames["simple_returns"], cfg.backtest.cost_bps, long_only=False)
        for s, w in unc.weights.items()
    }
    net_lo = pd.read_parquet(cache / "returns_net.parquet")
    turnover_lo = pd.read_parquet(cache / "turnover.parquet")
    lo_ids = [lo for _, lo in UNCONSTRAINED.values()]
    net = pd.concat([net_lo[lo_ids], pd.DataFrame({s: sim.net for s, sim in sims.items()})], axis=1)
    if net.isna().any().any():
        raise ValueError("long-only and unconstrained returns are not on the same dates")

    rows = {}
    for strategy in (*lo_ids, *UNCONSTRAINED_STRATEGIES):
        constrained = strategy in lo_ids
        turnover = turnover_lo[strategy] if constrained else sims[strategy].turnover
        weights = long_only.weights[strategy] if constrained else unc.weights[strategy]
        diagnostics = long_only.diagnostics if constrained else unc.diagnostics
        values = performance_metrics(net[strategy], turnover, ann, rf)
        pred = float(diagnostics[f"pred_vol_{strategy}"].mean())
        rows[strategy] = {
            "constraint": "long-only" if constrained else "unconstrained",
            "ann_vol": values["ann_vol"],
            "ann_return": values["ann_return"],
            "sharpe": values["sharpe"],
            "max_drawdown": values["max_drawdown"],
            "annual_turnover": values["annual_turnover"],
            "mean_max_weight": float(weights.max(axis=1).mean()),
            "mean_min_weight": float(weights.min(axis=1).mean()),
            "mean_leverage": float(weights.abs().sum(axis=1).mean()),
            "mean_pred_vol": pred,
            "realized_over_pred": values["ann_vol"] / pred,
        }
    metrics = pd.DataFrame.from_dict(rows, orient="index").rename_axis("strategy")
    tables, figures = root / cfg.paths.tables_dir, root / cfg.paths.figures_dir
    tables.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(tables / "unconstrained_metrics.csv")

    quadruples = {
        estimator: (
            f"GMV-{SHORT[estimator]}-UNC",
            "GMV-SAMPLE-UNC",
            f"GMV-{SHORT[estimator]}",
            "GMV-SAMPLE",
        )
        for estimator in ALTERNATIVES
    }
    s = cfg.stats
    h3 = log_var_double_difference(
        net,
        quadruples,
        block_length=s.block_length,
        n_boot=s.n_boot,
        seed=cfg.run.seed,
        alpha=s.alpha,
    )
    h3.to_csv(tables / "unconstrained_h3.csv", index=False)

    # ---- figure ----------------------------------------------------------------------------
    estimators = ("Sample", *ALTERNATIVES)
    vol = pd.DataFrame(
        {
            "Long-only": [metrics.loc[f"GMV-{SHORT[e]}", "ann_vol"] for e in estimators],
            "Unconstrained": [metrics.loc[f"GMV-{SHORT[e]}-UNC", "ann_vol"] for e in estimators],
        },
        index=list(estimators),
    )
    relative = 100 * (vol.loc[list(ALTERNATIVES)] / vol.loc["Sample"] - 1.0)
    leverage = pd.DataFrame(
        {e: unc.weights[f"GMV-{SHORT[e]}-UNC"].abs().sum(axis=1) for e in estimators}
    )
    save_figure(plot_h3_summary(100 * vol, relative, leverage), "unc_h3", figures)

    # ---- summary ---------------------------------------------------------------------------
    verdicts = []
    for _, r in h3.iterrows():
        if r["estimate"] < 0 and r["significant"]:
            verdict = "supported (advantage larger without the constraint, significant)"
        elif r["estimate"] < 0:
            verdict = "direction as H3 predicts, not significant"
        else:
            verdict = "not supported (advantage not larger without the constraint)"
        verdicts.append(
            f"  {r['name']:<12} gap unconstrained {r['gap_a']:+.3f}, long-only {r['gap_b']:+.3f}, "
            f"difference {r['estimate']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] "
            f"p={r['p_value']:.3f}: {verdict}"
        )
    relative_text = ", ".join(
        f"{e} {relative.loc[e, 'Long-only']:+.1f}/{relative.loc[e, 'Unconstrained']:+.1f}"
        for e in ALTERNATIVES
    )
    mean_leverage = metrics.loc[list(UNCONSTRAINED_STRATEGIES), "mean_leverage"].mean()
    ridge = [c for c in unc.diagnostics.columns if c.startswith("ridge_")]
    n_ridge = int((unc.diagnostics[ridge] > 0).sum().sum())
    lines = [
        f"{len(net)} daily net returns, {len(unc.weights['GMV-SAMPLE-UNC'])} rebalances, cost "
        f"{cfg.backtest.cost_bps:g} bps; shorts carry no borrow cost",
        f"rebalance x strategy cases needing a ridge: {n_ridge}",
        "",
        metrics.round(3).to_string(),
        "",
        "H3: log variance ratio vs the sample covariance, unconstrained minus long-only "
        "(negative = larger advantage without the constraint); block bootstrap "
        f"(block {s.block_length}, B={s.n_boot}):",
        *verdicts,
        "",
        "volatility relative to sample GMV (%, long-only / unconstrained): " + relative_text,
        f"mean unconstrained leverage {mean_leverage:.2f}",
    ]
    (tables / "unconstrained_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("summary written to %s", tables / "unconstrained_summary.txt")


if __name__ == "__main__":
    main()
