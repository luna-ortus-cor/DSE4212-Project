"""Sensitivity analyses (T, N / q, costs, weight cap, PCA K rule, HRP linkage).

Builds every scenario from the ``sensitivity`` section of ``config.yaml`` (one changed field each),
generates or loads cached weights, simulates net returns, and writes ``sensitivity_metrics.csv``
(own and common out-of-sample windows), ``sensitivity_trials.csv`` (number of variants tried) and
``sensitivity_summary.txt`` to ``results/tables`` plus ``sens_q`` and ``sens_other`` to
``results/figures``. Sensitivity only: nothing here selects the headline configuration. The first
run generates weights for each non-baseline scenario (several minutes each); later runs use caches.

Usage: ``python scripts/07_sensitivity.py --config config.yaml [--force]``
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import matplotlib
import pandas as pd

from pfopt.backtest.engine import ALL_STRATEGIES
from pfopt.backtest.sensitivity import (
    HRP_STRATEGIES,
    Scenario,
    build_scenarios,
    common_start,
    metrics_long,
    run_scenarios,
    scenario_metrics,
    trial_counts,
)
from pfopt.config import load_config
from pfopt.data.cleaning import read_processed
from pfopt.viz.plots import plot_heatmap_panels
from pfopt.viz.style import save_figure

logger = logging.getLogger("sensitivity")


def _columns(
    scenarios: list[Scenario], common: dict[str, pd.DataFrame], grid: str, labels: dict[str, str]
) -> dict[str, pd.DataFrame]:
    """Label -> common-window metrics of the baseline and the scenarios of one grid."""
    chosen = [s for s in scenarios if s.grid in ("baseline", grid)]
    return {labels.get(s.id, s.variant): common[s.id] for s in chosen}


def _matrix(columns: dict[str, pd.DataFrame], metric: str, rows: tuple[str, ...]) -> pd.DataFrame:
    frame = pd.DataFrame({label: table[metric] for label, table in columns.items()})
    return frame.loc[list(rows)]


def main() -> None:
    """Run all scenarios and write tables, figures and the summary."""
    parser = argparse.ArgumentParser(description="Sensitivity analyses.")
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    parser.add_argument("--force", action="store_true", help="regenerate all weights")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    matplotlib.use("Agg")
    started = time.perf_counter()

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    root = config_path.parent
    ann, rf = cfg.run.annualization, cfg.run.rf
    frames = read_processed(root / cfg.paths.processed_dir)
    log_returns = frames["log_returns"]

    scenarios, skipped = build_scenarios(cfg, log_returns)
    for message in skipped:
        logger.warning("skipped: %s", message)
    logger.info("%d scenarios (including the baseline)", len(scenarios))
    runs = run_scenarios(
        scenarios,
        log_returns,
        frames["dollar_volume"],
        frames["simple_returns"],
        root / cfg.paths.cache_dir,
        force=args.force,
    )

    start = common_start(runs)
    common = {s.id: scenario_metrics(runs[s.id], start, ann, rf) for s in scenarios}
    long = metrics_long(scenarios, runs, ann, rf)
    trials = trial_counts(scenarios)
    tables, figures = root / cfg.paths.tables_dir, root / cfg.paths.figures_dir
    tables.mkdir(parents=True, exist_ok=True)
    long.to_csv(tables / "sensitivity_metrics.csv", index=False)
    trials.to_csv(tables / "sensitivity_trials.csv", index=False)

    # ---- q grid (T and N): sorted by q = N / T --------------------------------------------
    q_scenarios = sorted(
        (s for s in scenarios if s.grid in ("baseline", "lookback", "universe")), key=lambda s: s.q
    )
    q_labels = {
        s.id: f"N={s.cfg.universe.n_assets}\nT={s.cfg.universe.lookback_days}\nq={s.q:.2f}"
        for s in q_scenarios
    }
    q_columns = {q_labels[s.id]: common[s.id] for s in q_scenarios}
    vol = _matrix(q_columns, "ann_vol", ALL_STRATEGIES)
    sharpe = _matrix(q_columns, "sharpe", ALL_STRATEGIES)
    turnover = _matrix(q_columns, "annual_turnover", ALL_STRATEGIES)
    vol_vs_ew = vol / vol.loc["EW"]
    save_figure(
        plot_heatmap_panels(
            [
                ("Annualized volatility (net)", vol, ".3f"),
                ("Sharpe ratio (net)", sharpe, ".2f"),
                ("Volatility relative to EW", vol_vs_ew, ".2f"),
                ("Annual turnover", turnover, ".1f"),
            ],
        ),
        "sens_q",
        figures,
    )

    # ---- cost, cap, K and linkage: Sharpe heatmaps ----------------------------------------
    cost_labels = {s.id: f"{s.cost_bps:g} bps" for s in scenarios if s.grid in ("baseline", "cost")}
    cost_columns = _columns(scenarios, common, "cost", cost_labels)
    cost_columns = dict(sorted(cost_columns.items(), key=lambda item: float(item[0].split()[0])))
    cap_columns = _columns(scenarios, common, "cap", {"baseline": "no cap"})
    k_columns = _columns(scenarios, common, "pca_k", {"baseline": f"{cfg.pca.k_rule.upper()}"})
    link_columns = _columns(scenarios, common, "hrp_linkage", {"baseline": cfg.hrp.linkage})
    other = [
        ("Sharpe vs cost (bps)", _matrix(cost_columns, "sharpe", ALL_STRATEGIES), ".2f"),
        ("Sharpe vs cap (HRP uncapped)", _matrix(cap_columns, "sharpe", ALL_STRATEGIES), ".2f"),
        ("GMV-PCA Sharpe vs K rule", _matrix(k_columns, "sharpe", ("GMV-PCA",)), ".2f"),
        ("HRP Sharpe vs linkage", _matrix(link_columns, "sharpe", HRP_STRATEGIES), ".2f"),
    ]
    save_figure(plot_heatmap_panels(other), "sens_other", figures)

    # ---- summary text ---------------------------------------------------------------------
    lines = [
        f"scenarios run (incl. baseline): {len(scenarios)}; common window starts {start.date()} "
        f"(all tables below use it; own-window numbers are in sensitivity_metrics.csv)",
        "skipped: " + ("; ".join(skipped) if skipped else "none"),
        "",
        "strategy-parameter combinations tried:",
        trials.to_string(index=False),
        "",
        "volatility by q (net, annualized):",
        vol.round(3).to_string(),
        "",
        "Sharpe by q (net):",
        sharpe.round(2).to_string(),
        "",
        "volatility relative to EW by q:",
        vol_vs_ew.round(2).to_string(),
        "",
        "annual turnover by q:",
        turnover.round(1).to_string(),
        "",
        "Sharpe vs cost (bps):",
        _matrix(cost_columns, "sharpe", ALL_STRATEGIES).round(2).to_string(),
        "",
        "volatility vs weight cap (HRP has no cap mechanism and stays uncapped):",
        _matrix(cap_columns, "ann_vol", ALL_STRATEGIES).round(3).to_string(),
        "Sharpe vs weight cap:",
        _matrix(cap_columns, "sharpe", ALL_STRATEGIES).round(2).to_string(),
        "",
        "GMV-PCA volatility / Sharpe vs K rule:",
        _matrix(k_columns, "ann_vol", ("GMV-PCA",)).round(3).to_string(),
        _matrix(k_columns, "sharpe", ("GMV-PCA",)).round(2).to_string(),
        "",
        "HRP volatility / Sharpe vs linkage:",
        _matrix(link_columns, "ann_vol", HRP_STRATEGIES).round(3).to_string(),
        _matrix(link_columns, "sharpe", HRP_STRATEGIES).round(2).to_string(),
    ]
    (tables / "sensitivity_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    elapsed = time.perf_counter() - started
    logger.info("done in %.0f s; summary in %s", elapsed, tables / "sensitivity_summary.txt")


if __name__ == "__main__":
    main()
