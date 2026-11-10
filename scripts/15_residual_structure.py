"""Residual-structure diagnostic (sector blocks left by the PCA factors).

For the estimation window of every rebalance, the part of the sample correlation matrix discarded
by the K-factor (Marchenko-Pastur) model is tested for sector structure with static sector labels
(a current snapshot, used as labels only): same-sector against cross-sector correlation, a
permutation test, residual eigenvalues above the MP edge and their sector share. The per-window
diagnostics are then correlated with the realized volatility of GMV-RMT, GMV-PCA and GMV-LW
relative to sample GMV.

Reads the universe and ``returns_net.parquet`` from the cache of ``02_run_backtest.py``. Writes
``residual_structure_windows.csv``, ``residual_structure_blocks.csv``,
``residual_structure_correlations.csv`` and ``residual_structure_summary.txt`` to
``results/tables`` and the figures ``res_blocks`` and ``res_timeseries`` to ``results/figures``.

Usage: ``python scripts/15_residual_structure.py --config config.yaml``
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from pfopt.backtest.engine import read_weights_result
from pfopt.backtest.residual import residual_diagnostics
from pfopt.backtest.structure import diagnostic_correlations, relative_performance
from pfopt.config import load_config
from pfopt.data.cleaning import read_processed
from pfopt.data.loader import load_companies
from pfopt.viz.plots import plot_heatmap_panels, plot_residual_timeseries
from pfopt.viz.style import save_figure

logger = logging.getLogger("residual_structure")
BASELINE = "GMV-SAMPLE"
COMPARED = ("GMV-LW", "GMV-RMT", "GMV-PCA")
DIAGNOSTICS = ("raw_gap", "resid_gap", "perm_z", "k_factors", "resid_n_signal")


def main() -> None:
    """Compute the diagnostics, write the tables and figures and a short summary."""
    parser = argparse.ArgumentParser(description="Residual-structure diagnostic.")
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    matplotlib.use("Agg")
    started = time.perf_counter()

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    root = config_path.parent
    cache = root / cfg.paths.cache_dir / cfg.cache_key()
    if not (cache / "returns_net.parquet").exists():
        raise SystemExit(f"{cache / 'returns_net.parquet'} not found; run 02_run_backtest.py first")
    result = read_weights_result(cache)
    net = pd.read_parquet(cache / "returns_net.parquet")
    log_returns = read_processed(root / cfg.paths.processed_dir)["log_returns"]
    companies = load_companies(
        root / cfg.paths.raw_dir / cfg.data.companies_file, cfg.data.companies_columns
    )
    sectors = companies.set_index("Symbol")["Sector"]

    frame, blocks = residual_diagnostics(
        log_returns,
        result.universe,
        sectors,
        cfg.universe.lookback_days,
        cfg.data.winsor_lower,
        cfg.data.winsor_upper,
        cfg.residual.n_perm,
        cfg.run.seed,
    )
    performance = relative_performance(net, frame.index, BASELINE, COMPARED, cfg.run.annualization)
    targets = performance[[f"logvol_{s}" for s in COMPARED]]
    correlations = diagnostic_correlations(frame.loc[targets.index, list(DIAGNOSTICS)], targets)

    tables, figures = root / cfg.paths.tables_dir, root / cfg.paths.figures_dir
    tables.mkdir(parents=True, exist_ok=True)
    frame.to_csv(tables / "residual_structure_windows.csv")
    blocks.to_csv(tables / "residual_structure_blocks.csv")
    correlations.to_csv(tables / "residual_structure_correlations.csv", index=False)
    save_figure(
        plot_heatmap_panels(
            [("Mean residual correlation by sector pair", blocks, ".2f", "RdBu_r")],
            ncols=1,
            panel_height=4.8,
            xtick_rotation=40,
        ),
        "res_blocks",
        figures,
    )
    save_figure(plot_residual_timeseries(frame), "res_timeseries", figures)

    diag = np.diag(blocks.to_numpy())
    off = blocks.to_numpy()[~np.eye(len(blocks), dtype=bool)]
    within = pd.Series(diag, index=blocks.index).dropna().sort_values(ascending=False)
    significant = float((frame["perm_p"] < 0.05).mean())
    lines = [
        f"{len(frame)} rebalance windows, T={cfg.universe.lookback_days}, "
        f"N={cfg.universe.n_assets}, {cfg.residual.n_perm} permutations per window; sector "
        "labels are a current snapshot used as labels only",
        f"retained factors K (MP rule): mean {frame['k_factors'].mean():.1f}, range "
        f"{frame['k_factors'].min():.0f} to {frame['k_factors'].max():.0f}",
        "",
        "same-sector minus cross-sector correlation (mean over windows):",
        f"  raw correlation      {frame['raw_gap'].mean():.3f}",
        f"  PCA residual         {frame['resid_gap'].mean():.3f}  (shuffled labels "
        f"{frame['perm_null_mean'].mean():.3f}, sd {frame['perm_null_sd'].mean():.3f}; "
        f"mean z {frame['perm_z'].mean():.1f})",
        f"  windows with permutation p < 0.05: {100 * significant:.0f}% "
        "(overlapping windows, so not independent; 5% expected under no structure)",
        "",
        f"residual eigenvalues above the MP edge: mean {frame['resid_n_signal'].mean():.1f} "
        f"(range {frame['resid_n_signal'].min():.0f} to {frame['resid_n_signal'].max():.0f})",
        f"their share in the span of the sector indicators: "
        f"{frame['resid_sector_share'].mean():.2f} against {frame['sector_share_null'].mean():.2f} "
        f"for a random direction ({int(frame['resid_sector_share'].notna().sum())} windows with at "
        "least one such eigenvalue)",
        "",
        f"mean residual correlation: within sector {np.nanmean(diag):.3f}, between sectors "
        f"{np.nanmean(off):.3f}",
        "within-sector residual correlation by sector (mean over windows):",
        within.round(3).to_string(),
        "",
        "Spearman correlation of window diagnostics with log(vol ratio vs GMV-SAMPLE) "
        "(p-values optimistic: overlapping windows):",
        correlations.round(3).to_string(index=False),
    ]
    (tables / "residual_structure_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    logger.info(
        "done in %.0f s; summary in %s",
        time.perf_counter() - started,
        tables / "residual_structure_summary.txt",
    )


if __name__ == "__main__":
    main()
