"""Step 5.4: weight concentration, turnover dynamics, sector exposure and spectrum diagnostics.

Reads the cached weights, ``returns_net.parquet`` and ``turnover.parquet`` written by
``02_run_backtest.py`` (for the config in ``config.yaml``) and the static sector labels of
``sp500_companies.csv`` (a current snapshot, used as labels only). Writes ``structure_*.csv`` and
``structure_summary.txt`` to ``results/tables`` and the figures ``struct_weights``,
``struct_dynamics``, ``struct_sectors`` and ``struct_eigen`` to ``results/figures``.

Usage: ``python scripts/08_structure.py --config config.yaml``
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
import pandas as pd

from pfopt.backtest.engine import ALL_STRATEGIES, read_weights_result
from pfopt.backtest.structure import (
    concentration_table,
    diagnostic_correlations,
    effective_n_table,
    mean_sector_table,
    pooled_member_weights,
    relative_performance,
    window_eigen_diagnostics,
)
from pfopt.config import load_config
from pfopt.data.cleaning import read_processed
from pfopt.data.loader import load_companies
from pfopt.viz.plots import (
    plot_eigen_vs_performance,
    plot_heatmap_panels,
    plot_weight_dynamics,
    plot_weight_survival,
)
from pfopt.viz.style import save_figure

logger = logging.getLogger("structure")
BASELINE = "GMV-SAMPLE"
COMPARED = ("GMV-LW", "GMV-RMT", "GMV-PCA")


def main() -> None:
    """Compute the structure tables and figures and write a short summary."""
    parser = argparse.ArgumentParser(description="Weight and portfolio structure analysis.")
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    matplotlib.use("Agg")

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    root = config_path.parent
    cache = root / cfg.paths.cache_dir / cfg.cache_key()
    for name in ("returns_net.parquet", "turnover.parquet"):
        if not (cache / name).exists():
            raise SystemExit(f"{cache / name} not found; run scripts/02_run_backtest.py first")
    result = read_weights_result(cache)
    net = pd.read_parquet(cache / "returns_net.parquet")
    turnover = pd.read_parquet(cache / "turnover.parquet").iloc[1:]
    weights = {s: result.weights[s] for s in ALL_STRATEGIES}
    log_returns = read_processed(root / cfg.paths.processed_dir)["log_returns"]
    companies = load_companies(
        root / cfg.paths.raw_dir / cfg.data.companies_file, cfg.data.companies_columns
    )
    sectors = companies.set_index("Symbol")["Sector"]
    tables, figures = root / cfg.paths.tables_dir, root / cfg.paths.figures_dir
    tables.mkdir(parents=True, exist_ok=True)

    # ---- concentration and weight distribution --------------------------------------------
    concentration = concentration_table(weights, result.universe)
    concentration.to_csv(tables / "structure_concentration.csv")
    effective_n = effective_n_table(weights)
    pooled = {s: pooled_member_weights(weights[s], result.universe) for s in ALL_STRATEGIES}
    save_figure(plot_weight_survival(pooled), "struct_weights", figures)
    save_figure(plot_weight_dynamics(effective_n, turnover), "struct_dynamics", figures)

    # ---- sector exposure --------------------------------------------------------------------
    mean_sector = mean_sector_table(weights, sectors)
    mean_sector = mean_sector[mean_sector.loc["EW"].sort_values(ascending=False).index]
    active = mean_sector - mean_sector.loc["EW"]
    mean_sector.to_csv(tables / "structure_sector_mean.csv")
    active.to_csv(tables / "structure_sector_active.csv")
    save_figure(
        plot_heatmap_panels(
            [
                ("Mean sector weight (%)", 100 * mean_sector, ".1f"),
                ("Active sector weight vs EW (percentage points)", 100 * active, ".1f", "RdBu_r"),
            ],
            ncols=1,
            panel_height=3.0,
            xtick_rotation=30,
        ),
        "struct_sectors",
        figures,
    )

    # ---- spectrum diagnostics versus relative performance --------------------------------
    spectrum = window_eigen_diagnostics(
        log_returns,
        result.universe,
        cfg.universe.lookback_days,
        cfg.data.winsor_lower,
        cfg.data.winsor_upper,
    )
    spectrum["rmt_n_signal"] = result.diagnostics["rmt_n_signal"]
    performance = relative_performance(
        net, result.diagnostics.index, BASELINE, COMPARED, cfg.run.annualization
    )
    periods = spectrum.join(performance, how="inner")
    periods.to_csv(tables / "structure_eigen_periods.csv")
    correlations = diagnostic_correlations(spectrum.loc[performance.index], performance)
    correlations.to_csv(tables / "structure_eigen_correlations.csv", index=False)
    save_figure(
        plot_eigen_vs_performance(spectrum, performance, ("GMV-RMT", "GMV-PCA")),
        "struct_eigen",
        figures,
    )

    # ---- summary ----------------------------------------------------------------------------
    high = periods["top_eig_share"] > periods["top_eig_share"].median()
    split = pd.DataFrame(
        {
            "market mode high": periods.loc[high, [f"logvol_{s}" for s in COMPARED]].mean(),
            "market mode low": periods.loc[~high, [f"logvol_{s}" for s in COMPARED]].mean(),
        }
    )
    tilts = []
    for strategy in (BASELINE, "GMV-RMT", "MV-SAMPLE"):
        row = active.loc[strategy].sort_values()
        tilts.append(
            f"{strategy}: most underweight {row.index[0]} ({100 * row.iloc[0]:+.1f} pp), "
            f"most overweight {row.index[-1]} ({100 * row.iloc[-1]:+.1f} pp)"
        )
    lines = [
        f"{len(weights['EW'])} rebalances; {len(periods)} complete holding periods; sectors are a "
        f"current snapshot used as labels only",
        "",
        "concentration (averages over rebalances; share_zero = member holdings at or below 0.1%):",
        concentration.round(3).to_string(),
        "",
        "mean sector weight (%):",
        (100 * mean_sector).round(1).to_string(),
        "",
        *tilts,
        "",
        "Spearman correlation of window diagnostics with log(vol ratio vs GMV-SAMPLE) and return "
        "difference (p-values optimistic: overlapping windows):",
        correlations.round(3).to_string(index=False),
        "",
        "mean log(vol / vol of GMV-SAMPLE) by market-mode share (median split; negative = "
        "lower risk):",
        split.round(3).to_string(),
        f"mean market-mode share {spectrum['top_eig_share'].mean():.3f}, "
        f"RMT signal count {spectrum['rmt_n_signal'].min():.0f} to "
        f"{spectrum['rmt_n_signal'].max():.0f}",
    ]
    (tables / "structure_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("summary written to %s", tables / "structure_summary.txt")


if __name__ == "__main__":
    main()
