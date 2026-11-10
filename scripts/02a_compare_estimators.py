"""Compare the covariance estimators and core strategies at three snapshot dates.

Snapshots: first feasible date, the stress date from ``config.yaml`` and the last date (as in the
EDA). At each date the point-in-time top-N universe's T-day window is winsorized per column and
fed to the four covariance estimators and the seven core strategies. Writes to ``results/tables``:
``cmp_condition_numbers.csv``, ``cmp_estimator_info.csv``, ``cmp_weight_stats.csv``,
``cmp_weights_<date key>.csv`` and a short ``cmp_summary.txt``; figures ``cmp_eigenvalue_spectra``
and ``cmp_sorted_weights`` (PNG + PDF) to ``results/figures``.

Usage: ``python scripts/02a_compare_estimators.py --config config.yaml``
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from pfopt.backtest.covquality import weight_stats
from pfopt.backtest.strategies import CORE_STRATEGIES, ESTIMATORS, core_strategies
from pfopt.config import load_config
from pfopt.data.cleaning import read_processed
from pfopt.data.eda import get_window
from pfopt.data.universe import select_universe, snapshot_dates
from pfopt.estimators.pca_factor import n_factors
from pfopt.estimators.utils import condition_number, winsorize_cols
from pfopt.viz.plots import plot_cov_spectra, plot_sorted_weights
from pfopt.viz.style import save_figure

logger = logging.getLogger("compare")
_LABELS = {"first": "First", "stress": "Stress", "last": "Last"}


def _fmt_row(values: dict[str, float], fmt: str) -> str:
    return " | ".join(f"{name} {fmt.format(value)}" for name, value in values.items())


def main() -> None:
    """Build estimators and strategies at the three dates; write tables, figures and a summary."""
    parser = argparse.ArgumentParser(description="Compare covariance estimators and strategies.")
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

    cond_rows, info_rows, stat_rows, summary = [], [], [], []
    spectra_panels, weight_panels = [], []
    for key, date in snapshots.items():
        universe = select_universe(log_returns, dollar_volume, date, t, n, w)
        window = get_window(log_returns, date, t, universe)
        x = winsorize_cols(window.to_numpy(), cfg.data.winsor_lower, cfg.data.winsor_upper)
        result = core_strategies(x, cfg)
        title = f"{_LABELS[key]}: {date:%Y-%m-%d}"

        spectra, conds = {}, {}
        for name in ESTIMATORS:
            eigenvalues = np.linalg.eigh(result.covariances[name])[0][::-1] * ann
            spectra[name] = eigenvalues
            conds[name] = condition_number(result.covariances[name])
            cond_rows.append(
                {
                    "date": f"{date:%Y-%m-%d}",
                    "estimator": name,
                    "condition_number": conds[name],
                    "lambda_min_annualized": float(eigenvalues[-1]),
                    "lambda_max_annualized": float(eigenvalues[0]),
                }
            )
        spectra_panels.append((title, spectra))
        weight_panels.append((title, result.weights))

        k_baing = n_factors(x, rule="baing", kmax=cfg.pca.baing_kmax)
        info = result.info
        info_rows.append(
            {
                "date": f"{date:%Y-%m-%d}",
                "n_assets": x.shape[1],
                "n_obs": x.shape[0],
                "lw_shrinkage": info["lw_shrinkage"],
                "rmt_n_signal": info["rmt_n_signal"],
                "pca_K_rule": info["pca_K"],
                "pca_K_baing_diagnostic": k_baing,
                "pca_explained_variance": info["pca_explained"],
            }
        )
        stats = {}
        for strategy in CORE_STRATEGIES:
            stats[strategy] = weight_stats(result.weights[strategy])
            stat_rows.append({"date": f"{date:%Y-%m-%d}", "strategy": strategy, **stats[strategy]})
        pd.DataFrame(result.weights, index=pd.Index(universe, name="Symbol")).to_csv(
            tables / f"cmp_weights_{key}.csv"
        )

        summary += [
            f"{title} | LW shrinkage {info['lw_shrinkage']:.3f} | RMT n_signal "
            f"{info['rmt_n_signal']} | PCA K (mp rule) {info['pca_K']}, Bai-Ng {k_baing}",
            "  condition numbers: " + _fmt_row(conds, "{:.3g}"),
            "  effective N: " + _fmt_row({s: v["effective_n"] for s, v in stats.items()}, "{:.1f}"),
            "  max weight: " + _fmt_row({s: v["max_weight"] for s, v in stats.items()}, "{:.3f}"),
            "  positions > 0.1%: "
            + _fmt_row({s: v["n_positions"] for s, v in stats.items()}, "{:.0f}"),
        ]

    pd.DataFrame(cond_rows).to_csv(tables / "cmp_condition_numbers.csv", index=False)
    pd.DataFrame(info_rows).to_csv(tables / "cmp_estimator_info.csv", index=False)
    pd.DataFrame(stat_rows).to_csv(tables / "cmp_weight_stats.csv", index=False)
    (tables / "cmp_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    save_figure(plot_cov_spectra(spectra_panels), "cmp_eigenvalue_spectra", figures)
    save_figure(plot_sorted_weights(weight_panels, CORE_STRATEGIES), "cmp_sorted_weights", figures)
    logger.info("done; summary in %s, figures in %s", tables / "cmp_summary.txt", figures)


if __name__ == "__main__":
    main()
