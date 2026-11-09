"""Step 8.18: covariance forecast loss on random portfolios.

At every rebalance date each covariance estimator (sample, Ledoit-Wolf, RMT, PCA, nonlinear
shrinkage, PCA plus sector blocks) forecasts the variance of the same random long-only portfolios
(Dirichlet weights, diffuse and concentrated, plus equal weight); the forecast is scored against the
realized variance over the next holding period with the QLIKE loss. This tests the estimators
beyond their own minimum-variance portfolios. Sector labels are the static snapshot of
``sp500_companies.csv``, used as labels only.

Needs only the processed data. Writes ``forecastloss_dates.csv``, ``forecastloss_summary.csv`` and
``forecastloss_summary.txt`` to ``results/tables``.

Usage: ``python scripts/20_forecast_loss.py --config config.yaml``
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from pfopt.backtest.forecastloss import (
    EQUAL_WEIGHT,
    REFERENCE,
    forecast_loss_dates,
    summarize_loss,
)
from pfopt.config import load_config
from pfopt.data.cleaning import read_processed
from pfopt.data.loader import load_companies

logger = logging.getLogger("forecast_loss")


def main() -> None:
    """Compute the forecast losses and write the tables and a short summary."""
    parser = argparse.ArgumentParser(description="Covariance forecast loss on random portfolios.")
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    started = time.perf_counter()

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    root = config_path.parent
    frames = read_processed(root / cfg.paths.processed_dir)
    companies = load_companies(
        root / cfg.paths.raw_dir / cfg.data.companies_file, cfg.data.companies_columns
    )
    sectors = companies.set_index("Symbol")["Sector"]

    dates = forecast_loss_dates(
        frames["log_returns"],
        frames["simple_returns"],
        frames["dollar_volume"],
        sectors,
        cfg,
    )
    summary = summarize_loss(dates, cfg.forecast.hac_lag, cfg.stats.alpha)
    tables = root / cfg.paths.tables_dir
    tables.mkdir(parents=True, exist_ok=True)
    dates.to_csv(tables / "forecastloss_dates.csv", index=False)
    summary.to_csv(tables / "forecastloss_summary.csv", index=False)

    lines = [
        f"{dates['date'].nunique()} rebalance dates with a full {cfg.backtest.rebalance_days}-day "
        f"holding period, T={cfg.universe.lookback_days}, N={cfg.universe.n_assets}; "
        f"{cfg.forecast.n_portfolios} random long-only portfolios per Dirichlet family and date "
        "(identical for every estimator)",
        "QLIKE = RV/F - log(RV/F) - 1 (lower is better); bias = mean log(F/RV) (negative = risk "
        "under-forecast); diff = QLIKE minus the sample covariance's, Newey-West "
        f"lag {cfg.forecast.hac_lag}, * = significant after Holm within the family",
    ]
    for family, block in summary.groupby("family", sort=False):
        shown = block.set_index("estimator").drop(columns=["family", "n_dates"])
        star = (shown["p_holm"] < cfg.stats.alpha).map({True: "*", False: ""})
        shown["diff"] = shown["diff_qlike"].map("{:+.4f}".format) + star
        shown = shown[["mean_qlike", "mean_log_bias", "diff", "p_holm"]].round(4)
        label = f"{family} (one portfolio)" if family == EQUAL_WEIGHT else family
        lines += ["", f"{label}; reference {REFERENCE}:", shown.to_string()]
    path = tables / "forecastloss_summary.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("done in %.0f s; summary in %s", time.perf_counter() - started, path)


if __name__ == "__main__":
    main()
