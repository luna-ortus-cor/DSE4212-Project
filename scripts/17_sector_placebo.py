"""Step 8.25: random-label placebo for the sector-block strategy on the real data.

``GMV-PCA-SECTOR`` of step 8.24 is rerun ``residual.n_placebo`` times with the known sector labels
shuffled among the universe members at every rebalance (group sizes and unknown members kept). If
the gain over ``GMV-PCA`` comes from sector structure, the true-label strategy should beat the
placebo draws; if any block restriction of the residual helps, the draws should gain as much.

Reads ``returns_net.parquet`` of ``02_run_backtest.py`` and the true-label weights cached by
``16_sector_block.py``. Writes ``sectorblock_placebo.csv`` and ``sectorblock_placebo_summary.txt``
to ``results/tables``. Each draw's net returns are cached.

Usage: ``python scripts/17_sector_placebo.py --config config.yaml [--force]``
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pfopt.backtest.engine import simulate
from pfopt.backtest.sectorblock import STRATEGY, generate_sector_block_weights
from pfopt.config import load_config
from pfopt.data.cleaning import read_processed
from pfopt.data.loader import load_companies

logger = logging.getLogger("sector_placebo")


def _log_var_ratio(net: pd.Series, reference: pd.Series) -> float:
    """Log ratio of the sample variances (twice the log volatility ratio)."""
    return float(np.log(net.var(ddof=1) / reference.var(ddof=1)))


def _vol_ratio(log_var_ratio: float) -> float:
    """Volatility ratio from a log variance ratio."""
    return float(np.exp(log_var_ratio / 2.0))


def main() -> None:
    """Run the placebo draws and write the table and the summary."""
    parser = argparse.ArgumentParser(description="Random-label placebo for the sector blocks.")
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    parser.add_argument("--force", action="store_true", help="recompute all draws")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    root = config_path.parent
    cache = root / cfg.paths.cache_dir / cfg.cache_key()
    true_path = cache / "sector_block" / "weights.parquet"
    if not (cache / "returns_net.parquet").exists() or not true_path.exists():
        raise SystemExit("run 02_run_backtest.py and 16_sector_block.py first")
    frames = read_processed(root / cfg.paths.processed_dir)
    companies = load_companies(
        root / cfg.paths.raw_dir / cfg.data.companies_file, cfg.data.companies_columns
    )
    sectors = companies.set_index("Symbol")["Sector"]
    core = pd.read_parquet(cache / "returns_net.parquet")
    true_net = simulate(
        pd.read_parquet(true_path), frames["simple_returns"], cfg.backtest.cost_bps
    ).net
    pca, sample = core["GMV-PCA"], core["GMV-SAMPLE"]

    folder = cache / "sector_placebo"
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for draw in range(cfg.residual.n_placebo):
        path = folder / f"draw_{draw:03d}.parquet"
        if path.exists() and not args.force:
            net = pd.read_parquet(path)["net"]
        else:
            started = time.perf_counter()
            weights = generate_sector_block_weights(
                frames["log_returns"],
                frames["dollar_volume"],
                sectors,
                cfg,
                placebo_draw=draw,
            ).weights
            net = simulate(weights, frames["simple_returns"], cfg.backtest.cost_bps).net
            net.rename("net").to_frame().to_parquet(path)
            logger.info("draw %d done in %.0f s", draw, time.perf_counter() - started)
        if not net.index.equals(pca.index):
            raise ValueError("placebo returns are not on the core dates")
        rows.append(
            {
                "draw": draw,
                "vol": float(net.std(ddof=1) * np.sqrt(cfg.run.annualization)),
                "log_var_vs_pca": _log_var_ratio(net, pca),
                "log_var_vs_sample": _log_var_ratio(net, sample),
            }
        )
    table = pd.DataFrame(rows)
    true = {
        "vol": float(true_net.std(ddof=1) * np.sqrt(cfg.run.annualization)),
        "log_var_vs_pca": _log_var_ratio(true_net, pca),
        "log_var_vs_sample": _log_var_ratio(true_net, sample),
    }
    tables = root / cfg.paths.tables_dir
    tables.mkdir(parents=True, exist_ok=True)
    table.to_csv(tables / "sectorblock_placebo.csv", index=False)

    n = len(table)
    vs_pca = table["log_var_vs_pca"]
    # one-sided rank p-value: how often a random labelling is at least as good as the true one
    p_rank = (1 + int((vs_pca <= true["log_var_vs_pca"]).sum())) / (n + 1)
    gain_true = -true["log_var_vs_pca"]
    gain_placebo = -float(vs_pca.mean())
    ratio = _vol_ratio
    share = (
        f"share of the true-label gain over PCA reached by random labels (mean): "
        f"{100 * gain_placebo / gain_true:.0f}%"
        if gain_true > 0
        else "true labels do not beat PCA"
    )
    lines = [
        f"{n} random-label draws (known labels shuffled among members at every rebalance; group "
        f"sizes and unknown members kept); true labels = the 8.24 strategy {STRATEGY}",
        "",
        "volatility ratio against GMV-PCA (below 1 = lower volatility than PCA):",
        f"  true sector labels   {ratio(true['log_var_vs_pca']):.3f}",
        f"  random labels        mean {ratio(float(vs_pca.mean())):.3f}, "
        f"min {ratio(float(vs_pca.min())):.3f}, max {ratio(float(vs_pca.max())):.3f}",
        "volatility ratio against GMV-SAMPLE:",
        f"  true sector labels   {ratio(true['log_var_vs_sample']):.3f}",
        f"  random labels        mean {ratio(float(table['log_var_vs_sample'].mean())):.3f}, "
        f"min {ratio(float(table['log_var_vs_sample'].min())):.3f}, "
        f"max {ratio(float(table['log_var_vs_sample'].max())):.3f}",
        "",
        share,
        f"rank p-value (random labelling at least as good as the true one): {p_rank:.3f} "
        f"(smallest possible {1 / (n + 1):.3f}; a single data path, draws share the same returns)",
    ]
    path = tables / "sectorblock_placebo_summary.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("summary written to %s", path)


if __name__ == "__main__":
    main()
