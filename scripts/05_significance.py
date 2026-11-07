"""Step 5.1: bootstrap significance tests of the net-of-cost strategy returns.

Reads ``returns_net.parquet`` from the cache written by ``02_run_backtest.py`` and writes
``significance_tests.csv`` (main block length), ``significance_block_sensitivity.csv`` (all block
lengths) and a short ``significance_summary.txt`` to ``results/tables``. Settings come from the
``stats`` section of ``config.yaml``.

Usage: ``python scripts/05_significance.py --config config.yaml``
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pfopt.backtest.stats import block_length_sensitivity, compare_to_references
from pfopt.config import load_config

logger = logging.getLogger("significance")


def _sharpe_lines(table: pd.DataFrame, reference: str) -> list[str]:
    rows = table[(table.reference == reference) & (table.test == "sharpe_diff")]
    lines = [f"Sharpe difference vs {reference} (net): estimate [95% CI], p-value"]
    for _, r in rows.iterrows():
        flag = "*" if r["significant"] else " "
        lines.append(
            f"  {r['strategy']:<11} {r['estimate']:6.2f} [{r['ci_low']:6.2f}, {r['ci_high']:6.2f}]"
            f"  p={r['p_value']:.3f} {flag}"
        )
    return lines


def _volatility_lines(table: pd.DataFrame, reference: str) -> list[str]:
    rows = table[(table.reference == reference) & (table.test == "log_var_ratio")]
    lines = [f"Volatility ratio vs {reference} (strategy / reference): ratio [95% CI], p-value"]
    for _, r in rows.iterrows():
        flag = "*" if r["significant"] else " "
        ratio, low, high = (float(np.exp(r[c] / 2.0)) for c in ("estimate", "ci_low", "ci_high"))
        lines.append(
            f"  {r['strategy']:<11} {ratio:6.3f} [{low:6.3f}, {high:6.3f}]  p={r['p_value']:.3f} {flag}"
        )
    return lines


def main() -> None:
    """Run the bootstrap tests and write the tables and the summary."""
    parser = argparse.ArgumentParser(description="Bootstrap significance tests.")
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    started = time.perf_counter()

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    root = config_path.parent
    path = root / cfg.paths.cache_dir / cfg.cache_key() / "returns_net.parquet"
    if not path.exists():
        raise SystemExit(f"{path} not found; run scripts/02_run_backtest.py first")
    returns = pd.read_parquet(path)
    s = cfg.stats
    common = {
        "n_boot": s.n_boot,
        "seed": cfg.run.seed,
        "annualization": cfg.run.annualization,
        "rf": cfg.run.rf,
        "alpha": s.alpha,
    }

    main_table = compare_to_references(
        returns, s.references, block_length=s.block_length, **common
    )
    sensitivity = block_length_sensitivity(
        returns, s.references, block_lengths=s.sensitivity_block_lengths, **common
    )
    tables = root / cfg.paths.tables_dir
    tables.mkdir(parents=True, exist_ok=True)
    main_table.to_csv(tables / "significance_tests.csv", index=False)
    sensitivity.to_csv(tables / "significance_block_sensitivity.csv", index=False)

    key = ["reference", "strategy", "test"]
    flags = sensitivity.groupby(key)["significant"].agg(["min", "max"])
    unstable = flags[flags["min"] != flags["max"]]
    lines = [
        f"{len(returns)} daily net returns, block length {s.block_length}, B={s.n_boot}, "
        f"alpha={s.alpha}; * = significant at alpha",
        f"hypothesis tests in the main table: {len(main_table)} "
        f"(no multiple-testing correction applied); in the sensitivity table: {len(sensitivity)}",
        "",
        *_sharpe_lines(main_table, "EW"),
        "",
        *_sharpe_lines(main_table, "GMV-SAMPLE"),
        "",
        *_volatility_lines(main_table, "GMV-SAMPLE"),
        "",
        *_volatility_lines(main_table, "EW"),
        "",
        f"tests whose significance flips with the block length {tuple(s.sensitivity_block_lengths)}: "
        f"{len(unstable)} of {len(flags)}",
    ]
    for (reference, strategy, test), _ in unstable.iterrows():
        lines.append(f"  {test}: {strategy} vs {reference}")
    (tables / "significance_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("done in %.0f s; summary in %s", time.perf_counter() - started, tables / "significance_summary.txt")


if __name__ == "__main__":
    main()
