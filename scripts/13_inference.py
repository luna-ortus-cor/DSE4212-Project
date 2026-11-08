"""Step 8.21: studentized bootstrap tests, multiple-testing adjustment and detectable effects.

Reads ``returns_net.parquet`` from the cache written by ``02_run_backtest.py`` (as step 5.1 does)
and writes ``inference_tests.csv`` (studentized tests with Holm / Benjamini-Hochberg adjusted
p-values and the minimum detectable effect, plus the percentile-bootstrap p-value of step 5.1 for
comparison) and ``inference_summary.txt`` to ``results/tables``. Settings come from ``stats``.

Usage: ``python scripts/13_inference.py --config config.yaml``
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pfopt.backtest.inference import studentized_compare
from pfopt.backtest.stats import compare_to_references
from pfopt.config import load_config

logger = logging.getLogger("inference")
KEYS = ["reference", "strategy", "test"]


def _lines(table: pd.DataFrame, reference: str, test: str, alpha: float) -> list[str]:
    rows = table[(table["reference"] == reference) & (table["test"] == test)]
    if test == "sharpe_diff":
        head = f"Sharpe difference vs {reference}: estimate [CI] (se), p / Holm / BH, detectable"
    else:
        head = f"Volatility ratio vs {reference}: ratio [CI], p / Holm / BH, detectable change"
    lines = [head]
    for _, r in rows.iterrows():
        flag = "*" if r["p_holm"] < alpha else " "
        tail = f"p={r['p_value']:.3f} / {r['p_holm']:.3f} / {r['p_bh']:.3f} {flag}"
        if test == "sharpe_diff":
            lines.append(
                f"  {r['strategy']:<11} {r['estimate']:6.2f} [{r['ci_low']:6.2f}, "
                f"{r['ci_high']:6.2f}] (se {r['se']:.2f})  {tail}  mde {r['mde']:.2f}"
            )
        else:
            cols = ("estimate", "ci_low", "ci_high")
            ratio, low, high = (float(np.exp(r[c] / 2.0)) for c in cols)
            lines.append(
                f"  {r['strategy']:<11} {ratio:6.3f} [{low:6.3f}, {high:6.3f}]  {tail}  "
                f"mde {r['mde_vol_pct']:.1f}%"
            )
    return lines


def main() -> None:
    """Run the studentized tests and write the table and the summary."""
    parser = argparse.ArgumentParser(description="Studentized bootstrap inference.")
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
        "block_length": s.block_length,
        "n_boot": s.n_boot,
        "seed": cfg.run.seed,
        "annualization": cfg.run.annualization,
        "rf": cfg.run.rf,
        "alpha": s.alpha,
    }
    table = studentized_compare(returns, s.references, power=s.mde_power, **common)
    percentile = compare_to_references(returns, s.references, **common)
    table = table.merge(
        percentile[[*KEYS, "p_value", "significant"]].rename(
            columns={"p_value": "p_percentile", "significant": "significant_percentile"}
        ),
        on=KEYS,
    )
    tables = root / cfg.paths.tables_dir
    tables.mkdir(parents=True, exist_ok=True)
    table.to_csv(tables / "inference_tests.csv", index=False)

    n = len(table)
    lines = [
        f"{len(returns)} daily net returns, block length {s.block_length}, B={s.n_boot}, "
        f"alpha={s.alpha}, power {s.mde_power:g}; * = significant after Holm within each "
        "(reference, test) family",
        f"{n} comparisons; significant at alpha: "
        f"percentile {int(table.significant_percentile.sum())}, "
        f"studentized {int(table.significant.sum())}, Holm {int(table.significant_holm.sum())}, "
        f"BH {int((table.p_bh < s.alpha).sum())}",
        "mde = true difference detected with the stated power (Sharpe units, or change of the "
        "volatility ratio); a smaller non-significant difference is not evidence of equality",
        "",
    ]
    for reference in s.references:
        for test in ("sharpe_diff", "log_var_ratio"):
            lines += [*_lines(table, reference, test, s.alpha), ""]
    flipped = table[table.significant != table.significant_percentile]
    lines.append(
        f"tests whose significance differs between percentile and studentized: {len(flipped)} "
        f"of {n}"
    )
    for _, r in flipped.iterrows():
        lines.append(f"  {r['test']}: {r['strategy']} vs {r['reference']}")
    (tables / "inference_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(
        "done in %.0f s; summary in %s",
        time.perf_counter() - started,
        tables / "inference_summary.txt",
    )


if __name__ == "__main__":
    main()
