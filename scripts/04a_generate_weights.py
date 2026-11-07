"""Step 4.1: generate leakage-free weights for all strategies on the real data.

Reads the processed return matrices, runs :func:`pfopt.backtest.engine.generate_weights` and caches
the result in ``results/cache/<config hash>/`` (reused unless ``--force`` is given). Writes a short
``results/tables/weights_summary.txt``. With ``--leakage-check`` it repeats the run on data whose
future (after the middle rebalance date) is replaced by noise and verifies that all earlier weights
are bit-identical (doubles the runtime).

Usage: ``python scripts/04a_generate_weights.py --config config.yaml [--force] [--leakage-check]``
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pfopt.backtest.covquality import weight_stats
from pfopt.backtest.engine import (
    ALL_STRATEGIES,
    WeightsResult,
    generate_weights,
    read_weights_result,
    write_weights_result,
)
from pfopt.config import load_config
from pfopt.data.cleaning import read_processed

logger = logging.getLogger("weights")


def _summary_lines(result: WeightsResult) -> list[str]:
    dates = result.diagnostics.index
    lines = [
        f"rebalance dates: {len(dates)} from {dates[0].date()} to {dates[-1].date()}",
        f"universe size: {int(result.diagnostics['n_universe'].min())} to "
        f"{int(result.diagnostics['n_universe'].max())}",
        f"solver fallbacks (rebalances with at least one): "
        f"{int((result.diagnostics['n_solver_fallbacks'] > 0).sum())}",
    ]
    for strategy in ALL_STRATEGIES:
        frame = result.weights[strategy]
        stats = pd.DataFrame([weight_stats(row) for row in frame.to_numpy()])
        change = 0.5 * np.abs(np.diff(frame.to_numpy(), axis=0)).sum(axis=1)
        lines.append(
            f"{strategy}: mean eff.N {stats['effective_n'].mean():.1f}, mean max weight "
            f"{stats['max_weight'].mean():.3f}, "
            f"mean positions>0.1% {stats['n_positions'].mean():.0f}, "
            f"mean one-way change between rebalances {change.mean():.3f}"
        )
    return lines


def main() -> None:
    """Generate (or load) the weights, write the summary, optionally run the leakage check."""
    parser = argparse.ArgumentParser(description="Generate leakage-free strategy weights.")
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    parser.add_argument("--force", action="store_true", help="ignore the cache and recompute")
    parser.add_argument("--leakage-check", action="store_true", help="also run the leakage check")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    root = config_path.parent
    cache = root / cfg.paths.cache_dir / cfg.cache_key()
    frames = read_processed(root / cfg.paths.processed_dir)
    log_returns, dollar_volume = frames["log_returns"], frames["dollar_volume"]

    if (cache / "diagnostics.parquet").exists() and not args.force:
        logger.info("loading cached weights from %s", cache)
        result = read_weights_result(cache)
    else:
        start = time.perf_counter()
        result = generate_weights(log_returns, dollar_volume, cfg)
        logger.info("weights generated in %.1f s", time.perf_counter() - start)
        write_weights_result(result, cache)
        logger.info("cached in %s", cache)

    tables = root / cfg.paths.tables_dir
    tables.mkdir(parents=True, exist_ok=True)
    lines = _summary_lines(result)
    (tables / "weights_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    result.diagnostics.to_csv(tables / "weights_diagnostics.csv", date_format="%Y-%m-%d")
    logger.info("summary written to %s", tables / "weights_summary.txt")

    if args.leakage_check:
        dates = result.diagnostics.index
        cutoff = dates[len(dates) // 2]
        rng = np.random.default_rng(cfg.run.seed)
        future = log_returns.index > cutoff
        bad_returns, bad_volume = log_returns.copy(), dollar_volume.copy()
        bad_returns.loc[future] = 0.02 * rng.normal(size=bad_returns.loc[future].shape)
        bad_volume.loc[future] = rng.uniform(size=bad_volume.loc[future].shape)
        corrupted = generate_weights(bad_returns, bad_volume, cfg)
        past = dates <= cutoff
        for strategy in ALL_STRATEGIES:
            before = result.weights[strategy].loc[past].to_numpy()
            after = (
                corrupted.weights[strategy].loc[corrupted.diagnostics.index <= cutoff].to_numpy()
            )
            if not np.array_equal(before, after):
                raise AssertionError(f"{strategy}: weights up to {cutoff.date()} changed")
        logger.info("leakage check PASSED up to %s (%d rebalance dates)", cutoff.date(), past.sum())


if __name__ == "__main__":
    main()
