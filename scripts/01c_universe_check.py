"""Inspect the point-in-time universe on the real data.

Reads the processed parquet files, writes ``results/tables/universe_eligible_counts.csv``, logs the
number of eligible symbols over time, the first feasible date, example universes at three dates,
and runs a leakage self-check on the real data (corrupting everything after a middle date must not
change the universe chosen at that date).

Usage: ``python scripts/01c_universe_check.py --config config.yaml``
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from pfopt.config import load_config
from pfopt.data.cleaning import read_processed
from pfopt.data.universe import eligible_counts, first_feasible_date, select_universe

logger = logging.getLogger("universe")
_SHOWN = 10


def main() -> None:
    """Report eligibility over time and example universes, and self-check for look-ahead."""
    parser = argparse.ArgumentParser(description="Inspect the point-in-time universe.")
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    root = config_path.parent
    n, t, w = cfg.universe.n_assets, cfg.universe.lookback_days, cfg.universe.dollar_volume_window

    frames = read_processed(root / cfg.paths.processed_dir)
    log_returns, dollar_volume = frames["log_returns"], frames["dollar_volume"]

    counts = eligible_counts(log_returns, t)
    tables = root / cfg.paths.tables_dir
    tables.mkdir(parents=True, exist_ok=True)
    counts_path = tables / "universe_eligible_counts.csv"
    counts.rename_axis("Date").to_csv(counts_path, date_format="%Y-%m-%d")
    logger.info(
        "eligible symbols (T=%d): min=%d median=%d max=%d over %d dates, first %s last %s",
        t,
        counts.min(),
        int(counts.median()),
        counts.max(),
        len(counts),
        counts.index[0].date(),
        counts.index[-1].date(),
    )

    first = first_feasible_date(log_returns, t, n)
    dates = counts.index[counts.index >= first]
    middle = dates[len(dates) // 2]
    logger.info("first date with >= %d eligible symbols: %s", n, first.date())
    since = counts.loc[dates]
    logger.info(
        "dates since then with fewer than %d eligible symbols: %d of %d",
        n,
        int((since < n).sum()),
        len(since),
    )

    chosen = {}
    for label, day in (("first", first), ("middle", middle), ("last", dates[-1])):
        universe = select_universe(log_returns, dollar_volume, day, t, n, w)
        chosen[label] = set(universe)
        logger.info(
            "%s feasible date %s: eligible=%d, universe size=%d, top %d by dollar volume: %s",
            label,
            day.date(),
            int(counts.loc[day]),
            len(universe),
            _SHOWN,
            universe[:_SHOWN],
        )
    logger.info("overlap first vs middle: %d of %d", len(chosen["first"] & chosen["middle"]), n)
    logger.info("overlap middle vs last: %d of %d", len(chosen["middle"] & chosen["last"]), n)
    logger.info("overlap first vs last: %d of %d", len(chosen["first"] & chosen["last"]), n)

    rng = np.random.default_rng(cfg.run.seed)
    future = log_returns.index > middle
    corrupted_returns = log_returns.copy()
    corrupted_volume = dollar_volume.copy()
    corrupted_returns.loc[future] = rng.normal(size=corrupted_returns.loc[future].shape)
    corrupted_volume.loc[future] = rng.uniform(size=corrupted_volume.loc[future].shape)
    before = select_universe(log_returns, dollar_volume, middle, t, n, w)
    after = select_universe(corrupted_returns, corrupted_volume, middle, t, n, w)
    if before != after:
        raise AssertionError("universe at the middle date changed when future data was corrupted")
    logger.info("leakage self-check PASSED at %s (future data corrupted)", middle.date())


if __name__ == "__main__":
    main()
