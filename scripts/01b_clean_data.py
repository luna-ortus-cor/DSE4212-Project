"""Clean the raw stock panel and write return/volume matrices.

Writes ``data/processed/{log_returns,simple_returns,volume,dollar_volume}.parquet`` and the tables
``results/tables/{cleaning_log,dropped_symbols,suspect_returns}.csv``.

Usage: ``python scripts/01b_clean_data.py --config config.yaml``
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from pfopt.config import load_config
from pfopt.data.cleaning import clean_prices, write_processed
from pfopt.data.loader import load_stocks

logger = logging.getLogger("clean")
_NUM_EXAMPLES = 15


def main() -> None:
    """Load, clean, write parquet and tables, and log shapes, NaN counts and examples."""
    parser = argparse.ArgumentParser(description="Clean prices and build return matrices.")
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    root = config_path.parent

    stocks = load_stocks(root / cfg.paths.raw_dir / cfg.data.stocks_file, cfg.data.stocks_columns)
    result = clean_prices(
        stocks,
        start_date=cfg.data.start_date,
        end_date=cfg.data.end_date,
        ffill_limit=cfg.data.ffill_limit,
        max_missing_frac=cfg.data.max_missing_frac,
        suspect_abs_logret=cfg.data.suspect_abs_logret,
    )

    paths = write_processed(result, root / cfg.paths.processed_dir)
    tables = root / cfg.paths.tables_dir
    tables.mkdir(parents=True, exist_ok=True)
    result.log.to_csv(tables / "cleaning_log.csv", index=False)
    result.dropped.to_csv(tables / "dropped_symbols.csv", index=False)
    result.suspect.to_csv(tables / "suspect_returns.csv", index=False, date_format="%Y-%m-%d")

    for path in paths:
        logger.info("wrote %s", path)
    for name in ("log_returns", "simple_returns", "volume", "dollar_volume"):
        frame = getattr(result, name)
        logger.info(
            "%s: shape=%s, NaN=%d, index monotonic=%s",
            name,
            frame.shape,
            int(frame.isna().sum().sum()),
            frame.index.is_monotonic_increasing,
        )
    logger.info("cleaning log:\n%s", result.log.to_string(index=False))
    logger.info("dropped symbols (first 10):\n%s", result.dropped.head(10).to_string(index=False))
    examples = result.suspect.sort_values(
        "log_return", key=lambda s: s.abs(), ascending=False
    ).head(_NUM_EXAMPLES)
    logger.info("largest suspect returns:\n%s", examples.to_string(index=False))


if __name__ == "__main__":
    main()
