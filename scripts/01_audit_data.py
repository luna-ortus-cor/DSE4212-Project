"""Step 1.1: audit the raw CSVs and save ``results/tables/data_audit.csv``.

Usage: ``python scripts/01_audit_data.py --config config.yaml``
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from pfopt.config import load_config
from pfopt.data.audit import run_audit
from pfopt.data.loader import load_companies, load_index, load_stocks

logger = logging.getLogger("audit")


def main() -> None:
    """Load the three raw files, audit them, write the table and log it."""
    parser = argparse.ArgumentParser(description="Audit the raw S&P 500 CSV files.")
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    root = config_path.parent
    raw = root / cfg.paths.raw_dir

    stocks = load_stocks(raw / cfg.data.stocks_file, cfg.data.stocks_columns)
    companies = load_companies(raw / cfg.data.companies_file, cfg.data.companies_columns)
    index = load_index(raw / cfg.data.index_file, cfg.data.index_columns)

    table = run_audit(stocks, companies, index, ffill_limit=cfg.data.ffill_limit)

    out = root / cfg.paths.tables_dir / "data_audit.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    logger.info("Audit table (%d rows) written to %s", len(table), out)
    logger.info("\n%s", table.to_string(index=False))


if __name__ == "__main__":
    main()
