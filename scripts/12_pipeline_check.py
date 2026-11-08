"""Step 8.20: validate the backtest pipeline end to end on synthetic panels with a known truth.

Draws ``pipeline_check.n_panels`` synthetic Date x Symbol panels from the factor model of the
simulation study, runs each through the unchanged engine and scores the weights and the P&L
against the true covariance (see ``pfopt.simulation.pipeline``). Per-panel scores are cached in
``results/cache/pipeline_check/<key>``; ``--force`` recomputes them.

Writes ``pipeline_check_accounting.csv``, ``pipeline_check_identity.csv``,
``pipeline_check_estimation.csv``, ``pipeline_check_estimation_dates.csv`` and
``pipeline_check_summary.txt`` to ``results/tables``.

Usage: ``python scripts/12_pipeline_check.py --config config.yaml [--force] [--n-panels P]``
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import time
from pathlib import Path

import pandas as pd

from pfopt.config import Config, load_config
from pfopt.simulation.pipeline import (
    accounting_summary,
    estimation_summary,
    identity_summary,
    run_panel,
)

logger = logging.getLogger("pipeline_check")


def _cache_key(cfg: Config) -> str:
    """Hash of everything that determines the synthetic panels and their scores."""
    full = cfg.to_dict()
    payload = {k: full[k] for k in ("simulation", "pipeline_check")}
    payload["pipeline"] = cfg.cache_key()
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def _run_or_load(cfg: Config, cache: Path, force: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    accounting_path, estimation_path = cache / "accounting.parquet", cache / "estimation.parquet"
    if accounting_path.exists() and estimation_path.exists() and not force:
        logger.info("loading cached panel scores from %s", cache)
        return pd.read_parquet(accounting_path), pd.read_parquet(estimation_path)
    accounting, estimation = [], []
    for panel in range(cfg.pipeline_check.n_panels):
        started = time.perf_counter()
        out = run_panel(cfg, panel)
        accounting.append(out["accounting"])
        estimation.append(out["estimation"])
        logger.info("panel %d finished in %.0f s", panel, time.perf_counter() - started)
    cache.mkdir(parents=True, exist_ok=True)
    result = (
        pd.concat(accounting, ignore_index=True),
        pd.concat(estimation, ignore_index=True),
    )
    result[0].to_parquet(accounting_path)
    result[1].to_parquet(estimation_path)
    return result


def _study_reference(tables: Path, lookback: int) -> pd.Series | None:
    """Paired log ratios of step 6.1 (Gaussian, long-only) at the engine's window length."""
    path = tables / "simulation_paired.csv"
    if not path.exists():
        return None
    paired = pd.read_csv(path)
    rows = paired[
        (paired["distribution"] == "gaussian")
        & (paired["constraint"] == "long-only")
        & (paired["T"] == lookback)
    ]
    return rows.set_index("method")["mean_log_ratio"] if len(rows) else None


def main() -> None:
    """Run (or load) the panels and write the tables and the summary."""
    parser = argparse.ArgumentParser(description="Synthetic-panel pipeline validation.")
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    parser.add_argument("--force", action="store_true", help="recompute the panels")
    parser.add_argument("--n-panels", type=int, help="override pipeline_check.n_panels")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    if args.n_panels is not None:
        check = dataclasses.replace(cfg.pipeline_check, n_panels=args.n_panels)
        cfg = dataclasses.replace(cfg, pipeline_check=check)
    root, pc = config_path.parent, cfg.pipeline_check
    cache = root / cfg.paths.cache_dir / "pipeline_check" / _cache_key(cfg)
    accounting, estimation = _run_or_load(cfg, cache, args.force)

    acc = accounting_summary(accounting, pc.z2_tolerance)
    identity = identity_summary(estimation, pc.identity_tolerance)
    ranking = estimation_summary(estimation, cfg.stats.alpha)
    tables = root / cfg.paths.tables_dir
    tables.mkdir(parents=True, exist_ok=True)
    acc.to_csv(tables / "pipeline_check_accounting.csv", index=False)
    identity.to_csv(tables / "pipeline_check_identity.csv", index=False)
    ranking.to_csv(tables / "pipeline_check_estimation.csv", index=False)
    estimation.to_csv(tables / "pipeline_check_estimation_dates.csv", index=False)

    study = _study_reference(tables, cfg.universe.lookback_days)
    ranking = ranking.set_index("strategy")
    ranking["study_6_1"] = study.reindex(ranking.index) if study is not None else float("nan")
    n_dates = estimation.groupby("panel")["date"].nunique().mean()
    ok = bool(acc["passes"].all() and identity["passes"].all())
    lines = [
        f"{pc.n_panels} synthetic panels of {pc.n_symbols} symbols x {pc.n_days} days; universe "
        f"N={cfg.universe.n_assets}, T={cfg.universe.lookback_days}, about {n_dates:.0f} "
        f"rebalance dates per panel; step 6.1 factor model, K={1 + cfg.simulation.n_groups}",
        f"VERDICT: accounting and identity checks {'PASS' if ok else 'FAIL'}",
        "",
        "1. accounting: mean of r^2 / (w' Sigma w) should be 1 "
        f"(pass if |z| <= {pc.z2_tolerance:g}); realized/predicted = realized volatility over the "
        "mean volatility forecast:",
        acc.round(4).to_string(index=False),
        "",
        f"2. identity with the simulation study (pass if max |diff| <= {pc.identity_tolerance:g}):",
        identity.to_string(index=False, float_format=lambda v: f"{v:.2e}"),
        "",
        "3. ranking: mean log(true variance / sample GMV true variance) over rebalance dates "
        f"(negative = beats sample); {100 * (1 - cfg.stats.alpha):.0f}% t-interval across panels;"
        " study_6_1 = same quantity from step 6.1 at the same T (Gaussian, long-only, different "
        "random model):",
        ranking.round(4).to_string(),
    ]
    summary = tables / "pipeline_check_summary.txt"
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("summary written to %s", summary)


if __name__ == "__main__":
    main()
