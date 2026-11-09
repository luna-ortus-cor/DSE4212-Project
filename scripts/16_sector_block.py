"""Step 8.24: PCA plus sector-block residual covariance (direct test of the sector explanation).

Real data: long-only GMV with the sector-block covariance (``GMV-PCA-SECTOR``) on exactly the dates
and universes of the core strategies, tested against ``GMV-PCA`` (only the blocks differ) and
``GMV-SAMPLE`` with the studentized bootstrap of step 8.21. Sector labels are the static snapshot
of ``sp500_companies.csv``, used as labels only. Positive and negative controls: the same
estimators in simulations with and without weak group factors, with true and partly wrong labels.

Reads ``returns_net.parquet`` written by ``02_run_backtest.py``. Writes
``sectorblock_real_metrics.csv``, ``sectorblock_real_tests.csv``, ``sectorblock_sim_paired.csv``,
``sectorblock_sim_vs_pca.csv`` and ``sectorblock_summary.txt`` to ``results/tables``. Weights and
simulation scores are cached.

Usage: ``python scripts/16_sector_block.py --config config.yaml [--force]``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pfopt.backtest.engine import simulate
from pfopt.backtest.inference import studentized_compare
from pfopt.backtest.metrics import performance_metrics
from pfopt.backtest.sectorblock import STRATEGY, SectorBlockResult, generate_sector_block_weights
from pfopt.config import Config, load_config
from pfopt.data.cleaning import read_processed
from pfopt.data.loader import load_companies
from pfopt.simulation.sectorblock import DESIGNS, paired_control, paired_vs_pca, run_control

logger = logging.getLogger("sector_block")
CORE = ("GMV-SAMPLE", "GMV-LW", "GMV-RMT", "GMV-PCA")


def _load_or_generate(
    cfg: Config,
    folder: Path,
    frames: dict[str, pd.DataFrame],
    sectors: pd.Series,
    force: bool,
) -> SectorBlockResult:
    marker = folder / "diagnostics.parquet"
    if marker.exists() and not force:
        logger.info("loading cached sector-block weights from %s", folder)
        return SectorBlockResult(
            weights=pd.read_parquet(folder / "weights.parquet"),
            diagnostics=pd.read_parquet(marker),
        )
    started = time.perf_counter()
    result = generate_sector_block_weights(
        frames["log_returns"], frames["dollar_volume"], sectors, cfg
    )
    logger.info("sector-block weights generated in %.0f s", time.perf_counter() - started)
    folder.mkdir(parents=True, exist_ok=True)
    result.weights.to_parquet(folder / "weights.parquet")
    result.diagnostics.to_parquet(marker)
    return result


def _sim_key(cfg: Config) -> str:
    full = cfg.to_dict()
    payload = {k: full[k] for k in ("simulation", "residual", "pca", "hrp")}
    payload["run"] = {"seed": cfg.run.seed, "annualization": cfg.run.annualization}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def _control_lines(paired: pd.DataFrame, versus_pca: pd.DataFrame, sizes: tuple[int, ...]) -> list:
    def table(frame: pd.DataFrame, design: str) -> str:
        rows = frame[frame["design"] == design].copy()
        star = (rows["ci_low"] > 0) | (rows["ci_high"] < 0)
        rows["cell"] = rows["mean_log_ratio"].map("{:+.3f}".format) + np.where(star, "*", " ")
        return rows.pivot(index="method", columns="T", values="cell")[list(sizes)].to_string()

    lines: list[str] = []
    for design in DESIGNS:
        others = paired[paired["method"] != "GMV-SAMPLE"]
        lines += [f"{design}, against GMV-SAMPLE:", table(others, design)]
        lines += [f"{design}, against GMV-PCA:", table(versus_pca, design), ""]
    return lines


def main() -> None:
    """Run the real-data strategy and the controls and write the tables and the summary."""
    parser = argparse.ArgumentParser(description="PCA plus sector-block covariance test.")
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    parser.add_argument("--force", action="store_true", help="recompute weights and simulation")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    root, s = config_path.parent, cfg.stats
    cache = root / cfg.paths.cache_dir / cfg.cache_key()
    if not (cache / "returns_net.parquet").exists():
        raise SystemExit(f"{cache / 'returns_net.parquet'} not found; run 02_run_backtest.py first")
    ann, rf = cfg.run.annualization, cfg.run.rf
    frames = read_processed(root / cfg.paths.processed_dir)
    companies = load_companies(
        root / cfg.paths.raw_dir / cfg.data.companies_file, cfg.data.companies_columns
    )
    sectors = companies.set_index("Symbol")["Sector"]

    # ---- real data -------------------------------------------------------------------------
    result = _load_or_generate(cfg, cache / "sector_block", frames, sectors, args.force)
    sim = simulate(result.weights, frames["simple_returns"], cfg.backtest.cost_bps)
    core_net = pd.read_parquet(cache / "returns_net.parquet")[list(CORE)]
    net = pd.concat([core_net, sim.net.rename(STRATEGY)], axis=1)
    if net.isna().any().any():
        raise ValueError("core and sector-block returns are not on the same dates")
    core_diag = pd.read_parquet(cache / "diagnostics.parquet")
    core_turnover = pd.read_parquet(cache / "turnover.parquet")
    rows = {}
    for strategy in (*CORE, STRATEGY):
        new = strategy == STRATEGY
        turnover = sim.turnover if new else core_turnover[strategy]
        values = performance_metrics(net[strategy], turnover, ann, rf)
        pred = float(
            (result.diagnostics["pred_vol"] if new else core_diag[f"pred_vol_{strategy}"]).mean()
        )
        rows[strategy] = {
            "ann_vol": values["ann_vol"],
            "ann_return": values["ann_return"],
            "sharpe": values["sharpe"],
            "max_drawdown": values["max_drawdown"],
            "annual_turnover": values["annual_turnover"],
            "mean_pred_vol": pred,
            "realized_over_pred": values["ann_vol"] / pred,
        }
    metrics = pd.DataFrame.from_dict(rows, orient="index").rename_axis("strategy")
    tests = studentized_compare(
        net,
        ("GMV-PCA", "GMV-SAMPLE"),
        block_length=s.block_length,
        n_boot=s.n_boot,
        seed=cfg.run.seed,
        annualization=ann,
        rf=rf,
        alpha=s.alpha,
        power=s.mde_power,
    )

    # ---- controls --------------------------------------------------------------------------
    sim_cache = root / cfg.paths.cache_dir / "sector_block_sim" / _sim_key(cfg)
    scores_path = sim_cache / "scores.parquet"
    if scores_path.exists() and not args.force:
        logger.info("loading cached simulation scores from %s", sim_cache)
        scores = pd.read_parquet(scores_path)
    else:
        parts = []
        for structured in (True, False):
            started = time.perf_counter()
            parts.append(run_control(cfg, structured))
            logger.info("control finished in %.0f s", time.perf_counter() - started)
        scores = pd.concat(parts, ignore_index=True)
        sim_cache.mkdir(parents=True, exist_ok=True)
        scores.to_parquet(scores_path)
    paired = paired_control(scores, cfg.simulation.alpha)
    versus_pca = paired_vs_pca(scores, cfg.simulation.alpha)

    # ---- outputs ---------------------------------------------------------------------------
    tables = root / cfg.paths.tables_dir
    tables.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(tables / "sectorblock_real_metrics.csv")
    tests.to_csv(tables / "sectorblock_real_tests.csv", index=False)
    paired.to_csv(tables / "sectorblock_sim_paired.csv", index=False)
    versus_pca.to_csv(tables / "sectorblock_sim_vs_pca.csv", index=False)

    vol = tests[tests["test"] == "log_var_ratio"].set_index(["reference", "strategy"])
    sharpe = tests[tests["test"] == "sharpe_diff"].set_index(["reference", "strategy"])

    def line(reference: str, strategy: str) -> str:
        v, d = vol.loc[(reference, strategy)], sharpe.loc[(reference, strategy)]
        ratio, low, high = (float(np.exp(v[c] / 2.0)) for c in ("estimate", "ci_low", "ci_high"))
        flag = "*" if v["p_holm"] < s.alpha else " "
        return (
            f"  {strategy:<15} vs {reference:<10} vol ratio {ratio:6.3f} [{low:6.3f}, {high:6.3f}] "
            f"p={v['p_value']:.3f} Holm {v['p_holm']:.3f} {flag} detectable {v['mde_vol_pct']:.1f}%"
            f" | Sharpe diff {d['estimate']:+.2f} [{d['ci_low']:+.2f}, {d['ci_high']:+.2f}]"
        )

    diag = result.diagnostics
    lines = [
        f"{len(net)} daily net returns, {len(result.weights)} rebalances, window "
        f"T={cfg.universe.lookback_days}, N={cfg.universe.n_assets}; sector labels are a current "
        f"snapshot used as labels only ({diag['n_known_sector'].mean():.0f} of "
        f"{diag['n_universe'].mean():.0f} members have a known sector on average)",
        f"K (same rule as PCA): mean {diag['K'].mean():.1f}; share of the residual's squared "
        f"off-diagonal mass inside the kept blocks: mean {diag['block_share'].mean():.2f}",
        "",
        "REAL DATA (long-only GMV, net of costs)",
        metrics.round(3).to_string(),
        "",
        f"volatility ratios (studentized block bootstrap, block {s.block_length}, B={s.n_boot}; "
        "* = significant after Holm within the reference):",
        line("GMV-PCA", STRATEGY),
        line("GMV-SAMPLE", STRATEGY),
        "",
        f"POSITIVE / NEGATIVE CONTROL: paired log(true variance ratio), long-only, Gaussian, "
        f"{cfg.residual.sim_n_reps} replications, {cfg.residual.sim_n_groups} groups, "
        f"{100 * cfg.residual.sim_label_error:.0f}% wrong labels in the NOISY variant "
        "(negative = lower true variance; * = interval excludes 0):",
        *_control_lines(paired, versus_pca, cfg.residual.sim_sample_sizes),
    ]
    path = tables / "sectorblock_summary.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("summary written to %s", path)


if __name__ == "__main__":
    main()
