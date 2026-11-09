"""Step 8.2: nonlinear shrinkage (quadratic-inverse shrinkage, QIS) against RMT clipping.

Real data: long-only GMV with the QIS covariance (``GMV-NLS``) on exactly the dates and universes of
the core strategies, tested against ``GMV-SAMPLE`` and ``GMV-RMT`` (hard threshold versus smooth
shrinkage of the eigenvalues) with the studentized bootstrap of step 8.21. Simulation: the
factor-model study of step 6.1 (stationary) and 6.3 (time-varying) rerun with NLS added, scored by
the covariance error and the true variance of the GMV weights relative to the oracle.

Reads ``returns_net.parquet`` written by ``02_run_backtest.py``. Writes ``nls_real_metrics.csv``,
``nls_real_tests.csv``, ``nls_sim_paired.csv``, ``nls_sim_frobenius.csv`` and ``nls_summary.txt`` to
``results/tables``. Weights and simulation scores are cached.

Usage: ``python scripts/19_nonlinear_shrinkage.py --config config.yaml [--force]``
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
from pfopt.backtest.nonlinear import STRATEGY, NlsResult, generate_nls_weights
from pfopt.config import Config, load_config
from pfopt.data.cleaning import read_processed
from pfopt.simulation.nls import METHODS, SCENARIOS, run_nls_grid
from pfopt.simulation.run import COVARIANCE, LONG_ONLY, UNCONSTRAINED, paired_vs_sample

logger = logging.getLogger("nonlinear_shrinkage")
CORE = ("GMV-SAMPLE", "GMV-LW", "GMV-RMT", "GMV-PCA")
REFERENCES = ("GMV-SAMPLE", "GMV-RMT")


def _load_or_generate(
    cfg: Config, folder: Path, frames: dict[str, pd.DataFrame], force: bool
) -> NlsResult:
    marker = folder / "diagnostics.parquet"
    if marker.exists() and not force:
        logger.info("loading cached NLS weights from %s", folder)
        return NlsResult(
            weights=pd.read_parquet(folder / "weights.parquet"),
            diagnostics=pd.read_parquet(marker),
        )
    started = time.perf_counter()
    result = generate_nls_weights(frames["log_returns"], frames["dollar_volume"], cfg)
    logger.info("NLS weights generated in %.0f s", time.perf_counter() - started)
    folder.mkdir(parents=True, exist_ok=True)
    result.weights.to_parquet(folder / "weights.parquet")
    result.diagnostics.to_parquet(marker)
    return result


def _sim_key(cfg: Config) -> str:
    full = cfg.to_dict()
    payload = {k: full[k] for k in ("simulation", "pca", "hrp")}
    payload["run"] = {"seed": cfg.run.seed, "annualization": cfg.run.annualization}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def _paired_lines(paired: pd.DataFrame, sizes: tuple[int, ...]) -> list[str]:
    lines: list[str] = []
    for scenario in SCENARIOS:
        for constraint in (LONG_ONLY, UNCONSTRAINED):
            cell = paired[
                (paired["distribution"] == scenario)
                & (paired["constraint"] == constraint)
                & (paired["method"] != "GMV-SAMPLE")
            ].copy()
            star = (cell["ci_low"] > 0) | (cell["ci_high"] < 0)
            cell["text"] = cell["mean_log_ratio"].map("{:+.3f}".format) + np.where(star, "*", " ")
            table = cell.pivot(index="method", columns="T", values="text")[list(sizes)]
            lines += [f"{scenario}, {constraint} GMV, against GMV-SAMPLE:", table.to_string(), ""]
    return lines


def main() -> None:
    """Run the real-data strategy and the simulation and write the tables and the summary."""
    parser = argparse.ArgumentParser(description="Nonlinear shrinkage (QIS) test.")
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

    # ---- real data -------------------------------------------------------------------------
    result = _load_or_generate(cfg, cache / "nls", frames, args.force)
    sim = simulate(result.weights, frames["simple_returns"], cfg.backtest.cost_bps)
    net = pd.concat(
        [pd.read_parquet(cache / "returns_net.parquet")[list(CORE)], sim.net.rename(STRATEGY)],
        axis=1,
    )
    if net.isna().any().any():
        raise ValueError("core and NLS returns are not on the same dates")
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
        REFERENCES,
        block_length=s.block_length,
        n_boot=s.n_boot,
        seed=cfg.run.seed,
        annualization=ann,
        rf=rf,
        alpha=s.alpha,
        power=s.mde_power,
    )

    # ---- simulation ------------------------------------------------------------------------
    sim_cache = root / cfg.paths.cache_dir / "nls_sim" / _sim_key(cfg)
    scores_path = sim_cache / "scores.parquet"
    if scores_path.exists() and not args.force:
        logger.info("loading cached simulation scores from %s", sim_cache)
        scores = pd.read_parquet(scores_path)
    else:
        parts = []
        for scenario in SCENARIOS:
            started = time.perf_counter()
            parts.append(run_nls_grid(cfg, scenario))
            logger.info("%s finished in %.0f s", scenario, time.perf_counter() - started)
        scores = pd.concat(parts, ignore_index=True)
        sim_cache.mkdir(parents=True, exist_ok=True)
        scores.to_parquet(scores_path)
    paired = paired_vs_sample(scores, cfg.simulation.alpha)
    paired = paired[paired["method"].isin(METHODS)]
    frobenius = (
        scores[scores["constraint"] == COVARIANCE]
        .groupby(["distribution", "T", "method"])["frob_error"]
        .mean()
        .reset_index()
    )

    # ---- outputs ---------------------------------------------------------------------------
    tables = root / cfg.paths.tables_dir
    tables.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(tables / "nls_real_metrics.csv")
    tests.to_csv(tables / "nls_real_tests.csv", index=False)
    paired.to_csv(tables / "nls_sim_paired.csv", index=False)
    frobenius.to_csv(tables / "nls_sim_frobenius.csv", index=False)

    vol = tests[tests["test"] == "log_var_ratio"].set_index(["reference", "strategy"])
    sharpe = tests[tests["test"] == "sharpe_diff"].set_index(["reference", "strategy"])

    def line(reference: str, strategy: str) -> str:
        v, d = vol.loc[(reference, strategy)], sharpe.loc[(reference, strategy)]
        ratio, low, high = (float(np.exp(v[c] / 2.0)) for c in ("estimate", "ci_low", "ci_high"))
        flag = "*" if v["p_holm"] < s.alpha else " "
        return (
            f"  {strategy:<11} vs {reference:<10} vol ratio {ratio:6.3f} [{low:6.3f}, {high:6.3f}] "
            f"p={v['p_value']:.3f} Holm {v['p_holm']:.3f} {flag} detectable {v['mde_vol_pct']:.1f}%"
            f" | Sharpe diff {d['estimate']:+.2f} [{d['ci_low']:+.2f}, {d['ci_high']:+.2f}]"
        )

    diag = result.diagnostics
    sizes = cfg.simulation.sample_sizes
    frob_text = frobenius.pivot_table(
        index=["distribution", "method"], columns="T", values="frob_error"
    )[list(sizes)]
    lines = [
        f"{len(net)} daily net returns, {len(result.weights)} rebalances, window "
        f"T={cfg.universe.lookback_days}, N={cfg.universe.n_assets}; QIS bandwidth h mean "
        f"{diag['h'].mean():.3f}, condition number mean {diag['cond'].mean():.0f}",
        "",
        "REAL DATA (long-only GMV, net of costs)",
        metrics.round(3).to_string(),
        "",
        f"volatility ratios (studentized block bootstrap, block {s.block_length}, B={s.n_boot}; "
        "* = significant after Holm within the reference):",
        *[line(ref, STRATEGY) for ref in REFERENCES],
        "",
        f"SIMULATION (known factor model, {cfg.simulation.n_reps} replications; paired log true "
        "variance ratio against GMV-SAMPLE, negative = lower true variance; * = interval "
        "excludes 0):",
        *_paired_lines(paired, sizes),
        "mean relative Frobenius error of the covariance:",
        frob_text.round(3).to_string(),
    ]
    path = tables / "nls_summary.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("summary written to %s", path)


if __name__ == "__main__":
    main()
