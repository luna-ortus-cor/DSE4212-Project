"""Step 8.22: non-stationarity test (EWMA strategies, volatility calibration, positive control).

Real data: long-only GMV with the exponentially weighted covariance and its RMT-clipped version
(half-lives from ``nonstationarity.halflives``), on exactly the dates and universes of the core
strategies, tested against sample GMV with the studentized bootstrap of step 8.21. Calibration:
the log-volatility dynamics of the market factor are estimated from the equal-weight market return
of the supplied data. Positive control: the same estimators in a stationary and two time-varying
simulations (a priori and calibrated parameters).

Reads ``returns_net.parquet`` written by ``02_run_backtest.py``. Writes
``nonstat_real_metrics.csv``, ``nonstat_real_tests.csv``, ``nonstat_calibration.csv``,
``nonstat_sim_paired.csv`` and ``nonstat_summary.txt`` to ``results/tables``. Weights and simulation
scores are cached.

Usage: ``python scripts/14_nonstationarity.py --config config.yaml [--force]``
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
from pfopt.backtest.nonstationary import EwmaResult, ewma_strategy_ids, generate_ewma_weights
from pfopt.config import Config, load_config
from pfopt.data.cleaning import read_processed
from pfopt.simulation.nonstationary import (
    DESIGNS,
    calibrate_market_volatility,
    paired_control,
    run_design,
)

logger = logging.getLogger("nonstationarity")
CORE = ("GMV-SAMPLE", "GMV-LW", "GMV-RMT", "GMV-PCA")


def _load_or_generate(
    cfg: Config, folder: Path, frames: dict[str, pd.DataFrame], force: bool
) -> EwmaResult:
    ids = ewma_strategy_ids(cfg.nonstationarity.halflives)
    marker = folder / "diagnostics.parquet"
    if marker.exists() and not force:
        logger.info("loading cached EWMA weights from %s", folder)
        weights = {s: pd.read_parquet(folder / f"{s}.parquet") for s in ids}
        return EwmaResult(weights=weights, diagnostics=pd.read_parquet(marker))
    started = time.perf_counter()
    result = generate_ewma_weights(frames["log_returns"], frames["dollar_volume"], cfg)
    logger.info("EWMA weights generated in %.0f s", time.perf_counter() - started)
    folder.mkdir(parents=True, exist_ok=True)
    for strategy, frame in result.weights.items():
        frame.to_parquet(folder / f"{strategy}.parquet")
    result.diagnostics.to_parquet(marker)
    return result


def _sim_key(cfg: Config, calibrated: dict[str, float]) -> str:
    full = cfg.to_dict()
    payload = {k: full[k] for k in ("simulation", "nonstationarity", "pca", "hrp")}
    payload["run"] = {"seed": cfg.run.seed, "annualization": cfg.run.annualization}
    payload["calibrated"] = {k: round(v, 6) for k, v in calibrated.items()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def _control_text(paired: pd.DataFrame, sizes: tuple[int, ...]) -> list[str]:
    lines = []
    for design in DESIGNS:
        rows = paired[(paired["design"] == design) & (paired["method"] != "GMV-SAMPLE")].copy()
        star = (rows["ci_low"] > 0) | (rows["ci_high"] < 0)
        rows["cell"] = rows["mean_log_ratio"].map("{:+.3f}".format) + np.where(star, "*", " ")
        table = rows.pivot(index="method", columns="T", values="cell")[list(sizes)]
        lines += [f"{design}:", table.to_string(), ""]
    return lines


def main() -> None:
    """Run the real-data EWMA strategies, the calibration and the positive control."""
    parser = argparse.ArgumentParser(description="Non-stationarity test.")
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    parser.add_argument("--force", action="store_true", help="recompute weights and simulation")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    root, ns, s = config_path.parent, cfg.nonstationarity, cfg.stats
    cache = root / cfg.paths.cache_dir / cfg.cache_key()
    if not (cache / "returns_net.parquet").exists():
        raise SystemExit(f"{cache / 'returns_net.parquet'} not found; run 02_run_backtest.py first")
    ann, rf = cfg.run.annualization, cfg.run.rf
    frames = read_processed(root / cfg.paths.processed_dir)
    ids = ewma_strategy_ids(ns.halflives)

    # ---- real data -------------------------------------------------------------------------
    folder = cache / ("ewma_" + "-".join(str(h) for h in ns.halflives))
    result = _load_or_generate(cfg, folder, frames, args.force)
    sims = {
        i: simulate(w, frames["simple_returns"], cfg.backtest.cost_bps)
        for i, w in result.weights.items()
    }
    net = pd.concat(
        [
            pd.read_parquet(cache / "returns_net.parquet")[list(CORE)],
            pd.DataFrame({i: sim.net for i, sim in sims.items()}),
        ],
        axis=1,
    )
    if net.isna().any().any():
        raise ValueError("core and EWMA returns are not on the same dates")
    core_diag = pd.read_parquet(cache / "diagnostics.parquet")
    rows = {}
    for strategy in (*CORE, *ids):
        turnover = sims[strategy].turnover if strategy in sims else None
        if turnover is None:
            turnover = pd.read_parquet(cache / "turnover.parquet")[strategy]
        values = performance_metrics(net[strategy], turnover, ann, rf)
        diagnostics = result.diagnostics if strategy in sims else core_diag
        pred = float(diagnostics[f"pred_vol_{strategy}"].mean())
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
    references = ("GMV-SAMPLE", *(f"GMV-EWMA{h}" for h in ns.halflives))
    tests = studentized_compare(
        net,
        references,
        block_length=s.block_length,
        n_boot=s.n_boot,
        seed=cfg.run.seed,
        annualization=ann,
        rf=rf,
        alpha=s.alpha,
        power=s.mde_power,
    )

    # ---- calibration -----------------------------------------------------------------------
    log_returns = frames["log_returns"]
    market = log_returns.mean(axis=1)[log_returns.notna().sum(axis=1) >= cfg.universe.n_assets]
    calibrated = calibrate_market_volatility(market.to_numpy(), ns.calibration_block_days)

    # ---- positive control ------------------------------------------------------------------
    sim_cache = root / cfg.paths.cache_dir / "nonstationarity" / _sim_key(cfg, calibrated)
    scores_path = sim_cache / "scores.parquet"
    if scores_path.exists() and not args.force:
        logger.info("loading cached simulation scores from %s", sim_cache)
        scores = pd.read_parquet(scores_path)
    else:
        parts = []
        for design in DESIGNS:
            started = time.perf_counter()
            parts.append(run_design(cfg, design, calibrated))
            logger.info("%s finished in %.0f s", design, time.perf_counter() - started)
        scores = pd.concat(parts, ignore_index=True)
        sim_cache.mkdir(parents=True, exist_ok=True)
        scores.to_parquet(scores_path)
    paired = paired_control(scores, cfg.simulation.alpha)

    # ---- outputs ---------------------------------------------------------------------------
    tables = root / cfg.paths.tables_dir
    tables.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(tables / "nonstat_real_metrics.csv")
    tests.to_csv(tables / "nonstat_real_tests.csv", index=False)
    pd.DataFrame(
        [
            {
                "parameter": name,
                "a_priori": getattr(cfg.simulation, name),
                "calibrated": calibrated[name],
            }
            for name in ("vol_log_sd", "vol_persistence")
        ]
    ).to_csv(tables / "nonstat_calibration.csv", index=False)
    paired.to_csv(tables / "nonstat_sim_paired.csv", index=False)

    vol = tests[tests["test"] == "log_var_ratio"].set_index(["reference", "strategy"])

    def vol_line(reference: str, strategy: str) -> str:
        r = vol.loc[(reference, strategy)]
        ratio, low, high = (float(np.exp(r[c] / 2.0)) for c in ("estimate", "ci_low", "ci_high"))
        flag = "*" if r["p_holm"] < s.alpha else " "
        return (
            f"  {strategy:<16} vs {reference:<14} ratio {ratio:6.3f} [{low:6.3f}, {high:6.3f}]  "
            f"p={r['p_value']:.3f} Holm {r['p_holm']:.3f} {flag} detectable {r['mde_vol_pct']:.1f}%"
        )

    t_eff = ", ".join(f"h={h}: {result.diagnostics[f't_eff_{h}'].mean():.0f}" for h in ns.halflives)
    lines = [
        f"{len(net)} daily net returns, {len(result.weights[ids[0]])} rebalances, window "
        f"T={cfg.universe.lookback_days}, N={cfg.universe.n_assets}; half-lives {ns.halflives}; "
        f"mean effective sample size {t_eff}",
        "",
        "REAL DATA (long-only GMV, net of costs)",
        metrics.round(3).to_string(),
        "",
        f"volatility ratios (studentized block bootstrap, block {s.block_length}, B={s.n_boot}; "
        "* = significant after Holm):",
        *(vol_line("GMV-SAMPLE", i) for i in ("GMV-LW", "GMV-RMT", "GMV-PCA", *ids)),
        *(vol_line(f"GMV-EWMA{h}", f"GMV-EWMA{h}-RMT") for h in ns.halflives),
        "",
        "CALIBRATION of the market-factor log-volatility dynamics "
        f"({int(calibrated['n_blocks'])} blocks of {ns.calibration_block_days} days; dates with at "
        f"least {cfg.universe.n_assets} symbols):",
        f"  vol_log_sd      a priori {cfg.simulation.vol_log_sd:.3f}  calibrated "
        f"{calibrated['vol_log_sd']:.3f}",
        f"  vol_persistence a priori {cfg.simulation.vol_persistence:.4f}  calibrated "
        f"{calibrated['vol_persistence']:.4f}",
        "",
        "POSITIVE CONTROL: paired log(true variance / sample GMV true variance), long-only, "
        f"Gaussian, {ns.sim_n_reps} replications",
        "(negative = beats sample; * = interval excludes 0):",
        *_control_text(paired, ns.sim_sample_sizes),
    ]
    path = tables / "nonstat_summary.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("summary written to %s", path)


if __name__ == "__main__":
    main()
