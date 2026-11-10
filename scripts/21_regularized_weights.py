"""Regularized weights (ridge, shrinkage to 1/N, L1 bound, elastic net).

Real data: regularized versions of sample-covariance GMV on exactly the dates and universes of the
core strategies, net of costs, tested with the studentized bootstrap against
``GMV-SAMPLE`` (does regularizing the weights help?) and ``EW``. ``GMV-LW`` and ``GMV-RMT`` are
included for context. Ridge and shrinkage are long-only; L1 and elastic net allow shorting
(no borrow cost).

Reads ``returns_net.parquet`` written by ``02_run_backtest.py``. Writes
``regularized_metrics.csv``, ``regularized_tests.csv`` and ``regularized_summary.txt`` to
``results/tables``. Weights are cached.

Usage: ``python scripts/21_regularized_weights.py --config config.yaml [--force]``
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pfopt.backtest.engine import read_weights_result, simulate
from pfopt.backtest.inference import studentized_compare
from pfopt.backtest.metrics import performance_metrics
from pfopt.backtest.regularized import (
    RegularizedResult,
    generate_regularized_weights,
    strategy_ids,
)
from pfopt.config import Config, load_config
from pfopt.data.cleaning import read_processed

logger = logging.getLogger("regularized")
CONTEXT = ("EW", "GMV-SAMPLE", "GMV-LW", "GMV-RMT")
REFERENCES = ("GMV-SAMPLE", "EW")


def _load_or_generate(
    cfg: Config, folder: Path, frames: dict[str, pd.DataFrame], force: bool
) -> RegularizedResult:
    marker = folder / "diagnostics.parquet"
    ids = strategy_ids(cfg)
    if marker.exists() and not force and all((folder / f"{s}.parquet").exists() for s in ids):
        logger.info("loading cached regularized weights from %s", folder)
        weights = {s: pd.read_parquet(folder / f"{s}.parquet") for s in ids}
        return RegularizedResult(weights=weights, diagnostics=pd.read_parquet(marker))
    started = time.perf_counter()
    result = generate_regularized_weights(frames["log_returns"], frames["dollar_volume"], cfg)
    logger.info("regularized weights generated in %.0f s", time.perf_counter() - started)
    folder.mkdir(parents=True, exist_ok=True)
    for strategy, frame in result.weights.items():
        frame.to_parquet(folder / f"{strategy}.parquet")
    result.diagnostics.to_parquet(marker)
    return result


def main() -> None:
    """Run the regularized strategies and write the tables and the summary."""
    parser = argparse.ArgumentParser(description="Regularized weights.")
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    parser.add_argument("--force", action="store_true", help="recompute the weights")
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
    result = _load_or_generate(cfg, cache / "regularized", frames, args.force)
    ids = strategy_ids(cfg)

    cost, simple = cfg.backtest.cost_bps, frames["simple_returns"]
    sims = {
        strategy: simulate(result.weights[strategy], simple, cost, long_only=ids[strategy])
        for strategy in ids
    }
    core = read_weights_result(cache)
    core_net = pd.read_parquet(cache / "returns_net.parquet")[list(CONTEXT)]
    net = pd.concat([pd.DataFrame({k: v.net for k, v in sims.items()}), core_net], axis=1)
    if net.isna().any().any():
        raise ValueError("core and regularized returns are not on the same dates")
    core_turnover = pd.read_parquet(cache / "turnover.parquet")
    rows = {}
    for strategy in (*ids, *CONTEXT):
        new = strategy in sims
        turnover = sims[strategy].turnover if new else core_turnover[strategy]
        values = performance_metrics(net[strategy], turnover, ann, rf)
        weights = (result.weights if new else core.weights)[strategy].to_numpy()
        row = {
            "ann_vol": values["ann_vol"],
            "ann_return": values["ann_return"],
            "sharpe": values["sharpe"],
            "max_drawdown": values["max_drawdown"],
            "annual_turnover": values["annual_turnover"],
            "mean_leverage": float(np.abs(weights).sum(axis=1).mean()),
            "mean_effective_n": float((1.0 / (weights**2).sum(axis=1)).mean()),
        }
        if strategy != "EW":
            diag = result.diagnostics if new else core.diagnostics
            pred = float(diag[f"pred_vol_{strategy}"].mean())
            row["mean_pred_vol"] = pred
            row["realized_over_pred"] = values["ann_vol"] / pred
        rows[strategy] = row
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

    tables = root / cfg.paths.tables_dir
    tables.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(tables / "regularized_metrics.csv")
    tests.to_csv(tables / "regularized_tests.csv", index=False)

    vol = tests[tests["test"] == "log_var_ratio"].set_index(["reference", "strategy"])
    sharpe = tests[tests["test"] == "sharpe_diff"].set_index(["reference", "strategy"])

    def line(reference: str, strategy: str) -> str:
        v, d = vol.loc[(reference, strategy)], sharpe.loc[(reference, strategy)]
        ratio, low, high = (float(np.exp(v[c] / 2.0)) for c in ("estimate", "ci_low", "ci_high"))
        flag = "*" if v["p_holm"] < s.alpha else " "
        return (
            f"  {strategy:<18} vs {reference:<10} vol ratio {ratio:6.3f} [{low:6.3f}, {high:6.3f}]"
            f" p={v['p_value']:.3f} Holm {v['p_holm']:.3f} {flag}"
            f" detectable {v['mde_vol_pct']:.1f}%"
            f" | Sharpe diff {d['estimate']:+.2f} [{d['ci_low']:+.2f}, {d['ci_high']:+.2f}]"
        )

    lines = [
        f"{len(net)} daily net returns, {len(next(iter(result.weights.values())))} rebalances, "
        f"window T={cfg.universe.lookback_days}, N={cfg.universe.n_assets}; sample covariance",
        "RIDGE / SHRINK are long-only; L1 / ENET allow shorting without borrow cost",
        "",
        "REAL DATA (net of costs)",
        metrics.round(3).to_string(),
        "",
        f"volatility ratios (studentized block bootstrap, block {s.block_length}, B={s.n_boot}; "
        "* = significant after Holm within the reference):",
    ]
    for reference in REFERENCES:
        lines += [f"against {reference}:"]
        lines += [line(reference, x) for x in net.columns if x != reference]
    path = tables / "regularized_summary.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("summary written to %s", path)


if __name__ == "__main__":
    main()
