"""ML return forecasts feeding the optimizer.

Stage 1 (signal): walk-forward Ridge and random-forest forecasts of the next-21-day cross-sectional
return rank from momentum, 1-month return, volatility and dollar volume, judged by the
out-of-sample rank information coefficient (IC) against the single features.

Stage 2 (portfolio): a model whose mean IC is positive with t-statistic at least ``ml.ic_gate_t``
feeds long-only max-Sharpe with the RMT covariance (``MV-ML-<model>-RMT``). A model that fails the
gate is not turned into a portfolio unless ``ml.force_portfolio`` is true or ``--ignore-gate`` is
passed (the output is then labelled "gate failed"). All strategies, including
the core references, are simulated from the first evaluation date on the same calendar, from cash,
with the same costs, tested with the studentized bootstrap.

Reads the core weights written by ``02_run_backtest.py``. Writes ``ml_signal_ic.csv``,
``ml_ic_by_date.csv``, ``ml_metrics.csv``, ``ml_tests.csv`` and ``ml_summary.txt`` to
``results/tables``. Predictions are cached under ``results/cache/<key>/ml/<ml hash>``.

Usage: ``python scripts/22_ml_forecasts.py --config config.yaml [--force] [--ignore-gate]``
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

from pfopt.backtest.engine import read_weights_result, simulate
from pfopt.backtest.inference import studentized_compare
from pfopt.backtest.metrics import performance_metrics
from pfopt.backtest.mlportfolio import ML_STRATEGIES, generate_ml_weights
from pfopt.backtest.mlsignal import (
    MODELS,
    SignalResult,
    build_panel,
    ic_gate,
    ic_summary,
    walk_forward_signals,
)
from pfopt.config import Config, load_config
from pfopt.data.cleaning import read_processed

logger = logging.getLogger("ml_forecasts")
CONTEXT = ("EW", "GMV-SAMPLE", "GMV-RMT", "MV-SAMPLE", "MV-RMT")
REFERENCES = ("MV-RMT", "GMV-RMT", "EW")


def _load_or_compute(
    cfg: Config, folder: Path, frames: dict[str, pd.DataFrame], force: bool
) -> SignalResult:
    names = [f"pred_{m}.parquet" for m in MODELS] + ["ic.parquet", "fits.parquet"]
    if not force and all((folder / n).exists() for n in names):
        logger.info("loading cached ML signals from %s", folder)
        return SignalResult(
            predictions={m: pd.read_parquet(folder / f"pred_{m}.parquet") for m in MODELS},
            ic=pd.read_parquet(folder / "ic.parquet"),
            fits=pd.read_parquet(folder / "fits.parquet"),
        )
    started = time.perf_counter()
    panel = build_panel(frames["log_returns"], frames["dollar_volume"], cfg)
    result = walk_forward_signals(panel, cfg)
    logger.info("ML signals computed in %.0f s", time.perf_counter() - started)
    folder.mkdir(parents=True, exist_ok=True)
    for model, frame in result.predictions.items():
        frame.to_parquet(folder / f"pred_{model}.parquet")
    result.ic.to_parquet(folder / "ic.parquet")
    result.fits.to_parquet(folder / "fits.parquet")
    return result


def main() -> None:
    """Run the signal evaluation and, for models passing the IC gate, the portfolios."""
    parser = argparse.ArgumentParser(description="ML return forecasts feeding the optimizer.")
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    parser.add_argument("--force", action="store_true", help="recompute the signals")
    parser.add_argument(
        "--ignore-gate", action="store_true", help="build portfolios even if the IC gate fails"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    root, s, m = config_path.parent, cfg.stats, cfg.ml
    cache = root / cfg.paths.cache_dir / cfg.cache_key()
    if not (cache / "diagnostics.parquet").exists():
        raise SystemExit(f"core weights not found in {cache}; run 02_run_backtest.py first")
    ml_hash = hashlib.sha256(json.dumps(cfg.to_dict()["ml"], sort_keys=True).encode()).hexdigest()
    frames = read_processed(root / cfg.paths.processed_dir)
    signal = _load_or_compute(cfg, cache / "ml" / ml_hash[:12], frames, args.force)

    summary = ic_summary(signal.ic)
    gate = ic_gate(summary, MODELS, m.ic_gate_t)
    tables = root / cfg.paths.tables_dir
    tables.mkdir(parents=True, exist_ok=True)
    summary.to_csv(tables / "ml_signal_ic.csv")
    signal.ic.rename_axis("date").to_csv(tables / "ml_ic_by_date.csv")

    first_date, last_date = signal.ic.index[0], signal.ic.index[-1]
    lines = [
        f"{len(signal.ic)} evaluation dates {first_date.date()} to {last_date.date()}, "
        f"{len(signal.fits)} refits, N={cfg.universe.n_assets}, horizon {m.horizon_days} days",
        f"training samples at the first / last fit: {int(signal.fits['n_samples'].iloc[0])} / "
        f"{int(signal.fits['n_samples'].iloc[-1])}",
        "",
        "OUT-OF-SAMPLE RANK IC (Spearman, per rebalance date; models and single features)",
        summary.round(4).to_string(),
        "",
        f"gate (mean IC > 0 and t >= {m.ic_gate_t:g}): "
        + ", ".join(f"{name} {'pass' if ok else 'FAIL'}" for name, ok in gate.items()),
    ]
    force = args.ignore_gate or m.force_portfolio
    built = [name for name, ok in gate.items() if ok or force]
    for stale in ("ml_metrics.csv", "ml_tests.csv"):
        (tables / stale).unlink(missing_ok=True)
    if not built:
        lines += ["no model passed the gate: portfolios not built (set ml.force_portfolio)"]
        _write(tables / "ml_summary.txt", lines)
        return
    failed = [name for name in built if not gate[name]]
    if failed:
        lines += [f"GATE FAILED for {', '.join(failed)}: portfolios shown for completeness only"]

    strategies = [sid for sid, model in ML_STRATEGIES.items() if model in built]
    predictions = {model: signal.predictions[model] for model in built}
    result = generate_ml_weights(frames["log_returns"], predictions, cfg, strategies)
    dates = result.diagnostics.index

    cost, simple = cfg.backtest.cost_bps, frames["simple_returns"]
    core = read_weights_result(cache)
    absent = [x for x in CONTEXT if x not in core.weights]
    if absent:
        raise SystemExit(f"core weights lack {absent}; rerun 02_run_backtest.py")
    sims = {x: simulate(result.weights[x], simple, cost) for x in strategies}
    sims.update({x: simulate(core.weights[x].loc[dates], simple, cost) for x in CONTEXT})
    net = pd.DataFrame({x: sim.net for x, sim in sims.items()})
    if net.isna().any().any():
        raise ValueError("strategies are not on the same dates")

    ann, rf = cfg.run.annualization, cfg.run.rf
    rows = {}
    for strategy in (*strategies, *CONTEXT):
        values = performance_metrics(net[strategy], sims[strategy].turnover, ann, rf)
        weights = (result.weights if strategy in strategies else core.weights)[strategy]
        weights = weights.loc[dates].to_numpy()
        row = {
            "ann_vol": values["ann_vol"],
            "ann_return": values["ann_return"],
            "sharpe": values["sharpe"],
            "max_drawdown": values["max_drawdown"],
            "annual_turnover": values["annual_turnover"],
            "mean_effective_n": float((1.0 / (weights**2).sum(axis=1)).mean()),
        }
        if strategy != "EW":
            diag = result.diagnostics if strategy in strategies else core.diagnostics.loc[dates]
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
    metrics.to_csv(tables / "ml_metrics.csv")
    tests.to_csv(tables / "ml_tests.csv", index=False)

    vol = tests[tests["test"] == "log_var_ratio"].set_index(["reference", "strategy"])
    sharpe = tests[tests["test"] == "sharpe_diff"].set_index(["reference", "strategy"])

    def line(reference: str, strategy: str) -> str:
        v, d = vol.loc[(reference, strategy)], sharpe.loc[(reference, strategy)]
        ratio, low, high = (float(np.exp(v[c] / 2.0)) for c in ("estimate", "ci_low", "ci_high"))
        flag = "*" if v["p_holm"] < s.alpha else " "
        return (
            f"  {strategy:<16} vs {reference:<10} vol ratio {ratio:6.3f} [{low:6.3f}, {high:6.3f}]"
            f" Holm {v['p_holm']:.3f} {flag}"
            f" | Sharpe diff {d['estimate']:+.2f} [{d['ci_low']:+.2f}, {d['ci_high']:+.2f}]"
            f" p={d['p_value']:.3f} Holm {d['p_holm']:.3f}"
        )

    lines += [
        "",
        f"PORTFOLIOS (net of costs; {len(net)} daily returns from {net.index[0].date()}; all "
        "strategies simulated from cash on the same dates)",
        metrics.round(3).to_string(),
        "",
        f"studentized block bootstrap, block {s.block_length}, B={s.n_boot} "
        "(* = vol ratio significant after Holm within the reference):",
    ]
    for reference in REFERENCES:
        lines += [f"against {reference}:"]
        lines += [line(reference, x) for x in strategies if x != reference]
    _write(tables / "ml_summary.txt", lines)


def _write(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("summary written to %s", path)


if __name__ == "__main__":
    main()
