"""Factor-model simulation study of covariance and GMV estimation error.

Simulates returns from a known 1 + ``n_groups`` factor covariance (Gaussian = main result, and a
Student-t variant with the same covariance), fits the sample, Ledoit-Wolf, RMT and PCA covariances
(plus HRP and equal weight) for each sample size, and measures the Frobenius error of the
covariance, the true variance of the estimated GMV weights relative to the oracle (long-only and
unconstrained) and the L1 distance of the weights to the oracle weights.

Raw scores are cached in ``results/cache/simulation/<key>``; ``--force`` recomputes them. Writes
``simulation_summary.csv``, ``simulation_paired.csv``, ``simulation_winners.csv``,
``simulation_detection.csv`` and ``simulation_summary.txt`` to ``results/tables`` and the figures
``sim_gaussian`` / ``sim_student_t`` to ``results/figures``.

Scenario runs: ``--n-groups 5`` replaces ``simulation.n_groups``; ``--time-varying`` switches on
stochastic factor volatility and drifting loadings. Every output then gets a suffix (``_groups5``,
``_tv``, ``_groups5_tv``, or ``--tag NAME``), leaving the baseline files untouched.

Usage: ``python scripts/10_simulation.py --config config.yaml [--force] [--n-groups G]
[--time-varying] [--tag T]``
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from pfopt.config import Config, load_config
from pfopt.simulation.run import (
    COVARIANCE,
    LONG_ONLY,
    UNCONSTRAINED,
    paired_vs_sample,
    run_grid,
    summarize,
    true_model,
    winner_table,
)
from pfopt.viz.plots import plot_simulation_accuracy
from pfopt.viz.style import save_figure

logger = logging.getLogger("simulation")
ORACLE_TOLERANCE = 1e-6
TIME_VARYING_FIELDS = (
    "time_varying",
    "horizon_days",
    "vol_log_sd",
    "vol_persistence",
    "loading_drift_sd",
    "loading_persistence",
)


def _cache_key(cfg: Config) -> str:
    """Hash of everything that determines the simulated scores."""
    full = cfg.to_dict()
    keys = ("simulation", "pca", "hrp")
    payload = {k: full[k] for k in keys}
    if not cfg.simulation.time_varying:  # keeps the baseline cache valid when the 6.3 fields change
        for name in TIME_VARYING_FIELDS:
            payload["simulation"].pop(name)
    payload["run"] = {"seed": cfg.run.seed, "annualization": cfg.run.annualization}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def _run_or_load(cfg: Config, cache: Path, force: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores_path, info_path = cache / "scores.parquet", cache / "detection.parquet"
    if scores_path.exists() and info_path.exists() and not force:
        logger.info("loading cached simulation results from %s", cache)
        return pd.read_parquet(scores_path), pd.read_parquet(info_path)
    model = true_model(cfg)
    scores, infos = [], []
    for distribution in cfg.simulation.distributions:
        started = time.perf_counter()
        s, i = run_grid(cfg, distribution, model)
        scores.append(s)
        infos.append(i)
        logger.info("%s finished in %.0f s", distribution, time.perf_counter() - started)
    cache.mkdir(parents=True, exist_ok=True)
    out = pd.concat(scores, ignore_index=True), pd.concat(infos, ignore_index=True)
    out[0].to_parquet(scores_path)
    out[1].to_parquet(info_path)
    return out


def _detectability(sigma: np.ndarray, k_true: int, sizes: tuple[int, ...]) -> list[str]:
    """True correlation eigenvalues against the Marchenko-Pastur edge of the noise bulk.

    The bulk level is the mean of the eigenvalues beyond the first ``k_true``; the edge at sample
    size ``T`` is ``bulk * (1 + sqrt(N / T))^2``. A factor whose eigenvalue is below the edge cannot
    be told apart from noise by the MP rule.
    """
    n = sigma.shape[0]
    scale = np.sqrt(np.diag(sigma))
    eigenvalues = np.sort(np.linalg.eigvalsh(sigma / np.outer(scale, scale)))[::-1]
    bulk = float(eigenvalues[k_true:].mean())
    lines = [
        "true correlation eigenvalues of the K factors: "
        + ", ".join(f"{v:.2f}" for v in eigenvalues[:k_true])
        + f"; noise bulk level {bulk:.2f}"
    ]
    for n_obs in sizes:
        edge = bulk * (1.0 + np.sqrt(n / n_obs)) ** 2
        above = int((eigenvalues[:k_true] > edge).sum())
        lines.append(f"  T={n_obs}: MP edge {edge:.2f}, factors above it: {above} of {k_true}")
    return lines


def _pivot(summary: pd.DataFrame, dist: str, constraint: str, metric: str) -> pd.DataFrame:
    rows = summary[
        (summary["distribution"] == dist)
        & (summary["constraint"] == constraint)
        & (summary["metric"] == metric)
    ]
    return rows.pivot(index="method", columns="T", values="mean")


def _paired_text(paired: pd.DataFrame, dist: str, constraint: str) -> str:
    rows = paired[
        (paired["distribution"] == dist)
        & (paired["constraint"] == constraint)
        & (paired["method"] != "GMV-SAMPLE")
    ].copy()
    star = (rows["ci_low"] > 0) | (rows["ci_high"] < 0)
    rows["cell"] = rows["mean_log_ratio"].map("{:+.3f}".format) + np.where(star, "*", " ")
    return rows.pivot(index="method", columns="T", values="cell").to_string()


def main() -> None:
    """Run (or load) the simulation study and write tables, figures and the summary."""
    parser = argparse.ArgumentParser(description="Factor-model simulation study.")
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    parser.add_argument("--force", action="store_true", help="recompute the simulation")
    parser.add_argument("--n-groups", type=int, help="override simulation.n_groups (scenario run)")
    parser.add_argument("--time-varying", action="store_true", help="time-varying covariance")
    parser.add_argument("--tag", help="output suffix (default built from the scenario flags)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    matplotlib.use("Agg")

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    if args.n_groups is not None:
        sim_cfg = dataclasses.replace(cfg.simulation, n_groups=args.n_groups)
        cfg = dataclasses.replace(cfg, simulation=sim_cfg)
    if args.time_varying:
        cfg = dataclasses.replace(
            cfg, simulation=dataclasses.replace(cfg.simulation, time_varying=True)
        )
    parts = [f"groups{args.n_groups}"] if args.n_groups is not None else []
    parts += ["tv"] if args.time_varying else []
    tag = args.tag or "_".join(parts)

    def named(base: str) -> str:
        return f"{base}_{tag}" if tag else base

    root, sim = config_path.parent, cfg.simulation
    cache = root / cfg.paths.cache_dir / "simulation" / _cache_key(cfg)
    scores, detection = _run_or_load(cfg, cache, args.force)

    summary = summarize(scores, sim.alpha)
    paired = paired_vs_sample(scores, sim.alpha)
    winners = winner_table(summary)
    detect = (
        detection.groupby(["distribution", "T"])[["rmt_n_signal", "pca_K", "lw_shrinkage"]]
        .mean()
        .reset_index()
    )
    tables, figures = root / cfg.paths.tables_dir, root / cfg.paths.figures_dir
    tables.mkdir(parents=True, exist_ok=True)
    summary.to_csv(tables / f"{named('simulation_summary')}.csv", index=False)
    paired.to_csv(tables / f"{named('simulation_paired')}.csv", index=False)
    winners.to_csv(tables / f"{named('simulation_winners')}.csv", index=False)
    detect.to_csv(tables / f"{named('simulation_detection')}.csv", index=False)
    for dist in sim.distributions:
        base = "sim_gaussian" if dist == "gaussian" else "sim_student_t"
        save_figure(plot_simulation_accuracy(summary, dist), named(base), figures)

    true_k = 1 + sim.n_groups
    design = (
        f"time-varying: factor log-volatility sd {sim.vol_log_sd:g} (persistence "
        f"{sim.vol_persistence:g}), loading drift sd {sim.loading_drift_sd:g} (persistence "
        f"{sim.loading_persistence:g}); truth = average conditional covariance over the next "
        f"{sim.horizon_days} days"
        if sim.time_varying
        else "stationary covariance"
    )
    lines = [
        f"design: {design}",
        f"N={sim.n_assets}, true factors K={true_k} (1 market + {sim.n_groups} group), "
        f"T in {list(sim.sample_sizes)}, {sim.n_reps} replications, t df={sim.t_df:g}; "
        f"bands are {100 * (1 - sim.alpha):.0f}% normal intervals; "
        f"main result: {sim.distributions[0]}",
    ]
    lines += ["", *_detectability(true_model(cfg).sigma, true_k, sim.sample_sizes)]
    for dist in sim.distributions:
        weights = scores[(scores["distribution"] == dist) & scores["var_ratio"].notna()]
        lowest = float(weights["var_ratio"].min())
        frob = _pivot(summary, dist, COVARIANCE, "frob_error")
        decreasing = frob.diff(axis=1).iloc[:, 1:].lt(0).all(axis=None)
        lines += [
            "",
            f"===== {dist} =====",
            f"lowest true-variance ratio over all replications: {lowest:.8f} "
            f"(oracle bound holds: {lowest >= 1 - ORACLE_TOLERANCE})",
            f"covariance error decreases with T for every estimator: {bool(decreasing)}",
            "",
            "relative Frobenius error (mean):",
            frob.round(3).to_string(),
            "",
            "true variance / oracle, long-only (mean):",
            _pivot(summary, dist, LONG_ONLY, "var_ratio").round(3).to_string(),
            "",
            "true variance / oracle, unconstrained (mean):",
            _pivot(summary, dist, UNCONSTRAINED, "var_ratio").round(3).to_string(),
            "",
            "L1 distance to oracle weights, long-only (mean):",
            _pivot(summary, dist, LONG_ONLY, "l1_dist").round(3).to_string(),
            "",
            "paired log(true var / sample GMV true var), long-only (negative = beats sample; "
            "* = interval excludes 0):",
            _paired_text(paired, dist, LONG_ONLY),
            "",
            "same, unconstrained:",
            _paired_text(paired, dist, UNCONSTRAINED),
            "",
            f"detected factors (true K={true_k}): mean RMT n_signal / PCA K / LW shrinkage:",
            detect[detect["distribution"] == dist]
            .drop(columns="distribution")
            .round(3)
            .to_string(index=False),
            "",
            "lowest-mean-variance method per cell:",
            winners[winners["distribution"] == dist]
            .drop(columns="distribution")
            .round(3)
            .to_string(index=False),
        ]
    summary_path = tables / f"{named('simulation_summary')}.txt"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("summary written to %s", summary_path)


if __name__ == "__main__":
    main()
