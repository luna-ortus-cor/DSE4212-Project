"""Step 7.1: regenerate every figure and table of the report from the cached results.

First re-runs the analysis stage scripts (each reads the cached weights and returns, so this takes
minutes, not hours), then copies the report figures (PNG and PDF) and tables (CSV, Markdown,
LaTeX) to ``results/report`` under report numbering, checks every file against the style rules
and writes ``manifest.csv`` and ``report_assets_summary.txt`` there. Exits with status 1 if a
stage failed or an asset is missing or breaks a rule.

Prerequisite: the cleaned data (``scripts/01b_clean_data.py``) and a completed backtest
(``scripts/02_run_backtest.py``; it is re-run here and uses its cache).

Usage: ``python scripts/11_make_report_assets.py --config config.yaml [--assemble-only]``
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from pfopt.config import load_config
from pfopt.report.assets import (
    APPENDIX,
    FIGURES,
    MAIN,
    TABLES,
    build_table,
    check_figure_files,
    format_table,
    numeric_columns,
    parameter_table,
    to_latex,
    to_markdown,
)

logger = logging.getLogger("report_assets")
STAGES = (
    "03_eda.py",
    "03b_markowitz_frontier.py",
    "02a_compare_estimators.py",
    "02_run_backtest.py",
    "05_significance.py",
    "06_subperiods.py",
    "07_sensitivity.py",
    "08_structure.py",
    "09_unconstrained.py",
    "10_simulation.py",
    "12_pipeline_check.py",
    "13_inference.py",
    "14_nonstationarity.py",
    "15_residual_structure.py",
    "16_sector_block.py",
    "17_sector_placebo.py",
    "18_risk_based.py",
    "19_nonlinear_shrinkage.py",
    "20_forecast_loss.py",
)


def _run_stages(root: Path, config_path: Path, log_dir: Path) -> list[dict[str, object]]:
    """Run every stage script; the output of each goes to ``log_dir/<stage>.log``."""
    log_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for stage in STAGES:
        started = time.perf_counter()
        log_path = log_dir / f"{Path(stage).stem}.log"
        with log_path.open("w", encoding="utf-8") as log:
            code = subprocess.run(  # noqa: S603
                [sys.executable, str(root / "scripts" / stage), "--config", str(config_path)],
                cwd=root,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            ).returncode
        seconds = time.perf_counter() - started
        status = "ok" if code == 0 else "FAILED"
        logger.info("%-28s %s in %5.0f s (log: %s)", stage, status, seconds, log_path.name)
        results.append({"stage": stage, "returncode": code, "seconds": round(seconds, 1)})
    return results


def _copy_figures(source: Path, target: Path) -> list[dict[str, object]]:
    rows = []
    counters = {MAIN: 0, APPENDIX: 0}
    for spec in FIGURES:
        counters[spec.role] += 1
        prefix = "fig" if spec.role == MAIN else "figA"
        name = f"{prefix}{counters[spec.role]:02d}_{spec.key}"
        problems = check_figure_files(source, spec.source)
        if not any("missing" in p for p in problems):
            for extension in ("png", "pdf"):
                if (source / f"{spec.source}.{extension}").exists():
                    origin = source / f"{spec.source}.{extension}"
                    shutil.copyfile(origin, target / f"{name}.{extension}")
        rows.append(
            {"kind": "figure", "role": spec.role, "name": name, "source": spec.source,
             "caption": spec.caption, "problems": "; ".join(problems)}
        )  # fmt: skip
    return rows


def _write_table(frame: pd.DataFrame, name: str, caption: str, label: str, target: Path) -> None:
    frame.to_csv(target / f"{name}.csv", index=False)
    text = format_table(frame)
    (target / f"{name}.md").write_text(to_markdown(text), encoding="utf-8")
    latex = to_latex(text, caption, label, numeric_columns(frame))
    (target / f"{name}.tex").write_text(latex, encoding="utf-8")


def _build_tables(
    tables_dir: Path, parameters: pd.DataFrame, target: Path
) -> list[dict[str, object]]:
    rows = []
    counters = {MAIN: 0, APPENDIX: 0}
    for spec in TABLES:
        counters[spec.role] += 1
        prefix = "tab" if spec.role == MAIN else "tabA"
        name = f"{prefix}{counters[spec.role]:02d}_{spec.key}"
        problem = ""
        try:
            _write_table(build_table(spec, tables_dir), name, spec.caption, spec.key, target)
        except (OSError, KeyError, ValueError) as exc:
            problem = f"{type(exc).__name__}: {exc}"
        rows.append(
            {"kind": "table", "role": spec.role, "name": name, "source": spec.source,
             "caption": spec.caption, "problems": problem}
        )  # fmt: skip
    counters[MAIN] += 1
    name = f"tab{counters[MAIN]:02d}_parameters"
    _write_table(parameters, name, "Fixed a priori parameters.", "parameters", target)
    rows.append(
        {"kind": "table", "role": MAIN, "name": name, "source": "config.yaml",
         "caption": "Fixed a priori parameters.", "problems": ""}
    )  # fmt: skip
    return rows


def main() -> None:
    """Re-run the stages, assemble the report assets and write the manifest."""
    parser = argparse.ArgumentParser(description="Assemble the report figures and tables.")
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    parser.add_argument("--assemble-only", action="store_true", help="skip re-running the stages")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    root = config_path.parent
    if not any((root / cfg.paths.processed_dir).glob("*.parquet")):
        raise SystemExit("no processed data found; run scripts/01b_clean_data.py first")
    out = root / "results" / "report"
    figures_out, tables_out = out / "figures", out / "tables"
    for folder in (figures_out, tables_out):
        folder.mkdir(parents=True, exist_ok=True)

    stages = [] if args.assemble_only else _run_stages(root, config_path, out / "logs")
    rows = _copy_figures(root / cfg.paths.figures_dir, figures_out)
    rows += _build_tables(root / cfg.paths.tables_dir, parameter_table(cfg), tables_out)
    manifest = pd.DataFrame(rows)
    manifest.to_csv(out / "manifest.csv", index=False)

    failed = [s for s in stages if s["returncode"] != 0]
    broken = manifest[manifest["problems"] != ""]
    lines = [
        f"stages run: {len(stages)}, failed: {len(failed)}"
        + (f" ({', '.join(str(s['stage']) for s in failed)})" if failed else ""),
        *(f"  {s['stage']:<28} {s['seconds']:>7.1f} s" for s in stages),
        f"figures: {int((manifest['kind'] == 'figure').sum())} "
        f"({int(((manifest['kind'] == 'figure') & (manifest['role'] == MAIN)).sum())} main), "
        f"tables: {int((manifest['kind'] == 'table').sum())} "
        f"({int(((manifest['kind'] == 'table') & (manifest['role'] == MAIN)).sum())} main)",
        f"assets with problems: {len(broken)}",
        *(f"  {r['name']}: {r['problems']}" for _, r in broken.iterrows()),
    ]
    (out / "report_assets_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("summary written to %s", out / "report_assets_summary.txt")
    if failed or len(broken):
        sys.exit(1)


if __name__ == "__main__":
    main()
