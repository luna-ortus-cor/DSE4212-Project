"""Assembly and checking of the report figures and tables (PLAN.md step 7.1).

The stage scripts write their outputs to ``results/figures`` and ``results/tables``. This module
defines which of them enter the report (:data:`FIGURES`, :data:`TABLES`), checks the figure files
against the style rules of ``CLAUDE.md`` section 8 (PNG plus PDF, 300 dpi, at most 6.3 in wide),
selects and formats the table columns and renders Markdown and LaTeX versions. Nothing here
contains result numbers: every value comes from the CSV files written by the stage scripts.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pfopt.config import Config
from pfopt.viz.style import DPI, FIG_WIDTH_IN

MAIN, APPENDIX = "main", "appendix"
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_DPI_TOLERANCE = 1.0
_WIDTH_TOLERANCE_IN = 0.01
_METERS_PER_INCH = 0.0254


@dataclass(frozen=True)
class FigureSpec:
    """A report figure: source file stem in ``results/figures`` and its place in the report."""

    key: str
    source: str
    role: str
    caption: str


@dataclass(frozen=True)
class TableSpec:
    """A report table built from one stage CSV.

    Attributes
    ----------
    key, source, role, caption
        Short name, CSV file in ``results/tables``, ``main`` or ``appendix``, caption text.
    columns : tuple of str or None
        Columns kept, in order (``None`` keeps all).
    where : tuple of (str, str) or None
        Keep rows whose column (first entry) equals the value (second entry, compared as text).
    index_label : str
        Name given to the first column when the CSV stores an unnamed index.
    """

    key: str
    source: str
    role: str
    caption: str
    columns: tuple[str, ...] | None = None
    where: tuple[str, str] | None = None
    index_label: str = "strategy"


FIGURES = (
    FigureSpec(
        "mp_spectrum", "eda_mp_spectrum", MAIN, "Eigenvalue spectrum against Marchenko-Pastur."
    ),
    FigureSpec("corr_heatmap", "eda_corr_heatmap", MAIN, "Clustered correlation matrices."),
    FigureSpec("spectrum_time", "eda_spectrum_timeseries", MAIN, "Signal eigenvalues over time."),
    FigureSpec("frontier", "mv_frontier", MAIN, "Estimated against realized efficient frontier."),
    FigureSpec(
        "equity", "bt_equity_since_spx", MAIN, "Net equity curves with 1/N and the S&P 500."
    ),
    FigureSpec("drawdowns", "bt_drawdowns", MAIN, "Drawdowns of the core strategies."),
    FigureSpec(
        "vol_forecast", "bt_vol_forecast_ratio", MAIN, "Realized over predicted volatility."
    ),
    FigureSpec(
        "turnover", "bt_turnover_effective_n", MAIN, "Turnover and effective number of names."
    ),
    FigureSpec(
        "subperiods", "subperiod_sharpe_drawdown", MAIN, "Sharpe and drawdown by subperiod."
    ),
    FigureSpec("sens_q", "sens_q", MAIN, "Sensitivity to N and T (ratio q = N / T)."),
    FigureSpec("h3", "unc_h3", MAIN, "Long-only against unconstrained GMV (hypothesis H3)."),
    FigureSpec("simulation", "sim_gaussian", MAIN, "Simulation study, Gaussian returns."),
    FigureSpec("return_hist", "eda_return_hist", APPENDIX, "Pooled return distribution."),
    FigureSpec("rolling_corr", "eda_rolling_corr", APPENDIX, "Rolling average correlation."),
    FigureSpec("cov_spectra", "cmp_eigenvalue_spectra", APPENDIX, "Covariance estimator spectra."),
    FigureSpec("sorted_weights", "cmp_sorted_weights", APPENDIX, "Sorted weights at one date."),
    FigureSpec("equity_full", "bt_equity_curves", APPENDIX, "Net equity curves, full sample."),
    FigureSpec("risk_return", "bt_risk_return", APPENDIX, "Risk against return."),
    FigureSpec(
        "rolling_sharpe", "bt_rolling_sharpe_vol", APPENDIX, "Rolling Sharpe and volatility."
    ),
    FigureSpec("sens_other", "sens_other", APPENDIX, "Sensitivity to costs, cap, K and linkage."),
    FigureSpec("struct_weights", "struct_weights", APPENDIX, "Pooled weight distributions."),
    FigureSpec("struct_dynamics", "struct_dynamics", APPENDIX, "Weight dynamics."),
    FigureSpec("struct_sectors", "struct_sectors", APPENDIX, "Sector exposure."),
    FigureSpec("struct_eigen", "struct_eigen", APPENDIX, "Spectrum diagnostics and performance."),
    FigureSpec("simulation_t", "sim_student_t", APPENDIX, "Simulation study, Student-t returns."),
)

_PERFORMANCE = (
    "strategy", "ann_return", "ann_vol", "sharpe", "sortino", "max_drawdown", "calmar",
    "cvar_95", "annual_turnover",
)  # fmt: skip

TABLES = (
    TableSpec("main_net", "main_metrics.csv", MAIN, "Out-of-sample metrics, net of costs.",
              _PERFORMANCE, ("basis", "net")),
    TableSpec("main_gross", "main_metrics.csv", MAIN, "Out-of-sample metrics, gross of costs.",
              _PERFORMANCE, ("basis", "gross")),
    TableSpec("significance", "significance_tests.csv", MAIN, "Bootstrap tests against references.",
              ("reference", "strategy", "test", "estimate", "ci_low", "ci_high", "p_value")),
    TableSpec("subperiods", "subperiod_metrics.csv", MAIN, "Metrics by subperiod.",
              ("period", "strategy", "ann_return", "ann_vol", "sharpe", "max_drawdown")),
    TableSpec("sensitivity", "sensitivity_metrics.csv", MAIN, "Sensitivity analyses.",
              ("grid", "variant", "strategy", "period", "ann_vol", "sharpe", "annual_turnover")),
    TableSpec("h3_metrics", "unconstrained_metrics.csv", MAIN,
              "Long-only and unconstrained GMV.",
              ("strategy", "constraint", "ann_vol", "sharpe", "annual_turnover",
               "mean_max_weight", "mean_leverage", "realized_over_pred")),
    TableSpec("h3_tests", "unconstrained_h3.csv", MAIN, "Hypothesis H3 bootstrap test.",
              index_label="name"),
    TableSpec("common_period", "common_period_metrics.csv", APPENDIX,
              "Metrics on the common period with the S&P 500.", _PERFORMANCE),
    TableSpec("concentration", "structure_concentration.csv", APPENDIX,
              "Weight concentration."),
    TableSpec("risk_forecast", "risk_forecast_quality.csv", APPENDIX,
              "Volatility forecast quality."),
    TableSpec("simulation_paired", "simulation_paired.csv", APPENDIX,
              "Simulation: paired comparison against the sample covariance.",
              ("distribution", "T", "constraint", "method", "mean_log_ratio", "ci_low", "ci_high",
               "beats_sample")),
    TableSpec("simulation_detection", "simulation_detection.csv", APPENDIX,
              "Simulation: detected number of factors."),
)  # fmt: skip

_EXCLUDED_PARAMETERS = ("paths", "stocks_file", "companies_file", "index_file")


def png_info(path: Path) -> tuple[int, int, float | None]:
    """Width and height in pixels and the resolution in dpi (``None`` if not stored) of a PNG."""
    data = Path(path).read_bytes()
    if data[:8] != _PNG_SIGNATURE:
        raise ValueError(f"{path} is not a PNG file")
    width, height = struct.unpack(">II", data[16:24])
    dpi = None
    position = 8
    while position + 8 <= len(data):
        length, kind = struct.unpack(">I4s", data[position : position + 8])
        if kind == b"pHYs":
            x_per_meter, _, unit = struct.unpack(">IIB", data[position + 8 : position + 17])
            dpi = x_per_meter * _METERS_PER_INCH if unit == 1 else None
            break
        position += 12 + length
    return int(width), int(height), dpi


def check_figure_files(directory: Path, stem: str) -> list[str]:
    """Problems of the figure ``stem`` against the style rules (empty list = fine).

    Checks that the PNG and the PDF exist, that the PDF starts with ``%PDF``, that the PNG has
    300 dpi and a width of at most 6.3 inches.
    """
    problems = []
    png, pdf = Path(directory) / f"{stem}.png", Path(directory) / f"{stem}.pdf"
    if not pdf.exists() or pdf.stat().st_size == 0:
        problems.append(f"{stem}: PDF missing or empty")
    elif pdf.read_bytes()[:4] != b"%PDF":
        problems.append(f"{stem}: PDF has no %PDF header")
    if not png.exists():
        problems.append(f"{stem}: PNG missing")
        return problems
    width, _, dpi = png_info(png)
    if dpi is None or abs(dpi - DPI) > _DPI_TOLERANCE:
        problems.append(f"{stem}: PNG resolution is {dpi}, expected {DPI} dpi")
    elif width / dpi > FIG_WIDTH_IN + _WIDTH_TOLERANCE_IN:
        problems.append(f"{stem}: PNG is {width / dpi:.2f} in wide, limit {FIG_WIDTH_IN} in")
    return problems


def build_table(spec: TableSpec, tables_dir: Path) -> pd.DataFrame:
    """Read the source CSV of ``spec``, apply its row filter and column selection."""
    frame = pd.read_csv(Path(tables_dir) / spec.source)
    first = frame.columns[0]
    if str(first).startswith("Unnamed"):
        frame = frame.rename(columns={first: spec.index_label})
    if spec.where is not None:
        column, value = spec.where
        frame = frame[frame[column].astype(str) == value]
    if spec.columns is not None:
        missing = [c for c in spec.columns if c not in frame.columns]
        if missing:
            raise KeyError(f"{spec.source}: missing column(s) {missing}")
        frame = frame[list(spec.columns)]
    if frame.empty:
        raise ValueError(f"{spec.key}: no rows left after filtering {spec.where}")
    return frame.reset_index(drop=True)


def parameter_table(cfg: Config) -> pd.DataFrame:
    """Flat ``section / parameter / value`` table of the configuration (no paths or file names)."""
    rows = []
    for section, values in cfg.to_dict().items():
        if section in _EXCLUDED_PARAMETERS:
            continue
        for name, value in values.items():
            if name in _EXCLUDED_PARAMETERS or name.endswith("_columns"):
                continue
            text = "none" if value is None else str(value).replace("'", "")
            rows.append({"section": section, "parameter": name, "value": text})
    return pd.DataFrame(rows)


def _format_value(value: object, integer: bool) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float | int) and not isinstance(value, bool):
        if pd.isna(value):
            return ""
        if integer:
            return str(int(value))
        if value != 0 and abs(value) < 1e-3:
            return f"{value:.1e}"
        return f"{value:.3f}"
    return "" if value is None or (isinstance(value, float) and pd.isna(value)) else str(value)


def format_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Text version of ``frame``: three decimals, small numbers in scientific notation.

    Columns whose name starts with ``n_`` are shown as integers; booleans as yes / no.
    """
    out = pd.DataFrame(index=frame.index)
    for column in frame.columns:
        integer = str(column).startswith("n_")
        out[column] = [_format_value(v, integer) for v in frame[column].tolist()]
    return out


def to_markdown(frame: pd.DataFrame) -> str:
    """GitHub-flavoured Markdown table of an already formatted frame."""
    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    rule = "| " + " | ".join("---" for _ in frame.columns) + " |"
    body = ["| " + " | ".join(row) + " |" for row in frame.astype(str).to_numpy().tolist()]
    return "\n".join([header, rule, *body]) + "\n"


def _escape_latex(text: str) -> str:
    for char in ("\\", "&", "%", "$", "#", "_", "{", "}"):
        if char == "\\":
            text = text.replace(char, r"\textbackslash{}")
        else:
            text = text.replace(char, "\\" + char)
    return text


def to_latex(frame: pd.DataFrame, caption: str, label: str, numeric: list[bool]) -> str:
    """LaTeX ``table`` (booktabs, scaled to the text width) of an already formatted frame.

    Parameters
    ----------
    frame : pandas.DataFrame
        Formatted text cells.
    caption, label : str
        Caption text and label suffix (the label becomes ``tab:<label>``).
    numeric : list of bool
        Per column, whether it is right-aligned (numeric) or left-aligned.
    """
    spec = "".join("r" if is_num else "l" for is_num in numeric)
    header = " & ".join(_escape_latex(str(c)) for c in frame.columns) + r" \\"
    body = [
        " & ".join(_escape_latex(cell) for cell in row) + r" \\"
        for row in frame.astype(str).to_numpy().tolist()
    ]
    lines = [
        r"\begin{table}[t]",
        r"\centering\footnotesize",
        rf"\caption{{{_escape_latex(caption)}}}\label{{tab:{label}}}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{spec}}}",
        r"\toprule",
        header,
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def numeric_columns(frame: pd.DataFrame) -> list[bool]:
    """Per column of the raw (unformatted) frame, whether it holds numbers (not booleans)."""
    return [
        bool(pd.api.types.is_numeric_dtype(frame[c]) and not pd.api.types.is_bool_dtype(frame[c]))
        for c in frame.columns
    ]
