"""Tests for the report asset helpers."""

import dataclasses

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from pfopt.config import Config, UniverseConfig  # noqa: E402
from pfopt.report.assets import (  # noqa: E402
    FIGURES,
    MAIN,
    TABLES,
    TableSpec,
    build_table,
    check_figure_files,
    format_table,
    numeric_columns,
    parameter_table,
    png_info,
    to_latex,
    to_markdown,
)
from pfopt.viz.style import save_figure  # noqa: E402


def _figure(width: float = 4.0) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(width, 2.5))
    ax.plot([0, 1], [0, 1])
    return fig


def test_png_info_reads_size_and_resolution(tmp_path) -> None:
    png, _ = save_figure(_figure(4.0), "good", tmp_path)
    width, height, dpi = png_info(png)
    assert dpi == pytest.approx(300.0, abs=0.5)
    assert width / dpi == pytest.approx(4.0, abs=0.05) and height > 0
    (tmp_path / "bad.png").write_bytes(b"not a png")
    with pytest.raises(ValueError, match="not a PNG"):
        png_info(tmp_path / "bad.png")


def test_check_figure_files_accepts_a_saved_figure_and_flags_problems(tmp_path) -> None:
    save_figure(_figure(), "ok_fig", tmp_path)
    assert check_figure_files(tmp_path, "ok_fig") == []
    assert check_figure_files(tmp_path, "absent")  # both files missing
    (tmp_path / "ok_fig.pdf").unlink()
    assert any("PDF missing" in p for p in check_figure_files(tmp_path, "ok_fig"))
    save_figure(_figure(), "stub", tmp_path)
    (tmp_path / "stub.pdf").write_bytes(b"garbage")
    assert any("%PDF" in p for p in check_figure_files(tmp_path, "stub"))


def test_check_figure_files_flags_wrong_resolution(tmp_path) -> None:
    paths = save_figure(_figure(4.0), "fine", tmp_path)
    _figure(4.0).savefig(paths[0], dpi=150)
    problems = check_figure_files(tmp_path, "fine")
    assert any("resolution" in p for p in problems)


def test_check_figure_files_flags_a_png_wider_than_the_limit(tmp_path) -> None:
    paths = save_figure(_figure(4.0), "toowide", tmp_path)
    _figure(7.5).savefig(paths[0], dpi=300)
    problems = check_figure_files(tmp_path, "toowide")
    assert any("wide" in p for p in problems)


def test_figure_and_table_specs_are_unique_and_cover_the_plan() -> None:
    keys = [f.key for f in FIGURES]
    assert len(keys) == len(set(keys)) and len({f.source for f in FIGURES}) == len(FIGURES)
    assert sum(f.role == MAIN for f in FIGURES) >= 11
    assert len({t.key for t in TABLES}) == len(TABLES)


def test_extension_tables_are_registered_in_the_appendix() -> None:
    extension = [t for t in TABLES if t.source.startswith(("pipeline_", "inference_", "nonstat_"))]
    assert len(extension) == 9
    assert all(t.role == "appendix" for t in extension)


def test_build_table_filters_selects_and_names_the_index(tmp_path) -> None:
    pd.DataFrame(
        {"basis": ["net", "gross", "net"], "strategy": ["A", "A", "B"], "x": [1.0, 2.0, 3.0]}
    ).to_csv(tmp_path / "m.csv", index=False)
    spec = TableSpec("k", "m.csv", MAIN, "c", ("strategy", "x"), ("basis", "net"))
    frame = build_table(spec, tmp_path)
    assert list(frame.columns) == ["strategy", "x"] and frame["x"].tolist() == [1.0, 3.0]

    pd.DataFrame({"a": [1.0, 2.0]}, index=["EW", "HRP"]).to_csv(tmp_path / "i.csv")
    named = build_table(TableSpec("i", "i.csv", MAIN, "c", index_label="strategy"), tmp_path)
    assert list(named.columns) == ["strategy", "a"]

    with pytest.raises(KeyError, match="nope"):
        build_table(TableSpec("k", "m.csv", MAIN, "c", ("nope",)), tmp_path)
    with pytest.raises(ValueError, match="no rows"):
        build_table(TableSpec("k", "m.csv", MAIN, "c", None, ("basis", "zzz")), tmp_path)


def test_format_table_and_renderers() -> None:
    frame = pd.DataFrame(
        {
            "strategy": ["GMV_RMT", "EW"],
            "n_days": [3263.0, 100.0],
            "p_value": [0.0004, 0.5],
            "sharpe": [0.7449, np.nan],
            "flag": [True, False],
        }
    )
    text = format_table(frame)
    assert text["n_days"].tolist() == ["3263", "100"]
    assert text["p_value"].tolist() == ["4.0e-04", "0.500"]
    assert text["sharpe"].tolist() == ["0.745", ""]
    assert text["flag"].tolist() == ["yes", "no"]
    markdown = to_markdown(text)
    assert markdown.splitlines()[1] == "| --- | --- | --- | --- | --- |"
    assert "| GMV_RMT | 3263 | 4.0e-04 | 0.745 | yes |" in markdown
    assert numeric_columns(frame) == [False, True, True, True, False]
    latex = to_latex(text, "Caption with 50% & more", "lbl", numeric_columns(frame))
    assert r"\caption{Caption with 50\% \& more}\label{tab:lbl}" in latex
    assert "GMV\\_RMT" in latex and r"\begin{tabular}{lrrrl}" in latex
    assert latex.count(r"\\") == 3  # header and two rows


def test_parameter_table_lists_parameters_without_paths_or_file_names() -> None:
    cfg = dataclasses.replace(Config(), universe=UniverseConfig(n_assets=80))
    table = parameter_table(cfg)
    assert set(table.columns) == {"section", "parameter", "value"}
    row = table[(table["section"] == "universe") & (table["parameter"] == "n_assets")]
    assert row["value"].tolist() == ["80"]
    assert not table["section"].eq("paths").any()
    assert not table["parameter"].str.endswith("_columns").any()
    assert not table["parameter"].isin(["stocks_file", "companies_file", "index_file"]).any()
    cap = table[(table["section"] == "backtest") & (table["parameter"] == "max_weight")]
    assert cap["value"].tolist() == ["none"]
