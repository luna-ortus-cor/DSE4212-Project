"""Tests for the synthetic-panel pipeline validation."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from pfopt.config import BacktestConfig, Config, PipelineCheckConfig, UniverseConfig
from pfopt.simulation.pipeline import (
    PIPELINE_STRATEGIES,
    accounting_summary,
    estimation_summary,
    identity_summary,
    make_panel,
    run_panel,
)

CFG = dataclasses.replace(
    Config(),
    universe=UniverseConfig(n_assets=8, lookback_days=40, dollar_volume_window=20),
    backtest=BacktestConfig(rebalance_days=5),
    pipeline_check=PipelineCheckConfig(n_symbols=14, n_days=400, n_panels=2),
)


@pytest.fixture(scope="module")
def panel_result():
    return run_panel(CFG, 0)


def test_panel_shapes_reproducibility_and_independence() -> None:
    a, b, c = make_panel(CFG, 0), make_panel(CFG, 0), make_panel(CFG, 1)
    assert a.log_returns.shape == a.dollar_volume.shape == (400, 14)
    assert a.model.n_assets == 14
    pd.testing.assert_frame_equal(a.log_returns, b.log_returns)
    assert not np.allclose(a.log_returns.to_numpy(), c.log_returns.to_numpy())
    np.testing.assert_allclose(a.simple_returns.to_numpy(), np.expm1(a.log_returns.to_numpy()))
    assert (a.dollar_volume.to_numpy() > 0).all()


def test_universe_has_the_configured_size(panel_result) -> None:
    sizes = panel_result["estimation"]["n_universe"]
    assert (sizes == CFG.universe.n_assets).all()


def test_engine_matches_the_simulation_study(panel_result) -> None:
    estimation = panel_result["estimation"]
    assert set(estimation["strategy"]) == set(PIPELINE_STRATEGIES)
    assert (estimation["var_ratio"] - estimation["var_ratio_study"]).abs().max() < 1e-6
    assert (estimation["var_ratio"] >= 1.0 - 1e-9).all()
    ew = estimation[estimation["strategy"] == "EW"]
    assert (ew["var_ratio"] > 1.0).all()


def test_accounting_is_unbiased_for_the_true_variance(panel_result) -> None:
    accounting = panel_result["accounting"]
    z = (accounting["mean_z2"] - 1.0) / accounting["se_z2"]
    assert (z.abs() < 5.0).all()
    assert (accounting["n_days"] > 300).all()
    assert (accounting["true_vol"] > 0).all()


def test_accounting_summary_flags_a_biased_series() -> None:
    frame = pd.DataFrame(
        {
            "panel": [0, 1, 0, 1],
            "strategy": ["A", "A", "B", "B"],
            "mean_z2": [1.01, 0.99, 1.30, 1.28],
            "se_z2": [0.02, 0.02, 0.02, 0.02],
            "realized_vol": [0.1, 0.1, 0.2, 0.2],
            "true_vol": [0.1, 0.1, 0.1, 0.1],
            "pred_vol": [0.05, 0.05, 0.1, 0.1],
        }
    )
    out = accounting_summary(frame, 3.0).set_index("strategy")
    assert out.loc["A", "mean_z2"] == pytest.approx(1.0)
    assert out.loc["A", "se"] == pytest.approx(0.02 / np.sqrt(2.0))
    assert out.loc["A", "passes"] and not out.loc["B", "passes"]
    assert out.loc["B", "realized_over_true"] == pytest.approx(2.0)
    assert out.loc["A", "realized_over_predicted"] == pytest.approx(2.0)


def test_identity_summary_detects_a_difference() -> None:
    frame = pd.DataFrame(
        {
            "strategy": ["A", "A", "B", "B"],
            "var_ratio": [1.2, 1.3, 1.5, 1.6],
            "var_ratio_study": [1.2, 1.3 + 1e-9, 1.5, 1.7],
        }
    )
    out = identity_summary(frame, 1e-6).set_index("strategy")
    assert out.loc["A", "passes"] and not out.loc["B", "passes"]
    assert out.loc["B", "max_abs_diff"] == pytest.approx(0.1)


def test_estimation_summary_known_values() -> None:
    rows = []
    for panel, shift in enumerate([0.0, 0.02, -0.02]):
        for date in range(3):
            rows.append(("GMV-SAMPLE", panel, date, 1.5, 0.5))
            rows.append(("GMV-RMT", panel, date, 1.5 * np.exp(-0.05 + shift), 0.4))
    frame = pd.DataFrame(rows, columns=["strategy", "panel", "date", "var_ratio", "l1_dist"])
    out = estimation_summary(frame).set_index("strategy")
    assert out.loc["GMV-SAMPLE", "mean_log_ratio"] == pytest.approx(0.0)
    assert out.loc["GMV-RMT", "mean_log_ratio"] == pytest.approx(-0.05)
    assert out.loc["GMV-RMT", "se"] == pytest.approx(0.02 / np.sqrt(3.0))
    assert out.loc["GMV-RMT", "ci_low"] < -0.05 < out.loc["GMV-RMT", "ci_high"]
    assert out.loc["GMV-RMT", "n_panels"] == 3
    assert out.loc["GMV-RMT", "mean_l1_dist"] == pytest.approx(0.4)


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        PipelineCheckConfig(n_panels=1)
    with pytest.raises(ValueError):
        PipelineCheckConfig(dv_persistence=1.0)
    with pytest.raises(ValueError):
        PipelineCheckConfig(identity_tolerance=0.0)
