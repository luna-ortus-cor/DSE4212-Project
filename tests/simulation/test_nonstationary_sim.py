"""Tests for the volatility calibration and the positive control (step 8.22)."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from pfopt.config import Config, NonStationarityConfig, SimulationConfig
from pfopt.simulation.dgp import ar1_paths
from pfopt.simulation.nonstationary import (
    DESIGNS,
    calibrate_market_volatility,
    design_config,
    methods,
    paired_control,
    run_design,
)
from pfopt.simulation.run import true_model

SIM = SimulationConfig(n_assets=12, n_groups=2, sample_sizes=(30, 60), n_reps=4)
CFG = dataclasses.replace(
    Config(),
    simulation=SIM,
    nonstationarity=NonStationarityConfig(halflives=(20,), sim_sample_sizes=(40, 60), sim_n_reps=3),
)
CALIBRATED = {"vol_log_sd": 0.3, "vol_persistence": 0.95}


def test_calibration_recovers_known_parameters() -> None:
    rng = np.random.default_rng(11)
    n_days, sd, phi = 60_000, 0.5, 0.98
    log_vol = ar1_paths(n_days, 1, phi, sd, rng)[:, 0]
    market = np.exp(log_vol - sd**2) * 0.01 * rng.standard_normal(n_days)
    out = calibrate_market_volatility(market, 21)
    # Block means are slightly more persistent and less dispersed than the daily process
    # (about +0.007 and -7% for these values), so the tolerances allow for that approximation.
    assert out["vol_log_sd"] == pytest.approx(sd, rel=0.15)
    assert out["vol_persistence"] == pytest.approx(phi, abs=0.02)
    assert out["n_blocks"] == n_days // 21


def test_calibration_of_constant_volatility_fails_or_is_small() -> None:
    market = 0.01 * np.random.default_rng(2).standard_normal(5000)
    try:
        out = calibrate_market_volatility(market, 21)
    except ValueError:
        return
    assert out["vol_log_sd"] < 0.15


def test_calibration_validation() -> None:
    with pytest.raises(ValueError, match="at least 10 blocks"):
        calibrate_market_volatility(np.ones(50) * 0.01, 21)
    with pytest.raises(ValueError, match="finite"):
        calibrate_market_volatility(np.full(500, np.nan), 21)


def test_design_configs() -> None:
    stationary, apriori, calibrated = (design_config(SIM, d, CALIBRATED) for d in DESIGNS)
    assert not stationary.time_varying and apriori.time_varying and calibrated.time_varying
    assert apriori.vol_log_sd == SIM.vol_log_sd
    assert calibrated.vol_log_sd == 0.3 and calibrated.vol_persistence == 0.95
    with pytest.raises(ValueError):
        design_config(SIM, "nope", CALIBRATED)


@pytest.mark.parametrize("design", DESIGNS)
def test_run_design_shape_bound_and_reproducibility(design) -> None:
    model = true_model(CFG)
    a = run_design(CFG, design, CALIBRATED, model)
    b = run_design(CFG, design, CALIBRATED, model)
    pd.testing.assert_frame_equal(a, b)
    expected_methods = set(methods((20,)))
    assert set(a["method"]) == expected_methods
    assert len(a) == 2 * 3 * len(expected_methods)
    assert (a["var_ratio"] >= 1.0 - 1e-6).all()
    assert (a["distribution"] == design).all()


def test_paired_control_columns_and_sample_row() -> None:
    scores = run_design(CFG, DESIGNS[0], CALIBRATED)
    paired = paired_control(scores)
    assert {"design", "T", "method", "mean_log_ratio", "ci_low", "ci_high"} <= set(paired.columns)
    sample = paired[paired["method"] == "GMV-SAMPLE"]
    assert (sample["mean_log_ratio"] == 0.0).all()
