"""Tests for the nonlinear-shrinkage simulation check."""

import dataclasses

import numpy as np
import pytest

from pfopt.config import Config
from pfopt.simulation.nls import METHODS, run_nls_grid, scenario_config
from pfopt.simulation.run import COVARIANCE, LONG_ONLY, UNCONSTRAINED, paired_vs_sample


def small_config() -> Config:
    cfg = Config()
    sim = dataclasses.replace(cfg.simulation, n_assets=20, sample_sizes=(60,), n_reps=3)
    return dataclasses.replace(cfg, simulation=sim)


def test_scenario_config_switches_time_variation() -> None:
    cfg = small_config()
    assert not scenario_config(cfg, "stationary").simulation.time_varying
    assert scenario_config(cfg, "time-varying").simulation.time_varying
    with pytest.raises(ValueError):
        scenario_config(cfg, "other")


@pytest.mark.parametrize("scenario", ["stationary", "time-varying"])
def test_grid_shape_oracle_bound_and_reproducibility(scenario: str) -> None:
    cfg = small_config()
    scores = run_nls_grid(cfg, scenario)
    assert len(scores) == 3 * len(METHODS) * 3
    assert set(scores["method"]) == set(METHODS) and set(scores["distribution"]) == {scenario}
    ratios = scores.loc[scores["constraint"].isin([LONG_ONLY, UNCONSTRAINED]), "var_ratio"]
    assert (ratios >= 1.0 - 1e-6).all()
    frob = scores.loc[scores["constraint"] == COVARIANCE, "frob_error"]
    assert np.isfinite(frob).all() and (frob > 0).all()
    again = run_nls_grid(cfg, scenario)
    np.testing.assert_allclose(
        scores["var_ratio"].to_numpy(), again["var_ratio"].to_numpy(), equal_nan=True
    )


def test_paired_table_covers_nls() -> None:
    paired = paired_vs_sample(run_nls_grid(small_config(), "stationary"))
    assert "GMV-NLS" in set(paired["method"])
