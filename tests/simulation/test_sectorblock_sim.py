"""Tests for the positive and negative control of the sector-block estimator."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from pfopt.config import Config, ResidualConfig, SimulationConfig
from pfopt.simulation.sectorblock import (
    DESIGNS,
    METHODS,
    arbitrary_labels,
    control_config,
    mislabel,
    paired_control,
    paired_vs_pca,
    run_control,
)

CFG = dataclasses.replace(
    Config(),
    simulation=SimulationConfig(n_assets=40, n_groups=2, sample_sizes=(100,), n_reps=2),
    residual=ResidualConfig(
        sim_n_groups=4,
        sim_group_vol=0.08,
        sim_label_error=0.25,
        sim_sample_sizes=(100,),
        sim_n_reps=3,
    ),
)


def test_control_config_switches_the_group_factors() -> None:
    structured = control_config(CFG, True).simulation
    plain = control_config(CFG, False).simulation
    assert structured.n_groups == 4 and structured.group_vols == (0.08,)
    assert plain.n_groups == 0 and not structured.time_varying
    assert structured.sample_sizes == (100,)


def test_mislabel_keeps_the_label_counts_and_changes_at_most_the_chosen_entries() -> None:
    groups = np.repeat(np.arange(4), 10)
    original = groups.copy()
    out = mislabel(groups, 0.25, np.random.default_rng(0))
    np.testing.assert_array_equal(groups, original)
    assert sorted(out.tolist()) == sorted(groups.tolist())
    assert 0 < (out != groups).sum() <= 10
    np.testing.assert_array_equal(mislabel(groups, 0.0, np.random.default_rng(0)), groups)


def test_arbitrary_labels_have_near_equal_group_sizes() -> None:
    labels = arbitrary_labels(100, 10, np.random.default_rng(1))
    assert np.bincount(labels).tolist() == [10] * 10
    assert not np.array_equal(labels, np.sort(labels))


@pytest.fixture(scope="module")
def scores():
    return pd.concat([run_control(CFG, True), run_control(CFG, False)], ignore_index=True)


def test_run_control_shape_oracle_bound_and_reproducibility(scores) -> None:
    assert set(scores["distribution"]) == set(DESIGNS)
    assert len(scores) == len(DESIGNS) * len(METHODS) * 3
    assert set(scores["method"]) == set(METHODS)
    assert (scores["var_ratio"] >= 1.0 - 1e-6).all()
    again = run_control(CFG, True)
    first = scores[scores["distribution"] == DESIGNS[0]].reset_index(drop=True)
    np.testing.assert_array_equal(first["var_ratio"].to_numpy(), again["var_ratio"].to_numpy())


def test_paired_tables(scores) -> None:
    paired = paired_control(scores)
    assert {"design", "T", "method", "mean_log_ratio", "ci_low", "ci_high"} <= set(paired.columns)
    versus = paired_vs_pca(scores)
    assert len(versus) == len(DESIGNS) * 2 and (versus["n"] == 3).all()
    assert (versus["ci_low"] <= versus["mean_log_ratio"]).all()
    assert (versus["mean_log_ratio"] <= versus["ci_high"]).all()
    assert set(versus["method"]) == {"GMV-PCA-SECTOR", "GMV-PCA-SECTOR-NOISY"}
