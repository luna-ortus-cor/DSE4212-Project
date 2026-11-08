"""Tests for the simulation runner and its summaries."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from pfopt.config import Config, SimulationConfig
from pfopt.simulation.dgp import simulate_returns
from pfopt.simulation.run import (
    COVARIANCE,
    LONG_ONLY,
    METHODS,
    UNCONSTRAINED,
    evaluate_sample,
    oracle_portfolios,
    paired_vs_sample,
    run_grid,
    summarize,
    true_model,
    winner_table,
)

SIM = SimulationConfig(n_assets=12, n_groups=2, sample_sizes=(30, 60), n_reps=4)
CFG = dataclasses.replace(Config(), simulation=SIM)


@pytest.fixture(scope="module")
def model():
    return true_model(CFG)


@pytest.fixture(scope="module")
def grid(model):
    return run_grid(CFG, "gaussian", model)


def test_oracle_weights_and_variances(model) -> None:
    oracle = oracle_portfolios(model.sigma)
    long_only, free = oracle.weights[LONG_ONLY], oracle.weights[UNCONSTRAINED]
    assert long_only.min() >= -1e-9 and long_only.sum() == pytest.approx(1.0)
    assert free.sum() == pytest.approx(1.0)
    assert oracle.variance[UNCONSTRAINED] <= oracle.variance[LONG_ONLY] + 1e-15
    assert oracle.variance[LONG_ONLY] == pytest.approx(long_only @ model.sigma @ long_only)


def test_evaluate_sample_rows_and_oracle_bound(model) -> None:
    oracle = oracle_portfolios(model.sigma)
    x = simulate_returns(model, 60, np.random.default_rng(1))
    rows, info = evaluate_sample(x, oracle, CFG)
    frame = pd.DataFrame(rows)
    assert len(frame) == 4 + 8 + 2
    assert set(frame["method"]) == set(METHODS)
    assert set(frame[frame["constraint"] == UNCONSTRAINED]["method"]) == {
        "GMV-SAMPLE",
        "GMV-LW",
        "GMV-RMT",
        "GMV-PCA",
    }
    weights = frame[frame["constraint"] != COVARIANCE]
    assert (weights["var_ratio"] >= 1 - 1e-6).all()
    assert (weights["l1_dist"] >= 0).all()
    covariance = frame[frame["constraint"] == COVARIANCE]
    assert covariance["frob_error"].notna().all() and weights["frob_error"].isna().all()
    assert set(info) == {"rmt_n_signal", "pca_K", "lw_shrinkage"}


def test_large_sample_recovers_the_oracle(model) -> None:
    oracle = oracle_portfolios(model.sigma)
    x = simulate_returns(model, 40_000, np.random.default_rng(2))
    frame = pd.DataFrame(evaluate_sample(x, oracle, CFG)[0])
    sample = frame[frame["method"] == "GMV-SAMPLE"].set_index("constraint")
    assert sample.loc[COVARIANCE, "frob_error"] < 0.05
    assert sample.loc[LONG_ONLY, "var_ratio"] < 1.01
    assert sample.loc[UNCONSTRAINED, "var_ratio"] < 1.01
    assert sample.loc[UNCONSTRAINED, "l1_dist"] < 0.5


def test_grid_shape_and_no_replication_below_the_oracle(grid) -> None:
    scores, info = grid
    assert len(scores) == 2 * 4 * 14 and len(info) == 2 * 4
    assert set(scores["T"]) == {30, 60} and set(scores["distribution"]) == {"gaussian"}
    assert scores["var_ratio"].min() >= 1 - 1e-6


def test_replications_do_not_depend_on_the_number_of_replications(model, grid) -> None:
    more = dataclasses.replace(CFG, simulation=dataclasses.replace(SIM, n_reps=5))
    scores, _ = run_grid(more, "gaussian", model)
    kept = scores[scores["rep"] < 4].reset_index(drop=True)
    pd.testing.assert_frame_equal(kept, grid[0])


def test_student_t_run_differs_from_gaussian(model, grid) -> None:
    scores, _ = run_grid(CFG, "student_t", model)
    assert set(scores["distribution"]) == {"student_t"}
    assert not np.allclose(
        scores["frob_error"].to_numpy(dtype=float),
        grid[0]["frob_error"].to_numpy(dtype=float),
        equal_nan=True,
    )


def test_summarize_matches_hand_computation(grid) -> None:
    scores, _ = grid
    summary = summarize(scores, 0.05)
    cell = scores[
        (scores["T"] == 30) & (scores["method"] == "GMV-LW") & (scores["constraint"] == LONG_ONLY)
    ]["var_ratio"]
    row = summary[
        (summary["T"] == 30)
        & (summary["method"] == "GMV-LW")
        & (summary["constraint"] == LONG_ONLY)
        & (summary["metric"] == "var_ratio")
    ].iloc[0]
    assert row["mean"] == pytest.approx(cell.mean())
    assert row["se"] == pytest.approx(cell.std(ddof=1) / 2.0)
    assert row["ci_low"] < row["mean"] < row["ci_high"] and row["n"] == 4
    assert row["median"] == pytest.approx(cell.median())
    assert summary[summary["metric"] == "frob_error"]["constraint"].eq(COVARIANCE).all()


def _toy_scores() -> pd.DataFrame:
    rows = []
    for rep in range(4):
        ratios = {"GMV-SAMPLE": 2.0, "GMV-LW": 1.0 + 0.1 * rep, "GMV-RMT": 3.0}
        for method, value in ratios.items():
            rows.append(
                {
                    "distribution": "gaussian",
                    "T": 50,
                    "rep": rep,
                    "method": method,
                    "constraint": LONG_ONLY,
                    "frob_error": np.nan,
                    "var_ratio": value,
                    "l1_dist": 0.0,
                }
            )
    return pd.DataFrame(rows)


def test_paired_comparison_and_winners_on_a_toy_frame() -> None:
    scores = _toy_scores()
    paired = paired_vs_sample(scores).set_index("method")
    assert paired.loc["GMV-SAMPLE", "mean_log_ratio"] == 0.0
    assert paired.loc["GMV-SAMPLE", "beats_sample"] == 0.0
    expected = np.mean([np.log((1.0 + 0.1 * r) / 2.0) for r in range(4)])
    assert paired.loc["GMV-LW", "mean_log_ratio"] == pytest.approx(expected)
    assert paired.loc["GMV-LW", "beats_sample"] == 1.0 and paired.loc["GMV-LW", "best_rate"] == 1.0
    assert paired.loc["GMV-RMT", "mean_log_ratio"] == pytest.approx(np.log(1.5))
    assert paired.loc["GMV-RMT", "beats_sample"] == 0.0
    assert paired["best_rate"].sum() == pytest.approx(1.0)
    winners = winner_table(summarize(scores))
    assert len(winners) == 1 and winners.iloc[0]["method"] == "GMV-LW"


def test_best_rates_sum_to_one_in_every_cell(grid) -> None:
    paired = paired_vs_sample(grid[0])
    totals = paired.groupby(["distribution", "T", "constraint"])["best_rate"].sum()
    np.testing.assert_allclose(totals.to_numpy(), 1.0)


def test_time_varying_grid_has_a_replication_specific_oracle(model) -> None:
    tv_sim = dataclasses.replace(SIM, time_varying=True, horizon_days=5, n_reps=3)
    tv_cfg = dataclasses.replace(CFG, simulation=tv_sim)
    scores, info = run_grid(tv_cfg, "gaussian", model)
    assert len(scores) == 2 * 3 * 14 and len(info) == 2 * 3
    assert scores["var_ratio"].min() >= 1 - 1e-6
    same_size = dataclasses.replace(CFG, simulation=dataclasses.replace(SIM, n_reps=3))
    static, _ = run_grid(same_size, "gaussian", model)
    assert not np.allclose(
        scores["frob_error"].to_numpy(dtype=float),
        static["frob_error"].to_numpy(dtype=float),
        equal_nan=True,
    )
