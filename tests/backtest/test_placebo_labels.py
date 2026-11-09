"""Tests for the random-label placebo of the sector-block strategy (step 8.25)."""

import dataclasses

import numpy as np
import pandas as pd

from pfopt.backtest.residual import UNKNOWN_CODE
from pfopt.backtest.sectorblock import generate_sector_block_weights, placebo_codes
from pfopt.config import BacktestConfig, Config, UniverseConfig

CFG = dataclasses.replace(
    Config(),
    universe=UniverseConfig(n_assets=8, lookback_days=40, dollar_volume_window=20),
    backtest=BacktestConfig(rebalance_days=5),
)


def make_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2020-01-01", periods=130)
    symbols = [f"S{i:02d}" for i in range(14)]
    group = rng.normal(0.0, 0.01, size=(130, 3))
    codes = np.arange(14) % 3
    returns = group[:, codes] + rng.normal(0.0, 0.01, size=(130, 14))
    volume = rng.uniform(1.0, 3.0, size=(130, 14))
    labels = ["Tech", "Energy", "Health"] * 4 + ["Unknown", np.nan]
    return (
        pd.DataFrame(returns, index=dates, columns=symbols),
        pd.DataFrame(volume, index=dates, columns=symbols),
        pd.Series(labels, index=symbols),
    )


def test_placebo_codes_keep_sizes_and_unknown_members() -> None:
    groups = np.array([0, 0, 1, 1, 1, 2, UNKNOWN_CODE, UNKNOWN_CODE])
    out = placebo_codes(groups, np.random.default_rng(0))
    assert (out == UNKNOWN_CODE).tolist() == (groups == UNKNOWN_CODE).tolist()
    assert sorted(out[out != UNKNOWN_CODE]) == sorted(groups[groups != UNKNOWN_CODE])
    assert groups.tolist() == [0, 0, 1, 1, 1, 2, UNKNOWN_CODE, UNKNOWN_CODE]


def test_placebo_codes_actually_move_labels_and_are_seeded() -> None:
    groups = np.repeat(np.arange(4), 10)
    a = placebo_codes(groups, np.random.default_rng(1))
    b = placebo_codes(groups, np.random.default_rng(1))
    assert (a == b).all() and (a != groups).any()


def test_placebo_weights_are_valid_reproducible_and_differ_from_true_labels() -> None:
    data = make_data()
    true = generate_sector_block_weights(*data, CFG).weights
    one = generate_sector_block_weights(*data, CFG, placebo_draw=0).weights
    again = generate_sector_block_weights(*data, CFG, placebo_draw=0).weights
    other = generate_sector_block_weights(*data, CFG, placebo_draw=1).weights
    pd.testing.assert_frame_equal(one, again)
    np.testing.assert_allclose(one.sum(axis=1), 1.0, atol=1e-10)
    assert (one.to_numpy() >= -1e-12).all()
    assert not np.allclose(one.to_numpy(), true.to_numpy())
    assert not np.allclose(one.to_numpy(), other.to_numpy())
