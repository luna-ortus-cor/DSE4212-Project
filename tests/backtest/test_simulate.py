"""Tests for the P&L simulator on toy panels with hand-computed answers."""

import logging

import numpy as np
import pandas as pd
import pytest

from pfopt.backtest.engine import WeightsResult, simulate, simulate_all


def frame(rows: list[list[float]], index: pd.DatetimeIndex, columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(rows, index=index, columns=columns)


DATES = pd.bdate_range("2020-01-01", periods=6)  # d0..d5
COLS = ["A", "B"]


def returns_frame(rows: list[list[float]]) -> pd.DataFrame:
    return frame(rows, DATES[: len(rows)], COLS)


def test_hand_computed_drift_turnover_and_costs() -> None:
    # Rebalance at d0 and d2, both to 50/50. Day 1 returns (10%, 0%) drift the weights.
    returns = returns_frame([[0.0, 0.0], [0.10, 0.0], [0.0, 0.0], [0.02, 0.02]])
    weights = frame([[0.5, 0.5], [0.5, 0.5]], DATES[[0, 2]], COLS)
    sim = simulate(weights, returns, cost_bps=10.0)

    drifted = np.array([0.5 * 1.1, 0.5]) / 1.05
    expected_turnover = np.abs(np.array([0.5, 0.5]) - drifted).sum()
    assert expected_turnover == pytest.approx(0.047619047619)
    np.testing.assert_allclose(sim.turnover.to_numpy(), [1.0, expected_turnover])
    np.testing.assert_allclose(sim.weights_pre.iloc[1].to_numpy(), drifted)
    np.testing.assert_allclose(sim.weights_pre.iloc[0].to_numpy(), [0.0, 0.0])

    assert list(sim.gross.index) == list(DATES[1:4])
    np.testing.assert_allclose(sim.gross.to_numpy(), [0.05, 0.0, 0.02])
    # d1 carries the initial investment cost, d2 the rebalance cost, d3 nothing.
    costs = np.array([10.0 / 1e4 * 1.0, 10.0 / 1e4 * expected_turnover, 0.0])
    np.testing.assert_allclose(sim.net.to_numpy(), sim.gross.to_numpy() - costs)
    assert sim.n_missing_returns == 0


def test_constant_target_weights_zero_cost_equals_weighted_return() -> None:
    rng = np.random.default_rng(0)
    rows = rng.normal(0.0005, 0.01, size=(6, 2)).tolist()
    returns = returns_frame(rows)
    weights = frame([[0.7, 0.3]] * 6, DATES, COLS)  # rebalance to the same target every day
    sim = simulate(weights, returns, cost_bps=0.0)
    expected = returns.iloc[1:].to_numpy() @ np.array([0.7, 0.3])
    np.testing.assert_allclose(sim.gross.to_numpy(), expected)
    np.testing.assert_array_equal(sim.gross.to_numpy(), sim.net.to_numpy())


def test_equal_weight_without_rebalancing_is_buy_and_hold() -> None:
    rng = np.random.default_rng(1)
    returns = returns_frame(rng.normal(0.001, 0.02, size=(6, 2)).tolist())
    weights = frame([[0.5, 0.5]], DATES[[0]], COLS)
    sim = simulate(weights, returns, cost_bps=10.0)
    growth = (1.0 + returns.iloc[1:]).cumprod()
    value = 0.5 * growth.sum(axis=1)  # initial value 1
    previous = np.concatenate([[1.0], value.to_numpy()[:-1]])
    np.testing.assert_allclose(sim.gross.to_numpy(), value.to_numpy() / previous - 1.0)
    np.testing.assert_allclose(sim.turnover.to_numpy(), [1.0])
    # only the initial investment is charged
    assert (sim.gross - sim.net).iloc[0] == pytest.approx(10.0 / 1e4)
    np.testing.assert_allclose((sim.gross - sim.net).iloc[1:], 0.0)


def test_net_never_exceeds_gross_and_equals_it_without_costs() -> None:
    rng = np.random.default_rng(2)
    returns = frame(rng.normal(0, 0.01, size=(6, 2)).tolist(), DATES, COLS)
    raw = rng.dirichlet(np.ones(2), size=3)
    weights = frame(raw.tolist(), DATES[[0, 2, 4]], COLS)
    with_costs = simulate(weights, returns, cost_bps=25.0)
    without = simulate(weights, returns, cost_bps=0.0)
    assert (with_costs.net <= with_costs.gross + 1e-15).all()
    assert (with_costs.net < with_costs.gross).any()
    np.testing.assert_array_equal(without.net.to_numpy(), without.gross.to_numpy())
    np.testing.assert_array_equal(with_costs.gross.to_numpy(), without.gross.to_numpy())


def test_total_cost_is_proportional_to_cost_bps() -> None:
    rng = np.random.default_rng(3)
    returns = frame(rng.normal(0, 0.01, size=(6, 2)).tolist(), DATES, COLS)
    weights = frame(rng.dirichlet(np.ones(2), size=3).tolist(), DATES[[0, 2, 4]], COLS)
    sim5 = simulate(weights, returns, cost_bps=5.0)
    sim20 = simulate(weights, returns, cost_bps=20.0)
    drag5, drag20 = (sim5.gross - sim5.net).sum(), (sim20.gross - sim20.net).sum()
    assert drag20 == pytest.approx(4.0 * drag5)
    assert drag5 == pytest.approx(5.0 / 1e4 * sim5.turnover.sum())


def test_assets_entering_and_leaving_are_counted_in_turnover() -> None:
    returns = returns_frame([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    weights = frame([[1.0, 0.0], [0.0, 1.0]], DATES[[0, 2]], COLS)  # full switch A -> B
    sim = simulate(weights, returns, cost_bps=10.0)
    np.testing.assert_allclose(sim.turnover.to_numpy(), [1.0, 2.0])
    np.testing.assert_allclose(sim.weights_pre.iloc[1].to_numpy(), [1.0, 0.0])


def test_missing_returns_of_held_assets_are_flat_and_counted(
    caplog: pytest.LogCaptureFixture,
) -> None:
    returns = returns_frame([[0.0, 0.0], [np.nan, 0.01], [np.nan, 0.02], [0.0, 0.0]])
    held_both = frame([[0.5, 0.5]], DATES[[0]], COLS)
    with caplog.at_level(logging.WARNING, logger="pfopt.backtest.engine"):
        sim = simulate(held_both, returns, cost_bps=0.0)
    assert sim.n_missing_returns == 2  # A is held on d1 and d2
    assert any("missing" in rec.message for rec in caplog.records)
    assert sim.gross.iloc[0] == pytest.approx(0.5 * 0.01)
    only_b = frame([[0.0, 1.0]], DATES[[0]], COLS)
    assert simulate(only_b, returns, cost_bps=0.0).n_missing_returns == 0  # A not held


def test_returns_up_to_d_do_not_depend_on_later_data_or_on_the_rebalance_day_return() -> None:
    rng = np.random.default_rng(4)
    rows = rng.normal(0, 0.01, size=(6, 2))
    weights = frame(rng.dirichlet(np.ones(2), size=2).tolist(), DATES[[0, 3]], COLS)
    base = simulate(weights, frame(rows.tolist(), DATES, COLS), cost_bps=10.0)
    changed = rows.copy()
    changed[0] = [0.5, -0.5]  # return on the first rebalance date: never earned
    changed[4:] = [[0.3, 0.3], [-0.2, 0.1]]  # later days
    other = simulate(weights, frame(changed.tolist(), DATES, COLS), cost_bps=10.0)
    np.testing.assert_array_equal(base.gross.iloc[:3].to_numpy(), other.gross.iloc[:3].to_numpy())
    np.testing.assert_array_equal(base.net.iloc[:3].to_numpy(), other.net.iloc[:3].to_numpy())


def test_rebalance_on_the_last_date_charges_cost_but_holds_nothing_after() -> None:
    returns = returns_frame([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    weights = frame([[0.5, 0.5], [0.2, 0.8]], DATES[[0, 2]], COLS)
    sim = simulate(weights, returns, cost_bps=100.0)
    assert list(sim.gross.index) == list(DATES[1:3])
    assert sim.net.iloc[-1] == pytest.approx(-0.01 * 0.6)  # turnover 0.6 at 100 bps


def test_daily_weights_are_the_start_of_day_holdings() -> None:
    returns = returns_frame([[0.0, 0.0], [0.10, 0.0], [0.0, 0.0], [0.0, 0.0]])
    weights = frame([[0.5, 0.5], [0.5, 0.5]], DATES[[0, 2]], COLS)
    sim = simulate(weights, returns, cost_bps=0.0)
    np.testing.assert_allclose(sim.daily_weights.iloc[0].to_numpy(), [0.5, 0.5])
    np.testing.assert_allclose(sim.daily_weights.iloc[1].to_numpy(), np.array([0.55, 0.5]) / 1.05)
    np.testing.assert_allclose(sim.daily_weights.iloc[2].to_numpy(), [0.5, 0.5])  # after rebalancing
    np.testing.assert_allclose(sim.daily_weights.sum(axis=1), 1.0)


def test_simulate_all_matches_individual_runs() -> None:
    rng = np.random.default_rng(5)
    returns = frame(rng.normal(0, 0.01, size=(6, 2)).tolist(), DATES, COLS)
    a = frame([[0.5, 0.5], [0.6, 0.4]], DATES[[0, 3]], COLS)
    b = frame([[0.1, 0.9], [0.9, 0.1]], DATES[[0, 3]], COLS)
    result = WeightsResult(
        weights={"A": a, "B": b},
        diagnostics=pd.DataFrame(index=a.index),
        universe=pd.DataFrame(True, index=a.index, columns=COLS),
    )
    out = simulate_all(result, returns, cost_bps=10.0)
    assert set(out) == {"A", "B"}
    pd.testing.assert_series_equal(out["B"].net, simulate(b, returns, 10.0).net)


def test_invalid_inputs_raise() -> None:
    returns = returns_frame([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    good = frame([[0.5, 0.5]], DATES[[0]], COLS)
    with pytest.raises(ValueError, match="cost_bps"):
        simulate(good, returns, cost_bps=-1.0)
    with pytest.raises(ValueError, match="missing"):
        simulate(frame([[1.0]], DATES[[0]], ["Z"]), returns, cost_bps=0.0)
    with pytest.raises(ValueError, match="not in simple_returns"):
        simulate(frame([[0.5, 0.5]], pd.DatetimeIndex(["2019-01-01"]), COLS), returns, 0.0)
    with pytest.raises(ValueError, match="sum"):
        simulate(frame([[0.5, 0.4]], DATES[[0]], COLS), returns, 0.0)
    with pytest.raises(ValueError, match="increasing"):
        simulate(frame([[0.5, 0.5], [0.5, 0.5]], DATES[[2, 0]], COLS), returns, 0.0)
    with pytest.raises(ValueError, match="no holding day"):
        simulate(frame([[0.5, 0.5]], DATES[[2]], COLS), returns, 0.0)
