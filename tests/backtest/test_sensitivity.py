"""Tests for the sensitivity-analysis driver on a small synthetic panel."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from pfopt.backtest import sensitivity as sens
from pfopt.backtest.engine import ALL_STRATEGIES
from pfopt.backtest.metrics import performance_metrics
from pfopt.config import BacktestConfig, Config, SensitivityConfig, UniverseConfig

N_DATES, N_SYMBOLS = 130, 14
CFG = dataclasses.replace(
    Config(),
    universe=UniverseConfig(n_assets=8, lookback_days=40, dollar_volume_window=20),
    backtest=BacktestConfig(rebalance_days=5),
    sensitivity=SensitivityConfig(
        n_assets=(5, 8, 30),
        lookback_days=(20, 40, 60),
        cost_bps=(0.0, 5.0, 10.0, 20.0),
        max_weight=(0.5,),
        pca_k_fixed=(1, 2),
        hrp_linkage=("single", "ward"),
    ),
)


@pytest.fixture(scope="module")
def data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2020-01-01", periods=N_DATES)
    symbols = [f"S{i:02d}" for i in range(N_SYMBOLS)]
    market = rng.normal(0.0, 0.008, size=(N_DATES, 1))
    log_r = pd.DataFrame(
        0.0004 + market + rng.normal(0.0, 0.01, size=(N_DATES, N_SYMBOLS)),
        index=dates,
        columns=symbols,
    )
    volume = pd.DataFrame(
        rng.uniform(1.0, 3.0, size=(N_DATES, N_SYMBOLS)), index=dates, columns=symbols
    )
    return log_r, volume, np.expm1(log_r)


@pytest.fixture(scope="module")
def built(data):
    return sens.build_scenarios(CFG, data[0])


def test_build_scenarios_counts_and_skips(built) -> None:
    scenarios, skipped = built
    ids = [s.id for s in scenarios]
    assert ids[0] == "baseline" and len(set(ids)) == len(ids)
    by_grid = {g: [s for s in scenarios if s.grid == g] for g in sens.GRIDS}
    assert [s.variant for s in by_grid["lookback"]] == ["T=20", "T=60"]
    assert [s.variant for s in by_grid["universe"]] == ["N=5"]
    assert [s.cost_bps for s in by_grid["cost"]] == [0.0, 5.0, 20.0]
    assert [s.variant for s in by_grid["pca_k"]] == ["K=1", "K=2", "Bai-Ng"]
    assert [s.variant for s in by_grid["hrp_linkage"]] == ["ward"]
    assert len(scenarios) == 12
    assert len(skipped) == 1 and skipped[0].startswith("N=30")


def test_each_scenario_changes_exactly_one_setting(built) -> None:
    scenarios, _ = built
    base = scenarios[0].cfg.to_dict()
    for s in scenarios[1:]:
        other = s.cfg.to_dict()
        changed = [
            (section, key)
            for section in ("universe", "backtest", "pca", "hrp")
            for key in base[section]
            if base[section][key] != other[section][key]
        ]
        if s.grid == "cost":
            assert changed == [] and s.cost_bps != CFG.backtest.cost_bps
        elif s.grid == "pca_k":
            assert {sec for sec, _ in changed} == {"pca"}
        else:
            assert len(changed) == 1, (s.id, changed)


def test_cost_scenarios_share_baseline_weights(built) -> None:
    scenarios, _ = built
    key = scenarios[0].cfg.cache_key()
    assert all(s.cfg.cache_key() == key for s in scenarios if s.grid == "cost")
    assert len({s.cfg.cache_key() for s in scenarios}) == len(scenarios) - 3


def test_strategy_subsets(built) -> None:
    scenarios, _ = built
    for s in scenarios:
        if s.grid == "pca_k":
            assert s.strategies == ("GMV-PCA",)
        elif s.grid == "hrp_linkage":
            assert s.strategies == sens.HRP_STRATEGIES
        else:
            assert s.strategies == ALL_STRATEGIES


def test_trial_counts(built) -> None:
    scenarios, _ = built
    table = sens.trial_counts(scenarios).set_index("grid")
    assert table.loc["baseline", "combinations"] == len(ALL_STRATEGIES)
    assert table.loc["pca_k", "combinations"] == 3
    assert table.loc["hrp_linkage", "combinations"] == 2
    assert table.loc["total", "variants"] == len(scenarios)
    assert table.loc["total", "combinations"] == table.drop("total")["combinations"].sum()


def test_run_scenarios_costs_cache_and_metrics(built, data, tmp_path, monkeypatch) -> None:
    scenarios, _ = built
    picked = [
        s for s in scenarios if s.id in ("baseline", "cost:0 bps", "cost:20 bps", "pca_k:K=1")
    ]
    args = (picked, data[0], data[1], data[2], tmp_path)
    runs = sens.run_scenarios(*args)

    free, base, dear = (runs[i]["GMV-SAMPLE"] for i in ("cost:0 bps", "baseline", "cost:20 bps"))
    pd.testing.assert_series_equal(free.net, free.gross, check_names=False)
    assert (free.net >= base.net - 1e-15).all() and (base.net >= dear.net - 1e-15).all()
    assert list(runs["pca_k:K=1"]) == ["GMV-PCA"]

    def fail(*_args, **_kwargs):
        raise AssertionError("weights regenerated although cached")

    monkeypatch.setattr(sens, "generate_weights", fail)
    again = sens.run_scenarios(*args)
    for key in runs:
        for strategy in runs[key]:
            pd.testing.assert_series_equal(runs[key][strategy].net, again[key][strategy].net)

    start = sens.common_start(runs)
    assert start == max(sim.net.index[0] for sims in runs.values() for sim in sims.values())
    table = sens.scenario_metrics(base_sims := runs["baseline"], start, 252, 0.0)
    net = base_sims["EW"].net
    expected = performance_metrics(net.loc[net.index >= start])
    assert table.loc["EW", "sharpe"] == pytest.approx(expected["sharpe"])
    assert table.loc["EW", "n_days"] == (net.index >= start).sum()

    long = sens.metrics_long(picked, runs, 252, 0.0)
    assert set(long["period"]) == {"own", "common"}
    assert len(long) == 2 * sum(len(s.strategies) for s in picked)
    common = long[long.period == "common"]
    assert common.groupby(["grid", "variant", "strategy"])["n_days"].first().nunique() == 1


def test_turnover_base_is_the_rebalance_held_on_the_start_day() -> None:
    dates = pd.to_datetime(["2020-01-03", "2020-01-10", "2020-01-17", "2020-01-24"])
    turnover = pd.Series([1.0, 0.2, 0.3, 0.4], index=dates)
    kept = sens._turnover_from(turnover, pd.Timestamp("2020-01-14"))
    assert list(kept.index) == list(dates[1:])
    kept = sens._turnover_from(turnover, pd.Timestamp("2020-01-02"))
    assert list(kept.index) == list(dates)
