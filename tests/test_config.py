"""Tests for config loading, validation, round-tripping and hashing."""

import dataclasses
import logging
from pathlib import Path

import pytest
import yaml

from pfopt.config import Config, config_from_dict, dump_config, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config.yaml"


def test_repo_config_loads_with_plan_defaults() -> None:
    cfg = load_config(CONFIG_PATH)
    assert cfg.run.seed == 42
    assert cfg.run.annualization == 252
    assert cfg.run.rf == 0.0
    assert cfg.universe.n_assets == 100
    assert cfg.universe.lookback_days == 504
    assert cfg.universe.dollar_volume_window == 252
    assert cfg.backtest.rebalance_days == 21
    assert cfg.backtest.cost_bps == 10.0
    assert cfg.backtest.max_weight is None
    assert cfg.pca.k_rule == "mp"
    assert cfg.hrp.linkage == "single"
    assert cfg.sensitivity.lookback_days == (252, 504, 756)
    assert cfg.simulation.n_groups == 2 and cfg.simulation.distributions[0] == "gaussian"
    assert cfg.data.stocks_file == "sp500_stocks.csv"
    assert cfg.data.index_columns == ("Date", "S&P500")
    assert cfg.data.start_date == "2010-01-04"
    assert cfg.data.end_date == "2024-12-20"
    assert cfg.eda.stress_date == "2020-03-31"
    assert cfg.eda.highlight_years == (2020, 2022)
    assert cfg.eda.frontier_horizon_days == 252 and cfg.eda.frontier_points == 50
    assert cfg.backtest.mu_convention == "arithmetic"
    assert cfg.regimes.covid_start == "2020-02-19" and cfg.regimes.stress_year == 2022
    assert cfg.regimes.vol_window_days == 63 and cfg.regimes.min_history_days == 252
    assert cfg.stats.block_length == 10 and cfg.stats.n_boot == 5000
    assert cfg.stats.sensitivity_block_lengths == (5, 10, 20)
    assert cfg.stats.references == ("GMV-SAMPLE", "EW", "GMV-LW")
    assert cfg.stats.mde_power == 0.8
    assert cfg.pipeline_check.n_symbols == 150 and cfg.pipeline_check.n_panels == 4


def test_unquoted_yaml_date_rejected(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("data:\n  start_date: 2010-01-04\n", encoding="utf-8")
    with pytest.raises(ValueError, match="quoted ISO string"):
        load_config(path)


def test_yaml_matches_dataclass_defaults() -> None:
    assert load_config(CONFIG_PATH) == Config()


def test_round_trip(tmp_path: Path) -> None:
    cfg = load_config(CONFIG_PATH)
    out = tmp_path / "roundtrip.yaml"
    dump_config(cfg, out)
    assert load_config(out) == cfg
    assert yaml.safe_load(out.read_text()) == cfg.to_dict()


def test_hash_stable_and_sensitive() -> None:
    cfg = Config()
    assert cfg.config_hash() == Config().config_hash()
    changed = dataclasses.replace(cfg, run=dataclasses.replace(cfg.run, seed=7))
    assert changed.config_hash() != cfg.config_hash()
    assert len(cfg.config_hash(8)) == 8


def test_cache_key_depends_only_on_result_determining_sections() -> None:
    base = Config()
    unrelated = dataclasses.replace(
        base,
        eda=dataclasses.replace(base.eda, frontier_points=20),
        stats=dataclasses.replace(base.stats, n_boot=100),
        regimes=dataclasses.replace(base.regimes, vol_window_days=21),
    )
    assert unrelated.cache_key() == base.cache_key()
    assert unrelated.config_hash() != base.config_hash()
    for changed in (
        dataclasses.replace(base, universe=dataclasses.replace(base.universe, n_assets=50)),
        dataclasses.replace(base, backtest=dataclasses.replace(base.backtest, cost_bps=5.0)),
        dataclasses.replace(base, run=dataclasses.replace(base.run, seed=1)),
    ):
        assert changed.cache_key() != base.cache_key()
    assert len(base.cache_key(8)) == 8


def test_missing_keys_use_defaults() -> None:
    assert config_from_dict({"universe": {"n_assets": 50}}).universe.n_assets == 50
    assert config_from_dict({}) == Config()


def test_unknown_key_rejected() -> None:
    with pytest.raises(ValueError, match="unknown key"):
        config_from_dict({"universe": {"n_assetz": 50}})
    with pytest.raises(ValueError, match="unknown key"):
        config_from_dict({"bogus": {}})


@pytest.mark.parametrize(
    ("section", "values"),
    [
        ("backtest", {"rebalance_days": 0}),
        ("backtest", {"cost_bps": -1.0}),
        ("backtest", {"max_weight": 0.0}),
        ("backtest", {"max_weight": 1.5}),
        ("universe", {"n_assets": 0}),
        ("universe", {"lookback_days": -5}),
        ("run", {"annualization": 0}),
        ("hrp", {"linkage": "complete"}),
        ("pca", {"k_rule": "magic"}),
        ("pca", {"k_rule": "fixed"}),
        ("data", {"start_date": "not-a-date"}),
        ("data", {"start_date": "2020-02-01", "end_date": "2020-01-01"}),
        ("data", {"winsor_lower": 0.9, "winsor_upper": 0.1}),
        ("data", {"stocks_file": ""}),
        ("eda", {"stress_date": "march"}),
        ("eda", {"corr_window_days": 1}),
        ("regimes", {"covid_start": "march"}),
        ("regimes", {"covid_start": "2021-01-01", "covid_end": "2020-12-31"}),
        ("regimes", {"stress_year": 2020}),
        ("regimes", {"vol_window_days": 1}),
        ("regimes", {"min_history_days": 0}),
        ("stats", {"block_length": 0}),
        ("stats", {"sensitivity_block_lengths": [5, 0]}),
        ("stats", {"n_boot": 0}),
        ("stats", {"references": []}),
        ("stats", {"alpha": 1.0}),
        ("stats", {"mde_power": 1.0}),
        ("pipeline_check", {"n_panels": 1}),
        ("pipeline_check", {"dv_persistence": 1.0}),
        ("pipeline_check", {"z2_tolerance": 0.0}),
        ("eda", {"frontier_horizon_days": 1}),
        ("eda", {"frontier_points": 1}),
        ("backtest", {"mu_convention": "geometric"}),
        ("eda", {"index_vol_window_days": 0}),
        ("data", {"stocks_columns": ["Symbol", "Close"]}),
        ("data", {"index_columns": ["Date"]}),
        ("simulation", {"n_groups": -1}),
        ("simulation", {"sample_sizes": [50, 252]}),
        ("simulation", {"n_reps": 1}),
        ("simulation", {"distributions": ["laplace"]}),
        ("simulation", {"t_df": 4.0}),
        ("simulation", {"group_vols": []}),
        ("simulation", {"loading_low": 2.0, "loading_high": 1.0}),
        ("simulation", {"idio_vol_low": 0.0}),
        ("simulation", {"alpha": 0.0}),
        ("simulation", {"horizon_days": 0}),
        ("simulation", {"vol_log_sd": -0.1}),
        ("simulation", {"vol_persistence": 1.0}),
        ("simulation", {"loading_persistence": -0.5}),
    ],
)
def test_invalid_values_rejected(section: str, values: dict) -> None:
    with pytest.raises(ValueError):
        config_from_dict({section: values})


def test_infeasible_max_weight_rejected() -> None:
    with pytest.raises(ValueError, match="infeasible"):
        config_from_dict({"universe": {"n_assets": 100}, "backtest": {"max_weight": 0.005}})


def test_t_not_greater_than_n_warns(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="pfopt.config"):
        config_from_dict({"universe": {"n_assets": 200, "lookback_days": 100}})
    assert any("rank-deficient" in rec.message for rec in caplog.records)


def test_no_warning_when_t_exceeds_n(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="pfopt.config"):
        Config()
    assert not caplog.records


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")
