"""Tests for the studentized bootstrap, multiple-testing adjustment and detectable effects."""

import numpy as np
import pandas as pd
import pytest

from pfopt.backtest.inference import (
    _block_sums,
    _pair_statistic,
    adjust_pvalues,
    minimum_detectable_effect,
    studentized_compare,
)


def test_adjust_pvalues_known_values() -> None:
    p = [0.01, 0.04, 0.03, 0.005]
    np.testing.assert_allclose(adjust_pvalues(p, "holm"), [0.03, 0.06, 0.06, 0.02])
    np.testing.assert_allclose(adjust_pvalues(p, "bh"), [0.02, 0.04, 0.04, 0.02])
    np.testing.assert_allclose(adjust_pvalues([0.4], "holm"), [0.4])
    np.testing.assert_allclose(adjust_pvalues([0.3, 0.2], "holm"), [0.4, 0.4])
    np.testing.assert_allclose(adjust_pvalues([0.9, 0.8], "holm"), [1.0, 1.0])


def test_adjust_pvalues_matches_statsmodels_and_validates() -> None:
    multipletests = pytest.importorskip("statsmodels.stats.multitest").multipletests
    p = np.random.default_rng(0).uniform(size=25) ** 2
    for ours, theirs in (("holm", "holm"), ("bh", "fdr_bh")):
        np.testing.assert_allclose(adjust_pvalues(p, ours), multipletests(p, method=theirs)[1])
    with pytest.raises(ValueError):
        adjust_pvalues([0.1, 1.2])
    with pytest.raises(ValueError):
        adjust_pvalues([0.1], "bonferroni")
    with pytest.raises(ValueError):
        adjust_pvalues([])


def test_minimum_detectable_effect() -> None:
    assert minimum_detectable_effect(1.0, 0.05, 0.8) == pytest.approx(2.8016, abs=1e-4)
    out = minimum_detectable_effect(np.array([1.0, 2.0]))
    np.testing.assert_allclose(out, [2.8016, 5.6032], atol=1e-3)
    with pytest.raises(ValueError):
        minimum_detectable_effect(1.0, 0.05, 1.0)


@pytest.mark.parametrize("test", ["log_var_ratio", "sharpe_diff"])
def test_delta_method_standard_error_matches_monte_carlo(test) -> None:
    rng = np.random.default_rng(5)
    n_obs, n_rep = 400, 600
    cov = np.array([[1.0, 0.5], [0.5, 1.5]]) * 1e-4
    chol = np.linalg.cholesky(cov)
    d, se = np.empty(n_rep), np.empty(n_rep)
    for i in range(n_rep):
        x = 0.0005 + rng.standard_normal((n_obs, 2)) @ chol.T
        mean, blocks = _block_sums(np.concatenate([x, x**2], axis=1)[None], 5)
        d_i, se_i = _pair_statistic(test, mean, blocks, 0, 1, 2, n_obs)
        d[i], se[i] = d_i[0], se_i[0]
    assert se.mean() / d.std(ddof=1) == pytest.approx(1.0, abs=0.12)


def make_returns(seed: int = 1, n_obs: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0004, 0.01, size=n_obs)
    other = rng.normal(0.0004, 0.01, size=n_obs)
    return pd.DataFrame(
        {
            "ref": base,
            "half": 0.5 * base + 0.001 * other,
            "indep": other,
        }
    )


def test_table_structure_and_known_scaling() -> None:
    frame = make_returns()
    out = studentized_compare(frame, ["ref"], block_length=5, n_boot=300, seed=1)
    assert len(out) == 2 * 2
    half = out[(out.strategy == "half") & (out.test == "log_var_ratio")].iloc[0]
    assert half["estimate"] == pytest.approx(np.log(0.25), abs=0.05)
    assert half["ci_high"] < 0 and half["significant"] and half["significant_holm"]
    assert half["mde"] > 0 and half["mde_vol_pct"] == pytest.approx(
        100 * (np.exp(half["mde"] / 2) - 1)
    )
    assert out.loc[out.test == "sharpe_diff", "mde_vol_pct"].isna().all()
    for _, r in out.iterrows():
        assert r["ci_low"] < r["estimate"] < r["ci_high"]
        assert 0.0 < r["p_value"] <= 1.0
        assert r["p_value"] <= r["p_holm"] + 1e-12 and r["p_bh"] <= r["p_holm"] + 1e-12


def test_sharpe_estimate_is_the_annualized_difference() -> None:
    frame = make_returns(seed=2)
    out = studentized_compare(frame, ["ref"], block_length=5, n_boot=50, seed=1, annualization=252)
    row = out[(out.strategy == "indep") & (out.test == "sharpe_diff")].iloc[0]
    sr = frame.mean() / frame.std(ddof=0) * np.sqrt(252)
    assert row["estimate"] == pytest.approx(sr["indep"] - sr["ref"])


def test_identical_series_and_chunk_invariance() -> None:
    frame = make_returns(seed=3)
    frame["copy"] = frame["ref"]
    a = studentized_compare(frame, ["ref"], block_length=5, n_boot=40, seed=1, chunk_size=7)
    b = studentized_compare(frame, ["ref"], block_length=5, n_boot=40, seed=1, chunk_size=100)
    pd.testing.assert_frame_equal(a, b)
    same = a[a.strategy == "copy"]
    assert (same["estimate"].abs() < 1e-12).all() and (same["p_value"] == 1.0).all()
    again = studentized_compare(frame, ["ref"], block_length=5, n_boot=40, seed=1)
    pd.testing.assert_frame_equal(a, again)


def test_input_validation() -> None:
    frame = make_returns(n_obs=50)
    with pytest.raises(ValueError, match="not found"):
        studentized_compare(frame, ["zzz"], block_length=5, n_boot=10)
    with pytest.raises(ValueError, match="two blocks"):
        studentized_compare(frame, ["ref"], block_length=40, n_boot=10)
    bad = frame.copy()
    bad.iloc[3, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        studentized_compare(bad, ["ref"], block_length=5, n_boot=10)
