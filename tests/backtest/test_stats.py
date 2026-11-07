"""Tests for the block-bootstrap significance tests."""

import numpy as np
import pandas as pd
import pytest

from pfopt.backtest.stats import (
    block_length_sensitivity,
    bootstrap_moments,
    compare_to_references,
    draw_block_starts,
    expand_block_indices,
    sharpe_ratio,
)


def frame(*columns: np.ndarray, names: str = "AB") -> pd.DataFrame:
    index = pd.bdate_range("2000-01-03", periods=len(columns[0]))
    return pd.DataFrame(dict(zip(names, columns, strict=False)), index=index)


# ------------------------------------------------------------------ resampling


def test_blocks_are_consecutive_circular_and_cut_to_length() -> None:
    n, b = 10, 3
    starts = draw_block_starts(n, b, 50, np.random.default_rng(0))
    assert starts.shape == (50, 4)  # ceil(10 / 3) blocks
    idx = expand_block_indices(starts, n, b)
    assert idx.shape == (50, n)
    assert idx.min() >= 0 and idx.max() < n
    # inside a block, consecutive positions differ by 1 modulo n
    for row in idx:
        for block_start in range(0, n - 1, b):
            for step in range(b - 1):
                pos = block_start + step
                if pos + 1 < n:
                    assert (row[pos + 1] - row[pos]) % n == 1


def test_block_of_full_length_is_a_rotation() -> None:
    idx = expand_block_indices(np.array([[4]]), 7, 7)
    np.testing.assert_array_equal(idx[0], [4, 5, 6, 0, 1, 2, 3])


def test_invalid_block_arguments_raise() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="block_length"):
        draw_block_starts(10, 0, 5, rng)
    with pytest.raises(ValueError, match="block_length"):
        draw_block_starts(10, 11, 5, rng)
    with pytest.raises(ValueError, match="n_boot"):
        draw_block_starts(10, 2, 0, rng)


def test_bootstrap_moments_are_reproducible_and_independent_of_chunk_size() -> None:
    x = np.random.default_rng(1).normal(size=(200, 3))
    a = bootstrap_moments(x, 5, 300, seed=7, chunk_size=64)
    b = bootstrap_moments(x, 5, 300, seed=7, chunk_size=300)
    c = bootstrap_moments(x, 5, 300, seed=8)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    assert not np.array_equal(a[0], c[0])
    assert a[0].shape == a[1].shape == (300, 3)


def test_bootstrap_distribution_of_the_mean_matches_theory_for_iid_data() -> None:
    x = np.random.default_rng(2).normal(0.0, 0.01, size=(1000, 1))
    means, _ = bootstrap_moments(x, 1, 3000, seed=0)
    assert means.mean() == pytest.approx(x.mean(), abs=2e-4)
    assert means.std() == pytest.approx(x.std(ddof=1) / np.sqrt(1000), rel=0.1)


def test_columns_are_resampled_jointly() -> None:
    base = np.random.default_rng(3).normal(size=500)
    x = np.column_stack([base, 2.0 * base])
    means, variances = bootstrap_moments(x, 10, 200, seed=0)
    np.testing.assert_allclose(means[:, 1], 2.0 * means[:, 0])
    np.testing.assert_allclose(variances[:, 1], 4.0 * variances[:, 0])


# ------------------------------------------------------------------ tests of differences


def test_sharpe_ratio_matches_manual_computation() -> None:
    x = np.random.default_rng(4).normal(0.0005, 0.01, size=(300, 2))
    manual = x.mean(axis=0) / x.std(axis=0, ddof=1) * np.sqrt(252)
    np.testing.assert_allclose(sharpe_ratio(x), manual)
    with_rf = (x.mean(axis=0) - 0.03 / 252) / x.std(axis=0, ddof=1) * np.sqrt(252)
    np.testing.assert_allclose(sharpe_ratio(x, rf=0.03), with_rf)


def test_clear_sharpe_difference_is_detected() -> None:
    rng = np.random.default_rng(5)
    returns = frame(rng.normal(0.002, 0.01, 2000), rng.normal(0.0, 0.01, 2000))
    table = compare_to_references(returns, ["B"], n_boot=1000)
    row = table[(table.strategy == "A") & (table.test == "sharpe_diff")].iloc[0]
    assert row["estimate"] == pytest.approx(
        sharpe_ratio(returns.to_numpy())[0] - sharpe_ratio(returns.to_numpy())[1]
    )
    assert row["estimate"] > 2.0
    assert row["ci_low"] > 0 and row["p_value"] < 0.01 and bool(row["significant"])


def test_clear_variance_ratio_is_detected() -> None:
    rng = np.random.default_rng(6)
    returns = frame(rng.normal(0.0, 0.02, 2000), rng.normal(0.0, 0.01, 2000))
    table = compare_to_references(returns, ["B"], n_boot=1000)
    row = table[(table.strategy == "A") & (table.test == "log_var_ratio")].iloc[0]
    assert row["estimate"] == pytest.approx(np.log(4.0), abs=0.2)
    assert row["ci_low"] > 0 and row["p_value"] < 0.01


def test_table_structure_and_symmetry_between_reference_and_strategy() -> None:
    rng = np.random.default_rng(7)
    returns = frame(*(rng.normal(0.0003, 0.01, 800) for _ in range(3)), names="ABC")
    table = compare_to_references(returns, ["A", "B"], n_boot=500, block_length=5)
    assert len(table) == 2 * 2 * 2  # references x other strategies x tests
    assert set(table.test) == {"sharpe_diff", "log_var_ratio"}
    assert set(table.reference) == {"A", "B"}
    assert not (table.reference == table.strategy).any()
    assert (table.block_length == 5).all() and (table.n_boot == 500).all()

    def row(reference: str, strategy: str, test: str) -> pd.Series:
        sel = table[
            (table.reference == reference) & (table.strategy == strategy) & (table.test == test)
        ]
        return sel.iloc[0]

    for test in ("sharpe_diff", "log_var_ratio"):
        forward, backward = row("A", "B", test), row("B", "A", test)
        assert forward["estimate"] == pytest.approx(-backward["estimate"])
        assert forward["ci_low"] == pytest.approx(-backward["ci_high"])
        assert forward["ci_high"] == pytest.approx(-backward["ci_low"])
        assert forward["p_value"] == pytest.approx(backward["p_value"])


def test_results_are_reproducible_and_only_ci_and_p_change_with_the_seed() -> None:
    rng = np.random.default_rng(8)
    returns = frame(rng.normal(0, 0.01, 600), rng.normal(0, 0.01, 600))
    a = compare_to_references(returns, ["B"], n_boot=400, seed=1)
    b = compare_to_references(returns, ["B"], n_boot=400, seed=1)
    c = compare_to_references(returns, ["B"], n_boot=400, seed=2)
    pd.testing.assert_frame_equal(a, b)
    np.testing.assert_array_equal(a["estimate"], c["estimate"])
    assert not np.array_equal(a["ci_low"], c["ci_low"])


def test_block_length_sensitivity_stacks_results_with_same_estimates() -> None:
    rng = np.random.default_rng(9)
    returns = frame(rng.normal(0, 0.01, 600), rng.normal(0, 0.01, 600))
    table = block_length_sensitivity(returns, ["B"], block_lengths=(5, 10, 20), n_boot=300)
    assert sorted(table.block_length.unique()) == [5, 10, 20]
    sharpe_rows = table[table.test == "sharpe_diff"]
    assert sharpe_rows["estimate"].nunique() == 1


def test_invalid_inputs_raise() -> None:
    rng = np.random.default_rng(10)
    returns = frame(rng.normal(0, 0.01, 100), rng.normal(0, 0.01, 100))
    with pytest.raises(TypeError, match="DataFrame"):
        compare_to_references(returns.to_numpy(), ["B"])
    with pytest.raises(ValueError, match="not found"):
        compare_to_references(returns, ["Z"], n_boot=10)
    with pytest.raises(ValueError, match="alpha"):
        compare_to_references(returns, ["B"], n_boot=10, alpha=1.5)
    bad = returns.copy()
    bad.iloc[3, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        compare_to_references(bad, ["B"], n_boot=10)


# ------------------------------------------------------------------ null calibration


def rejection_rates(make_pair, n_sim: int = 300, alpha: float = 0.10) -> dict[str, float]:
    """Share of simulations in which each test rejects at level ``alpha``.

    With B = 400 resamples the +1 smoothing makes the test reject when the smaller tail holds at
    most 19 draws (4.75%), so the expected rate is about 9.5%; the standard error over 300
    simulations is about 1.7 percentage points.
    """
    rejections = {"sharpe_diff": 0, "log_var_ratio": 0}
    for sim in range(n_sim):
        a, b = make_pair(np.random.default_rng(1000 + sim))
        table = compare_to_references(
            frame(a, b), ["B"], block_length=5, n_boot=400, seed=sim, alpha=alpha
        )
        for _, row in table.iterrows():
            rejections[row["test"]] += int(row["significant"])
    return {test: count / n_sim for test, count in rejections.items()}


def test_null_calibration_series_plus_tiny_noise() -> None:
    """A and B differ only by tiny independent noise: the true differences are exactly zero."""

    def pair(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        x = rng.normal(0.0004, 0.01, 500)
        return x, x + rng.normal(0.0, 1e-4, 500)

    for rate in rejection_rates(pair).values():
        assert 0.05 <= rate <= 0.16


def test_null_calibration_independent_series_with_equal_distribution() -> None:
    def pair(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        return rng.normal(0.0004, 0.01, 500), rng.normal(0.0004, 0.01, 500)

    for rate in rejection_rates(pair).values():
        assert 0.05 <= rate <= 0.16
