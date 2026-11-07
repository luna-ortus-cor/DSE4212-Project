"""Tests for expected-return estimates."""

import numpy as np
import pytest

from pfopt.estimators.mean import expected_returns


def test_hand_computed_example() -> None:
    x = np.array([[0.01], [-0.01], [0.03]])  # mean 0.01, sample variance 0.0004
    np.testing.assert_allclose(expected_returns(x, "log"), [0.01])
    np.testing.assert_allclose(expected_returns(x, "arithmetic"), [0.0102])
    np.testing.assert_allclose(expected_returns(x), [0.0102])


def test_arithmetic_convention_matches_mean_simple_return_for_lognormal_data() -> None:
    rng = np.random.default_rng(0)
    r = rng.normal(0.0005, 0.02, size=(500_000, 3))
    simple_mean = (np.exp(r) - 1.0).mean(axis=0)
    np.testing.assert_allclose(expected_returns(r, "arithmetic"), simple_mean, atol=2e-6)
    assert np.all(expected_returns(r, "log") < simple_mean)  # geometric drag


def test_correction_grows_with_volatility() -> None:
    rng = np.random.default_rng(1)
    r = np.column_stack([rng.normal(0, 0.01, 1000), rng.normal(0, 0.03, 1000)])
    gap = expected_returns(r, "arithmetic") - expected_returns(r, "log")
    assert gap[1] > gap[0] > 0


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError, match="convention"):
        expected_returns(np.ones((5, 2)), "geometric")
    with pytest.raises(ValueError, match="2-D"):
        expected_returns(np.arange(5.0))
    bad = np.ones((5, 2))
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        expected_returns(bad)


def test_input_not_mutated() -> None:
    x = np.random.default_rng(2).normal(size=(50, 4))
    before = x.copy()
    expected_returns(x)
    np.testing.assert_array_equal(x, before)
