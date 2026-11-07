"""Tests for the shared weight validator."""

import numpy as np
import pandas as pd
import pytest

from pfopt.weights import check_weights


def test_valid_weights_pass_for_arrays_and_series() -> None:
    check_weights(np.array([0.5, 0.3, 0.2]))
    check_weights(pd.Series([0.25, 0.75], index=["A", "B"]))
    check_weights(np.array([1.0]))


def test_budget_constraint_enforced_with_tolerance() -> None:
    check_weights(np.array([0.5, 0.5 + 5e-9]))
    with pytest.raises(ValueError, match="sum"):
        check_weights(np.array([0.5, 0.4]))
    check_weights(np.array([0.7, 0.7]), total=1.4)


def test_long_only_and_cap() -> None:
    with pytest.raises(ValueError, match="negative"):
        check_weights(np.array([1.2, -0.2]))
    check_weights(np.array([1.2, -0.2]), long_only=False)
    check_weights(np.array([0.5, 0.5, -1e-12 + 0.0]), tol=1e-8)
    with pytest.raises(ValueError, match="cap"):
        check_weights(np.array([0.6, 0.4]), max_weight=0.5)
    check_weights(np.array([0.5, 0.5]), max_weight=0.5)


def test_non_finite_and_empty_rejected() -> None:
    with pytest.raises(ValueError, match="NaN"):
        check_weights(np.array([np.nan, 1.0]))
    with pytest.raises(ValueError, match="empty"):
        check_weights(np.array([]))
