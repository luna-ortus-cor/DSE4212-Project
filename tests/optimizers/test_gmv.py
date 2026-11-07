"""Tests for long-only and unconstrained global minimum variance portfolios."""

import logging

import numpy as np
import pytest

from pfopt.estimators.sample import sample_cov
from pfopt.optimizers import gmv as gmv_module
from pfopt.optimizers.gmv import gmv_long_only, gmv_unconstrained
from pfopt.weights import check_weights

BETAS = np.linspace(0.2, 2.0, 12)
# One-factor covariance: the unconstrained GMV shorts the high-beta assets (hand-checked sign),
# so the long-only constraint binds for several of them.
FACTOR_COV = np.outer(BETAS, BETAS) + 0.2 * np.eye(12)
CORNER_COV = np.array([[1.0, 1.5], [1.5, 4.0]])  # unconstrained GMV = [1.25, -0.25]


def assert_kkt(
    sigma: np.ndarray, w: np.ndarray, cap: float | None = None, tol: float = 1e-6
) -> None:
    """KKT conditions of min w'Sigma w with 1'w = 1 and 0 <= w <= cap (Sigma rescaled)."""
    s = sigma / (np.trace(sigma) / sigma.shape[0])
    grad = s @ w
    upper = np.inf if cap is None else cap
    at_zero = w <= 1e-8
    at_cap = w >= upper - 1e-8
    free = ~at_zero & ~at_cap
    assert free.any()
    nu = grad[free].mean()
    np.testing.assert_allclose(grad[free], nu, atol=tol)
    assert (grad[at_zero] >= nu - tol).all()
    assert (grad[at_cap] <= nu + tol).all()


def test_corner_solution_known_answer() -> None:
    np.testing.assert_allclose(gmv_unconstrained(CORNER_COV), [1.25, -0.25], atol=1e-12)
    np.testing.assert_allclose(gmv_long_only(CORNER_COV), [1.0, 0.0], atol=1e-7)


def test_diagonal_covariance_gives_inverse_variance_weights() -> None:
    expected = np.array([4.0, 2.0, 1.0]) / 7.0
    np.testing.assert_allclose(gmv_long_only(np.diag([1.0, 2.0, 4.0])), expected, atol=1e-7)


def test_matches_closed_form_when_unconstrained_solution_is_nonnegative() -> None:
    sd = np.arange(1.0, 6.0)
    corr = 0.9 * np.eye(5) + 0.1
    cov = np.outer(sd, sd) * corr
    closed = gmv_unconstrained(cov)
    assert (closed >= 0).all()
    np.testing.assert_allclose(gmv_long_only(cov), closed, atol=1e-7)


def test_unconstrained_matches_explicit_inverse_formula() -> None:
    inv = np.linalg.inv(FACTOR_COV)
    expected = inv @ np.ones(12) / (np.ones(12) @ inv @ np.ones(12))
    w, info = gmv_unconstrained(FACTOR_COV, return_info=True)
    np.testing.assert_allclose(w, expected, atol=1e-10)
    assert w.sum() == pytest.approx(1.0)
    assert (w < 0).any()
    assert info["ridge"] == 0.0
    assert info["leverage"] == pytest.approx(np.abs(w).sum())
    assert info["leverage"] > 1.0


def test_long_only_satisfies_kkt_and_has_active_constraints() -> None:
    w = gmv_long_only(FACTOR_COV)
    check_weights(w)
    assert (w < 1e-8).sum() >= 2
    assert (w > 1e-8).sum() >= 2
    assert_kkt(FACTOR_COV, w)


def test_capped_solution_satisfies_kkt_and_cap() -> None:
    cap = 0.15
    w = gmv_long_only(FACTOR_COV, max_weight=cap)
    check_weights(w, max_weight=cap)
    assert (w >= cap - 1e-8).any()  # the cap actually binds
    assert_kkt(FACTOR_COV, w, cap=cap)
    uncapped = gmv_long_only(FACTOR_COV)
    assert w @ FACTOR_COV @ w >= uncapped @ FACTOR_COV @ uncapped - 1e-12


def test_cap_equal_to_one_over_n_gives_equal_weights_and_smaller_cap_is_infeasible() -> None:
    np.testing.assert_allclose(gmv_long_only(FACTOR_COV, max_weight=1 / 12), np.full(12, 1 / 12))
    with pytest.raises(ValueError, match="infeasible"):
        gmv_long_only(FACTOR_COV, max_weight=0.08)
    with pytest.raises(ValueError, match="max_weight"):
        gmv_long_only(FACTOR_COV, max_weight=0.0)
    with pytest.raises(ValueError, match="max_weight"):
        gmv_long_only(FACTOR_COV, max_weight=1.5)


def test_solution_beats_random_feasible_portfolios_and_equal_weight() -> None:
    w = gmv_long_only(FACTOR_COV)
    best = w @ FACTOR_COV @ w
    draws = np.random.default_rng(0).dirichlet(np.ones(12), size=500)
    random_values = np.einsum("ij,jk,ik->i", draws, FACTOR_COV, draws)
    assert best <= random_values.min() + 1e-10
    equal = np.full(12, 1 / 12)
    assert best <= equal @ FACTOR_COV @ equal


def test_weights_do_not_depend_on_the_scale_of_sigma() -> None:
    np.testing.assert_allclose(
        gmv_long_only(1e-4 * FACTOR_COV), gmv_long_only(FACTOR_COV), atol=1e-6
    )


def test_info_reports_solver_and_objective() -> None:
    w, info = gmv_long_only(FACTOR_COV, return_info=True)
    assert info["solver"] == "CLARABEL" and info["fallback"] is False
    assert info["objective"] == pytest.approx(w @ FACTOR_COV @ w)


def test_singular_sample_covariance_is_handled() -> None:
    x = np.random.default_rng(1).normal(size=(20, 50))
    cov = sample_cov(x)  # rank 19: tiny negative eigenvalues from round-off are possible
    w = gmv_long_only(cov)
    check_weights(w)
    equal = np.full(50, 1 / 50)
    assert w @ cov @ w <= equal @ cov @ equal


def test_unconstrained_singular_matrix_gets_a_logged_ridge(
    caplog: pytest.LogCaptureFixture,
) -> None:
    cov = sample_cov(np.random.default_rng(2).normal(size=(20, 50)))
    with caplog.at_level(logging.WARNING, logger="pfopt.optimizers.gmv"):
        w, info = gmv_unconstrained(cov, return_info=True)
    assert any("singular" in rec.message for rec in caplog.records)
    assert info["ridge"] == pytest.approx(1e-8 * np.trace(cov) / 50)
    assert w.sum() == pytest.approx(1.0)
    assert np.isfinite(w).all()


def test_falls_back_to_slsqp_and_logs(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    primary = gmv_long_only(FACTOR_COV)

    def broken(*args: object, **kwargs: object) -> np.ndarray:
        raise RuntimeError("boom")

    monkeypatch.setattr(gmv_module, "_solve_cvxpy", broken)
    with caplog.at_level(logging.WARNING, logger="pfopt.optimizers.gmv"):
        w, info = gmv_long_only(FACTOR_COV, return_info=True)
    assert info["solver"] == "SLSQP" and info["fallback"] is True
    assert any("falling back" in rec.message for rec in caplog.records)
    np.testing.assert_allclose(w, primary, atol=1e-5)
    check_weights(w)


def test_raises_when_both_solvers_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*args: object, **kwargs: object) -> np.ndarray:
        raise RuntimeError("boom")

    monkeypatch.setattr(gmv_module, "_solve_cvxpy", broken)
    monkeypatch.setattr(gmv_module, "_solve_slsqp", broken)
    with pytest.raises(RuntimeError, match="boom"):
        gmv_long_only(FACTOR_COV)


def test_single_asset_and_permutation_equivariance() -> None:
    np.testing.assert_array_equal(gmv_long_only(np.array([[2.0]])), [1.0])
    perm = np.random.default_rng(3).permutation(12)
    base = gmv_long_only(FACTOR_COV)
    np.testing.assert_allclose(
        gmv_long_only(FACTOR_COV[np.ix_(perm, perm)]), base[perm], atol=1e-6
    )


def test_invalid_matrices_raise() -> None:
    for func in (gmv_long_only, gmv_unconstrained):
        with pytest.raises(ValueError, match="square"):
            func(np.ones((2, 3)))
        with pytest.raises(ValueError, match="finite"):
            func(np.array([[1.0, np.nan], [np.nan, 1.0]]))
    with pytest.raises(ValueError, match="not PSD"):
        gmv_long_only(np.array([[1.0, 2.0], [2.0, 1.0]]))
    with pytest.raises(ValueError, match="positive trace"):
        gmv_long_only(np.zeros((3, 3)))
