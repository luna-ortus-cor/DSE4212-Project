"""Tests for long-only maximum-Sharpe portfolios and the efficient frontier."""

import logging

import numpy as np
import pytest

from pfopt.optimizers import markowitz as mk
from pfopt.optimizers.gmv import gmv_long_only
from pfopt.optimizers.markowitz import (
    efficient_frontier,
    max_sharpe_long_only,
    min_variance_for_return,
)
from pfopt.weights import check_weights

MU2 = np.array([0.1, 0.2])
SIGMA2 = np.diag([0.04, 0.09])  # uncorrelated: tangency weights are analytic


def random_problem(n: int = 10, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n))
    return rng.normal(0.05, 0.05, size=n), a @ a.T / n + 0.1 * np.eye(n)


def sharpe(w: np.ndarray, mu: np.ndarray, sigma: np.ndarray, rf: float = 0.0) -> float:
    return float((mu @ w - rf) / np.sqrt(w @ sigma @ w))


def test_two_asset_tangency_known_answers() -> None:
    w = max_sharpe_long_only(MU2, SIGMA2)
    np.testing.assert_allclose(w, [2.5 / (2.5 + 2.0 / 0.9), 1 - 2.5 / (2.5 + 2.0 / 0.9)], atol=1e-7)
    assert sharpe(w, MU2, SIGMA2) == pytest.approx(np.sqrt(0.25 + 0.04 / 0.09), abs=1e-7)
    # With rf = 0.05 the excess returns are (0.05, 0.15): w proportional to (1.25, 1.6667).
    w_rf = max_sharpe_long_only(MU2, SIGMA2, rf=0.05)
    np.testing.assert_allclose(w_rf, [3 / 7, 4 / 7], atol=1e-7)


def test_long_only_constraint_binds_at_a_corner() -> None:
    sigma = np.array([[0.04, 0.05], [0.05, 0.09]])  # unconstrained tangency would short asset 1
    np.testing.assert_allclose(max_sharpe_long_only(MU2, sigma), [0.0, 1.0], atol=1e-7)


def test_falls_back_to_gmv_when_no_asset_beats_rf(caplog: pytest.LogCaptureFixture) -> None:
    mu, sigma = np.array([-0.01, -0.02]), np.array([[0.04, 0.01], [0.01, 0.09]])
    with caplog.at_level(logging.WARNING, logger="pfopt.optimizers.markowitz"):
        w, info = max_sharpe_long_only(mu, sigma, return_info=True)
    np.testing.assert_allclose(w, gmv_long_only(sigma), atol=1e-8)
    assert info["fallback_gmv"] is True and info["solver"] == "gmv-fallback"
    assert any("falling back to GMV" in rec.message for rec in caplog.records)
    w_rf = max_sharpe_long_only(np.array([0.01, 0.02]), sigma, rf=0.03)
    np.testing.assert_allclose(w_rf, gmv_long_only(sigma), atol=1e-8)


def test_solution_is_optimal_against_random_portfolios() -> None:
    mu, sigma = random_problem(8, seed=1)
    w = max_sharpe_long_only(mu, sigma)
    check_weights(w)
    draws = np.random.default_rng(0).dirichlet(np.ones(8), size=2000)
    best_random = max(sharpe(d, mu, sigma) for d in draws)
    assert sharpe(w, mu, sigma) >= best_random - 1e-9


@pytest.mark.parametrize("cap", [None, 0.25])
def test_no_feasible_pairwise_shift_improves_the_sharpe_ratio(cap: float | None) -> None:
    mu, sigma = random_problem(8, seed=2)
    w = max_sharpe_long_only(mu, sigma, max_weight=cap)
    upper = 1.0 if cap is None else cap
    check_weights(w, max_weight=cap)
    base = sharpe(w, mu, sigma)
    for i in range(8):
        for j in range(8):
            if i == j:
                continue
            trial = w.copy()
            shift = min(1e-3, trial[i], upper - trial[j])
            trial[i] -= shift
            trial[j] += shift
            assert sharpe(trial, mu, sigma) <= base + 1e-9


def test_cap_cannot_improve_the_sharpe_ratio_and_is_respected() -> None:
    mu, sigma = random_problem(8, seed=3)
    free = max_sharpe_long_only(mu, sigma)
    capped = max_sharpe_long_only(mu, sigma, max_weight=0.2)
    assert capped.max() <= 0.2 + 1e-8
    assert sharpe(capped, mu, sigma) <= sharpe(free, mu, sigma) + 1e-9
    np.testing.assert_allclose(max_sharpe_long_only(mu, sigma, max_weight=1 / 8), np.full(8, 1 / 8))
    with pytest.raises(ValueError, match="infeasible"):
        max_sharpe_long_only(mu, sigma, max_weight=0.1)


def test_weights_invariant_to_positive_rescaling_of_mu_and_sigma() -> None:
    mu, sigma = random_problem(8, seed=4)
    np.testing.assert_allclose(
        max_sharpe_long_only(50.0 * mu, 0.01 * sigma), max_sharpe_long_only(mu, sigma), atol=1e-6
    )


def test_info_reports_solver_and_sharpe() -> None:
    mu, sigma = random_problem(8, seed=5)
    w, info = max_sharpe_long_only(mu, sigma, return_info=True)
    assert info["solver"] == "CLARABEL"
    assert info["fallback_gmv"] is False and info["fallback_slsqp"] is False
    assert info["sharpe"] == pytest.approx(sharpe(w, mu, sigma))


def test_slsqp_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    mu, sigma = random_problem(8, seed=6)
    primary = max_sharpe_long_only(mu, sigma)
    target = float(mu @ gmv_long_only(sigma)) + 0.3 * (mu.max() - float(mu @ gmv_long_only(sigma)))
    primary_mv = min_variance_for_return(mu, sigma, target)

    def broken(*args: object, **kwargs: object) -> np.ndarray:
        raise RuntimeError("boom")

    monkeypatch.setattr(mk, "_cvx_tangency", broken)
    monkeypatch.setattr(mk, "_cvx_min_variance", broken)
    w, info = max_sharpe_long_only(mu, sigma, return_info=True)
    assert info["solver"] == "SLSQP" and info["fallback_slsqp"] is True
    np.testing.assert_allclose(w, primary, atol=1e-3)
    np.testing.assert_allclose(min_variance_for_return(mu, sigma, target), primary_mv, atol=1e-5)


def test_two_asset_frontier_matches_analytic_curve() -> None:
    frontier = efficient_frontier(MU2, SIGMA2, n_points=9)
    r_gmv = 0.1 * (0.09 / 0.13) + 0.2 * (0.04 / 0.13)
    np.testing.assert_allclose(frontier.returns, np.linspace(r_gmv, 0.2, 9), atol=1e-8)
    w2 = (frontier.returns - 0.1) / 0.1
    expected_vol = np.sqrt(0.04 * (1 - w2) ** 2 + 0.09 * w2**2)
    np.testing.assert_allclose(frontier.volatilities, expected_vol, atol=1e-7)
    np.testing.assert_allclose(frontier.weights[:, 1], w2, atol=1e-6)
    np.testing.assert_allclose(frontier.weights[0], [0.09 / 0.13, 0.04 / 0.13], atol=1e-8)
    np.testing.assert_allclose(frontier.weights[-1], [0.0, 1.0], atol=1e-12)


def test_frontier_is_monotone_valid_and_contains_the_max_sharpe_portfolio() -> None:
    mu, sigma = random_problem(10, seed=7)
    frontier = efficient_frontier(mu, sigma, n_points=20)
    assert np.all(np.diff(frontier.returns) > 0)
    assert np.all(np.diff(frontier.volatilities) >= -1e-9)
    for w in frontier.weights:
        check_weights(w)
    np.testing.assert_allclose(frontier.weights[0], gmv_long_only(sigma), atol=1e-8)
    expected_last = np.zeros(10)
    expected_last[np.argmax(mu)] = 1.0
    np.testing.assert_allclose(frontier.weights[-1], expected_last, atol=1e-12)

    w_ms = max_sharpe_long_only(mu, sigma)
    on_frontier = min_variance_for_return(mu, sigma, float(mu @ w_ms))
    assert on_frontier @ sigma @ on_frontier == pytest.approx(w_ms @ sigma @ w_ms, rel=1e-6)
    sharpes = frontier.returns / frontier.volatilities
    assert sharpe(w_ms, mu, sigma) >= sharpes.max() - 1e-9


def test_capped_frontier_respects_cap_and_ends_at_greedy_portfolio() -> None:
    mu, sigma = random_problem(10, seed=8)
    frontier = efficient_frontier(mu, sigma, n_points=12, max_weight=0.3)
    assert frontier.weights.max() <= 0.3 + 1e-8
    top = np.argsort(-mu)[:4]
    expected_last = np.zeros(10)
    expected_last[top] = [0.3, 0.3, 0.3, 0.1]
    np.testing.assert_allclose(frontier.weights[-1], expected_last, atol=1e-12)
    np.testing.assert_allclose(frontier.weights[0], gmv_long_only(sigma, max_weight=0.3), atol=1e-8)


def test_equal_expected_returns_collapse_the_frontier() -> None:
    _, sigma = random_problem(6, seed=9)
    frontier = efficient_frontier(np.full(6, 0.05), sigma, n_points=5)
    np.testing.assert_allclose(frontier.returns, 0.05)
    np.testing.assert_allclose(frontier.weights, np.tile(gmv_long_only(sigma), (5, 1)), atol=1e-12)


def test_target_return_range_is_enforced() -> None:
    mu, sigma = random_problem(6, seed=10)
    r_gmv = float(mu @ gmv_long_only(sigma))
    with pytest.raises(ValueError, match="exceeds"):
        min_variance_for_return(mu, sigma, mu.max() + 0.01)
    with pytest.raises(ValueError, match="below"):
        min_variance_for_return(mu, sigma, r_gmv - 0.01)
    w_top = min_variance_for_return(mu, sigma, mu.max())
    assert w_top[np.argmax(mu)] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="n_points"):
        efficient_frontier(mu, sigma, n_points=1)


def test_invalid_inputs_raise() -> None:
    mu, sigma = random_problem(4, seed=11)
    with pytest.raises(ValueError, match="entries"):
        max_sharpe_long_only(mu[:3], sigma)
    bad = mu.copy()
    bad[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        max_sharpe_long_only(bad, sigma)
    with pytest.raises(ValueError, match="not PSD"):
        max_sharpe_long_only(np.array([0.1, 0.2]), np.array([[1.0, 2.0], [2.0, 1.0]]))
    with pytest.raises(ValueError, match="square"):
        efficient_frontier(mu, np.ones((4, 3)))
