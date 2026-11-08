"""Inference upgrade for the headline comparisons (PLAN.md step 8.21).

Builds on the circular block bootstrap of :mod:`pfopt.backtest.stats` and adds

- a **studentized** (bootstrap-t) test for the Sharpe-ratio difference and the log variance ratio,
  following Ledoit and Wolf (2008): the statistic is ``d = g(m_a, m_b)`` of the first and second
  moments ``m = (E r, E r^2)`` of the two series, its delta-method standard error is
  ``sqrt(grad' Psi grad / T)`` with ``Psi`` the long-run covariance of ``(r_a, r_a^2, r_b, r_b^2)``
  estimated from non-overlapping block sums (batch means), and the p-value is the share of
  bootstrap ``|d* - d| / se*`` at least as large as the observed ``|d| / se``. The batch-means
  estimator replaces the prewhitened kernel estimator of the paper; blocks of ``block_length`` are
  used for both the resampling and the standard error, and an incomplete last block is dropped
  from the standard error only. Variances use ``ddof=0`` here (a ``1/T`` difference from the
  5.1 point estimates);
- **multiple-testing adjustment** (Holm 1979 family-wise error, Benjamini-Hochberg 1995 false
  discovery rate) within each family of comparisons;
- the **minimum detectable effect**, ``(z_{1-alpha/2} + z_{power}) * se``: the true difference that
  the test would detect with the given power. A non-significant result for a smaller difference
  is uninformative, not evidence of equality.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from pfopt.backtest.stats import draw_block_starts, expand_block_indices

INFERENCE_TESTS = ("sharpe_diff", "log_var_ratio")
ADJUSTMENTS = ("holm", "bh")


def adjust_pvalues(p_values: Sequence[float], method: str = "holm") -> np.ndarray:
    """Multiple-testing adjusted p-values, in the input order.

    Parameters
    ----------
    p_values : sequence of float
        Raw p-values of one family, each in ``[0, 1]``.
    method : {"holm", "bh"}
        ``"holm"``: step-down family-wise error control, ``max_{j <= i} (m - j + 1) p_(j)``.
        ``"bh"``: Benjamini-Hochberg step-up false discovery rate, ``min_{j >= i} m p_(j) / j``.
        Both are capped at 1.

    Returns
    -------
    numpy.ndarray
        Adjusted p-values.
    """
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or p.size == 0 or np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("p_values must be a non-empty 1-D array of values in [0, 1]")
    if method not in ADJUSTMENTS:
        raise ValueError(f"method must be one of {ADJUSTMENTS}, got {method!r}")
    m = p.size
    order = np.argsort(p, kind="stable")
    ranked = p[order]
    if method == "holm":
        adjusted = np.maximum.accumulate((m - np.arange(m)) * ranked)
    else:
        adjusted = np.minimum.accumulate((m / np.arange(1, m + 1) * ranked)[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def minimum_detectable_effect(se: float | np.ndarray, alpha: float = 0.05, power: float = 0.8):
    """Smallest true effect detected with probability ``power`` by a two-sided level-``alpha`` test.

    ``(z_{1 - alpha / 2} + z_{power}) * se`` (normal approximation).
    """
    if not (0.0 < alpha < 1.0 and 0.0 < power < 1.0):
        raise ValueError("alpha and power must be in (0, 1)")
    return (stats.norm.ppf(1.0 - alpha / 2.0) + stats.norm.ppf(power)) * np.asarray(se)


def _gradient(test: str, mu: np.ndarray, gamma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Value and derivative of the per-series statistic w.r.t. ``(mu, gamma)``.

    ``mu`` and ``gamma`` hold ``E r`` and ``E r^2`` and may have any (matching) shape; returns the
    statistic and the stacked partials ``(d/dmu, d/dgamma)`` along a new last axis.
    """
    var = gamma - mu**2
    if test == "sharpe_diff":
        value = mu / np.sqrt(var)
        grad = np.stack([gamma / var**1.5, -mu / (2.0 * var**1.5)], axis=-1)
    else:
        value = np.log(var)
        grad = np.stack([-2.0 * mu / var, 1.0 / var], axis=-1)
    return value, grad


def _pair_statistic(
    test: str, moments: np.ndarray, blocks: np.ndarray, a: int, b: int, n_series: int, n_obs: int
) -> tuple[np.ndarray, np.ndarray]:
    """Difference ``d`` of the statistic of series ``a`` and ``b`` and its delta-method s.e.

    Parameters
    ----------
    moments : numpy.ndarray
        ``c x 2S`` means of ``(r, r^2)`` of ``c`` samples.
    blocks : numpy.ndarray
        ``c x l x 2S`` centred block sums divided by ``sqrt(block_length)``.
    """
    cols = [a, n_series + a, b, n_series + b]
    mu = moments[:, [a, b]]
    gamma = moments[:, [n_series + a, n_series + b]]
    value, grad = _gradient(test, mu, gamma)
    d = value[:, 0] - value[:, 1]
    g = np.stack([grad[:, 0, 0], grad[:, 0, 1], -grad[:, 1, 0], -grad[:, 1, 1]], axis=1)
    z = blocks[:, :, cols]
    psi = np.einsum("clp,clq->cpq", z, z) / blocks.shape[1]
    var = np.einsum("cp,cpq,cq->c", g, psi, g) / n_obs
    return d, np.sqrt(np.maximum(var, 0.0))


def _block_sums(sample: np.ndarray, block_length: int) -> tuple[np.ndarray, np.ndarray]:
    """Means ``c x 2S`` and scaled centred block sums ``c x l x 2S`` of ``c x T x 2S`` samples."""
    n_obs = sample.shape[1]
    n_blocks = n_obs // block_length
    mean = sample.mean(axis=1)
    used = sample[:, : n_blocks * block_length]
    sums = used.reshape(sample.shape[0], n_blocks, block_length, -1).sum(axis=2)
    centred = (sums - block_length * mean[:, None, :]) / np.sqrt(block_length)
    return mean, centred


def studentized_compare(
    returns: pd.DataFrame,
    references: Sequence[str],
    block_length: int = 10,
    n_boot: int = 5000,
    seed: int = 42,
    annualization: int = 252,
    rf: float = 0.0,
    alpha: float = 0.05,
    power: float = 0.8,
    chunk_size: int = 100,
) -> pd.DataFrame:
    """Studentized block-bootstrap tests of every strategy against each reference.

    Parameters
    ----------
    returns : pandas.DataFrame
        Date x strategy daily returns (finite), at least ``2 * block_length`` rows.
    references : sequence of str
        Benchmark columns.
    block_length, n_boot, seed
        Circular block bootstrap settings (the generator is seeded with
        ``[seed, block_length, 8]`` and the block starts are drawn up front).
    annualization, rf : float
        Trading days per year and annual risk-free rate (Sharpe only).
    alpha, power : float
        Test level and power of the minimum detectable effect.
    chunk_size : int, default 100
        Resamples processed at a time (memory only; does not change the result).

    Returns
    -------
    pandas.DataFrame
        One row per (reference, strategy, test): ``estimate`` (strategy minus reference; Sharpe in
        annual units), ``se``, ``t_stat``, ``ci_low``, ``ci_high`` (symmetric bootstrap-t),
        ``p_value``, ``significant``, ``p_holm``, ``p_bh`` (within each reference and test),
        ``significant_holm``, ``mde`` (minimum detectable effect in the units of ``estimate``) and
        ``mde_vol_pct`` (for the log variance ratio: the detectable change of the volatility ratio
        in percent, ``100 (exp(mde / 2) - 1)``; NaN for the Sharpe test).
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame")
    x = returns.to_numpy(dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("returns must be finite")
    n_obs, n_series = x.shape
    if n_obs < 2 * block_length:
        raise ValueError("need at least two blocks of observations")
    columns = list(returns.columns)
    missing = [r for r in references if r not in columns]
    if missing:
        raise ValueError(f"reference(s) {missing} not found in returns")

    excess = x - rf / annualization
    base = np.concatenate([excess, excess**2], axis=1)
    starts = draw_block_starts(
        n_obs, block_length, n_boot, np.random.default_rng([seed, block_length, 8])
    )
    mean_hat, blocks_hat = _block_sums(base[None], block_length)
    pairs = [(r, s) for r in references for s in columns if s != r]
    scale = {"sharpe_diff": np.sqrt(annualization), "log_var_ratio": 1.0}
    keys = [(r, s, t) for r, s in pairs for t in INFERENCE_TESTS]
    t_star = {key: np.empty(n_boot) for key in keys}
    observed = {}
    for r, s, test in keys:
        a, b = columns.index(s), columns.index(r)
        d, se = _pair_statistic(test, mean_hat, blocks_hat, a, b, n_series, n_obs)
        observed[(r, s, test)] = (float(d[0]), float(se[0]))
    for lo in range(0, n_boot, chunk_size):
        hi = min(lo + chunk_size, n_boot)
        sample = base[expand_block_indices(starts[lo:hi], n_obs, block_length)]
        mean_b, blocks_b = _block_sums(sample, block_length)
        for r, s, test in keys:
            a, b = columns.index(s), columns.index(r)
            d_b, se_b = _pair_statistic(test, mean_b, blocks_b, a, b, n_series, n_obs)
            with np.errstate(divide="ignore", invalid="ignore"):
                t_star[(r, s, test)][lo:hi] = np.where(
                    se_b > 0, (d_b - observed[(r, s, test)][0]) / se_b, 0.0
                )

    rows = []
    for r, s in pairs:
        for test in INFERENCE_TESTS:
            d, se = observed[(r, s, test)]
            draws = np.abs(t_star[(r, s, test)])
            t_stat = d / se if se > 0 else 0.0
            p_value = (1.0 + float(np.sum(draws >= abs(t_stat)))) / (n_boot + 1.0)
            crit = float(np.quantile(draws, 1.0 - alpha))
            k = scale[test]
            mde = float(minimum_detectable_effect(se * k, alpha, power))
            rows.append(
                {
                    "reference": r,
                    "strategy": s,
                    "test": test,
                    "estimate": d * k,
                    "se": se * k,
                    "t_stat": t_stat,
                    "ci_low": (d - crit * se) * k,
                    "ci_high": (d + crit * se) * k,
                    "p_value": min(1.0, p_value),
                    "significant": p_value < alpha,
                    "mde": mde,
                    "mde_vol_pct": 100.0 * (np.exp(mde / 2.0) - 1.0)
                    if test == "log_var_ratio"
                    else float("nan"),
                }
            )
    table = pd.DataFrame(rows)
    for method, name in (("holm", "p_holm"), ("bh", "p_bh")):
        table[name] = table.groupby(["reference", "test"])["p_value"].transform(
            lambda p, m=method: adjust_pvalues(p.to_numpy(), m)
        )
    table["significant_holm"] = table["p_holm"] < alpha
    return table
