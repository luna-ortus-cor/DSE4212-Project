"""Statistical significance of performance differences (PLAN.md step 5.1).

Circular block bootstrap (Politis and Romano, 1992; Ledoit and Wolf, 2008 for the Sharpe test):
the daily return series of all strategies are resampled *jointly*, by concatenating blocks of
``block_length`` consecutive days that start at uniformly random positions and wrap around the
end of the sample. This keeps the cross-correlation between strategies and the serial dependence
within a block.

Two statistics are tested for every strategy against a reference strategy:

- Sharpe-ratio difference: ``SR_strategy - SR_reference`` with
  ``SR = mean(r - rf / A) / std(r, ddof=1) * sqrt(A)``;
- log variance ratio: ``log(var_strategy / var_reference)`` (``ddof=1``); a negative value means
  the strategy is less volatile than the reference. This is the cleanest test of the claim that
  denoised, factor and hierarchical portfolios have lower volatility.

Inference is by the percentile method: the confidence interval is the ``alpha / 2`` and
``1 - alpha / 2`` quantiles of the bootstrap distribution of the statistic, and the two-sided
p-value is ``min(1, 2 (1 + min(#{d* <= 0}, #{d* >= 0})) / (B + 1))``, which equals the smallest
significance level at which the percentile interval excludes zero. (The studentized version of
Ledoit and Wolf is not implemented.)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

TEST_NAMES = ("sharpe_diff", "log_var_ratio")


def draw_block_starts(
    n_obs: int, block_length: int, n_boot: int, rng: np.random.Generator
) -> np.ndarray:
    """Random block start positions, shape ``(n_boot, ceil(n_obs / block_length))``."""
    if not 1 <= block_length <= n_obs:
        raise ValueError(f"block_length must be in [1, {n_obs}], got {block_length}")
    if n_boot < 1:
        raise ValueError(f"n_boot must be >= 1, got {n_boot}")
    n_blocks = -(-n_obs // block_length)
    return rng.integers(0, n_obs, size=(n_boot, n_blocks))


def expand_block_indices(starts: np.ndarray, n_obs: int, block_length: int) -> np.ndarray:
    """Time indices ``(n_boot, n_obs)`` of the circular block bootstrap samples.

    Each block covers ``start, start + 1, ...`` modulo ``n_obs``; the concatenation is cut to
    ``n_obs`` observations.
    """
    offsets = np.arange(block_length)
    indices = (starts[:, :, None] + offsets) % n_obs
    return indices.reshape(starts.shape[0], -1)[:, :n_obs]


def bootstrap_moments(
    returns: np.ndarray,
    block_length: int,
    n_boot: int,
    seed: int,
    chunk_size: int = 250,
) -> tuple[np.ndarray, np.ndarray]:
    """Means and sample variances (``ddof=1``) of circular-block-bootstrap resamples.

    All columns of ``returns`` are resampled with the same time indices. The random block starts
    are drawn up front, so the result does not depend on ``chunk_size`` (which only bounds memory).

    Parameters
    ----------
    returns : numpy.ndarray
        ``T x S`` array of daily returns.
    block_length : int
        Block length in days.
    n_boot : int
        Number of bootstrap replicates ``B``.
    seed : int
        Seed; the generator is seeded with ``[seed, block_length]`` so each block length has its
        own reproducible stream.
    chunk_size : int, default 250
        Replicates processed at a time.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        Bootstrap means and variances, each ``B x S``.
    """
    x = np.asarray(returns, dtype=float)
    n_obs, n_series = x.shape
    rng = np.random.default_rng([seed, block_length])
    starts = draw_block_starts(n_obs, block_length, n_boot, rng)
    means = np.empty((n_boot, n_series))
    variances = np.empty((n_boot, n_series))
    for lo in range(0, n_boot, chunk_size):
        hi = min(lo + chunk_size, n_boot)
        sample = x[expand_block_indices(starts[lo:hi], n_obs, block_length)]
        means[lo:hi] = sample.mean(axis=1)
        variances[lo:hi] = sample.var(axis=1, ddof=1)
    return means, variances


def sharpe_ratio(returns: np.ndarray, annualization: int = 252, rf: float = 0.0) -> np.ndarray:
    """Annualized Sharpe ratio of each column of a ``T x S`` array (``ddof=1``)."""
    x = np.asarray(returns, dtype=float)
    return (x.mean(axis=0) - rf / annualization) / x.std(axis=0, ddof=1) * np.sqrt(annualization)


def _percentile_summary(draws: np.ndarray, alpha: float) -> tuple[float, float, float]:
    """Lower and upper percentile bounds and the two-sided p-value for the null value 0."""
    low, high = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    n = draws.size
    tail = min(int(np.sum(draws <= 0.0)), int(np.sum(draws >= 0.0)))
    return float(low), float(high), min(1.0, 2.0 * (1.0 + tail) / (n + 1.0))


def compare_to_references(
    returns: pd.DataFrame,
    references: Sequence[str],
    block_length: int = 10,
    n_boot: int = 5000,
    seed: int = 42,
    annualization: int = 252,
    rf: float = 0.0,
    alpha: float = 0.05,
    chunk_size: int = 250,
) -> pd.DataFrame:
    """Bootstrap tests of every strategy against each reference strategy.

    Parameters
    ----------
    returns : pandas.DataFrame
        Date x strategy daily returns (finite).
    references : sequence of str
        Column names used as the benchmark in the comparisons.
    block_length, n_boot, seed, chunk_size
        See :func:`bootstrap_moments`.
    annualization : int, default 252
        Trading days per year.
    rf : float, default 0.0
        Annual risk-free rate for the Sharpe ratio.
    alpha : float, default 0.05
        Significance level of the confidence intervals and the ``significant`` flag.

    Returns
    -------
    pandas.DataFrame
        One row per (reference, strategy, test) with columns ``reference``, ``strategy``,
        ``test`` (``sharpe_diff`` or ``log_var_ratio``), ``estimate`` (strategy minus reference),
        ``ci_low``, ``ci_high``, ``p_value``, ``significant`` (p-value below ``alpha``),
        ``block_length`` and ``n_boot``.
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame")
    if not np.isfinite(returns.to_numpy(dtype=float)).all():
        raise ValueError("returns must be finite")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    columns = list(returns.columns)
    missing = [r for r in references if r not in columns]
    if missing:
        raise ValueError(f"reference(s) {missing} not found in returns")
    x = returns.to_numpy(dtype=float)
    daily_rf = rf / annualization
    sharpe_hat = sharpe_ratio(x, annualization, rf)
    log_var_hat = np.log(x.var(axis=0, ddof=1))
    means, variances = bootstrap_moments(x, block_length, n_boot, seed, chunk_size)
    sharpe_star = (means - daily_rf) / np.sqrt(variances) * np.sqrt(annualization)
    log_var_star = np.log(variances)

    rows = []
    for reference in references:
        j = columns.index(reference)
        for k, strategy in enumerate(columns):
            if k == j:
                continue
            for test, hat, star in (
                ("sharpe_diff", sharpe_hat, sharpe_star),
                ("log_var_ratio", log_var_hat, log_var_star),
            ):
                low, high, p_value = _percentile_summary(star[:, k] - star[:, j], alpha)
                rows.append(
                    {
                        "reference": reference,
                        "strategy": strategy,
                        "test": test,
                        "estimate": float(hat[k] - hat[j]),
                        "ci_low": low,
                        "ci_high": high,
                        "p_value": p_value,
                        "significant": p_value < alpha,
                        "block_length": block_length,
                        "n_boot": n_boot,
                    }
                )
    return pd.DataFrame(rows)


def block_length_sensitivity(
    returns: pd.DataFrame,
    references: Sequence[str],
    block_lengths: Sequence[int] = (5, 10, 20),
    **kwargs: object,
) -> pd.DataFrame:
    """Run :func:`compare_to_references` for several block lengths and stack the results."""
    return pd.concat(
        [
            compare_to_references(returns, references, block_length=b, **kwargs)
            for b in block_lengths
        ],
        ignore_index=True,
    )


def log_var_double_difference(
    returns: pd.DataFrame,
    quadruples: Mapping[str, tuple[str, str, str, str]],
    block_length: int = 10,
    n_boot: int = 5000,
    seed: int = 42,
    alpha: float = 0.05,
    chunk_size: int = 250,
) -> pd.DataFrame:
    """Bootstrap test of a difference in log variance ratios (the H3 statistic, step 5.5).

    For each named quadruple ``(a, a_ref, b, b_ref)`` the statistic is
    ``[log var(a) - log var(a_ref)] - [log var(b) - log var(b_ref)]``. With ``a`` an
    unconstrained estimator-based GMV, ``a_ref`` unconstrained sample GMV, and ``b``, ``b_ref``
    their long-only counterparts, a negative value means the estimator's volatility advantage
    over the sample covariance is larger without the long-only constraint (H3). The same
    resampled days are used for all series (joint circular block bootstrap, percentile interval
    and p-value).

    Parameters
    ----------
    returns : pandas.DataFrame
        Date x strategy daily returns (finite).
    quadruples : mapping of str to tuple of 4 str
        Name to ``(a, a_ref, b, b_ref)`` column names.
    block_length, n_boot, seed, chunk_size
        See :func:`bootstrap_moments`.
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    pandas.DataFrame
        One row per name with ``gap_a`` (``log var(a) - log var(a_ref)``), ``gap_b``, ``estimate``
        (their difference), ``ci_low``, ``ci_high``, ``p_value`` and ``significant``.
    """
    if not np.isfinite(returns.to_numpy(dtype=float)).all():
        raise ValueError("returns must be finite")
    columns = list(returns.columns)
    needed = {c for quad in quadruples.values() for c in quad}
    missing = sorted(needed - set(columns))
    if missing:
        raise ValueError(f"column(s) {missing} not found in returns")
    x = returns.to_numpy(dtype=float)
    log_var_hat = np.log(x.var(axis=0, ddof=1))
    _, variances = bootstrap_moments(x, block_length, n_boot, seed, chunk_size)
    log_var_star = np.log(variances)
    rows = []
    for name, (a, a_ref, b, b_ref) in quadruples.items():
        i, j, k, m = (columns.index(c) for c in (a, a_ref, b, b_ref))
        gap_a, gap_b = log_var_hat[i] - log_var_hat[j], log_var_hat[k] - log_var_hat[m]
        star_a = log_var_star[:, i] - log_var_star[:, j]
        draws = star_a - (log_var_star[:, k] - log_var_star[:, m])
        low, high, p_value = _percentile_summary(draws, alpha)
        rows.append(
            {
                "name": name,
                "gap_a": float(gap_a),
                "gap_b": float(gap_b),
                "estimate": float(gap_a - gap_b),
                "ci_low": low,
                "ci_high": high,
                "p_value": p_value,
                "significant": p_value < alpha,
            }
        )
    return pd.DataFrame(rows)
