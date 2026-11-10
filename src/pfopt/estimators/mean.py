"""Expected-return estimates for Markowitz."""

from __future__ import annotations

import numpy as np

from pfopt.estimators.utils import validate_returns

MU_CONVENTIONS = ("arithmetic", "log")


def expected_returns(returns: np.ndarray, convention: str = "arithmetic") -> np.ndarray:
    """Per-period expected return of each asset, estimated from a window of log returns.

    Mean-variance optimization needs the expected *simple* return, because a portfolio's simple
    return is the weighted sum of the assets' simple returns. If the log return ``r`` is
    approximately normal with mean ``m`` and variance ``s2``, then ``E[exp(r) - 1] = exp(m + s2 / 2)
    - 1``, which is approximately ``m + s2 / 2``. The mean of log returns alone is the geometric
    growth rate and understates the expected simple return by about ``s2 / 2`` per period.

    Parameters
    ----------
    returns : numpy.ndarray
        ``T x N`` array of log returns without NaN.
    convention : {"arithmetic", "log"}, default "arithmetic"
        ``"arithmetic"``: ``mean(r) + var(r, ddof=1) / 2``. ``"log"``: ``mean(r)``.

    Returns
    -------
    numpy.ndarray
        Length-``N`` vector of per-period expected returns.
    """
    if convention not in MU_CONVENTIONS:
        raise ValueError(f"convention must be one of {MU_CONVENTIONS}, got {convention!r}")
    x = validate_returns(returns)
    mean = x.mean(axis=0)
    if convention == "log":
        return mean
    return mean + 0.5 * x.var(axis=0, ddof=1)
