"""Factor-model data-generating process for the simulation study (PLAN.md step 6.1).

Daily returns follow ``r = B f + e`` with zero mean, where ``f`` are ``1 + n_groups`` independent
factors (a market factor loading on every asset and ``n_groups`` group factors, each loading on its
own disjoint block of assets) and ``e`` is independent idiosyncratic noise. The true covariance is

    Sigma = B F B' + D,   F = diag(factor variances),   D = diag(idiosyncratic variances).

Two return distributions share exactly the same ``Sigma``:

- ``gaussian``: ``f`` and ``e`` normal;
- ``student_t``: the Gaussian vector times a common daily scale ``sqrt((df - 2) / chi2_df)``, i.e. a
  multivariate Student-t with ``df`` degrees of freedom rescaled to covariance ``Sigma``
  (Kotz and Nadarajah, 2004). Marginal excess kurtosis is ``6 / (df - 4)`` (6 for ``df = 5``) and
  extreme days hit all assets together.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pfopt.config import SimulationConfig


@dataclass(frozen=True)
class FactorModel:
    """A fixed true factor model (daily units).

    Attributes
    ----------
    loadings : numpy.ndarray
        ``N x K`` loadings; column 0 is the market, columns ``1..n_groups`` the group factors.
    factor_var : numpy.ndarray
        Length-``K`` daily factor variances.
    idio_var : numpy.ndarray
        Length-``N`` daily idiosyncratic variances.
    groups : numpy.ndarray
        Length-``N`` block label of each asset (``-1`` if there are no groups).
    """

    loadings: np.ndarray
    factor_var: np.ndarray
    idio_var: np.ndarray
    groups: np.ndarray

    @property
    def n_assets(self) -> int:
        """Number of assets ``N``."""
        return self.loadings.shape[0]

    @property
    def n_factors(self) -> int:
        """True number of factors ``K``."""
        return self.loadings.shape[1]

    @property
    def sigma(self) -> np.ndarray:
        """True covariance ``B F B' + D`` (symmetric, positive definite)."""
        cov = (self.loadings * self.factor_var) @ self.loadings.T + np.diag(self.idio_var)
        return (cov + cov.T) / 2.0


def make_factor_model(
    cfg: SimulationConfig, annualization: int, rng: np.random.Generator
) -> FactorModel:
    """Draw the true factor model.

    Parameters
    ----------
    cfg : SimulationConfig
        Sizes, factor volatilities and loading / idiosyncratic ranges.
    annualization : int
        Trading days per year (annual variances are divided by it).
    rng : numpy.random.Generator
        Source of the loadings and idiosyncratic volatilities.

    Returns
    -------
    FactorModel
        Group blocks are contiguous and of near-equal size (``numpy.array_split``).
    """
    n, g = cfg.n_assets, cfg.n_groups
    loadings = np.zeros((n, 1 + g))
    loadings[:, 0] = rng.uniform(cfg.loading_low, cfg.loading_high, size=n)
    groups = np.full(n, -1)
    for block, members in enumerate(np.array_split(np.arange(n), g) if g else []):
        groups[members] = block
        loadings[members, 1 + block] = rng.uniform(
            cfg.loading_low, cfg.loading_high, size=len(members)
        )
    vols = np.array(
        [cfg.market_vol] + [cfg.group_vols[k % len(cfg.group_vols)] for k in range(g)], dtype=float
    )
    idio_vol = rng.uniform(cfg.idio_vol_low, cfg.idio_vol_high, size=n)
    return FactorModel(
        loadings=loadings,
        factor_var=vols**2 / annualization,
        idio_var=idio_vol**2 / annualization,
        groups=groups,
    )


def simulate_returns(
    model: FactorModel,
    n_obs: int,
    rng: np.random.Generator,
    distribution: str = "gaussian",
    df: float = 5.0,
) -> np.ndarray:
    """Simulate ``n_obs`` daily return vectors with covariance ``model.sigma``.

    Parameters
    ----------
    model : FactorModel
        True model.
    n_obs : int
        Number of days ``T``.
    rng : numpy.random.Generator
        The normal draws come first, so the same generator state gives the same underlying normals
        for both distributions (the Student-t scale is drawn afterwards).
    distribution : {"gaussian", "student_t"}
        Return distribution.
    df : float, default 5.0
        Degrees of freedom of the Student-t variant (must exceed 2).

    Returns
    -------
    numpy.ndarray
        ``T x N`` returns.
    """
    if n_obs < 1:
        raise ValueError(f"n_obs must be >= 1, got {n_obs}")
    k, n = model.n_factors, model.n_assets
    factors = rng.standard_normal((n_obs, k)) * np.sqrt(model.factor_var)
    noise = rng.standard_normal((n_obs, n)) * np.sqrt(model.idio_var)
    returns = factors @ model.loadings.T + noise
    if distribution == "gaussian":
        return returns
    if distribution == "student_t":
        if df <= 2.0:
            raise ValueError(f"df must be > 2, got {df}")
        scale = np.sqrt((df - 2.0) / rng.chisquare(df, size=n_obs))
        return returns * scale[:, None]
    raise ValueError(f"unknown distribution {distribution!r}")
