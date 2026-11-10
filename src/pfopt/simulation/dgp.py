"""Factor-model data-generating process for the simulation study.

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


def ar1_paths(
    n_steps: int, n_series: int, persistence: float, sd: float, rng: np.random.Generator
) -> np.ndarray:
    """Stationary AR(1) paths ``x_t = phi x_{t-1} + e_t`` with marginal standard deviation ``sd``.

    Parameters
    ----------
    n_steps, n_series : int
        Output shape ``(n_steps, n_series)``.
    persistence : float
        Autocorrelation ``phi`` in ``[0, 1)``.
    sd : float
        Stationary standard deviation (the innovation sd is ``sd * sqrt(1 - phi^2)``).
    rng : numpy.random.Generator
        The first row is drawn from the stationary distribution.
    """
    shocks = rng.standard_normal((n_steps, n_series))
    out = np.empty((n_steps, n_series))
    out[0] = sd * shocks[0]
    innovation = sd * np.sqrt(1.0 - persistence**2)
    for t in range(1, n_steps):
        out[t] = persistence * out[t - 1] + innovation * shocks[t]
    return out


def loading_paths(
    model: FactorModel, n_steps: int, persistence: float, sd: float, rng: np.random.Generator
) -> np.ndarray:
    """Time-varying loadings ``B_t = B + u_t`` with AR(1) drift ``u_t`` on the non-zero loadings.

    Returns
    -------
    numpy.ndarray
        ``n_steps x N x K`` loadings; entries that are zero in ``model.loadings`` stay zero, so
        the block structure is preserved.
    """
    base = model.loadings
    drift = ar1_paths(n_steps, base.size, persistence, sd, rng).reshape(n_steps, *base.shape)
    return base[None, :, :] + drift * (base != 0.0)[None, :, :]


def simulate_time_varying(
    model: FactorModel,
    n_obs: int,
    horizon: int,
    rng: np.random.Generator,
    cfg: SimulationConfig,
    distribution: str = "gaussian",
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate returns with stochastic factor volatility and drifting loadings.

    On day ``t`` the return is ``r_t = B_t (s_t * f_t) + e_t`` with ``f_t`` normal with the model's
    factor variances, independent idiosyncratic noise ``e_t`` (constant variances) and

    - ``s_t = exp(x_t - v)``, ``x_t`` a stationary AR(1) log volatility with standard deviation
      ``sqrt(v)`` per factor (``cfg.vol_log_sd``, ``cfg.vol_persistence``), so ``E[s_t^2] = 1``;
    - ``B_t = B + u_t`` with AR(1) loading drift (``cfg.loading_drift_sd``,
      ``cfg.loading_persistence``) on the non-zero loadings.

    Correlations therefore rise when the market volatility is high and the factor structure moves.
    The Student-t variant multiplies each day by the same common scale as in
    :func:`simulate_returns`. The generator draws, in this order, the log volatilities, the loading
    drift, the factor normals, the idiosyncratic normals and (Student-t only) the chi-square scale.

    Parameters
    ----------
    model : FactorModel
        Base (unconditional) model.
    n_obs : int
        Length ``T`` of the estimation sample.
    horizon : int
        Number of days after the sample over which the target covariance is averaged.
    rng : numpy.random.Generator
        Random source.
    cfg : SimulationConfig
        Time-variation parameters and the Student-t degrees of freedom.
    distribution : {"gaussian", "student_t"}
        Return distribution.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        The ``T x N`` sample and the ``N x N`` forward covariance: the average over the next
        ``horizon`` days of the conditional covariance ``B_t diag(s_t^2 F) B_t' + D``.
    """
    if n_obs < 1 or horizon < 1:
        raise ValueError("n_obs and horizon must be >= 1")
    total = n_obs + horizon
    k, n = model.n_factors, model.n_assets
    variance = cfg.vol_log_sd**2
    log_vol = ar1_paths(total, k, cfg.vol_persistence, cfg.vol_log_sd, rng)
    scale = np.exp(log_vol - variance)
    loadings = loading_paths(model, total, cfg.loading_persistence, cfg.loading_drift_sd, rng)
    factors = rng.standard_normal((total, k)) * np.sqrt(model.factor_var) * scale
    noise = rng.standard_normal((total, n)) * np.sqrt(model.idio_var)
    returns = np.einsum("tnk,tk->tn", loadings, factors) + noise
    if distribution == "student_t":
        if cfg.t_df <= 2.0:
            raise ValueError(f"df must be > 2, got {cfg.t_df}")
        returns = returns * np.sqrt((cfg.t_df - 2.0) / rng.chisquare(cfg.t_df, size=total))[:, None]
    elif distribution != "gaussian":
        raise ValueError(f"unknown distribution {distribution!r}")
    ahead = slice(n_obs, total)
    factor_var = model.factor_var[None, :] * scale[ahead] ** 2
    forward = np.einsum("tnk,tk,tmk->nm", loadings[ahead], factor_var, loadings[ahead]) / horizon
    forward = forward + np.diag(model.idio_var)
    return returns[:n_obs], (forward + forward.T) / 2.0
