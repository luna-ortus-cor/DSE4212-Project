"""Typed, validated project configuration loaded from ``config.yaml``.

All tunable parameters of the project live in ``config.yaml`` (fixed *a priori*, see
``CLAUDE.md`` section 6). This module turns that file into immutable dataclasses, validates
them, and provides a stable hash used as a cache key for expensive results.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, get_origin, get_type_hints

import yaml

logger = logging.getLogger(__name__)

HRP_LINKAGES = ("single", "ward", "average")
PCA_K_RULES = ("mp", "fixed", "baing")
SIM_DISTRIBUTIONS = ("gaussian", "student_t")


@dataclass(frozen=True)
class RunConfig:
    """Global run settings.

    Attributes
    ----------
    seed : int
        Global random seed.
    annualization : int
        Trading days per year used to annualize returns and volatilities.
    rf : float
        Annual risk-free rate (0 by design; stated as a simplification in the report).
    """

    seed: int = 42
    annualization: int = 252
    rf: float = 0.0

    def __post_init__(self) -> None:
        if self.annualization <= 0:
            raise ValueError(f"annualization must be > 0, got {self.annualization}")


@dataclass(frozen=True)
class PathsConfig:
    """Directories, relative to the repository root."""

    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    cache_dir: str = "results/cache"
    tables_dir: str = "results/tables"
    figures_dir: str = "results/figures"


@dataclass(frozen=True)
class DataConfig:
    """Cleaning parameters (PLAN.md step 1.2).

    Attributes
    ----------
    start_date, end_date : str or None
        Quoted ISO dates (``"YYYY-MM-DD"``) restricting the sample. Taken from the step 1.1 audit
        of the supplied file: first and last date with a valid price.
    ffill_limit : int
        Maximum number of consecutive missing price days that are forward-filled.
    max_missing_frac : float
        Symbols with a larger missing share (over their own listed span, after filling) are dropped.
    suspect_abs_logret : float
        Absolute log return above which a return is flagged as suspect.
    winsor_lower, winsor_upper : float
        Per-column quantiles for winsorization inside each estimation window.
    stocks_file, companies_file, index_file : str
        Raw CSV filenames inside ``paths.raw_dir``.
    stocks_columns, companies_columns, index_columns : tuple of str
        Columns read from each CSV. A column missing from a file raises an error in the loader.
    """

    start_date: str | None = "2010-01-04"
    end_date: str | None = "2024-12-20"
    ffill_limit: int = 3
    max_missing_frac: float = 0.05
    suspect_abs_logret: float = 0.40
    winsor_lower: float = 0.001
    winsor_upper: float = 0.999
    stocks_file: str = "sp500_stocks.csv"
    companies_file: str = "sp500_companies.csv"
    index_file: str = "sp500_index.csv"
    stocks_columns: tuple[str, ...] = (
        "Date",
        "Symbol",
        "Adj Close",
        "Close",
        "High",
        "Low",
        "Open",
        "Volume",
    )
    companies_columns: tuple[str, ...] = ("Symbol", "Sector", "Industry", "Marketcap", "Weight")
    index_columns: tuple[str, ...] = ("Date", "S&P500")

    def __post_init__(self) -> None:
        for name in ("stocks_file", "companies_file", "index_file"):
            if not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty filename")
        for name, required in (
            ("stocks_columns", ("Date", "Symbol", "Adj Close", "Close", "Volume")),
            ("companies_columns", ("Symbol",)),
            ("index_columns", ("Date",)),
        ):
            missing = [c for c in required if c not in getattr(self, name)]
            if missing:
                raise ValueError(f"{name} must include {missing}")
        if len(self.index_columns) != 2:
            raise ValueError("index_columns must be exactly (Date, <index level column>)")
        for name in ("start_date", "end_date"):
            value = getattr(self, name)
            if value is not None:
                if not isinstance(value, str):
                    raise ValueError(f"{name} must be a quoted ISO string, got {value!r}")
                try:
                    date.fromisoformat(value)
                except ValueError as exc:
                    raise ValueError(f"{name} must be an ISO date, got {value!r}") from exc
        if self.start_date and self.end_date and str(self.start_date) > str(self.end_date):
            raise ValueError("start_date must not be after end_date")
        if self.ffill_limit < 0:
            raise ValueError(f"ffill_limit must be >= 0, got {self.ffill_limit}")
        if not 0.0 <= self.max_missing_frac < 1.0:
            raise ValueError(f"max_missing_frac must be in [0, 1), got {self.max_missing_frac}")
        if self.suspect_abs_logret <= 0:
            raise ValueError(f"suspect_abs_logret must be > 0, got {self.suspect_abs_logret}")
        if not 0.0 <= self.winsor_lower < self.winsor_upper <= 1.0:
            raise ValueError("winsor quantiles must satisfy 0 <= lower < upper <= 1")


@dataclass(frozen=True)
class UniverseConfig:
    """Universe and estimation window (PLAN.md step 1.3).

    Attributes
    ----------
    n_assets : int
        Universe size ``N``.
    lookback_days : int
        Estimation window ``T`` in trading days.
    dollar_volume_window : int
        Trailing window (days) of average dollar volume used to rank eligible names.
    """

    n_assets: int = 100
    lookback_days: int = 504
    dollar_volume_window: int = 252

    def __post_init__(self) -> None:
        for name in ("n_assets", "lookback_days", "dollar_volume_window"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)}")


@dataclass(frozen=True)
class BacktestConfig:
    """Backtest settings (PLAN.md Phase 4).

    Attributes
    ----------
    rebalance_days : int
        Trading days between rebalances.
    cost_bps : float
        Proportional cost in basis points per unit of one-way turnover ``sum |dw|``.
    max_weight : float or None
        Optional per-asset weight cap (``None`` means no cap).
    mu_convention : {"arithmetic", "log"}
        Expected-return estimate used by Markowitz (Track B). ``"arithmetic"`` is the window mean
        of log returns plus half the sample variance (the expected simple return if returns are
        lognormal); ``"log"`` is the plain mean of log returns.
    """

    rebalance_days: int = 21
    cost_bps: float = 10.0
    max_weight: float | None = None
    mu_convention: str = "arithmetic"

    def __post_init__(self) -> None:
        if self.mu_convention not in ("arithmetic", "log"):
            raise ValueError(f"mu_convention must be arithmetic or log, got {self.mu_convention!r}")
        if self.rebalance_days <= 0:
            raise ValueError(f"rebalance_days must be > 0, got {self.rebalance_days}")
        if self.cost_bps < 0:
            raise ValueError(f"cost_bps must be >= 0, got {self.cost_bps}")
        if self.max_weight is not None and not 0.0 < self.max_weight <= 1.0:
            raise ValueError(f"max_weight must be None or in (0, 1], got {self.max_weight}")


@dataclass(frozen=True)
class PCAConfig:
    """PCA-factor covariance settings (PLAN.md step 2.3).

    Attributes
    ----------
    k_rule : str
        ``"mp"`` (eigenvalues above the Marchenko-Pastur edge), ``"fixed"`` or ``"baing"``.
    k_fixed : int or None
        Number of factors when ``k_rule == "fixed"``.
    baing_kmax : int
        Maximum number of factors searched by the Bai-Ng criterion.
    """

    k_rule: str = "mp"
    k_fixed: int | None = None
    baing_kmax: int = 10

    def __post_init__(self) -> None:
        if self.k_rule not in PCA_K_RULES:
            raise ValueError(f"k_rule must be one of {PCA_K_RULES}, got {self.k_rule!r}")
        if self.k_rule == "fixed" and (self.k_fixed is None or self.k_fixed < 1):
            raise ValueError("k_fixed must be an int >= 1 when k_rule == 'fixed'")
        if self.baing_kmax < 1:
            raise ValueError(f"baing_kmax must be >= 1, got {self.baing_kmax}")


@dataclass(frozen=True)
class HRPConfig:
    """Hierarchical Risk Parity settings (PLAN.md step 2.4)."""

    linkage: str = "single"

    def __post_init__(self) -> None:
        if self.linkage not in HRP_LINKAGES:
            raise ValueError(f"linkage must be one of {HRP_LINKAGES}, got {self.linkage!r}")


@dataclass(frozen=True)
class SensitivityConfig:
    """Sensitivity grids (PLAN.md step 5.3). Reported as sensitivity only, never used to tune."""

    n_assets: tuple[int, ...] = (50, 100, 200)
    lookback_days: tuple[int, ...] = (252, 504, 756)
    cost_bps: tuple[float, ...] = (0.0, 5.0, 10.0, 20.0)
    max_weight: tuple[float, ...] = (0.10,)
    pca_k_fixed: tuple[int, ...] = (1, 3, 5, 10)
    hrp_linkage: tuple[str, ...] = HRP_LINKAGES

    def __post_init__(self) -> None:
        if any(n <= 0 for n in self.n_assets) or any(t <= 0 for t in self.lookback_days):
            raise ValueError("sensitivity grids for n_assets / lookback_days must be positive")
        if any(c < 0 for c in self.cost_bps):
            raise ValueError("sensitivity cost_bps values must be >= 0")
        if any(not 0.0 < w <= 1.0 for w in self.max_weight):
            raise ValueError("sensitivity max_weight values must be in (0, 1]")
        if any(k < 1 for k in self.pca_k_fixed):
            raise ValueError("sensitivity pca_k_fixed values must be >= 1")
        bad = [m for m in self.hrp_linkage if m not in HRP_LINKAGES]
        if bad:
            raise ValueError(f"unknown HRP linkage(s) in sensitivity grid: {bad}")


@dataclass(frozen=True)
class EdaConfig:
    """Exploratory data analysis settings (PLAN.md step 1.4).

    Attributes
    ----------
    stress_date : str
        Quoted ISO date of the stress-period snapshot (inside the 2020 COVID stress period).
        The other two snapshots are the first feasible and the last date, derived from the data.
    corr_window_days : int
        Window of the rolling cross-sectional average correlation.
    index_vol_window_days : int
        Window of the index realized volatility overlay.
    highlight_years : tuple of int
        Calendar years shaded in the rolling-correlation figure.
    frontier_horizon_days : int
        Number of trading days after the as-of date over which the realized risk and return of
        efficient-frontier portfolios are measured (step 3.2).
    frontier_points : int
        Number of portfolios on each efficient frontier.
    """

    stress_date: str = "2020-03-31"
    corr_window_days: int = 21
    index_vol_window_days: int = 21
    highlight_years: tuple[int, ...] = (2020, 2022)
    frontier_horizon_days: int = 252
    frontier_points: int = 50

    def __post_init__(self) -> None:
        if not isinstance(self.stress_date, str):
            raise ValueError(f"stress_date must be a quoted ISO string, got {self.stress_date!r}")
        try:
            date.fromisoformat(self.stress_date)
        except ValueError as exc:
            raise ValueError(f"stress_date must be an ISO date, got {self.stress_date!r}") from exc
        if self.corr_window_days < 2 or self.index_vol_window_days < 2:
            raise ValueError("corr_window_days and index_vol_window_days must be >= 2")
        if self.frontier_horizon_days < 2 or self.frontier_points < 2:
            raise ValueError("frontier_horizon_days and frontier_points must be >= 2")


@dataclass(frozen=True)
class StatsConfig:
    """Bootstrap significance tests (PLAN.md step 5.1).

    Attributes
    ----------
    block_length : int
        Block length (days) of the circular block bootstrap in the main table.
    n_boot : int
        Number of bootstrap replicates.
    sensitivity_block_lengths : tuple of int
        Block lengths of the sensitivity table.
    references : tuple of str
        Benchmark strategies every other strategy is compared with.
    alpha : float
        Significance level of the confidence intervals and the significance flag.
    mde_power : float
        Power used for the minimum detectable effect of the inference upgrade (step 8.21).
    """

    block_length: int = 10
    n_boot: int = 5000
    sensitivity_block_lengths: tuple[int, ...] = (5, 10, 20)
    references: tuple[str, ...] = ("GMV-SAMPLE", "EW", "GMV-LW")
    alpha: float = 0.05
    mde_power: float = 0.8

    def __post_init__(self) -> None:
        if self.block_length < 1 or any(b < 1 for b in self.sensitivity_block_lengths):
            raise ValueError("block lengths must be >= 1")
        if self.n_boot < 1:
            raise ValueError(f"n_boot must be >= 1, got {self.n_boot}")
        if not self.references:
            raise ValueError("references must not be empty")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}")
        if not 0.0 < self.mde_power < 1.0:
            raise ValueError(f"mde_power must be in (0, 1), got {self.mde_power}")


@dataclass(frozen=True)
class RegimeConfig:
    """Subperiod and volatility-regime analysis (PLAN.md step 5.2).

    Attributes
    ----------
    covid_start, covid_end : str
        Quoted ISO dates of the COVID stress window (inclusive).
    stress_year : int
        Calendar year analysed as a second stress period (2022). It must not overlap the COVID
        window.
    vol_window_days : int
        Window of the S&P 500 trailing realized volatility that defines the regimes.
    min_history_days : int
        Minimum number of volatility observations before the expanding median is used; earlier
        days are left unclassified.
    """

    covid_start: str = "2020-02-19"
    covid_end: str = "2020-12-31"
    stress_year: int = 2022
    vol_window_days: int = 63
    min_history_days: int = 252

    def __post_init__(self) -> None:
        for name in ("covid_start", "covid_end"):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise ValueError(f"{name} must be a quoted ISO string, got {value!r}")
            try:
                date.fromisoformat(value)
            except ValueError as exc:
                raise ValueError(f"{name} must be an ISO date, got {value!r}") from exc
        if self.covid_start > self.covid_end:
            raise ValueError("covid_start must not be after covid_end")
        year_start, year_end = f"{self.stress_year}-01-01", f"{self.stress_year}-12-31"
        if self.covid_start <= year_end and self.covid_end >= year_start:
            raise ValueError("the COVID window and the stress year must not overlap")
        if self.vol_window_days < 2 or self.min_history_days < 1:
            raise ValueError("vol_window_days must be >= 2 and min_history_days >= 1")


@dataclass(frozen=True)
class SimulationConfig:
    """Factor-model simulation study (PLAN.md step 6.1).

    The true covariance is ``B F B' + D`` with one market factor (all assets), ``n_groups`` group
    factors (each loading on its own disjoint block of assets, blocks of near-equal size) and
    idiosyncratic noise. The true model is drawn once and shared by all grid cells and both
    distributions, so replications differ only through sampling.

    Attributes
    ----------
    n_assets : int
        Number of simulated assets ``N``.
    n_groups : int
        Number of group factors / disjoint asset blocks (``K_true = 1 + n_groups``).
    sample_sizes : tuple of int
        Sample lengths ``T`` of the grid.
    n_reps : int
        Replications per grid cell.
    distributions : tuple of str
        Subset of ``("gaussian", "student_t")``; the first one is the main result.
    t_df : float
        Degrees of freedom of the multivariate Student-t variant (must exceed 4 so that the
        kurtosis exists); the covariance is scaled to equal the Gaussian one.
    market_vol : float
        Annual volatility of the market factor.
    group_vols : tuple of float
        Annual volatilities of the group factors, cycled if ``n_groups`` exceeds its length.
    loading_low, loading_high : float
        Bounds of the uniform distribution of the factor loadings.
    idio_vol_low, idio_vol_high : float
        Bounds of the uniform distribution of annual idiosyncratic volatilities.
    alpha : float
        Significance level of the confidence bands (normal approximation across replications).
    time_varying : bool
        If true (step 6.3), the covariance changes over time: every factor has its own
        stochastic volatility and the loadings drift. The target of every estimator is then the
        average conditional covariance over the ``horizon_days`` days after the sample.
    horizon_days : int
        Length of the forward evaluation period in the time-varying design.
    vol_log_sd : float
        Stationary standard deviation of the log volatility of each factor (mean-one variance
        scale, so the unconditional factor variance is unchanged).
    vol_persistence : float
        Daily autocorrelation of the log factor volatilities.
    loading_drift_sd : float
        Stationary standard deviation of the loading drift (only non-zero loadings drift).
    loading_persistence : float
        Daily autocorrelation of the loading drift.
    """

    n_assets: int = 100
    n_groups: int = 2
    sample_sizes: tuple[int, ...] = (126, 252, 504, 1000)
    n_reps: int = 200
    distributions: tuple[str, ...] = ("gaussian", "student_t")
    t_df: float = 5.0
    market_vol: float = 0.15
    group_vols: tuple[float, ...] = (0.08, 0.06)
    loading_low: float = 0.5
    loading_high: float = 1.5
    idio_vol_low: float = 0.15
    idio_vol_high: float = 0.45
    alpha: float = 0.05
    time_varying: bool = False
    horizon_days: int = 21
    vol_log_sd: float = 0.4
    vol_persistence: float = 0.98
    loading_drift_sd: float = 0.2
    loading_persistence: float = 0.996

    def __post_init__(self) -> None:
        if self.n_groups < 0 or self.n_assets < 2 * max(self.n_groups, 1):
            raise ValueError("need n_groups >= 0 and at least two assets per group")
        if not self.sample_sizes or any(t <= self.n_assets for t in self.sample_sizes):
            raise ValueError("every sample size must exceed n_assets (sample covariance full rank)")
        if self.n_reps < 2:
            raise ValueError(f"n_reps must be >= 2, got {self.n_reps}")
        if not self.distributions or any(d not in SIM_DISTRIBUTIONS for d in self.distributions):
            raise ValueError(f"distributions must be a non-empty subset of {SIM_DISTRIBUTIONS}")
        if self.t_df <= 4.0:
            raise ValueError(f"t_df must be > 4, got {self.t_df}")
        if self.market_vol <= 0 or not self.group_vols or any(v <= 0 for v in self.group_vols):
            raise ValueError("factor volatilities must be positive")
        if not 0.0 < self.loading_low <= self.loading_high:
            raise ValueError("need 0 < loading_low <= loading_high")
        if not 0.0 < self.idio_vol_low <= self.idio_vol_high:
            raise ValueError("need 0 < idio_vol_low <= idio_vol_high")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}")
        if self.horizon_days < 1:
            raise ValueError(f"horizon_days must be >= 1, got {self.horizon_days}")
        if self.vol_log_sd < 0 or self.loading_drift_sd < 0:
            raise ValueError("vol_log_sd and loading_drift_sd must be >= 0")
        if not (0.0 <= self.vol_persistence < 1.0 and 0.0 <= self.loading_persistence < 1.0):
            raise ValueError("persistence parameters must be in [0, 1)")


@dataclass(frozen=True)
class PipelineCheckConfig:
    """Synthetic-panel validation of the backtest pipeline (PLAN.md step 8.20).

    Attributes
    ----------
    n_symbols : int
        Symbols in each synthetic panel (more than ``universe.n_assets``, so the point-in-time
        universe has to choose).
    n_days : int
        Trading days of each panel.
    n_panels : int
        Independent panels (each with its own true model and returns).
    dv_persistence, dv_sd : float
        Daily autocorrelation and stationary standard deviation of every symbol's log dollar
        volume (an AR(1)), which makes the top-N membership change over time.
    z2_tolerance : float
        The accounting check passes if the mean standardized squared return is within this many
        standard errors of 1.
    identity_tolerance : float
        The engine/simulation-study identity check passes if the true-variance ratios differ by
        less than this.
    """

    n_symbols: int = 150
    n_days: int = 2500
    n_panels: int = 4
    dv_persistence: float = 0.995
    dv_sd: float = 1.0
    z2_tolerance: float = 3.0
    identity_tolerance: float = 1e-6

    def __post_init__(self) -> None:
        if self.n_symbols < 4 or self.n_days < 10 or self.n_panels < 2:
            raise ValueError("need n_symbols >= 4, n_days >= 10 and n_panels >= 2")
        if not 0.0 <= self.dv_persistence < 1.0 or self.dv_sd < 0:
            raise ValueError("need dv_persistence in [0, 1) and dv_sd >= 0")
        if self.z2_tolerance <= 0 or self.identity_tolerance <= 0:
            raise ValueError("tolerances must be positive")


@dataclass(frozen=True)
class NonStationarityConfig:
    """Non-stationarity test: EWMA strategies and simulated positive control (PLAN.md step 8.22).

    Attributes
    ----------
    halflives : tuple of int
        EWMA half-lives in days (fixed a priori; each gives ``GMV-EWMA<h>`` and
        ``GMV-EWMA<h>-RMT``).
    calibration_block_days : int
        Length of the non-overlapping blocks whose log realized variance of the equal-weight market
        return is used to calibrate the volatility dynamics of the time-varying simulation.
    sim_sample_sizes : tuple of int
        Sample sizes of the simulated positive control (must exceed ``simulation.n_assets``).
    sim_n_reps : int
        Replications per sample size and design in the positive control.
    """

    halflives: tuple[int, ...] = (126, 63)
    calibration_block_days: int = 21
    sim_sample_sizes: tuple[int, ...] = (252, 504)
    sim_n_reps: int = 100

    def __post_init__(self) -> None:
        if not self.halflives or any(h < 1 for h in self.halflives):
            raise ValueError("halflives must be non-empty positive integers")
        if self.calibration_block_days < 5:
            raise ValueError("calibration_block_days must be >= 5")
        if not self.sim_sample_sizes or any(t < 2 for t in self.sim_sample_sizes):
            raise ValueError("sim_sample_sizes must be non-empty and >= 2")
        if self.sim_n_reps < 2:
            raise ValueError(f"sim_n_reps must be >= 2, got {self.sim_n_reps}")


@dataclass(frozen=True)
class ResidualConfig:
    """Residual-structure diagnostic and sector-block test (PLAN.md steps 8.23 and 8.24).

    Attributes
    ----------
    n_perm : int
        Sector-label permutations per estimation window for the residual-gap test (at least 19 so
        that a one-sided p-value can reach 0.05).
    sim_n_groups : int
        Number of weak group factors in the simulated positive control of step 8.24.
    sim_group_vol : float
        Annual volatility of each group factor of the positive control (chosen so that the
        within-group residual correlation is about the 0.06 found in step 8.23).
    sim_label_error : float
        Fraction of the group labels that are wrong in the noisy-label variant (the sector
        snapshot problem).
    sim_sample_sizes : tuple of int
        Sample sizes of the positive control (must exceed ``simulation.n_assets``).
    sim_n_reps : int
        Replications per sample size and design.
    n_placebo : int
        Random-label draws of the real-data placebo of step 8.25 (at least 19 so that a one-sided
        rank p-value can reach 0.05).
    """

    n_perm: int = 200
    sim_n_groups: int = 10
    sim_group_vol: float = 0.08
    sim_label_error: float = 0.2
    sim_sample_sizes: tuple[int, ...] = (252, 504)
    sim_n_reps: int = 100
    n_placebo: int = 20

    def __post_init__(self) -> None:
        if self.n_perm < 19:
            raise ValueError(f"n_perm must be >= 19, got {self.n_perm}")
        if self.sim_n_groups < 2 or self.sim_group_vol <= 0:
            raise ValueError("need sim_n_groups >= 2 and sim_group_vol > 0")
        if not 0.0 <= self.sim_label_error < 1.0:
            raise ValueError(f"sim_label_error must be in [0, 1), got {self.sim_label_error}")
        if not self.sim_sample_sizes or any(t < 2 for t in self.sim_sample_sizes):
            raise ValueError("sim_sample_sizes must be non-empty and >= 2")
        if self.sim_n_reps < 2:
            raise ValueError(f"sim_n_reps must be >= 2, got {self.sim_n_reps}")
        if self.n_placebo < 19:
            raise ValueError(f"n_placebo must be >= 19, got {self.n_placebo}")


@dataclass(frozen=True)
class ForecastConfig:
    """Covariance forecast loss on random portfolios (PLAN.md step 8.18).

    Attributes
    ----------
    n_portfolios : int
        Random long-only portfolios per rebalance date and concentration level.
    dirichlet_alphas : tuple of float
        Concentration parameters of the Dirichlet weights (1 = uniform on the simplex, diffuse;
        small values give concentrated portfolios).
    hac_lag : int
        Bartlett lag of the Newey-West variance used for the paired loss differences across
        rebalance dates.
    """

    n_portfolios: int = 500
    dirichlet_alphas: tuple[float, ...] = (1.0, 0.1)
    hac_lag: int = 3

    def __post_init__(self) -> None:
        if self.n_portfolios < 1:
            raise ValueError(f"n_portfolios must be >= 1, got {self.n_portfolios}")
        if not self.dirichlet_alphas or any(a <= 0 for a in self.dirichlet_alphas):
            raise ValueError("dirichlet_alphas must be non-empty and positive")
        if self.hac_lag < 0:
            raise ValueError(f"hac_lag must be >= 0, got {self.hac_lag}")


@dataclass(frozen=True)
class Config:
    """Top-level project configuration."""

    run: RunConfig = field(default_factory=RunConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    eda: EdaConfig = field(default_factory=EdaConfig)
    stats: StatsConfig = field(default_factory=StatsConfig)
    regimes: RegimeConfig = field(default_factory=RegimeConfig)
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    pca: PCAConfig = field(default_factory=PCAConfig)
    hrp: HRPConfig = field(default_factory=HRPConfig)
    sensitivity: SensitivityConfig = field(default_factory=SensitivityConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    pipeline_check: PipelineCheckConfig = field(default_factory=PipelineCheckConfig)
    nonstationarity: NonStationarityConfig = field(default_factory=NonStationarityConfig)
    residual: ResidualConfig = field(default_factory=ResidualConfig)
    forecast: ForecastConfig = field(default_factory=ForecastConfig)

    def __post_init__(self) -> None:
        n, t = self.universe.n_assets, self.universe.lookback_days
        cap = self.backtest.max_weight
        if cap is not None and cap < 1.0 / n:
            raise ValueError(f"max_weight={cap} is infeasible for n_assets={n} (needs >= 1/N)")
        if t <= n:
            logger.warning(
                "lookback_days T=%d <= n_assets N=%d: the sample covariance is rank-deficient", t, n
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON/YAML-serializable nested dict (tuples become lists)."""
        return json.loads(json.dumps(dataclasses.asdict(self)))

    def cache_key(self, length: int = 12) -> str:
        """Hex digest of the sections that determine the weights and returns, for cache folders.

        Only ``run``, ``data``, ``universe``, ``backtest``, ``pca`` and ``hrp`` are included, so
        editing analysis-only sections (``eda``, ``stats``, ``regimes``, ``sensitivity``,
        ``simulation``, paths)
        does not invalidate cached weights and returns.

        Parameters
        ----------
        length : int, default 12
            Number of hex characters returned.
        """
        sections = ("run", "data", "universe", "backtest", "pca", "hrp")
        full = self.to_dict()
        payload = json.dumps({name: full[name] for name in sections}, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]

    def config_hash(self, length: int = 12) -> str:
        """Stable hex digest of the full configuration, used as a cache key.

        Parameters
        ----------
        length : int, default 12
            Number of hex characters returned.
        """
        payload = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def _build(cls: type, data: dict[str, Any]) -> Any:
    """Recursively build dataclass ``cls`` from ``data``, rejecting unknown keys."""
    if not isinstance(data, dict):
        raise ValueError(f"{cls.__name__} section must be a mapping, got {type(data).__name__}")
    hints = get_type_hints(cls)
    unknown = set(data) - {f.name for f in dataclasses.fields(cls)}
    if unknown:
        raise ValueError(f"unknown key(s) in {cls.__name__}: {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        tp = hints[name]
        if dataclasses.is_dataclass(tp):
            kwargs[name] = _build(tp, value)
        elif get_origin(tp) is tuple:
            kwargs[name] = tuple(value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def config_from_dict(data: dict[str, Any]) -> Config:
    """Build and validate a :class:`Config` from a nested dict (missing keys use defaults)."""
    return _build(Config, data)


def load_config(path: str | Path = "config.yaml") -> Config:
    """Load and validate the YAML configuration file.

    Parameters
    ----------
    path : str or Path, default "config.yaml"
        Location of the YAML file.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the file contains unknown keys or invalid values.
    """
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return config_from_dict(raw)


def dump_config(cfg: Config, path: str | Path) -> None:
    """Write ``cfg`` to ``path`` as YAML (inverse of :func:`load_config`)."""
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg.to_dict(), handle, sort_keys=False)
