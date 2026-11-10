"""Sensitivity analyses driver.

One declarative list of :class:`Scenario` objects covers every grid: estimation window ``T``,
universe size ``N`` (so ``q = N / T`` varies), transaction costs, a 10% weight cap, the PCA factor
rule and the HRP linkage. Each scenario is a full copy of the configuration with exactly one field
changed, so no result mixes parameters from different runs. Weights are cached by
``Config.cache_key()`` (cost scenarios share the baseline weights because costs do not change
weights). The baseline is the headline configuration of ``config.yaml``; these runs are sensitivity
only and are never used to select it.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pfopt.backtest.engine import (
    ALL_STRATEGIES,
    SimulationResult,
    WeightsResult,
    generate_weights,
    read_weights_result,
    simulate_all,
    write_weights_result,
)
from pfopt.backtest.metrics import performance_metrics
from pfopt.config import Config, PCAConfig
from pfopt.data.universe import eligible_counts

logger = logging.getLogger(__name__)

HRP_STRATEGIES = ("HRP", "HRP-RMT")
GRIDS = ("lookback", "universe", "cost", "cap", "pca_k", "hrp_linkage")
METRIC_COLUMNS = ("ann_return", "ann_vol", "sharpe", "max_drawdown", "annual_turnover")


@dataclass(frozen=True)
class Scenario:
    """One sensitivity run: a configuration, a cost level and the strategies to evaluate.

    Attributes
    ----------
    id : str
        Unique id, ``"baseline"`` or ``"<grid>:<variant>"``.
    grid, variant : str
        Grid name and the human-readable variant label (``"T=252"``, ``"K=3"``, ...).
    cfg : Config
        Configuration that determines the weights.
    cost_bps : float
        Proportional cost used in the simulation.
    strategies : tuple of str
        Strategy ids evaluated in this scenario.
    """

    id: str
    grid: str
    variant: str
    cfg: Config
    cost_bps: float
    strategies: tuple[str, ...]

    @property
    def q(self) -> float:
        """Concentration ratio ``N / T`` of the scenario."""
        return self.cfg.universe.n_assets / self.cfg.universe.lookback_days


def _universe(cfg: Config, **changes: int) -> Config:
    return dataclasses.replace(cfg, universe=dataclasses.replace(cfg.universe, **changes))


def build_scenarios(cfg: Config, log_returns: pd.DataFrame) -> tuple[list[Scenario], list[str]]:
    """List every sensitivity scenario of the configured grids.

    Values equal to the baseline are not repeated (the baseline is scenario ``"baseline"``).
    Combinations of ``(N, T)`` that are never feasible on the data (fewer than ``N`` symbols with
    a full ``T``-day window) are skipped and reported.

    Parameters
    ----------
    cfg : Config
        Baseline configuration; the grids come from ``cfg.sensitivity``.
    log_returns : pandas.DataFrame
        Date x Symbol log returns, used only to test feasibility.

    Returns
    -------
    (list of Scenario, list of str)
        The scenarios (baseline first) and one message per skipped variant.
    """
    s, base_n, base_t = cfg.sensitivity, cfg.universe.n_assets, cfg.universe.lookback_days
    cost = cfg.backtest.cost_bps
    scenarios = [Scenario("baseline", "baseline", "baseline", cfg, cost, ALL_STRATEGIES)]
    skipped: list[str] = []

    def add(grid: str, variant: str, config: Config, strategies: tuple[str, ...], c: float = cost):
        scenarios.append(Scenario(f"{grid}:{variant}", grid, variant, config, c, strategies))

    def feasible(n: int, t: int) -> bool:
        counts = eligible_counts(log_returns, t)
        return bool(len(counts) and counts.max() >= n)

    for t in s.lookback_days:
        if t == base_t:
            continue
        if feasible(base_n, t):
            add("lookback", f"T={t}", _universe(cfg, lookback_days=t), ALL_STRATEGIES)
        else:
            skipped.append(f"T={t}: fewer than N={base_n} symbols ever have a full window")
    for n in s.n_assets:
        if n == base_n:
            continue
        if feasible(n, base_t):
            add("universe", f"N={n}", _universe(cfg, n_assets=n), ALL_STRATEGIES)
        else:
            counts = eligible_counts(log_returns, base_t)
            most = int(counts.max()) if len(counts) else 0
            skipped.append(f"N={n}: at most {most} symbols have a full {base_t}-day window")
    for c in s.cost_bps:
        if c != cost:
            add("cost", f"{c:g} bps", cfg, ALL_STRATEGIES, c)
    for cap in s.max_weight:
        backtest = dataclasses.replace(cfg.backtest, max_weight=cap)
        add("cap", f"cap {cap:.0%}", dataclasses.replace(cfg, backtest=backtest), ALL_STRATEGIES)
    for k in s.pca_k_fixed:
        pca = PCAConfig(k_rule="fixed", k_fixed=k, baing_kmax=cfg.pca.baing_kmax)
        add("pca_k", f"K={k}", dataclasses.replace(cfg, pca=pca), ("GMV-PCA",))
    if cfg.pca.k_rule != "baing":
        pca = PCAConfig(k_rule="baing", baing_kmax=cfg.pca.baing_kmax)
        add("pca_k", "Bai-Ng", dataclasses.replace(cfg, pca=pca), ("GMV-PCA",))
    for linkage in s.hrp_linkage:
        if linkage != cfg.hrp.linkage:
            hrp = dataclasses.replace(cfg.hrp, linkage=linkage)
            add("hrp_linkage", linkage, dataclasses.replace(cfg, hrp=hrp), HRP_STRATEGIES)
    return scenarios, skipped


def _weights_for(
    scenario: Scenario,
    log_returns: pd.DataFrame,
    dollar_volume: pd.DataFrame,
    cache_root: Path,
    force: bool,
) -> WeightsResult:
    """Cached weights of a scenario (all requested strategies present) or freshly generated."""
    folder = cache_root / scenario.cfg.cache_key()
    if not force and (folder / "diagnostics.parquet").exists():
        cached = read_weights_result(folder)
        if set(scenario.strategies) <= set(cached.weights):
            logger.info("%s: using cached weights from %s", scenario.id, folder)
            return WeightsResult(
                weights={s: cached.weights[s] for s in scenario.strategies},
                diagnostics=cached.diagnostics,
                universe=cached.universe,
            )
    logger.info("%s: generating weights", scenario.id)
    result = generate_weights(log_returns, dollar_volume, scenario.cfg, scenario.strategies)
    write_weights_result(result, folder)
    return result


def run_scenarios(
    scenarios: list[Scenario],
    log_returns: pd.DataFrame,
    dollar_volume: pd.DataFrame,
    simple_returns: pd.DataFrame,
    cache_root: str | Path,
    force: bool = False,
) -> dict[str, dict[str, SimulationResult]]:
    """Generate (or load) weights and simulate every scenario.

    Scenarios that share a configuration (cost variants) reuse the same weights in memory.

    Returns
    -------
    dict
        Scenario id to ``{strategy: SimulationResult}``.
    """
    root = Path(cache_root)
    memo: dict[tuple[str, tuple[str, ...]], WeightsResult] = {}
    runs: dict[str, dict[str, SimulationResult]] = {}
    for scenario in scenarios:
        key = (scenario.cfg.cache_key(), scenario.strategies)
        if key not in memo:
            memo[key] = _weights_for(scenario, log_returns, dollar_volume, root, force)
        runs[scenario.id] = simulate_all(memo[key], simple_returns, scenario.cost_bps)
    return runs


def common_start(runs: dict[str, dict[str, SimulationResult]]) -> pd.Timestamp:
    """Latest first return date over all scenarios and strategies (start of the common window)."""
    return max(sim.net.index[0] for sims in runs.values() for sim in sims.values())


def _turnover_from(turnover: pd.Series, start: pd.Timestamp) -> pd.Series:
    """Turnover from the rebalance whose weights are held on ``start`` (its entry is the base)."""
    before = turnover.index[turnover.index < start]
    first = before[-1] if len(before) else turnover.index[0]
    return turnover.loc[turnover.index >= first]


def scenario_metrics(
    sims: dict[str, SimulationResult],
    start: pd.Timestamp | None,
    annualization: int,
    rf: float,
) -> pd.DataFrame:
    """Net-of-cost metrics per strategy, over the whole sample or from ``start`` onward.

    Parameters
    ----------
    sims : dict
        Simulation results of one scenario.
    start : Timestamp or None
        First day included (``None`` keeps each strategy's own full sample).
    annualization, rf
        As in :func:`pfopt.backtest.metrics.performance_metrics`.

    Returns
    -------
    pandas.DataFrame
        Strategy x (``n_days`` + :data:`METRIC_COLUMNS`).
    """
    rows = {}
    for strategy, sim in sims.items():
        net, turnover = sim.net, sim.turnover
        if start is not None:
            net, turnover = net.loc[net.index >= start], _turnover_from(turnover, start)
        values = performance_metrics(net, turnover, annualization, rf)
        rows[strategy] = {"n_days": len(net), **{c: values[c] for c in METRIC_COLUMNS}}
    return pd.DataFrame.from_dict(rows, orient="index")


def metrics_long(
    scenarios: list[Scenario],
    runs: dict[str, dict[str, SimulationResult]],
    annualization: int,
    rf: float,
) -> pd.DataFrame:
    """Long table of metrics for every scenario, strategy and period (``own`` / ``common``).

    ``common`` restricts all scenarios to the same days (from :func:`common_start`) so that
    different ``T`` are compared on identical out-of-sample dates.
    """
    start = common_start(runs)
    frames = []
    for scenario in scenarios:
        for period, begin in (("own", None), ("common", start)):
            table = scenario_metrics(runs[scenario.id], begin, annualization, rf)
            table.insert(0, "period", period)
            table.insert(0, "strategy", table.index)
            table.insert(0, "cost_bps", scenario.cost_bps)
            table.insert(0, "q", scenario.q)
            table.insert(0, "lookback_days", scenario.cfg.universe.lookback_days)
            table.insert(0, "n_assets", scenario.cfg.universe.n_assets)
            table.insert(0, "variant", scenario.variant)
            table.insert(0, "grid", scenario.grid)
            frames.append(table.reset_index(drop=True))
    out = pd.concat(frames, ignore_index=True)
    capped = out["grid"] == "cap"
    out["note"] = ""
    out.loc[capped & out["strategy"].isin(HRP_STRATEGIES), "note"] = "HRP has no cap mechanism"
    return out


def trial_counts(scenarios: list[Scenario]) -> pd.DataFrame:
    """Number of strategy-parameter combinations tried, per grid and in total.

    A combination is one (scenario, strategy) pair; the baseline is counted once under
    ``baseline``. Needed for the multiple-testing discussion.
    """
    rows = [
        {
            "grid": grid,
            "variants": sum(1 for s in scenarios if s.grid == grid),
            "combinations": sum(len(s.strategies) for s in scenarios if s.grid == grid),
        }
        for grid in ("baseline", *GRIDS)
    ]
    table = pd.DataFrame(rows)
    total = {"grid": "total", "variants": table["variants"].sum()}
    total["combinations"] = table["combinations"].sum()
    return pd.concat([table, pd.DataFrame([total])], ignore_index=True)
