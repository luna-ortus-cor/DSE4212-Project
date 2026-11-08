"""End-to-end validation of the backtest pipeline on a synthetic panel (PLAN.md step 8.20).

A Date x Symbol panel of log returns and dollar volumes is drawn from the factor model of the
simulation study (more symbols than the universe size, persistent random dollar volumes so the
point-in-time universe changes) and pushed through the *unchanged* engine
(:func:`pfopt.backtest.engine.generate_weights` and :func:`pfopt.backtest.engine.simulate`).
Because the true covariance is known, three things can be checked:

- **accounting**: the gross return of day ``d`` has true variance ``w_d' Sigma w_d`` for the
  drifted weights ``w_d`` held that day, so ``z_d^2 = r_d^2 / (w_d' Sigma w_d)`` has mean 1;
- **identity**: at every rebalance date the true-variance ratio (to the oracle long-only GMV on
  the same universe) of the engine's weights equals the ratio that
  :func:`pfopt.simulation.run.evaluate_sample` gives for the same window;
- **ranking**: the paired log true-variance ratio of each strategy against ``GMV-SAMPLE`` over
  rebalance dates, which can be compared with the stationary simulation study of step 6.1.

Returns are simulated as log returns; the simple returns used for P&L are ``exp(r) - 1``.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from pfopt.backtest.engine import generate_weights, simulate
from pfopt.config import Config
from pfopt.data.eda import get_window
from pfopt.data.universe import select_universe
from pfopt.estimators.utils import winsorize_cols
from pfopt.simulation.dgp import FactorModel, ar1_paths, make_factor_model, simulate_returns
from pfopt.simulation.run import LONG_ONLY, evaluate_sample, oracle_portfolios

logger = logging.getLogger(__name__)

PIPELINE_STRATEGIES = ("EW", "GMV-SAMPLE", "GMV-LW", "GMV-RMT", "GMV-PCA", "HRP")
SAMPLE = "GMV-SAMPLE"
START_DATE = "2000-01-03"


@dataclass(frozen=True)
class SyntheticPanel:
    """A synthetic data panel and the true model behind it.

    Attributes
    ----------
    log_returns, dollar_volume, simple_returns : pandas.DataFrame
        Date x Symbol frames with identical index and columns.
    model : FactorModel
        True model of the ``n_symbols`` symbols (column order of the frames).
    """

    log_returns: pd.DataFrame
    dollar_volume: pd.DataFrame
    simple_returns: pd.DataFrame
    model: FactorModel


def make_panel(cfg: Config, panel: int) -> SyntheticPanel:
    """Draw synthetic panel number ``panel``.

    The true model, the returns and the dollar volumes use independent generators seeded with
    ``[seed, 2, panel]``, ``[seed, 3, panel]`` and ``[seed, 4, panel]``, so a panel does not depend
    on the number of panels.

    Parameters
    ----------
    cfg : Config
        Uses ``pipeline_check``, ``simulation`` (model parameters), ``run``.
    panel : int
        Panel index.
    """
    pc, seed = cfg.pipeline_check, cfg.run.seed
    sim = dataclasses.replace(cfg.simulation, n_assets=pc.n_symbols, sample_sizes=(pc.n_days,))
    model = make_factor_model(sim, cfg.run.annualization, np.random.default_rng([seed, 2, panel]))
    log_returns = simulate_returns(
        model, pc.n_days, np.random.default_rng([seed, 3, panel]), "gaussian"
    )
    log_volume = ar1_paths(
        pc.n_days,
        pc.n_symbols,
        pc.dv_persistence,
        pc.dv_sd,
        np.random.default_rng([seed, 4, panel]),
    )
    index = pd.bdate_range(START_DATE, periods=pc.n_days, name="Date")
    columns = [f"S{i:03d}" for i in range(pc.n_symbols)]
    logs = pd.DataFrame(log_returns, index=index, columns=columns)
    volume = pd.DataFrame(np.exp(log_volume), index=index, columns=columns)
    return SyntheticPanel(logs, volume, np.expm1(logs), model)


def run_panel(
    cfg: Config, panel: int, strategies: tuple[str, ...] = PIPELINE_STRATEGIES
) -> dict[str, pd.DataFrame]:
    """Run one synthetic panel through the engine and score it against the truth.

    Parameters
    ----------
    cfg : Config
        Full configuration (universe, backtest, pca, hrp, data, run, simulation, pipeline_check).
    panel : int
        Panel index.
    strategies : tuple of str
        Strategy ids; all must be among the long-only methods of the simulation study.

    Returns
    -------
    dict of str to pandas.DataFrame
        ``accounting`` (per strategy: days, ``mean_z2``, ``se_z2``, realized and true annualized
        volatility, mean predicted volatility), ``estimation`` (per rebalance date and strategy:
        engine variance ratio ``var_ratio``, study variance ratio ``var_ratio_study``, ``l1_dist``
        to the oracle) and ``info`` (universe size and panel number per date).
    """
    data = make_panel(cfg, panel)
    weights = generate_weights(data.log_returns, data.dollar_volume, cfg, strategies=strategies)
    sigma = data.model.sigma
    columns = list(data.log_returns.columns)
    position = {symbol: i for i, symbol in enumerate(columns)}
    uni, a = cfg.universe, float(cfg.run.annualization)

    rows = []
    for date in weights.diagnostics.index:
        members = select_universe(
            data.log_returns,
            data.dollar_volume,
            date,
            uni.lookback_days,
            uni.n_assets,
            uni.dollar_volume_window,
        )
        idx = np.array([position[s] for s in members])
        sigma_u = sigma[np.ix_(idx, idx)]
        oracle = oracle_portfolios(sigma_u)
        window = get_window(data.log_returns, date, uni.lookback_days, members)
        x = winsorize_cols(window.to_numpy(), cfg.data.winsor_lower, cfg.data.winsor_upper)
        study, _ = evaluate_sample(x, oracle, cfg)
        study_ratio = {r["method"]: r["var_ratio"] for r in study if r["constraint"] == LONG_ONLY}
        reference = oracle.weights[LONG_ONLY]
        for strategy in strategies:
            w = weights.weights[strategy].loc[date, members].to_numpy()
            rows.append(
                {
                    "panel": panel,
                    "date": date,
                    "strategy": strategy,
                    "n_universe": len(members),
                    "var_ratio": float(w @ sigma_u @ w) / oracle.variance[LONG_ONLY],
                    "var_ratio_study": float(study_ratio[strategy]),
                    "l1_dist": float(np.abs(w - reference).sum()),
                }
            )

    accounting = []
    for strategy in strategies:
        result = simulate(weights.weights[strategy], data.simple_returns, cfg.backtest.cost_bps)
        daily = result.daily_weights.to_numpy()
        true_var = np.einsum("ti,ij,tj->t", daily, sigma, daily)
        z2 = result.gross.to_numpy() ** 2 / true_var
        pred = weights.diagnostics.get(f"pred_vol_{strategy}")
        accounting.append(
            {
                "panel": panel,
                "strategy": strategy,
                "n_days": len(z2),
                "mean_z2": float(z2.mean()),
                "se_z2": float(z2.std(ddof=1) / np.sqrt(len(z2))),
                "realized_vol": float(result.gross.std(ddof=1) * np.sqrt(a)),
                "true_vol": float(np.sqrt(true_var.mean() * a)),
                "pred_vol": float("nan") if pred is None else float(pred.mean()),
            }
        )
    logger.info("panel %d done (%d rebalance dates)", panel, len(weights.diagnostics))
    return {"accounting": pd.DataFrame(accounting), "estimation": pd.DataFrame(rows)}


def accounting_summary(accounting: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    """Combine the accounting check across panels.

    Per strategy: mean of ``mean_z2`` over panels, its combined standard error
    ``sqrt(sum se^2) / n_panels``, the z-statistic against 1, the pass flag
    (``|z| <= tolerance``) and the mean realized/true and realized/predicted volatility ratios.
    """
    rows = []
    for strategy, cell in accounting.groupby("strategy", sort=False):
        n = len(cell)
        mean = float(cell["mean_z2"].mean())
        se = float(np.sqrt((cell["se_z2"] ** 2).sum()) / n)
        z = (mean - 1.0) / se
        rows.append(
            {
                "strategy": strategy,
                "n_panels": n,
                "mean_z2": mean,
                "se": se,
                "z": z,
                "passes": abs(z) <= tolerance,
                "realized_over_true": float((cell["realized_vol"] / cell["true_vol"]).mean()),
                "realized_over_predicted": float((cell["realized_vol"] / cell["pred_vol"]).mean()),
            }
        )
    return pd.DataFrame(rows)


def identity_summary(estimation: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    """Largest absolute difference between engine and study variance ratios, per strategy."""
    diff = (estimation["var_ratio"] - estimation["var_ratio_study"]).abs()
    out = diff.groupby(estimation["strategy"], sort=False).max().rename("max_abs_diff")
    out = out.reset_index()
    out["passes"] = out["max_abs_diff"] <= tolerance
    return out


def estimation_summary(estimation: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Paired log true-variance ratio against the sample-covariance GMV, across panels.

    The log ratio is averaged over the rebalance dates of each panel first; panels are
    independent, so the mean and a ``t`` interval are taken over panels (overlapping windows make a
    standard error over dates meaningless).

    Returns
    -------
    pandas.DataFrame
        Per strategy: ``mean_log_ratio``, ``se``, ``ci_low``, ``ci_high``, ``n_panels``, the mean
        variance ratio to the oracle and the mean L1 distance to the oracle weights.
    """
    log_ratio = np.log(estimation["var_ratio"])
    wide = log_ratio.groupby([estimation["panel"], estimation["strategy"]]).mean().unstack()
    paired = wide.sub(wide[SAMPLE], axis=0)
    n = len(paired)
    crit = float(stats.t.ppf(1.0 - alpha / 2.0, n - 1))
    ratio = estimation.groupby("strategy", sort=False)[["var_ratio", "l1_dist"]].mean()
    rows = []
    for strategy in estimation["strategy"].drop_duplicates():
        col = paired[strategy]
        se = float(col.std(ddof=1) / np.sqrt(n))
        rows.append(
            {
                "strategy": strategy,
                "mean_log_ratio": float(col.mean()),
                "se": se,
                "ci_low": float(col.mean()) - crit * se,
                "ci_high": float(col.mean()) + crit * se,
                "n_panels": n,
                "mean_var_ratio": float(ratio.loc[strategy, "var_ratio"]),
                "mean_l1_dist": float(ratio.loc[strategy, "l1_dist"]),
            }
        )
    return pd.DataFrame(rows)
