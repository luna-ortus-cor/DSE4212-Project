# Where does the estimation error go?

DSE4212/QF4212 Project 2 ("Portfolio Optimization Beyond Markowitz"): classical Markowitz / sample-covariance
portfolios against RMT-denoised, PCA-factor and HRP portfolios (with Ledoit-Wolf shrinkage and 1/N as baselines),
compared by a rolling out-of-sample backtest on the supplied daily S&P 500 data.

Status: the analysis pipeline (steps 0.1 to 6.3 and the extensions 8.2, 8.7, 8.18 and 8.20 to 8.25) is implemented. The final notebook
(7.2) and the packaging and full run-order documentation (7.3) are not done yet. `PLAN.md` holds the methodology,
`PROGRESS.md` the step log with results, and `CLAUDE.md` the working rules.

## Setup

```
pip install -e . && pip install -r requirements.txt
pytest -q
ruff check . && ruff format --check .
```

Python 3.11+ (developed on 3.12, system interpreter, no virtualenv). Place the Kaggle CSVs in `data/raw/`
(gitignored): `sp500_stocks.csv`, `sp500_companies.csv`, `sp500_index.csv`.

**Data scope.** Only the supplied stock file is used. Only 172 of its 502 symbols have prices, and the
universe is built from the survivors, so absolute performance carries survivorship bias. No symbol counts or dates
are hard-coded; every parameter is in `config.yaml`.

## Method in brief

- Log returns for estimation, simple returns for P&L. Per-window winsorization (0.1% / 99.9%). No look-ahead:
  weights at a rebalance date use data up to that date and are applied from the next day.
- Point-in-time universe: the top N = 100 names by trailing dollar volume with a full T = 504 day window;
  rebalance every 21 days; 10 bps proportional cost; long-only, fully invested.
- Estimators (written by hand in NumPy/SciPy): sample, Ledoit-Wolf, RMT eigenvalue clipping at the Marchenko-Pastur
  edge, PCA factor model, HRP. Strategies: equal weight, global minimum variance (GMV) for each covariance,
  HRP, and long-only max-Sharpe Markowitz.

## Run order

Run from the repository root; every script takes `--config config.yaml`. Expensive results are cached in
`results/cache/` and keyed by the configuration.

| Order | Script | Step | Purpose |
|---|---|---|---|
| 1 | `01_audit_data.py` | 1.1 | Raw data audit |
| 2 | `01b_clean_data.py` | 1.2 | Clean panel, return and volume matrices |
| 3 | `01c_universe_check.py` | 1.3 | Point-in-time universe check |
| 4 | `03_eda.py` | 1.4 | Exploratory analysis, Marchenko-Pastur spectrum |
| 5 | `02a_compare_estimators.py` | 2.5 | Estimator comparison at three dates |
| 6 | `03b_markowitz_frontier.py` | 3.2 | Estimated versus realized frontiers |
| 7 | `04a_generate_weights.py` | 4.1 | Leakage-free weights on the real data |
| 8 | `02_run_backtest.py` | 4.5 | Main backtest (writes the return cache used below) |
| 9 | `05_significance.py` | 5.1 | Block-bootstrap tests |
| 10 | `06_subperiods.py` | 5.2 | Subperiods and volatility regimes |
| 11 | `07_sensitivity.py` | 5.3 | Sensitivity to T, N, costs, cap, PCA K, linkage |
| 12 | `08_structure.py` | 5.4 | Concentration, turnover, sectors, spectrum diagnostics |
| 13 | `09_unconstrained.py` | 5.5 | Unconstrained GMV and hypothesis H3 |
| 14 | `10_simulation.py` | 6.1 to 6.3 | Factor-model simulation; `--n-groups G` and `--time-varying` give scenarios |
| 15 | `12_pipeline_check.py` | 8.20 | Backtest pipeline validated on synthetic panels |
| 16 | `13_inference.py` | 8.21 | Studentized bootstrap, Holm / BH, detectable effects |
| 17 | `14_nonstationarity.py` | 8.22 | EWMA strategies, volatility calibration, simulated positive control |
| 18 | `15_residual_structure.py` | 8.23 | Sector blocks left in the PCA residual correlation |
| 19 | `16_sector_block.py` | 8.24 | PCA plus sector-block covariance, with simulated controls |
| 20 | `17_sector_placebo.py` | 8.25 | Random-label placebo for the sector blocks on the real data |
| 21 | `18_risk_based.py` | 8.7 | Inverse-volatility and equal-risk-contribution portfolios |
| 22 | `19_nonlinear_shrinkage.py` | 8.2 | Quadratic-inverse (nonlinear) shrinkage against RMT clipping, with simulation |
| 23 | `20_forecast_loss.py` | 8.18 | Covariance forecast loss on random portfolios |
| 24 | `11_make_report_assets.py` | 7.1 | Re-runs the stage scripts and exports all figures and tables to `results/report/` |

Outputs: tables in `results/tables/` (CSV plus a short `*_summary.txt` per stage), figures in `results/figures/`
(300 dpi PNG and PDF, width at most 6.3 in), report-ready copies and `manifest.csv` in `results/report/`.

## Repository map

```
config.yaml            all parameters (windows, universe, costs, seeds, simulation, checks)
src/pfopt/
  config.py            load and validate the configuration
  data/                loader, audit, cleaning, point-in-time universe, EDA helpers
  estimators/          sample + Ledoit-Wolf, RMT, nonlinear shrinkage, PCA factor, sector-block, HRP, EWMA, expected returns
  optimizers/          long-only and unconstrained GMV, max-Sharpe Markowitz
  backtest/            engine, strategies, metrics, covariance quality, bootstrap stats,
                       inference (8.21), EWMA strategies (8.22), regimes, sensitivity, structure,
                       unconstrained
  simulation/          factor-model DGP (static and time-varying), study runner,
                       synthetic-panel pipeline check (8.20)
  report/              figure and table registries for the report
  viz/                 shared plot style and plotting functions
scripts/               thin command-line runners, one per stage
tests/                 pytest suite mirroring src/
```

## Results so far

Real data (172 priced names, top-100 universe, 156 monthly rebalances, 10 bps costs):

- GMV portfolios cut volatility by about 22 to 25% against equal weight (ratio 0.75 to 0.78, significant after Holm).
- No Sharpe-ratio difference is statistically detectable. The test could only detect a gap of about 0.5 to 0.6
  against equal weight, so this is lack of power, not evidence of equality.
- Denoising did not help: PCA has 3.3% higher volatility than sample GMV (significant), RMT 4.5% higher (not
  significant at block 10, p = 0.060; significant at block 5 and not at 20, so only suggestive), and Ledoit-Wolf is indistinguishable in practice.
- The long-only constraint already regularizes: without it, leverage is 2.7 to 3.7, turnover 6 to 12 times higher,
  and hypothesis H3 holds only for Ledoit-Wolf.

Simulation (known factor model, N = 100):

- With a stationary covariance, RMT and PCA beat sample GMV, but by about 1% of variance at T = 504; the gain grows
  with N/T. More factors do not change this.
- With time-varying volatility and loadings, the RMT/PCA advantage disappears for T of 504 or more: the error is
  mostly non-stationarity, which denoising cannot fix. It does not reproduce the size of the real-data loss.
- The real-data engine is not the cause: run on synthetic stationary panels, it matches the simulation study to
  machine precision and reproduces its ranking (step 8.20).

These are the results of single runs on one data path and the parameters in `config.yaml`, which were fixed in
advance and not tuned on the test period. Null and negative results are reported as such. Extensions (Phase 8)
are reported separately from the core results.
