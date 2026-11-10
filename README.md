# Where does the estimation error go?

DSE4212/QF4212 Project 2 ("Portfolio Optimization Beyond Markowitz"): classical Markowitz / sample-covariance
portfolios against RMT-denoised, PCA-factor and HRP portfolios (with Ledoit-Wolf shrinkage and 1/N as baselines),
compared by a rolling out-of-sample backtest on the supplied daily S&P 500 data.

The submission contains the analysis code (`src/pfopt/`, `scripts/`, `tests/`), all parameters (`config.yaml`),
the final notebook (`notebooks/main.ipynb`), the report source (`report/main.tex`, with its PDF) and the report
figures and tables (`results/report/`, `results/tables/`, `results/figures/`).

## Setup

```
pip install -e . && pip install -r requirements.txt
pytest -q
ruff check . && ruff format --check .
```

Python 3.11+ (developed on 3.12). Place the Kaggle CSVs in `data/raw/` (not part of the submission):
`sp500_stocks.csv`, `sp500_companies.csv`, `sp500_index.csv`.

**Data scope.** Only the supplied stock file is used. Only 172 of its 502 symbols have prices, and the
universe is built from the survivors, so absolute performance carries survivorship bias. No symbol counts or dates
are hard-coded; every parameter is in `config.yaml`.

## Method in brief

- Log returns for estimation, simple returns for P&L. Per-window winsorization (0.1% / 99.9%). No look-ahead:
  weights at a rebalance date use data up to that date and are applied from the next day.
- Point-in-time universe: the top N = 100 names by trailing dollar volume with a full T = 504 day window;
  rebalance every 21 days; 10 bps proportional cost; long-only, fully invested.
- Estimators (written by hand in NumPy/SciPy): sample, Ledoit-Wolf, RMT eigenvalue clipping at the Marchenko-Pastur
  edge, PCA factor model, HRP, plus nonlinear (quadratic-inverse) shrinkage, EWMA and a sector-block covariance.
  Strategies: equal weight, global minimum variance (GMV) for each covariance, HRP, long-only max-Sharpe Markowitz,
  and the extensions (inverse volatility, equal risk contribution, regularized weights, ML-forecast max-Sharpe).

## Run order

Run from the repository root; every script takes `--config config.yaml`. Expensive results are cached in
`results/cache/` and keyed by the configuration.

| Order | Script | Purpose |
|---|---|---|
| 1 | `01_audit_data.py` | Raw data audit |
| 2 | `01b_clean_data.py` | Clean panel, return and volume matrices |
| 3 | `01c_universe_check.py` | Point-in-time universe check |
| 4 | `03_eda.py` | Exploratory analysis, Marchenko-Pastur spectrum |
| 5 | `02a_compare_estimators.py` | Estimator comparison at three dates |
| 6 | `03b_markowitz_frontier.py` | Estimated versus realized frontiers |
| 7 | `04a_generate_weights.py` | Leakage-free weights on the real data |
| 8 | `02_run_backtest.py` | Main backtest (writes the return cache used below) |
| 9 | `05_significance.py` | Block-bootstrap tests |
| 10 | `06_subperiods.py` | Subperiods and volatility regimes |
| 11 | `07_sensitivity.py` | Sensitivity to T, N, costs, cap, PCA K, linkage |
| 12 | `08_structure.py` | Concentration, turnover, sectors, spectrum diagnostics |
| 13 | `09_unconstrained.py` | Unconstrained GMV and hypothesis H3 |
| 14 | `10_simulation.py` | Factor-model simulation; `--n-groups G` and `--time-varying` give scenarios |
| 15 | `12_pipeline_check.py` | Backtest pipeline validated on synthetic panels |
| 16 | `13_inference.py` | Studentized bootstrap, Holm / BH, detectable effects |
| 17 | `14_nonstationarity.py` | EWMA strategies, volatility calibration, simulated positive control |
| 18 | `15_residual_structure.py` | Sector blocks left in the PCA residual correlation |
| 19 | `16_sector_block.py` | PCA plus sector-block covariance, with simulated controls |
| 20 | `17_sector_placebo.py` | Random-label placebo for the sector blocks on the real data |
| 21 | `18_risk_based.py` | Inverse-volatility and equal-risk-contribution portfolios |
| 22 | `19_nonlinear_shrinkage.py` | Quadratic-inverse (nonlinear) shrinkage against RMT clipping, with simulation |
| 23 | `20_forecast_loss.py` | Covariance forecast loss on random portfolios |
| 24 | `21_regularized_weights.py` | Ridge, shrinkage, L1-bounded and elastic-net weights |
| 25 | `22_ml_forecasts.py` | Walk-forward Ridge and random-forest return signals, rank IC, ML-fed max-Sharpe |
| 26 | `11_make_report_assets.py` | Re-runs the analysis stages (steps 4, 5, 6 and 8 to 25) and exports all figures and tables to `results/report/` |

Step 26 needs the cleaned data (step 2) and a completed backtest (steps 7 and 8). It then re-runs the analysis
stages itself, so it is the single command that regenerates every table and figure:
`python scripts/11_make_report_assets.py --config config.yaml`. It also deletes files in
`results/report/figures` and `results/report/tables` that the manifest does not list.

Outputs: tables in `results/tables/` (CSV plus a short `*_summary.txt` per stage), figures in `results/figures/`
(300 dpi PNG and PDF, width at most 6.3 in), report-ready copies (CSV, Markdown and LaTeX tables, PNG and PDF
figures) and `manifest.csv` in `results/report/`.

**Expected runtime.** The main backtest takes about 70 s. With a warm cache, step 26 re-runs every stage in a few
minutes; per-stage timings are written to `results/report/report_assets_summary.txt`. A cold run (no cache) takes
longer, mostly in the simulation, bootstrap and ML stages.

## Notebook

`notebooks/main.ipynb` is the narrative layer: it loads the exported tables and figures through
`results/report/manifest.csv`, recomputes a few headline quantities from the cached backtest as a consistency check,
and holds no result numbers of its own. Run step 26 first, then *Restart & Run All* (or
`jupyter nbconvert --to notebook --execute --inplace notebooks/main.ipynb`). Cells that need the processed data or
the cache print a message and are skipped when those are missing.

## Report

`report/main.tex` (11 pt, A4, 1.5 spacing) reads the figures and tables from `results/report/`, so run step 26
first. Compile from inside `report/`, twice so that cross-references resolve:

```
cd report && pdflatex main && pdflatex main
```

It needs a TeX distribution with the packages `booktabs`, `adjustbox`, `float`, `placeins`, `caption`, `setspace`
and `hyperref` (on Debian/Ubuntu: `texlive-latex-recommended texlive-latex-extra lmodern`).

## Repository map

```
config.yaml            all parameters (windows, universe, costs, seeds, simulation, checks)
src/pfopt/
  config.py            load and validate the configuration
  data/                loader, audit, cleaning, point-in-time universe, EDA helpers
  estimators/          sample + Ledoit-Wolf, RMT, nonlinear shrinkage, PCA factor, sector-block, HRP, EWMA, expected returns
  optimizers/          long-only and unconstrained GMV, max-Sharpe Markowitz, risk-based, regularized
  backtest/            engine, strategies, metrics, covariance quality, bootstrap stats, inference,
                       EWMA strategies, regimes, sensitivity, structure, unconstrained, forecast loss,
                       ML signals and ML portfolios
  simulation/          factor-model DGP (static and time-varying), study runner,
                       synthetic-panel pipeline check
  report/              figure and table registries for the report
  viz/                 shared plot style and plotting functions
scripts/               thin command-line runners, one per stage
tests/                 pytest suite mirroring src/
notebooks/main.ipynb   final notebook
report/main.tex        report source (LaTeX)
results/               tables, figures, report assets; results/cache is not part of the submission
```

## Results in brief

Real data (172 priced names, top-100 universe, 156 monthly rebalances, 10 bps costs):

- GMV portfolios cut volatility by about 22 to 25% against equal weight (ratio 0.75 to 0.78, significant after Holm).
- No Sharpe-ratio difference is statistically detectable. The test could only detect a gap of about 0.5 to 0.6
  against equal weight, so this is lack of power, not evidence of equality.
- Denoising did not help: PCA has 3.3% higher volatility than sample GMV (significant), RMT 4.5% higher (not
  significant at block 10, p = 0.060; significant at block 5 and not at 20, so only suggestive), and Ledoit-Wolf is
  indistinguishable in practice.
- The long-only constraint already regularizes: without it, leverage is 2.7 to 3.7, turnover 6 to 12 times higher,
  and hypothesis H3 holds only for Ledoit-Wolf.

Simulation (known factor model, N = 100):

- With a stationary covariance, RMT and PCA beat sample GMV, but by about 1% of variance at T = 504; the gain grows
  with N/T. More factors do not change this.
- With time-varying volatility and loadings, the RMT/PCA advantage disappears for T of 504 or more: the error is
  mostly non-stationarity, which denoising cannot fix. It does not reproduce the size of the real-data loss.
- The real-data engine is not the cause: run on synthetic stationary panels, it agrees with the direct evaluation to
  1e-15 and reproduces the simulation study's ranking.

Extensions (reported separately from the core results):

- Equal-risk-contribution portfolios are almost insensitive to the covariance estimator, so the denoising loss is
  specific to the minimum-variance optimizer. Ridge and shrinkage-to-1/N weights bring realized volatility closer to
  the predicted level at the price of higher volatility. Nonlinear shrinkage does not beat sample GMV.
- Sector correlation survives in the PCA residual; restoring sector blocks removes about three quarters of the PCA
  loss but does not beat sample GMV, and random labels recover only a small part of the gain.
- As forecasters of the variance of random portfolios, RMT and PCA are not detectably worse than the sample
  covariance, which points to how the optimizer uses the estimate. EWMA covariances are consistent with
  non-stationarity but the real-data test is underpowered.
- Ridge and random-forest return signals have an out-of-sample rank IC that is small and not significant, so the
  pre-set gate fails; the forced ML portfolios are more concentrated and more volatile than the references.

These are the results of single runs on one data path and the parameters in `config.yaml`, which were fixed in
advance and not tuned on the test period. Null and negative results are reported as such.
