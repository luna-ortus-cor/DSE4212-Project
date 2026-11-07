# Where does the estimation error go?

DSE4212/QF4212 Project 2: Markowitz vs RMT-denoised vs PCA-factor vs HRP portfolios on daily S&P 500 data.

Status: scaffold only (step 0.1). Full documentation (setup, data placement, run order, runtime, repo map)
is written in step 7.3.

## Quick start

```
pip install -e . && pip install -r requirements.txt
pytest -q
ruff check . && ruff format --check .
```

Place the Kaggle CSVs in `data/raw/` (gitignored). See `PLAN.md` for methodology and `CLAUDE.md` for working rules.
