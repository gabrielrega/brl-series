# CLAUDE.md — BRL-Series Project Guide

## Project Overview

This is a **USD/BRL exchange rate time series forecasting project** that downloads historical data from Brazil's Central Bank and applies two forecasting methodologies:

1. **ARIMA** — Classical statistical approach with stationarity testing and grid-search parameter optimization
2. **Prophet** — Facebook's modern forecasting library with seasonality decomposition and holiday effects

---

## Repository Structure

```
brl-series/
├── download_data.py        # Fetches USD/BRL historical data from BCB API
├── arima_analysis.py       # ARIMA modeling, diagnostics, and forecasting
├── prophet_analysis.py     # Prophet-based forecasting with cross-validation
├── usd_brl_history.csv     # Input data: date + exchange rate (1,257+ rows)
├── future_forecast.csv     # ARIMA output: future forecasts with confidence intervals
├── prophet_forecast.csv    # Prophet output: full forecast series
└── *.png                   # Generated visualizations (ACF, PACF, forecasts, etc.)
```

No subdirectories beyond `.git`. All source and output files live at the root.

---

## Running the Scripts

Run in this order:

```bash
# 1. Download latest data from Brazil Central Bank API
python download_data.py

# 2. Run ARIMA analysis (produces future_forecast.csv + PNG plots)
python arima_analysis.py

# 3. Run Prophet analysis (produces prophet_forecast.csv + PNG plots)
python prophet_analysis.py
```

There is no build system, Makefile, or test suite.

---

## Dependencies

No `requirements.txt` or `pyproject.toml` exists. Install manually:

```bash
pip install pandas numpy matplotlib statsmodels prophet scikit-learn scipy requests
```

Key packages and their roles:
| Package | Role |
|---|---|
| `pandas` | Data loading, manipulation, date indexing |
| `numpy` | Numerical operations |
| `matplotlib` | Saving plots to PNG |
| `statsmodels` | ARIMA, ADF test, KPSS test, ACF/PACF, Ljung-Box |
| `prophet` | Facebook Prophet forecasting + holidays |
| `scikit-learn` | Metrics (MAE, RMSE) |
| `scipy` | Statistical tests |
| `requests` | BCB API calls |

---

## Data

### Input: `usd_brl_history.csv`
- Source: [BCB API](http://api.bcb.gov.br/dados/serie/bcdata.sgs.1/dados) (series code `1`)
- Columns: `data` (DD/MM/YYYY), `valor` (float exchange rate)
- ~5 years of business-day data, roughly 1,257 rows

### Outputs
- `future_forecast.csv` — ARIMA near-term point forecasts + 95% confidence intervals
- `prophet_forecast.csv` — Full Prophet forecast series (`ds`, `yhat`, `yhat_lower`, `yhat_upper`)
- `prophet_cv_metrics.csv` — Cross-validation metrics by forecast horizon

---

## Code Conventions

- **Language:** Python 3, snake_case throughout
- **Style:** One script per methodology; steps within each script are commented as phases (Phase 1, Phase 2, etc.)
- **Logging:** `print()` statements for progress; no logging framework
- **Error handling:** `try/except` blocks around model fitting; warnings suppressed with `warnings.filterwarnings`
- **Plots:** Saved directly to disk via `plt.savefig()`, not displayed interactively (`plt.close()` after each)
- **No hardcoded credentials** — BCB API is public and unauthenticated

---

## Key Implementation Details

### `download_data.py`
- Calls BCB API with a 5-year lookback window (calculated from today's date at runtime)
- Writes `usd_brl_history.csv` with columns `data`, `valor`

### `arima_analysis.py`
- Reads `usd_brl_history.csv`, parses `data` as `dayfirst=True`
- Stationarity: ADF + KPSS tests on the raw series
- Differencing: `d=1` fixed; grid search over `p ∈ {0,1,2}`, `q ∈ {0,1,2}` by AIC
- Diagnostics: Ljung-Box (residual autocorrelation), Jarque-Bera (normality)
- Validation: Walk-forward out-of-sample evaluation
- Forecast horizon: through `2025-12-31`

### `prophet_analysis.py`
- Renames columns to `ds`/`y` as required by Prophet
- Uses additive seasonality; adds Brazilian holidays via `add_country_holidays('BR')`
- Hyperparameter tuning over `changepoint_prior_scale` and `seasonality_prior_scale`
- Cross-validation via `prophet.diagnostics.cross_validation` + `performance_metrics`

---

## Development Notes

- **No CI/CD** — no GitHub Actions, no pre-commit hooks configured
- **No tests** — scripts are research/analysis oriented, not production services
- **Git branches:** main development branch is `master`; feature work on `claude/*` branches
- **Remote:** `http://local_proxy@127.0.0.1:44641/git/gabrielrega/brl-series`

When adding new analysis scripts, follow the existing pattern:
1. One script per methodology
2. Read from `usd_brl_history.csv` as the canonical input
3. Save all plots as PNGs at the root level
4. Export forecast results as CSV with clear column names

---

## Common Tasks for AI Assistants

### Adding a new forecasting model
1. Create `<model>_analysis.py` at the root
2. Import `usd_brl_history.csv` using pandas with `dayfirst=True` date parsing
3. Follow the Phase 1/2/3 comment structure from existing scripts
4. Export forecasts to `<model>_forecast.csv` and plots to `<model>_*.png`

### Updating the data
```bash
python download_data.py
```
This overwrites `usd_brl_history.csv` with the latest available data from BCB.

### Adding dependencies
Since no `requirements.txt` exists, either:
- Create one listing all current imports, or
- Add the new package to an existing one if you create it

### Interpreting ARIMA results
- `future_forecast.csv` columns: date index, `mean`, `mean_ci_lower`, `mean_ci_upper`
- AIC-selected (p, d, q) parameters are printed to stdout during the run

### Interpreting Prophet results
- `prophet_forecast.csv` contains historical fitted values + future projections
- `prophet_cv_metrics.csv` contains MAE/RMSE/MAPE by forecast horizon (in days)
