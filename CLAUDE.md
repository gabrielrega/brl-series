# CLAUDE.md — BRL-Series Project Guide

## Project Overview

This is a **USD/BRL exchange rate time series forecasting project** that downloads historical data from Brazil's Central Bank and applies four forecasting methodologies:

1. **ARIMA** — Classical statistical approach with stationarity testing and grid-search parameter optimization
2. **Prophet** — Facebook's modern forecasting library with seasonality decomposition and holiday effects
3. **GARCH** — Volatility clustering model (Engle 1982, Bollerslev 1986); models conditional variance of FX returns
4. **VAR** — Vector Autoregression of USD/BRL + SELIC rate; grounded in Uncovered Interest Rate Parity (UIP)

---

## Repository Structure

```
brl-series/
├── download_data.py        # Fetches USD/BRL and supplementary BCB series
├── arima_analysis.py       # ARIMA modeling, diagnostics, and forecasting
├── prophet_analysis.py     # Prophet-based forecasting with cross-validation
├── garch_analysis.py       # GARCH(1,1) volatility modeling and forecast
├── var_analysis.py         # VAR model: USD/BRL + SELIC, Granger causality, IRF
├── requirements.txt        # Python dependencies
├── usd_brl_history.csv     # Input data: date + exchange rate (1,257+ rows)
├── bcb_series_432.csv      # SELIC rate data (created by download_data.py --all)
├── future_forecast.csv     # ARIMA output: future forecasts with confidence intervals
├── prophet_forecast.csv    # Prophet output: full forecast series
├── garch_forecast.csv      # GARCH output: 1-year volatility forecast
├── var_forecast.csv        # VAR output: USD/BRL + SELIC level forecasts
└── *.png                   # Generated visualizations
```

No subdirectories beyond `.git`. All source and output files live at the root.

---

## Running the Scripts

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download latest data from Brazil Central Bank API
#    --all also fetches SELIC (432), IPCA (433), CDI (11)
python download_data.py
python download_data.py --all   # required before running var_analysis.py

# 3. Run models (ARIMA and GARCH need only usd_brl_history.csv)
python arima_analysis.py
python prophet_analysis.py
python garch_analysis.py

# 4. VAR requires bcb_series_432.csv (from --all above)
python var_analysis.py
```

There is no build system, Makefile, or test suite.

---

## Dependencies

Install via:

```bash
pip install -r requirements.txt
```

Key packages and their roles:

| Package | Role |
|---|---|
| `pandas` | Data loading, manipulation, date indexing |
| `numpy` | Numerical operations |
| `matplotlib` | Saving plots to PNG |
| `statsmodels` | ARIMA, ADF/KPSS tests, ACF/PACF, Ljung-Box, VAR |
| `prophet` | Facebook Prophet forecasting + holidays |
| `scikit-learn` | Metrics (MAE, RMSE) |
| `scipy` | Statistical tests |
| `requests` | BCB API calls |
| `arch` | GARCH/ARCH models (Kevin Sheppard's library) |

---

## Data

### Input: `usd_brl_history.csv`
- Source: [BCB API](http://api.bcb.gov.br/dados/serie/bcdata.sgs.1/dados) (series code `1`)
- Columns: `data` (YYYY-MM-DD), `valor` (float exchange rate)
- ~5 years of business-day data, roughly 1,257 rows

### Supplementary: `bcb_series_{id}.csv`
Created by `python download_data.py --all`. Known series:

| File | Series ID | Description |
|---|---|---|
| `bcb_series_432.csv` | 432 | SELIC overnight rate (% p.a.) |
| `bcb_series_433.csv` | 433 | IPCA monthly inflation (%) |
| `bcb_series_11.csv` | 11 | CDI interbank rate (% p.a.) |

### Outputs

| File | Source | Contents |
|---|---|---|
| `future_forecast.csv` | ARIMA | Date, Forecast, Lower CI, Upper CI |
| `prophet_forecast.csv` | Prophet | ds, yhat, yhat_lower, yhat_upper |
| `prophet_cv_metrics.csv` | Prophet | CV metrics by horizon |
| `garch_forecast.csv` | GARCH | Date, conditional_variance, annualized_volatility |
| `var_forecast.csv` | VAR | Date, usd_brl_forecast, selic_rate_forecast |

---

## Code Conventions

- **Language:** Python 3, snake_case throughout
- **Style:** One script per methodology; steps within each script are commented as phases (Phase 1, Phase 2, etc.)
- **Logging:** `print()` statements for progress; no logging framework
- **Error handling:** `except Exception:` around model fitting; warnings suppressed with `warnings.filterwarnings`
- **Plots:** Saved directly to disk via `plt.savefig()`, followed by `plt.close()` to free memory
- **No hardcoded credentials** — BCB API is public and unauthenticated
- **Forecast horizon:** Dynamic — 1 year from the last data point (`pd.DateOffset(years=1)`), not a hardcoded date

---

## Key Implementation Details

### `download_data.py`
- `SERIES_CATALOG` dict at module level documents all known BCB series IDs
- Default (no args): downloads only series 1 → `usd_brl_history.csv`
- `--all` flag: also downloads SELIC (432), IPCA (433), CDI (11) → `bcb_series_{id}.csv`
- Uses 5-year lookback window calculated from today's date at runtime

### `arima_analysis.py`
- Reads `usd_brl_history.csv`, parses `data` column with `pd.to_datetime`
- Stationarity: ADF + KPSS tests on raw series and 1st-differenced series
- Grid search over `p ∈ {0,1,2}`, `q ∈ {0,1,2}`, `d=1` fixed, by AIC
- Diagnostics: Ljung-Box (residual autocorrelation), Jarque-Bera (normality)
- Validation: Walk-forward out-of-sample evaluation on last 60 observations
- Forecast horizon: 1 year from last data point (dynamic)

### `prophet_analysis.py`
- Renames columns to `ds`/`y` as required by Prophet
- Uses additive seasonality; adds Brazilian holidays via `add_country_holidays('BR')`
- Cross-validation via `prophet.diagnostics.cross_validation` + `performance_metrics`
- Forecast horizon: 1 year from last data point (dynamic)

### `garch_analysis.py`
- Computes percentage log returns: `100 * ln(P_t / P_{t-1})`
- Tests for ARCH effects (ARCH-LM test) before fitting
- Fits GARCH(1,1) with Student-t distribution via the `arch` library
- Prints persistence `alpha + beta` — values near 1 indicate slow volatility decay
- Annualizes volatility: `sqrt(variance_daily) / 100 * sqrt(252)`
- Forecast horizon: 252 business days (1 trading year)

### `var_analysis.py`
- Requires `bcb_series_432.csv` — exits with a helpful message if missing
- Inner-merges USD/BRL and SELIC on date to align the two daily series
- Transforms to stationary: log-diff for FX, first-diff for SELIC
- Selects VAR lag order by AIC (`maxlags=10`)
- Granger causality test: does SELIC predict USD/BRL?
- Orthogonalized IRF (20 periods): effect of a SELIC rate shock on BRL
- Inverts differencing via `cumsum` to recover level forecasts

---

## Development Notes

- **No CI/CD** — no GitHub Actions, no pre-commit hooks configured
- **No tests** — scripts are research/analysis oriented, not production services
- **Git branches:** main development branch is `master`; feature work on `claude/*` branches
- **Remote:** `http://local_proxy@127.0.0.1:44641/git/gabrielrega/brl-series`

When adding new analysis scripts, follow the existing pattern:
1. One script per methodology
2. Read from `usd_brl_history.csv` (and optionally `bcb_series_{id}.csv`) as inputs
3. Use `last_date + pd.DateOffset(years=1)` for forecast horizon — never hardcode a date
4. Save all plots as PNGs at the root level with `plt.savefig()` followed by `plt.close()`
5. Export forecast results as CSV with clear column names

---

## Common Tasks for AI Assistants

### Adding a new forecasting model
1. Create `<model>_analysis.py` at the root
2. Import `usd_brl_history.csv` using pandas with `pd.to_datetime`
3. Follow the Phase 1/2/3/4 comment structure from existing scripts
4. Use `last_date + pd.DateOffset(years=1)` for forecast horizon
5. Export forecasts to `<model>_forecast.csv` and plots to `<model>_*.png`
6. Add the new dependency to `requirements.txt` if needed

### Updating the data
```bash
python download_data.py         # USD/BRL only
python download_data.py --all   # USD/BRL + SELIC + IPCA + CDI
```

### Adding a new BCB series
Add an entry to `SERIES_CATALOG` in `download_data.py`:
```python
SERIES_CATALOG = {
    1:   "USD/BRL exchange rate",
    432: "SELIC overnight rate (% p.a.)",
    # add new series here
}
```
Then run `python download_data.py --all`.

### Interpreting ARIMA results
- `future_forecast.csv` columns: Date (index), `Forecast`, `Lower CI`, `Upper CI`
- AIC-selected (p, d, q) parameters are printed to stdout during the run

### Interpreting Prophet results
- `prophet_forecast.csv` contains historical fitted values + future projections
- `prophet_cv_metrics.csv` contains MAE/RMSE/MAPE by forecast horizon (in days)

### Interpreting GARCH results
- `garch_forecast.csv` columns: Date (index), `conditional_variance`, `annualized_volatility`
- `annualized_volatility` is a fraction (e.g., 0.12 = 12% per year); multiply by 100 for display
- `alpha + beta` persistence printed to stdout — values > 0.95 indicate slow mean reversion

### Interpreting VAR results
- `var_forecast.csv` columns: Date (index), `usd_brl_forecast`, `selic_rate_forecast`
- Granger causality p-value printed to stdout — p < 0.05 means SELIC helps predict BRL
- `var_irf.png` shows the IRF: under UIP, a positive SELIC shock should appreciate BRL (USD/BRL falls)
