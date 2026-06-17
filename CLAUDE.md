# CLAUDE.md — brl-series

Time-series analysis of the **BRL/USD exchange rate** (Brazilian Central Bank
data). Several models are fit and, crucially, scored against each other on a
single shared cross-validation so the comparison is honest. The headline result
is negative and intentional: **no model significantly beats a random walk**
(Meese-Rogoff holds for daily FX).

## Environment & commands

Managed with [uv](https://docs.astral.sh/uv/); deps pinned in `requirements.txt`
(no `pyproject.toml`/lockfile).

```bash
uv venv && uv pip install -r requirements.txt
python main.py        # full pipeline: download + all models + comparison tables
python test_models.py # offline sanity suite (no network), 21 assertions
```

`main.py` needs network (BCB API). `test_models.py` runs on synthetic data plus
the cached `data/usd_brl_history.csv` if present, and uses the `Agg` matplotlib
backend so it never opens a window.

## Architecture

**Modular line.** `main.py` is the orchestrator; `evaluation.py` is the core;
each model lives in its own `*_analysis.py` exposing a `run_*_analysis(...)`
function plus a thin `*_forecast(train, future_index)` callable. There is no
`download_data.py` and the model files are **not** meant to be run standalone —
they are imported by `main.py`. (An older monolithic line with root-level CSVs
and standalone scripts was retired; don't reintroduce that pattern.)

```
main.py            Downloads BCB series, runs every model, prints two comparison
                   tables (level + volatility) with Diebold-Mariano columns.
evaluation.py      THE shared machinery — read this first:
                     rolling_origin_cv()      level CV (ARIMA/ETS/Prophet/VAR)
                     rolling_origin_vol_cv()  volatility CV (GARCH)
                     naive_rw_forecast()      random-walk level baseline
                     constant_vol_forecast()  constant-vol volatility baseline
                     diebold_mariano()        significance test (HLN + Newey-West)
                     INITIAL/PERIOD/HORIZON    the one CV scheme everyone uses
arima_analysis.py  ARIMA, AIC grid search over (p,1,q), p,q in 0..2.
ets_analysis.py    ETS, damped additive trend, NO seasonal (it was inert on this
                   series — see fit_ets docstring).
prophet_analysis.py  Prophet, weekly+yearly seasonality, BR holidays.
garch_analysis.py  GARCH(1,1) Student-t (Normal fallback) — targets VOLATILITY,
                   not the level, so it is scored on its own CV.
var_analysis.py    Bivariate VAR(USD/BRL, SELIC), differenced; UIP angle +
                   Granger causality + IRF.
```

### Why the shared CV matters (the central design idea)

Every level model is scored by `rolling_origin_cv` with the **same** cutoffs,
horizon, and target dates (`INITIAL=750`, `PERIOD=60`, `HORIZON=60` business
days → ~8 folds). That is what makes the MAE/MAPE/RMSE numbers directly
comparable. Each model only supplies a `forecast_fn(train, future_index)`; the
CV harness owns the rest. The CV returns per-point `errors` (indexed by
`(cutoff, date)`) so two models can be aligned point-by-point and fed to
`diebold_mariano`, which tests whether a MAE gap is *real* or just noise
(Harvey-Leybourne-Newbold small-sample correction, Newey-West long-run variance
truncated at `horizon-1`). A negative DM stat means the model beats the
benchmark; significance needs `p < 0.05`.

**GARCH is the exception:** it forecasts conditional variance, not the level, so
it has a parallel `rolling_origin_vol_cv` against realized volatility and is
compared to a constant-vol baseline — never mixed into the level table.

## Data flow & conventions

- Inputs: BCB SGS series **1** (USD/BRL) and **432** (SELIC), 5y window, via
  `download_bcb_data()` in `main.py` (uses `https://`).
- `data/` holds downloaded CSVs; `assets/` holds generated plots/CSVs. **Both
  are git-ignored** (along with `*.csv`/`*.png`). Don't commit data or outputs.
- Warning hygiene: each model silences only its known-benign warnings by
  category (e.g. `ValueWarning`, `DataScaleWarning`) and deliberately keeps
  `ConvergenceWarning` visible so a window that fails to fit isn't hidden. Do
  **not** add a blanket `warnings.filterwarnings("ignore")` — it was removed for
  exactly this reason.

## Findings (honest caveats — keep them honest)

With 8 folds + DM, no level model significantly beats the random walk (VAR's
edge is noise: DM −0.166, p 0.868). SELIC Granger-causes USD/BRL in-sample
(p 0.008) but that doesn't convert to detectable out-of-sample gains (weak UIP).
GARCH lands numerically below constant-vol but not significantly (p 0.615). With
only ~8 folds the DM test has low power: "not significant" means "no evidence",
not "equal". When reporting results, preserve that distinction.

## Versioning

Follow the repo-wide conventions in `../CLAUDE.md`: Conventional Commits (type in
English, description in Portuguese), SemVer in annotated git tags, and the
`CHANGELOG.md` in Keep a Changelog format (Portuguese). Mirrored to GitHub
(`origin`) and Codeberg.
