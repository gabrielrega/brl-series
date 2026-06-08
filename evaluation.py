import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)


def naive_rw_forecast(train, future_index):
    """Random-walk baseline: forecast the last observed value for the whole horizon.

    For an exchange rate (close to a random walk) this is the benchmark every model
    must beat to justify its complexity. Plugged into the same rolling_origin_cv as
    the real models so the comparison is apples-to-apples.
    """
    return np.full(len(future_index), train.iloc[-1])

# Shared cross-validation parameters (in observations / business days).
# Using the same scheme for every model is what makes their metrics comparable.
INITIAL = 750   # ~3 years of training before the first forecast
PERIOD = 60     # step between cutoffs; == HORIZON tiles the test windows
HORIZON = 60    # ~3 months forecast horizon evaluated at each cutoff


def rolling_origin_cv(forecast_fn, series, initial=INITIAL, period=PERIOD,
                      horizon=HORIZON, label=""):
    """
    Rolling-origin cross-validation shared by every model so their error
    metrics are directly comparable: same cutoffs, same horizon, same target
    dates and the same metric maths.

    Args:
        forecast_fn: callable(train_series, future_index) -> array-like aligned
            to future_index. Each model builds and fits its own estimator here.
        series (pd.Series): full time series, datetime-indexed.
        initial (int): size of the first training window (observations).
        period (int): step between successive cutoffs (observations).
        horizon (int): forecast horizon evaluated at each cutoff (observations).
        label (str): model name, used only for logging.

    Returns:
        dict with pooled 'mae', 'mape', 'rmse' across all folds, 'n_folds' and
        'errors' (a pd.Series of actual-minus-pred indexed by a (cutoff, date)
        MultiIndex, so two models can be aligned point-by-point for a
        Diebold-Mariano test). None if no fold could be evaluated.
    """
    n = len(series)
    actuals, preds, keys = [], [], []
    folds = 0

    cutoff = initial
    while cutoff + horizon <= n:
        train = series.iloc[:cutoff]
        test = series.iloc[cutoff:cutoff + horizon]
        try:
            yhat = np.asarray(forecast_fn(train, test.index), dtype=float)
        except Exception as e:
            print(f"  [{label}] fold at cutoff={cutoff} failed: {e}")
            cutoff += period
            continue

        if len(yhat) < len(test):
            print(f"  [{label}] fold at cutoff={cutoff} returned too few points; skipping")
            cutoff += period
            continue

        actuals.extend(test.values.tolist())
        preds.extend(yhat[:len(test)].tolist())
        keys.extend((cutoff, d) for d in test.index)
        folds += 1
        cutoff += period

    if not preds:
        return None

    actuals = np.array(actuals)
    preds = np.array(preds)
    errors = pd.Series(actuals - preds,
                       index=pd.MultiIndex.from_tuples(keys, names=["cutoff", "date"]))
    print(f"  [{label}] {folds} folds, {len(preds)} pooled predictions")
    return {
        "mae": mean_absolute_error(actuals, preds),
        "mape": mean_absolute_percentage_error(actuals, preds),
        "rmse": np.sqrt(mean_squared_error(actuals, preds)),
        "n_folds": folds,
        "errors": errors,
    }


# --- Volatility evaluation -------------------------------------------------
# GARCH forecasts the *conditional variance*, not the level, so it cannot share
# the level MAE/MAPE table above. These helpers give volatility models their own
# apples-to-apples target: realized volatility over the next `horizon` days,
# scored on the same rolling-origin cutoffs.

TRADING_DAYS = 252  # business days per year, the usual annualization factor


def log_returns(level_series):
    """Percentage log-returns, 100*ln(P_t / P_{t-1}) — the standard GARCH input.

    Scaling by 100 keeps the optimizer away from tiny numbers (the convention the
    `arch` library expects) and makes the resulting volatility read as a percent.
    """
    return (100 * np.log(level_series / level_series.shift(1))).dropna()


def realized_vol(returns_window):
    """Annualized realized volatility (%) implied by a window of %-log-returns.

    Uses the root-mean-square of returns so it lines up directly with a variance
    forecast (mean predicted variance), then scales by sqrt(252) to annualize.
    """
    r = np.asarray(returns_window, dtype=float)
    return float(np.sqrt(np.mean(r ** 2)) * np.sqrt(TRADING_DAYS))


def constant_vol_forecast(train_returns, horizon):
    """Naive volatility baseline: next-`horizon` vol = trailing annualized vol.

    The volatility analogue of the random-walk level baseline (`naive_rw_forecast`):
    a GARCH model must beat this to justify modelling the conditional variance.
    `horizon` is unused — a flat forecast doesn't depend on how far ahead we look.
    """
    return realized_vol(train_returns)


def rolling_origin_vol_cv(vol_forecast_fn, returns, initial=INITIAL, period=PERIOD,
                          horizon=HORIZON, label=""):
    """Rolling-origin CV for volatility forecasts (annualized %, one value per fold).

    Mirrors `rolling_origin_cv` but the target is realized volatility over the next
    `horizon` days rather than the level path, so volatility models are comparable
    to each other on the same cutoffs and horizon.

    Args:
        vol_forecast_fn: callable(train_returns, horizon) -> single annualized-vol
            number forecast for the upcoming `horizon` days.
        returns (pd.Series): %-log-returns, datetime-indexed.
        initial, period, horizon (int): same scheme as the level CV.
        label (str): model name, used only for logging.

    Returns:
        dict with pooled 'mae', 'mape', 'rmse' (in annualized vol points),
        'n_folds' and 'errors' (a pd.Series of realized-minus-forecast indexed by
        cutoff, one per fold, for a Diebold-Mariano test). None if no fold could
        be evaluated.
    """
    n = len(returns)
    actuals, preds, keys = [], [], []
    folds = 0

    cutoff = initial
    while cutoff + horizon <= n:
        train = returns.iloc[:cutoff]
        test = returns.iloc[cutoff:cutoff + horizon]
        try:
            vhat = float(vol_forecast_fn(train, horizon))
        except Exception as e:
            print(f"  [{label}] fold at cutoff={cutoff} failed: {e}")
            cutoff += period
            continue

        actuals.append(realized_vol(test))
        preds.append(vhat)
        keys.append(cutoff)
        folds += 1
        cutoff += period

    if not preds:
        return None

    actuals = np.array(actuals)
    preds = np.array(preds)
    errors = pd.Series(actuals - preds, index=pd.Index(keys, name="cutoff"))
    print(f"  [{label}] {folds} folds, {len(preds)} pooled vol forecasts")
    return {
        "mae": mean_absolute_error(actuals, preds),
        "mape": mean_absolute_percentage_error(actuals, preds),
        "rmse": np.sqrt(mean_squared_error(actuals, preds)),
        "n_folds": folds,
        "errors": errors,
    }


# --- Significance: Diebold-Mariano -----------------------------------------
# A lower MAE doesn't mean a model is *significantly* better — with only a
# handful of folds the gap can be noise. Diebold & Mariano (1995) test whether
# two forecasts have equal expected loss on the same target points.


def diebold_mariano(errors_model, errors_bench, horizon=HORIZON, loss="abs"):
    """Diebold-Mariano test of equal predictive accuracy (HLN small-sample corrected).

    H0: `errors_model` and `errors_bench` carry the same expected loss. The two
    error series are aligned on their common index first (the (cutoff, date) keys
    produced by the rolling-origin CV), so only points both models forecast are
    compared. A **negative** statistic means the model beats the benchmark; the
    p-value (two-sided, Student-t with n-1 df) says whether that gap is real.

    Multi-step forecasts overlap, so the loss-differential is serially
    correlated: the long-run variance uses a Newey-West/Bartlett estimator
    truncated at `horizon-1` lags, and the statistic carries the Harvey,
    Leybourne & Newbold (1997) finite-sample correction.

    Args:
        errors_model, errors_bench (pd.Series): actual-minus-forecast errors,
            indexed compatibly (e.g. both from rolling_origin_cv).
        horizon (int): forecast horizon, sets the autocorrelation truncation.
        loss ('abs'|'sq'): absolute loss (matches the MAE table) or squared.

    Returns:
        dict with 'stat', 'p_value', 'n' (aligned points) and 'mean_diff'
        (mean loss differential, model minus benchmark; negative => model wins),
        or None if there is no overlap or the variance is degenerate.
    """
    common = errors_model.index.intersection(errors_bench.index)
    if len(common) < 3:
        return None

    em = np.asarray(errors_model.loc[common], dtype=float)
    eb = np.asarray(errors_bench.loc[common], dtype=float)
    lm = np.abs(em) if loss == "abs" else em ** 2
    lb = np.abs(eb) if loss == "abs" else eb ** 2

    d = lm - lb
    T = len(d)
    dbar = d.mean()

    # Newey-West long-run variance, Bartlett weights, truncated at horizon-1
    # (the MA order of optimal h-step forecast errors); guaranteed non-negative.
    h = max(int(horizon), 1)
    gamma0 = np.mean((d - dbar) ** 2)
    var = gamma0
    for k in range(1, min(h, T)):
        cov = np.mean((d[k:] - dbar) * (d[:-k] - dbar))
        var += 2.0 * (1.0 - k / h) * cov
    var_dbar = var / T
    if var_dbar <= 0:
        # Degenerate loss-differential. Zero variance with zero mean means the
        # two forecasts are indistinguishable (no evidence against H0); a nonzero
        # mean with zero variance is pathological, so we decline to test.
        if abs(dbar) < 1e-12:
            return {"stat": 0.0, "p_value": 1.0, "n": T, "mean_diff": 0.0}
        return None

    dm = dbar / np.sqrt(var_dbar)
    # Harvey, Leybourne & Newbold (1997) finite-sample correction
    corr = np.sqrt(max((T + 1 - 2 * h + h * (h - 1) / T) / T, 0.0))
    dm_hln = dm * corr
    p_value = 2 * student_t.cdf(-abs(dm_hln), df=T - 1)
    return {"stat": dm_hln, "p_value": p_value, "n": T, "mean_diff": dbar}
