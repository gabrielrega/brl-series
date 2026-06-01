import numpy as np
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
PERIOD = 120    # ~6 months between successive cutoffs
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
        dict with pooled 'mae', 'mape', 'rmse' across all folds and 'n_folds',
        or None if no fold could be evaluated.
    """
    n = len(series)
    actuals, preds = [], []
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
        folds += 1
        cutoff += period

    if not preds:
        return None

    actuals = np.array(actuals)
    preds = np.array(preds)
    print(f"  [{label}] {folds} folds, {len(preds)} pooled predictions")
    return {
        "mae": mean_absolute_error(actuals, preds),
        "mape": mean_absolute_percentage_error(actuals, preds),
        "rmse": np.sqrt(mean_squared_error(actuals, preds)),
        "n_folds": folds,
    }
