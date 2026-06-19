import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.sm_exceptions import ValueWarning
from datetime import timedelta
import warnings

from evaluation import rolling_origin_cv, HORIZON

# The date index carries no explicit frequency, so statsmodels emits a
# ValueWarning on every fit; silence just that, keeping real fit/convergence
# warnings visible.
warnings.simplefilter("ignore", ValueWarning)

MAX_LAGS = 10  # ceiling for AIC lag selection


def adf_test(series, name):
    result = adfuller(series.dropna(), autolag='AIC')
    verdict = " [stationary]" if result[1] < 0.05 else " [non-stationary]"
    print(f"  ADF Test — {name}: stat={result[0]:.4f}, p-value={result[1]:.4f}{verdict}")


def _build_diff(fx_level, rate_level):
    """Stationary inputs for the VAR: FX log-returns and the rate first-difference.

    FX is close to a random walk in levels, so we model log-differences (returns);
    the rate variable (the Brazil-US interest differential, or the SELIC level when
    Fed Funds is unavailable) is differenced once. Returned aligned and NaN-free,
    ready to feed VAR().
    """
    df = pd.DataFrame({
        'usd_brl_diff': np.log(fx_level).diff(),
        'rate_diff': rate_level.diff(),
    })
    return df.dropna()


def _select_lag(df_diff):
    """AIC-selected lag order, floored at 1 (VAR needs at least one lag)."""
    order = VAR(df_diff).select_order(maxlags=MAX_LAGS).selected_orders['aic']
    return max(int(order), 1)


def var_forecast(train, future_index, rate_series):
    """Multi-step USD/BRL level forecast from a bivariate VAR(FX, rate).

    The rate variable is the Brazil-US interest differential (SELIC - Fed Funds),
    or the SELIC level alone when Fed Funds is unavailable. Refits the VAR on every
    training window (closing over the aligned rate series), forecasts the
    differenced series `len(future_index)` steps ahead and inverts the
    log-differences back to a price level anchored at the last training observation.
    Signature matches `rolling_origin_cv` so VAR is scored on the same level
    cutoffs/horizon as the univariate models.
    """
    rate_train = rate_series.loc[train.index]
    df_diff = _build_diff(train, rate_train)

    lag = _select_lag(df_diff)
    var_fit = VAR(df_diff).fit(lag)

    steps = len(future_index)
    fc = var_fit.forecast(df_diff.values[-lag:], steps=steps)
    fx_diff_fc = fc[:, 0]  # usd_brl_diff is the first column
    return train.iloc[-1] * np.exp(np.cumsum(fx_diff_fc))


def run_var_analysis(fx_series, rate_series, rate_label="interest differential (SELIC - Fed Funds)"):
    """
    Runs the complete VAR analysis on USD/BRL jointly with an interest-rate variable.

    Args:
        fx_series (pd.Series): USD/BRL price level, datetime-indexed.
        rate_series (pd.Series): the rate variable (% p.a.) — the Brazil-US interest
            differential (SELIC - Fed Funds), or the SELIC level when Fed Funds is
            unavailable — datetime-indexed, already aligned to `fx_series`.
        rate_label (str): human-readable name of the rate variable, used in logs and
            plot titles.

    Returns:
        dict: level forecast accuracy metrics, comparable to the univariate models.
    """
    print(f"--- VAR Analysis: USD/BRL + {rate_label} ---")
    print("Economic basis: Uncovered Interest Rate Parity (UIP) — UIP is about the")
    print("Brazil-US rate *differential*, not the domestic policy rate alone.")
    print(f"Granger test: does the {rate_label} help predict BRL movements?\n")

    # Phase 1: Stationarity
    print("--- Phase 1: Stationarity ---")
    print(f"Aligned dataset: {len(fx_series)} observations from "
          f"{fx_series.index[0].date()} to {fx_series.index[-1].date()}")

    print("\nADF tests on levels:")
    adf_test(fx_series, 'USD/BRL (level)')
    adf_test(rate_series, f'{rate_label} (level)')

    df_diff = _build_diff(fx_series, rate_series)
    print("\nADF tests on differenced series:")
    adf_test(df_diff['usd_brl_diff'], 'USD/BRL log-diff')
    adf_test(df_diff['rate_diff'], f'{rate_label} first-diff')

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes[0, 0].plot(fx_series.index, fx_series.values)
    axes[0, 0].set_title('USD/BRL Exchange Rate (Level)')
    axes[0, 1].plot(rate_series.index, rate_series.values)
    axes[0, 1].set_title(f'{rate_label} — % p.a. (Level)')
    axes[1, 0].plot(df_diff.index, df_diff['usd_brl_diff'])
    axes[1, 0].set_title('USD/BRL Log-Difference (Returns)')
    axes[1, 1].plot(df_diff.index, df_diff['rate_diff'])
    axes[1, 1].set_title(f'{rate_label} First-Difference')
    plt.tight_layout()
    plt.savefig('assets/var_input_series.png')
    plt.close()
    print("Input series plot saved to 'assets/var_input_series.png'")

    # Phase 2: Estimation & Granger causality
    print("\n--- Phase 2: VAR Estimation & Granger Causality ---")
    optimal_lag = _select_lag(df_diff)
    print(f"Optimal lag order (AIC, floored at 1): {optimal_lag}")

    var_fit = VAR(df_diff).fit(optimal_lag)
    print(var_fit.summary())

    print(f"\nGranger Causality Test: {rate_label} -> USD/BRL")
    gc = var_fit.test_causality('usd_brl_diff', ['rate_diff'], kind='f')
    print(gc.summary())
    if gc.pvalue < 0.05:
        print(f"  -> {rate_label} Granger-causes USD/BRL (p < 0.05): past rates improve FX forecasts.")
    else:
        print(f"  -> No significant Granger causality (p >= 0.05): {rate_label} does not reliably predict BRL.")

    # Phase 3: Impulse response functions
    print("\n--- Phase 3: Impulse Response Functions ---")
    irf = var_fit.irf(periods=20)
    irf.plot(orth=True, figsize=(12, 8))
    plt.suptitle('Orthogonalized IRF: SELIC shock vs USD/BRL', y=1.01)
    plt.tight_layout()
    plt.savefig('assets/var_irf.png')
    plt.close()
    print("IRF plot saved to 'assets/var_irf.png'")

    # Phase 4: Forecast evaluation (shared rolling-origin CV, level target)
    print("\n--- Phase 4: Forecast Evaluation ---")
    cv = rolling_origin_cv(
        lambda tr, idx: var_forecast(tr, idx, rate_series),
        fx_series,
        label="VAR",
    )
    mae, mape, rmse = cv["mae"], cv["mape"], cv["rmse"]
    print(f"\nForecast Accuracy Metrics ({cv['n_folds']}-fold CV, horizon={HORIZON}):")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.4f}")
    print(f"RMSE: {rmse:.4f}")

    # Final held-out window (multi-step level forecast vs actual)
    test = fx_series.iloc[-HORIZON:]
    predictions = var_forecast(fx_series.iloc[:-HORIZON], test.index, rate_series)
    plt.figure(figsize=(12, 6))
    plt.plot(test.index, test.values, label='Actual')
    plt.plot(test.index, predictions, color='red', label='Forecast')
    plt.title(f'VAR Forecast vs Actual (last {HORIZON} days, multi-step)')
    plt.legend()
    plt.savefig('assets/var_forecast_plots.png')
    plt.close()
    print("Forecast plots saved to 'assets/var_forecast_plots.png'")

    # Phase 5: Forward forecast (one trading year ahead, level)
    print("\n--- Phase 5: Future Forecasting (1 Year Ahead) ---")
    horizon = 252
    fc = var_fit.forecast(df_diff.values[-optimal_lag:], steps=horizon)
    fc_df = pd.DataFrame(fc, columns=['usd_brl_diff_fc', 'rate_diff_fc'])

    last_date = df_diff.index[-1]
    fc_df.index = pd.date_range(start=last_date + timedelta(days=1), periods=horizon, freq='B')
    fc_df['usd_brl_forecast'] = fx_series.iloc[-1] * np.exp(fc_df['usd_brl_diff_fc'].cumsum())
    fc_df['rate_forecast'] = rate_series.iloc[-1] + fc_df['rate_diff_fc'].cumsum()
    fc_df[['usd_brl_forecast', 'rate_forecast']].to_csv('assets/var_future_forecast.csv')
    print("Future forecast saved to 'assets/var_future_forecast.csv'")

    history = fx_series[fx_series.index >= last_date - timedelta(days=180)]
    plt.figure(figsize=(14, 6))
    plt.plot(history.index, history.values, label='History (last 6 months)')
    plt.plot(fc_df.index, fc_df['usd_brl_forecast'], color='red', label='VAR Forecast')
    plt.title(f'VAR Forecast: USD/BRL until {fc_df.index[-1].date()}')
    plt.ylabel('USD/BRL')
    plt.legend()
    plt.grid(True)
    plt.savefig('assets/var_future_forecast_plot.png')
    plt.close()
    print("Future forecast plot saved to 'assets/var_future_forecast_plot.png'")

    return {
        "mae": mae,
        "mape": mape,
        "rmse": rmse,
        "errors": cv["errors"],
    }
