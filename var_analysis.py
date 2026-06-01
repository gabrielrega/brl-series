import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import adfuller
from datetime import timedelta
import warnings

from evaluation import rolling_origin_cv, HORIZON

warnings.filterwarnings("ignore")

MAX_LAGS = 10  # ceiling for AIC lag selection


def adf_test(series, name):
    result = adfuller(series.dropna(), autolag='AIC')
    verdict = " [stationary]" if result[1] < 0.05 else " [non-stationary]"
    print(f"  ADF Test — {name}: stat={result[0]:.4f}, p-value={result[1]:.4f}{verdict}")


def _build_diff(fx_level, selic_level):
    """Stationary inputs for the VAR: FX log-returns and SELIC first-difference.

    FX is close to a random walk in levels, so we model log-differences (returns);
    the policy rate is differenced once (change in rate). Returned aligned and
    NaN-free, ready to feed VAR().
    """
    df = pd.DataFrame({
        'usd_brl_diff': np.log(fx_level).diff(),
        'selic_diff': selic_level.diff(),
    })
    return df.dropna()


def _select_lag(df_diff):
    """AIC-selected lag order, floored at 1 (VAR needs at least one lag)."""
    order = VAR(df_diff).select_order(maxlags=MAX_LAGS).selected_orders['aic']
    return max(int(order), 1)


def var_forecast(train, future_index, selic_series):
    """Multi-step USD/BRL level forecast from a bivariate VAR(FX, SELIC).

    Refits the VAR on every training window (closing over the aligned SELIC series),
    forecasts the differenced series `len(future_index)` steps ahead and inverts the
    log-differences back to a price level anchored at the last training observation.
    Signature matches `rolling_origin_cv` so VAR is scored on the same level
    cutoffs/horizon as the univariate models.
    """
    selic_train = selic_series.loc[train.index]
    df_diff = _build_diff(train, selic_train)

    lag = _select_lag(df_diff)
    var_fit = VAR(df_diff).fit(lag)

    steps = len(future_index)
    fc = var_fit.forecast(df_diff.values[-lag:], steps=steps)
    fx_diff_fc = fc[:, 0]  # usd_brl_diff is the first column
    return train.iloc[-1] * np.exp(np.cumsum(fx_diff_fc))


def run_var_analysis(fx_series, selic_series):
    """
    Runs the complete VAR analysis on USD/BRL jointly with the SELIC policy rate.

    Args:
        fx_series (pd.Series): USD/BRL price level, datetime-indexed.
        selic_series (pd.Series): SELIC rate (% p.a.), datetime-indexed, already
            aligned to `fx_series` (same index).

    Returns:
        dict: level forecast accuracy metrics, comparable to the univariate models.
    """
    print("--- VAR Analysis: USD/BRL + SELIC Interest Rate ---")
    print("Economic basis: Uncovered Interest Rate Parity (UIP).")
    print("Granger test: does the SELIC rate help predict BRL movements?\n")

    # Phase 1: Stationarity
    print("--- Phase 1: Stationarity ---")
    print(f"Aligned dataset: {len(fx_series)} observations from "
          f"{fx_series.index[0].date()} to {fx_series.index[-1].date()}")

    print("\nADF tests on levels:")
    adf_test(fx_series, 'USD/BRL (level)')
    adf_test(selic_series, 'SELIC (level)')

    df_diff = _build_diff(fx_series, selic_series)
    print("\nADF tests on differenced series:")
    adf_test(df_diff['usd_brl_diff'], 'USD/BRL log-diff')
    adf_test(df_diff['selic_diff'], 'SELIC first-diff')

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes[0, 0].plot(fx_series.index, fx_series.values)
    axes[0, 0].set_title('USD/BRL Exchange Rate (Level)')
    axes[0, 1].plot(selic_series.index, selic_series.values)
    axes[0, 1].set_title('SELIC Rate — % p.a. (Level)')
    axes[1, 0].plot(df_diff.index, df_diff['usd_brl_diff'])
    axes[1, 0].set_title('USD/BRL Log-Difference (Returns)')
    axes[1, 1].plot(df_diff.index, df_diff['selic_diff'])
    axes[1, 1].set_title('SELIC First-Difference')
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

    print("\nGranger Causality Test: SELIC -> USD/BRL")
    gc = var_fit.test_causality('usd_brl_diff', ['selic_diff'], kind='f')
    print(gc.summary())
    if gc.pvalue < 0.05:
        print("  -> SELIC Granger-causes USD/BRL (p < 0.05): past rates improve FX forecasts.")
    else:
        print("  -> No significant Granger causality (p >= 0.05): SELIC does not reliably predict BRL.")

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
        lambda tr, idx: var_forecast(tr, idx, selic_series),
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
    predictions = var_forecast(fx_series.iloc[:-HORIZON], test.index, selic_series)
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
    fc_df = pd.DataFrame(fc, columns=['usd_brl_diff_fc', 'selic_diff_fc'])

    last_date = df_diff.index[-1]
    fc_df.index = pd.date_range(start=last_date + timedelta(days=1), periods=horizon, freq='B')
    fc_df['usd_brl_forecast'] = fx_series.iloc[-1] * np.exp(fc_df['usd_brl_diff_fc'].cumsum())
    fc_df['selic_rate_forecast'] = selic_series.iloc[-1] + fc_df['selic_diff_fc'].cumsum()
    fc_df[['usd_brl_forecast', 'selic_rate_forecast']].to_csv('assets/var_future_forecast.csv')
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
    }
