import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import adfuller
from datetime import timedelta
import sys
import warnings

warnings.filterwarnings("ignore")


def adf_test(series, name):
    result = adfuller(series.dropna(), autolag='AIC')
    print(f"  ADF Test — {name}: stat={result[0]:.4f}, p-value={result[1]:.4f}", end="")
    print(" [stationary]" if result[1] < 0.05 else " [non-stationary]")


def main():
    print("--- VAR Analysis: USD/BRL + SELIC Interest Rate ---")
    print("Economic basis: Uncovered Interest Rate Parity (UIP)")
    print("Expected FX changes equal the interest rate differential (Brazil vs. US).")
    print("Granger causality test: does SELIC help predict BRL movements?\n")

    # Phase 1: Data Loading & Stationarity
    print("--- Phase 1: Data Loading & Stationarity ---")

    try:
        df_fx = pd.read_csv('usd_brl_history.csv')
        df_fx['data'] = pd.to_datetime(df_fx['data'])
    except FileNotFoundError:
        print("Error: 'usd_brl_history.csv' not found. Please run download_data.py first.")
        sys.exit(1)

    try:
        df_selic = pd.read_csv('bcb_series_432.csv')
        df_selic['data'] = pd.to_datetime(df_selic['data'])
    except FileNotFoundError:
        print("Error: 'bcb_series_432.csv' not found.")
        print("Please run: python download_data.py --all")
        sys.exit(1)

    # Inner merge on date to align both daily series
    df = pd.merge(df_fx, df_selic, on='data', how='inner', suffixes=('_fx', '_selic'))
    df = df.rename(columns={'valor_fx': 'usd_brl', 'valor_selic': 'selic_rate'})
    df.set_index('data', inplace=True)
    df = df.sort_index()

    print(f"Merged dataset: {len(df)} observations from {df.index[0].date()} to {df.index[-1].date()}")

    # Stationarity check on levels
    print("\nADF tests on levels:")
    adf_test(df['usd_brl'], 'USD/BRL (level)')
    adf_test(df['selic_rate'], 'SELIC (level)')

    # Transform to stationary series
    # FX: log-difference (percentage returns)
    df['usd_brl_diff'] = np.log(df['usd_brl']).diff()
    # SELIC: first difference (change in rate)
    df['selic_diff'] = df['selic_rate'].diff()
    df_diff = df[['usd_brl_diff', 'selic_diff']].dropna()

    print("\nADF tests on differenced series:")
    adf_test(df_diff['usd_brl_diff'], 'USD/BRL log-diff')
    adf_test(df_diff['selic_diff'], 'SELIC first-diff')

    # Plot levels and differences
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes[0, 0].plot(df.index, df['usd_brl'])
    axes[0, 0].set_title('USD/BRL Exchange Rate (Level)')
    axes[0, 1].plot(df.index, df['selic_rate'])
    axes[0, 1].set_title('SELIC Rate — % p.a. (Level)')
    axes[1, 0].plot(df_diff.index, df_diff['usd_brl_diff'])
    axes[1, 0].set_title('USD/BRL Log-Difference (Returns)')
    axes[1, 1].plot(df_diff.index, df_diff['selic_diff'])
    axes[1, 1].set_title('SELIC First-Difference')
    plt.tight_layout()
    plt.savefig('var_input_series.png')
    plt.close()
    print("Input series plot saved to 'var_input_series.png'")

    # Phase 2: VAR Model Selection & Estimation
    print("\n--- Phase 2: VAR Model Selection & Estimation ---")
    model = VAR(df_diff)
    lag_order = model.select_order(maxlags=10)
    print(lag_order.summary())

    optimal_lag = lag_order.selected_orders['aic']
    if optimal_lag == 0:
        print("AIC selected 0 lags — using lag=1 as minimum for VAR.")
        optimal_lag = 1
    print(f"Optimal lag order (AIC): {optimal_lag}")

    var_fit = model.fit(optimal_lag)
    print(var_fit.summary())

    # Granger causality: does SELIC help predict USD/BRL?
    print("\nGranger Causality Test: SELIC -> USD/BRL")
    gc = var_fit.test_causality('usd_brl_diff', ['selic_diff'], kind='f')
    print(gc.summary())
    if gc.pvalue < 0.05:
        print("  -> SELIC Granger-causes USD/BRL (p < 0.05): past SELIC rates improve FX forecasts.")
    else:
        print("  -> No significant Granger causality (p >= 0.05): SELIC does not reliably predict BRL.")

    # Phase 3: Diagnostics & Impulse Response Functions
    print("\n--- Phase 3: Impulse Response Functions ---")
    irf = var_fit.irf(periods=20)
    fig = irf.plot(orth=True, figsize=(12, 8))
    plt.suptitle('Orthogonalized IRF: Effect of SELIC Shock on USD/BRL (and vice versa)', y=1.01)
    plt.tight_layout()
    plt.savefig('var_irf.png')
    plt.close()
    print("IRF plot saved to 'var_irf.png'")
    print("  Interpretation: A positive shock to SELIC (rate hike) should strengthen BRL")
    print("  (BRL appreciates = USD/BRL falls) under Uncovered Interest Rate Parity.")

    # Phase 4: Forecasting
    print("\n--- Phase 4: Forecast (1 Year Ahead) ---")
    horizon = 252  # 1 trading year
    last_values = df_diff.values[-optimal_lag:]

    fc = var_fit.forecast(last_values, steps=horizon)
    fc_df = pd.DataFrame(fc, columns=['usd_brl_diff_fc', 'selic_diff_fc'])

    last_date = df_diff.index[-1]
    forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=horizon, freq='B')
    fc_df.index = forecast_dates

    # Invert log-difference to get USD/BRL level forecast
    last_fx_level = df['usd_brl'].iloc[-1]
    fc_df['usd_brl_forecast'] = last_fx_level * np.exp(fc_df['usd_brl_diff_fc'].cumsum())

    # Invert first-difference to get SELIC level forecast
    last_selic_level = df['selic_rate'].iloc[-1]
    fc_df['selic_rate_forecast'] = last_selic_level + fc_df['selic_diff_fc'].cumsum()

    fc_df[['usd_brl_forecast', 'selic_rate_forecast']].to_csv('var_forecast.csv')
    print("Forecast saved to 'var_forecast.csv'")

    # Plot USD/BRL forecast vs recent history
    history_start = last_date - timedelta(days=180)
    history = df['usd_brl'][df.index >= history_start]

    plt.figure(figsize=(14, 6))
    plt.plot(history.index, history.values, label='History (last 6 months)')
    plt.plot(fc_df.index, fc_df['usd_brl_forecast'], color='red', label='VAR Forecast')
    plt.title(f'VAR Forecast: USD/BRL until {fc_df.index[-1].date()}')
    plt.ylabel('USD/BRL')
    plt.legend()
    plt.grid(True)
    plt.savefig('var_forecast_plot.png')
    plt.close()
    print("Forecast plot saved to 'var_forecast_plot.png'")


if __name__ == "__main__":
    main()
