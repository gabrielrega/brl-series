import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from datetime import timedelta
import sys
import warnings

warnings.filterwarnings("ignore")


def main():
    print("--- GARCH Volatility Analysis for USD/BRL ---")
    print("Economic basis: Volatility clustering (Engle 1982, Bollerslev 1986)")
    print("Conditional variance of FX returns is predictable from past squared returns.\n")

    # Phase 1: Data Preparation & Return Calculation
    print("--- Phase 1: Data Preparation ---")
    try:
        df = pd.read_csv('usd_brl_history.csv')
        df['data'] = pd.to_datetime(df['data'])
        df.set_index('data', inplace=True)
        series = df['valor']
    except FileNotFoundError:
        print("Error: 'usd_brl_history.csv' not found. Please run download_data.py first.")
        sys.exit(1)

    # Compute percentage log returns: 100 * ln(P_t / P_{t-1})
    # Scaling by 100 is conventional for the arch library and avoids tiny numbers
    returns = 100 * np.log(series / series.shift(1)).dropna()
    returns.name = 'log_returns'

    print(f"Series length: {len(series)} observations")
    print(f"Returns computed: {len(returns)} observations")
    print(f"Return stats: mean={returns.mean():.4f}%, std={returns.std():.4f}%")

    # Test for ARCH effects (Lagrange Multiplier test)
    lm_stat, lm_pvalue, f_stat, f_pvalue = het_arch(returns, nlags=10)
    print(f"\nARCH-LM Test (lags=10): LM stat={lm_stat:.4f}, p-value={lm_pvalue:.4f}")
    if lm_pvalue < 0.05:
        print("  -> ARCH effects confirmed (p < 0.05): GARCH modeling is appropriate.")
    else:
        print("  -> No significant ARCH effects found (p >= 0.05).")

    # Plot returns and squared returns (visual evidence of volatility clustering)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    axes[0].plot(returns.index, returns.values)
    axes[0].set_title('USD/BRL Log Returns (%)')
    axes[0].set_ylabel('Return (%)')
    axes[1].plot(returns.index, returns.values ** 2)
    axes[1].set_title('Squared Returns — Volatility Clustering')
    axes[1].set_ylabel('Squared Return')
    plt.tight_layout()
    plt.savefig('garch_returns.png')
    plt.close()
    print("Returns plot saved to 'garch_returns.png'")

    # Phase 2: GARCH(1,1) Model Estimation
    print("\n--- Phase 2: GARCH(1,1) Estimation ---")
    # GARCH(1,1) with Student-t distribution to account for fat tails
    am = arch_model(returns, vol='Garch', p=1, q=1, dist='t', mean='Constant')
    try:
        res = am.fit(disp='off', options={'maxiter': 1000})
    except Exception as e:
        print(f"GARCH(1,1) with t-dist failed ({e}), retrying with Normal distribution...")
        am = arch_model(returns, vol='Garch', p=1, q=1, dist='Normal', mean='Constant')
        res = am.fit(disp='off', options={'maxiter': 1000})

    print(res.summary())

    alpha = res.params.get('alpha[1]', res.params.get('alpha1', 0))
    beta = res.params.get('beta[1]', res.params.get('beta1', 0))
    persistence = alpha + beta
    print(f"\nGARCH persistence (alpha + beta) = {persistence:.4f}")
    if persistence > 0.95:
        print("  -> High persistence: volatility shocks decay slowly (typical for FX).")
    else:
        print("  -> Moderate persistence: volatility shocks revert to mean more quickly.")

    # Phase 3: Diagnostics
    print("\n--- Phase 3: Diagnostics ---")
    std_resid = res.resid / res.conditional_volatility
    std_resid = std_resid.dropna()

    lb_resid = acorr_ljungbox(std_resid, lags=[10], return_df=True)
    lb_sq = acorr_ljungbox(std_resid ** 2, lags=[10], return_df=True)
    print("Ljung-Box on standardized residuals (should be insignificant):")
    print(lb_resid.to_string())
    print("\nLjung-Box on squared standardized residuals (should be insignificant):")
    print(lb_sq.to_string())

    # Diagnostics plot: standardized residuals + ACF of squared residuals
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    axes[0].plot(std_resid.index, std_resid.values)
    axes[0].axhline(0, color='black', linewidth=0.8)
    axes[0].set_title('Standardized Residuals')
    axes[0].set_ylabel('Std. Residual')
    axes[1].plot(std_resid.index, std_resid.values ** 2)
    axes[1].set_title('Squared Standardized Residuals (no clustering = good fit)')
    axes[1].set_ylabel('Squared Std. Residual')
    plt.tight_layout()
    plt.savefig('garch_diagnostics.png')
    plt.close()
    print("Diagnostics plot saved to 'garch_diagnostics.png'")

    # Conditional volatility plot (annualized)
    # Conditional volatility from arch is in % units (since returns were * 100)
    # Annualize: vol_annual = vol_daily * sqrt(252)
    cond_vol_annual = res.conditional_volatility * np.sqrt(252)

    plt.figure(figsize=(14, 5))
    plt.plot(cond_vol_annual.index, cond_vol_annual.values, color='steelblue')
    plt.title('GARCH(1,1) Conditional Volatility — Annualized (%)')
    plt.ylabel('Annualized Volatility (%)')
    plt.grid(True)
    plt.savefig('garch_volatility.png')
    plt.close()
    print("Conditional volatility plot saved to 'garch_volatility.png'")

    # Phase 4: Volatility Forecasting
    print("\n--- Phase 4: Volatility Forecast (1 Year Ahead) ---")
    horizon = 252  # 1 trading year
    forecasts = res.forecast(horizon=horizon, reindex=False)

    # forecasts.variance is in %-squared units (because returns were * 100)
    # Annualized volatility: sqrt(variance_daily) / 100 * sqrt(252)
    forecast_var = forecasts.variance.iloc[-1].values  # shape (horizon,)
    forecast_vol_annual = np.sqrt(forecast_var) / 100 * np.sqrt(252)

    last_date = returns.index[-1]
    forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=horizon, freq='B')

    forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'conditional_variance': forecast_var,
        'annualized_volatility': forecast_vol_annual,
    })
    forecast_df.set_index('Date', inplace=True)
    forecast_df.to_csv('garch_forecast.csv')
    print("Forecast saved to 'garch_forecast.csv'")

    mean_vol = forecast_vol_annual.mean() * 100
    print(f"Mean forecasted annualized volatility (next year): {mean_vol:.2f}%")

    # Plot volatility forecast
    last_90d = cond_vol_annual[cond_vol_annual.index >= last_date - timedelta(days=90)]
    plt.figure(figsize=(14, 5))
    plt.plot(last_90d.index, last_90d.values, label='In-sample (last 90 days)', color='steelblue')
    plt.plot(forecast_df.index, forecast_df['annualized_volatility'] * 100,
             label='Forecast', color='darkorange')
    plt.title('GARCH(1,1) Volatility Forecast — Annualized (%)')
    plt.ylabel('Annualized Volatility (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig('garch_forecast_plot.png')
    plt.close()
    print("Forecast plot saved to 'garch_forecast_plot.png'")


if __name__ == "__main__":
    main()
