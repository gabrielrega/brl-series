import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from arch.utility.exceptions import DataScaleWarning, StartingValueWarning
from datetime import timedelta
import warnings

# Silence only the benign arch diagnostics about data scale / starting values
# (returns are already scaled by 100); ConvergenceWarning stays visible so a
# window that fails to fit is not hidden — fit_garch falls back to Normal there.
warnings.simplefilter("ignore", DataScaleWarning)
warnings.simplefilter("ignore", StartingValueWarning)

from evaluation import (
    log_returns,
    rolling_origin_vol_cv,
    constant_vol_forecast,
    TRADING_DAYS,
    HORIZON,
)

warnings.filterwarnings("ignore")


def fit_garch(returns):
    """Single source of truth for the GARCH configuration used everywhere.

    GARCH(1,1) with a Student-t innovation to capture the fat tails of FX returns
    (Engle 1982, Bollerslev 1986). Falls back to a Normal distribution if the
    t-distribution optimization fails to converge on a given training window.
    """
    am = arch_model(returns, vol='Garch', p=1, q=1, dist='t', mean='Constant')
    try:
        return am.fit(disp='off', options={'maxiter': 1000})
    except Exception:
        am = arch_model(returns, vol='Garch', p=1, q=1, dist='Normal', mean='Constant')
        return am.fit(disp='off', options={'maxiter': 1000})


def garch_forecast(train_returns, horizon):
    """Annualized volatility (%) forecast over the next `horizon` days.

    Fits a GARCH(1,1) on the training returns, forecasts the daily conditional
    variances for the horizon and collapses them to a single annualized number,
    matching `evaluation.realized_vol` so the rolling-origin CV compares like with
    like. This is the volatility analogue of the level multi-step forecasts.
    """
    res = fit_garch(train_returns)
    fc = res.forecast(horizon=horizon, reindex=False)
    daily_var = fc.variance.iloc[-1].values  # %-squared, one per day in the horizon
    return float(np.sqrt(daily_var.mean()) * np.sqrt(TRADING_DAYS))


def run_garch_analysis(series):
    """
    Runs the complete GARCH volatility analysis on a price level series.

    Unlike the level models (ARIMA/ETS/Prophet/VAR), GARCH targets the conditional
    variance of returns, so it is evaluated against realized volatility on its own
    rolling-origin CV rather than the level MAE/MAPE table.

    Args:
        series (pd.Series): the price level time series (USD/BRL), datetime-indexed.

    Returns:
        dict: volatility forecast accuracy metrics (annualized vol points).
    """
    print("--- GARCH Volatility Analysis for USD/BRL ---")
    print("Economic basis: volatility clustering (Engle 1982, Bollerslev 1986).")
    print("The conditional variance of FX returns is predictable from past squared returns.\n")

    # Phase 1: Returns & ARCH effects
    print("--- Phase 1: Returns & ARCH Effects ---")
    returns = log_returns(series)
    returns.name = 'log_returns'
    print(f"Series length: {len(series)} observations")
    print(f"Returns computed: {len(returns)} observations")
    print(f"Return stats: mean={returns.mean():.4f}%, std={returns.std():.4f}%")

    lm_stat, lm_pvalue, _, _ = het_arch(returns, nlags=10)
    print(f"\nARCH-LM Test (lags=10): LM stat={lm_stat:.4f}, p-value={lm_pvalue:.4f}")
    if lm_pvalue < 0.05:
        print("  -> ARCH effects confirmed (p < 0.05): GARCH modeling is appropriate.")
    else:
        print("  -> No significant ARCH effects found (p >= 0.05).")

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    axes[0].plot(returns.index, returns.values)
    axes[0].set_title('USD/BRL Log Returns (%)')
    axes[0].set_ylabel('Return (%)')
    axes[1].plot(returns.index, returns.values ** 2)
    axes[1].set_title('Squared Returns — Volatility Clustering')
    axes[1].set_ylabel('Squared Return')
    plt.tight_layout()
    plt.savefig('assets/garch_returns.png')
    plt.close()
    print("Returns plot saved to 'assets/garch_returns.png'")

    # Phase 2: GARCH(1,1) estimation on the full sample
    print("\n--- Phase 2: GARCH(1,1) Estimation ---")
    res = fit_garch(returns)
    print(res.summary())

    alpha = res.params.get('alpha[1]', res.params.get('alpha1', 0))
    beta = res.params.get('beta[1]', res.params.get('beta1', 0))
    persistence = alpha + beta
    print(f"\nGARCH persistence (alpha + beta) = {persistence:.4f}")
    if persistence > 0.95:
        print("  -> High persistence: volatility shocks decay slowly (typical for FX).")
    else:
        print("  -> Moderate persistence: shocks revert to the mean more quickly.")

    # Phase 3: Diagnostics
    print("\n--- Phase 3: Diagnostics ---")
    std_resid = (res.resid / res.conditional_volatility).dropna()
    lb_resid = acorr_ljungbox(std_resid, lags=[10], return_df=True)
    lb_sq = acorr_ljungbox(std_resid ** 2, lags=[10], return_df=True)
    print("Ljung-Box on standardized residuals (should be insignificant):")
    print(lb_resid.to_string())
    print("\nLjung-Box on squared standardized residuals (should be insignificant):")
    print(lb_sq.to_string())

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    axes[0].plot(std_resid.index, std_resid.values)
    axes[0].axhline(0, color='black', linewidth=0.8)
    axes[0].set_title('Standardized Residuals')
    axes[0].set_ylabel('Std. Residual')
    axes[1].plot(std_resid.index, std_resid.values ** 2)
    axes[1].set_title('Squared Standardized Residuals (no clustering = good fit)')
    axes[1].set_ylabel('Squared Std. Residual')
    plt.tight_layout()
    plt.savefig('assets/garch_diagnostics.png')
    plt.close()
    print("Diagnostics plot saved to 'assets/garch_diagnostics.png'")

    cond_vol_annual = res.conditional_volatility * np.sqrt(TRADING_DAYS)
    plt.figure(figsize=(14, 5))
    plt.plot(cond_vol_annual.index, cond_vol_annual.values, color='steelblue')
    plt.title('GARCH(1,1) Conditional Volatility — Annualized (%)')
    plt.ylabel('Annualized Volatility (%)')
    plt.grid(True)
    plt.savefig('assets/garch_volatility.png')
    plt.close()
    print("Conditional volatility plot saved to 'assets/garch_volatility.png'")

    # Phase 4: Evaluation against realized volatility (shared rolling-origin CV)
    print("\n--- Phase 4: Volatility Forecast Evaluation ---")
    cv = rolling_origin_vol_cv(garch_forecast, returns, label="GARCH")
    baseline = rolling_origin_vol_cv(constant_vol_forecast, returns, label="Const Vol")

    print(f"\nRealized-volatility CV ({cv['n_folds']}-fold, horizon={HORIZON}, "
          f"annualized vol points):")
    print(f"{'Model':<12} {'MAE':>8} {'MAPE':>8} {'RMSE':>8}")
    for name, m in (("Const Vol", baseline), ("GARCH", cv)):
        print(f"{name:<12} {m['mae']:>8.4f} {m['mape']:>8.4f} {m['rmse']:>8.4f}")

    # Phase 5: Forward volatility forecast (one trading year ahead)
    print("\n--- Phase 5: Volatility Forecast (1 Year Ahead) ---")
    horizon = TRADING_DAYS
    forecasts = res.forecast(horizon=horizon, reindex=False)
    forecast_var = forecasts.variance.iloc[-1].values
    forecast_vol_annual = np.sqrt(forecast_var) * np.sqrt(TRADING_DAYS)  # annualized %

    last_date = returns.index[-1]
    forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=horizon, freq='B')
    forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'conditional_variance': forecast_var,
        'annualized_volatility': forecast_vol_annual,
    }).set_index('Date')
    forecast_df.to_csv('assets/garch_future_forecast.csv')
    print("Future forecast saved to 'assets/garch_future_forecast.csv'")
    print(f"Mean forecasted annualized volatility (next year): "
          f"{forecast_vol_annual.mean():.2f}%")

    last_90d = cond_vol_annual[cond_vol_annual.index >= last_date - timedelta(days=90)]
    plt.figure(figsize=(14, 5))
    plt.plot(last_90d.index, last_90d.values, label='In-sample (last 90 days)', color='steelblue')
    plt.plot(forecast_df.index, forecast_df['annualized_volatility'],
             label='Forecast', color='darkorange')
    plt.title('GARCH(1,1) Volatility Forecast — Annualized (%)')
    plt.ylabel('Annualized Volatility (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig('assets/garch_future_forecast_plot.png')
    plt.close()
    print("Future forecast plot saved to 'assets/garch_future_forecast_plot.png'")

    return {
        "mae": cv["mae"],
        "mape": cv["mape"],
        "rmse": cv["rmse"],
        "baseline": baseline,
    }
