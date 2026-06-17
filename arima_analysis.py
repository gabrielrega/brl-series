import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import jarque_bera
from datetime import timedelta
import warnings

from statsmodels.tools.sm_exceptions import ValueWarning, InterpolationWarning

from evaluation import rolling_origin_cv, HORIZON

# Silence only the known-benign noise, not everything: the date index has no
# explicit frequency (ValueWarning) and KPSS reports p-values at the table
# bounds (InterpolationWarning). ConvergenceWarning is left visible on purpose
# so a model that genuinely fails to fit still surfaces.
warnings.simplefilter("ignore", ValueWarning)
warnings.simplefilter("ignore", InterpolationWarning)

def check_stationarity(timeseries):
    print("Results of Dickey-Fuller Test:")
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
    
    print("\nResults of KPSS Test:")
    kpsstest = kpss(timeseries, regression='c', nlags="auto")
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
    for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = value
    print(kpss_output)

def arima_forecast(train, future_index, order):
    """Multi-step forecast used by the shared cross-validation and final-window plot.

    Fits an ARIMA(order) on `train` and forecasts len(future_index) steps ahead,
    aligned positionally to the held-out observations.
    """
    model_fit = ARIMA(train, order=order).fit()
    return model_fit.get_forecast(steps=len(future_index)).predicted_mean.values

def run_arima_analysis(series):
    """
    Runs the complete ARIMA analysis on a time series.

    Args:
        series (pd.Series): The time series data.

    Returns:
        dict: A dictionary with the forecast accuracy metrics.
    """
    # Phase 1: Identification
    print("--- Phase 1: Model Identification ---")
    
    # Stationarity Tests
    print("\nStationarity Check (Original Series):")
    check_stationarity(series)
    
    # Differencing
    diff_series = series.diff().dropna()
    print("\nStationarity Check (1st Differenced Series):")
    check_stationarity(diff_series)
    
    # Plots
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    
    # Original
    axes[0, 0].plot(series)
    axes[0, 0].set_title('Original Series')
    plot_acf(series, ax=axes[0, 1])
    
    # 1st Diff
    axes[1, 0].plot(diff_series)
    axes[1, 0].set_title('1st Differenced Series')
    plot_acf(diff_series, ax=axes[1, 1])
    plot_pacf(diff_series, ax=axes[2, 0])
    
    plt.tight_layout()
    plt.savefig('assets/stationarity_plots.png')
    print("\nStationarity plots saved to 'assets/stationarity_plots.png'")

    # Grid Search for p, d, q (Simplified)
    # Based on differencing, d=1 seems likely. Let's check p and q.
    print("\nGrid Search for ARIMA parameters (AIC):")
    best_aic = float("inf")
    best_order = None
    
    # Small grid for demonstration
    for p in range(3):
        for q in range(3):
            try:
                model = ARIMA(series, order=(p, 1, q))
                results = model.fit()
                print(f'ARIMA({p},1,{q}) - AIC:{results.aic:.2f}')
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_order = (p, 1, q)
            except Exception as e:
                print(f'ARIMA({p},1,{q}) - skipped ({type(e).__name__})')
                continue
                
    print(f"\nBest ARIMA Order: {best_order}")

    # Phase 2: Estimation
    print("\n--- Phase 2: Model Estimation ---")
    model = ARIMA(series, order=best_order)
    model_fit = model.fit()
    print(model_fit.summary())
    
    # Residual Analysis
    residuals = pd.DataFrame(model_fit.resid)
    
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    residuals.plot(title="Residuals", ax=ax[0])
    residuals.plot(kind='kde', title='Density', ax=ax[1])
    plt.savefig('assets/residual_plots.png')
    print("\nResidual plots saved to 'assets/residual_plots.png'")
    
    # Ljung-Box Test
    lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
    print("\nLjung-Box Test for Autocorrelation:")
    print(lb_test)
    
    # Jarque-Bera Test
    jb_score, p_value = jarque_bera(residuals)
    print(f"\nJarque-Bera Test: Score={jb_score:.2f}, p-value={p_value:.4f}")

    # Phase 3: Forecasting & Evaluation (shared rolling-origin cross-validation)
    print("\n--- Phase 3: Forecasting & Evaluation ---")

    cv = rolling_origin_cv(
        lambda tr, idx: arima_forecast(tr, idx, best_order),
        series,
        label="ARIMA",
    )
    mae, mape, rmse = cv["mae"], cv["mape"], cv["rmse"]

    print(f"\nForecast Accuracy Metrics ({cv['n_folds']}-fold CV, horizon={HORIZON}):")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.4f}")
    print(f"RMSE: {rmse:.4f}")

    # Plot the final held-out window (multi-step forecast vs actual)
    test = series.iloc[-HORIZON:]
    predictions = arima_forecast(series.iloc[:-HORIZON], test.index, best_order)
    plt.figure(figsize=(12, 6))
    plt.plot(test.index, test.values, label='Actual')
    plt.plot(test.index, predictions, color='red', label='Forecast')
    plt.title(f'ARIMA Forecast vs Actual (last {HORIZON} days, multi-step)')
    plt.legend()
    plt.savefig('assets/forecast_plots.png')
    print("\nForecast plots saved to 'assets/forecast_plots.png'")

    # Phase 4: Future Forecasting (one year beyond the last observation)
    # Retrain on full dataset
    final_model = ARIMA(series, order=best_order)
    final_model_fit = final_model.fit()

    # Calculate steps to forecast
    last_date = series.index[-1]
    target_date = last_date + timedelta(days=365)
    print(f"\n--- Phase 4: Future Forecasting (until {target_date.date()}) ---")
    
    # Create date range for forecast
    future_dates = pd.date_range(start=last_date + timedelta(days=1), end=target_date)
    steps = len(future_dates)
    
    print(f"Forecasting {steps} days from {future_dates[0].date()} to {future_dates[-1].date()}")
    
    # Get forecast and confidence intervals
    forecast_result = final_model_fit.get_forecast(steps=steps)
    forecast_values = forecast_result.predicted_mean
    conf_int = forecast_result.conf_int()
    
    # Create DataFrame for results
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Forecast': forecast_values.values,
        'Lower CI': conf_int.iloc[:, 0].values,
        'Upper CI': conf_int.iloc[:, 1].values
    })
    forecast_df.set_index('Date', inplace=True)
    
    # Save to CSV
    forecast_df.to_csv('assets/future_forecast.csv')
    print("Future forecast saved to 'assets/future_forecast.csv'")
    
    # Plot Future Forecast
    plt.figure(figsize=(12, 6))
    # Plot last 6 months of history for context
    history_subset = series[series.index >= (last_date - timedelta(days=180))]
    plt.plot(history_subset.index, history_subset.values, label='History')
    
    plt.plot(forecast_df.index, forecast_df['Forecast'], color='red', label='Forecast')
    plt.fill_between(forecast_df.index, 
                     forecast_df['Lower CI'], 
                     forecast_df['Upper CI'], 
                     color='pink', alpha=0.3, label='95% CI')
    
    plt.title(f'BRL/USD ARIMA Forecast: next {steps} days (until {target_date.date()})')
    plt.legend()
    plt.grid(True)
    plt.savefig('assets/future_forecast_plot.png')
    print("Future forecast plot saved to 'assets/future_forecast_plot.png'")

    return {
        "mae": mae,
        "mape": mape,
        "rmse": rmse,
        "errors": cv["errors"],
    }
