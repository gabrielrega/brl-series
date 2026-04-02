import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import jarque_bera
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from datetime import datetime, timedelta
import warnings
import sys

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def check_stationarity(timeseries):
    print("Results of Dickey-Fuller Test:")
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput[f'Critical Value ({key})'] = value
    print(dfoutput)

    print("\nResults of KPSS Test:")
    kpsstest = kpss(timeseries, regression='c', nlags="auto")
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
    for key,value in kpsstest[3].items():
        kpss_output[f'Critical Value ({key})'] = value
    print(kpss_output)

def evaluate_arima_model(train, test, order):
    history = [x for x in train]
    predictions = list()
    # Walk-forward validation
    for t in range(len(test)):
        model = ARIMA(history, order=order)
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
    
    mae = mean_absolute_error(test, predictions)
    mape = mean_absolute_percentage_error(test, predictions)
    rmse = np.sqrt(mean_squared_error(test, predictions))
    
    return predictions, mae, mape, rmse

def main():
    # Load data
    try:
        df = pd.read_csv('usd_brl_history.csv')
        df['data'] = pd.to_datetime(df['data'])
        df.set_index('data', inplace=True)
        series = df['valor']
    except FileNotFoundError:
        print("Error: 'usd_brl_history.csv' not found. Please run download_data.py first.")
        sys.exit(1)

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
    plt.savefig('stationarity_plots.png')
    plt.close()
    print("\nStationarity plots saved to 'stationarity_plots.png'")

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
            except Exception:
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
    plt.savefig('residual_plots.png')
    plt.close()
    print("\nResidual plots saved to 'residual_plots.png'")
    
    # Ljung-Box Test
    lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
    print("\nLjung-Box Test for Autocorrelation:")
    print(lb_test)
    
    # Jarque-Bera Test
    jb_score, p_value = jarque_bera(residuals)
    print(f"\nJarque-Bera Test: Score={jb_score:.2f}, p-value={p_value:.4f}")

    # Phase 3: Forecasting & Evaluation
    print("\n--- Phase 3: Forecasting & Evaluation ---")
    
    train_size = int(len(series) - 60)
    train, test = series[0:train_size], series[train_size:len(series)]
    
    print(f"Training size: {len(train)}, Test size: {len(test)}")
    
    predictions, mae, mape, rmse = evaluate_arima_model(train, test.values, best_order)
    
    print(f"\nForecast Accuracy Metrics:")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    # Plot Forecasts
    plt.figure(figsize=(12, 6))
    plt.plot(test.index, test.values, label='Actual')
    plt.plot(test.index, predictions, color='red', label='Forecast')
    plt.title('ARIMA Forecast vs Actual (Walk-Forward)')
    plt.legend()
    plt.savefig('forecast_plots.png')
    plt.close()
    print("\nForecast plots saved to 'forecast_plots.png'")

    # Phase 4: Future Forecasting
    # Retrain on full dataset
    final_model = ARIMA(series, order=best_order)
    final_model_fit = final_model.fit()

    # Calculate steps to forecast: 1 year from last data point
    last_date = series.index[-1]
    target_date = last_date + pd.DateOffset(years=1)

    print(f"\n--- Phase 4: Future Forecasting (Until {target_date.date()}) ---")

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
    forecast_df.to_csv('future_forecast.csv')
    print("Future forecast saved to 'future_forecast.csv'")
    
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
    
    plt.title(f'BRL/USD Forecast until {target_date.date()}')
    plt.legend()
    plt.grid(True)
    plt.savefig('future_forecast_plot.png')
    plt.close()
    print("Future forecast plot saved to 'future_forecast_plot.png'")

if __name__ == "__main__":
    main()

