import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from datetime import timedelta

def run_ets_analysis(series):
    """
    Runs the complete ETS analysis on a time series.

    Args:
        series (pd.Series): The time series data.

    Returns:
        dict: A dictionary with the forecast accuracy metrics.
    """
    print("--- ETS Modeling for BRL/USD ---")

    # 1. Model Configuration and Training
    print("\n1. Configuring and Training ETS Model...")
    # Let's try an additive trend and additive seasonality model
    model = ExponentialSmoothing(
        series,
        trend='add',
        seasonal='add',
        seasonal_periods=12  # Assuming monthly seasonality
    ).fit()

    print(model.summary())

    # 2. Forecasting & Evaluation
    print("\n2. Forecasting & Evaluation...")
    train_size = int(len(series) - 60)
    train, test = series[0:train_size], series[train_size:len(series)]

    print(f"Training size: {len(train)}, Test size: {len(test)}")

    # Walk-forward validation is not standard for ETS, so we'll do a simple forecast
    predictions = model.forecast(len(test))

    mae = mean_absolute_error(test, predictions)
    mape = mean_absolute_percentage_error(test, predictions)
    rmse = np.sqrt(mean_squared_error(test, predictions))

    print(f"\nForecast Accuracy Metrics:")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.4f}")
    print(f"RMSE: {rmse:.4f}")

    # Plot Forecasts
    plt.figure(figsize=(12, 6))
    plt.plot(test.index, test.values, label='Actual')
    plt.plot(test.index, predictions, color='red', label='Forecast')
    plt.title('ETS Forecast vs Actual')
    plt.legend()
    plt.savefig('assets/ets_forecast_plots.png')
    print("\nForecast plots saved to 'assets/ets_forecast_plots.png'")

    # 3. Future Forecasting
    print("\n3. Future Forecasting...")
    
    # Retrain on full dataset
    final_model = ExponentialSmoothing(
        series,
        trend='add',
        seasonal='add',
        seasonal_periods=12
    ).fit()
    
    # Calculate steps to forecast
    last_date = series.index[-1]
    target_date = last_date + timedelta(days=365)
    
    # Create date range for forecast
    future_dates = pd.date_range(start=last_date + timedelta(days=1), end=target_date)
    steps = len(future_dates)
    
    print(f"Forecasting {steps} days from {future_dates[0].date()} to {future_dates[-1].date()}")
    
    # Get forecast
    forecast_values = final_model.forecast(steps)
    
    # Create DataFrame for results
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Forecast': forecast_values.values
    })
    forecast_df.set_index('Date', inplace=True)
    
    # Save to CSV
    forecast_df.to_csv('assets/ets_future_forecast.csv')
    print("Future forecast saved to 'assets/ets_future_forecast.csv'")
    
    # Plot Future Forecast
    plt.figure(figsize=(12, 6))
    # Plot last 6 months of history for context
    history_subset = series[series.index >= (last_date - timedelta(days=180))]
    plt.plot(history_subset.index, history_subset.values, label='History')
    
    plt.plot(forecast_df.index, forecast_df['Forecast'], color='red', label='Forecast')
    
    plt.title('BRL/USD ETS Forecast')
    plt.legend()
    plt.grid(True)
    plt.savefig('assets/ets_future_forecast_plot.png')
    print("Future forecast plot saved to 'assets/ets_future_forecast_plot.png'")

    return {
        "mae": mae,
        "mape": mape,
        "rmse": rmse
    }
