import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
from datetime import datetime, timedelta
import sys
import logging

# Configure logging
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
logging.getLogger('prophet').setLevel(logging.WARNING)

def run_prophet_analysis(df):
    """
    Runs the complete Prophet analysis on a time series.

    Args:
        df (pd.DataFrame): The time series data, with 'ds' and 'y' columns.

    Returns:
        dict: A dictionary with the forecast accuracy metrics.
    """
    print("--- Prophet Modeling for BRL/USD ---")

    # 2. Model Configuration
    print("\n2. Configuring Prophet Model...")
    # Daily data, so daily_seasonality=False (unless intraday), weekly and yearly are relevant.
    # Adding Brazil holidays.
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05, # Default is 0.05, increasing makes it more flexible
        seasonality_mode='additive' # Additive is usually good for exchange rates unless variance grows with trend
    )
    model.add_country_holidays(country_name='BR')

    # 3. Model Training
    print("\n3. Training Model...")
    model.fit(df)

    # 4. Forecasting
    print("\n4. Generating Forecast...")
    # Create future dataframe for 1 year
    last_date = df['ds'].max()
    days_to_predict = 365
    
    print(f"Forecasting {days_to_predict} days into the future (until {(last_date + timedelta(days=days_to_predict)).date()})...")
    
    future = model.make_future_dataframe(periods=days_to_predict)
    forecast = model.predict(future)
    
    # Save forecast
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv('assets/prophet_forecast.csv', index=False)
    print("Forecast saved to 'assets/prophet_forecast.csv'")

    # 5. Visualization
    print("\n5. Generating Visualizations...")
    
    # Forecast Plot
    fig1 = model.plot(forecast)
    plt.title('BRL/USD Prophet Forecast')
    plt.savefig('assets/prophet_forecast_plot.png')
    print("Forecast plot saved to 'assets/prophet_forecast_plot.png'")
    
    # Components Plot
    fig2 = model.plot_components(forecast)
    plt.savefig('assets/prophet_components.png')
    print("Components plot saved to 'assets/prophet_components.png'")

    # 6. Cross-Validation & Evaluation
    print("\n6. Running Cross-Validation (this may take time)...")
    # Initial training period: 3 years (~1095 days)
    # Period: 180 days
    # Horizon: 60 days
    try:
        df_cv = cross_validation(model, initial='1095 days', period='180 days', horizon='60 days')
        
        df_p = performance_metrics(df_cv)
        print("\nPerformance Metrics (Average):")
        metrics = df_p[['horizon', 'mae', 'mape', 'rmse']].mean()
        print(metrics)
        
        df_p.to_csv('assets/prophet_cv_metrics.csv')
        print("CV metrics saved to 'assets/prophet_cv_metrics.csv'")
        
        # Plot MAPE
        fig3 = plot_cross_validation_metric(df_cv, metric='mape')
        plt.title('Cross-Validation MAPE')
        plt.savefig('assets/prophet_cv_mape.png')
        print("CV MAPE plot saved to 'assets/prophet_cv_mape.png'")
        
        return {
            "mae": metrics['mae'],
            "mape": metrics['mape'],
            "rmse": metrics['rmse']
        }
        
    except Exception as e:
        print(f"Cross-validation failed: {e}")
        print("Skipping CV metrics.")
        return None
