import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from datetime import datetime, timedelta
import sys
import logging

from evaluation import rolling_origin_cv, HORIZON

# Configure logging
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
logging.getLogger('prophet').setLevel(logging.WARNING)


def _build_prophet():
    """Single source of truth for the Prophet configuration used everywhere."""
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05,
        seasonality_mode='additive',
    )
    model.add_country_holidays(country_name='BR')
    return model


def prophet_forecast(train, future_index):
    """Multi-step forecast used by the shared cross-validation.

    Fits a fresh Prophet on `train` (a date-indexed Series) and predicts exactly
    at the held-out dates in `future_index`, so it lines up with the same target
    business days used to score ARIMA and ETS.
    """
    dfp = train.reset_index()
    dfp.columns = ['ds', 'y']
    model = _build_prophet()
    model.fit(dfp)
    future = pd.DataFrame({'ds': pd.to_datetime(future_index)})
    return model.predict(future)['yhat'].values

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
    # Brazil holidays added inside _build_prophet (shared with the CV folds).
    model = _build_prophet()

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

    # 6. Cross-Validation & Evaluation (shared rolling-origin CV, same as ARIMA/ETS)
    print("\n6. Running Cross-Validation (this may take time)...")
    series = df.set_index('ds')['y']
    cv = rolling_origin_cv(prophet_forecast, series, label="Prophet")

    if cv is None:
        print("Cross-validation produced no folds. Skipping CV metrics.")
        return None

    print(f"\nForecast Accuracy Metrics ({cv['n_folds']}-fold CV, horizon={HORIZON}):")
    print(f"MAE: {cv['mae']:.4f}")
    print(f"MAPE: {cv['mape']:.4f}")
    print(f"RMSE: {cv['rmse']:.4f}")

    return {
        "mae": cv["mae"],
        "mape": cv["mape"],
        "rmse": cv["rmse"],
    }
