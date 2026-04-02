import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
import sys
import logging

# Configure logging
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
logging.getLogger('prophet').setLevel(logging.WARNING)

def main():
    print("--- Prophet Modeling for BRL/USD ---")

    # 1. Data Preprocessing
    print("\n1. Loading and Preprocessing Data...")
    try:
        df = pd.read_csv('usd_brl_history.csv')
        # Prophet requires columns 'ds' and 'y'
        df = df.rename(columns={'data': 'ds', 'valor': 'y'})
        df['ds'] = pd.to_datetime(df['ds'])
        print(f"Data loaded: {len(df)} rows")
        print(df.head())
    except FileNotFoundError:
        print("Error: 'usd_brl_history.csv' not found.")
        sys.exit(1)

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
    # Create future dataframe: 1 year from last data point
    last_date = df['ds'].max()
    target_date = last_date + pd.DateOffset(years=1)
    days_to_predict = (target_date - last_date).days

    print(f"Forecasting {days_to_predict} days into the future (until {target_date.date()})...")
    
    future = model.make_future_dataframe(periods=days_to_predict)
    forecast = model.predict(future)
    
    # Save forecast
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv('prophet_forecast.csv', index=False)
    print("Forecast saved to 'prophet_forecast.csv'")

    # 5. Visualization
    print("\n5. Generating Visualizations...")
    
    # Forecast Plot
    fig1 = model.plot(forecast)
    plt.title('BRL/USD Prophet Forecast')
    plt.savefig('prophet_forecast_plot.png')
    plt.close()
    print("Forecast plot saved to 'prophet_forecast_plot.png'")

    # Components Plot
    fig2 = model.plot_components(forecast)
    plt.savefig('prophet_components.png')
    plt.close()
    print("Components plot saved to 'prophet_components.png'")

    # 6. Cross-Validation & Evaluation
    print("\n6. Running Cross-Validation (this may take time)...")
    # Initial training period: 3 years (~1095 days)
    # Period: 180 days
    # Horizon: 60 days
    try:
        df_cv = cross_validation(model, initial='1095 days', period='180 days', horizon='60 days')
        
        df_p = performance_metrics(df_cv)
        print("\nPerformance Metrics (Average):")
        print(df_p[['horizon', 'mae', 'mape', 'rmse']].mean())
        
        df_p.to_csv('prophet_cv_metrics.csv')
        print("CV metrics saved to 'prophet_cv_metrics.csv'")
        
        # Plot MAPE
        fig3 = plot_cross_validation_metric(df_cv, metric='mape')
        plt.title('Cross-Validation MAPE')
        plt.savefig('prophet_cv_mape.png')
        plt.close()
        print("CV MAPE plot saved to 'prophet_cv_mape.png'")
        
    except Exception as e:
        print(f"Cross-validation failed: {e}")
        print("Skipping CV metrics.")

if __name__ == "__main__":
    main()
