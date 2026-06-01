import os
import requests
import pandas as pd
from datetime import datetime, timedelta
import io

from arima_analysis import run_arima_analysis
from prophet_analysis import run_prophet_analysis
from ets_analysis import run_ets_analysis

def download_bcb_data(series_id, start_date, end_date):
    """
    Downloads time series data from the Brazilian Central Bank API.

    Args:
        series_id (int): The ID of the time series to download.
        start_date (datetime): The start date for the data.
        end_date (datetime): The end date for the data.

    Returns:
        pd.DataFrame: A DataFrame with the time series data.
    """
    # Format dates as dd/MM/yyyy
    start_str = start_date.strftime('%d/%m/%Y')
    end_str = end_date.strftime('%d/%m/%Y')

    # Construct URL with parameters
    url = f"http://api.bcb.gov.br/dados/serie/bcdata.sgs.{series_id}/dados?formato=json&dataInicial={start_str}&dataFinal={end_str}"

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    print(f"Requesting URL: {url}")

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        data = response.json()

        df = pd.DataFrame(data)
        df['data'] = pd.to_datetime(df['data'], format='%d/%m/%Y')
        df['valor'] = pd.to_numeric(df['valor'])

        return df

    except Exception as e:
        print(f"Error downloading data: {e}")
        if 'response' in locals():
            print(f"Response status: {response.status_code}")
            print(f"Response text snippet: {response.text[:200]}")
        return None

def main():
    """
    Main function to run the time series analysis.
    """
    # Create output directories if they don't exist
    for directory in ('data', 'assets'):
        os.makedirs(directory, exist_ok=True)

    # Download data
    print("Downloading data...")
    # Series ID 1 is the exchange rate (USD/BRL)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    df = download_bcb_data(1, start_date, end_date)

    if df is not None:
        output_file = "data/usd_brl_history.csv"
        df.to_csv(output_file, index=False)
        print(f"Data saved to {output_file}")
    else:
        print("Failed to download data.")
        return

    # Load data for analysis
    df_prophet = df.rename(columns={'data': 'ds', 'valor': 'y'})
    
    df_series = df.set_index('data')['valor']

    # Run analyses
    arima_metrics = run_arima_analysis(df_series)
    prophet_metrics = run_prophet_analysis(df_prophet)
    ets_metrics = run_ets_analysis(df_series)

    # Compare models
    print("\n--- Model Comparison ---")
    if arima_metrics:
        print("\nARIMA Metrics:")
        print(f"MAE: {arima_metrics['mae']:.4f}")
        print(f"MAPE: {arima_metrics['mape']:.4f}")
        print(f"RMSE: {arima_metrics['rmse']:.4f}")

    if prophet_metrics:
        print("\nProphet Metrics:")
        print(f"MAE: {prophet_metrics['mae']:.4f}")
        print(f"MAPE: {prophet_metrics['mape']:.4f}")
        print(f"RMSE: {prophet_metrics['rmse']:.4f}")

    if ets_metrics:
        print("\nETS Metrics:")
        print(f"MAE: {ets_metrics['mae']:.4f}")
        print(f"MAPE: {ets_metrics['mape']:.4f}")
        print(f"RMSE: {ets_metrics['rmse']:.4f}")

if __name__ == "__main__":
    main()