import requests
import pandas as pd
from datetime import datetime, timedelta
import io

def download_bcb_data(series_id):
    # Calculate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    
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

if __name__ == "__main__":
    # Series ID 1 is the exchange rate (USD/BRL)
    print("Downloading data...")
    df = download_bcb_data(1)
    
    if df is not None:
        output_file = "usd_brl_history.csv"
        df.to_csv(output_file, index=False)
        print(f"Data saved to {output_file}")
        print(df.head())
        print(df.tail())
    else:
        print("Failed to download data.")
