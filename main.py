import os
import requests
import pandas as pd
from datetime import datetime, timedelta
import io

from arima_analysis import run_arima_analysis
from prophet_analysis import run_prophet_analysis
from ets_analysis import run_ets_analysis
from garch_analysis import run_garch_analysis
from var_analysis import run_var_analysis
from evaluation import rolling_origin_cv, naive_rw_forecast, diebold_mariano, HORIZON

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
    url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{series_id}/dados?formato=json&dataInicial={start_str}&dataFinal={end_str}"

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


def parse_fred_csv(text):
    """Parse a FRED `fredgraph.csv` payload into a ['data', 'valor'] DataFrame.

    Kept separate from the HTTP call so the parsing (the part prone to silent
    breakage) is unit-testable offline. FRED's first column is a date (header
    'observation_date' on current exports, 'DATE' on older ones) and the second is
    the series itself; missing observations are encoded as '.', which become NaN
    and are dropped.
    """
    df = pd.read_csv(io.StringIO(text))
    date_col, val_col = df.columns[0], df.columns[1]
    df = df.rename(columns={date_col: 'data', val_col: 'valor'})
    df['data'] = pd.to_datetime(df['data'])
    df['valor'] = pd.to_numeric(df['valor'], errors='coerce')
    return df.dropna(subset=['valor'])[['data', 'valor']]


def download_fred_data(series_id, start_date, end_date):
    """
    Downloads a daily series from FRED as CSV (no API key required).

    Used for the US policy rate (Fed Funds) so the VAR can use the Brazil-US
    interest *differential* (SELIC - Fed Funds), which is what Uncovered Interest
    Rate Parity is actually about, rather than the SELIC level alone.

    Args:
        series_id (str): FRED series id (e.g. 'DFF' for the Federal Funds rate).
        start_date (datetime), end_date (datetime): inclusive date window.

    Returns:
        pd.DataFrame with columns ['data', 'valor'] (datetime, float), or None.
    """
    url = (f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
           f"&cosd={start_date:%Y-%m-%d}&coed={end_date:%Y-%m-%d}")

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    print(f"Requesting URL: {url}")

    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return parse_fred_csv(response.text)

    except Exception as e:
        print(f"Error downloading FRED data: {e}")
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
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)

    # Series 1 is the exchange rate (USD/BRL); series 432 is the SELIC target rate.
    # The Fed Funds rate (FRED 'DFF', the US overnight policy rate, the natural
    # counterpart of the SELIC) lets the VAR use the Brazil-US interest
    # *differential* — which is what Uncovered Interest Rate Parity is about.
    df = download_bcb_data(1, start_date, end_date)
    df_selic = download_bcb_data(432, start_date, end_date)
    df_fedfunds = download_fred_data('DFF', start_date, end_date)

    if df is None:
        print("Failed to download exchange-rate data.")
        return

    output_file = "data/usd_brl_history.csv"
    df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")
    if df_selic is not None:
        df_selic.to_csv("data/selic_history.csv", index=False)
        print("SELIC data saved to data/selic_history.csv")
    if df_fedfunds is not None:
        df_fedfunds.to_csv("data/fedfunds_history.csv", index=False)
        print("Fed Funds data saved to data/fedfunds_history.csv")

    # Load data for analysis
    df_prophet = df.rename(columns={'data': 'ds', 'valor': 'y'})

    df_series = df.set_index('data')['valor']

    # Naive random-walk baseline (the benchmark every level model must beat)
    print("\nComputing naive random-walk baseline...")
    naive_metrics = rolling_origin_cv(naive_rw_forecast, df_series, label="Naive RW")

    # Run univariate level analyses
    arima_metrics = run_arima_analysis(df_series)
    prophet_metrics = run_prophet_analysis(df_prophet)
    ets_metrics = run_ets_analysis(df_series)

    # GARCH targets volatility, not the level: it is scored separately below.
    garch_metrics = run_garch_analysis(df_series)

    # VAR needs a policy-rate series aligned to the exchange rate (inner merge on
    # date). When the Fed Funds rate is available, the VAR's rate variable is the
    # Brazil-US interest differential (SELIC - Fed Funds), the UIP-consistent
    # quantity; otherwise it falls back to the SELIC level alone.
    var_metrics = None
    if df_selic is not None:
        merged = pd.merge(df, df_selic, on='data', how='inner', suffixes=('_fx', '_selic'))
        merged = merged.rename(columns={'valor_fx': 'usd_brl', 'valor_selic': 'selic'}).set_index('data').sort_index()

        if df_fedfunds is not None:
            # Align the daily Fed Funds rate onto the FX business-day index,
            # forward-filling across US-holiday gaps (a policy rate is flat
            # between moves), then take the differential.
            fed = df_fedfunds.set_index('data')['valor'].sort_index()
            merged['fed_funds'] = fed.reindex(merged.index, method='ffill')
            merged = merged.dropna(subset=['fed_funds'])
            merged['rate_diff'] = merged['selic'] - merged['fed_funds']
            var_metrics = run_var_analysis(
                merged['usd_brl'], merged['rate_diff'],
                rate_label="interest differential (SELIC - Fed Funds)",
            )
        else:
            print("\nFed Funds unavailable; VAR falls back to the SELIC level.")
            var_metrics = run_var_analysis(
                merged['usd_brl'], merged['selic'], rate_label="SELIC",
            )
    else:
        print("\nSkipping VAR: SELIC series unavailable.")

    # Level comparison (all scored on the same rolling-origin CV, horizon=HORIZON).
    # DM = Diebold-Mariano vs the Naive RW baseline on the absolute-error loss:
    # a negative stat with p < 0.05 means the model significantly beats the RW.
    print(f"\n--- Level Model Comparison ({HORIZON}-day horizon CV) ---")
    print(f"{'Model':<12} {'MAE':>8} {'MAPE':>8} {'RMSE':>8} {'DM vs RW':>9} {'p':>7}")
    for name, metrics in (
        ("Naive RW", naive_metrics),
        ("ARIMA", arima_metrics),
        ("ETS", ets_metrics),
        ("Prophet", prophet_metrics),
        ("VAR", var_metrics),
    ):
        if not metrics:
            print(f"{name:<12} {'—':>8} {'—':>8} {'—':>8} {'—':>9} {'—':>7}")
            continue
        if name == "Naive RW":
            dm_cell, p_cell = f"{'—':>9}", f"{'—':>7}"
        else:
            dm = diebold_mariano(metrics["errors"], naive_metrics["errors"], loss="abs")
            dm_cell = f"{dm['stat']:>9.3f}" if dm else f"{'—':>9}"
            p_cell = f"{dm['p_value']:>7.3f}" if dm else f"{'—':>7}"
        print(f"{name:<12} {metrics['mae']:>8.4f} {metrics['mape']:>8.4f} "
              f"{metrics['rmse']:>8.4f} {dm_cell} {p_cell}")

    # Volatility comparison (separate target: realized vol, annualized %).
    # DM here is GARCH vs the constant-vol baseline (one forecast per fold, so
    # horizon=1 for the autocorrelation correction).
    print(f"\n--- Volatility Model Comparison ({HORIZON}-day horizon CV) ---")
    print(f"{'Model':<12} {'MAE':>8} {'MAPE':>8} {'RMSE':>8} {'DM vs CV':>9} {'p':>7}")
    baseline = garch_metrics.get("baseline") if garch_metrics else None
    vol_rows = []
    if garch_metrics:
        vol_rows.append(("Const Vol", baseline))
        vol_rows.append(("GARCH", garch_metrics))
    for name, metrics in vol_rows:
        if not metrics:
            print(f"{name:<12} {'—':>8} {'—':>8} {'—':>8} {'—':>9} {'—':>7}")
            continue
        if name == "GARCH" and baseline and baseline.get("errors") is not None:
            dm = diebold_mariano(metrics["errors"], baseline["errors"], horizon=1, loss="abs")
            dm_cell = f"{dm['stat']:>9.3f}" if dm else f"{'—':>9}"
            p_cell = f"{dm['p_value']:>7.3f}" if dm else f"{'—':>7}"
        else:
            dm_cell, p_cell = f"{'—':>9}", f"{'—':>7}"
        print(f"{name:<12} {metrics['mae']:>8.4f} {metrics['mape']:>8.4f} "
              f"{metrics['rmse']:>8.4f} {dm_cell} {p_cell}")

if __name__ == "__main__":
    main()