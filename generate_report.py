"""
generate_report.py -- USD/BRL Monthly Forecast Report

Loads pre-computed forecast CSVs, fetches BCB Focus consensus, and assembles
a self-contained HTML report with embedded charts for email delivery.

Run after all analysis scripts have been executed:
    python download_data.py --all
    python arima_analysis.py
    python prophet_analysis.py
    python garch_analysis.py
    python var_analysis.py
    python generate_report.py
"""

import base64
import io
import sys
import warnings
from datetime import datetime, timedelta

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy.stats import norm, skew, kurtosis, probplot
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings('ignore')

REPORT_FILE = 'brl_report.html'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fig_to_img(fig):
    """Embed a matplotlib figure as a base64 PNG <img> tag."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    return (
        '<img src="data:image/png;base64,' + b64 + '" '
        'style="max-width:100%;margin:12px 0;display:block;">'
    )


def load_forecast(path):
    """Load a forecast CSV; return None with a warning if missing."""
    try:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    except FileNotFoundError:
        print(f'  Warning: {path} not found.')
        return None


def warn_box(msg):
    return f'<p class="warn">{msg}</p>'


def df_to_html(df):
    return df.to_html(classes='data-table', border=0)


# ---------------------------------------------------------------------------
# BCB Focus consensus
# ---------------------------------------------------------------------------

def fetch_focus_consensus():
    """
    Fetch BCB Focus Report annual consensus forecasts for USD/BRL (Cambio).

    Returns a DataFrame indexed by year-end date with columns:
        Media, Mediana, DesvioPadrao, Minimo, Maximo, Data (survey date)
    Uses the most recent survey. Returns None on failure.
    """
    url = (
        'https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/'
        'ExpectativaMercadoAnual(Indicador=@Indicador)'
        '?@Indicador=%27C%C3%A2mbio%27'
        '&$top=12'
        '&$orderby=Data%20desc%2CDataReferencia%20asc'
        '&$format=json'
        '&$select=Data%2CDataReferencia%2CMedia%2CMediana%2CDesvioPadrao%2CMinimo%2CMaximo'
    )
    try:
        resp = requests.get(url, timeout=30, headers={'Accept': 'application/json'})
        resp.raise_for_status()
        values = resp.json().get('value', [])
        if not values:
            print('  Warning: Focus API returned no data.')
            return None
        df = pd.DataFrame(values)
        df['Data'] = pd.to_datetime(df['Data'])
        # DataReferencia is the forecast year as a string, e.g. '2026'
        df['DataReferencia'] = (
            pd.to_datetime(df['DataReferencia'], format='%Y') + pd.offsets.YearEnd(0)
        )
        # Keep only the latest survey date
        df = df[df['Data'] == df['Data'].max()].copy()
        df = df.set_index('DataReferencia').sort_index()
        for col in ['Media', 'Mediana', 'DesvioPadrao', 'Minimo', 'Maximo']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        print(f'  Focus: {len(df)} year(s) from survey dated {df["Data"].iloc[0].date()}')
        return df
    except Exception as e:
        print(f'  Warning: Could not fetch Focus consensus: {e}')
        return None


# ---------------------------------------------------------------------------
# Section: Executive summary
# ---------------------------------------------------------------------------

def section_summary(df_hist, focus_df):
    last_price = df_hist['valor'].iloc[-1]
    last_date  = df_hist.index[-1].strftime('%Y-%m-%d')

    arima_fc = load_forecast('future_forecast.csv')
    var_fc   = load_forecast('var_forecast.csv')

    def end_val(fc, col):
        return f'{fc[col].iloc[-1]:.3f}' if fc is not None and col in fc.columns else 'N/A'

    focus_str = 'Unavailable'
    if focus_df is not None and 'Mediana' in focus_df.columns:
        future = focus_df[focus_df.index > df_hist.index[-1]]
        if len(future):
            yr = future.index[0].year
            focus_str = f'{future["Mediana"].iloc[0]:.3f}  (year-end {yr} consensus)'

    rows = {
        'Latest USD/BRL rate':         f'{last_price:.4f}  (as of {last_date})',
        'ARIMA 1-year-ahead forecast': end_val(arima_fc, 'Forecast'),
        'VAR 1-year-ahead forecast':   end_val(var_fc, 'usd_brl_forecast'),
        'BCB Focus consensus':         focus_str,
    }
    tbl = pd.DataFrame.from_dict(rows, orient='index', columns=['Value'])
    return (
        '<h2>Executive Summary</h2>'
        + df_to_html(tbl)
    )


# ---------------------------------------------------------------------------
# Section 1: Historical data exploration
# ---------------------------------------------------------------------------

def section_exploration(df):
    returns = (np.log(df['valor']).diff() * 100).dropna()

    # Chart A: price with 60-day rolling mean/band + returns
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    roll = df['valor'].rolling(60)
    axes[0].plot(df.index, df['valor'], lw=0.9, color='#1f77b4', label='USD/BRL')
    axes[0].plot(df.index, roll.mean(), color='orange', lw=1.5, label='60-day MA')
    axes[0].fill_between(
        df.index,
        roll.mean() - 1.5 * roll.std(),
        roll.mean() + 1.5 * roll.std(),
        alpha=0.15, color='orange', label='+/- 1.5 sigma band',
    )
    axes[0].set_title('USD/BRL Rate with 60-day Rolling Mean +/- 1.5 sigma')
    axes[0].set_ylabel('BRL per USD')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    thr = 2 * returns.std()
    axes[1].plot(returns.index, returns.values, lw=0.7, color='steelblue')
    axes[1].axhline(0,    color='black', lw=0.5)
    axes[1].axhline( thr, color='red', lw=0.8, ls='--', alpha=0.7, label='+2 sigma')
    axes[1].axhline(-thr, color='red', lw=0.8, ls='--', alpha=0.7, label='-2 sigma')
    axes[1].set_title('Daily Log Returns (%)')
    axes[1].set_ylabel('Return (%)')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    chart_a = fig_to_img(fig)

    # Chart B: return distribution vs Normal + Q-Q
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].hist(returns.values, bins=60, density=True, color='steelblue', alpha=0.7)
    x = np.linspace(returns.min(), returns.max(), 300)
    axes[0].plot(x, norm.pdf(x, returns.mean(), returns.std()),
                 'r--', lw=1.5, label='Normal fit')
    axes[0].set_title('Return Distribution vs Normal')
    axes[0].set_xlabel('Return (%)')
    axes[0].legend()

    probplot(returns.values, plot=axes[1])
    axes[1].set_title('Q-Q Plot (returns vs Normal)')
    plt.tight_layout()
    chart_b = fig_to_img(fig)

    # Chart C: annual summary bar
    yearly = df['valor'].resample('YE').agg(['mean', 'min', 'max', 'last'])
    fig, ax = plt.subplots(figsize=(13, 4))
    years = yearly.index.year
    ax.fill_between(years, yearly['min'], yearly['max'],
                    alpha=0.25, color='steelblue', label='Annual range')
    ax.plot(years, yearly['mean'],  marker='o', color='steelblue', label='Annual mean')
    ax.plot(years, yearly['last'],  marker='s', color='orange', ls='--', label='Year-end close')
    ax.set_title('Annual Summary: Mean, Year-end Close, and Range')
    ax.set_ylabel('BRL per USD')
    ax.set_xticks(years)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    chart_c = fig_to_img(fig)

    # Summary statistics table
    adf_level  = adfuller(df['valor'],  autolag='AIC')
    adf_return = adfuller(returns,       autolag='AIC')
    ann_vol      = returns.std() * np.sqrt(252)
    total_return = (df['valor'].iloc[-1] / df['valor'].iloc[0] - 1) * 100
    ann_return   = ((1 + total_return / 100) ** (252 / len(df)) - 1) * 100

    stats = {
        'Start date':                  df.index[0].strftime('%Y-%m-%d'),
        'End date':                    df.index[-1].strftime('%Y-%m-%d'),
        'Observations':                str(len(df)),
        'Current rate':                f'{df["valor"].iloc[-1]:.4f}',
        'Period low':                  f'{df["valor"].min():.4f}',
        'Period high':                 f'{df["valor"].max():.4f}',
        'Period mean':                 f'{df["valor"].mean():.4f}',
        '5-year total return':         f'{total_return:+.1f}%',
        'Annualised return':           f'{ann_return:+.1f}% p.a.',
        'Annualised volatility':       f'{ann_vol:.2f}%',
        'Return skewness':             f'{skew(returns):.3f}',
        'Return excess kurtosis':      f'{kurtosis(returns):.3f}',
        'ADF p-value (level)':         f'{adf_level[1]:.4f}  ({"stationary" if adf_level[1] < 0.05 else "non-stationary"})',
        'ADF p-value (log-diff)':      f'{adf_return[1]:.4f}  ({"stationary" if adf_return[1] < 0.05 else "non-stationary"})',
    }
    stats_df = pd.DataFrame.from_dict(stats, orient='index', columns=['Value'])

    return (
        '<h2>1. Historical Data Exploration</h2>'
        '<p>Source: Brazil Central Bank (BCB) API &mdash; Series 1 (official USD/BRL daily rate). '
        'This section uses only historical data; no forecast data appears here.</p>'
        + chart_a + chart_b + chart_c
        + '<h4>Summary Statistics</h4>'
        + df_to_html(stats_df)
    )


# ---------------------------------------------------------------------------
# Section 2: ARIMA
# ---------------------------------------------------------------------------

def section_arima(df_hist):
    fc = load_forecast('future_forecast.csv')
    if fc is None:
        return (
            '<h2>2. ARIMA Model</h2>'
            + warn_box('future_forecast.csv not found &mdash; run arima_analysis.py first.')
        )

    fig, ax = plt.subplots(figsize=(13, 5))
    hist = df_hist[df_hist.index >= fc.index[0] - timedelta(days=180)]
    ax.plot(hist.index, hist['valor'], color='steelblue', lw=1, label='Historical')
    ax.plot(fc.index, fc['Forecast'], color='darkorange', lw=1.5, label='ARIMA Forecast')
    ax.fill_between(fc.index, fc['Lower CI'], fc['Upper CI'],
                    color='darkorange', alpha=0.2, label='95% CI')
    ax.set_title(f'ARIMA Forecast  ({fc.index[0].date()} to {fc.index[-1].date()})')
    ax.set_ylabel('BRL per USD')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    chart = fig_to_img(fig)

    last = fc.iloc[-1]
    info = pd.DataFrame({
        'Forecast horizon end':  [fc.index[-1].strftime('%Y-%m-%d')],
        'Point forecast (end)':  [f'{last["Forecast"]:.4f}'],
        'Lower 95% CI':          [f'{last["Lower CI"]:.4f}'],
        'Upper 95% CI':          [f'{last["Upper CI"]:.4f}'],
        'Steps forecast':        [str(len(fc))],
    }).T
    info.columns = ['Value']

    return (
        '<h2>2. ARIMA Model</h2>'
        '<p>Grid-searched ARIMA(p,1,q) with AIC selection over p, q &isin; {0,1,2}. '
        'Walk-forward out-of-sample validation on the last 60 observations. '
        '<em>Economic interpretation:</em> captures autocorrelation and momentum in the rate level.</p>'
        + chart
        + '<h4>Forecast Summary</h4>'
        + df_to_html(info)
    )


# ---------------------------------------------------------------------------
# Section 3: Prophet
# ---------------------------------------------------------------------------

def section_prophet(df_hist):
    fc = load_forecast('prophet_forecast.csv')
    if fc is None:
        return (
            '<h2>3. Prophet Model</h2>'
            + warn_box('prophet_forecast.csv not found &mdash; run prophet_analysis.py first.')
        )

    if 'ds' in fc.columns:
        fc['ds'] = pd.to_datetime(fc['ds'])
        fc = fc.set_index('ds')

    last_hist = df_hist.index[-1]
    fc_future = fc[fc.index > last_hist]
    fc_insamp = fc[fc.index <= last_hist]

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(df_hist.index[-365:], df_hist['valor'].tail(365),
            color='steelblue', lw=1, label='Historical (last year)')
    ax.plot(fc_insamp.index[-365:], fc_insamp['yhat'].tail(365),
            color='gray', lw=0.8, ls='--', label='In-sample fit')
    if len(fc_future):
        ax.plot(fc_future.index, fc_future['yhat'],
                color='darkorange', lw=1.5, label='Prophet Forecast')
        ax.fill_between(fc_future.index,
                        fc_future['yhat_lower'], fc_future['yhat_upper'],
                        color='darkorange', alpha=0.2, label='Uncertainty interval')
    ax.set_title('Prophet Forecast')
    ax.set_ylabel('BRL per USD')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    chart = fig_to_img(fig)

    cv_html = ''
    cv = load_forecast('prophet_cv_metrics.csv')
    if cv is not None and 'mae' in cv.columns:
        sample = cv[['mae', 'rmse', 'mape']].dropna().head(10)
        cv_html = (
            '<h4>Cross-Validation Metrics (first 10 horizons)</h4>'
            + sample.to_html(classes='data-table', border=0, float_format='{:.4f}'.format)
        )

    end_fc = fc_future['yhat'].iloc[-1] if len(fc_future) else float('nan')
    info = pd.DataFrame({
        'Forecast horizon end':  [fc_future.index[-1].strftime('%Y-%m-%d') if len(fc_future) else 'N/A'],
        'Point forecast (end)':  [f'{end_fc:.4f}' if not np.isnan(end_fc) else 'N/A'],
    }).T
    info.columns = ['Value']

    return (
        '<h2>3. Prophet Model</h2>'
        '<p>Additive seasonality model with Brazilian public holidays. '
        'Weekly and yearly seasonality components. '
        '<em>Economic interpretation:</em> decomposes the rate into trend, seasonal patterns, '
        'and holiday effects.</p>'
        + chart
        + '<h4>Forecast Summary</h4>'
        + df_to_html(info)
        + cv_html
    )


# ---------------------------------------------------------------------------
# Section 4: GARCH
# ---------------------------------------------------------------------------

def section_garch(df_hist):
    fc = load_forecast('garch_forecast.csv')
    if fc is None:
        return (
            '<h2>4. GARCH Volatility Model</h2>'
            + warn_box('garch_forecast.csv not found &mdash; run garch_analysis.py first.')
        )

    if 'annualized_volatility' not in fc.columns:
        return (
            '<h2>4. GARCH Volatility Model</h2>'
            + warn_box('Unexpected format in garch_forecast.csv.')
        )

    returns  = (np.log(df_hist['valor']).diff() * 100).dropna()
    hist_vol = returns.rolling(21).std() * np.sqrt(252)   # annualised %
    fc_vol   = fc['annualized_volatility'] * 100           # fraction -> %

    fig, axes = plt.subplots(2, 1, figsize=(13, 7))
    axes[0].plot(hist_vol.index[-365:], hist_vol.tail(365),
                 color='steelblue', label='21-day rolling vol (annualised %)')
    axes[0].set_title('Historical Realised Volatility (last year)')
    axes[0].set_ylabel('Annualised Vol (%)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(fc.index, fc_vol, color='darkorange', label='GARCH(1,1) forecast')
    axes[1].axhline(fc_vol.mean(), color='red', ls='--', lw=0.9,
                    label=f'1-yr mean: {fc_vol.mean():.1f}%')
    axes[1].set_title('GARCH(1,1) Volatility Forecast (next year)')
    axes[1].set_ylabel('Annualised Vol (%)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    plt.tight_layout()
    chart = fig_to_img(fig)

    info = pd.DataFrame({
        'Current 21-day realised vol':  [f'{hist_vol.iloc[-1]:.1f}%'],
        'GARCH forecast mean (1 year)': [f'{fc_vol.mean():.1f}%'],
        'GARCH forecast at 1 month':    [f'{fc_vol.iloc[21]:.1f}%'  if len(fc_vol) > 21  else 'N/A'],
        'GARCH forecast at 6 months':   [f'{fc_vol.iloc[126]:.1f}%' if len(fc_vol) > 126 else 'N/A'],
    }).T
    info.columns = ['Value']

    return (
        '<h2>4. GARCH Volatility Model</h2>'
        '<p>GARCH(1,1) with Student-t errors fitted to USD/BRL daily log returns. '
        '<em>Economic interpretation:</em> volatility clustering (Engle 1982, Bollerslev 1986) &mdash; '
        'high-volatility regimes persist. This model forecasts <strong>uncertainty</strong> '
        'around the rate level, not the level itself. '
        'Persistence &alpha;+&beta; near 1 indicates slow mean-reversion of volatility (typical for FX).</p>'
        + chart
        + '<h4>Volatility Summary</h4>'
        + df_to_html(info)
    )


# ---------------------------------------------------------------------------
# Section 5: VAR
# ---------------------------------------------------------------------------

def section_var(df_hist):
    fc = load_forecast('var_forecast.csv')
    if fc is None:
        return (
            '<h2>5. VAR Model (USD/BRL + SELIC)</h2>'
            + warn_box(
                'var_forecast.csv not found &mdash; '
                'run <code>python download_data.py --all</code> then var_analysis.py.'
            )
        )

    if 'usd_brl_forecast' not in fc.columns:
        return (
            '<h2>5. VAR Model (USD/BRL + SELIC)</h2>'
            + warn_box('Unexpected format in var_forecast.csv.')
        )

    fig, axes = plt.subplots(2, 1, figsize=(13, 8))
    hist = df_hist[df_hist.index >= fc.index[0] - timedelta(days=180)]
    axes[0].plot(hist.index, hist['valor'], color='steelblue', lw=1, label='Historical')
    axes[0].plot(fc.index, fc['usd_brl_forecast'],
                 color='darkgreen', lw=1.5, label='VAR Forecast')
    axes[0].set_title('VAR Forecast: USD/BRL')
    axes[0].set_ylabel('BRL per USD')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if 'selic_rate_forecast' in fc.columns:
        axes[1].plot(fc.index, fc['selic_rate_forecast'], color='purple', lw=1.5)
        axes[1].set_title('VAR Forecast: SELIC Rate (% p.a.)')
        axes[1].set_ylabel('SELIC (%)')
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    chart = fig_to_img(fig)

    info = pd.DataFrame({
        'Forecast horizon end':          [fc.index[-1].strftime('%Y-%m-%d')],
        'USD/BRL forecast (end)':        [f'{fc["usd_brl_forecast"].iloc[-1]:.4f}'],
        'SELIC forecast end (% p.a.)':   [
            f'{fc["selic_rate_forecast"].iloc[-1]:.2f}%'
            if 'selic_rate_forecast' in fc.columns else 'N/A'
        ],
    }).T
    info.columns = ['Value']

    return (
        '<h2>5. VAR Model (USD/BRL + SELIC)</h2>'
        '<p>Bivariate Vector Autoregression jointly modelling USD/BRL and Brazil\'s SELIC rate. '
        '<em>Economic basis:</em> Uncovered Interest Rate Parity (UIP) &mdash; expected '
        'exchange-rate changes equal the interest-rate differential. '
        'The Granger causality test answers: '
        '&ldquo;do past SELIC values statistically improve USD/BRL forecasts?&rdquo;</p>'
        + chart
        + '<h4>Forecast Summary</h4>'
        + df_to_html(info)
    )


# ---------------------------------------------------------------------------
# Section 6: Consensus comparison
# ---------------------------------------------------------------------------

def _nearest(series, date):
    """Return the value in series nearest to date, or None."""
    try:
        idx = series.index.asof(date)
        return float(series.loc[idx]) if pd.notna(idx) else None
    except Exception:
        return None


def section_consensus(df_hist, focus_df):
    arima_fc   = load_forecast('future_forecast.csv')
    prophet_fc = load_forecast('prophet_forecast.csv')
    var_fc     = load_forecast('var_forecast.csv')

    if prophet_fc is not None and 'ds' in prophet_fc.columns:
        prophet_fc['ds'] = pd.to_datetime(prophet_fc['ds'])
        prophet_fc = prophet_fc.set_index('ds')

    last_hist = df_hist.index[-1]

    # --- Combined forecast chart ---
    fig, ax = plt.subplots(figsize=(13, 6))
    hist_90 = df_hist[df_hist.index >= last_hist - timedelta(days=90)]
    ax.plot(hist_90.index, hist_90['valor'],
            color='steelblue', lw=1.5, label='Historical (last 90 days)')

    if arima_fc is not None:
        ax.plot(arima_fc.index, arima_fc['Forecast'],
                color='darkorange', lw=1.2, ls='--', label='ARIMA')
        ax.fill_between(arima_fc.index, arima_fc['Lower CI'], arima_fc['Upper CI'],
                        color='darkorange', alpha=0.08)

    if prophet_fc is not None and 'yhat' in prophet_fc.columns:
        fp = prophet_fc[prophet_fc.index > last_hist]
        if len(fp):
            ax.plot(fp.index, fp['yhat'],
                    color='red', lw=1.2, ls='-.', label='Prophet')

    if var_fc is not None and 'usd_brl_forecast' in var_fc.columns:
        ax.plot(var_fc.index, var_fc['usd_brl_forecast'],
                color='darkgreen', lw=1.2, ls=':', label='VAR')

    if focus_df is not None and 'Mediana' in focus_df.columns:
        fut_focus = focus_df[focus_df.index > last_hist]
        if len(fut_focus):
            yerr = (fut_focus['DesvioPadrao'].values
                    if 'DesvioPadrao' in fut_focus.columns else None)
            ax.errorbar(fut_focus.index, fut_focus['Mediana'],
                        yerr=yerr, fmt='D', color='black',
                        ms=7, lw=1.5, capsize=5, label='Focus Median (+/- 1 SD)')

    ax.set_title('All Model Forecasts vs BCB Focus Market Consensus')
    ax.set_ylabel('BRL per USD')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    chart = fig_to_img(fig)

    # --- Horizon comparison table ---
    horizons = [
        ('3 months',  last_hist + pd.DateOffset(months=3)),
        ('6 months',  last_hist + pd.DateOffset(months=6)),
        ('12 months', last_hist + pd.DateOffset(months=12)),
    ]

    rows = []
    for label, tdate in horizons:
        row = {'Horizon': label, 'Target date': tdate.strftime('%Y-%m-%d')}

        row['ARIMA'] = (
            round(_nearest(arima_fc['Forecast'], tdate), 3)
            if arima_fc is not None else 'N/A'
        )

        if prophet_fc is not None and 'yhat' in prophet_fc.columns:
            fp = prophet_fc[prophet_fc.index > last_hist]
            row['Prophet'] = round(_nearest(fp['yhat'], tdate), 3) if len(fp) else 'N/A'
        else:
            row['Prophet'] = 'N/A'

        row['VAR'] = (
            round(_nearest(var_fc['usd_brl_forecast'], tdate), 3)
            if (var_fc is not None and 'usd_brl_forecast' in var_fc.columns) else 'N/A'
        )

        # Focus: use year-end median for the target year
        if focus_df is not None and 'Mediana' in focus_df.columns:
            year_end = pd.Timestamp(year=tdate.year, month=12, day=31)
            v = _nearest(focus_df['Mediana'], year_end)
            row['Focus Median'] = round(v, 3) if v is not None else '—'
            v2 = _nearest(focus_df['Media'], year_end)
            row['Focus Mean']   = round(v2, 3) if v2 is not None else '—'
        else:
            row['Focus Median'] = 'Unavailable'
            row['Focus Mean']   = 'Unavailable'

        rows.append(row)

    cmp_df = pd.DataFrame(rows).set_index('Horizon')

    if focus_df is not None and 'Data' in focus_df.columns:
        focus_note = (
            f'<p>BCB Focus survey date: <strong>{focus_df["Data"].iloc[0].strftime("%Y-%m-%d")}</strong>'
            f' &mdash; median and mean aggregated from ~120 financial institutions.</p>'
        )
    else:
        focus_note = warn_box(
            'BCB Focus consensus could not be retrieved. '
            'Check network connectivity and try again.'
        )

    return (
        '<h2>6. Model Comparison vs Market Consensus (BCB Focus)</h2>'
        '<p>The BCB Focus Report aggregates weekly forecasts from ~120 financial institutions. '
        'Focus year-end medians are compared against model point forecasts at 3-, 6-, and 12-month '
        'horizons. GARCH is excluded here because it forecasts <em>volatility</em>, not the rate level.</p>'
        + focus_note
        + chart
        + '<h4>Point Estimates at Key Horizons (USD/BRL)</h4>'
        + cmp_df.to_html(classes='data-table', border=0, na_rep='--')
    )


# ---------------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------------

_CSS = """
body  { font-family:'Helvetica Neue',Arial,sans-serif; font-size:14px; color:#222;
        max-width:1060px; margin:0 auto; padding:28px; background:#fff; }
h1    { color:#1a1a2e; border-bottom:3px solid #1a73e8; padding-bottom:8px; }
h2    { color:#1a73e8; margin-top:44px; border-bottom:1px solid #dde; padding-bottom:4px; }
h4    { color:#444; margin-top:18px; }
p     { line-height:1.65; }
code  { background:#f0f0f0; padding:1px 4px; border-radius:3px; font-size:12px; }
.warn { background:#fff3cd; border:1px solid #ffc107; border-radius:4px;
        padding:10px 14px; color:#856404; margin:8px 0; }
.meta { color:#666; font-size:12px; margin-bottom:32px; }
table.data-table { border-collapse:collapse; width:100%; margin:12px 0; font-size:13px; }
table.data-table th { background:#1a73e8; color:#fff; padding:8px 12px; text-align:left; }
table.data-table td { padding:6px 12px; border-bottom:1px solid #eee; }
table.data-table tr:nth-child(even) td { background:#f8f9fa; }
img   { display:block; }
hr    { border:none; border-top:1px solid #dde; margin:40px 0; }
"""


def build_html(sections):
    now  = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
    body = '\n'.join(sections)
    return (
        '<!DOCTYPE html>\n'
        '<html lang="en">\n'
        '<head>\n'
        '<meta charset="UTF-8">\n'
        '<meta name="viewport" content="width=device-width,initial-scale=1.0">\n'
        f'<title>USD/BRL Forecast Report &mdash; {now}</title>\n'
        f'<style>{_CSS}</style>\n'
        '</head>\n'
        '<body>\n'
        '<h1>USD/BRL Exchange Rate Forecast Report</h1>\n'
        f'<p class="meta">Generated: {now} &nbsp;|&nbsp; Source: Brazil Central Bank (BCB) API</p>\n'
        + body
        + '\n<hr>\n'
        '<p class="meta">'
        'Data: BCB Series 1 (USD/BRL) and Series 432 (SELIC). '
        'Consensus: BCB Focus Report.<br>'
        'Models: ARIMA, Prophet (Meta), GARCH(1,1), VAR. '
        'Report generated automatically via GitHub Actions.'
        '</p>\n'
        '</body>\n'
        '</html>'
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print('Generating USD/BRL forecast report...')

    try:
        df = pd.read_csv('usd_brl_history.csv')
        df['data'] = pd.to_datetime(df['data'])
        df = df.set_index('data').sort_index()
        print(f'  Historical data: {len(df)} rows '
              f'({df.index[0].date()} to {df.index[-1].date()})')
    except FileNotFoundError:
        print('Error: usd_brl_history.csv not found. Run download_data.py first.')
        sys.exit(1)

    print('  Fetching BCB Focus consensus...')
    focus_df = fetch_focus_consensus()

    print('  Building report sections...')
    sections = [
        section_summary(df, focus_df),
        section_exploration(df),
        section_arima(df),
        section_prophet(df),
        section_garch(df),
        section_var(df),
        section_consensus(df, focus_df),
    ]

    html = build_html(sections)

    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(html)

    size_kb = len(html.encode()) / 1024
    print(f'  Report saved: {REPORT_FILE} ({size_kb:.0f} KB)')


if __name__ == '__main__':
    main()
