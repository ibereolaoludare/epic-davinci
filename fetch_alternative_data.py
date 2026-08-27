"""
Fetch alternative data sources for USD/JPY V6 pipeline.
Downloads: COT positioning, US 10Y yield, VIX, DXY.
All data is merged into a single CSV with a daily date index.
"""
import pandas as pd
import numpy as np
import yfinance as yf
import warnings
import os
import requests
import zipfile
import io
from datetime import datetime

warnings.filterwarnings('ignore')

def fetch_cot_data():
    """Download COT legacy futures data from CFTC for Japanese Yen."""
    print("  Downloading COT data from CFTC...")
    all_dfs = []
    
    current_year = datetime.now().year
    for year in range(2023, current_year + 1):
        url = f"https://www.cftc.gov/files/dea/history/fut_fin_txt_{year}.zip"
        print(f"    Fetching {year}...", end=" ")
        try:
            resp = requests.get(url, timeout=60)
            if resp.status_code != 200:
                # Try legacy format
                url = f"https://www.cftc.gov/files/dea/history/deacot{year}.zip"
                resp = requests.get(url, timeout=60)
            
            if resp.status_code == 200:
                zf = zipfile.ZipFile(io.BytesIO(resp.content))
                for fname in zf.namelist():
                    if fname.endswith('.txt') or fname.endswith('.csv'):
                        df = pd.read_csv(zf.open(fname))
                        all_dfs.append(df)
                        print(f"OK ({len(df)} rows)")
                        break
            else:
                print(f"SKIP (HTTP {resp.status_code})")
        except Exception as e:
            print(f"SKIP ({e})")
    
    if not all_dfs:
        print("  WARNING: Could not download COT data. Using synthetic placeholder.")
        return pd.DataFrame()
    
    df_all = pd.concat(all_dfs, ignore_index=True)
    
    # Find JPY contract - try different column names
    name_col = None
    for col in ['Market_and_Exchange_Names', 'Market and Exchange Names', 'CONTRACT_MARKET_NAME']:
        if col in df_all.columns:
            name_col = col
            break
    
    if name_col is None:
        print(f"  WARNING: Cannot find market name column. Columns: {list(df_all.columns[:10])}")
        return pd.DataFrame()
    
    jpy = df_all[df_all[name_col].str.contains('YEN|JPY', case=False, na=False)].copy()
    
    if jpy.empty:
        print("  WARNING: No JPY data found in COT reports.")
        return pd.DataFrame()
    
    # Find date column
    date_col = None
    for col in ['Report_Date_as_YYYY-MM-DD', 'As of Date in Form YYYY-MM-DD', 'As_of_Date_In_Form_YYYY-MM-DD']:
        if col in jpy.columns:
            date_col = col
            break
    
    if date_col is None:
        print(f"  WARNING: Cannot find date column. Columns: {list(jpy.columns[:15])}")
        return pd.DataFrame()
    
    # Find positioning columns
    long_col, short_col, oi_col = None, None, None
    for col in jpy.columns:
        col_lower = col.lower().replace(' ', '_')
        if 'noncommercial' in col_lower or 'noncomm' in col_lower or 'dealer' in col_lower:
            if 'long' in col_lower and 'spread' not in col_lower and long_col is None:
                long_col = col
            elif 'short' in col_lower and 'spread' not in col_lower and short_col is None:
                short_col = col
        if 'open_interest' in col_lower or 'open interest' in col_lower:
            if oi_col is None:
                oi_col = col
    
    if long_col is None or short_col is None:
        print(f"  WARNING: Cannot find long/short columns.")
        print(f"  Available columns containing 'long': {[c for c in jpy.columns if 'long' in c.lower()]}")
        return pd.DataFrame()
    
    jpy['date'] = pd.to_datetime(jpy[date_col])
    jpy['COT_Long'] = pd.to_numeric(jpy[long_col], errors='coerce')
    jpy['COT_Short'] = pd.to_numeric(jpy[short_col], errors='coerce')
    jpy['COT_Net'] = jpy['COT_Long'] - jpy['COT_Short']
    
    if oi_col:
        jpy['COT_OI'] = pd.to_numeric(jpy[oi_col], errors='coerce')
    else:
        jpy['COT_OI'] = jpy['COT_Long'] + jpy['COT_Short']
    
    result = jpy[['date', 'COT_Net', 'COT_OI']].sort_values('date').drop_duplicates('date')
    print(f"  COT data: {len(result)} weekly observations from {result['date'].iloc[0].date()} to {result['date'].iloc[-1].date()}")
    return result


def fetch_market_data():
    """Download US 10Y yield, VIX, and DXY from yfinance."""
    print("  Downloading market data from yfinance...")
    
    tickers = {
        '^TNX': 'US10Y',
        '^VIX': 'VIX', 
        'DX-Y.NYB': 'DXY'
    }
    
    all_data = {}
    for ticker, name in tickers.items():
        print(f"    Fetching {name} ({ticker})...", end=" ")
        try:
            data = yf.download(ticker, start='2023-07-01', end=datetime.now().strftime('%Y-%m-%d'), 
                             interval='1d', progress=False)
            if len(data) > 0:
                # yfinance returns MultiIndex columns sometimes
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                all_data[name] = data['Close'].copy()
                print(f"OK ({len(data)} days)")
            else:
                print("SKIP (no data)")
        except Exception as e:
            print(f"SKIP ({e})")
    
    if not all_data:
        return pd.DataFrame()
    
    df_market = pd.DataFrame(all_data)
    df_market.index.name = 'date'
    df_market = df_market.reset_index()
    df_market['date'] = pd.to_datetime(df_market['date']).dt.tz_localize(None)
    
    # TNX is reported as yield * 10 (e.g., 45.2 = 4.52%)
    if 'US10Y' in df_market.columns:
        df_market['US10Y'] = df_market['US10Y'] / 10.0
    
    # Forward fill any gaps (weekends, holidays)
    df_market = df_market.sort_values('date').ffill()
    
    print(f"  Market data: {len(df_market)} days from {df_market['date'].iloc[0].date()} to {df_market['date'].iloc[-1].date()}")
    return df_market


def main():
    print("=" * 60)
    print("Fetching Alternative Data for USD/JPY V6 Pipeline")
    print("=" * 60)
    
    # 1. COT Data
    df_cot = fetch_cot_data()
    
    # 2. Market Data (Yields, VIX, DXY)
    df_market = fetch_market_data()
    
    # 3. Merge everything on date
    if not df_market.empty:
        df_combined = df_market.copy()
    else:
        print("FATAL: No market data available.")
        return
    
    if not df_cot.empty:
        df_cot['date'] = pd.to_datetime(df_cot['date']).dt.tz_localize(None)
        # Normalize both sides to same datetime resolution
        df_combined['date'] = pd.to_datetime(df_combined['date']).astype('datetime64[ns]')
        df_cot['date'] = pd.to_datetime(df_cot['date']).astype('datetime64[ns]')
        df_combined = pd.merge_asof(
            df_combined.sort_values('date'),
            df_cot.sort_values('date'),
            on='date',
            direction='backward'
        )
    else:
        df_combined['COT_Net'] = 0
        df_combined['COT_OI'] = 0
    
    # Save
    df_combined.to_csv('alternative_data.csv', index=False)
    print(f"\nSaved alternative_data.csv: {len(df_combined)} rows")
    print(f"Columns: {list(df_combined.columns)}")
    print(df_combined.tail())


if __name__ == "__main__":
    main()
