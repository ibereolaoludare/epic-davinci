"""
Feature Engineering V6: Zero-leakage pipeline with alternative data.
Merges technical features with COT positioning, yield spreads, VIX, and DXY.
All joins use merge_asof(direction='backward') to prevent look-ahead bias.
"""
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# Features that the model is allowed to see
EXCLUDE_COLS = ['Target', 'Time', 'Future_Close', 'Open', 'High', 'Low', 'Close', 
                'Volume', 'EMA_50', 'EMA_200', 'Hour', 'DayOfWeek', 'MACD_Signal',
                'Returns']  # Returns kept internally for Rolling_Vol but excluded from model


def process_features_v6(df_raw):
    df = df_raw.copy()
    df['Time'] = pd.to_datetime(df['Time'], utc=True)
    
    # ========================
    # 1. SESSION EMBEDDINGS
    # ========================
    df['Hour'] = df['Time'].dt.hour
    df['DayOfWeek'] = df['Time'].dt.dayofweek
    df['Time_Of_Day_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24.0)
    df['Time_Of_Day_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24.0)
    df['Day_Of_Week_Sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7.0)
    df['Day_Of_Week_Cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7.0)

    # ========================
    # 2. TECHNICAL INDICATORS (all backward-looking)
    # ========================
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()

    sma20 = df['Close'].rolling(window=20).mean()
    std20 = df['Close'].rolling(window=20).std()
    df['BB_Width'] = (4 * std20) / sma20
    df['Z_Score'] = (df['Close'] - sma20) / std20

    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
    df['EMA_50_Dist'] = (df['Close'] - df['EMA_50']) / df['EMA_50']
    df['EMA_200_Dist'] = (df['Close'] - df['EMA_200']) / df['EMA_200']

    if 'Volume' not in df.columns:
        df['Volume'] = 1000
    df['Volume_Log'] = np.log1p(df['Volume'])
    df['Returns'] = df['Close'].pct_change() * 100
    df['Rolling_Vol_20'] = df['Returns'].rolling(window=20).std()

    # ========================
    # 3. MACRO: CENTRAL BANK RATES (backward merge, no leakage)
    # ========================
    try:
        df_rates = pd.read_csv("central_bank_rates.csv")
        df_rates['date'] = pd.to_datetime(df_rates['date'], utc=True)
    except FileNotFoundError:
        df_rates = pd.DataFrame()

    if not df_rates.empty:
        df_rates = df_rates.sort_values('date')
        df = df.sort_values('Time')
        df = pd.merge_asof(df, df_rates[['date', 'fed_rate', 'boj_rate']], 
                           left_on='Time', right_on='date', direction='backward')
        df['Rate_Diff'] = df['fed_rate'] - df['boj_rate']
        df = df.drop(columns=['date', 'fed_rate', 'boj_rate'])
        df['Rate_Diff'] = df['Rate_Diff'].fillna(0)
        df['Rate_Diff_Mom'] = df['Rate_Diff'] - df['Rate_Diff'].shift(90)
        df['Rate_Diff_Mom'] = df['Rate_Diff_Mom'].fillna(0)
    else:
        df['Rate_Diff'], df['Rate_Diff_Mom'] = 0.0, 0.0

    try:
        df_int = pd.read_csv("interventions.csv")
        df_int['date'] = pd.to_datetime(df_int['date'], utc=True)
    except FileNotFoundError:
        df_int = pd.DataFrame()

    if not df_int.empty:
        df['Intervention_Dist'] = np.abs(df['Close'] - 160.0) / 160.0
        df['In_Intervention_Zone'] = (df['Intervention_Dist'] < 0.005).astype(int)
    else:
        df['Intervention_Dist'], df['In_Intervention_Zone'] = 0.0, 0

    # ========================
    # 4. ALTERNATIVE DATA (backward merge, no leakage)
    # ========================
    try:
        df_alt = pd.read_csv("alternative_data.csv")
        df_alt['date'] = pd.to_datetime(df_alt['date'], utc=True)
        df_alt = df_alt.sort_values('date')
    except FileNotFoundError:
        print("WARNING: alternative_data.csv not found. Alt features will be zero.")
        df_alt = pd.DataFrame()

    if not df_alt.empty:
        # Backward merge: each H4 candle gets the MOST RECENT daily/weekly data point
        df = pd.merge_asof(df.sort_values('Time'), df_alt, 
                           left_on='Time', right_on='date', direction='backward')
        df = df.drop(columns=['date'], errors='ignore')
        
        # --- COT Features ---
        if 'COT_Net' in df.columns:
            # Z-score of net positioning over ~52 weeks (52 * 7 / (4/6) ≈ ~546 H4 candles)
            # But COT is weekly so we use the forward-filled daily values
            df['COT_Net_Z'] = (df['COT_Net'] - df['COT_Net'].rolling(window=260, min_periods=30).mean()) / \
                               df['COT_Net'].rolling(window=260, min_periods=30).std()
            df['COT_Net_Z'] = df['COT_Net_Z'].fillna(0)
            
            # Week-over-week change (shift by ~42 H4 candles ≈ 1 week)
            df['COT_Change'] = df['COT_Net'] - df['COT_Net'].shift(42)
            df['COT_Change'] = df['COT_Change'].fillna(0)
        else:
            df['COT_Net'], df['COT_Net_Z'], df['COT_Change'], df['COT_OI'] = 0, 0, 0, 0
        
        # --- Yield Spread Features ---
        if 'US10Y' in df.columns:
            # Japan 10Y is not easily available via yfinance, approximate from Rate_Diff
            # The BOJ rate serves as a proxy for the short end; 
            # for the spread we use US10Y directly as the dominant driver
            df['US10Y'] = df['US10Y'].ffill().fillna(0)
            df['US10Y_Mom_20'] = df['US10Y'] - df['US10Y'].shift(120)  # ~20 trading days in H4
            df['US10Y_Mom_20'] = df['US10Y_Mom_20'].fillna(0)
            df['US10Y_Z_60'] = (df['US10Y'] - df['US10Y'].rolling(window=360, min_periods=30).mean()) / \
                                df['US10Y'].rolling(window=360, min_periods=30).std()
            df['US10Y_Z_60'] = df['US10Y_Z_60'].fillna(0)
        else:
            df['US10Y'], df['US10Y_Mom_20'], df['US10Y_Z_60'] = 0, 0, 0
        
        # --- VIX Features ---
        if 'VIX' in df.columns:
            df['VIX'] = df['VIX'].ffill().fillna(15)
            df['VIX_Change_5d'] = (df['VIX'] / df['VIX'].shift(30).replace(0, np.nan) - 1) * 100  # ~5 trading days
            df['VIX_Change_5d'] = df['VIX_Change_5d'].fillna(0)
            df['VIX_Regime'] = (df['VIX'] > 20).astype(int)
        else:
            df['VIX'], df['VIX_Change_5d'], df['VIX_Regime'] = 15, 0, 0
        
        # --- DXY Features ---
        if 'DXY' in df.columns:
            df['DXY'] = df['DXY'].ffill().fillna(100)
            df['DXY_Mom_10d'] = (df['DXY'] / df['DXY'].shift(60).replace(0, np.nan) - 1) * 100  # ~10 trading days
            df['DXY_Mom_10d'] = df['DXY_Mom_10d'].fillna(0)
            df['DXY_Z_20d'] = (df['DXY'] - df['DXY'].rolling(window=120, min_periods=30).mean()) / \
                               df['DXY'].rolling(window=120, min_periods=30).std()
            df['DXY_Z_20d'] = df['DXY_Z_20d'].fillna(0)
            # Drop raw DXY (non-stationary)
            df = df.drop(columns=['DXY'])
        else:
            df['DXY_Mom_10d'], df['DXY_Z_20d'] = 0, 0
    else:
        # No alt data - add zero placeholders
        for col in ['COT_Net', 'COT_OI', 'COT_Net_Z', 'COT_Change',
                     'US10Y', 'US10Y_Mom_20', 'US10Y_Z_60',
                     'VIX', 'VIX_Change_5d', 'VIX_Regime',
                     'DXY_Mom_10d', 'DXY_Z_20d']:
            df[col] = 0

    # ========================
    # 5. CLEANUP
    # ========================
    df = df.dropna().reset_index(drop=True)

    # ========================
    # 6. TRIPLE BARRIER TARGET (1.5 ATR, 12-candle horizon)
    # ========================
    TARGET_HORIZON = 12
    TARGET_ATR_MULT = 1.5
    
    targets = np.zeros(len(df))
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    atr = df['ATR'].values
    
    for i in range(len(df) - TARGET_HORIZON):
        current_close = close[i]
        current_atr = atr[i]
        
        upper_barrier = current_close + (current_atr * TARGET_ATR_MULT)
        lower_barrier = current_close - (current_atr * TARGET_ATR_MULT)
        
        for j in range(1, TARGET_HORIZON + 1):
            idx = i + j
            if high[idx] >= upper_barrier:
                targets[i] = 1
                break
            elif low[idx] <= lower_barrier:
                targets[i] = -1
                break
                
    df['Target'] = targets
    return df


if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else 'USDJPY'
    
    print(f"Building V6 feature set for {ticker}...")
    try:
        df_raw = pd.read_csv(f"{ticker}_H4.csv")
        df_features = process_features_v6(df_raw)
        out_csv = f"{ticker}_features_v6.csv"
        df_features.to_csv(out_csv, index=False)
        
        counts = df_features['Target'].value_counts()
        print(f"Saved {out_csv}: {len(df_features)} rows")
        print(f"Target Distribution:\n{counts}")
        
        # Show which features the model will use
        feature_cols = [c for c in df_features.columns if c not in EXCLUDE_COLS]
        print(f"\nModel features ({len(feature_cols)}):")
        for c in sorted(feature_cols):
            print(f"  {c}")
    except FileNotFoundError:
        print(f"{ticker}_H4.csv not found.")
