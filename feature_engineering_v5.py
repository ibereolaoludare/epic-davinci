import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

def process_features_v5(df_raw):
    df = df_raw.copy()
    df['Time'] = pd.to_datetime(df['Time'], utc=True)
    
    # Session Embeddings
    df['Hour'] = df['Time'].dt.hour
    df['DayOfWeek'] = df['Time'].dt.dayofweek
    df['Time_Of_Day_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24.0)
    df['Time_Of_Day_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24.0)
    df['Day_Of_Week_Sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7.0)
    df['Day_Of_Week_Cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7.0)

    # Technicals
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
        df['Volume'] = 1000  # Dummy volume
    df['Volume_Log'] = np.log1p(df['Volume'])
    df['Returns'] = df['Close'].pct_change() * 100
    
    # Replacement for leaky GARCH
    df['Rolling_Vol_20'] = df['Returns'].rolling(window=20).std()

    # Macro Data Loading & Alignment
    try:
        df_rates = pd.read_csv("central_bank_rates.csv")
        df_rates['date'] = pd.to_datetime(df_rates['date'], utc=True)
        df_int = pd.read_csv("interventions.csv")
        df_int['date'] = pd.to_datetime(df_int['date'], utc=True)
    except FileNotFoundError:
        df_rates, df_int = pd.DataFrame(), pd.DataFrame()

    if not df_rates.empty:
        df_rates = df_rates.sort_values('date')
        df = df.sort_values('Time')
        # backward direction ensures we only know rate at 'Time' or before
        df = pd.merge_asof(df, df_rates[['date', 'fed_rate', 'boj_rate']], left_on='Time', right_on='date', direction='backward')
        df['Rate_Diff'] = df['fed_rate'] - df['boj_rate']
        df = df.drop(columns=['date', 'fed_rate', 'boj_rate'])
        df['Rate_Diff'] = df['Rate_Diff'].fillna(0)
        
        # Momentum of Rate Diff
        df['Rate_Diff_Mom'] = df['Rate_Diff'] - df['Rate_Diff'].shift(90) # 90 H4 candles ~ 15 days
        df['Rate_Diff_Mom'] = df['Rate_Diff_Mom'].fillna(0)
        
        # REMOVED: Days_To_Next_Release (Leakage)
    else:
        df['Rate_Diff'], df['Rate_Diff_Mom'] = 0.0, 0.0

    if not df_int.empty:
        df['Intervention_Dist'] = np.abs(df['Close'] - 160.0) / 160.0
        df['In_Intervention_Zone'] = (df['Intervention_Dist'] < 0.005).astype(int)
    else:
        df['Intervention_Dist'], df['In_Intervention_Zone'] = 0.0, 0

    df = df.dropna().reset_index(drop=True)
    
    # REMOVED: GARCH & HMM (Leakage)

    # Triple Barrier Target V4 (1.5 ATR, 12 Horizon)
    TARGET_HORIZON = 12
    TARGET_ATR_MULT = 1.5
    
    # We will simulate the exact same Triple Barrier exits here to label the data.
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
    print("Loading USDJPY H4 data...")
    try:
        df_raw = pd.read_csv("USDJPY_H4.csv")
        df_features = process_features_v5(df_raw)
        df_features.to_csv("USDJPY_features_v5.csv", index=False)
        
        counts = df_features['Target'].value_counts()
        print(f"Saved USDJPY_features_v5.csv: {len(df_features)} rows")
        print(f"Target Distribution:\n{counts}")
    except FileNotFoundError:
        print("USDJPY_H4.csv not found. Run download_data.py first.")
