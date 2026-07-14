import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import joblib
from stable_baselines3 import PPO
from datetime import datetime

# Feature Engineering Functions
def calc_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calc_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist

def calc_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def calc_bollinger_bands(series, period=20, std_dev=2):
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, lower

def generate_features(df):
    # Basic technicals
    df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    
    df['RSI_14'] = calc_rsi(df['Close'], 14)
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = calc_macd(df['Close'])
    df['ATR'] = calc_atr(df['High'], df['Low'], df['Close'], 14)
    
    df['BB_Upper'], df['BB_Lower'] = calc_bollinger_bands(df['Close'], 20, 2)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['Close']
    
    # Kalman Filter (Simplified for speed)
    df['KF_Close'] = df['Close'].ewm(span=5, adjust=False).mean() 
    
    # FracDiff (Simplified approximation for speed)
    df['Frac_Diff_0.4'] = df['Close'] - 0.4 * df['Close'].shift(1) - 0.12 * df['Close'].shift(2)
    
    # GARCH Vol (Approximation using rolling variance for speed)
    df['GARCH_Vol'] = df['Close'].pct_change().rolling(window=20).std() * np.sqrt(252*96)
    
    # Log Returns
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def run_unseen_backtest():
    print("Loading data...")
    df = pd.read_csv("advanced_features_strict.csv")
    
    # The Meta-Model was trained on the last 5,000 candles (index 20k to 25k).
    # We will test on 10,000 candles from index 10,000 to 20,000. 
    # This is 100% OUT OF SAMPLE for the Meta-Model.
    df = df.iloc[10000:20000].reset_index(drop=True)
    
    print(f"Data Date Range: {df['Time'].min()} to {df['Time'].max()}")
    
    print("Loading AI Models...")
    ppo_model = PPO.load("ppo_deriv_bot")
    ensemble = joblib.load("ensemble_meta.pkl")
        
    feature_cols = [col for col in df.columns if col not in ['Time', 'Open', 'High', 'Low', 'Close', 'KF_Close', 'EMA_50']]
    obs_df = df[feature_cols]
    
    prices = df['Close'].values
    highs = df['High'].values
    lows = df['Low'].values
    
    # We'll test our best configuration
    SL_POINTS = 200
    TP_POINTS = 200
    
    balance = 0.0
    current_position = 0
    entry_price = 0.0
    trades_taken = 0
    wins = 0
    
    print(f"Running vectorized out-of-sample backtest with SL={SL_POINTS}, TP={TP_POINTS}...")
    
    for i in range(96, len(df)):
        obs_window = obs_df.iloc[i-96:i].values
        obs = np.array(obs_window, dtype=np.float32)
        
        # 1. PPO decides direction
        action, _ = ppo_model.predict(obs, deterministic=True)
        action = int(action)
        
        # 2. Ensemble verifies
        features_for_meta = list(obs[-1]) + [action]
        meta_input_df = pd.DataFrame([features_for_meta], columns=[f"feat_{j}" for j in range(14)] + ['direction'])
        win_prob = ensemble.predict_proba(meta_input_df)[0][1]
        
        # 3. Veto if confidence is < 60%
        veto = win_prob < 0.60
        
        close = prices[i]
        high = highs[i]
        low = lows[i]
        
        # Check SL / TP
        if current_position == 1:
            if low <= entry_price - SL_POINTS:
                profit = -SL_POINTS
                balance += profit
                trades_taken += 1
                current_position = 0
            elif high >= entry_price + TP_POINTS:
                profit = TP_POINTS
                balance += profit
                trades_taken += 1
                wins += 1
                current_position = 0
        elif current_position == 2:
            if high >= entry_price + SL_POINTS:
                profit = -SL_POINTS
                balance += profit
                trades_taken += 1
                current_position = 0
            elif low <= entry_price - TP_POINTS:
                profit = TP_POINTS
                balance += profit
                trades_taken += 1
                wins += 1
                current_position = 0
                
        # Trade Execution
        if current_position == 0:
            if action == 0 and not veto:
                current_position = 1
                entry_price = close
            elif action == 1 and not veto:
                current_position = 2
                entry_price = close
        else:
            # Signal flips
            if current_position == 1 and action == 1 and not veto:
                profit = close - entry_price
                balance += profit
                trades_taken += 1
                if profit > 0: wins += 1
                
                current_position = 2
                entry_price = close
                
            elif current_position == 2 and action == 0 and not veto:
                profit = entry_price - close
                balance += profit
                trades_taken += 1
                if profit > 0: wins += 1
                
                current_position = 1
                entry_price = close
                
    # Close open position
    if current_position == 1:
        profit = close - entry_price
        balance += profit
        trades_taken += 1
        if profit > 0: wins += 1
    elif current_position == 2:
        profit = entry_price - close
        balance += profit
        trades_taken += 1
        if profit > 0: wins += 1
        
    win_rate = (wins / trades_taken * 100) if trades_taken > 0 else 0
    
    print("\n========================================================")
    print("      COMPLETELY UNSEEN OUT-OF-SAMPLE RESULTS           ")
    print("========================================================")
    print(f"Candles Tested:    {len(df)}")
    print(f"Date Range:        {df['Time'].min()} to {df['Time'].max()}")
    print(f"Stop Loss:         {SL_POINTS} pts")
    print(f"Take Profit:       {TP_POINTS} pts")
    print(f"Total Trades:      {trades_taken}")
    print(f"Win Rate:          {win_rate:.1f}%")
    print(f"Total Net Profit:  {balance:.1f} points")
    print("========================================================")

if __name__ == "__main__":
    run_unseen_backtest()
