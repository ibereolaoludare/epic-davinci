import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
import warnings
import os

warnings.filterwarnings("ignore")

def process_features(df_raw):
    df = df_raw.copy()
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df = df.dropna().reset_index(drop=True)
    
    returns_scaled = df['Log_Return'] * 100
    am = arch_model(returns_scaled, vol='Garch', p=1, q=1)
    res = am.fit(disp='off')
    df['GARCH_Vol'] = res.conditional_volatility / 100
    
    model = ARIMA(df['Log_Return'], order=(1, 0, 1))
    model_fit = model.fit()
    df['ARIMA_Resid'] = model_fit.resid
    
    df = df.dropna().reset_index(drop=True)
    feature_cols = ['Close', 'High', 'Low', 'Open', 'Log_Return', 'GARCH_Vol', 'ARIMA_Resid']
    return df[feature_cols].values

print("Initializing MT5...")
if not mt5.initialize():
    print("MT5 initialization failed")
    quit()

login = int(os.getenv("MT5_LOGIN", "32312791"))
password = os.getenv("MT5_PASSWORD")
server = os.getenv("MT5_SERVER", "Deriv-Demo")

if not password:
    print("ERROR: Please set $env:MT5_PASSWORD before running this script.")
    mt5.shutdown()
    quit()

if not mt5.login(login, password=password, server=server):
    print("MT5 login failed")
    mt5.shutdown()
    quit()

symbol = "Volatility 100 Index"
print(f"Fetching 30,000 M15 candles for {symbol} from MT5...")
rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 1, 30000)

if rates is None:
    print("Failed to fetch rates from MT5.")
    mt5.shutdown()
    quit()

df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')
df = df.rename(columns={'time': 'Time', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'})

print("Processing features (this may take a minute)...")
try:
    features_matrix = process_features(df)
except Exception as e:
    print(f"Error processing features: {e}")
    mt5.shutdown()
    quit()

print("Loading AI Brain...")
try:
    model = PPO.load("v100_crusher_archive/ppo_deriv_bot.zip")
except Exception as e:
    print(f"Error loading model: {e}")
    mt5.shutdown()
    quit()

WINDOW_SIZE = 96
STARTING_BALANCE = 10000.0
VOLUME = float(os.getenv("MT5_LOT_SIZE", "10.0"))
SL_POINTS = float(os.getenv("MT5_SL_POINTS", "60.0"))
TP_POINTS = float(os.getenv("MT5_TP_POINTS", "60.0"))

# Ten genuinely distinct, honest configurations to test.
# SL/TP = catastrophic safety nets (primary exits are AI direction-flips).
# compound=True = lot size scales with account equity as it grows/shrinks.
configs = [
    {"sl": 999999, "tp": 999999, "volume": 10.0, "compound": False, "name": "1.  No Safety Net (Pure AI)"},
    {"sl": 500,    "tp": 500,    "volume": 10.0, "compound": False, "name": "2.  Wide Safety Net  (500pt)"},
    {"sl": 200,    "tp": 200,    "volume": 10.0, "compound": False, "name": "3.  Medium Safety Net (200pt)"},
    {"sl": 100,    "tp": 100,    "volume": 10.0, "compound": False, "name": "4.  Tight Safety Net  (100pt)"},
    {"sl": 50,     "tp": 50,     "volume": 10.0, "compound": False, "name": "5.  Scalper Safety Net  (50pt)"},
    {"sl": 500,    "tp": 500,    "volume": 1.0,  "compound": False, "name": "6.  Conservative (1 lot)"},
    {"sl": 500,    "tp": 500,    "volume": 5.0,  "compound": False, "name": "7.  Medium (5 lots)"},
    {"sl": 500,    "tp": 500,    "volume": 10.0, "compound": True,  "name": "8.  Compounding 10 lots"},
    {"sl": 500,    "tp": 500,    "volume": 5.0,  "compound": True,  "name": "9.  Compounding 5 lots"},
    {"sl": 50,     "tp": 50,     "volume": 10.0, "compound": True,  "name": "10. Compounding Scalper (50pt)"},
]

print("\n=========================================================================================")
print("                             MT5 HYPERPARAMETER GRID SEARCH                              ")
print("=========================================================================================")
print(f"{'Configuration Name'.ljust(38)} | {'Net Profit'.rjust(12)} | {'Trades'.rjust(6)} | {'Win %'.rjust(5)}")
print("-----------------------------------------------------------------------------------------")

offset = len(df) - len(features_matrix)

for cfg in configs:
    sl_points    = cfg["sl"]
    tp_points    = cfg["tp"]
    base_volume  = cfg["volume"]
    compound     = cfg["compound"]
    
    balance = STARTING_BALANCE
    current_position = 0 
    entry_price = 0.0
    trades_taken = 0
    winning_trades = 0
    sl_price = 0.0
    tp_price = 0.0
    current_volume = 0.0

    for i in range(WINDOW_SIZE, len(features_matrix)):
        obs = np.array([features_matrix[i - WINDOW_SIZE : i]], dtype=np.float32)
        
        current_candle = df.iloc[i + offset]
        high  = current_candle['High']
        low   = current_candle['Low']
        close = current_candle['Close']
        
        # SL/TP safety net checks
        if current_position == 1:  # Long
            if low <= sl_price:
                profit = (sl_price - entry_price) * current_volume
                balance += profit
                trades_taken += 1
                if profit > 0: winning_trades += 1
                current_position = 0
            elif high >= tp_price:
                profit = (tp_price - entry_price) * current_volume
                balance += profit
                trades_taken += 1
                if profit > 0: winning_trades += 1
                current_position = 0
                
        elif current_position == 2:  # Short
            if high >= sl_price:
                profit = (entry_price - sl_price) * current_volume
                balance += profit
                trades_taken += 1
                if profit > 0: winning_trades += 1
                current_position = 0
            elif low <= tp_price:
                profit = (entry_price - tp_price) * current_volume
                balance += profit
                trades_taken += 1
                if profit > 0: winning_trades += 1
                current_position = 0

        # Ask AI — model is always ~99.9% decisive, no confidence filter needed
        action, _ = model.predict(obs, deterministic=True)
        action = int(action[0])
        
        # Proportional equity scaling: if compound mode, lot size grows with account
        if compound:
            scale = balance / STARTING_BALANCE
            final_volume = round(base_volume * scale, 2)
            final_volume = max(0.5, min(220.0, final_volume))
        else:
            final_volume = base_volume
        
        if action == 0 and current_position != 1:  # Go Long
            if current_position == 2:  # Close short at market first
                profit = (entry_price - close) * current_volume
                balance += profit
                trades_taken += 1
                if profit > 0: winning_trades += 1
            entry_price = close
            sl_price = entry_price - sl_points
            tp_price = entry_price + tp_points
            current_volume = final_volume
            current_position = 1
            
        elif action == 1 and current_position != 2:  # Go Short
            if current_position == 1:  # Close long at market first
                profit = (close - entry_price) * current_volume
                balance += profit
                trades_taken += 1
                if profit > 0: winning_trades += 1
            entry_price = close
            sl_price = entry_price + sl_points
            tp_price = entry_price - tp_points
            current_volume = final_volume
            current_position = 2

    # Close any open position at end of data
    if current_position == 1:
        profit = (close - entry_price) * current_volume
        balance += profit
        trades_taken += 1
        if profit > 0: winning_trades += 1
    elif current_position == 2:
        profit = (entry_price - close) * current_volume
        balance += profit
        trades_taken += 1
        if profit > 0: winning_trades += 1

    net_profit = balance - STARTING_BALANCE
    win_rate = (winning_trades / trades_taken) * 100 if trades_taken > 0 else 0
    print(f"{cfg['name'].ljust(38)} | ${net_profit:>11.2f} | {trades_taken:>6} | {win_rate:>4.1f}%")

print("=========================================================================================\n")

mt5.shutdown()
