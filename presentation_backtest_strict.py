import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from stable_baselines3 import PPO
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
import warnings
import os
from feature_engineering import process_features

warnings.filterwarnings("ignore")

print("Initializing MT5...")
if not mt5.initialize():
    print("MT5 initialization failed")
    quit()

login = int(os.getenv("MT5_LOGIN", "32312791"))
password = os.getenv("MT5_PASSWORD")
server = os.getenv("MT5_SERVER", "Deriv-Demo")

if password:
    mt5.login(login, password=password, server=server)

symbol = "Volatility 100 Index"
# Fetch 6000 candles: 5000 for the initial lookback, and 1000 for the walk-forward backtest (approx 10.4 days)
print(f"Fetching 6,000 strictly UNSEEN M15 candles for {symbol} from MT5 (offset by 30,000 candles)...")
rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 30000, 6000)

if rates is None or len(rates) == 0:
    print("Failed to fetch rates from MT5.")
    mt5.shutdown()
    quit()

df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')
df = df.rename(columns={'time': 'Time', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'})

print(f"Data Date Range: {df['Time'].min()} to {df['Time'].max()}")

print("Loading AI Brain...")
model = PPO.load("ppo_deriv_bot.zip")

STARTING_BALANCE = 10000.0
VOLUME = 10.0
SL_POINTS = 60.0
TP_POINTS = 60.0

print("\n=========================================================================================")
print("             STRICT UNSEEN DATA WALK-FORWARD BACKTEST (EXACT LIVE REPLICA)               ")
print("=========================================================================================")
print("This simulates the exact live environment by re-fitting GARCH & ARIMA on a sliding 5,000 candle window for EVERY step.")
print("Processing 1,000 steps. This will take ~5 minutes...")

balance = STARTING_BALANCE
equity_curve = []
time_axis = []

current_position = 0 
entry_price = 0.0
trades_taken = 0
winning_trades = 0
sl_price = 0.0
tp_price = 0.0
current_volume = VOLUME
entry_volume = VOLUME

start_time = time.time()

# Walk-forward starting at index 5000
for i in range(5000, len(df)):
    
    if i % 100 == 0:
        elapsed = time.time() - start_time
        print(f"Processed {i - 5000}/1000 steps... ({elapsed:.1f}s elapsed)")
    
    # 1. STRICT LIVE REPLICA: Slice the exactly preceding 5000 candles
    df_slice = df.iloc[i - 5000 : i]
    
    # 2. Re-fit models on this exact 5000 candle window (takes ~0.3s per step)
    features_matrix = process_features(df_slice)
    
    # 3. Take the last 96 rows for the observation
    obs = np.array([features_matrix[-96:]], dtype=np.float32)
    
    # The active candle for the backtest
    current_candle = df.iloc[i]
    high  = current_candle['High']
    low   = current_candle['Low']
    close = current_candle['Close']
    current_time = current_candle['Time']
    
    # SL/TP safety net checks
    if current_position == 1:  # Long
        if low <= sl_price:
            profit = (sl_price - entry_price) * entry_volume
            balance += profit
            trades_taken += 1
            if profit > 0: winning_trades += 1
            current_position = 0
        elif high >= tp_price:
            profit = (tp_price - entry_price) * entry_volume
            balance += profit
            trades_taken += 1
            if profit > 0: winning_trades += 1
            current_position = 0
            
    elif current_position == 2:  # Short
        if high >= sl_price:
            profit = (entry_price - sl_price) * entry_volume
            balance += profit
            trades_taken += 1
            if profit > 0: winning_trades += 1
            current_position = 0
        elif low <= tp_price:
            profit = (entry_price - tp_price) * entry_volume
            balance += profit
            trades_taken += 1
            if profit > 0: winning_trades += 1
            current_position = 0

    # Ask AI 
    action, _ = model.predict(obs, deterministic=True)
    action = int(action[0])
    
    # --- Volatility-Targeted Position Sizing ---
    current_garch_vol = features_matrix[-1][6]
    
    target_volatility = 0.01 # Baseline 1% volatility target
    if current_garch_vol > 0:
        vol_multiplier = target_volatility / current_garch_vol
    else:
        vol_multiplier = 1.0
        
    vol_multiplier = max(0.2, min(2.5, vol_multiplier))
    
    base_volume = 10.0
    current_volume = round(base_volume * vol_multiplier, 2)
    # -------------------------------------------
    
    # --- Meta-Labeling (The Veto Network) ---
    meta_veto = False
    if os.path.exists("meta_model.pkl"):
        try:
            import joblib
            rf = joblib.load("meta_model.pkl")
            # Build feature row: all 14 features + the proposed direction
            # The exact order MUST match train_meta_model.py: [feat_0...feat_13, direction]
            meta_input = list(features_matrix[-1]) + [action]
            meta_input_df = pd.DataFrame([meta_input], columns=[f"feat_{i}" for i in range(len(features_matrix[-1]))] + ['direction'])
            win_prob = rf.predict_proba(meta_input_df)[0][1] # Probability of Class 1 (Win)
            if win_prob < 0.50:
                meta_veto = True
        except Exception as e:
            pass # Ignore if missing
    # ----------------------------------------
    
    if action == 0 and current_position != 1 and not meta_veto:  # Go Long
        if current_position == 2:  # Close short at market first
            profit = (entry_price - close) * entry_volume
            balance += profit
            trades_taken += 1
            if profit > 0: winning_trades += 1
        entry_price = close
        entry_volume = current_volume
        sl_price = entry_price - SL_POINTS
        tp_price = entry_price + TP_POINTS
        current_position = 1
        
    elif action == 1 and current_position != 2 and not meta_veto:  # Go Short
        if current_position == 1:  # Close long at market first
            profit = (close - entry_price) * entry_volume
            balance += profit
            trades_taken += 1
            if profit > 0: winning_trades += 1
        entry_price = close
        entry_volume = current_volume
        sl_price = entry_price + SL_POINTS
        tp_price = entry_price - TP_POINTS
        current_position = 2

    # Track floating equity for charting
    floating_profit = 0
    if current_position == 1:
        floating_profit = (close - entry_price) * entry_volume
    elif current_position == 2:
        floating_profit = (entry_price - close) * entry_volume
        
    equity_curve.append(balance + floating_profit)
    time_axis.append(current_time)

# Close any open position at end of data
if current_position == 1:
    profit = (close - entry_price) * entry_volume
    balance += profit
    trades_taken += 1
    if profit > 0: winning_trades += 1
elif current_position == 2:
    profit = (entry_price - close) * entry_volume
    balance += profit
    trades_taken += 1
    if profit > 0: winning_trades += 1

net_profit = balance - STARTING_BALANCE
win_rate = (winning_trades / trades_taken) * 100 if trades_taken > 0 else 0

print(f"\nTotal Trades: {trades_taken}")
print(f"Win Rate:     {win_rate:.1f}%")
print(f"Net Profit:   ${net_profit:,.2f}")
print("=========================================================================================\n")

mt5.shutdown()

# Generate presentation plot
plt.figure(figsize=(12, 6))
plt.plot(time_axis, equity_curve, color='blue', linewidth=1.5)
plt.title(f'Strict Live-Replica AI Performance on UNSEEN Data\n{time_axis[0].date()} to {time_axis[-1].date()}', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Account Equity ($)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.fill_between(time_axis, equity_curve, STARTING_BALANCE, where=(np.array(equity_curve) >= STARTING_BALANCE), color='green', alpha=0.1)
plt.fill_between(time_axis, equity_curve, STARTING_BALANCE, where=(np.array(equity_curve) < STARTING_BALANCE), color='red', alpha=0.1)
plt.axhline(y=STARTING_BALANCE, color='black', linestyle='-', linewidth=1)

# Add stats box
stats_text = f"Trades: {trades_taken}\nWin Rate: {win_rate:.1f}%\nNet Profit: ${net_profit:,.2f}"
plt.text(0.02, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=12,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plot_path = "strict_unseen_data_presentation.png"
plt.savefig(plot_path, dpi=300)
print(f"Saved strict presentation plot to {plot_path}")
