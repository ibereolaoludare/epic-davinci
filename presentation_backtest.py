import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

if password:
    mt5.login(login, password=password, server=server)

symbol = "Volatility 100 Index"
# Fetching UNSEEN data: from candle 30,000 back to 45,000 (roughly 15,000 candles from over a year ago)
print(f"Fetching 15,000 strictly UNSEEN M15 candles for {symbol} from MT5 (offset by 30,000 candles)...")
rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 30000, 15000)

if rates is None or len(rates) == 0:
    print("Failed to fetch rates from MT5.")
    mt5.shutdown()
    quit()

df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')
df = df.rename(columns={'time': 'Time', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'})

print(f"Data Date Range: {df['Time'].min()} to {df['Time'].max()}")
print("Processing features (this may take a minute)...")
features_matrix = process_features(df)

print("Loading AI Brain...")
model = PPO.load("v100_crusher_archive/ppo_deriv_bot.zip")

WINDOW_SIZE = 96
STARTING_BALANCE = 10000.0
VOLUME = 10.0
SL_POINTS = 60.0
TP_POINTS = 60.0

print("\n=========================================================================================")
print("                   UNSEEN DATA WALK-FORWARD BACKTEST (PRESENTATION)                      ")
print("=========================================================================================")

offset = len(df) - len(features_matrix)

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

for i in range(WINDOW_SIZE, len(features_matrix)):
    obs = np.array([features_matrix[i - WINDOW_SIZE : i]], dtype=np.float32)
    
    current_candle = df.iloc[i + offset]
    high  = current_candle['High']
    low   = current_candle['Low']
    close = current_candle['Close']
    current_time = current_candle['Time']
    
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

    # Ask AI 
    action, _ = model.predict(obs, deterministic=True)
    action = int(action[0])
    
    if action == 0 and current_position != 1:  # Go Long
        if current_position == 2:  # Close short at market first
            profit = (entry_price - close) * current_volume
            balance += profit
            trades_taken += 1
            if profit > 0: winning_trades += 1
        entry_price = close
        sl_price = entry_price - SL_POINTS
        tp_price = entry_price + TP_POINTS
        current_position = 1
        
    elif action == 1 and current_position != 2:  # Go Short
        if current_position == 1:  # Close long at market first
            profit = (close - entry_price) * current_volume
            balance += profit
            trades_taken += 1
            if profit > 0: winning_trades += 1
        entry_price = close
        sl_price = entry_price + SL_POINTS
        tp_price = entry_price - TP_POINTS
        current_position = 2

    # Track floating equity for charting
    floating_profit = 0
    if current_position == 1:
        floating_profit = (close - entry_price) * current_volume
    elif current_position == 2:
        floating_profit = (entry_price - close) * current_volume
        
    equity_curve.append(balance + floating_profit)
    time_axis.append(current_time)

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

print(f"Total Trades: {trades_taken}")
print(f"Win Rate:     {win_rate:.1f}%")
print(f"Net Profit:   ${net_profit:,.2f}")
print("=========================================================================================\n")

mt5.shutdown()

# Generate presentation plot
plt.figure(figsize=(12, 6))
plt.plot(time_axis, equity_curve, color='blue', linewidth=1.5)
plt.title(f'AI Performance on UNSEEN Data ({symbol})\n{df["Time"].min().date()} to {df["Time"].max().date()}', fontsize=16)
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
plot_path = "unseen_data_presentation.png"
plt.savefig(plot_path, dpi=300)
print(f"Saved beautiful presentation plot to {plot_path}")
