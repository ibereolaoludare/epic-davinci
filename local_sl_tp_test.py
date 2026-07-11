import pandas as pd
import numpy as np
from stable_baselines3 import PPO

print("Loading historical data from CSV...")
df = pd.read_csv('v100_crusher_archive/advanced_features.csv')

print("Loading AI Brain...")
model = PPO.load("v100_crusher_archive/ppo_deriv_bot.zip")

WINDOW_SIZE = 96
STARTING_BALANCE = 10000.0
VOLUME = 10.0

# Extract the exact 7 features the model expects
feature_cols = ['Close', 'High', 'Low', 'Open', 'Log_Return', 'GARCH_Vol', 'ARIMA_Resid']
features_matrix = df[feature_cols].values

configs = [
    (None, None),  # No SL/TP
    (50, 50),
    (60, 60),
    (100, 100),
    (200, 200),
    (500, 500)
]

print("\nRunning simulations for different SL/TP distances...\n")

for sl_points, tp_points in configs:
    balance = STARTING_BALANCE
    current_position = 0
    entry_price = 0.0
    trades_taken = 0
    winning_trades = 0
    
    sl_price = 0.0
    tp_price = 0.0

    for i in range(WINDOW_SIZE, len(features_matrix)):
        obs = np.array([features_matrix[i - WINDOW_SIZE : i]], dtype=np.float32)
        
        high = df.iloc[i]['High']
        low = df.iloc[i]['Low']
        close = df.iloc[i]['Close']
        
        if sl_points is not None and tp_points is not None:
            # Check SL/TP
            if current_position == 1:
                if low <= sl_price:
                    profit = (sl_price - entry_price) * VOLUME
                    balance += profit
                    trades_taken += 1
                    if profit > 0: winning_trades += 1
                    current_position = 0
                elif high >= tp_price:
                    profit = (tp_price - entry_price) * VOLUME
                    balance += profit
                    trades_taken += 1
                    if profit > 0: winning_trades += 1
                    current_position = 0
            elif current_position == 2:
                if high >= sl_price:
                    profit = (entry_price - sl_price) * VOLUME
                    balance += profit
                    trades_taken += 1
                    if profit > 0: winning_trades += 1
                    current_position = 0
                elif low <= tp_price:
                    profit = (entry_price - tp_price) * VOLUME
                    balance += profit
                    trades_taken += 1
                    if profit > 0: winning_trades += 1
                    current_position = 0

        action, _ = model.predict(obs, deterministic=True) 
        action = int(action[0])
        
        if action == 0 and current_position != 1:
            if current_position == 2:
                profit = (entry_price - close) * VOLUME
                balance += profit
                trades_taken += 1
                if profit > 0: winning_trades += 1
            entry_price = close
            if sl_points:
                sl_price = entry_price - sl_points
                tp_price = entry_price + tp_points
            current_position = 1
            
        elif action == 1 and current_position != 2:
            if current_position == 1:
                profit = (close - entry_price) * VOLUME
                balance += profit
                trades_taken += 1
                if profit > 0: winning_trades += 1
            entry_price = close
            if sl_points:
                sl_price = entry_price + sl_points
                tp_price = entry_price - tp_points
            current_position = 2

    # Force close at end
    if current_position == 1:
        profit = (close - entry_price) * VOLUME
        balance += profit
        trades_taken += 1
        if profit > 0: winning_trades += 1
    elif current_position == 2:
        profit = (entry_price - close) * VOLUME
        balance += profit
        trades_taken += 1
        if profit > 0: winning_trades += 1

    net_profit = balance - STARTING_BALANCE
    win_rate = (winning_trades / trades_taken) * 100 if trades_taken > 0 else 0
    
    label = f"SL/TP: {sl_points if sl_points else 'NONE'}"
    print(f"{label.ljust(15)} | Net Profit: ${net_profit:+.2f} | Trades: {trades_taken} | Win Rate: {win_rate:.1f}%")
