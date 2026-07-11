import numpy as np
import pandas as pd
from stable_baselines3 import PPO
import warnings

warnings.filterwarnings("ignore")

STARTING_BALANCE = 10000.0
VOLUME = 10.0  # 10 Lots
SL_POINTS = 60.0
TP_POINTS = 60.0
WINDOW_SIZE = 96 # AI observation window

print("Loading data...")
df = pd.read_csv('v100_crusher_archive/advanced_features.csv')

print("Loading AI Brain...")
model = PPO.load("v100_crusher_archive/ppo_deriv_bot.zip")

# We will split the dataset into 5 equal chronological folds
n_folds = 5
fold_size = len(df) // n_folds

print(f"Total Candles: {len(df)}")
print(f"Testing {n_folds} separate folds using SCALPING (SL={SL_POINTS}, TP={TP_POINTS})...\n")

total_profit = 0

feature_cols = ['Close', 'High', 'Low', 'Open', 'Log_Return', 'GARCH_Vol', 'ARIMA_Resid']

for fold in range(n_folds):
    start_idx = fold * fold_size
    end_idx = start_idx + fold_size if fold < n_folds - 1 else len(df)
    
    df_fold = df.iloc[start_idx:end_idx].reset_index(drop=True)
    features_matrix = df_fold[feature_cols].values
    
    balance = STARTING_BALANCE
    current_position = 0 
    entry_price = 0.0
    trades_taken = 0
    winning_trades = 0
    
    sl_price = 0.0
    tp_price = 0.0
        
    for i in range(WINDOW_SIZE, len(features_matrix)):
        obs = np.array([features_matrix[i - WINDOW_SIZE : i]], dtype=np.float32)
        
        high = df_fold.iloc[i]['High']
        low = df_fold.iloc[i]['Low']
        close = df_fold.iloc[i]['Close']
        
        # Check SL/TP using High/Low of the candle
        if current_position == 1: # Long
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
                
        elif current_position == 2: # Short
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

        # Ask AI for decision
        action, _ = model.predict(obs, deterministic=True) 
        action = int(action[0])
        
        if action == 0 and current_position != 1: # Go Long
            if current_position == 2: # Close short first
                profit = (entry_price - close) * VOLUME
                balance += profit
                trades_taken += 1
                if profit > 0: winning_trades += 1
            entry_price = close
            sl_price = entry_price - SL_POINTS
            tp_price = entry_price + TP_POINTS
            current_position = 1
            
        elif action == 1 and current_position != 2: # Go Short
            if current_position == 1: # Close long first
                profit = (close - entry_price) * VOLUME
                balance += profit
                trades_taken += 1
                if profit > 0: winning_trades += 1
            entry_price = close
            sl_price = entry_price + SL_POINTS
            tp_price = entry_price - TP_POINTS
            current_position = 2

    # Close any open position at the end of the fold
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
    total_profit += net_profit
    win_rate = (winning_trades / trades_taken) * 100 if trades_taken > 0 else 0
    
    print(f"Fold {fold+1} (Rows {start_idx} to {end_idx}):")
    print(f"  Net Profit:   ${net_profit:+.2f}")
    print(f"  Total Trades: {trades_taken}")
    print(f"  Win Rate:     {win_rate:.1f}%")
    print("-" * 40)

print(f"Overall Net Profit across all scalping folds: ${total_profit:+.2f}")
