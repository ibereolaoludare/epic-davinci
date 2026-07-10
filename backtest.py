import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

# =========================
# Backtest Configuration
# =========================
STARTING_BALANCE = 10000.0
TRADE_SIZE = 10  # Number of contracts/units
SPREAD = 0.5     # Points lost to spread per trade
WINDOW_SIZE = 30 # AI observation window

# Load Data
print("Loading advanced quantitative features...")
df = pd.read_csv('advanced_features.csv')
times_all = pd.to_datetime(df['Time'].values)
df = df.drop(columns=['Time'], errors='ignore')

# 80/20 Train/Test Split (Prevent Overfitting)
split_index = int(len(df) * 0.8)
df_test = df.iloc[split_index:].reset_index(drop=True)
times = times_all[split_index:]

print(f"Running strict Out-Of-Sample backtest on final {len(df_test)} candles...")

# Load AI Brain
print("Loading AI Brain...")
try:
    model = PPO.load("ppo_deriv_bot")
except Exception as e:
    print("Error: Could not load 'ppo_deriv_bot.zip'.")
    quit()

# Simulation Variables
balance = STARTING_BALANCE
equity_curve = []
current_position = 0 # 0=Flat, 1=Long, 2=Short
entry_price = 0.0
trades_taken = 0
winning_trades = 0

print("Starting Simulation Loop...")

# Pre-fill the 30-step observation window with the first 30 rows of test data
price_history = []
for i in range(WINDOW_SIZE):
    price_history.append(df_test.iloc[i].values.tolist())

# Start stepping through time from index WINDOW_SIZE onwards
for i in range(WINDOW_SIZE, len(df_test)):
    current_row = df_test.iloc[i]
    current_price = current_row['Close']
    
    # Update sliding window
    price_history.append(current_row.values.tolist())
    price_history.pop(0)
    
    # Calculate ongoing PnL for equity curve
    unrealized_pnl = 0.0
    if current_position == 1: # Long
        unrealized_pnl = (current_price - entry_price) * TRADE_SIZE
    elif current_position == 2: # Short
        unrealized_pnl = (entry_price - current_price) * TRADE_SIZE
        
    equity_curve.append(balance + unrealized_pnl)
    
    # Ask AI for decision
    obs = np.array([price_history], dtype=np.float32)
    # Turn OFF deterministic mode so it can explore. In true production, this is True.
    action, _states = model.predict(obs, deterministic=False) 
    action = action[0]
    
    # Execute Trades based on AI action
    if action == 1 and current_position != 1: # Go Long
        if current_position == 2:
            profit = (entry_price - current_price) * TRADE_SIZE - (SPREAD * TRADE_SIZE)
            balance += profit
            trades_taken += 1
            if profit > 0: winning_trades += 1
        entry_price = current_price
        current_position = 1
        balance -= (SPREAD * TRADE_SIZE) 
        
    elif action == 2 and current_position != 2: # Go Short
        if current_position == 1:
            profit = (current_price - entry_price) * TRADE_SIZE - (SPREAD * TRADE_SIZE)
            balance += profit
            trades_taken += 1
            if profit > 0: winning_trades += 1
        entry_price = current_price
        current_position = 2
        balance -= (SPREAD * TRADE_SIZE) 
        
    elif action == 0 and current_position != 0: # Close positions and hold flat
        if current_position == 1:
            profit = (current_price - entry_price) * TRADE_SIZE - (SPREAD * TRADE_SIZE)
            balance += profit
            trades_taken += 1
            if profit > 0: winning_trades += 1
        elif current_position == 2:
            profit = (entry_price - current_price) * TRADE_SIZE - (SPREAD * TRADE_SIZE)
            balance += profit
            trades_taken += 1
            if profit > 0: winning_trades += 1
        current_position = 0

# Print Stats
net_profit = balance - STARTING_BALANCE
win_rate = (winning_trades / trades_taken) * 100 if trades_taken > 0 else 0

print("====================================")
print("       BACKTEST RESULTS             ")
print("====================================")
print(f"Starting Balance: ${STARTING_BALANCE:.2f}")
print(f"Ending Balance:   ${balance:.2f}")
print(f"Net Profit:       ${net_profit:.2f}")
print(f"Total Trades:     {trades_taken}")
print(f"Win Rate:         {win_rate:.1f}%")
print("====================================")

# Plot Equity Curve
plt.figure(figsize=(10, 5))
# Make sure x-axis matches equity curve length
plot_times = times[WINDOW_SIZE:WINDOW_SIZE+len(equity_curve)]
plt.plot(plot_times, equity_curve, label='Account Equity', color='blue')
plt.title('AI Trading Bot - STRICT OUT-OF-SAMPLE Equity Curve')
plt.xlabel('Time')
plt.ylabel('Account Balance ($)')
plt.grid(True)
plt.legend()
plt.savefig('backtest_results.png')
print("Saved chart to 'backtest_results.png'")
