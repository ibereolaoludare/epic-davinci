import pandas as pd
import numpy as np
import joblib
from stable_baselines3 import PPO

def optimize():
    print("Loading data and models...")
    df = pd.read_csv("advanced_features_strict.csv")
    
    # Use the last 10,000 candles for optimization testing
    df_test = df.iloc[-10000:].reset_index(drop=True)
    
    feature_cols = [col for col in df_test.columns if col not in ['Time', 'Open', 'High', 'Low', 'Close', 'KF_Close', 'EMA_50']]
    obs_df = df_test[feature_cols]
    
    prices = df_test['Close'].values
    highs = df_test['High'].values
    lows = df_test['Low'].values
    
    model = PPO.load("ppo_deriv_bot")
    rf = joblib.load("meta_model.pkl")
    
    print("Pre-calculating AI Signals and Vetoes (this takes a few seconds)...")
    actions = np.zeros(len(df_test), dtype=int)
    vetoes = np.zeros(len(df_test), dtype=bool)
    
    for i in range(96, len(df_test)):
        obs_window = obs_df.iloc[i-96:i].values
        obs = np.array(obs_window, dtype=np.float32)
        
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        actions[i] = action
        
        latest_features = obs[-1]
        meta_input = list(latest_features) + [action]
        meta_input_df = pd.DataFrame([meta_input], columns=[f"feat_{j}" for j in range(len(latest_features))] + ['direction'])
        win_prob = rf.predict_proba(meta_input_df)[0][1]
        
        if win_prob < 0.50:
            vetoes[i] = True

    print("Starting Grid Search...")
    best_profit = -999999
    best_params = (0, 0)
    best_stats = {}
    
    # SL and TP ranges from 100 to 1000 in steps of 100
    for SL_POINTS in range(100, 1001, 100):
        for TP_POINTS in range(100, 1001, 100):
            
            balance = 0.0
            current_position = 0
            entry_price = 0.0
            trades_taken = 0
            wins = 0
            
            for i in range(96, len(df_test)):
                close = prices[i]
                high = highs[i]
                low = lows[i]
                
                action = actions[i]
                veto = vetoes[i]
                
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
                        
            # End of test, close position
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
            
            # Print significant results
            if trades_taken > 5 and balance > 0:
                print(f"SL: {SL_POINTS:4d} | TP: {TP_POINTS:4d} | Trades: {trades_taken:3d} | Win Rate: {win_rate:5.1f}% | Net Profit: {balance:8.1f} pts")
                
            if trades_taken > 5 and balance > best_profit:
                best_profit = balance
                best_params = (SL_POINTS, TP_POINTS)
                best_stats = {'trades': trades_taken, 'win_rate': win_rate, 'profit': balance}
                
    print("\n========================================================")
    print("                 OPTIMIZATION COMPLETE                  ")
    print("========================================================")
    print(f"BEST CONFIGURATION:")
    print(f"Stop Loss (SL):    {best_params[0]} points")
    print(f"Take Profit (TP):  {best_params[1]} points")
    print(f"Total Trades:      {best_stats['trades']}")
    print(f"Win Rate:          {best_stats['win_rate']:.1f}%")
    print(f"Total Net Profit:  {best_stats['profit']:.1f} points")
    print("========================================================")

if __name__ == "__main__":
    optimize()
