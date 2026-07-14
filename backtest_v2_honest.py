"""
Honest Out-of-Sample Backtest v2
=================================
Tests on candles 17,500–end. Neither the PPO nor the Ensemble
has EVER seen this data. Features were computed with GARCH/HMM
fitted only on candles 0–10,000. This is the real test.
"""

import pandas as pd
import numpy as np
import joblib
from sb3_contrib import RecurrentPPO
from feature_engineering_v2 import FEATURE_COLS

# ── Boundaries (must match train scripts) ─────────
PPO_TRAIN_END = 10000
ENSEMBLE_TRAIN_END = 17500


def run_honest_backtest():
    print("Loading leak-free features v2...")
    df_all = pd.read_csv("features_v2_strict.csv")

    print("Loading models...")
    ppo = RecurrentPPO.load("ppo_v2_lstm_strict")

    # Feature cols — canonical order
    feature_cols = [c for c in FEATURE_COLS if c in df_all.columns]

    # ── Slice COMPLETELY UNSEEN test data ─────────────────────────
    # We will just test on everything past 10000, since we aren't using the ensemble
    TEST_START = 10000
    df = df_all.iloc[TEST_START:].reset_index(drop=True)
    print(f"Test data: candles {TEST_START}–{len(df_all)} ({len(df)} candles)")
    print(f"Date range: {df['Time'].min()} to {df['Time'].max()}")

    obs_df = df[feature_cols]
    prices = df['Close'].values

    balance = 0.0
    current_position = 0 # 1: Long, 0: Flat, -1: Short
    entry_price = 0.0
    trades_taken = 0
    wins = 0
    trade_log = []
    
    commission = 20.0 # Spread penalty for changing positions

    print(f"Running continuous dynamic backtest...\n")

    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)

    for i in range(96, len(df)):
        obs_window = obs_df.iloc[i-96:i].values
        obs = np.array(obs_window, dtype=np.float32)

        # 1. PPO decides position continuously
        action, lstm_states = ppo.predict(
            obs,
            state=lstm_states,
            episode_start=episode_starts,
            deterministic=True
        )
        episode_starts = np.zeros((1,), dtype=bool)
        
        action = int(action)
        # 0: Short (-1), 1: Flat (0), 2: Long (1)
        target_position = action - 1

        close = prices[i]

        if current_position != target_position:
            # Close old position
            if current_position != 0:
                if current_position == 1:
                    profit = close - entry_price
                elif current_position == -1:
                    profit = entry_price - close
                balance += profit
                trades_taken += 1
                if profit > 0: wins += 1
                trade_log.append(profit)
                
            # Pay spread/commission on state change
            balance -= commission
            
            # Open new position
            if target_position != 0:
                entry_price = close
                
            current_position = target_position

    # Close any open position at end
    if current_position != 0:
        if current_position == 1:
            profit = prices[-1] - entry_price
        elif current_position == -1:
            profit = entry_price - prices[-1]
        balance += profit
        trades_taken += 1
        if profit > 0: wins += 1
        trade_log.append(profit)

    win_rate = (wins / trades_taken * 100) if trades_taken > 0 else 0

    print("=" * 60)
    print("   HONEST OUT-OF-SAMPLE RESULTS (Continuous - Zero Leakage)")
    print("=" * 60)
    print(f"  PPO trained on:        candles 0-{PPO_TRAIN_END}")
    print(f"  TEST DATA:             candles {TEST_START}-{len(df_all)}")
    print(f"  Date Range:            {df['Time'].min()} to {df['Time'].max()}")
    print("-" * 60)
    print(f"  Total Trades Taken:    {trades_taken}")
    print(f"  Wins:                  {wins}")
    print(f"  Win Rate:              {win_rate:.1f}%")
    print(f"  Net Profit:            {balance:.1f} points")
    print("=" * 60)

    if len(trade_log) > 0:
        arr = np.array(trade_log)
        cum = np.cumsum(arr)
        max_dd = np.min(cum - np.maximum.accumulate(cum))
        print(f"\n  Avg Trade:             {np.mean(arr):.1f} pts")
        print(f"  Max Drawdown:          {max_dd:.1f} pts")
        profit_factor = np.sum(arr[arr>0]) / max(-np.sum(arr[arr<0]), 1)
        print(f"  Profit Factor:         {profit_factor:.2f}")


if __name__ == "__main__":
    run_honest_backtest()
