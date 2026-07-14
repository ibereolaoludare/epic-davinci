"""
Train PPO v2 — Strict Temporal Isolation
=========================================
PPO is trained ONLY on candles 0–10,000 from the leak-free feature set.
This ensures candles 10,000+ are completely unseen by the PPO.
"""

import numpy as np
import pandas as pd
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv
from trading_env import DerivTradingEnv
from feature_engineering_v2 import FEATURE_COLS

if __name__ == "__main__":
    print("Loading leak-free features (v2)...")
    df = pd.read_csv('features_v2_strict.csv')
    print(f"Total data: {len(df)} candles")

    # ── STRICT SPLIT: Train PPO on first 10,000 candles ONLY ──────
    PPO_TRAIN_END = 10000
    df_train = df.iloc[:PPO_TRAIN_END].reset_index(drop=True)

    # Drop columns the env doesn't use as features
    df_train_env = df_train.drop(columns=['Time'], errors='ignore')

    print(f"PPO Training Data: candles 0–{PPO_TRAIN_END} ({len(df_train)} candles)")
    print(f"Reserved for Ensemble + Test: candles {PPO_TRAIN_END}–{len(df)} ({len(df) - PPO_TRAIN_END} candles)")

    # Verify feature columns match canonical order
    env_feature_cols = [c for c in df_train_env.columns if c not in ['Open', 'High', 'Low', 'Close', 'KF_Close', 'EMA_50']]
    print(f"Feature cols ({len(env_feature_cols)}): {env_feature_cols}")

    # Setup environment
    env = DummyVecEnv([lambda: DerivTradingEnv(df=df_train_env, window_size=96)])

    # Train from scratch
    print("Initializing RecurrentPPO (LSTM) from scratch...")
    policy_kwargs = dict(lstm_hidden_size=64, n_lstm_layers=2)
    model = RecurrentPPO("MlpLstmPolicy", env, verbose=1, learning_rate=0.00005,
                         ent_coef=0.0001, batch_size=128, device="auto", policy_kwargs=policy_kwargs)

    print("Training RecurrentPPO on leak-free data...")
    model.learn(total_timesteps=1000000)

    model.save("ppo_v2_lstm_strict")
    print("Saved ppo_v2_lstm_strict!")
