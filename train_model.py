import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from trading_env import DerivTradingEnv
import os

if __name__ == "__main__":
    print("Loading advanced quantitative features (STRICT ZERO-LEAKAGE DATASET)...")
    df = pd.read_csv('advanced_features_strict.csv')
    
    # Drop Time column for the AI
    df = df.drop(columns=['Time'], errors='ignore')
    
    # 80/20 Train/Test Split (Prevent Overfitting)
    split_index = int(len(df) * 0.8)
    df_train = df.iloc[:split_index].reset_index(drop=True)
    
    print(f"Total Data: {len(df)} candles")
    print(f"Training Data: {len(df_train)} candles (80%)")
    print(f"Reserved for Backtesting: {len(df) - len(df_train)} candles (20%)")
    
    # 2. Setup Environment
    # We wrap it in a DummyVecEnv which is required by Stable-Baselines
    env = DummyVecEnv([lambda: DerivTradingEnv(df=df_train, window_size=96)])
    
    # 3. Initialize or Load the AI Brain (PPO)
    print("Initializing brand new PPO Neural Network from scratch to ensure a clean slate...")
    # Using 'MlpPolicy' since our data is tabular numeric data
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.00005, ent_coef=0.0001, batch_size=128, device="auto")
    
    # 4. Train the AI
    print("Starting Training! This is where your CPU/GPU works hard...")
    model.learn(total_timesteps=2000000)
    
    # 5. Save the trained brain
    print("Training complete. Saving AI brain to 'ppo_deriv_bot'...")
    model.save("ppo_deriv_bot")
    
    print("Successfully saved! You can now run backtests.")
