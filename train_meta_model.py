import pandas as pd
import numpy as np
import joblib
from stable_baselines3 import PPO
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import os

def train_meta_model():
    print("Loading pre-calculated features...")
    df = pd.read_csv("advanced_features_strict.csv")
    
    print("Loading trained PPO model...")
    try:
        model = PPO.load("ppo_deriv_bot")
    except Exception as e:
        print(f"Error loading PPO model: {e}")
        return

    # Backtest configuration
    # Use the test split (last 20%) to train the meta-model so it learns on out-of-sample data!
    split_index = int(len(df) * 0.8)
    df_test = df.iloc[split_index:].reset_index(drop=True)
    
    prices = df_test['Close'].values
    highs = df_test['High'].values
    lows = df_test['Low'].values
    
    # Exclude 'Time' and 'Close', 'High', 'Low' etc. for observations if they are not in training.
    feature_cols = [col for col in df_test.columns if col not in ['Time', 'Open', 'High', 'Low', 'Close', 'KF_Close', 'EMA_50']]
    obs_df = df_test[feature_cols]
    
    X_meta = []
    y_meta = []
    
    print(f"Running vectorized backtest over {len(df_test)} candles for Meta-Labeling...")
    
    current_position = 0
    entry_price = 0
    trade_start_idx = 0
    trade_features = None
    trade_direction = 0
    
    SL_POINTS = 60
    TP_POINTS = 60
    
    # Play out the trades
    # Start from index 96 to have a full window
    for i in range(96, len(df_test)):
        # Construct the 96-step window
        obs_window = obs_df.iloc[i-96:i].values
        # The model expects shape (n_env, 96, 14). We can pass (96, 14) and SB3 handles the batch dimension if needed.
        # Ensure it is a numpy array of shape (96, 14)
        obs = np.array(obs_window, dtype=np.float32)
        
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        
        close = prices[i]
        
        # We need the most recent row's features for the Meta-Model training data
        latest_features = obs[-1]
        
        # Check if SL/TP hit on existing position
        if current_position == 1:
            if lows[i] <= entry_price - SL_POINTS:
                y_meta.append(0) # Loss
                current_position = 0
            elif highs[i] >= entry_price + TP_POINTS:
                y_meta.append(1) # Win
                current_position = 0
        elif current_position == 2:
            if highs[i] >= entry_price + SL_POINTS:
                y_meta.append(0) # Loss
                current_position = 0
            elif lows[i] <= entry_price - TP_POINTS:
                y_meta.append(1) # Win
                current_position = 0
                
        # Handle signal flips or new entries
        if current_position == 0:
            if action == 0: # Go long
                current_position = 1
                entry_price = close
                trade_features = list(latest_features) + [action]
                X_meta.append(trade_features)
            elif action == 1: # Go short
                current_position = 2
                entry_price = close
                trade_features = list(latest_features) + [action]
                X_meta.append(trade_features)
        else:
            # If we hold a position and get an opposite signal, it flips.
            if current_position == 1 and action == 1:
                # Close long
                profit = close - entry_price
                y_meta.append(1 if profit > 0 else 0)
                
                # Open short
                current_position = 2
                entry_price = close
                trade_features = list(latest_features) + [action]
                X_meta.append(trade_features)
                
            elif current_position == 2 and action == 0:
                # Close short
                profit = entry_price - close
                y_meta.append(1 if profit > 0 else 0)
                
                # Open long
                current_position = 1
                entry_price = close
                trade_features = list(latest_features) + [action]
                X_meta.append(trade_features)
                
    # If a trade is still open at the end, drop it as we don't know the outcome
    if len(X_meta) > len(y_meta):
        X_meta = X_meta[:len(y_meta)]
        
    print(f"Collected {len(X_meta)} trades for Meta-Model training.")
    if len(X_meta) == 0:
        print("No trades generated. Cannot train meta-model.")
        return
        
    print("Training Random Forest Veto Network...")
    feature_names = [f"feat_{j}" for j in range(len(X_meta[0])-1)] + ['direction']
    X_df = pd.DataFrame(X_meta, columns=feature_names)
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced')
    rf.fit(X_df, y_meta)
    
    preds = rf.predict(X_df)
    print("\nMeta-Model Training Accuracy on Out-Of-Sample Trades:")
    print(classification_report(y_meta, preds))
    
    joblib.dump(rf, "meta_model.pkl")
    print("Meta-model saved as 'meta_model.pkl'!")

if __name__ == "__main__":
    train_meta_model()
