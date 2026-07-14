import pandas as pd
import numpy as np
import joblib
import warnings
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score, precision_score
from sklearn.ensemble import VotingClassifier

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import shap

warnings.filterwarnings("ignore")

def extract_ppo_trades(df, model, feature_cols, SL_POINTS=60, TP_POINTS=60):
    print(f"Running vectorized PPO backtest over {len(df)} candles to extract trades...")
    obs_df = df[feature_cols]
    prices = df['Close'].values
    highs = df['High'].values
    lows = df['Low'].values
    
    X_meta = []
    y_meta = []
    
    current_position = 0
    entry_price = 0
    
    for i in range(96, len(df)):
        obs_window = obs_df.iloc[i-96:i].values
        obs = np.array(obs_window, dtype=np.float32)
        
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        close = prices[i]
        latest_features = list(obs[-1])
        
        if current_position == 1:
            if lows[i] <= entry_price - SL_POINTS:
                y_meta.append(0)
                current_position = 0
            elif highs[i] >= entry_price + TP_POINTS:
                y_meta.append(1)
                current_position = 0
        elif current_position == 2:
            if highs[i] >= entry_price + SL_POINTS:
                y_meta.append(0)
                current_position = 0
            elif lows[i] <= entry_price - TP_POINTS:
                y_meta.append(1)
                current_position = 0
                
        if current_position == 0:
            if action == 0:
                current_position = 1
                entry_price = close
                X_meta.append(latest_features + [action])
            elif action == 1:
                current_position = 2
                entry_price = close
                X_meta.append(latest_features + [action])
        else:
            if current_position == 1 and action == 1:
                profit = close - entry_price
                y_meta.append(1 if profit > 0 else 0)
                current_position = 2
                entry_price = close
                X_meta.append(latest_features + [action])
            elif current_position == 2 and action == 0:
                profit = entry_price - close
                y_meta.append(1 if profit > 0 else 0)
                current_position = 1
                entry_price = close
                X_meta.append(latest_features + [action])
                
    if len(X_meta) > len(y_meta):
        X_meta = X_meta[:len(y_meta)]
        
    return X_meta, y_meta

def train_ensemble():
    print("Loading data...")
    df = pd.read_csv("advanced_features_strict.csv")
    
    # CRITICAL: We MUST restrict training to the first 15000 candles so the last 10000 remain completely unseen!
    df = df.iloc[:15000]
    
    print("Loading PPO...")
    model = PPO.load("ppo_deriv_bot")
    
    feature_cols = [col for col in df.columns if col not in ['Time', 'Open', 'High', 'Low', 'Close', 'KF_Close', 'EMA_50']]
    
    # Extract trades using the new target 200/200 config
    X_meta, y_meta = extract_ppo_trades(df, model, feature_cols, SL_POINTS=200, TP_POINTS=200)
    print(f"Extracted {len(X_meta)} trades for Meta-Model training.")
    
    feature_names = [f"feat_{j}" for j in range(len(X_meta[0])-1)] + ['direction']
    X = pd.DataFrame(X_meta, columns=feature_names)
    y = np.array(y_meta)
    
    print("\n--- Purged Time-Series Cross Validation ---")
    tscv = TimeSeriesSplit(n_splits=5)
    
    xgb_clf = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, scale_pos_weight=(len(y)-sum(y))/sum(y), n_jobs=-1)
    lgb_clf = lgb.LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, class_weight='balanced', n_jobs=-1, verbose=-1)
    cat_clf = CatBoostClassifier(iterations=100, depth=4, learning_rate=0.05, random_seed=42, auto_class_weights='Balanced', verbose=0)
    
    ensemble = VotingClassifier(estimators=[
        ('xgb', xgb_clf),
        ('lgb', lgb_clf),
        ('cat', cat_clf)
    ], voting='soft')
    
    fold = 1
    precisions = []
    
    for train_index, test_index in tscv.split(X):
        # Implement a tiny purge by dropping the last 10 samples of train to prevent data leakage
        purge_size = 10
        if len(train_index) > purge_size:
            train_index = train_index[:-purge_size]
            
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        ensemble.fit(X_train, y_train)
        
        # Predict probabilities
        probs = ensemble.predict_proba(X_test)[:, 1]
        
        # Test strict thresholds
        for thresh in [0.50, 0.55, 0.60, 0.65]:
            preds = (probs > thresh).astype(int)
            prec = precision_score(y_test, preds, zero_division=0)
            if thresh == 0.60:
                precisions.append(prec)
                print(f"Fold {fold} - Threshold {thresh:.2f} Precision: {prec:.3f} (Trades: {sum(preds)})")
        fold += 1
        
    print(f"Average CV Precision: {np.mean(precisions):.3f}")
    
    print("\n--- SHAP Feature Importance ---")
    # Fit XGB on all data to extract SHAP values
    xgb_clf.fit(X, y)
    explainer = shap.TreeExplainer(xgb_clf)
    shap_values = explainer.shap_values(X)
    
    # Calculate mean absolute SHAP values for each feature
    shap_sum = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame([X.columns.tolist(), shap_sum.tolist()]).T
    importance_df.columns = ['Feature', 'SHAP_Importance']
    importance_df = importance_df.sort_values('SHAP_Importance', ascending=False)
    
    print("Top 5 Most Important Features:")
    print(importance_df.head(5))
    print("\nBottom 5 Least Important Features (Potential Noise):")
    print(importance_df.tail(5))
    
    print("\nTraining Final Ensemble on ALL Data...")
    ensemble.fit(X, y)
    
    joblib.dump(ensemble, "ensemble_meta.pkl")
    print("Successfully saved 'ensemble_meta.pkl'!")

if __name__ == "__main__":
    train_ensemble()
