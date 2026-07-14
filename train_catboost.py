import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import joblib
import json

import sys

def main():
    ticker = sys.argv[1] if len(sys.argv) > 1 else 'USDJPY'
    print(f"Loading V6 data for {ticker}...")
    try:
        df = pd.read_csv(f'{ticker}_features_v6.csv').dropna()
    except FileNotFoundError:
        print(f"{ticker}_features_v6.csv not found.")
        return

    EXCLUDE_COLS = ['Target', 'Time', 'Future_Close', 'Open', 'High', 'Low', 'Close', 'Volume', 'EMA_50', 'EMA_200', 'Hour', 'DayOfWeek']
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]

    total_len = len(df)
    TRAIN_END = int(total_len * 0.70)
    VAL_END = int(total_len * 0.85)

    df_train = df.iloc[:TRAIN_END]
    df_val = df.iloc[TRAIN_END:VAL_END]
    df_test = df.iloc[VAL_END:]

    print(f"Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

    X_train = df_train[feature_cols]
    y_train = df_train['Target']
    X_val = df_val[feature_cols]
    y_val = df_val['Target']

    print(f"Training CatBoost Multi-class model for {ticker} H4...")
    
    try:
        with open('best_params.json', 'r') as f:
            params = json.load(f)
    except FileNotFoundError:
        params = {
            'iterations': 1000,
            'learning_rate': 0.1,
            'depth': 4,
            'l2_leaf_reg': 1,
            'loss_function': 'MultiClass',
            'eval_metric': 'MultiClass',
            'auto_class_weights': 'Balanced',
            'random_seed': 42
        }

    # Override iterations to train a bit longer since we have early stopping
    params['iterations'] = 500
    params['verbose'] = 100

    model = CatBoostClassifier(**params)

    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
        use_best_model=True
    )

    model_path = f'catboost_model_{ticker.lower()}_v6.pkl'
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
