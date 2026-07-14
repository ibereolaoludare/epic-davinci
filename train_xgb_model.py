import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from feature_engineering_v2 import FEATURE_COLS

def main():
    print("Loading data...")
    try:
        df = pd.read_csv('USDJPY_features_strict.csv')
    except FileNotFoundError:
        print("USDJPY_features_strict.csv not found.")
        return

    # Create target: 1 if Close in HORIZON candles > current Close, else 0
    HORIZON = 6 # 6 candles on H4 = 24 hours
    df['Future_Close'] = df['Close'].shift(-HORIZON)

    # Drop the last HORIZON rows which will have NaN
    df = df.dropna(subset=['Future_Close']).reset_index(drop=True)

    df['Target'] = (df['Future_Close'] > df['Close']).astype(int)

    # Use canonical features
    feature_cols = [c for c in FEATURE_COLS if c in df.columns]

    # Split data (40/30/30) due to variable Forex data length
    total_len = len(df)
    TRAIN_END = int(total_len * 0.4)
    VAL_END = int(total_len * 0.7)

    df_train = df.iloc[:TRAIN_END]
    df_val = df.iloc[TRAIN_END:VAL_END]
    df_test = df.iloc[VAL_END:]

    print(f"Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

    X_train = df_train[feature_cols]
    y_train = df_train['Target']

    X_val = df_val[feature_cols]
    y_val = df_val['Target']

    print("Training XGBoost model for USDJPY H4...")
    model = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        early_stopping_rounds=50,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100
    )

    joblib.dump(model, 'xgb_model_usdjpy.pkl')
    print("Model saved to xgb_model_usdjpy.pkl")

if __name__ == "__main__":
    main()
