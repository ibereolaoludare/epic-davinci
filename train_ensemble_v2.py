"""
Train Ensemble v2 — Honest Pipeline
====================================
- Uses ppo_v2_strict (trained on candles 0–10,000)
- Extracts PPO trades on candles 10,000–17,500 (PPO-unseen data)
- Trains the ensemble with Purged K-Fold CV
- Candles 17,500–end are reserved for final testing
"""

import pandas as pd
import numpy as np
import joblib
import warnings
from sb3_contrib import RecurrentPPO
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score
from sklearn.ensemble import VotingClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from feature_engineering_v2 import FEATURE_COLS

warnings.filterwarnings("ignore")

# ── Boundaries ─────────────────────────────────────
PPO_TRAIN_END = 10000      # PPO saw 0–10,000
ENSEMBLE_TRAIN_END = 17500  # Ensemble trains on 10,000–17,500
# Test: 17,500–end          # Neither model has seen this


def extract_ppo_trades(df, model, feature_cols, SL_POINTS=200, TP_POINTS=200):
    """Run the PPO over a dataframe and record (features, win/loss) for each trade."""
    print(f"  Extracting trades over {len(df)} candles (SL={SL_POINTS}, TP={TP_POINTS})...")
    obs_df = df[feature_cols]
    prices = df['Close'].values
    highs = df['High'].values
    lows = df['Low'].values

    X_meta, y_meta = [], []
    current_position = 0
    entry_price = 0

    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)

    for i in range(96, len(df)):
        obs_window = obs_df.iloc[i-96:i].values
        obs = np.array(obs_window, dtype=np.float32)
        
        action, lstm_states = model.predict(
            obs, 
            state=lstm_states, 
            episode_start=episode_starts,
            deterministic=True
        )
        episode_starts = np.zeros((1,), dtype=bool)
        
        action = int(action)
        close = prices[i]
        latest_features = list(obs[-1])

        # Check SL/TP on existing position
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

        # Enter or flip position
        if current_position == 0:
            if action in (0, 1):
                current_position = 1 if action == 0 else 2
                entry_price = close
                X_meta.append(latest_features)
        else:
            if current_position == 1 and action == 1:
                profit = close - entry_price
                y_meta.append(1 if profit > 0 else 0)
                current_position = 2
                entry_price = close
                X_meta.append(latest_features)
            elif current_position == 2 and action == 0:
                profit = entry_price - close
                y_meta.append(1 if profit > 0 else 0)
                current_position = 1
                entry_price = close
                X_meta.append(latest_features)

    if len(X_meta) > len(y_meta):
        X_meta = X_meta[:len(y_meta)]

    return X_meta, y_meta


def train_ensemble():
    print("Loading leak-free features...")
    df_all = pd.read_csv("features_v2_strict.csv")
    
    print("Loading RecurrentPPO v2 (trained on candles 0–10,000 only)...")
    model = RecurrentPPO.load("ppo_v2_lstm_strict")
    
    # Get feature cols from canonical order
    feature_cols = [c for c in FEATURE_COLS if c in df_all.columns]
    print(f"Using {len(feature_cols)} canonical features: {feature_cols}")

    # ── Extract trades on ENSEMBLE partition (PPO-unseen) ─────────
    df_ensemble = df_all.iloc[PPO_TRAIN_END:ENSEMBLE_TRAIN_END].reset_index(drop=True)
    print(f"\nEnsemble training data: candles {PPO_TRAIN_END}–{ENSEMBLE_TRAIN_END} ({len(df_ensemble)} candles)")

    X_meta, y_meta = extract_ppo_trades(df_ensemble, model, feature_cols, SL_POINTS=200, TP_POINTS=200)
    print(f"  Extracted {len(X_meta)} trades (Win rate: {sum(y_meta)/len(y_meta)*100:.1f}%)")

    feature_names = [f"feat_{j}" for j in range(len(X_meta[0]))]
    X = pd.DataFrame(X_meta, columns=feature_names)
    y = np.array(y_meta)

    # ── Purged Time-Series CV ─────────────────────────────────────
    print("\n--- Purged Time-Series Cross Validation ---")
    tscv = TimeSeriesSplit(n_splits=5)
    PURGE_GAP = 100  # 100 trades (~many candles) between folds

    xgb_clf = xgb.XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        random_state=42, scale_pos_weight=(len(y)-sum(y))/max(sum(y),1), n_jobs=-1
    )
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        random_state=42, class_weight='balanced', n_jobs=-1, verbose=-1
    )
    cat_clf = CatBoostClassifier(
        iterations=100, depth=3, learning_rate=0.05,
        random_seed=42, auto_class_weights='Balanced', verbose=0
    )

    ensemble = VotingClassifier(estimators=[
        ('xgb', xgb_clf), ('lgb', lgb_clf), ('cat', cat_clf)
    ], voting='soft')

    for thresh in [0.50, 0.55, 0.60]:
        fold_precs = []
        fold_counts = []
        fold = 1
        for train_idx, test_idx in tscv.split(X):
            # Purge: drop last PURGE_GAP from train
            if len(train_idx) > PURGE_GAP:
                train_idx = train_idx[:-PURGE_GAP]

            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            ensemble.fit(X_tr, y_tr)
            probs = ensemble.predict_proba(X_te)[:, 1]
            preds = (probs > thresh).astype(int)

            prec = precision_score(y_te, preds, zero_division=0)
            n_trades = int(sum(preds))
            fold_precs.append(prec)
            fold_counts.append(n_trades)
            fold += 1

        avg = np.mean(fold_precs)
        total_trades = sum(fold_counts)
        print(f"  Thresh {thresh:.2f} — Avg Precision: {avg:.3f} | Total Predicted Trades: {total_trades} | Per-fold: {[f'{p:.3f}({n})' for p,n in zip(fold_precs, fold_counts)]}")

    # ── Final fit on all ensemble data ────────────────────────────
    print("\nTraining Final Ensemble on all ensemble-partition data...")
    ensemble.fit(X, y)

    joblib.dump(ensemble, "ensemble_v2.pkl")
    joblib.dump(feature_names, "ensemble_v2_feature_names.pkl")
    print(f"Saved ensemble_v2.pkl (trained on {len(X)} trades)")


if __name__ == "__main__":
    train_ensemble()
