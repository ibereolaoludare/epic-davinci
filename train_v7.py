"""
Train V7: Meta-Labeling Architecture for USDJPY.
  Primary model:  Predict direction (1=long, -1=short) using CatBoost multiclass.
  Secondary model: Predict whether primary's call will be correct (1=yes, 0=no).
  At inference: only take trades where secondary model says YES (meta-label=1).

Walk-forward: rolling 6-month train / 2-month validation windows.
Saves both models plus the feature column list.
"""
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
import joblib
import json
import sys
from feature_engineering_v7 import get_feature_cols


def build_walk_forward_splits(df, train_months=6, test_months=2):
    """Generate (train_start, train_end, test_start, test_end) rolling windows."""
    df = df.sort_values('Time').reset_index(drop=True)
    df['Time'] = pd.to_datetime(df['Time'], utc=True)

    start = df['Time'].min()
    end = df['Time'].max()

    splits = []
    train_delta = pd.DateOffset(months=train_months)
    test_delta = pd.DateOffset(months=test_months)

    cursor = start
    while cursor + train_delta + test_delta <= end:
        tr_start = cursor
        tr_end = cursor + train_delta
        te_start = tr_end
        te_end = te_start + test_delta

        tr_mask = (df['Time'] >= tr_start) & (df['Time'] < tr_end)
        te_mask = (df['Time'] >= te_start) & (df['Time'] < te_end)

        if tr_mask.sum() > 200 and te_mask.sum() > 20:
            splits.append((tr_start, tr_end, te_start, te_end))

        cursor = te_end  # non-overlapping rolling windows

    return splits


def train_primary_model(X_train, y_train, X_val, y_val, params=None):
    """Train direction prediction model (multiclass: 1, -1, 0)."""
    if params is None:
        params = {
            'iterations': 500,
            'learning_rate': 0.05,
            'depth': 5,
            'l2_leaf_reg': 3,
            'loss_function': 'MultiClass',
            'eval_metric': 'MultiClass',
            'auto_class_weights': 'Balanced',
            'random_seed': 42,
            'verbose': 0,
        }

    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val),
              early_stopping_rounds=50, use_best_model=True)
    return model


def train_meta_model(X_train, primary_preds, actual_outcomes, X_val, primary_val, actual_val):
    """Train secondary model to predict if primary model's call is correct.
    Label: 1 if primary prediction matches actual outcome, 0 otherwise.
    """
    # Build meta-labels
    meta_y_train = (primary_preds == actual_outcomes).astype(int)
    meta_y_val = (primary_val == actual_val).astype(int)

    # Add primary prediction as a feature to the secondary model
    X_train_meta = X_train.copy()
    X_train_meta['Primary_Pred'] = primary_preds
    X_val_meta = X_val.copy()
    X_val_meta['Primary_Pred'] = primary_val

    # Filter to only rows where primary made a prediction (not 0)
    mask_train = primary_preds != 0
    mask_val = primary_val != 0

    if mask_train.sum() < 50 or mask_val.sum() < 10:
        return None

    model = CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=4,
        l2_leaf_reg=5,
        loss_function='Logloss',
        eval_metric='Logloss',
        auto_class_weights='Balanced',
        random_seed=42,
        verbose=0,
    )
    model.fit(X_train_meta[mask_train], meta_y_train[mask_train],
              eval_set=(X_val_meta[mask_val], meta_y_val[mask_val]),
              early_stopping_rounds=30, use_best_model=True)
    return model


def main():
    ticker = sys.argv[1] if len(sys.argv) > 1 else 'USDJPY'
    print(f"Loading V7 data for {ticker}...")
    try:
        df = pd.read_csv(f'{ticker}_features_v7.csv').dropna()
    except FileNotFoundError:
        print(f"{ticker}_features_v7.csv not found. Run feature_engineering_v7.py first.")
        return

    df['Time'] = pd.to_datetime(df['Time'], utc=True)
    df = df.sort_values('Time').reset_index(drop=True)

    feature_cols = get_feature_cols(df)
    print(f"Features: {len(feature_cols)}")
    print(f"Total rows: {len(df)}")

    # Walk-forward splits
    splits = build_walk_forward_splits(df, train_months=6, test_months=2)
    print(f"Walk-forward splits: {len(splits)}")

    if len(splits) < 2:
        print("Not enough data for walk-forward. Using single 70/15/15 split.")
        total_len = len(df)
        tr_end = int(total_len * 0.70)
        val_end = int(total_len * 0.85)
        splits = [(0, tr_end, tr_end, val_end, val_end, total_len)]
        index_based = True
    else:
        index_based = False

    # Train final models on the LARGEST available training set (all but last test window)
    if not index_based:
        last_test_start = splits[-1][2]
        train_mask = df['Time'] < last_test_start
        val_mask = (df['Time'] >= splits[-1][0]) & (df['Time'] < splits[-1][2])
        # Use last split's train period as val for final model
        final_train = df[train_mask]
        final_val = df[(df['Time'] >= splits[-2][0]) & (df['Time'] < splits[-2][2])] if len(splits) > 1 else final_train.tail(200)
    else:
        final_train = df.iloc[:splits[0][1]]
        final_val = df.iloc[splits[0][1]:splits[0][2]]

    print(f"\n=== Training FINAL Primary Model ===")
    print(f"Train: {len(final_train)}, Val: {len(final_val)}")

    X_tr = final_train[feature_cols]
    y_tr = final_train['Target']
    X_val = final_val[feature_cols]
    y_val = final_val['Target']

    primary_model = train_primary_model(X_tr, y_tr, X_val, y_val)

    # Get primary predictions on train for meta-labeling
    primary_preds_train = primary_model.predict(X_tr).flatten().astype(float)
    primary_preds_val = primary_model.predict(X_val).flatten().astype(float)

    train_acc = (primary_preds_train == y_tr.values).mean()
    val_acc = (primary_preds_val == y_val.values).mean()
    print(f"Primary Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}")

    print(f"\n=== Training FINAL Meta Model ===")
    meta_model = train_meta_model(
        X_tr, primary_preds_train, y_tr.values,
        X_val, primary_preds_val, y_val.values
    )

    if meta_model is not None:
        meta_preds_val = meta_model.predict(
            X_val.assign(Primary_Pred=primary_preds_val)
        ).flatten()
        meta_mask = meta_preds_val == 1
        if meta_mask.sum() > 0:
            filtered_acc = (primary_preds_val[meta_mask] == y_val.values[meta_mask]).mean()
            print(f"Meta-filtered Val Acc: {filtered_acc:.3f} (on {meta_mask.sum()} filtered trades)")
            print(f"Meta filter rate: {meta_mask.mean():.2%} of trades passed")

    # Save models
    primary_path = f'v7_primary_{ticker.lower()}.pkl'
    meta_path = f'v7_meta_{ticker.lower()}.pkl'
    joblib.dump(primary_model, primary_path)
    if meta_model is not None:
        joblib.dump(meta_model, meta_path)
    joblib.dump(feature_cols, f'v7_features_{ticker.lower()}.pkl')

    print(f"\nSaved: {primary_path}, {meta_path}")
    return primary_model, meta_model, feature_cols


if __name__ == "__main__":
    main()
