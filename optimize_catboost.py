import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss
import json

def run_grid_search():
    df = pd.read_csv('USDJPY_features_v6.csv').dropna()
    
    total_len = len(df)
    train_end = int(total_len * 0.70)
    val_end = int(total_len * 0.85)
    
    df_train = df.iloc[:train_end]
    df_val = df.iloc[train_end:val_end]
    
    EXCLUDE_COLS = ['Target', 'Time', 'Future_Close', 'Open', 'High', 'Low', 'Close', 'Volume', 'EMA_50', 'EMA_200', 'Hour', 'DayOfWeek']
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    
    X_train = df_train[feature_cols]
    y_train = df_train['Target']
    X_val = df_val[feature_cols]
    y_val = df_val['Target']
    
    learning_rates = [0.01, 0.03, 0.1]
    depths = [3, 4, 5, 6]
    l2_leaf_regs = [1, 3, 5]
    
    best_loss = float('inf')
    best_params = {}
    
    print("Running CatBoost Grid Search...")
    for lr in learning_rates:
        for depth in depths:
            for l2 in l2_leaf_regs:
                params = {
                    'iterations': 100,
                    'depth': depth,
                    'learning_rate': lr,
                    'l2_leaf_reg': l2,
                    'loss_function': 'MultiClass',
                    'eval_metric': 'MultiClass',
                    'auto_class_weights': 'Balanced',
                    'random_seed': 42,
                    'verbose': 0
                }
                
                model = CatBoostClassifier(**params)
                model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=20)
                
                preds = model.predict_proba(X_val)
                loss = log_loss(y_val, preds)
                
                if loss < best_loss:
                    best_loss = loss
                    best_params = params.copy()
                    
    print(f"Best Log Loss: {best_loss:.4f}")
    print("Best params:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
        
    with open('best_params.json', 'w') as f:
        json.dump(best_params, f)

if __name__ == '__main__':
    run_grid_search()
