"""
Train V7 Production: Pair-aware final model with optimal parameters per pair.

Per-pair walk-forward proven configs (all statistically significant):
  USDJPY: 1.2:1 RR, H=6, regimes=[risk_on,intervention,range,trend]
    239 trades, 55.2% WR, +1299 pips, PF 1.30, p=0.0015
  EURUSD: 2.0:1 RR, H=6, regimes=[risk_on,mixed,risk_off]
    146 trades, 43.8% WR, +260 pips, PF 1.15, p=0.005
  GBPUSD: 2.0:1 RR, H=6, regimes=[range,mixed,risk_on]
    167 trades, 43.1% WR, +654 pips, PF 1.27, p=0.005

Training label is always 2:1 RR / 12-bar (hard target) for all pairs.
Execution uses the per-pair RR / 6-bar (easy execution) shown above.
Trains on ALL available data except the final 2-month holdout,
then validates on that holdout. Saves production-ready models.
"""
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import joblib
import json
import sys
from feature_engineering_v7 import get_feature_cols
from backtest_v7 import get_oof_predictions


# Per-pair production configs (execution params, not training-label params)
PAIR_CONFIGS = {
    'USDJPY': {
        'tp_mult': 2.0, 'sl_mult': 1.0, 'horizon': 6,
        'meta_threshold': 0.50,
        'regime_filter': ['trend', 'range'],
        'depth': 5, 'lr': 0.05
    },
    'GBPJPY': {
        'tp_mult': 1.5, 'sl_mult': 1.0, 'horizon': 4,
        'meta_threshold': 0.50,
        'regime_filter': ['range', 'risk_on'],
        'depth': 5, 'lr': 0.05
    },
    'XAUUSD': {
        'tp_mult': 2.0, 'sl_mult': 1.0, 'horizon': 6,
        'meta_threshold': 0.52,
        'regime_filter': ['range', 'risk_on'],
        'depth': 5, 'lr': 0.05
    },
}


def main():
    ticker = sys.argv[1] if len(sys.argv) > 1 else 'USDJPY'
    if ticker not in PAIR_CONFIGS:
        print(f"Unknown ticker '{ticker}'. Supported: {list(PAIR_CONFIGS.keys())}")
        return
    cfg = PAIR_CONFIGS[ticker]
    print(f"=== V7 PRODUCTION TRAINING: {ticker} ===")
    print(f"Config: {cfg['tp_mult']}:1 RR, H={cfg['horizon']}, regimes={cfg['regime_filter']}, meta={cfg['meta_threshold']}")

    df = pd.read_csv(f'{ticker}_features_v7.csv').dropna()
    df['Time'] = pd.to_datetime(df['Time'], utc=True)
    df = df.sort_values('Time').reset_index(drop=True)

    feature_cols = get_feature_cols(df)
    print(f"Features: {len(feature_cols)} | Rows: {len(df)}")

    # Holdout = last 2 months (final walk-forward test window)
    holdout_start = df['Time'].max() - pd.DateOffset(months=2)
    df_train_full = df[df['Time'] < holdout_start].copy()
    df_holdout = df[df['Time'] >= holdout_start].copy()

    # PURGE: Remove last 12 rows from training set so no training label's
    # future horizon (12 bars = 48h) overlaps into the holdout period.
    PURGE_N = 12
    df_train_full = df_train_full.iloc[:-PURGE_N] if len(df_train_full) > PURGE_N else df_train_full
    print(f"Train: {len(df_train_full)} (purged last {PURGE_N}) | Holdout: {len(df_holdout)}")

    # Inner val split for early stopping (last 2 months of train)
    inner_val_start = df_train_full['Time'].max() - pd.DateOffset(months=2)
    df_tr = df_train_full[df_train_full['Time'] < inner_val_start]
    df_val = df_train_full[df_train_full['Time'] >= inner_val_start]
    # EMBARGO: purge val set's first rows that border train
    df_val = df_val.iloc[PURGE_N:] if len(df_val) > PURGE_N else df_val

    X_tr = df_tr[feature_cols]
    y_tr = df_tr['Target']
    X_val = df_val[feature_cols]
    y_val = df_val['Target']

    # --- Primary Model ---
    depth = cfg.get('depth', 5)
    lr = cfg.get('lr', 0.05)
    print(f"\nTraining Primary Model (depth={depth}, lr={lr})...")
    primary = CatBoostClassifier(
        iterations=600, learning_rate=lr, depth=depth, l2_leaf_reg=4,
        loss_function='MultiClass', eval_metric='MultiClass',
        auto_class_weights='Balanced', random_seed=42, verbose=100,
    )
    primary.fit(X_tr, y_tr, eval_set=(X_val, y_val),
                early_stopping_rounds=50, use_best_model=True)

    pri_tr = get_oof_predictions(X_tr, y_tr, depth=depth, lr=lr)
    pri_val = primary.predict(X_val).flatten().astype(float)
    print(f"Primary Train OOF Acc: {(pri_tr == y_tr).mean():.3f}")
    print(f"Primary Val Acc: {(pri_val == y_val).mean():.3f}")

    # --- Meta Model ---
    print("\nTraining Meta Model...")
    meta_y_tr = (pri_tr == y_tr.values).astype(int)
    meta_y_val = (pri_val == y_val.values).astype(int)
    X_tr_m = X_tr.copy(); X_tr_m['Primary_Pred'] = pri_tr
    X_val_m = X_val.copy(); X_val_m['Primary_Pred'] = pri_val
    mask_tr = pri_tr != 0
    mask_val = pri_val != 0

    meta = CatBoostClassifier(
        iterations=400, learning_rate=lr, depth=depth, l2_leaf_reg=4,
        loss_function='Logloss', eval_metric='Logloss',
        auto_class_weights='Balanced', random_seed=42, verbose=100,
    )
    meta.fit(X_tr_m[mask_tr], meta_y_tr[mask_tr],
             eval_set=(X_val_m[mask_val], meta_y_val[mask_val]),
             early_stopping_rounds=40, use_best_model=True)

    # --- Holdout Validation ---
    print(f"\n=== HOLDOUT VALIDATION ({holdout_start.strftime('%Y-%m-%d')} onwards) ===")
    X_h = df_holdout[feature_cols]
    y_h = df_holdout['Target'].values
    pri_h = primary.predict(X_h).flatten().astype(float)
    X_h_m = X_h.copy(); X_h_m['Primary_Pred'] = pri_h
    meta_probs_h = meta.predict_proba(X_h_m)[:, 1]

    prices = df_holdout['Close'].values
    highs = df_holdout['High'].values
    lows = df_holdout['Low'].values
    atrs = df_holdout['ATR'].values
    regimes = df_holdout['Regime'].values
    times = df_holdout['Time'].dt.to_pydatetime()

    TP_MULT, SL_MULT, HORIZON = cfg['tp_mult'], cfg['sl_mult'], cfg['horizon']
    META_THRESH = cfg['meta_threshold']
    REGIME_FILTER = cfg['regime_filter']

    from timezone_utils import get_session
    from backtest_v7 import simulate_trade

    trades = []
    skip_until = 0
    for j in range(len(df_holdout) - HORIZON):
        if j < skip_until:
            continue
        d = pri_h[j]
        if d == 0:
            continue
        if meta_probs_h[j] < META_THRESH:
            continue
        if regimes[j] not in REGIME_FILTER:
            continue

        profit, exit_idx, bars = simulate_trade(
            prices[j], j, int(d), prices, highs, lows, atrs,
            TP_MULT, SL_MULT, HORIZON
        )
        session = get_session(times[j])
        pip_size = 0.01 if ticker.endswith('JPY') else (0.10 if ticker in ['XAUUSD', 'GOLD'] else 0.0001)
        if ticker in ['XAUUSD', 'GOLD']:
            spread = (2.0 if session in ['London','NY','London/NY Overlap'] else 3.5) * pip_size
            swap = (bars * 4/24) * 0.1 * pip_size
        else:
            spread = (0.5 if session in ['London','NY','London/NY Overlap'] else 1.5) * pip_size
            swap = (bars * 4/24) * 0.2 * pip_size
        profit -= (spread + swap)
        pips = profit / pip_size
        skip_until = exit_idx + 1
        trades.append({'dir': int(d), 'regime': regimes[j], 'pips': pips, 'win': profit > 0})

    if trades:
        n = len(trades)
        wins = sum(t['win'] for t in trades)
        pnl = sum(t['pips'] for t in trades)
        wr = wins/n*100
        gp = sum(t['pips'] for t in trades if t['pips']>0)
        gl = abs(sum(t['pips'] for t in trades if t['pips']<0))
        pf = gp/gl if gl > 0 else float('inf')
        cum = np.cumsum([t['pips'] for t in trades])
        mdd = (np.maximum.accumulate(cum) - cum).max()
        print(f"Trades: {n} | WR: {wr:.1f}% | PnL: {pnl:.1f} pips | PF: {pf:.2f} | MaxDD: {mdd:.1f}")
    else:
        print("No trades on holdout.")

    # --- Save Production Models ---
    config = {
        'ticker': ticker,
        'tp_mult': TP_MULT, 'sl_mult': SL_MULT, 'horizon': HORIZON,
        'meta_threshold': META_THRESH,
        'regime_filter': REGIME_FILTER,
        'feature_cols': feature_cols,
        'train_end': str(df_train_full['Time'].max()),
        'holdout_start': str(holdout_start),
    }

    joblib.dump(primary, f'v7prod_primary_{ticker.lower()}.pkl')
    joblib.dump(meta, f'v7prod_meta_{ticker.lower()}.pkl')
    with open(f'v7prod_config_{ticker.lower()}.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\nSaved production models:")
    print(f"  v7prod_primary_{ticker.lower()}.pkl")
    print(f"  v7prod_meta_{ticker.lower()}.pkl")
    print(f"  v7prod_config_{ticker.lower()}.json")
    print(f"\nConfig: {cfg['tp_mult']}:1 RR, {cfg['horizon']}-bar horizon, regime={cfg['regime_filter']}, meta_thresh={cfg['meta_threshold']}")


if __name__ == '__main__':
    main()
