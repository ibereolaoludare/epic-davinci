"""
Backtest V7: Walk-forward regime-adaptive backtest with meta-labeling.

For each walk-forward window:
  1. Train primary model on past 6 months
  2. Train meta-model on primary's predictions
  3. Test on next 2 months (completely out-of-sample)
  4. Only trade when meta-model approves (confidence > threshold)
  5. Gate by regime (skip trades in unfavorable regimes)
  6. Use asymmetric 2:1 RR barriers

Reports: WR, PnL, max drawdown, profit factor, p-value, regime breakdown.
"""
import pandas as pd
import numpy as np
import joblib
from scipy import stats
from catboost import CatBoostClassifier
from feature_engineering_v7 import get_feature_cols
from timezone_utils import get_session
import sys
import warnings

warnings.filterwarnings('ignore')


def build_walk_forward_splits(df, train_months=6, test_months=2):
    df = df.sort_values('Time').reset_index(drop=True)
    df['Time'] = pd.to_datetime(df['Time'], utc=True)
    start = df['Time'].min()
    end = df['Time'].max()
    splits = []
    cursor = start
    while cursor + pd.DateOffset(months=train_months + test_months) <= end:
        tr_start = cursor
        tr_end = cursor + pd.DateOffset(months=train_months)
        te_start = tr_end
        te_end = te_start + pd.DateOffset(months=test_months)
        tr_mask = (df['Time'] >= tr_start) & (df['Time'] < tr_end)
        te_mask = (df['Time'] >= te_start) & (df['Time'] < te_end)
        if tr_mask.sum() > 200 and te_mask.sum() > 20:
            splits.append((tr_start, tr_end, te_start, te_end))
        cursor = te_end
    return splits


def train_primary(X_train, y_train, X_val, y_val, depth=5, lr=0.05):
    model = CatBoostClassifier(
        iterations=600, learning_rate=lr, depth=depth, l2_leaf_reg=4,
        loss_function='MultiClass', eval_metric='MultiClass',
        auto_class_weights='Balanced', random_seed=42, verbose=0,
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val),
              early_stopping_rounds=50, use_best_model=True)
    return model


def train_meta(X_train, primary_preds, y_train, X_val, primary_val, y_val, depth=5, lr=0.05):
    meta_y_tr = (primary_preds == y_train).astype(int)
    meta_y_val = (primary_val == y_val).astype(int)
    X_tr_meta = X_train.copy()
    X_tr_meta['Primary_Pred'] = primary_preds
    X_val_meta = X_val.copy()
    X_val_meta['Primary_Pred'] = primary_val
    mask_tr = primary_preds != 0
    mask_val = primary_val != 0
    if mask_tr.sum() < 50 or mask_val.sum() < 10:
        return None
    model = CatBoostClassifier(
        iterations=400, learning_rate=lr, depth=depth, l2_leaf_reg=4,
        loss_function='Logloss', eval_metric='Logloss',
        auto_class_weights='Balanced', random_seed=42, verbose=0,
    )
    model.fit(X_tr_meta[mask_tr], meta_y_tr[mask_tr],
              eval_set=(X_val_meta[mask_val], meta_y_val[mask_val]),
              early_stopping_rounds=40, use_best_model=True)
    return model


def simulate_trade(entry_price, entry_idx, trade_dir, prices, highs, lows,
                   atrs, tp_mult, sl_mult, horizon):
    """Simulate asymmetric triple-barrier trade. Returns (profit_in_price, exit_idx, days_held)."""
    atr = atrs[entry_idx]
    if atr <= 0 or np.isnan(atr):
        return 0, entry_idx + horizon, 0

    if trade_dir == 1:  # Long
        tp_price = entry_price + (atr * tp_mult)
        sl_price = entry_price - (atr * sl_mult)
    else:  # Short
        tp_price = entry_price - (atr * tp_mult)
        sl_price = entry_price + (atr * sl_mult)

    for j in range(1, horizon + 1):
        idx = entry_idx + j
        if idx >= len(prices):
            break
        if trade_dir == 1:
            if highs[idx] >= tp_price:
                return (tp_price - entry_price), idx, j
            elif lows[idx] <= sl_price:
                return (sl_price - entry_price), idx, j
        else:
            if lows[idx] <= tp_price:
                return (entry_price - tp_price), idx, j
            elif highs[idx] >= sl_price:
                return (entry_price - sl_price), idx, j

    # No barrier hit: exit at horizon close
    exit_idx = min(entry_idx + horizon, len(prices) - 1)
    exit_price = prices[exit_idx]
    profit = (exit_price - entry_price) if trade_dir == 1 else (entry_price - exit_price)
    return profit, exit_idx, horizon


def run_walk_forward_backtest(ticker='USDJPY', meta_threshold=0.5,
                               tp_mult=2.0, sl_mult=1.0, horizon=12,
                               regime_filter=None):
    """Full walk-forward backtest with meta-labeling and regime gating."""
    try:
        df = pd.read_csv(f'{ticker}_features_v7.csv').dropna()
    except FileNotFoundError:
        print(f"{ticker}_features_v7.csv not found.")
        return

    df['Time'] = pd.to_datetime(df['Time'], utc=True)
    df = df.sort_values('Time').reset_index(drop=True)
    feature_cols = get_feature_cols(df)

    splits = build_walk_forward_splits(df, train_months=6, test_months=2)
    if len(splits) < 2:
        print("Not enough data for walk-forward.")
        return

    # Aggregated results
    all_trades = []
    regime_stats = {}

    for i, (tr_s, tr_e, te_s, te_e) in enumerate(splits):
        tr_mask = (df['Time'] >= tr_s) & (df['Time'] < tr_e)
        te_mask = (df['Time'] >= te_s) & (df['Time'] < te_e)
        df_tr = df[tr_mask].copy()
        df_te = df[te_mask].copy()

        if len(df_tr) < 200 or len(df_te) < 20:
            continue

        # PURGE: Remove last TARGET_HORIZON rows from training set so that
        # no training label's future horizon overlaps into the test period.
        # This is the López de Prado purged k-fold technique — without it,
        # the last 12 training labels peek 48 hours into the test set.
        purge_n = 12  # TARGET_HORIZON used in feature_engineering_v7.py
        df_tr = df_tr.iloc[:-purge_n] if len(df_tr) > purge_n else df_tr

        # Split train into train/val for early stopping (80/20)
        tr_val_split = int(len(df_tr) * 0.8)
        df_tr_inner = df_tr.iloc[:tr_val_split]
        df_val_inner = df_tr.iloc[tr_val_split:]

        # EMBARGO: Also purge the first few rows of the val set that border
        # the inner train set, to prevent label leakage across that boundary.
        df_val_inner = df_val_inner.iloc[purge_n:] if len(df_val_inner) > purge_n else df_val_inner

        X_tr = df_tr_inner[feature_cols]
        y_tr = df_tr_inner['Target']
        X_val = df_val_inner[feature_cols]
        y_val = df_val_inner['Target']

        # Train primary
        primary = train_primary(X_tr, y_tr, X_val, y_val)

        # Get primary predictions on inner train and val for meta
        pri_preds_tr = primary.predict(X_tr).flatten().astype(float)
        pri_preds_val = primary.predict(X_val).flatten().astype(float)

        # Train meta
        meta = train_meta(X_tr, pri_preds_tr, y_tr.values,
                          X_val, pri_preds_val, y_val.values)

        if meta is None:
            print(f"  Split {i+1}: Meta model failed, skipping")
            continue

        # Predict on TEST set
        X_te = df_te[feature_cols]
        pri_preds_te = primary.predict(X_te).flatten().astype(float)
        X_te_meta = X_te.copy()
        X_te_meta['Primary_Pred'] = pri_preds_te
        meta_probs_te = meta.predict_proba(X_te_meta)[:, 1]  # P(meta=1)

        # Simulate trades
        prices = df_te['Close'].values
        highs = df_te['High'].values
        lows = df_te['Low'].values
        atrs = df_te['ATR'].values
        times = df_te['Time'].dt.to_pydatetime()
        regimes = df_te['Regime'].values

        skip_until = 0
        split_trades = []

        for j in range(len(df_te) - horizon):
            if j < skip_until:
                continue

            direction = pri_preds_te[j]
            if direction == 0:
                continue

            meta_conf = meta_probs_te[j]
            if meta_conf < meta_threshold:
                continue

            # Regime filter
            reg = regimes[j]
            if regime_filter is not None and reg not in regime_filter:
                continue

            # Macro VIX Circuit Breaker: lockout during extreme volatility spikes
            if 'Macro_VIX_Lockout' in df_te.columns and df_te['Macro_VIX_Lockout'].iloc[j] == 1:
                continue

            entry_price = prices[j]
            entry_time = times[j]

            profit, exit_idx, bars_held = simulate_trade(
                entry_price, j, int(direction),
                prices, highs, lows, atrs,
                tp_mult, sl_mult, horizon
            )

            # Costs (pair-aware pip sizes)
            pip_size = 0.01 if ticker.endswith('JPY') else 0.0001
            session = get_session(entry_time)
            spread_pips = 0.5 if session in ['London', 'NY', 'London/NY Overlap'] else 1.5
            spread_cost = spread_pips * pip_size
            days_held = bars_held * 4 / 24  # H4 candles to days
            swap_cost = days_held * 0.2 * pip_size
            total_cost = spread_cost + swap_cost
            profit -= total_cost

            # Pair-aware pip conversion
            profit_pips = profit / pip_size

            skip_until = exit_idx + 1

            trade = {
                'split': i, 'entry_time': entry_time, 'direction': int(direction),
                'regime': reg, 'meta_conf': meta_conf,
                'profit_price': profit, 'profit_pips': profit_pips,
                'bars_held': bars_held, 'win': profit > 0,
            }
            split_trades.append(trade)
            all_trades.append(trade)

        n = len(split_trades)
        if n > 0:
            wr = sum(t['win'] for t in split_trades) / n * 100
            pnl = sum(t['profit_pips'] for t in split_trades)
            print(f"  Split {i+1} ({te_s.strftime('%Y-%m')}-{te_e.strftime('%Y-%m')}): "
                  f"Trades={n}, WR={wr:.1f}%, PnL={pnl:.1f} pips")
        else:
            print(f"  Split {i+1} ({te_s.strftime('%Y-%m')}-{te_e.strftime('%Y-%m')}): 0 trades")

    # Final aggregated results
    print("\n" + "=" * 70)
    print(f"WALK-FORWARD RESULTS: {ticker}")
    print(f"Meta threshold: {meta_threshold} | RR: {tp_mult}:{sl_mult} | "
          f"Regime filter: {regime_filter or 'ALL'}")
    print("=" * 70)

    if not all_trades:
        print("NO TRADES TAKEN. Lower meta_threshold or relax regime filter.")
        return

    total_trades = len(all_trades)
    total_wins = sum(t['win'] for t in all_trades)
    total_pnl = sum(t['profit_pips'] for t in all_trades)
    wr = total_wins / total_trades * 100

    # Statistical significance (binomial test vs BREAK-EVEN WR, not 50%)
    # For 2:1 RR, break-even = 1/(2+1) = 33.3%. For 3:1, break-even = 25%.
    break_even_wr = sl_mult / (tp_mult + sl_mult)
    p_value = 1 - stats.binom.cdf(total_wins - 1, total_trades, break_even_wr)
    p_value_50 = 1 - stats.binom.cdf(total_wins - 1, total_trades, 0.5)

    # Profit factor
    gross_profit = sum(t['profit_pips'] for t in all_trades if t['profit_pips'] > 0)
    gross_loss = abs(sum(t['profit_pips'] for t in all_trades if t['profit_pips'] < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Max drawdown
    cumulative = np.cumsum([t['profit_pips'] for t in all_trades])
    running_max = np.maximum.accumulate(cumulative)
    drawdown = running_max - cumulative
    max_dd = drawdown.max() if len(drawdown) > 0 else 0

    # Expectancy
    expectancy = total_pnl / total_trades

    print(f"Total Trades:    {total_trades}")
    print(f"Win Rate:        {wr:.1f}% (break-even at {break_even_wr*100:.1f}%)")
    print(f"Total PnL:       {total_pnl:.1f} pips")
    print(f"Profit Factor:   {pf:.2f}")
    print(f"Max Drawdown:    {max_dd:.1f} pips")
    print(f"Expectancy:      {expectancy:.2f} pips/trade")
    print(f"p-value (vs break-even {break_even_wr*100:.0f}%): {p_value:.6f} {'*** SIGNIFICANT' if p_value < 0.05 else '(not significant)'}")
    print(f"p-value (vs 50%): {p_value_50:.6f}")

    # Regime breakdown
    print(f"\n--- Regime Breakdown ---")
    for reg in sorted(set(t['regime'] for t in all_trades)):
        reg_trades = [t for t in all_trades if t['regime'] == reg]
        reg_wins = sum(t['win'] for t in reg_trades)
        reg_pnl = sum(t['profit_pips'] for t in reg_trades)
        reg_wr = reg_wins / len(reg_trades) * 100
        print(f"  {reg:15s}: Trades={len(reg_trades):3d}, WR={reg_wr:.1f}%, "
              f"PnL={reg_pnl:8.1f} pips")

    # Monthly PnL
    print(f"\n--- Monthly PnL ---")
    monthly = {}
    for t in all_trades:
        key = pd.Timestamp(t['entry_time']).strftime('%Y-%m')
        monthly[key] = monthly.get(key, 0) + t['profit_pips']
    for k in sorted(monthly.keys()):
        bar = '+' * int(abs(monthly[k]) / 10) if monthly[k] > 0 else '-' * int(abs(monthly[k]) / 10)
        print(f"  {k}: {monthly[k]:8.1f} pips  {bar}")

    return all_trades


if __name__ == '__main__':
    ticker = sys.argv[1] if len(sys.argv) > 1 else 'USDJPY'

    print(f"\n{'='*70}")
    print(f"V7 WALK-FORWARD BACKTEST: {ticker}")
    print(f"{'='*70}")

    # Test different meta thresholds
    for thresh in [0.40, 0.45, 0.50, 0.55, 0.60]:
        print(f"\n--- Meta Threshold: {thresh} ---")
        run_walk_forward_backtest(ticker, meta_threshold=thresh,
                                   tp_mult=2.0, sl_mult=1.0, horizon=12)
