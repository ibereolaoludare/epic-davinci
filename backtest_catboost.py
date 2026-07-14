import pandas as pd
import numpy as np
import joblib
from scipy import stats
from timezone_utils import get_session

def run_backtest(ticker='USDJPY', prob_threshold=0.5):
    try:
        df = pd.read_csv(f'{ticker}_features_v6.csv').dropna()
        model = joblib.load(f'catboost_model_{ticker.lower()}_v6.pkl')
    except FileNotFoundError:
        print(f"Data or model not found for {ticker}.")
        return 0.0
    
    EXCLUDE_COLS = ['Target', 'Time', 'Future_Close', 'Open', 'High', 'Low', 'Close', 'Volume', 'EMA_50', 'EMA_200', 'Hour', 'DayOfWeek']
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    
    total_len = len(df)
    VAL_END = int(total_len * 0.85)
    df_test = df.iloc[VAL_END:].reset_index(drop=True)
    df_test['Time'] = pd.to_datetime(df_test['Time'], utc=True)
    
    X_test = df_test[feature_cols]
    
    classes = model.classes_
    probs = model.predict_proba(X_test)
    
    class_map = {c: i for i, c in enumerate(classes)}
    
    idx_long = class_map.get(1)
    idx_short = class_map.get(-1)
    
    prices = df_test['Close'].values
    highs = df_test['High'].values
    lows = df_test['Low'].values
    times = df_test['Time'].dt.to_pydatetime()
    atrs = df_test['ATR'].values if 'ATR' in df_test.columns else np.full(len(prices), 0.5)
    
    HORIZON = 12  # 48 hours on H4
    TARGET_ATR_MULT = 1.5
    
    from datetime import timezone
    known_intervention_dates = [
        pd.Timestamp('2022-09-22 08:00:00', tz=timezone.utc),
        pd.Timestamp('2022-10-21 14:00:00', tz=timezone.utc),
        pd.Timestamp('2024-04-29 04:00:00', tz=timezone.utc),
        pd.Timestamp('2024-05-01 20:00:00', tz=timezone.utc),
        pd.Timestamp('2026-04-30 04:00:00', tz=timezone.utc)
    ]
    
    def is_event_window(t):
        for ev in known_intervention_dates:
            if abs((pd.Timestamp(t) - ev).total_seconds()) <= 7 * 86400:
                return True
        return False

    results = {
        'all': {'balance': 0.0, 'trades': 0, 'wins': 0, 'log': []},
        'event': {'balance': 0.0, 'trades': 0, 'wins': 0, 'log': []},
        'quiet': {'balance': 0.0, 'trades': 0, 'wins': 0, 'log': []}
    }
    
    # ONE-TRADE-AT-A-TIME: skip_until tracks when we're free to enter again
    skip_until = 0

    for i in range(len(df_test) - HORIZON):
        # Enforce one position at a time
        if i < skip_until:
            continue
            
        prob_l = probs[i, idx_long] if idx_long is not None else 0
        prob_s = probs[i, idx_short] if idx_short is not None else 0
        
        if prob_l >= prob_threshold and prob_l > prob_s:
            trade_dir = 1
        elif prob_s >= prob_threshold and prob_s > prob_l:
            trade_dir = -1
        else:
            continue
            
        entry_price = prices[i]
        entry_time = times[i]
        atr = atrs[i]
        
        tp_price = entry_price + (atr * TARGET_ATR_MULT) if trade_dir == 1 else entry_price - (atr * TARGET_ATR_MULT)
        sl_price = entry_price - (atr * TARGET_ATR_MULT) if trade_dir == 1 else entry_price + (atr * TARGET_ATR_MULT)
        
        profit = 0
        days_held = 0
        hit = False
        exit_candle = i + HORIZON  # default: hold to horizon
        
        for j in range(1, HORIZON + 1):
            idx = i + j
            days_held = (times[idx] - entry_time).total_seconds() / 86400
            
            if trade_dir == 1:
                if highs[idx] >= tp_price:
                    profit = tp_price - entry_price
                    exit_candle = idx
                    hit = True; break
                elif lows[idx] <= sl_price:
                    profit = sl_price - entry_price
                    exit_candle = idx
                    hit = True; break
            else:
                if lows[idx] <= tp_price:
                    profit = entry_price - tp_price
                    exit_candle = idx
                    hit = True; break
                elif highs[idx] >= sl_price:
                    profit = entry_price - sl_price
                    exit_candle = idx
                    hit = True; break
                    
        if not hit:
            exit_price = prices[i + HORIZON]
            profit = (exit_price - entry_price) if trade_dir == 1 else (entry_price - exit_price)
            days_held = (times[i + HORIZON] - entry_time).total_seconds() / 86400
        
        # Block new entries until after this trade exits
        skip_until = exit_candle + 1
            
        session = get_session(entry_time)
        spread_pips = 0.5 if session in ['London', 'NY', 'London/NY Overlap'] else 1.5
        total_cost = (spread_pips * 0.01) + (days_held * 0.2 * 0.01)
        profit -= total_cost
        
        bucket = 'event' if is_event_window(entry_time) else 'quiet'
        for b in ['all', bucket]:
            results[b]['balance'] += profit
            results[b]['trades'] += 1
            if profit > 0: results[b]['wins'] += 1
            results[b]['log'].append(profit)
                
    def print_bucket(name, r):
        win_rate = (r['wins'] / r['trades'] * 100) if r['trades'] > 0 else 0
        p_value = 1 - stats.binom.cdf(r['wins'] - 1, r['trades'], 0.5) if r['trades'] > 0 else 1.0
        print(f"[{name.upper()}] Trades: {r['trades']} | WR: {win_rate:.1f}% | PnL: {r['balance']:.3f} | p-value: {p_value:.4f}")
        
    print("=" * 60)
    print(f"PROB_THRESHOLD: {prob_threshold}")
    print_bucket('all', results['all'])
    
    return results['all']['balance']

if __name__ == '__main__':
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else 'USDJPY'
    print(f"Running CatBoost V6 backtest on {ticker} H4 (one-trade-at-a-time)...")
    for t in [0.30, 0.35, 0.40, 0.45, 0.50]:
        run_backtest(ticker, t)
        print()
