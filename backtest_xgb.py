import pandas as pd
import numpy as np
import joblib
from feature_engineering_v2 import FEATURE_COLS
from scipy import stats
from timezone_utils import get_session

def run_backtest(threshold=0.55):
    try:
        df = pd.read_csv('USDJPY_features_strict.csv')
        model = joblib.load('xgb_model_usdjpy.pkl')
    except FileNotFoundError:
        print("Data or model not found. Run previous scripts first.")
        return
    
    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    
    total_len = len(df)
    TEST_START = int(total_len * 0.7)
    df_test = df.iloc[TEST_START:].reset_index(drop=True)
    df_test['Time'] = pd.to_datetime(df_test['Time'], utc=True)
    
    X_test = df_test[feature_cols]
    probs = model.predict_proba(X_test)[:, 1]
    
    prices = df_test['Close'].values
    times = df_test['Time'].dt.to_pydatetime()
    
    HORIZON = 6 # 24 hours on H4
    
    in_trade = False
    trade_dir = 0
    entry_price = 0.0
    entry_idx = 0
    entry_time = None
    
    # Trackers for Event Window vs Quiet Period
    # Event Window: Within 7 days of a known intervention
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
            if abs((pd.Timestamp(t) - ev).total_seconds()) <= 7 * 86400: # 7 days
                return True
        return False

    results = {
        'all': {'balance': 0.0, 'trades': 0, 'wins': 0, 'log': []},
        'event': {'balance': 0.0, 'trades': 0, 'wins': 0, 'log': []},
        'quiet': {'balance': 0.0, 'trades': 0, 'wins': 0, 'log': []}
    }
    
    for i in range(len(df_test) - HORIZON):
        if in_trade:
            if i - entry_idx == HORIZON:
                close_price = prices[i]
                close_time = times[i]
                
                # Forex Costs calculation
                session = get_session(close_time)
                # Pip spread (e.g., 0.5 pips for London/NY, 1.5 pips for Tokyo/Offhours)
                spread_pips = 0.5 if session in ['London', 'NY', 'London/NY Overlap'] else 1.5
                spread_cost = spread_pips * 0.01 # USDJPY pip is 0.01
                
                # Swap costs (simulated) for holding across 21:00 UTC
                # Roughly 0.2 pips per day
                days_held = (close_time - entry_time).total_seconds() / 86400
                swap_cost = days_held * 0.2 * 0.01
                
                total_cost = spread_cost + swap_cost
                
                if trade_dir == 1:
                    profit = (close_price - entry_price) - total_cost
                else:
                    profit = (entry_price - close_price) - total_cost
                
                bucket = 'event' if is_event_window(entry_time) else 'quiet'
                
                for b in ['all', bucket]:
                    results[b]['balance'] += profit
                    results[b]['trades'] += 1
                    if profit > 0: results[b]['wins'] += 1
                    results[b]['log'].append(profit)
                
                in_trade = False
            else:
                continue
                
        if not in_trade:
            prob = probs[i]
            if prob >= threshold:
                in_trade = True; trade_dir = 1
                entry_price = prices[i]; entry_idx = i; entry_time = times[i]
            elif prob <= (1 - threshold):
                in_trade = True; trade_dir = -1
                entry_price = prices[i]; entry_idx = i; entry_time = times[i]
                
    def print_bucket(name, r):
        win_rate = (r['wins'] / r['trades'] * 100) if r['trades'] > 0 else 0
        p_value = 1 - stats.binom.cdf(r['wins'] - 1, r['trades'], 0.5) if r['trades'] > 0 else 1.0
        print(f"[{name.upper()}] Trades: {r['trades']} | WR: {win_rate:.1f}% | PnL: {r['balance']:.3f} | p-value: {p_value:.4f}")
        
    print("=" * 60)
    print(f"THRESHOLD: {threshold}")
    print_bucket('all', results['all'])
    print_bucket('event', results['event'])
    print_bucket('quiet', results['quiet'])

if __name__ == '__main__':
    print("Running backtest for multiple thresholds on USDJPY H4...")
    for t in [0.5, 0.55, 0.60, 0.65]:
        run_backtest(t)
        print()
