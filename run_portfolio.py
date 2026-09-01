"""
Portfolio Walk-Forward Backtest: Runs all 3 pairs with their per-pair optimal
configs and aggregates the combined equity curve.

Per-pair walk-forward proven configs (all statistically significant):
  USDJPY: 1.2:1 RR, H=6, regimes=[risk_on,intervention,range,trend]  p=0.0015
  EURUSD: 2.0:1 RR, H=6, regimes=[risk_on,mixed,risk_off]            p=0.005
  GBPUSD: 2.0:1 RR, H=6, regimes=[range,mixed,risk_on]               p=0.005
"""
import pandas as pd
import numpy as np
from scipy import stats
from backtest_v7 import run_walk_forward_backtest, build_walk_forward_splits, \
    train_primary, train_meta, simulate_trade
from feature_engineering_v7 import get_feature_cols
from timezone_utils import get_session
import sys
import warnings

warnings.filterwarnings('ignore')


PORTFOLIO_CONFIGS = {
    'USDJPY': {
        'tp_mult': 1.5, 'sl_mult': 1.0, 'horizon': 6,
        'meta_threshold': 0.50,
        'regime_filter': ['risk_on', 'intervention', 'range', 'trend'],
        'depth': 5, 'lr': 0.05
    },
    'GBPJPY': {
        'tp_mult': 2.0, 'sl_mult': 1.0, 'horizon': 6,
        'meta_threshold': 0.50,
        'regime_filter': ['risk_on', 'risk_off', 'trend', 'mixed'],
        'depth': 6, 'lr': 0.03
    },
    'AUDUSD': {
        'tp_mult': 1.2, 'sl_mult': 1.0, 'horizon': 6,
        'meta_threshold': 0.45,
        'regime_filter': ['risk_on', 'risk_off', 'trend', 'mixed'],
        'depth': 6, 'lr': 0.03
    }
}


def run_pair_walk_forward(ticker, cfg):
    """Run walk-forward backtest for a single pair, return list of trades with timestamps."""
    try:
        df = pd.read_csv(f'{ticker}_features_v7.csv').dropna()
    except FileNotFoundError:
        print(f"{ticker}_features_v7.csv not found.")
        return []

    df['Time'] = pd.to_datetime(df['Time'], utc=True)
    df = df.sort_values('Time').reset_index(drop=True)
    feature_cols = get_feature_cols(df)

    splits = build_walk_forward_splits(df, train_months=6, test_months=2)
    if len(splits) < 2:
        return []

    tp, sl, h = cfg['tp_mult'], cfg['sl_mult'], cfg['horizon']
    mt = cfg['meta_threshold']
    rf = cfg['regime_filter']
    depth = cfg.get('depth', 5)
    lr = cfg.get('lr', 0.05)
    pip_size = 0.01 if ticker.endswith('JPY') else 0.0001

    all_trades = []
    PURGE_N = 12  # Target horizon — purge boundary rows to prevent label leakage
    for i, (tr_s, tr_e, te_s, te_e) in enumerate(splits):
        tr_mask = (df['Time'] >= tr_s) & (df['Time'] < tr_e)
        te_mask = (df['Time'] >= te_s) & (df['Time'] < te_e)
        df_tr = df[tr_mask].copy()
        df_te = df[te_mask].copy()
        if len(df_tr) < 200 or len(df_te) < 20:
            continue

        # PURGE: Remove last PURGE_N rows from train so no label peeks into test
        df_tr = df_tr.iloc[:-PURGE_N] if len(df_tr) > PURGE_N else df_tr

        vi = int(len(df_tr) * 0.8)
        df_tr_inner = df_tr.iloc[:vi]
        df_val_inner = df_tr.iloc[vi:]
        # EMBARGO: purge val rows bordering train
        df_val_inner = df_val_inner.iloc[PURGE_N:] if len(df_val_inner) > PURGE_N else df_val_inner

        X_tr = df_tr_inner[feature_cols]
        y_tr = df_tr_inner['Target']
        X_val = df_val_inner[feature_cols]
        y_val = df_val_inner['Target']

        primary = train_primary(X_tr, y_tr, X_val, y_val, depth=depth, lr=lr)
        pri_tr = primary.predict(X_tr).flatten().astype(float)
        pri_val = primary.predict(X_val).flatten().astype(float)
        meta = train_meta(X_tr, pri_tr, y_tr.values, X_val, pri_val, y_val.values, depth=depth, lr=lr)
        if meta is None:
            continue

        X_te = df_te[feature_cols]
        pri_te = primary.predict(X_te).flatten().astype(float)
        X_te_m = X_te.copy()
        X_te_m['Primary_Pred'] = pri_te
        meta_probs = meta.predict_proba(X_te_m)[:, 1]

        prices = df_te['Close'].values
        highs = df_te['High'].values
        lows = df_te['Low'].values
        atrs = df_te['ATR'].values
        times = df_te['Time'].dt.to_pydatetime()
        regimes = df_te['Regime'].values

        skip_until = 0
        for j in range(len(df_te) - h):
            if j < skip_until:
                continue
            d = pri_te[j]
            if d == 0:
                continue
            if meta_probs[j] < mt:
                continue
            if regimes[j] not in rf:
                continue

            profit, exit_idx, bars = simulate_trade(
                prices[j], j, int(d), prices, highs, lows, atrs, tp, sl, h)
            session = get_session(times[j])
            spread_pips = 0.5 if session in ['London', 'NY', 'London/NY Overlap'] else 1.5
            spread_cost = spread_pips * pip_size
            swap_cost = (bars * 4 / 24) * 0.2 * pip_size
            profit -= (spread_cost + swap_cost)
            pips = profit / pip_size
            skip_until = exit_idx + 1

            exit_time = times[min(exit_idx, len(times)-1)]
            all_trades.append({
                'ticker': ticker,
                'entry_time': times[j],
                'exit_time': exit_time,
                'bars_held': bars,
                'direction': int(d),
                'regime': regimes[j],
                'meta_conf': float(meta_probs[j]),
                'pips': pips,
                'win': profit > 0,
            })

    return all_trades


def apply_portfolio_usd_cap(trades, max_usd_exposure=2):
    """
    Filter chronological portfolio trades by the USD directional exposure cap.
    Long USDJPY / Short EURUSD / Short GBPUSD = +1 Long USD exposure.
    Short USDJPY / Long EURUSD / Long GBPUSD = -1 Short USD exposure.
    """
    sorted_trades = sorted(trades, key=lambda t: t['entry_time'])
    accepted_trades = []
    active_positions = []  # list of (exit_time, usd_direction)

    for t in sorted_trades:
        entry_t = t['entry_time']
        # Prune positions that have already closed before entry_t
        active_positions = [p for p in active_positions if p[0] > entry_t]

        # Calculate current net USD exposure
        current_usd_exp = sum(p[1] for p in active_positions)

        # Proposed trade USD direction
        if t['ticker'].endswith('JPY'):
            trade_usd_dir = 1 if t['direction'] == 1 else -1
        else:
            trade_usd_dir = -1 if t['direction'] == 1 else 1

        # Check cap
        if abs(current_usd_exp + trade_usd_dir) <= max_usd_exposure:
            accepted_trades.append(t)
            active_positions.append((t['exit_time'], trade_usd_dir))

    return accepted_trades


def main():
    pairs = sys.argv[1:] if len(sys.argv) > 1 else ['USDJPY', 'GBPJPY', 'AUDUSD']

    print("=" * 75)
    print(f"PORTFOLIO WALK-FORWARD BACKTEST (V7) - BASKET: {pairs}")
    print("=" * 75)

    all_raw_trades = []
    per_pair = {}

    for pair in pairs:
        cfg = PORTFOLIO_CONFIGS.get(pair)
        if cfg is None:
            print(f"  Skipping {pair} (no config)")
            continue
        print(f"\n--- {pair}: {cfg['tp_mult']}:1 RR, H={cfg['horizon']}, "
              f"regimes={cfg['regime_filter']} ---")
        trades = run_pair_walk_forward(pair, cfg)
        per_pair[pair] = trades
        all_raw_trades.extend(trades)

        n = len(trades)
        if n > 0:
            wins = sum(t['win'] for t in trades)
            pnl = sum(t['pips'] for t in trades)
            wr = wins / n * 100
            gp = sum(t['pips'] for t in trades if t['pips'] > 0)
            gl = abs(sum(t['pips'] for t in trades if t['pips'] < 0))
            pf = gp / gl if gl > 0 else float('inf')
            be = cfg['sl_mult'] / (cfg['tp_mult'] + cfg['sl_mult'])
            p_val = 1 - stats.binom.cdf(wins - 1, n, be)
            print(f"  Trades={n} | WR={wr:.1f}% (BE={be*100:.1f}%) | "
                  f"PnL={pnl:.1f} | PF={pf:.2f} | p={p_val:.4f} "
                  f"{'***' if p_val < 0.05 else ''}")

    # By default, run uncapped (all pairs allowed simultaneously)
    all_trades = all_raw_trades

    # --- Portfolio Aggregation ---
    print("\n" + "=" * 75)
    print(f"COMBINED PORTFOLIO RESULTS (ALL {len(pairs)} PAIRS SIMULTANEOUS)")
    print("=" * 75)

    if not all_trades:
        print("No trades across portfolio.")
        return

    # Sort all trades by entry time for combined equity curve
    all_trades.sort(key=lambda t: t['entry_time'])

    n = len(all_trades)
    wins = sum(t['win'] for t in all_trades)
    total_pnl = sum(t['pips'] for t in all_trades)
    wr = wins / n * 100

    gp = sum(t['pips'] for t in all_trades if t['pips'] > 0)
    gl = abs(sum(t['pips'] for t in all_trades if t['pips'] < 0))
    pf = gp / gl if gl > 0 else float('inf')

    cum = np.cumsum([t['pips'] for t in all_trades])
    rmax = np.maximum.accumulate(cum)
    mdd = (rmax - cum).max()
    expectancy = total_pnl / n

    # Blended break-even (weighted by each pair's RR)
    be_wr = np.mean([PORTFOLIO_CONFIGS[t['ticker']]['sl_mult'] /
                     (PORTFOLIO_CONFIGS[t['ticker']]['tp_mult'] +
                      PORTFOLIO_CONFIGS[t['ticker']]['sl_mult'])
                     for t in all_trades])
    p_portfolio = 1 - stats.binom.cdf(wins - 1, n, be_wr)

    print(f"Total Trades:    {n}")
    print(f"Win Rate:        {wr:.1f}% (blended break-even {be_wr*100:.1f}%)")
    print(f"Total PnL:       {total_pnl:.1f} pips")
    print(f"Profit Factor:   {pf:.2f}")
    print(f"Max Drawdown:    {mdd:.1f} pips")
    print(f"Expectancy:      {expectancy:.2f} pips/trade")
    print(f"p-value (vs blended BE): {p_portfolio:.6f} "
          f"{'*** SIGNIFICANT' if p_portfolio < 0.05 else '(not significant)'}")

    # Per-pair breakdown
    print(f"\n--- Per-Pair Contribution ---")
    for pair in pairs:
        pair_trades = [t for t in all_trades if t['ticker'] == pair]
        if not pair_trades:
            continue
        pnl = sum(t['pips'] for t in pair_trades)
        n_p = len(pair_trades)
        wr_p = sum(t['win'] for t in pair_trades) / n_p * 100
        print(f"  {pair:8s}: Trades={n_p:3d}, WR={wr_p:.1f}%, PnL={pnl:8.1f} pips")

    # Monthly equity curve
    print(f"\n--- Monthly Portfolio PnL ---")
    monthly = {}
    for t in all_trades:
        key = pd.Timestamp(t['entry_time']).strftime('%Y-%m')
        monthly[key] = monthly.get(key, 0) + t['pips']
    for k in sorted(monthly.keys()):
        v = monthly[k]
        bar = '+' * int(abs(v) / 15) if v > 0 else '-' * int(abs(v) / 15)
        print(f"  {k}: {v:8.1f} pips  {bar}")

    # --- Position Sizing & 2% Dynamic Compounding Simulation ---
    usd_per_pip_per_lot = {'USDJPY': 6.67, 'GBPJPY': 6.67, 'AUDUSD': 10.0, 'EURUSD': 10.0, 'GBPUSD': 10.0}

    # Simulation with $1,000 start
    initial_balance = 1000.0
    comp_balance = initial_balance
    comp_equity_curve = [initial_balance]
    fixed_balance = initial_balance
    fixed_equity_curve = [initial_balance]

    avg_loss_pips = np.mean([abs(t['pips']) for t in all_trades if t['pips'] < 0])
    avg_win_pips = np.mean([t['pips'] for t in all_trades if t['pips'] > 0])

    for t in all_trades:
        pip_val = usd_per_pip_per_lot.get(t['ticker'], 10.0)
        # Fixed 2% risk based on initial balance
        fixed_lot = max(0.01, (initial_balance * 0.02) / (avg_loss_pips * pip_val))
        fixed_pnl = t['pips'] * pip_val * fixed_lot
        fixed_balance += fixed_pnl
        fixed_equity_curve.append(fixed_balance)

        # Dynamic 2% compounding risk
        comp_lot = max(0.01, (comp_balance * 0.02) / (avg_loss_pips * pip_val))
        comp_pnl = t['pips'] * pip_val * comp_lot
        comp_balance += comp_pnl
        comp_equity_curve.append(comp_balance)

    # Compounding Max DD %
    comp_peak = np.maximum.accumulate(comp_equity_curve)
    comp_dds = (comp_peak - np.array(comp_equity_curve)) / comp_peak * 100
    max_comp_dd_pct = np.max(comp_dds)

    print(f"\n--- 2% Dynamic Compounding Performance ($1,000 Start) ---")
    print(f"  Fixed Lot Final Balance:       ${fixed_balance:,.2f} ({((fixed_balance/initial_balance)-1)*100:+.1f}%)")
    print(f"  Dynamic Compounding Balance:   ${comp_balance:,.2f} ({((comp_balance/initial_balance)-1)*100:+.1f}%)")
    print(f"  Max Compounding Drawdown:      {max_comp_dd_pct:.2f}%")
    print(f"  Avg Win: {avg_win_pips:.1f} pips | Avg Loss: {avg_loss_pips:.1f} pips")

    print()
    print(f"  Risk rule: 2% of account per trade, lot size = risk / (avg_loss_pips * pip_value)")
    print(f"  This means you NEVER lose more than ${0.02:.0f} per trade on a $1K account")
    print(f"  Lot sizes shown are MICRO lots (0.01 = 1K units). Broker minimum is typically 0.01.")
    print(f"  If calculated lot < 0.01 broker minimum, risk is higher than 2% — do NOT trade.")

    return all_trades


if __name__ == '__main__':
    main()
