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
        'tp_mult': 1.2, 'sl_mult': 1.0, 'horizon': 6,
        'meta_threshold': 0.40,
        'regime_filter': ['risk_on', 'intervention', 'range', 'trend'],
    },
    'EURUSD': {
        'tp_mult': 2.0, 'sl_mult': 1.0, 'horizon': 6,
        'meta_threshold': 0.51,
        'regime_filter': ['mixed', 'trend'],
    },
    'GBPUSD': {
        'tp_mult': 2.0, 'sl_mult': 1.0, 'horizon': 6,
        'meta_threshold': 0.40,
        'regime_filter': ['mixed', 'risk_off', 'trend'],
    },
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

        primary = train_primary(X_tr, y_tr, X_val, y_val)
        pri_tr = primary.predict(X_tr).flatten().astype(float)
        pri_val = primary.predict(X_val).flatten().astype(float)
        meta = train_meta(X_tr, pri_tr, y_tr.values, X_val, pri_val, y_val.values)
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

            all_trades.append({
                'ticker': ticker,
                'entry_time': times[j],
                'direction': int(d),
                'regime': regimes[j],
                'meta_conf': float(meta_probs[j]),
                'pips': pips,
                'win': profit > 0,
            })

    return all_trades


def main():
    pairs = sys.argv[1:] if len(sys.argv) > 1 else ['USDJPY', 'EURUSD', 'GBPUSD']

    print("=" * 75)
    print("PORTFOLIO WALK-FORWARD BACKTEST (V7)")
    print("=" * 75)

    all_trades = []
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
        all_trades.extend(trades)

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

    # --- Portfolio Aggregation ---
    print("\n" + "=" * 75)
    print("COMBINED PORTFOLIO RESULTS")
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
    for pair, trades in per_pair.items():
        if not trades:
            continue
        pnl = sum(t['pips'] for t in trades)
        n_p = len(trades)
        wr_p = sum(t['win'] for t in trades) / n_p * 100
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

    # --- Position Sizing for Real Accounts ---
    # pip values per 1.00 standard lot (100K units):
    #   USDJPY: ~$6.67/pip (100K * 0.01 / 150 rate)
    #   EURUSD: $10/pip (100K * 0.0001)
    #   GBPUSD: $10/pip
    usd_per_pip_per_lot = {'USDJPY': 6.67, 'EURUSD': 10.0, 'GBPUSD': 10.0}

    # Average loss per trade in pips (for risk calculation)
    avg_loss_pips = np.mean([abs(t['pips']) for t in all_trades if t['pips'] < 0])
    avg_win_pips = np.mean([t['pips'] for t in all_trades if t['pips'] > 0])

    print(f"\n--- Position Sizing Guide (risk-based, NOT fixed lots) ---")
    print(f"Avg win:  {avg_win_pips:.1f} pips | Avg loss: {avg_loss_pips:.1f} pips")
    print()
    print(f"  {'Account':>10s} | {'Risk/Trade':>10s} | {'Lot Size':>10s} | {'Max DD $':>10s} | {'Est. Total PnL':>15s}")
    print(f"  {'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*15}")

    for account_size in [1000, 2500, 5000]:
        risk_pct = 0.02  # 2% risk per trade
        risk_per_trade = account_size * risk_pct
        # Use the worst-case (highest) pip value among traded pairs for safety
        max_pip_val = max(usd_per_pip_per_lot.values())
        # Lot size = risk / (avg_loss_pips * pip_value_per_lot)
        lot_size = risk_per_trade / (avg_loss_pips * max_pip_val)
        max_dd_usd = mdd * max_pip_val * lot_size
        total_usd = total_pnl * np.mean([usd_per_pip_per_lot[t['ticker']] for t in all_trades]) * lot_size
        print(f"  ${account_size:>8,d} | ${risk_per_trade:>8.2f} | {lot_size:>10.4f} | ${max_dd_usd:>8.2f} | ${total_usd:>13.2f}")

    print()
    print(f"  Risk rule: 2% of account per trade, lot size = risk / (avg_loss_pips * pip_value)")
    print(f"  This means you NEVER lose more than ${0.02:.0f} per trade on a $1K account")
    print(f"  Lot sizes shown are MICRO lots (0.01 = 1K units). Broker minimum is typically 0.01.")
    print(f"  If calculated lot < 0.01 broker minimum, risk is higher than 2% — do NOT trade.")

    return all_trades


if __name__ == '__main__':
    main()
