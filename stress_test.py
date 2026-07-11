"""
Comprehensive Stress Test Suite (Fixed 10 Lots, 60pt SL/TP)
Tests:
  A. 5-Fold Walk-Forward Validation (does it work in EVERY 2-month window?)
  B. Fine SL/TP Sweep (is 60pt the true optimal, or just lucky?)
  C. Lot Size Sensitivity (how does lot size scale profit?)
  D. Drawdown Analysis (what's the worst losing streak we ever hit?)
"""
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
import warnings
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings("ignore")

STARTING_BALANCE = 10000.0
WINDOW_SIZE = 96
feature_cols = ['Close', 'High', 'Low', 'Open', 'Log_Return', 'GARCH_Vol', 'ARIMA_Resid']

print("Loading data and AI Brain...")
df = pd.read_csv('v100_crusher_archive/advanced_features.csv')
model = PPO.load("v100_crusher_archive/ppo_deriv_bot.zip")
features = df[feature_cols].values
print(f"Total candles: {len(df)}\n")

# -----------------------------------------------------------------------
def run_sim(feat, sl_pts, tp_pts, volume, start_idx=0, end_idx=None):
    """Runs a simulation on a slice of the feature matrix. Returns dict of results."""
    if end_idx is None:
        end_idx = len(feat)
    feat_slice = feat[start_idx:end_idx]

    balance = STARTING_BALANCE
    pos = 0  # 0=flat, 1=long, 2=short
    entry = 0.0
    sl_p = tp_p = 0.0
    vol_used = 0.0
    trades = wins = 0
    equity_curve = [STARTING_BALANCE]
    max_dd = 0.0
    peak = STARTING_BALANCE

    for i in range(WINDOW_SIZE, len(feat_slice)):
        obs = np.array([feat_slice[i - WINDOW_SIZE:i]], dtype=np.float32)
        high  = df.iloc[start_idx + i]['High']
        low   = df.iloc[start_idx + i]['Low']
        close = df.iloc[start_idx + i]['Close']

        # SL/TP check
        if pos == 1:
            if low <= sl_p:
                profit = (sl_p - entry) * vol_used
                balance += profit; trades += 1
                if profit > 0: wins += 1
                pos = 0
            elif high >= tp_p:
                profit = (tp_p - entry) * vol_used
                balance += profit; trades += 1
                if profit > 0: wins += 1
                pos = 0
        elif pos == 2:
            if high >= sl_p:
                profit = (entry - sl_p) * vol_used
                balance += profit; trades += 1
                if profit > 0: wins += 1
                pos = 0
            elif low <= tp_p:
                profit = (entry - tp_p) * vol_used
                balance += profit; trades += 1
                if profit > 0: wins += 1
                pos = 0

        action, _ = model.predict(obs, deterministic=True)
        action = int(action[0])

        if action == 0 and pos != 1:
            if pos == 2:
                profit = (entry - close) * vol_used
                balance += profit; trades += 1
                if profit > 0: wins += 1
            entry = close; sl_p = entry - sl_pts; tp_p = entry + tp_pts
            vol_used = volume; pos = 1
        elif action == 1 and pos != 2:
            if pos == 1:
                profit = (close - entry) * vol_used
                balance += profit; trades += 1
                if profit > 0: wins += 1
            entry = close; sl_p = entry + sl_pts; tp_p = entry - tp_pts
            vol_used = volume; pos = 2

        equity_curve.append(balance)
        if balance > peak:
            peak = balance
        dd = (peak - balance) / peak * 100
        if dd > max_dd:
            max_dd = dd

    # Close final position
    if pos == 1:
        profit = (close - entry) * vol_used
        balance += profit; trades += 1
        if profit > 0: wins += 1
    elif pos == 2:
        profit = (entry - close) * vol_used
        balance += profit; trades += 1
        if profit > 0: wins += 1

    return {
        "net_profit": balance - STARTING_BALANCE,
        "trades": trades,
        "win_rate": (wins / trades * 100) if trades > 0 else 0,
        "max_drawdown_pct": max_dd,
        "equity_curve": equity_curve,
    }

# ===================================================================
# A. 5-FOLD WALK-FORWARD VALIDATION (Optimal: 10 lots, 60pt)
# ===================================================================
print("=" * 70)
print("  A. 5-FOLD WALK-FORWARD: Fixed 10 lots, 60pt SL/TP")
print("=" * 70)
n_folds = 5
fold_size = len(features) // n_folds
total = 0
all_profitable = True
for fold in range(n_folds):
    s = fold * fold_size
    e = s + fold_size if fold < n_folds - 1 else len(features)
    r = run_sim(features, sl_pts=60, tp_pts=60, volume=10.0, start_idx=s, end_idx=e)
    total += r['net_profit']
    flag = "[OK]" if r['net_profit'] > 0 else "[LOSS]"
    if r['net_profit'] <= 0: all_profitable = False
    print(f"  Fold {fold+1} (~2 months): Profit=${r['net_profit']:>+9.2f}  "
          f"Trades={r['trades']:>4}  WinRate={r['win_rate']:.1f}%  "
          f"MaxDD={r['max_drawdown_pct']:.2f}%  {flag}")
print(f"  {'':->60}")
print(f"  Total across all folds: ${total:+.2f}")
print(f"  All folds profitable?   {'YES' if all_profitable else 'NO'}\n")

# ===================================================================
# B. FINE SL/TP SWEEP (fixed 10 lots on full dataset)
# ===================================================================
print("=" * 70)
print("  B. FINE SL/TP SWEEP: Fixed 10 lots, varying SL/TP (full dataset)")
print("=" * 70)
sweep_pts = [10, 20, 25, 30, 40, 50, 60, 75, 100, 150, 200, 500]
print(f"  {'SL/TP':>6} | {'Net Profit':>12} | {'Trades':>6} | {'Win %':>5} | {'Max DD':>7}")
print(f"  {'-'*55}")
best_profit = -999999
best_pts = 0
for pts in sweep_pts:
    r = run_sim(features, sl_pts=pts, tp_pts=pts, volume=10.0)
    marker = " <-- BEST" if r['net_profit'] > best_profit else ""
    if r['net_profit'] > best_profit:
        best_profit = r['net_profit']
        best_pts = pts
    print(f"  {pts:>6}pt | ${r['net_profit']:>11.2f} | {r['trades']:>6} | "
          f"{r['win_rate']:>4.1f}% | {r['max_drawdown_pct']:>6.2f}%{marker}")
print(f"\n  Optimal SL/TP: {best_pts}pt (${best_profit:+.2f})\n")

# ===================================================================
# C. LOT SIZE SENSITIVITY (60pt SL/TP on full dataset)
# ===================================================================
print("=" * 70)
print("  C. LOT SIZE SENSITIVITY: 60pt SL/TP, varying lot size")
print("=" * 70)
lots = [0.5, 1.0, 2.0, 5.0, 8.0, 10.0, 15.0, 20.0]
print(f"  {'Lots':>6} | {'Net Profit':>12} | {'$ per Trade':>12} | {'Risk/Trade':>10}")
print(f"  {'-'*55}")
for lot in lots:
    r = run_sim(features, sl_pts=60, tp_pts=60, volume=lot)
    per_trade = r['net_profit'] / r['trades'] if r['trades'] > 0 else 0
    risk = 60 * lot
    print(f"  {lot:>6} | ${r['net_profit']:>11.2f} | ${per_trade:>11.2f} | ${risk:>9.2f}")
print()

# ===================================================================
# D. DRAWDOWN DEEP-DIVE (Optimal full dataset)
# ===================================================================
print("=" * 70)
print("  D. DRAWDOWN DEEP-DIVE: Fixed 10 lots, 60pt (full dataset)")
print("=" * 70)
r = run_sim(features, sl_pts=60, tp_pts=60, volume=10.0)
curve = np.array(r['equity_curve'])
print(f"  Starting Balance: ${STARTING_BALANCE:,.2f}")
print(f"  Ending Balance:   ${STARTING_BALANCE + r['net_profit']:,.2f}")
print(f"  Net Profit:       ${r['net_profit']:+,.2f}")
print(f"  Total Trades:     {r['trades']}")
print(f"  Win Rate:         {r['win_rate']:.1f}%")
print(f"  Max Drawdown:     {r['max_drawdown_pct']:.2f}%  (${(r['max_drawdown_pct']/100)*max(curve):,.2f})")
print(f"  Lowest Balance:   ${min(curve):,.2f}")
print(f"  Peak Balance:     ${max(curve):,.2f}")
consecutive_losses = 0
max_consec_losses = 0
for i in range(1, len(curve)):
    if curve[i] < curve[i-1]:
        consecutive_losses += 1
        max_consec_losses = max(max_consec_losses, consecutive_losses)
    else:
        consecutive_losses = 0
print(f"  Max Consec. Loss Candles: {max_consec_losses}")
print()
print("  Final Verdict:")
verdict_profit_ok = r['net_profit'] > 5000
verdict_dd_ok     = r['max_drawdown_pct'] < 20
verdict_wr_ok     = r['win_rate'] > 50
print(f"    Profit > $5000?           {'YES' if verdict_profit_ok else 'NO'} (${r['net_profit']:,.2f})")
print(f"    Max Drawdown < 20%?       {'YES' if verdict_dd_ok else 'NO'} ({r['max_drawdown_pct']:.2f}%)")
print(f"    Win Rate > 50%?           {'YES' if verdict_wr_ok else 'NO'} ({r['win_rate']:.1f}%)")
print(f"    DEPLOYABLE?               {'ABSOLUTELY YES' if all([verdict_profit_ok, verdict_dd_ok, verdict_wr_ok]) else 'NEEDS REVIEW'}")
print()

# ===================================================================
# E. VISUALIZATION DASHBOARD
# ===================================================================
print("Generating visualization dashboard...")
sns.set_theme(style="darkgrid")
plt.figure(figsize=(14, 8))

# We'll plot the 60pt optimal curve since we found it to be the best.
r_opt = run_sim(features, sl_pts=60, tp_pts=60, volume=10.0)
opt_curve = r_opt['equity_curve']

plt.plot(opt_curve, color='cyan', linewidth=2, label='Equity Curve (60pt SL/TP)')
plt.fill_between(range(len(opt_curve)), opt_curve, min(opt_curve), color='cyan', alpha=0.1)

# Annotate peak and drawdown
peak_idx = np.argmax(opt_curve)
peak_val = opt_curve[peak_idx]
plt.scatter(peak_idx, peak_val, color='lime', s=100, zorder=5, label=f'Peak: ${peak_val:,.0f}')

plt.title('Backtest Equity Curve: V100 AI Bot (Optimal 60pt SL/TP)', fontsize=16, color='white', pad=20)
plt.xlabel('15-Minute Candles (10 Months)', fontsize=12, color='lightgray')
plt.ylabel('Account Balance ($)', fontsize=12, color='lightgray')

# Dark mode styling
ax = plt.gca()
ax.set_facecolor('#1e1e1e')
plt.gcf().patch.set_facecolor('#121212')
ax.tick_params(colors='lightgray')
for spine in ax.spines.values():
    spine.set_color('#333333')

plt.legend(facecolor='#1e1e1e', edgecolor='#333333', labelcolor='white')
plt.tight_layout()
plt.savefig('stress_test_dashboard.png', facecolor='#121212', dpi=150)
print("Saved dashboard to stress_test_dashboard.png")
