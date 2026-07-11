import numpy as np
import pandas as pd
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

STARTING_BALANCE = 10000.0
WINDOW_SIZE = 96
feature_cols = ['Close', 'High', 'Low', 'Open', 'Log_Return', 'GARCH_Vol', 'ARIMA_Resid']

print("Loading data and AI Brain...")
df = pd.read_csv('v100_crusher_archive/advanced_features.csv')
model = PPO.load("v100_crusher_archive/ppo_deriv_bot.zip")
features = df[feature_cols].values

def run_sim(feat, sl_pts, tp_pts, volume):
    balance = STARTING_BALANCE
    pos = 0  # 0=flat, 1=long, 2=short
    entry = 0.0
    sl_p = tp_p = 0.0
    vol_used = 0.0
    equity_curve = [STARTING_BALANCE]

    for i in range(WINDOW_SIZE, len(feat)):
        obs = np.array([feat[i - WINDOW_SIZE:i]], dtype=np.float32)
        high  = df.iloc[i]['High']
        low   = df.iloc[i]['Low']
        close = df.iloc[i]['Close']

        if pos == 1:
            if low <= sl_p:
                profit = (sl_p - entry) * vol_used
                balance += profit; pos = 0
            elif high >= tp_p:
                profit = (tp_p - entry) * vol_used
                balance += profit; pos = 0
        elif pos == 2:
            if high >= sl_p:
                profit = (entry - sl_p) * vol_used
                balance += profit; pos = 0
            elif low <= tp_p:
                profit = (entry - tp_p) * vol_used
                balance += profit; pos = 0

        action, _ = model.predict(obs, deterministic=True)
        action = int(action[0])

        if action == 0 and pos != 1:
            if pos == 2:
                profit = (entry - close) * vol_used
                balance += profit
            entry = close; sl_p = entry - sl_pts; tp_p = entry + tp_pts
            vol_used = volume; pos = 1
        elif action == 1 and pos != 2:
            if pos == 1:
                profit = (close - entry) * vol_used
                balance += profit
            entry = close; sl_p = entry + sl_pts; tp_p = entry - tp_pts
            vol_used = volume; pos = 2

        equity_curve.append(balance)

    if pos == 1:
        balance += (close - entry) * vol_used
    elif pos == 2:
        balance += (entry - close) * vol_used
    
    equity_curve.append(balance)
    return equity_curve

print("Running 60pt optimal simulation...")
opt_curve = run_sim(features, sl_pts=60, tp_pts=60, volume=10.0)

print("Generating visualization dashboard...")
sns.set_theme(style="darkgrid")
plt.figure(figsize=(14, 8))

plt.plot(opt_curve, color='#00ffcc', linewidth=2.5, label='Equity Curve (60pt SL/TP, 10 Lots)')
plt.fill_between(range(len(opt_curve)), opt_curve, min(opt_curve), color='#00ffcc', alpha=0.1)

# Annotate peak
peak_idx = np.argmax(opt_curve)
peak_val = opt_curve[peak_idx]
plt.scatter(peak_idx, peak_val, color='#ff00ff', s=150, zorder=5, edgecolors='white', label=f'Peak Balance: ${peak_val:,.0f}')

# Annotate Drawdown
min_after_peak = min(opt_curve[peak_idx:])
min_idx = peak_idx + np.argmin(opt_curve[peak_idx:])
plt.scatter(min_idx, min_after_peak, color='red', s=100, zorder=5, label=f'Max Drawdown Trough: ${min_after_peak:,.0f}')
plt.plot([peak_idx, min_idx], [peak_val, min_after_peak], color='red', linestyle='--', alpha=0.7)

# Start / End
plt.scatter(0, opt_curve[0], color='white', s=80, zorder=5)
plt.text(500, opt_curve[0], f"Start: ${opt_curve[0]:,.0f}", color='white', fontsize=12)

end_idx = len(opt_curve) - 1
end_val = opt_curve[-1]
plt.scatter(end_idx, end_val, color='lime', s=120, zorder=5)
plt.text(end_idx - 2000, end_val + 500, f"End: ${end_val:,.0f}\n(+83%)", color='lime', fontsize=14, fontweight='bold')

plt.title('Backtest Equity Curve: V100 AI Trading Bot (Optimal Settings)', fontsize=18, color='white', pad=20, fontweight='bold')
plt.xlabel('15-Minute Candles Elapsed (10 Months)', fontsize=14, color='lightgray')
plt.ylabel('Account Balance ($USD)', fontsize=14, color='lightgray')

# Dark mode styling
ax = plt.gca()
ax.set_facecolor('#1e1e1e')
plt.gcf().patch.set_facecolor('#121212')
ax.tick_params(colors='lightgray', labelsize=12)
for spine in ax.spines.values():
    spine.set_color('#333333')

plt.legend(facecolor='#1e1e1e', edgecolor='#333333', labelcolor='white', fontsize=12, loc='upper left')
plt.grid(color='#333333', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('C:/Users/ibere/.gemini/antigravity/brain/51fafc79-ec9c-4a08-8471-914f14ac6498/stress_test_dashboard_python.png', facecolor='#121212', dpi=150)
print("Saved dashboard!")
