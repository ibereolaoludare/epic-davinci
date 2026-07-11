import MetaTrader5 as mt5
import os
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

mt5.initialize()
mt5.login(32312791, password=os.getenv('MT5_PASSWORD'), server='Deriv-Demo')

# ── 1. Check symbol tick info ──────────────────────────────────────────────
sym = mt5.symbol_info('Volatility 100 Index')
print("=== SYMBOL INFO ===")
print(f"Tick Size:  {sym.trade_tick_size}")
print(f"Tick Value: {sym.trade_tick_value}")
print(f"Point:      {sym.point}")
print(f"Digits:     {sym.digits}")
print(f"1 lot, 50 point move = ${50 * sym.trade_tick_value / sym.trade_tick_size:.2f} USD")
print()

# ── 2. Check actual price scale ───────────────────────────────────────────
rates = mt5.copy_rates_from_pos('Volatility 100 Index', mt5.TIMEFRAME_M15, 0, 10)
df = pd.DataFrame(rates)
print("=== PRICE SCALE ===")
print("Sample closes:", df['close'].values)
mean_price = df['close'].mean()
print(f"Mean price: ~{mean_price:.2f}")
print(f"50 price units = {50/mean_price*100:.4f}% of price")
print()

# ── 3. Confidence distribution ────────────────────────────────────────────
print("=== AI CONFIDENCE AUDIT ===")
model = PPO.load('v100_crusher_archive/ppo_deriv_bot.zip')
rates_long = mt5.copy_rates_from_pos('Volatility 100 Index', mt5.TIMEFRAME_M15, 0, 300)
df_long = pd.DataFrame(rates_long)
df_long = df_long.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'})
df_long['Log_Return'] = np.log(df_long['Close'] / df_long['Close'].shift(1))
df_long = df_long.dropna().reset_index(drop=True)

sample_confs = []
for i in range(96, min(200, len(df_long))):
    obs_raw = df_long[['Close', 'High', 'Low', 'Open', 'Log_Return']].iloc[i-96:i].values
    obs = np.zeros((96, 7), dtype=np.float32)
    obs[:, :5] = obs_raw
    obs_t = torch.tensor(np.array([obs]), dtype=torch.float32)
    with torch.no_grad():
        dist = model.policy.get_distribution(obs_t)
        probs = dist.distribution.probs.cpu().numpy()[0]
    sample_confs.append(float(np.max(probs)))

print(f"Confidence stats over {len(sample_confs)} candles:")
print(f"  Min:              {min(sample_confs)*100:.1f}%")
print(f"  Max:              {max(sample_confs)*100:.1f}%")
print(f"  Mean:             {np.mean(sample_confs)*100:.1f}%")
print(f"  Fraction above 60%: {sum(c > 0.60 for c in sample_confs)/len(sample_confs)*100:.1f}%")
print(f"  Fraction above 65%: {sum(c > 0.65 for c in sample_confs)/len(sample_confs)*100:.1f}%")
print(f"  Fraction above 70%: {sum(c > 0.70 for c in sample_confs)/len(sample_confs)*100:.1f}%")

# ── 4. Profit logic check ─────────────────────────────────────────────────
print()
print("=== PROFIT LOGIC SANITY CHECK ===")
tick_value = sym.trade_tick_value
tick_size = sym.trade_tick_size
point_value_per_lot = tick_value / tick_size
print(f"Point value per lot: ${point_value_per_lot:.4f}")
print(f"50-point SL with 10 lots = ${50 * point_value_per_lot * 10:.2f} risk")
print(f"50-point SL with 1% Kelly (on $10k) = ${100:.2f} risk => volume = {100 / (50 * point_value_per_lot):.2f} lots")

mt5.shutdown()
