# Epic DaVinci: V7 Quantitative Trading Framework

A machine-learning-driven forex trading system that achieved a **statistically significant edge** on USDJPY, EURUSD, and GBPUSD using regime-adaptive meta-labeling with CatBoost.

## Results (Walk-Forward Validated, Purged K-Fold)

| Pair | Trades | Win Rate | Net PnL | Profit Factor | p-value |
|------|--------|----------|---------|---------------|---------|
| USDJPY | 195 | 53.8% | +1,122 pips | 1.32 | 0.011 *** |
| EURUSD | 145 | 44.8% | +337 pips | 1.20 | 0.003 *** |
| GBPUSD | 164 | 45.7% | +826 pips | 1.35 | 0.0007 *** |
| **Portfolio** | **504** | **48.6%** | **+2,285 pips** | **1.30** | **0.000001 ***** |

All results are out-of-sample, walk-forward (6-month train / 2-month test, rolling), with purged k-fold boundary leak prevention. 11 of 12 months profitable.

## Position Sizing (2% Risk Per Trade)

| Account | Risk/Trade | Lot Size | Max Drawdown | Est. Total PnL |
|---------|-----------|----------|-------------|----------------|
| $1,000 | $20 | 0.07 | $255 (25%) | $1,375 |
| $2,500 | $50 | 0.17 | $636 (25%) | $3,438 |
| $5,000 | $100 | 0.35 | $1,273 (25%) | $6,877 |

**WARNING:** Expect 25% max drawdown. Individual months can swing +/-300 pips. The final 2-month holdout showed only USDJPY profitable. Trade at your own risk.

---

## Project Structure

```
epic-davinci/
├── feature_engineering_v7.py   # 53-feature zero-leakage pipeline
├── backtest_v7.py              # Walk-forward backtest engine
├── train_v7_prod.py            # Production model trainer (per-pair configs)
├── run_portfolio.py            # 3-pair portfolio backtest + sizing guide
├── trade_live.py               # Live MT5 trading daemon
├── timezone_utils.py           # Forex session detection
├── download_data.py            # MT5 H4 data downloader
├── fetch_yfinance_data.py      # Yahoo Finance H4 data downloader (backup)
├── fetch_alternative_data.py   # COT/VIX/DXY/US10Y fetcher
├── requirements.txt            # Python dependencies
├── strategy_log.md             # Full research log (V1-V7)
│
├── USDJPY_H4.csv               # Raw H4 candles
├── EURUSD_H4.csv
├── GBPUSD_H4.csv
├── USDJPY_features_v7.csv      # Computed features + targets
├── EURUSD_features_v7.csv
├── GBPUSD_features_v7.csv
├── central_bank_rates.csv      # Fed/BOJ rate history
├── alternative_data.csv        # COT/VIX/DXY/US10Y daily data
│
├── v7prod_primary_usdjpy.pkl   # Production primary model
├── v7prod_meta_usdjpy.pkl      # Production meta model
├── v7prod_config_usdjpy.json   # Production config (RR, regimes, etc.)
├── v7prod_primary_eurusd.pkl
├── v7prod_meta_eurusd.pkl
├── v7prod_config_eurusd.json
├── v7prod_primary_gbpusd.pkl
├── v7prod_meta_gbpusd.pkl
├── v7prod_config_gbpusd.json
```

---

## How It Works

### The Edge: "Hard Target, Easy Execution"

The key discovery is a mismatch between training labels and execution parameters:

- **Training label:** 2:1 RR, 12-bar (48h) horizon — a HARD target that forces the model to learn deep directional signal
- **Execution:** 1.2-2:1 RR, 6-bar (24h) horizon — a TIGHT exit that captures the move before it reverses

Matching both at the same value collapses performance. The mismatch is what creates the edge.

### Per-Pair Configs

| Pair | Execution RR | Horizon | Regimes Traded | Break-Even WR |
|------|-------------|---------|---------------|----------------|
| USDJPY | 1.2:1 | 6 bars (24h) | risk_on, intervention, range, trend | 45.5% |
| EURUSD | 2.0:1 | 6 bars (24h) | risk_on, mixed, risk_off | 33.3% |
| GBPUSD | 2.0:1 | 6 bars (24h) | range, mixed, risk_on | 33.3% |

### Architecture

1. **Primary Model (CatBoost MultiClass):** Predicts 3-way direction (long/short/neutral) from 53 features
2. **Meta Model (CatBoost Logloss):** Predicts whether the primary model's prediction will be correct. Acts as a veto gate — trades only execute when meta confidence >= 0.40
3. **Regime Filter:** Each pair has winning and losing market regimes. Trades only fire in the pair's profitable regimes
4. **ATR-Based SL/TP:** Stop-loss and take-profit are set at ATR multiples, sent as broker orders

### Features (53 total)

- **Technical:** RSI (14), MACD, ATR, ATR percentile, Bollinger Band width, Z-score, EMA 50/200 distance, EMA alignment
- **Momentum:** 6/12/30-bar price momentum, rolling volatility
- **Time:** Hour and day-of-week as sin/cos cyclic features, London/NY/Tokyo open flags
- **Session:** Tokyo session range breakout, distance from Tokyo mid
- **Macro:** Fed/BOJ rate differential, rate differential momentum, BOJ intervention zone distance
- **Alternative:** COT net positioning (z-score, change, extremes), US10Y (momentum, z-score, acceleration), VIX (regime, spike, calm, change, USDJPY divergence), DXY (momentum, z-score)
- **Composite:** Carry score (rate diff x risk-on x DXY stability)

All features use strictly backward-looking operations (rolling windows, ewm, `merge_asof direction='backward'`). No look-ahead.

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Get Data

**Option A: From MetaTrader 5** (recommended, real tick data)

```bash
# Requires MT5 terminal running and logged in
python download_data.py
# This saves USDJPY_H4.csv and central_bank_rates.csv
# For EURUSD/GBPUSD, edit SYMBOL in download_data.py and re-run
```

**Option B: From Yahoo Finance** (free, no MT5 needed)

```bash
python fetch_yfinance_data.py USDJPY
python fetch_yfinance_data.py EURUSD
python fetch_yfinance_data.py GBPUSD
```

### 3. Fetch Alternative Data

```bash
python fetch_alternative_data.py
# Downloads COT, VIX, DXY, US10Y data -> alternative_data.csv
```

### 4. Build Features

```bash
python feature_engineering_v7.py USDJPY
python feature_engineering_v7.py EURUSD
python feature_engineering_v7.py GBPUSD
# Saves {TICKER}_features_v7.csv
```

### 5. Backtest

```bash
# Single pair walk-forward backtest
python backtest_v7.py USDJPY

# Full 3-pair portfolio backtest with position sizing
python run_portfolio.py
```

### 6. Train Production Models

```bash
python train_v7_prod.py USDJPY
python train_v7_prod.py EURUSD
python train_v7_prod.py GBPUSD
# Saves v7prod_primary_*.pkl, v7prod_meta_*.pkl, v7prod_config_*.json
```

### 7. Run Live Trading

```bash
# USDJPY with $5,000 account, 2% risk per trade
python trade_live.py USDJPY --account 5000 --risk 0.02

# EURUSD with $1,000 account, 2% risk
python trade_live.py EURUSD --account 1000 --risk 0.02

# GBPUSD with $2,500 account, 1% risk (more conservative)
python trade_live.py GBPUSD --account 2500 --risk 0.01
```

**Prerequisites for live trading:**
1. MetaTrader5 terminal running and logged in
2. Production models trained (step 6)
3. Alternative data fetched (step 3)
4. Central bank rates saved (step 2)

---

## Live vs Backtest Parity

`trade_live.py` replicates `backtest_v7.py` exactly:

| Logic | Backtest | Live |
|-------|----------|------|
| Feature pipeline | `feature_engineering_v7.py` | Same |
| Primary model | CatBoost MultiClass predict | Same |
| Meta model veto | `predict_proba >= 0.40` | Same |
| Regime filter | Config JSON `regime_filter` | Same |
| SL/TP calculation | `ATR * tp_mult` / `ATR * sl_mult` | Same (sent as broker orders) |
| Horizon exit | Close at `horizon` bar | Same (manual close if SL/TP not hit) |
| One position at a time | `skip_until = exit_idx + 1` | Same (skip if active trade exists) |
| Pip sizes | 0.01 (JPY) / 0.0001 (others) | Same |
| Position sizing | N/A (backtest) | 2% risk-based lot calculation |

The only unavoidable discrepancy is slippage: the backtest enters at candle close, live enters at market price (ask/bid). This is inherent to live trading.

---

## Research History

The full research journey from V1 (RL on synthetic indices) through V7 (profitable forex edge) is documented in `strategy_log.md`. Key milestones:

1. **V1-V3:** Reinforcement Learning (PPO/LSTM) on M15 synthetic indices — collapsed to coin-flip once look-ahead was removed
2. **V4-V5:** Pivot to supervised learning (XGBoost) — proved M15 technicals contain no signal
3. **V6:** Upgrade to H4 timeframe with macro/alternative data — marginal improvement
4. **V7:** Regime-adaptive meta-labeling with CatBoost + "hard target, easy execution" — **statistically significant edge found**

---

## Disclaimer

This software is for educational and research purposes. Past performance does not guarantee future results. Forex trading carries substantial risk of loss. Never trade with money you cannot afford to lose. The authors are not responsible for any financial losses.
