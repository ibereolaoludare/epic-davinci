# Epic DaVinci: V7 Quantitative Trading Framework

A machine-learning-driven algorithmic trading system that achieves an **institutional-grade quantitative edge** across a 4-Asset Multi-Asset Basket (**USDJPY**, **GBPJPY**, **AUDUSD**, and **XAUUSD (Gold)**) using regime-adaptive meta-labeling with CatBoost, Macro VIX circuit breakers, and dynamic 2% risk compounding.

---

## 📊 Walk-Forward Performance (Out-of-Sample, Purged K-Fold)

| Pair / Asset | Target (R:R) | Horizon | Trades | Win Rate | Break-Even Rate | Net PnL (pips) | Profit Factor | p-value (Luck Test) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🇬🇧 **GBPJPY** | **2.0 : 1** | 6 bars (24h) | 42 | **66.7%** | 33.3% | **+1,407.7 pips** | **3.02** | **p = 0.000003** 🌟 |
| 🇯🇵 **USDJPY** | **1.5 : 1** | 6 bars (24h) | 142 | **51.4%** | 40.0% | **+1,641.1 pips** | **1.69** | **p = 0.003900** ⭐ |
| 🇦🇺 **AUDUSD** *(Sniper)* | **1.5 : 1** | 4 bars (16h) | 55 | **56.4%** | 40.0% | **+205.7 pips** | **1.44** | **p = 0.010300** 🟢 |
| 🟡 **XAUUSD (Gold)** | **2.0 : 1** | 4 bars (16h) | 119 | **51.3%** | 33.3% | **+5,394.8 pips** | **1.57** | **p = 0.000000** 🌟 |
| 🏛️ **COMBINED BASKET** | **1.7 : 1 Avg** | **4–6 bars** | **358** | **53.9%** | **37.0%** | **+8,649.3 pips** | **1.66** | **p = 0.000000** 🚀 |

*All results are strictly out-of-sample walk-forward test folds on real H4 broker tick data with non-overlapping purge buffers and zero look-ahead bias.*

---

## 📈 2% Dynamic Compounding Growth (,000 Starting Equity)

| Account Model | Initial Capital | Final Balance | Net Return | Max Drawdown | Risk Cap Per Trade |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Fixed Lot Model** | ,000.00 | ,186.61 | **+218.7%** | — | Fixed 0.05 Lot |
| **2% Dynamic Compounding** | **,000.00** | **,151.76** | **+715.2%** | **17.44%** | **Max 2% Balance** |

\\text{Lot Size} = \\frac{\\text{Account Balance} \\times 0.02}{\\text{ATR Stop Loss (pips)} \\times \\text{Pip Value per Lot}}

*Micro-lot dynamic scaling mathematically guarantees that you never lose more than  on a ,000 account per trade.*

---

## 🛡️ Core Institutional Safeguards

1. **Macro VIX Circuit Breaker:**
   * Automatically trips if $\\text{VIX} \\ge 28$ or 5-day $\\text{VIX Spike} > 30\%$.
   * Locks out new entries for **24 hours** during global panic, emergency central bank rate cuts, or flash geopolitical shocks.
2. **Persistent 24-Hour Disk State System (live_state_*.json):**
   * Persists circuit breaker lockout expiration timers, loss streak counters, cooldown states, and active trade tickets.
   * If you close or restart the terminal, it immediately recovers remaining lockout timers and never starts a timer from zero or takes unauthorized trades.
3. **Gold Contract & Spread Alignment:**
   * Dynamic pip valuation (.00 / lot per .10 move) and custom spread tolerance during daily rollover.
4. **Regime-Gated Execution:**
   * Uses a primary directional CatBoost model coupled with a Meta-Model veto ($\\text{Confidence} \\ge 0.50$) and pair-specific regime filters (Trend, Risk-On, Intervention).
5. **Rollover Spread Shield:**
   * Defers market-order closes during the 21:00–22:00 UTC daily bank rollover window or when broker spreads widen beyond threshold.

---

## 📂 Project Structure

`
epic-davinci/
├── trade_all_live.py           # Master multi-asset concurrent live orchestrator
├── run_master_live.bat         # Single-click master launcher with auto venv detection
├── run_all_live_trades.bat     # Multi-window live launcher (4 assets)
├── trade_live.py               # Live MT5 trading daemon with state persistence
├── feature_engineering_v7.py   # 59-feature zero-leakage pipeline
├── backtest_v7.py              # Walk-forward backtest engine
├── train_v7_prod.py            # Production model trainer (per-pair configs)
├── run_portfolio.py            # 4-asset portfolio walk-forward backtest & compounding simulation
├── fetch_alternative_data.py   # COT/VIX/DXY/US10Y daily macro fetcher
├── timezone_utils.py           # Forex session detection & UTC conversion
├── requirements.txt            # Python dependencies
│
├── USDJPY_features_v7.csv      # Computed features + targets
├── GBPJPY_features_v7.csv
├── AUDUSD_features_v7.csv
├── XAUUSD_features_v7.csv
├── central_bank_rates.csv      # Fed/BOJ/BOE/RBA rate history
├── alternative_data.csv        # Macro COT/VIX/DXY data
│
├── v7prod_primary_usdjpy.pkl   # Production primary models
├── v7prod_meta_usdjpy.pkl      # Production meta models
├── v7prod_config_usdjpy.json   # Production configs
├── v7prod_primary_gbpjpy.pkl
├── v7prod_meta_gbpjpy.pkl
├── v7prod_config_gbpjpy.json
├── v7prod_primary_audusd.pkl
├── v7prod_meta_audusd.pkl
├── v7prod_config_audusd.json
├── v7prod_primary_xauusd.pkl
├── v7prod_meta_xauusd.pkl
├── v7prod_config_xauusd.json
└── live_state_*.json           # Persistent runtime state (auto-generated)
`

---

## 🚀 Quick Start & Live Execution

### 1. Install Dependencies
`ash
pip install -r requirements.txt
`

### 2. Fetch Macro Data & Build Features
`ash
python fetch_alternative_data.py
python feature_engineering_v7.py USDJPY
python feature_engineering_v7.py GBPJPY
python feature_engineering_v7.py AUDUSD
python feature_engineering_v7.py XAUUSD
`

### 3. Train Production Models
`ash
python train_v7_prod.py USDJPY
python train_v7_prod.py GBPJPY
python train_v7_prod.py AUDUSD
python train_v7_prod.py XAUUSD
`

### 4. Run Live Trading

**Option A: Single-Process Master Orchestrator (Recommended)**
Double-click 
un_master_live.bat or run:
`ash
python trade_all_live.py --risk 0.02
`

**Option B: Standalone Single Asset Daemon**
`ash
python trade_live.py USDJPY --risk 0.02
python trade_live.py GBPJPY --risk 0.02
python trade_live.py AUDUSD --risk 0.02
python trade_live.py XAUUSD --risk 0.02
`

---

## ⚖️ Disclaimer

This software is for educational, research, and algorithmic trading purposes. Past performance does not guarantee future results. Financial trading carries inherent market risk. Always test in demo environments and apply strict risk management.
