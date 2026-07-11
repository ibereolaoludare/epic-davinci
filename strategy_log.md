# Algorithmic Trading Strategy Log

This document serves as a persistent record of every model architecture, feature set, and environment rule we have tested. It documents exactly what worked, what failed (faults), and why (precedences). 

**Rule:** Always review this document before proposing new model architectures or hyperparameter changes to avoid hallucinating past successes or repeating past failures.

---

## V1: The "Nuclear Option" (Forced Market Exposure)
**Goal:** Prove that a mathematical edge exists in the data.
**Setup:**
- **Timeframe:** 15-minute candles.
- **Features:** Raw Unscaled Prices (Close), Log Returns, GARCH Volatility, ARIMA Residuals.
- **Action Space:** `Discrete(2)` - Long or Short. The "Hold/Flat" action was completely removed to prevent the AI from hiding in cash (0-variance safety net).
- **Reward Horizon:** 10 candles (150 minutes).
- **Hyperparameters:** Default SB3 PPO (ent_coef=0.0).

**Result:** SUCCESS
- **Win Rate:** 57.1%
- **Trades:** 7 trades over 6,000 OOS candles (2.5 months).
- **Net Profit:** +$608.50
- **Faults / Lessons Learned:** Because we fed the neural network raw absolute prices (e.g., Close = 2450.5) alongside tiny decimals (Log_Returns = 0.0003), the network weights were crushed. It struggled to find frequent setups and only took 7 trades. However, forcing it to be "Always In" successfully proved that the ARIMA/GARCH features contained a real alpha edge.

---

## V2: The "Thrashing" Failure (High Frequency)
**Goal:** Increase trade frequency from 7 trades to a realistic day-trading volume.
**Setup:**
- **Features:** Dropped raw prices. Added Normalized Distance from 50-EMA, RSI (14), MACD, ATR, GARCH, ARIMA. All features scaled.
- **Action Space:** `Discrete(2)` - Always in the market.
- **Reward Horizon:** 4 candles (60 minutes).
- **Hyperparameters:** `ent_coef=0.01` (Forced exploration/curiosity).

**Result:** MASSIVE FAILURE
- **Win Rate:** 41.5%
- **Trades:** 2,669 trades over 6,000 candles.
- **Net Profit:** -$26,885.10
- **Faults / Lessons Learned:** The combination of a high `ent_coef` and a short 1-hour horizon caused the bot to develop ADHD. It wildly flipped its position on almost every single candle. The sheer volume of broker spread/commission fees ($13,000+) completely destroyed the account. **Precedence:** Do not use `ent_coef >= 0.01` with short horizons on 15m data, as the spread costs outweigh the edge.

---

## V2.1: The Macro "Buy and Hold" Discovery
**Goal:** Stop the thrashing while keeping the powerful normalized features.
**Setup:**
- **Features:** Same as V2 (Normalized RSI, MACD, ATR, etc).
- **Reward Horizon:** Reverted to 10 candles (150 minutes).
- **Hyperparameters:** Reverted `ent_coef` down to `0.0001`.

**Result:** SUCCESS (But Unexpected Behavior)
- **Win Rate:** 100%
- **Trades:** 1 trade over 6,000 candles.
- **Net Profit:** +$669.10
- **Faults / Lessons Learned:** With the thrashing penalty removed, the AI used the normalized momentum features to identify the macro-trend for the entire 2.5-month out-of-sample period. It took a single Long position on day 1 and held it forever. **Precedence:** The normalized features work exceptionally well for macro-trend identification, but without forced Stop Loss / Take Profit mechanics, the AI acts as a long-term investor rather than a day trader.

---

## V3 (Phase 1): The Day Trader Test (SL/TP)
**Goal:** Force the AI to take high-probability intraday trades rather than holding macro positions for months.
**Setup:**
- **Action Space:** Reverted to `Discrete(3)` (Long, Short, Flat). Added a `-0.1` boredom penalty for staying Flat to encourage finding entries.
- **Mechanics:** AI can only choose the *entry* direction. Once in a trade, the environment locks the position and simulates it until a hard Take Profit (+50 pts) or Stop Loss (-20 pts) is hit.

**Result:** MASSIVE FAILURE
- **Win Rate:** 42.0%
- **Trades:** 2,963 trades over 6,000 candles.
- **Net Profit:** -$27,794.70
- **Faults / Lessons Learned:** Classic RL alignment failure. The `-0.1` boredom penalty was far too harsh relative to the `+1.0` and `-1.0` terminal rewards. The AI realized that staying flat for 10 candles guaranteed a `-1.0` penalty, so it mathematically calculated that violently jumping into trades randomly (even with negative expected value due to spread) was "safer" than suffering the guaranteed slow bleed of the boredom penalty. **Precedence:** Do not use aggressive continuous negative rewards (boredom penalties) in discrete trading environments without perfectly scaling them to the terminal rewards, otherwise the AI will thrash just to stop the bleeding.
