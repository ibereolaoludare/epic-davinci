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
## V3 (Phase 1 Fixed): The Day Trader Test (Zero Boredom Penalty)
**Goal:** Run the SL/TP Day Trader test but remove the boredom penalty to stop the AI from thrashing.
**Setup:**
- **Action Space:** `Discrete(3)` (Long, Short, Flat).
- **Rewards:** `0.0` reward for staying flat. Hard SL (-20 pts) / TP (+50 pts) for terminal trade rewards.

**Result:** THE PACIFIST BOT
- **Win Rate:** 0.0%
- **Trades:** 0 trades over 6,000 candles.
- **Net Profit:** $0.00
- **Faults / Lessons Learned:** Without a boredom penalty forcing it into the market, the AI correctly identified that the cost of the broker spread (-0.5 points) makes day-trading on the 15-minute timeframe a negative expected-value (EV) game. Because it was not penalized for holding cash, it realized the only mathematically winning move is not to play. It sat perfectly still for 2.5 months. **Precedence:** 15m momentum features alone are not strong enough to guarantee a 50pt Take Profit frequently enough to overcome the spread. Do not use day-trading SL/TP mechanics on the V100 index with this feature set.

---

## V3 (Phase 2): The Volatility 75 Asset Test
**Goal:** Test the highly successful "Macro Buy and Hold" setup (V2.1) on the legendary Volatility 75 Index (V75), which is known for massive, wild, mean-reverting price action.
**Setup:**
- **Mechanics:** Reverted to V2.1 mechanics (`Discrete(2)` action space, no SL/TP, continuous market exposure).
- **Asset:** V75 (R_75).

**Result:** SUCCESS
- **Win Rate:** 40.0%
- **Trades:** 5 trades over 6,000 candles.
- **Net Profit:** +$583.50
- **Faults / Lessons Learned:** The AI adapted perfectly to the volatile nature of the V75 index. While it took 1 trade on V100, it took 5 trades on V75 because the macro-trends broke more frequently. Although the win rate was only 40%, the net profit was highly positive. **Precedence:** The model successfully learned classic trend-following mechanics: "cut your losers short, and let your winners run." It took small losses when the V75 trend whipped around, and rode massive waves when the trend held. This is a robust algorithmic baseline for volatile assets.
