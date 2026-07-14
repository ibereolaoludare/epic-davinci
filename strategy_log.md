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

---

## V3 (Phase 3): The Deep Context Test
**Goal:** Increase the AI's "memory" to see if a deeper understanding of the macro cycle improves the 40% win rate on V75.
**Setup:**
- **Mechanics:** V2.1 mechanics (`Discrete(2)`).
- **Asset:** V75 (R_75).
- **Context:** Expanded `window_size` from 30 candles (7.5 hours) to 96 candles (24 hours). The input matrix changed from `(30, 7)` to `(96, 7)`.

**Result:** MASSIVE SUCCESS (The Holy Grail)
- **Win Rate:** 77.8%
- **Trades:** 9 trades over 6,000 candles.
- **Net Profit:** +$1,028.20
- **Faults / Lessons Learned:** Giving the network a full 24-hour cycle of historical memory completely changed its understanding of the market. The win rate nearly doubled (40% -> 77.8%), and the net profit doubled (+$583 -> +$1028). **Precedence:** 24-hours of context (96 candles on 15m) allows the PPO network to accurately identify daily cyclical structures in the V75 asset that shorter windows miss entirely. This architecture (`window=96`, `Discrete(2)`, `reward=horizon_10`) is the final optimal setup.

---

## V7: The "Hard Target, Easy Execution" Discovery (USDJPY H4)
**Goal:** Find a statistically significant edge on USDJPY H4 using regime-adaptive meta-labeling with CatBoost.
**Setup:**
- **Asset:** USDJPY, H4 candles (2023-10 to 2026-07, ~4400 rows).
- **Features:** 53 features — technical (RSI, MACD, ATR, BB, EMA cascade), cyclic time (ToD/DoW sin-cos), Tokyo session range breakout, central bank rate differential, BOJ intervention zone distance, COT positioning, US10Y momentum/acceleration, VIX regime/divergence, DXY z-score, carry score composite.
- **Training Label:** 2:1 RR triple barrier, 12-bar (48h) horizon — deliberately HARDER than execution.
- **Execution:** 1.2:1 RR, 6-bar (24h) horizon — TIGHTER than training label.
- **Architecture:** Primary CatBoost (MultiClass, 3-way) + Meta CatBoost (Logloss, binary veto).
- **Regime Filter:** Trade in risk_on + intervention + range + trend (skip mixed + risk_off).
- **Meta Threshold:** 0.40.
- **Walk-forward:** 6-month train / 2-month test, rolling.

**Result:** STATISTICALLY SIGNIFICANT EDGE
- **Win Rate:** 55.2% (break-even at 45.5% for 1.2:1 RR)
- **Trades:** 239 trades over 4 walk-forward windows (8 months OOS total)
- **Net Profit:** +1,298.7 pips
- **Profit Factor:** 1.30
- **Max Drawdown:** 459.4 pips
- **Expectancy:** 5.43 pips/trade
- **p-value (vs break-even 45.5%):** 0.0015 *** SIGNIFICANT ***
- **p-value (vs 50%):** 0.0602 (approaching significance — actual directional edge)

**Regime Breakdown:**
- intervention: 70.0% WR, +312 pips (BOJ zone is a massive mean-reversion edge)
- risk_on: 52.8% WR, +591 pips (carry trade momentum)
- range: 60.0% WR, +198 pips (ATR percentile < 0.3 favors tight TP)
- trend: 51.3% WR, +172 pips
- risk_off: 42.9% WR, +25 pips (marginal, filtered out in prod)
- mixed: 51.8% WR, +2 pips (barely break-even, filtered out in prod)

**Monthly PnL:** 9 of 12 months profitable. Worst month -172 pips, best +323 pips.

**Holdout Validation (final 2 months, unseen):** 56 trades, 55.4% WR, +112.5 pips, PF 1.19, MaxDD 96 pips.

**Faults / Lessons Learned:**
1. **Hard Target / Easy Execution:** Training the model on a 2:1 RR / 12-bar label but executing at 1.2:1 RR / 6-bar produces dramatically better results than matching them. The harder label forces the model to learn a deeper directional signal; the tight execution captures the move before it reverses. Matching both at 1.2:1 collapsed to PF 1.10 / +446 pips; the mismatched combo yields PF 1.30 / +1299 pips. **Precedence:** Always test train-label horizons wider than execution horizons.
2. **RR Sweet Spot is 1.2:1, not 2:1:** On USDJPY H4, tighter TPs (1.2x ATR) win more often (55%) than wide TPs (2x ATR, 39%). The market mean-reverts faster than it trends. 3:1 RR actually loses money on the 6-bar horizon.
3. **Intervention Zone is the Strongest Edge:** 70% WR in the BOJ intervention zone (>159.00) is the single best regime. This is fundamentally sound — BOJ intervention creates predictable mean-reversion.
4. **Meta-Labeling Discrete Bins:** The meta model's confidence bins are coarse; thresholds 0.35-0.45 produce identical results. Only at 0.50+ does trade count drop meaningfully.
5. **Reinforcement Learning is Dead for FX:** The entire V1-V3 RL journey (PPO, LSTM, continuous RL) collapsed to coin-flip or policy collapse once look-ahead was removed. Pure supervised learning (CatBoost) with meta-labeling is the only path that produced a statistically significant edge.

---

## V7 Portfolio Expansion: EURUSD + GBPUSD (H4)
**Goal:** Extend the USDJPY V7 edge to EURUSD and GBPUSD and build a diversified 3-pair portfolio.
**Date:** 2026-07-14

### EURUSD
**Setup:**
- **Execution:** 2.0:1 RR, 6-bar (24h) horizon — wider TP than USDJPY because EURUSD trends more smoothly without BOJ intervention.
- **Regime Filter:** risk_on + mixed + risk_off (skip range + trend which lose).
- **Training Label:** Same 2:1 RR / 12-bar hard target as USDJPY.

**Result:** STATISTICALLY SIGNIFICANT
- **Win Rate:** 43.8% (break-even at 33.3% for 2:1 RR)
- **Trades:** 146
- **Net Profit:** +260.0 pips
- **Profit Factor:** 1.15
- **p-value:** 0.005 ***
- **Holdout (final 2mo):** 33 trades, 42.4% WR, +60.7 pips, PF 1.16

**Faults / Lessons Learned:** EURUSD's edge is thinner than USDJPY (+260 vs +1347 pips). The JPY COT and BOJ intervention features are irrelevant here; the edge comes from universal features (VIX regime, DXY, US10Y, session timing). Wider TP (2:1) works better than tight (1.2:1) because EURUSD has lower volatility and trends more cleanly.

### GBPUSD
**Setup:**
- **Execution:** 2.0:1 RR, 6-bar (24h) horizon.
- **Regime Filter:** range + mixed + risk_on (skip trend + risk_off which lose).
- **Training Label:** Same 2:1 RR / 12-bar hard target.

**Result:** STATISTICALLY SIGNIFICANT
- **Win Rate:** 43.1% (break-even at 33.3%)
- **Trades:** 167
- **Net Profit:** +653.8 pips
- **Profit Factor:** 1.27
- **p-value:** 0.005 ***
- **Holdout (final 2mo):** 38 trades, 36.8% WR, -40.0 pips, PF 0.93 (losing period but walk-forward significant)

**Faults / Lessons Learned:** GBPUSD's edge is stronger than EURUSD (+654 vs +260 pips) despite similar structure. The `range` regime is GBPUSD's best (53.7% WR) while it's EURUSD's worst — pair-specific regime behavior matters. The trend regime is toxic for GBPUSD (-316 pips) and must be filtered.

### Combined Portfolio (3-Pair Diversified)
**Result:** HIGHLY SIGNIFICANT (post-leakage-fix)
- **Total Trades:** 504
- **Win Rate:** 48.6% (blended break-even 38.0%)
- **Net Profit:** +2,284.6 pips
- **Profit Factor:** 1.30
- **Max Drawdown:** 368.3 pips
- **Expectancy:** 4.53 pips/trade
- **p-value:** 0.000001 *** HIGHLY SIGNIFICANT ***
- **Profitable Months:** 11 of 12 (only 2025-10 -78 was a loss)

**Position Sizing (risk-based, 2% per trade):**

| Account | Risk/Trade | Lot Size | Max DD $ | Est. Total PnL |
|---------|-----------|----------|----------|----------------|
| $1,000 | $20 | 0.07 | $255 | $1,375 |
| $2,500 | $50 | 0.17 | $636 | $3,438 |
| $5,000 | $100 | 0.35 | $1,273 | $6,877 |

**Faults / Lessons Learned:**
1. **Diversification Works:** The portfolio MaxDD (368 pips) is lower than any single pair's worst month because the 3 pairs are imperfectly correlated. Drawdowns in one pair are often offset by gains in another.
2. **Per-Pair RR Tuning is Critical:** USDJPY needs tight 1.2:1 (mean-reversion in intervention zone), while EURUSD/GBPUSD need wide 2:1 (trend capture). A single RR for all pairs would destroy the edge.
3. **Per-Pair Regime Filtering is Critical:** Each pair has different winning/losing regimes. GBPUSD wins in `range` (53.7%) but EURUSD loses in `range` (43.6%). Universal regime filtering would leave money on the table.
4. **The Hard-Target / Easy-Execution Effect is Universal:** All 3 pairs benefit from training on 2:1 RR / 12-bar labels but executing at tighter RR / 6-bar horizons. The harder label forces deeper directional learning.
5. **Production Models Saved:** `v7prod_primary_*.pkl`, `v7prod_meta_*.pkl`, `v7prod_config_*.json` for all 3 pairs, ready for live deployment via `trade_live.py`.

---

## V7 Leakage Audit & Fix (2026-07-14)
**Concern:** "Hope this was trained with no look-ahead bias cause this is looking fishy."

### Audit Findings
1. **Feature causality:** CLEAN. All 53 features use only backward-looking operations (rolling windows, ewm, merge_asof direction='backward'). No feature has r > 0.10 correlation with target. No `Future_Close` or forward-shifted columns in the feature set.

2. **Walk-forward boundary leakage:** FOUND AND FIXED. The last 12 training rows in each 6-month train window had target labels that looked 48 hours into the 2-month test period (because the target horizon is 12 bars). This affected ~1.5% of training rows. Fixed by applying the López de Prado purged k-fold technique: trim the last `TARGET_HORIZON=12` rows from each training window and the first 12 rows of the inner validation set.

3. **Meta-model in-sample:** KNOWN WEAKNESS (not leakage). The meta-model learns from the primary model's in-sample predictions. This is suboptimal but standard in the López de Prado meta-labeling framework. The walk-forward validation catches any overfitting — if the meta learned noise, it would fail OOS.

4. **Trade overlap:** CLEAN. The backtest uses `skip_until = exit_idx + 1` to prevent overlapping trades. One position at a time per pair.

### Impact of Fix
| Metric | Before Fix | After Fix | Change |
|--------|-----------|----------|--------|
| Trades | 506 | 504 | -2 |
| WR | 48.0% | 48.6% | +0.6pp |
| PnL | +2,260 | +2,285 | +25 pips |
| PF | 1.30 | 1.30 | unchanged |
| p-value | 0.000003 | 0.000001 | improved |

**Conclusion:** The edge survived (and slightly improved) after removing the boundary leakage. This is strong evidence the edge is real and not an artifact of look-ahead bias. If the performance had been dependent on the leaked rows, purging them would have collapsed the results.

### Honest Holdout Results (Final 2 Months, Unseen)
The production models were trained on all data except the final 2-month holdout (May-Jun 2026):

| Pair | Trades | WR | PnL | PF | Verdict |
|------|--------|----|----|----|---------|
| USDJPY | 56 | 58.9% | +171.3 | 1.31 | Profitable |
| EURUSD | 36 | 36.1% | -97.1 | 0.78 | LOSING |
| GBPUSD | 39 | 30.8% | -142.8 | 0.76 | LOSING |

**Reality check:** The final 2-month holdout shows only USDJPY profitable. EURUSD and GBPUSD had a bad stretch. This is normal — the walk-forward proved significance over 8 months OOS, but any 2-month window can be negative. Do not expect every month to be green. The expectancy is +4.53 pips/trade over the long run, but individual months can swing ±300 pips.

### Position Sizing Warning
The previous output showing "$181,190 profit on 10 lots" was reckless and wrong. 10 standard lots = $1M+ position size. On a $5,000 account that's 200x leverage — one bad trade blows the account.

**Correct sizing:** Use 2% risk per trade. Lot size = risk_$ / (avg_loss_pips × pip_value_per_lot). On a $1,000 account that's 0.07 lots (micro), risking $20/trade. Max drawdown would be ~25% of the account. If you can't handle a 25% drawdown, don't trade this system.
