"""
Live Trading Daemon — V7 Production

Executes trades using the EXACT same logic as backtest_v7.py:
  1. Fetch latest closed H4 candles from MT5
  2. Compute V7 features (feature_engineering_v7.py)
  3. Primary CatBoost model predicts direction (1=long, -1=short, 0=skip)
  4. Meta CatBoost model predicts confidence; skip if < meta_threshold
  5. Regime filter: skip if current regime not in config's allowed list
  6. ATR-based SL/TP sent as broker orders (same formulas as backtest)
  7. If neither SL nor TP hits within horizon, close at horizon close
  8. One position at a time per symbol
  9. Risk-based lot sizing: 2% of account per trade

Usage:
  python trade_live.py USDJPY --risk 0.02
  python trade_live.py EURUSD --risk 0.02
  python trade_live.py GBPUSD --risk 0.02

Prerequisites:
  - MetaTrader5 terminal running and logged in
  - pip install MetaTrader5 pandas numpy catboost joblib ta pykalman hmmlearn arch statsmodels
  - Production models trained: python train_v7_prod.py USDJPY
  - Alternative data fetched: python fetch_alternative_data.py
  - Central bank rates: python download_data.py (saves central_bank_rates.csv)
"""
import time
import logging
import json
import argparse
import threading
import os
import subprocess
import sys
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timezone, timedelta
from feature_engineering_v7 import process_features_v7, get_feature_cols
from timezone_utils import get_session, to_utc

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
log = logging.getLogger('LiveV7')


def load_config(ticker):
    cfg_path = f'v7prod_config_{ticker.lower()}.json'
    with open(cfg_path) as f:
        cfg = json.load(f)
    primary = joblib.load(f'v7prod_primary_{ticker.lower()}.pkl')
    meta = joblib.load(f'v7prod_meta_{ticker.lower()}.pkl')
    return cfg, primary, meta


def calc_lot_size(account_balance, risk_pct, sl_pips, pip_value_per_lot=10.0, info=None, pip_size=0.0001):
    """
    Exact broker lot sizing: Volume = (Balance * Risk) / ((Entry - SL) * (TickValue / TickSize))
    Quantizes downward using broker volume_step and enforces volume_min / volume_max.
    Prevents blowing risk rules on small accounts.
    """
    risk_amount = account_balance * risk_pct
    if info and hasattr(info, 'trade_tick_value') and hasattr(info, 'trade_tick_size') and info.trade_tick_value > 0 and info.trade_tick_size > 0:
        sl_dist = sl_pips * pip_size
        value_per_unit = info.trade_tick_value / info.trade_tick_size
        raw_lot = risk_amount / (sl_dist * value_per_unit) if (sl_dist * value_per_unit) > 0 else 0.0
        effective_pip_value = value_per_unit * pip_size
    else:
        raw_lot = risk_amount / (sl_pips * pip_value_per_lot) if (sl_pips * pip_value_per_lot) > 0 else 0.0
        effective_pip_value = pip_value_per_lot

    step = info.volume_step if (info and hasattr(info, 'volume_step') and info.volume_step > 0) else 0.01
    min_lot = info.volume_min if (info and hasattr(info, 'volume_min') and info.volume_min > 0) else 0.01
    max_lot = info.volume_max if (info and hasattr(info, 'volume_max') and info.volume_max > 0) else 100.0

    # Quantize downward to nearest volume step
    lot = np.floor(raw_lot / step) * step

    if lot < min_lot:
        actual_risk_pct = (min_lot * sl_pips * effective_pip_value) / account_balance if account_balance > 0 else 0
        if actual_risk_pct > (risk_pct * 1.5):
            log.warning(f"Calculated lot ({raw_lot:.4f}) is below broker min ({min_lot}). "
                        f"Min lot would risk {actual_risk_pct*100:.1f}% (> target {risk_pct*100:.1f}%). Trade skipped.")
            return 0.0
        lot = min_lot

    lot = min(max_lot, lot)
    return round(lot, 2)


class LiveV7Trader:
    def __init__(self, ticker, risk_pct=0.02):
        self.ticker = ticker
        self.risk_pct = risk_pct
        self.account_balance = 0.0  # Will be populated by connect()
        self.pip_size = 0.01 if ticker.endswith('JPY') else (0.10 if ticker in ['XAUUSD', 'GOLD'] else 0.0001)
        self.pip_value_per_lot = 6.67 if ticker.endswith('JPY') else 10.0

        log.info(f"Loading production models for {ticker}...")
        self.cfg, self.primary, self.meta = load_config(ticker)
        self.tp_mult = self.cfg['tp_mult']
        self.sl_mult = self.cfg['sl_mult']
        self.horizon = self.cfg['horizon']
        self.meta_thresh = self.cfg['meta_threshold']
        self.regime_filter = self.cfg['regime_filter']
        self.feature_cols = self.cfg['feature_cols']

        log.info(f"Config: {self.tp_mult}:1 RR, H={self.horizon}, "
                 f"regimes={self.regime_filter}, meta>={self.meta_thresh}")
        log.info(f"Risk: {risk_pct*100}%/trade (Balance will be fetched from MT5)")

        self.mt5 = None
        self.active_trade = None
        self.broker_utc_offset = timedelta(hours=0)
        
        # Persistent state tracking across restarts
        self.load_state()

    def load_state(self):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.state_file = os.path.join(BASE_DIR, f'live_state_{self.ticker.lower()}.json')
        self.circuit_breaker_lockout_until = None
        self.consecutive_losses = 0
        self.cooldown_until = None
        self.last_processed_candle_time = None

        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    st = json.load(f)
                if st.get('circuit_breaker_lockout_until'):
                    self.circuit_breaker_lockout_until = datetime.fromisoformat(st['circuit_breaker_lockout_until'])
                if st.get('cooldown_until'):
                    self.cooldown_until = datetime.fromisoformat(st['cooldown_until'])
                self.consecutive_losses = st.get('consecutive_losses', 0)
                if st.get('last_processed_candle_time'):
                    self.last_processed_candle_time = st['last_processed_candle_time']
                
                # Check if circuit breaker is still active from prior run
                now = datetime.now(timezone.utc)
                if self.circuit_breaker_lockout_until and now < self.circuit_breaker_lockout_until:
                    rem_sec = (self.circuit_breaker_lockout_until - now).total_seconds()
                    log.warning(f"RESTORED STATE: Macro VIX Circuit Breaker lockout active for another "
                                f"{int(rem_sec//3600)}h{int((rem_sec%3600)//60)}m (until {self.circuit_breaker_lockout_until.strftime('%Y-%m-%d %H:%M UTC')}).")
                if self.cooldown_until and now < self.cooldown_until:
                    rem_sec = (self.cooldown_until - now).total_seconds()
                    log.warning(f"RESTORED STATE: Loss Cooldown active for another "
                                f"{int(rem_sec//3600)}h{int((rem_sec%3600)//60)}m (until {self.cooldown_until.strftime('%Y-%m-%d %H:%M UTC')}).")
            except Exception as e:
                log.warning(f"Failed to load state file: {e}")

    def save_state(self):
        try:
            st = {
                'ticker': self.ticker,
                'circuit_breaker_lockout_until': self.circuit_breaker_lockout_until.isoformat() if self.circuit_breaker_lockout_until else None,
                'consecutive_losses': self.consecutive_losses,
                'cooldown_until': self.cooldown_until.isoformat() if self.cooldown_until else None,
                'last_processed_candle_time': str(self.last_processed_candle_time) if self.last_processed_candle_time else None,
                'active_trade': self.active_trade,
                'last_saved_utc': datetime.now(timezone.utc).isoformat()
            }
            with open(self.state_file, 'w') as f:
                json.dump(st, f, indent=2)
        except Exception as e:
            log.warning(f"Failed to save state: {e}")

    def get_pip_value(self):
        """Calculate dynamic real-time pip value per standard lot from live market price / MT5."""
        if self.mt5:
            info = self.mt5.symbol_info(self.ticker)
            if info and info.trade_tick_value > 0 and info.trade_tick_size > 0:
                return (info.trade_tick_value / info.trade_tick_size) * self.pip_size
        if self.ticker.endswith('JPY'):
            tick = self.mt5.symbol_info_tick(self.ticker) if self.mt5 else None
            price = tick.bid if (tick and tick.bid > 0) else 155.0
            return (100000.0 * 0.01) / price
        return 10.0

    def check_portfolio_currency_matrix_exposure(self, proposed_direction, proposed_risk=None):
        """
        Safeguard: Dual Currency Matrix Exposure and Portfolio Heat Cap.
        Caps net USD positions at <= 2 and net JPY positions at <= 2.
        Also caps total portfolio heat at <= 4.5% of account balance.
        """
        if not self.mt5:
            return True
        positions = self.mt5.positions_get()
        if not positions:
            return True
            
        usd_exposure = 0
        jpy_exposure = 0
        total_risk_exposure = 0.0

        for pos in positions:
            if pos.magic == 234000:
                sym = pos.symbol
                is_buy = (pos.type == self.mt5.ORDER_TYPE_BUY)
                d = 1 if is_buy else -1

                # USD delta
                if sym == 'USDJPY':
                    usd_exposure += 1 if d == 1 else -1
                elif sym in ['AUDUSD', 'XAUUSD', 'GOLD', 'EURUSD', 'GBPUSD']:
                    usd_exposure += -1 if d == 1 else 1

                # JPY delta
                if sym in ['USDJPY', 'GBPJPY']:
                    jpy_exposure += -1 if d == 1 else 1

                # Calculate position dollar risk for portfolio heat limit
                if pos.sl > 0:
                    sl_dist = abs(pos.price_open - pos.sl)
                    sym_info = self.mt5.symbol_info(sym)
                    if sym_info and sym_info.trade_tick_value > 0 and sym_info.trade_tick_size > 0:
                        pos_risk = (sl_dist * (sym_info.trade_tick_value / sym_info.trade_tick_size)) * pos.volume
                    else:
                        pos_risk = pos.volume * 10.0 * 40.0
                    total_risk_exposure += pos_risk

        # Proposed trade deltas
        prop_usd = (1 if proposed_direction == 1 else -1) if self.ticker == 'USDJPY' else (-1 if proposed_direction == 1 else 1 if self.ticker in ['AUDUSD', 'XAUUSD', 'GOLD'] else 0)
        prop_jpy = (-1 if proposed_direction == 1 else 1) if self.ticker in ['USDJPY', 'GBPJPY'] else 0

        new_usd = usd_exposure + prop_usd
        new_jpy = jpy_exposure + prop_jpy

        MAX_USD = 2
        MAX_JPY = 2
        if abs(new_usd) > MAX_USD:
            log.warning(f"Portfolio USD Exposure Cap reached (Current={usd_exposure}, Proposed={prop_usd:+d}, Max={MAX_USD}). "
                        f"Skipping trade to prevent correlated USD drawdown.")
            return False

        if abs(new_jpy) > MAX_JPY:
            log.warning(f"Portfolio JPY Exposure Cap reached (Current={jpy_exposure}, Proposed={prop_jpy:+d}, Max={MAX_JPY}). "
                        f"Skipping trade to prevent correlated JPY drawdown.")
            return False

        # Check total portfolio heat (max 4.5% of account balance)
        risk_val = proposed_risk if proposed_risk is not None else (self.account_balance * self.risk_pct)
        if self.account_balance > 0 and (total_risk_exposure + risk_val) / self.account_balance > 0.045:
            heat_pct = (total_risk_exposure + risk_val) / self.account_balance * 100
            log.warning(f"Portfolio Heat Cap exceeded (Current heat=${total_risk_exposure:.2f}, Proposed=${risk_val:.2f}, "
                        f"Total heat={heat_pct:.1f}% > 4.5% limit). Skipping trade to preserve capital.")
            return False

        return True

    # Backward-compatible alias
    check_portfolio_usd_exposure = check_portfolio_currency_matrix_exposure

    def connect(self):
        import MetaTrader5 as mt5
        self.mt5 = mt5
        if not mt5.initialize():
            log.error("MT5 initialization failed. Is MetaTrader5 terminal running?")
            raise RuntimeError("MT5 init failed")

        info = mt5.symbol_info(self.ticker)
        if info is None:
            log.error(f"Symbol {self.ticker} not found in MT5")
            raise RuntimeError(f"Symbol {self.ticker} not found")

        if not info.visible:
            mt5.symbol_select(self.ticker, True)

        acc = mt5.account_info()
        if acc:
            self.account_balance = acc.balance
            log.info(f"Connected. Account: ${acc.balance:.2f} | "
                     f"Symbol: {self.ticker} | Digits: {info.digits}")

        # Auto-detect broker server time offset relative to true UTC
        tick = mt5.symbol_info_tick(self.ticker)
        if tick and tick.time > 0:
            server_time = datetime.fromtimestamp(tick.time, tz=timezone.utc)
            now_utc = datetime.now(timezone.utc)
            raw_diff = (server_time - now_utc).total_seconds() / 3600
            if abs(raw_diff) <= 14:
                offset_hours = round(raw_diff)
                self.broker_utc_offset = timedelta(hours=offset_hours)
                log.info(f"Broker server time offset detected: {offset_hours:+d}h relative to UTC")
            else:
                self.broker_utc_offset = timedelta(hours=0)
                log.info("Weekend/inactive tick detected. Using default broker offset (+0h).")

        # Reconcile existing open positions
        positions = mt5.positions_get(symbol=self.ticker)
        if positions:
            for pos in positions:
                if pos.magic == 234000:
                    direction = 1 if pos.type == mt5.ORDER_TYPE_BUY else -1
                    open_time = datetime.fromtimestamp(pos.time, tz=timezone.utc) - self.broker_utc_offset
                    now = datetime.now(timezone.utc)
                    candles_held = int((now - open_time).total_seconds() / 3600 // 4)
                    
                    self.active_trade = {
                        'ticket': pos.ticket,
                        'candles_held': candles_held,
                        'direction': direction,
                        'entry_price': pos.price_open,
                        'sl_price': pos.sl,
                        'tp_price': pos.tp,
                        'lot': pos.volume,
                    }
                    log.info(f"Recovered existing active trade: Ticket={pos.ticket}, "
                             f"Dir={'LONG' if direction==1 else 'SHORT'}, "
                             f"Volume={pos.volume}, Held approx {candles_held} bars.")
                    break

    def refresh_alternative_data_if_needed(self):
        alt_file = "alternative_data.csv"
        # Auto-refresh on background thread if missing or older than 12 hours
        if not os.path.exists(alt_file) or (time.time() - os.path.getmtime(alt_file)) > 43200:
            def _worker():
                log.info("Background thread: Refreshing alternative macro data (VIX, US10Y, DXY, COT)...")
                try:
                    subprocess.run([sys.executable, "fetch_alternative_data.py"], check=False)
                    log.info("Background thread: Macro data refresh complete.")
                except Exception as e:
                    log.warning(f"Background thread: Macro refresh failed: {e}")
            threading.Thread(target=_worker, daemon=True).start()

    def fetch_candles(self, n=300):
        # Auto-reconnect if terminal disconnected
        if not self.mt5 or not self.mt5.terminal_info():
            log.warning("MT5 terminal connection lost. Reconnecting...")
            try:
                self.connect()
            except Exception as e:
                log.error(f"Reconnection failed: {e}")
                return pd.DataFrame()

        rates = self.mt5.copy_rates_from_pos(self.ticker, self.mt5.TIMEFRAME_H4, 1, n)
        if rates is None or len(rates) == 0:
            return pd.DataFrame()
        df = pd.DataFrame(rates)
        if df.empty:
            return df
        # Convert to true UTC by adjusting for broker server offset
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True) - self.broker_utc_offset
        df = df.rename(columns={
            'time': 'Time', 'open': 'Open', 'high': 'High',
            'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume',
        })
        return df[['Time', 'Open', 'High', 'Low', 'Close', 'Volume']]

    def compute_features(self, df):
        try:
            feat_df = process_features_v7(df, ticker=self.ticker, is_training=False)
            if feat_df.empty:
                return None
            return feat_df
        except Exception as e:
            log.error(f"Error computing features: {e}", exc_info=True)
            return None

    def get_signal(self, feat_row):
        X = feat_row[self.feature_cols].values.reshape(1, -1)
        X_df = pd.DataFrame(X, columns=self.feature_cols)

        # CatBoost MultiClass predict returns shape (1,1) array — flatten
        # to match backtest_v7.py exactly: .flatten().astype(float)[0]
        direction = int(self.primary.predict(X_df).flatten().astype(float)[0])
        if direction == 0:
            return 0, 0.0, None

        X_meta = X_df.copy()
        X_meta['Primary_Pred'] = float(direction)
        meta_prob = float(self.meta.predict_proba(X_meta)[0, 1])

        regime = feat_row['Regime'] if 'Regime' in feat_row.index else 'mixed'
        return direction, meta_prob, regime

    def calc_sl_tp(self, entry_price, atr, direction, tp_mult=None, sl_mult=None):
        tp_m = tp_mult if tp_mult is not None else self.tp_mult
        sl_m = sl_mult if sl_mult is not None else self.sl_mult
        if direction == 1:
            sl = entry_price - (atr * sl_m)
            tp = entry_price + (atr * tp_m)
        else:
            sl = entry_price + (atr * sl_m)
            tp = entry_price - (atr * tp_m)
        return sl, tp

    def _get_filling_mode(self):
        """Detect the broker's supported filling mode for this symbol."""
        mt5 = self.mt5
        info = mt5.symbol_info(self.ticker)
        # Filling mode flags from symbol_info.filling_mode:
        #   bit 0 (1) = ORDER_FILLING_FOK supported
        #   bit 1 (2) = ORDER_FILLING_IOC supported
        #   ORDER_FILLING_RETURN is always available for non-market orders
        fm = info.filling_mode
        if fm & 1:
            return mt5.ORDER_FILLING_FOK
        elif fm & 2:
            return mt5.ORDER_FILLING_IOC
        else:
            return mt5.ORDER_FILLING_RETURN

    def execute_trade(self, direction, sl_price, tp_price, lot, comment=None):
        mt5 = self.mt5
        tick = mt5.symbol_info_tick(self.ticker)

        if direction == 1:
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
        else:
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid

        digits = mt5.symbol_info(self.ticker).digits
        filling = self._get_filling_mode()
        dev_pts = 20 if self.ticker == 'AUDUSD' else 50
        trade_comment = comment if comment else f'V7 {int(self.tp_mult*10)}x{int(self.sl_mult*10)}'
        request = {
            'action': mt5.TRADE_ACTION_DEAL,
            'symbol': self.ticker,
            'volume': lot,
            'type': order_type,
            'price': price,
            'sl': round(sl_price, digits),
            'tp': round(tp_price, digits),
            'deviation': dev_pts,
            'magic': 234000,
            'comment': trade_comment,
            'type_time': mt5.ORDER_TIME_GTC,
            'type_filling': filling,
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            log.error(f"Order failed: {result.comment} (retcode={result.retcode})")
            # If filling mode still rejected, try RETURN as fallback
            if result.retcode == 10030 and filling != mt5.ORDER_FILLING_RETURN:
                log.info("Retrying with ORDER_FILLING_RETURN...")
                request['type_filling'] = mt5.ORDER_FILLING_RETURN
                result = mt5.order_send(request)
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    log.error(f"Retry failed: {result.comment} (retcode={result.retcode})")
                    return None
            else:
                return None

        log.info(f"OPENED {('LONG' if direction==1 else 'SHORT')} "
                 f"{lot} lots @ {price:.{digits}f} | "
                 f"SL={sl_price:.{digits}f} TP={tp_price:.{digits}f} | "
                 f"Ticket={result.order}")
        return result.order

    def close_position(self, ticket, reason='horizon'):
        mt5 = self.mt5
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            log.info(f"Position {ticket} already closed (SL/TP hit)")
            return True

        pos = positions[0]
        close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        tick = mt5.symbol_info_tick(self.ticker)
        price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask

        request = {
            'action': mt5.TRADE_ACTION_DEAL,
            'symbol': self.ticker,
            'volume': pos.volume,
            'type': close_type,
            'position': pos.ticket,
            'price': price,
            'deviation': 50,
            'magic': 234000,
            'comment': f'V7 {reason}',
            'type_time': mt5.ORDER_TIME_GTC,
            'type_filling': self._get_filling_mode(),
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            # Retry with RETURN filling if unsupported
            if result.retcode == 10030:
                request['type_filling'] = mt5.ORDER_FILLING_RETURN
                result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                log.error(f"Close failed: {result.comment} (retcode={result.retcode})")
                return False

        log.info(f"CLOSED ticket {ticket} ({reason}) @ {price:.5f}")
        return True

    def _record_trade_closure(self, ticket, reason='SL/TP'):
        """
        Accurately query deal history for ONLY this trade ticket to record profit/loss,
        update consecutive loss streak, and trigger cooldowns.
        Prevents querying the whole account's deal history.
        """
        mt5 = self.mt5
        total_profit = None

        try:
            # 1. First attempt: query specifically by position ticket
            deals = mt5.history_deals_get(position=ticket)
            if deals:
                matching_deals = [d for d in deals if getattr(d, 'symbol', None) == self.ticker]
                if matching_deals:
                    total_profit = sum(d.profit + d.swap + d.fee for d in matching_deals)

            # 2. Second attempt: query by date window and strictly filter by position_id/order/symbol
            if total_profit is None:
                from_time = datetime.now(timezone.utc) - timedelta(days=7)
                to_time = datetime.now(timezone.utc) + timedelta(minutes=5)
                recent_deals = mt5.history_deals_get(from_time, to_time)
                if recent_deals:
                    matching_deals = [
                        d for d in recent_deals
                        if getattr(d, 'symbol', None) == self.ticker and (
                            getattr(d, 'position_id', None) == ticket or
                            getattr(d, 'order', None) == ticket
                        )
                    ]
                    if matching_deals:
                        total_profit = sum(d.profit + d.swap + d.fee for d in matching_deals)

            # 3. Third attempt: match the most recent exit deal on this specific symbol
            if total_profit is None:
                from_time = datetime.now(timezone.utc) - timedelta(hours=12)
                to_time = datetime.now(timezone.utc) + timedelta(minutes=5)
                recent_deals = mt5.history_deals_get(from_time, to_time)
                if recent_deals:
                    sym_exit_deals = [
                        d for d in recent_deals
                        if getattr(d, 'symbol', None) == self.ticker and getattr(d, 'entry', None) in [1, 2]
                    ]
                    if sym_exit_deals:
                        last_deal = sym_exit_deals[-1]
                        total_profit = last_deal.profit + last_deal.swap + last_deal.fee
                        log.info(f"Matched recent exit deal #{last_deal.ticket} on {self.ticker} for position #{ticket}")

            if total_profit is not None:
                if total_profit > 0:
                    log.info(f"Trade {ticket} closed ({reason}) in PROFIT (+${total_profit:.2f}). Resetting loss streak.")
                    self.consecutive_losses = 0
                    self.cooldown_until = None
                else:
                    self.consecutive_losses += 1
                    log.warning(f"Trade {ticket} closed ({reason}) at a LOSS (-${abs(total_profit):.2f}). "
                                f"Consecutive losses: {self.consecutive_losses}")
                    if self.consecutive_losses >= 2:
                        self.cooldown_until = datetime.now(timezone.utc) + timedelta(hours=24)
                        log.warning(f"2 consecutive losses reached on {self.ticker}. 24-hour pair cooldown ACTIVE until "
                                    f"{self.cooldown_until.strftime('%Y-%m-%d %H:%M UTC')}.")
            else:
                log.warning(f"Could not isolate MT5 deal record for ticket #{ticket} on {self.ticker}. "
                            f"Assuming trade closed ({reason}); leaving loss streak unchanged.")

        except Exception as e:
            log.warning(f"Error inspecting MT5 deal history for ticket #{ticket}: {e}")

        self.active_trade = None
        self.save_state()

    def manage_active_trade(self, feat_row=None):
        if not self.active_trade:
            return

        trade = self.active_trade
        trade['candles_held'] += 1

        mt5 = self.mt5
        positions = mt5.positions_get(ticket=trade['ticket'])
        if not positions:
            log.info(f"Trade {trade['ticket']} closed by SL/TP after "
                     f"{trade['candles_held']} bars")
            self._record_trade_closure(trade['ticket'], reason=f"SL/TP after {trade['candles_held']} bars")
            return

        # Check Adverse Signal Exit (For USDJPY and GBPJPY)
        if feat_row is not None and self.cfg.get('use_adverse_exit', False):
            direction, meta_conf, _ = self.get_signal(feat_row)
            if direction == -trade['direction'] and meta_conf >= 0.52:
                log.warning(f"ADVERSE MODEL FLIP DETECTED (New Dir={direction}, Active Dir={trade['direction']}, "
                            f"Meta Conf={meta_conf:.3f} >= 0.52). Early cutting position to prevent adverse loss...")
                self.close_position(trade['ticket'], 'adverse_signal_exit')
                self._record_trade_closure(trade['ticket'], reason="adverse_signal_exit")
                return

        trade_horizon = trade.get('horizon', self.horizon)
        if trade['candles_held'] >= trade_horizon:
            # Rollover spread protection: check current spread before market close
            tick = mt5.symbol_info_tick(self.ticker)
            current_spread = (tick.ask - tick.bid) / self.pip_size if tick else 1.0
            now_hour = datetime.now(timezone.utc).hour
            max_spread = 5.0 if self.ticker in ['XAUUSD', 'GOLD'] else (2.0 if self.ticker == 'AUDUSD' else 3.5)

            # If during daily rollover (21:00-22:00 UTC) with wide spread, defer market exit briefly
            if (21 <= now_hour <= 22 or current_spread > max_spread) and trade['candles_held'] == trade_horizon:
                log.warning(f"Trade {trade['ticket']} reached horizon, but elevated spread ({current_spread:.1f} pips > max {max_spread:.1f}) "
                            f"or rollover hour detected. Deferring market close until spread normalizes.")
                return

            log.info(f"Trade {trade['ticket']} reached horizon "
                     f"({trade_horizon} bars). Closing...")
            self.close_position(trade['ticket'], 'horizon_exit')
            self._record_trade_closure(trade['ticket'], reason=f"horizon_exit after {trade_horizon} bars")
        else:
            log.info(f"Trade {trade['ticket']} holding "
                     f"({trade['candles_held']}/{trade_horizon} bars)")
            self.save_state()

    def run(self):
        self.connect()
        log.info(f"Starting V7 live daemon for {self.ticker}. "
                 f"Waiting for H4 candle closes...")

        last_processed_time = None

        while True:
            try:
                now = datetime.now(timezone.utc)
                
                # Weekend idle check (Forex market closed from Friday 21:00 UTC to Sunday 21:00 UTC)
                if now.weekday() == 5 or (now.weekday() == 6 and now.hour < 21):
                    # Calculate seconds until Sunday 21:00 UTC
                    days_ahead = 6 - now.weekday()
                    sunday_open = now.replace(hour=21, minute=0, second=2, microsecond=0) + timedelta(days=days_ahead)
                    sleep_secs = max(60, (sunday_open - now).total_seconds())
                    log.info(f"Forex market is closed for the weekend. Sleeping {int(sleep_secs)//3600}h "
                             f"until Sunday 21:00 UTC...")
                    time.sleep(sleep_secs)
                    continue

                # If we are within the first 2 minutes of a new H4 block, process immediately
                if now.hour % 4 == 0 and now.minute < 2:
                    seconds_to_next = 0
                else:
                    next_hour = ((now.hour // 4) + 1) * 4
                    target = now.replace(minute=0, second=2, microsecond=0) + timedelta(hours=(next_hour - now.hour))
                    seconds_to_next = (target - now).total_seconds()

                if seconds_to_next > 65:
                    sleep_first = seconds_to_next - 60
                    log.info(f"Next H4 close in {int(seconds_to_next)//60}m{int(seconds_to_next)%60}s. "
                             f"Sleeping {int(sleep_first)//60}m{int(sleep_first)%60}s until macro pre-fetch window...")
                    time.sleep(sleep_first)
                    self.refresh_alternative_data_if_needed()
                    # Sleep remaining seconds until candle close
                    now = datetime.now(timezone.utc)
                    rem = max(0, (target - now).total_seconds())
                    if rem > 0:
                        time.sleep(rem)
                elif seconds_to_next > 0:
                    time.sleep(seconds_to_next)

                log.info("Fetching H4 candles (rolling 300 buffer)...")
                df = self.fetch_candles(300)
                if len(df) < 200:
                    log.error(f"Only {len(df)} candles, need 200+")
                    time.sleep(60)
                    continue

                latest_time = df.iloc[-1]['Time']
                if last_processed_time == latest_time:
                    log.info("Candle not updated yet (or already processed), sleeping 60s...")
                    time.sleep(60)
                    continue
                
                last_processed_time = latest_time

                log.info("Computing V7 features...")
                feat_df = self.compute_features(df)
                if feat_df is None or feat_df.empty:
                    log.error("Feature computation returned empty")
                    continue

                latest = feat_df.iloc[-1]
                self.manage_active_trade(latest)

                if self.active_trade:
                    log.info("Position active, skipping new entry.")
                    continue

                latest = feat_df.iloc[-1]
                atr = float(latest['ATR']) if 'ATR' in feat_df.columns else 0.0
                if atr <= 0 or np.isnan(atr):
                    log.warning(f"ATR is zero/NaN ({atr}), skipping")
                    continue

                direction, meta_conf, regime = self.get_signal(latest)
                close_price = float(latest['Close']) if 'Close' in feat_df.columns else 0.0

                log.info(f"Signal: dir={direction} meta={meta_conf:.3f} "
                         f"regime={regime} ATR={atr:.5f} close={close_price:.5f}")

                if direction == 0:
                    log.info("No trade (primary model neutral)")
                    continue

                now_utc = datetime.now(timezone.utc)
                candle_hour = latest_time.hour if hasattr(latest_time, 'hour') else now_utc.hour

                trade_tp_mult = self.tp_mult
                trade_sl_mult = self.sl_mult
                trade_horizon = self.horizon

                if 'modes' in self.cfg and self.cfg['modes']:
                    matched_mode = None
                    for mode in self.cfg['modes']:
                        if mode.get('allowed_hours') and candle_hour not in mode['allowed_hours']:
                            continue
                        if mode.get('regime_filter') and regime not in mode['regime_filter']:
                            continue
                        if meta_conf < mode['meta_threshold']:
                            continue
                        matched_mode = mode
                        break
                    if matched_mode is None:
                        log.info(f"Candle hour={candle_hour} or regime='{regime}' or meta={meta_conf:.3f} "
                                 f"did not match any active execution modes. Skipping.")
                        continue
                    trade_tp_mult = matched_mode['tp_mult']
                    trade_sl_mult = matched_mode['sl_mult']
                    trade_horizon = matched_mode['horizon']
                    log.info(f"Matched Mode: TP={trade_tp_mult} SL={trade_sl_mult} H={trade_horizon}")
                else:
                    if meta_conf < self.meta_thresh:
                        log.info(f"Meta confidence {meta_conf:.3f} < "
                                 f"{self.meta_thresh}, skipping")
                        continue

                    if self.regime_filter and regime not in self.regime_filter:
                        log.info(f"Regime '{regime}' not in filter "
                                 f"{self.regime_filter}, skipping")
                        continue

                    if self.cfg.get('allowed_hours'):
                        if candle_hour not in self.cfg['allowed_hours']:
                            log.info(f"Candle hour {candle_hour} not in allowed_hours "
                                     f"{self.cfg['allowed_hours']}, skipping")
                            continue

                # Persistent Circuit Breaker & Cooldown Lockout Check
                if self.circuit_breaker_lockout_until and now_utc < self.circuit_breaker_lockout_until:
                    rem_sec = (self.circuit_breaker_lockout_until - now_utc).total_seconds()
                    log.warning(f"Macro VIX Circuit Breaker ACTIVE ({int(rem_sec//3600)}h{int((rem_sec%3600)//60)}m remaining until {self.circuit_breaker_lockout_until.strftime('%Y-%m-%d %H:%M UTC')}). "
                                f"Skipping new trade entries.")
                    continue

                if self.cooldown_until and now_utc < self.cooldown_until:
                    rem_sec = (self.cooldown_until - now_utc).total_seconds()
                    log.warning(f"Consecutive Loss Cooldown ACTIVE ({int(rem_sec//3600)}h{int((rem_sec%3600)//60)}m remaining until {self.cooldown_until.strftime('%Y-%m-%d %H:%M UTC')}). "
                                f"Skipping new trade entries.")
                    continue

                # Macro VIX Circuit Breaker: lockout during global volatility spikes
                if 'Macro_VIX_Lockout' in feat_df.columns and float(latest['Macro_VIX_Lockout']) == 1:
                    self.circuit_breaker_lockout_until = now_utc + timedelta(hours=24)
                    self.save_state()
                    log.warning(f"Macro VIX Circuit Breaker TRIPPED (VIX Spike / Global Shock detected). "
                                f"Locking out new entries for 24h (until {self.circuit_breaker_lockout_until.strftime('%Y-%m-%d %H:%M UTC')}). State saved to disk.")
                    continue

                # Fractional conviction scaling: 1.5% to 2.5% based on meta conviction
                conviction_risk = 0.015 + max(0.0, min(0.010, (meta_conf - 0.50) * 0.10))

                # Query live account balance for dynamic auto-compounding
                acc = self.mt5.account_info()
                if acc and acc.balance > 0:
                    self.account_balance = acc.balance

                proposed_risk_usd = self.account_balance * conviction_risk

                # Portfolio-level risk check: ensure we do not over-expose USD/JPY or exceed 4.5% total heat
                if not self.check_portfolio_currency_matrix_exposure(direction, proposed_risk=proposed_risk_usd):
                    continue

                # Query real-time pip value & broker constraints
                pip_val = self.get_pip_value()
                info = self.mt5.symbol_info(self.ticker)
                sl_price, tp_price = self.calc_sl_tp(
                    close_price, atr, direction, tp_mult=trade_tp_mult, sl_mult=trade_sl_mult)
                sl_pips = abs(close_price - sl_price) / self.pip_size
                lot = calc_lot_size(
                    self.account_balance, conviction_risk,
                    sl_pips, pip_value_per_lot=pip_val, info=info, pip_size=self.pip_size)

                if lot <= 0.0:
                    log.info("Lot sizing calculated to 0.0 (minimum risk guard). Skipping trade.")
                    continue

                log.info(f"Entry signal: {('LONG' if direction==1 else 'SHORT')} "
                         f"| SL={sl_pips:.1f} pips | "
                         f"TP={abs(tp_price-close_price)/self.pip_size:.1f} pips | "
                         f"Lot={lot} | PipVal=${pip_val:.2f} | Risk={conviction_risk*100:.1f}% (${proposed_risk_usd:.2f})")

                comment_str = f'V7 {int(trade_tp_mult*10)}x{int(trade_sl_mult*10)}'
                ticket = self.execute_trade(direction, sl_price, tp_price, lot, comment=comment_str)
                if ticket:
                    self.active_trade = {
                        'ticket': ticket,
                        'candles_held': 0,
                        'direction': direction,
                        'entry_price': close_price,
                        'sl_price': sl_price,
                        'tp_price': tp_price,
                        'lot': lot,
                        'horizon': trade_horizon,
                    }
                    self.save_state()

            except KeyboardInterrupt:
                log.info("Shutting down by user request...")
                break
            except Exception as e:
                log.error(f"Main loop error: {e}", exc_info=True)
                time.sleep(60)

        if self.mt5:
            self.mt5.shutdown()
            log.info("MT5 connection closed.")


def main():
    parser = argparse.ArgumentParser(
        description='V7 Live Trading Daemon',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python trade_live.py USDJPY --risk 0.02
  python trade_live.py GBPJPY --risk 0.02
  python trade_live.py AUDUSD --risk 0.02
        """)
    parser.add_argument('ticker', type=str,
                        choices=['USDJPY', 'GBPJPY', 'AUDUSD', 'XAUUSD', 'GOLD', 'EURUSD', 'GBPUSD'],
                        help='Trading symbol')
    parser.add_argument('--risk', type=float, default=0.02,
                        help='Risk per trade as fraction (0.02 = 2%%)')
    args = parser.parse_args()

    trader = LiveV7Trader(args.ticker, args.risk)
    trader.run()


if __name__ == '__main__':
    main()
