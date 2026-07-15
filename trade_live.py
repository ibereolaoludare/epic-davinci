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
  python trade_live.py USDJPY --account 5000 --risk 0.02
  python trade_live.py EURUSD --account 1000 --risk 0.02
  python trade_live.py GBPUSD --account 2500 --risk 0.02

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


def calc_lot_size(account_balance, risk_pct, sl_pips, pip_value_per_lot):
    risk_amount = account_balance * risk_pct
    lot = risk_amount / (sl_pips * pip_value_per_lot)
    lot = max(0.01, round(lot, 2))
    return lot


class LiveV7Trader:
    def __init__(self, ticker, account_balance, risk_pct=0.02):
        self.ticker = ticker
        self.risk_pct = risk_pct
        self.account_balance = account_balance
        self.pip_size = 0.01 if ticker.endswith('JPY') else 0.0001
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
        log.info(f"Account: ${account_balance} | Risk: {risk_pct*100}%/trade")

        self.mt5 = None
        self.active_trade = None

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

    def fetch_candles(self, n=500):
        rates = self.mt5.copy_rates_from_pos(self.ticker, self.mt5.TIMEFRAME_H4, 1, n)
        if rates is None or len(rates) == 0:
            return pd.DataFrame()
        df = pd.DataFrame(rates)
        if df.empty:
            return df
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df = df.rename(columns={
            'time': 'Time', 'open': 'Open', 'high': 'High',
            'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume',
        })
        return df[['Time', 'Open', 'High', 'Low', 'Close', 'Volume']]

    def compute_features(self, df):
        feat_df = process_features_v7(df)
        if feat_df.empty:
            return None
        return feat_df

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

    def calc_sl_tp(self, entry_price, atr, direction):
        if direction == 1:
            sl = entry_price - (atr * self.sl_mult)
            tp = entry_price + (atr * self.tp_mult)
        else:
            sl = entry_price + (atr * self.sl_mult)
            tp = entry_price - (atr * self.tp_mult)
        return sl, tp

    def execute_trade(self, direction, sl_price, tp_price, lot):
        mt5 = self.mt5
        tick = mt5.symbol_info_tick(self.ticker)

        if direction == 1:
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
        else:
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid

        digits = mt5.symbol_info(self.ticker).digits
        request = {
            'action': mt5.TRADE_ACTION_DEAL,
            'symbol': self.ticker,
            'volume': lot,
            'type': order_type,
            'price': price,
            'sl': round(sl_price, digits),
            'tp': round(tp_price, digits),
            'deviation': 20,
            'magic': 234000,
            'comment': f'V7 {int(self.tp_mult*10)}x{int(self.sl_mult*10)}',
            'type_time': mt5.ORDER_TIME_GTC,
            'type_filling': mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            log.error(f"Order failed: {result.comment} (retcode={result.retcode})")
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
            'deviation': 20,
            'magic': 234000,
            'comment': f'V7 {reason}',
            'type_time': mt5.ORDER_TIME_GTC,
            'type_filling': mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            log.error(f"Close failed: {result.comment}")
            return False

        log.info(f"CLOSED ticket {ticket} ({reason}) @ {price:.5f}")
        return True

    def manage_active_trade(self):
        if not self.active_trade:
            return

        trade = self.active_trade
        trade['candles_held'] += 1

        mt5 = self.mt5
        positions = mt5.positions_get(ticket=trade['ticket'])
        if not positions:
            profit_pips = trade.get('profit_pips', 0)
            log.info(f"Trade {trade['ticket']} closed by SL/TP after "
                     f"{trade['candles_held']} bars")
            self.active_trade = None
            return

        if trade['candles_held'] >= self.horizon:
            log.info(f"Trade {trade['ticket']} reached horizon "
                     f"({self.horizon} bars). Closing...")
            self.close_position(trade['ticket'], 'horizon_exit')
            self.active_trade = None
        else:
            log.info(f"Trade {trade['ticket']} holding "
                     f"({trade['candles_held']}/{self.horizon} bars)")

    def run(self):
        self.connect()
        log.info(f"Starting V7 live daemon for {self.ticker}. "
                 f"Waiting for H4 candle closes...")

        while True:
            try:
                now = datetime.now(timezone.utc)
                hours_to_next = 4 - (now.hour % 4)
                if hours_to_next == 4 and now.minute == 0:
                    hours_to_next = 0

                minutes_to_next = (hours_to_next * 60) - now.minute
                seconds_to_next = (minutes_to_next * 60) - now.second + 2

                if seconds_to_next > 0:
                    log.info(f"Next H4 close in {seconds_to_next//60}m{seconds_to_next%60}s. "
                             f"Sleeping...")
                    time.sleep(seconds_to_next)

                self.manage_active_trade()

                if self.active_trade:
                    log.info("Position active, skipping new entry.")
                    continue

                log.info("Fetching H4 candles...")
                df = self.fetch_candles(500)
                if len(df) < 200:
                    log.error(f"Only {len(df)} candles, need 200+")
                    time.sleep(60)
                    continue

                log.info("Computing V7 features...")
                feat_df = self.compute_features(df)
                if feat_df is None or feat_df.empty:
                    log.error("Feature computation returned empty")
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

                if meta_conf < self.meta_thresh:
                    log.info(f"Meta confidence {meta_conf:.3f} < "
                             f"{self.meta_thresh}, skipping")
                    continue

                if regime not in self.regime_filter:
                    log.info(f"Regime '{regime}' not in filter "
                             f"{self.regime_filter}, skipping")
                    continue

                sl_price, tp_price = self.calc_sl_tp(
                    close_price, atr, direction)
                sl_pips = abs(close_price - sl_price) / self.pip_size
                lot = calc_lot_size(
                    self.account_balance, self.risk_pct,
                    sl_pips, self.pip_value_per_lot)

                log.info(f"Entry signal: {('LONG' if direction==1 else 'SHORT')} "
                         f"| SL={sl_pips:.1f} pips | "
                         f"TP={abs(tp_price-close_price)/self.pip_size:.1f} pips | "
                         f"Lot={lot} | Risk=${self.account_balance*self.risk_pct:.2f}")

                ticket = self.execute_trade(direction, sl_price, tp_price, lot)
                if ticket:
                    self.active_trade = {
                        'ticket': ticket,
                        'candles_held': 0,
                        'direction': direction,
                        'entry_price': close_price,
                        'sl_price': sl_price,
                        'tp_price': tp_price,
                        'lot': lot,
                    }

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
  python trade_live.py USDJPY --account 5000
  python trade_live.py EURUSD --account 1000 --risk 0.02
  python trade_live.py GBPUSD --account 2500 --risk 0.01

Prerequisites:
  1. MetaTrader5 terminal running and logged in
  2. Production models trained:
       python train_v7_prod.py USDJPY
       python train_v7_prod.py EURUSD
       python train_v7_prod.py GBPUSD
  3. Alternative data fetched:
       python fetch_alternative_data.py
  4. Central bank rates saved:
       python download_data.py
        """)
    parser.add_argument('ticker', type=str,
                        choices=['USDJPY', 'EURUSD', 'GBPUSD'],
                        help='Trading symbol')
    parser.add_argument('--account', type=float, default=5000,
                        help='Account balance in USD (for lot sizing)')
    parser.add_argument('--risk', type=float, default=0.02,
                        help='Risk per trade as fraction (0.02 = 2%%)')
    args = parser.parse_args()

    trader = LiveV7Trader(args.ticker, args.account, args.risk)
    trader.run()


if __name__ == '__main__':
    main()
