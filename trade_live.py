import time
import logging
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import joblib
from datetime import datetime, timezone
import os
from feature_engineering_v2 import process_features_strict, FEATURE_COLS
from timezone_utils import to_utc

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LiveXGBTrader:
    def __init__(self, symbol="USDJPY", timeframe=mt5.TIMEFRAME_H4, horizon=6, threshold=0.55, volume=1.0):
        self.symbol = symbol
        self.timeframe = timeframe
        self.horizon = horizon
        self.threshold = threshold
        self.volume = volume
        
        logging.info("Loading XGBoost model...")
        self.model = joblib.load('xgb_model_usdjpy.pkl')
        
        self.active_trades = [] # List of dicts: {'ticket': id, 'candles_held': 0}

    def connect(self):
        if not mt5.initialize():
            logging.error("Failed to initialize MT5")
            raise Exception("MT5 initialization failed")
        
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            logging.error(f"Symbol {self.symbol} not found")
            raise Exception(f"Symbol {self.symbol} not found")
            
        if not symbol_info.visible:
            logging.info(f"Symbol {self.symbol} is not visible, making it visible.")
            if not mt5.symbol_select(self.symbol, True):
                logging.error(f"Failed to select symbol {self.symbol}")
                raise Exception(f"Failed to select symbol {self.symbol}")
                
        logging.info(f"Successfully connected to MT5. Trading {self.symbol}.")

    def close_position(self, ticket):
        position = mt5.positions_get(ticket=ticket)
        if position is None or len(position) == 0:
            logging.warning(f"Position {ticket} not found. Maybe hit SL/TP?")
            return
            
        position = position[0]
        action = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(self.symbol).bid if action == mt5.ORDER_TYPE_SELL else mt5.symbol_info_tick(self.symbol).ask
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": position.volume,
            "type": action,
            "position": position.ticket,
            "price": price,
            "deviation": 20,
            "magic": 234000,
            "comment": "XGB Horizon Exit",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logging.error(f"Failed to close position {ticket}: {result.comment}")
        else:
            logging.info(f"Successfully closed position {ticket}.")

    def execute_trade(self, direction):
        action = mt5.ORDER_TYPE_BUY if direction == 1 else mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(self.symbol).ask if action == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(self.symbol).bid
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": self.volume,
            "type": action,
            "price": price,
            "deviation": 20,
            "magic": 234000,
            "comment": "XGB Entry",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logging.error(f"Failed to open position: {result.comment}")
        else:
            logging.info(f"Successfully opened position! Ticket: {result.order}")
            self.active_trades.append({'ticket': result.order, 'candles_held': 0})

    def run(self):
        self.connect()
        logging.info(f"Starting Live Daemon. Waiting for H4 candle closes...")
        
        while True:
            try:
                # Naive wait loop for H4 (wait until top of 4th hour)
                now = datetime.now()
                # On H4, candle closes are typically 00:00, 04:00, 08:00, 12:00, 16:00, 20:00
                hours_to_next = 4 - (now.hour % 4)
                if hours_to_next == 4 and now.minute == 0 and now.second == 0:
                    pass # We are exactly at the top of the hour
                else:
                    minutes_to_next = (hours_to_next * 60) - now.minute - 1
                    seconds_to_next = (minutes_to_next * 60) + (60 - now.second) + 2
                    logging.info(f"Sleeping {seconds_to_next} seconds until next H4 candle close...")
                    time.sleep(seconds_to_next)
                
                # Manage active trades
                trades_to_keep = []
                for trade in self.active_trades:
                    trade['candles_held'] += 1
                    if trade['candles_held'] >= self.horizon:
                        logging.info(f"Trade {trade['ticket']} reached horizon of {self.horizon} candles. Closing...")
                        self.close_position(trade['ticket'])
                    else:
                        trades_to_keep.append(trade)
                self.active_trades = trades_to_keep

                logging.info("Fetching recent H4 candles...")
                rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 1, 500)
                if rates is None or len(rates) < 200:
                    logging.error("Failed to fetch enough rates.")
                    time.sleep(60)
                    continue
                    
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df = df.rename(columns={'time': 'Time', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'})
                
                features_df = process_features_strict(df)
                if features_df.empty:
                    continue
                    
                latest_row = features_df.iloc[-1]
                
                # --- PRE-TRADE CHECKS & ALERTS ---
                # 1. MoF Intervention Danger Zone Alert
                if latest_row['In_Intervention_Zone'] == 1:
                    logging.critical("🚨 ALERT: USD/JPY IS WITHIN 0.5% OF MoF INTERVENTION LEVEL (¥160.0). EXTREME VOLATILITY RISK. 🚨")
                    
                # 2. Macro Release Circuit Breaker
                # Suppress trades if a high-impact release is within 12 hours (0.5 days)
                if latest_row['Days_To_Next_Release'] < 0.5:
                    logging.warning(f"CIRCUIT BREAKER: Macro release is in {latest_row['Days_To_Next_Release']*24:.1f} hours. Suppressing entries.")
                    continue
                # ----------------------------------
                
                # Extract features for XGBoost
                obs = latest_row[[c for c in FEATURE_COLS if c in features_df.columns]].values.reshape(1, -1)
                
                prob = self.model.predict_proba(obs)[0][1]
                logging.info(f"XGBoost UP Probability: {prob*100:.1f}%")
                
                if prob >= self.threshold:
                    logging.info("Probability > Threshold! Executing LONG.")
                    self.execute_trade(1)
                elif prob <= (1 - self.threshold):
                    logging.info("Probability < (1 - Threshold)! Executing SHORT.")
                    self.execute_trade(-1)
                else:
                    logging.info("No trade triggered (low confidence).")

            except Exception as e:
                logging.error(f"Error in main loop: {e}")
                time.sleep(60)

if __name__ == "__main__":
    bot = LiveXGBTrader(symbol="USDJPY", timeframe=mt5.TIMEFRAME_H4, horizon=6, threshold=0.55, volume=1.0)
    bot.run()
