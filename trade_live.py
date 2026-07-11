import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
from datetime import datetime
from stable_baselines3 import PPO
import ta
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
import warnings
import logging
import os

warnings.filterwarnings("ignore")

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("bot_execution.log"),
        logging.StreamHandler()
    ]
)

def process_features(df_raw):
    """Exact identical feature engineering to the V100 Crusher Brain training environment"""
    df = df_raw.copy()
    
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df = df.dropna().reset_index(drop=True)
    
    returns_scaled = df['Log_Return'] * 100
    am = arch_model(returns_scaled, vol='Garch', p=1, q=1)
    res = am.fit(disp='off')
    df['GARCH_Vol'] = res.conditional_volatility / 100
    
    model = ARIMA(df['Log_Return'], order=(1, 0, 1))
    model_fit = model.fit()
    df['ARIMA_Resid'] = model_fit.resid
    
    df = df.dropna().reset_index(drop=True)
    
    # The exact 7 columns the model was trained on, in the exact order:
    # ['Close', 'High', 'Low', 'Open', 'Log_Return', 'GARCH_Vol', 'ARIMA_Resid']
    feature_cols = ['Close', 'High', 'Low', 'Open', 'Log_Return', 'GARCH_Vol', 'ARIMA_Resid']
    
    return df[feature_cols].values


class DerivMT5Bot:
    def __init__(self):
        self.login = int(os.getenv("MT5_LOGIN", "32312791"))
        self.password = os.getenv("MT5_PASSWORD")
        self.server = os.getenv("MT5_SERVER", "Deriv-Demo")
        
        if not self.password:
            logging.error("MT5_PASSWORD environment variable not set.")
            raise ValueError("Missing MT5 Password. Please set $env:MT5_PASSWORD in your terminal.")
            
        self.symbol = "Volatility 100 Index"
        # Minimum lot size for Volatility 100 Index is 0.50
        self.volume = float(os.getenv("MT5_LOT_SIZE", "0.50"))
        self.window_size = 96
        
        # State Tracking
        self.current_ticket = None
        self.current_direction = None # 0 = UP (Buy), 1 = DOWN (Sell)
        
        logging.info("Loading V100 Crusher Brain...")
        self.model = PPO.load("v100_crusher_archive/ppo_deriv_bot.zip")

    def connect(self):
        logging.info("Initializing MetaTrader 5...")
        if not mt5.initialize():
            logging.error(f"MT5 initialize() failed, error code = {mt5.last_error()}")
            raise Exception("MT5 Initialization Failed")
            
        logging.info(f"Logging into MT5 account {self.login} on {self.server}...")
        authorized = mt5.login(self.login, password=self.password, server=self.server)
        
        if authorized:
            account_info = mt5.account_info()
            logging.info(f"Login successful! Balance: ${account_info.balance}")
        else:
            logging.error(f"MT5 login failed, error code = {mt5.last_error()}")
            raise Exception("MT5 Login Failed")
            
        # Ensure symbol is available in Market Watch
        if not mt5.symbol_select(self.symbol, True):
            logging.error(f"Failed to select symbol {self.symbol}")
            raise Exception(f"Symbol '{self.symbol}' not found in MT5")
            
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info:
            logging.info(f"Symbol specs: Min={symbol_info.volume_min}, Max={symbol_info.volume_max}, Step={symbol_info.volume_step}")
            if self.volume < symbol_info.volume_min:
                logging.warning(f"Configured volume {self.volume} is below minimum {symbol_info.volume_min}. Adjusting automatically.")
                self.volume = symbol_info.volume_min
            # ensure volume is a multiple of volume_step
            step = symbol_info.volume_step
            self.volume = round(round(self.volume / step) * step, 2)
            if self.volume < symbol_info.volume_min:
                self.volume += step
            logging.info(f"Final operating volume set to: {self.volume} lots")

    def close_position(self, ticket):
        position = mt5.positions_get(ticket=ticket)
        if not position:
            logging.warning(f"Position {ticket} not found or already closed.")
            return
            
        pos = position[0]
        close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(self.symbol).bid if close_type == mt5.ORDER_TYPE_SELL else mt5.symbol_info_tick(self.symbol).ask
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": pos.volume,
            "type": close_type,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": 1023,
            "comment": "AI Bot Close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
        
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logging.error(f"Failed to close position {ticket}: {result.comment}")
        else:
            logging.info(f"Successfully closed position {ticket}.")

    def execute_trade(self, action, sl_points, tp_points, volume):
        if self.current_ticket is not None:
            logging.info(f"Closing previous position {self.current_ticket}...")
            self.close_position(self.current_ticket)
            self.current_ticket = None
            self.current_direction = None
            
        order_type = mt5.ORDER_TYPE_BUY if action == 0 else mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(self.symbol).ask if order_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(self.symbol).bid
        
        if order_type == mt5.ORDER_TYPE_BUY:
            sl = price - sl_points
            tp = price + tp_points
        else:
            sl = price + sl_points
            tp = price - tp_points
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": 1023,
            "comment": "AI Bot Open",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
        
        logging.info(f"Executing {'Buy' if action == 0 else 'Sell'} {volume} lots with SL: {sl_points:.1f}, TP: {tp_points:.1f}...")
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logging.error(f"Failed to open position: {result.comment} (Retcode: {result.retcode})")
        else:
            self.current_ticket = result.order
            self.current_direction = action
            logging.info(f"Successfully opened position! Ticket: {self.current_ticket}")

    def run(self):
        self.connect()
        first_run = True
        
        while True:
            try:
                now = datetime.now()
                minutes_to_next = 15 - (now.minute % 15)
                seconds_to_next = (minutes_to_next * 60) - now.second
                
                if first_run:
                    first_run = False
                else:
                    logging.info(f"Sleeping for {seconds_to_next} seconds until the next 15m candle closes...")
                    time.sleep(seconds_to_next)
                
                logging.info("Fetching last 200 M15 candles from MT5...")
                rates = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M15, 0, 200)
                if rates is None or len(rates) < 200:
                    logging.error(f"Failed to fetch rates, got {len(rates) if rates is not None else 0}")
                    time.sleep(10)
                    continue
                    
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df = df.rename(columns={'time': 'Time', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'})
                
                try:
                    features_matrix = process_features(df)
                except Exception as e:
                    logging.error(f"Feature processing error: {e}")
                    time.sleep(10)
                    continue
                    
                if len(features_matrix) < self.window_size:
                    logging.warning(f"Not enough data to form {self.window_size} window. Got {len(features_matrix)}.")
                    time.sleep(10)
                    continue
                    
                obs = features_matrix[-self.window_size:]
                
                # Get AI action — NOTE: This model is always ~99.9% decisive.
                # Confidence thresholding has no effect and is intentionally omitted.
                action, _ = self.model.predict(obs, deterministic=True)
                action = int(action)
                
                direction_str = "Buy (UP)" if action == 0 else "Sell (DOWN)"
                logging.info(f"AI Prediction: {direction_str}")
                
                # SL/TP are used as catastrophic safety nets only.
                # Primary exits are triggered by the AI flipping direction.
                sl_points = float(os.getenv("MT5_SL_POINTS", "60.0"))
                tp_points = float(os.getenv("MT5_TP_POINTS", "60.0"))
                
                # --- PROPORTIONAL EQUITY SCALING ---
                # If MT5_COMPOUND=True, lot size scales proportionally with
                # account equity so profits compound as the account grows.
                # This is honest compounding — no broken SL-based Kelly formula.
                compound = os.getenv("MT5_COMPOUND", "False").lower() == "true"
                if compound:
                    account_info = mt5.account_info()
                    current_balance = account_info.balance
                    starting_balance = float(os.getenv("MT5_STARTING_BALANCE", "10000.0"))
                    scale_factor = current_balance / starting_balance
                    symbol_info = mt5.symbol_info(self.symbol)
                    step = symbol_info.volume_step
                    raw_volume = self.volume * scale_factor
                    final_volume = round(round(raw_volume / step) * step, 2)
                    final_volume = max(symbol_info.volume_min, min(symbol_info.volume_max, final_volume))
                    logging.info(f"Compound Mode: Balance=${current_balance:.2f}, Scale={scale_factor:.3f}x, Volume={final_volume}")
                else:
                    final_volume = self.volume
                
                if self.current_direction != action:
                    logging.info(f"Signal flipped! Triggering trade execution...")
                    self.execute_trade(action, sl_points, tp_points, final_volume)
                else:
                    logging.info(f"Holding current position ({direction_str}).")
                    # Track PnL
                    if self.current_ticket:
                        pos = mt5.positions_get(ticket=self.current_ticket)
                        if pos:
                            logging.info(f"Current P/L: ${pos[0].profit}")
                        else:
                            logging.warning("Position closed externally (TP/SL/Manual). Resetting internal state.")
                            self.current_ticket = None
                            self.current_direction = None
                    
            except Exception as e:
                logging.error(f"Strategy loop error: {e}")
                time.sleep(10)

if __name__ == "__main__":
    bot = DerivMT5Bot()
    try:
        bot.run()
    except KeyboardInterrupt:
        logging.info("Bot stopped by user.")
    finally:
        mt5.shutdown()
