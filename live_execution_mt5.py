import time
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from stable_baselines3 import PPO

# ==========================================
# CONFIGURATION - YOU MUST FILL THIS IN
# ==========================================
MT5_LOGIN = 32312791
MT5_PASSWORD = '1b3r301@/M' # <--- PASTE PASSWORD HERE
MT5_SERVER = 'Deriv-Demo'

# Deriv Volatility 100 Index in MT5 is typically named 'Volatility 100 Index'
SYMBOL = 'Volatility 100 Index' 
STAKE_VOLUME = 0.50 # Lot size
WINDOW_SIZE = 30
# ==========================================

print("Initializing MetaTrader 5...")
if not mt5.initialize():
    print(f"initialize() failed, error code = {mt5.last_error()}")
    quit()

print(f"Logging into {MT5_SERVER} as {MT5_LOGIN}...")
authorized = mt5.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)
if not authorized:
    print(f"Failed to connect to MT5, error code: {mt5.last_error()}")
    mt5.shutdown()
    quit()
print("Successfully connected to MT5!")

try:
    print("Loading AI Brain...")
    model = PPO.load("ppo_deriv_bot")
except Exception as e:
    print("Error: Could not load 'ppo_deriv_bot.zip'. Did you run train_model.py first?")
    mt5.shutdown()
    quit()

# Add symbol to Market Watch
selected = mt5.symbol_select(SYMBOL, True)
if not selected:
    print(f"Failed to select '{SYMBOL}'. Please verify the exact symbol name in your MT5.")
    mt5.shutdown()
    quit()

def calculate_zscore(prices):
    if len(prices) < 2: return 0.0
    mean = np.mean(prices[-30:])
    std = np.std(prices[-30:])
    if std == 0: return 0.0
    return (prices[-1] - mean) / std

def calculate_rsi(prices, period=14):
    if len(prices) < period + 1: return 50.0
    s = pd.Series(prices)
    delta = s.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]

def calculate_macd(prices, fast=12, slow=26, signal=9):
    if len(prices) < slow + signal: return 0.0
    s = pd.Series(prices)
    exp1 = s.ewm(span=fast, adjust=False).mean()
    exp2 = s.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - signal_line
    return macd_hist.iloc[-1]

def execute_mt5_trade(action_type):
    """Executes a market order in MT5"""
    tick = mt5.symbol_info_tick(SYMBOL)
    price = tick.ask if action_type == mt5.ORDER_TYPE_BUY else tick.bid
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": STAKE_VOLUME,
        "type": action_type,
        "price": price,
        "deviation": 20,
        "magic": 234000,
        "comment": "AI Bot Trade",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Trade failed! Error code: {result.retcode}")
    else:
        print(f"Trade Execution Result: {result.comment} at {price}")

def run_live_bot():
    price_history = []
    raw_price_buffer = []
    current_position = 0 # 0=Flat, 1=Long, 2=Short
    
    print(f"Listening to {SYMBOL}... waiting for {WINDOW_SIZE} ticks to build data window.")
    
    while True:
        tick = mt5.symbol_info_tick(SYMBOL)
        if tick is None:
            time.sleep(1)
            continue
            
        price = tick.bid # Using bid price for analysis
        
        # Avoid duplicate ticks
        if raw_price_buffer and raw_price_buffer[-1] == price:
            time.sleep(0.5)
            continue
            
        raw_price_buffer.append(price)
        if len(raw_price_buffer) > 100:
            raw_price_buffer.pop(0)
            
        z_score = calculate_zscore(raw_price_buffer)
        rsi = calculate_rsi(raw_price_buffer)
        macd = calculate_macd(raw_price_buffer)
        
        # [Open, High, Low, Close, ZScore, RSI, MACD]
        current_observation = [price, price, price, price, z_score, rsi, macd]
        price_history.append(current_observation)
        
        if len(price_history) > WINDOW_SIZE:
            price_history.pop(0)
            
        print(f"[{SYMBOL}] Price: {price} | Z: {z_score:.2f} | RSI: {rsi:.1f} | MACD: {macd:.2f} | Buffer: {len(price_history)}/{WINDOW_SIZE}")
        
        if len(price_history) == WINDOW_SIZE:
            obs = np.array([price_history], dtype=np.float32) # Model expects batch dimension
            action, _states = model.predict(obs, deterministic=True)
            action = action[0] # Extract from batch
            
            if action == 1 and current_position != 1:
                print("🤖 AI DECISION: BUY!")
                execute_mt5_trade(mt5.ORDER_TYPE_BUY)
                current_position = 1
                
            elif action == 2 and current_position != 2:
                print("🤖 AI DECISION: SELL!")
                execute_mt5_trade(mt5.ORDER_TYPE_SELL)
                current_position = 2
                
        time.sleep(0.5) # Fast polling for ticks

try:
    run_live_bot()
except KeyboardInterrupt:
    print("\nBot stopped by user.")
finally:
    mt5.shutdown()
