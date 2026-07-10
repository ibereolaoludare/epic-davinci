import asyncio
import json
import websockets
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

# ==========================================
# CONFIGURATION - YOU MUST FILL THIS IN
# ==========================================
DERIV_API_TOKEN = 'pat_8ef16c547498c0350e6febbc6adbed622177ecc72b077d5b1ad629afb9a2bf92' 
SYMBOL = 'R_100' # Example: Volatility 100 Index
STAKE_AMOUNT = 10 # Amount to risk per trade in USD

# Load our trained AI brain
try:
    print("Loading AI Brain...")
    model = PPO.load("ppo_deriv_bot")
except Exception as e:
    print("Error: Could not load 'ppo_deriv_bot.zip'. Did you run train_model.py first?")
    exit()

price_history = []
raw_price_buffer = [] # Larger buffer for accurate indicator calculation
WINDOW_SIZE = 30
current_position = 0 # 0=Flat, 1=Long, 2=Short

def calculate_zscore(prices):
    """Calculate the rolling Z-Score for the current price"""
    if len(prices) < 2:
        return 0.0
    mean = np.mean(prices[-30:]) # Use last 30 for Z-score to match training
    std = np.std(prices[-30:])
    if std == 0:
        return 0.0
    return (prices[-1] - mean) / std

def calculate_rsi(prices, period=14):
    if len(prices) < period + 1:
        return 50.0
    s = pd.Series(prices)
    delta = s.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]

def calculate_macd(prices, fast=12, slow=26, signal=9):
    if len(prices) < slow + signal:
        return 0.0
    s = pd.Series(prices)
    exp1 = s.ewm(span=fast, adjust=False).mean()
    exp2 = s.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - signal_line
    return macd_hist.iloc[-1]

async def execute_deriv_trade(websocket, contract_type):
    """Sends the actual execution JSON to Deriv API"""
    trade_request = {
        "buy": 1,
        "price": STAKE_AMOUNT,
        "parameters": {
            "amount": STAKE_AMOUNT,
            "basis": "stake",
            "contract_type": contract_type, # 'CALL' for Buy, 'PUT' for Sell
            "currency": "USD",
            "duration": 5, # Duration of the trade
            "duration_unit": "t", # 't' for ticks, 'm' for minutes
            "symbol": SYMBOL
        }
    }
    await websocket.send(json.dumps(trade_request))
    response = await websocket.recv()
    print(f"Trade Execution Response: {json.loads(response)}")

async def listen_to_market():
    global current_position
    uri = "wss://ws.binaryws.com/websockets/v3?app_id=1089" 
    
    async with websockets.connect(uri) as websocket:
        # 1. Authenticate
        auth_request = json.dumps({"authorize": DERIV_API_TOKEN})
        await websocket.send(auth_request)
        response = await websocket.recv()
        auth_data = json.loads(response)
        if 'error' in auth_data:
             print(f"Auth Error: {auth_data['error']['message']}")
             return
        print(f"Successfully Authenticated as {auth_data['authorize']['email']}")
        
        # 2. Subscribe to live price ticks
        tick_request = json.dumps({"ticks": SYMBOL, "subscribe": 1})
        await websocket.send(tick_request)
        print(f"Subscribed to {SYMBOL} live prices. Waiting for {WINDOW_SIZE} ticks to build data window...")

        # 3. Listen loop
        async for message in websocket:
            data = json.loads(message)
            if 'tick' in data:
                price = data['tick']['quote']
                
                # Maintain raw price array for math
                raw_price_buffer.append(price)
                if len(raw_price_buffer) > 100: # 100 is enough for accurate EWMA
                    raw_price_buffer.pop(0)
                
                # Compute live indicators
                z_score = calculate_zscore(raw_price_buffer)
                rsi = calculate_rsi(raw_price_buffer)
                macd = calculate_macd(raw_price_buffer)
                
                # Format to match training environment exactly [Open, High, Low, Close, ZScore, RSI, MACD]
                # (Assuming tick chart, OHLC are all the same price momentarily)
                current_observation = [price, price, price, price, z_score, rsi, macd] 
                price_history.append(current_observation)
                
                if len(price_history) > WINDOW_SIZE:
                    price_history.pop(0)
                
                # Print status
                print(f"[{SYMBOL}] Price: {price} | Z: {z_score:.2f} | RSI: {rsi:.1f} | MACD: {macd:.2f} | Buffer: {len(price_history)}/{WINDOW_SIZE}")
                
                # Ask AI for decision
                if len(price_history) == WINDOW_SIZE:
                    obs = np.array(price_history, dtype=np.float32)
                    
                    action, _states = model.predict(obs, deterministic=True)
                    
                    # Prevent spamming the same order if we are already in position
                    if action == 1 and current_position != 1:
                        print("🤖 AI DECISION: BUY (CALL)!")
                        await execute_deriv_trade(websocket, "CALL")
                        current_position = 1
                        
                    elif action == 2 and current_position != 2:
                        print("🤖 AI DECISION: SELL (PUT)!")
                        await execute_deriv_trade(websocket, "PUT")
                        current_position = 2

if __name__ == "__main__":
    asyncio.run(listen_to_market())
