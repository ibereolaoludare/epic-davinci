import asyncio
import json
import websockets
import pandas as pd
from datetime import datetime

SYMBOL = 'R_100'

async def download_history():
    uri = "wss://ws.binaryws.com/websockets/v3?app_id=1089" 
    target_count = 50000
    all_candles = []
    end_time = "latest"
    
    async with websockets.connect(uri) as websocket:
        print(f"Starting paginated download for {target_count} candles...")
        while len(all_candles) < target_count:
            print(f"Requesting 5,000 candles ending at {end_time}...")
            
            history_request = json.dumps({
                "ticks_history": SYMBOL,
                "adjust_start_time": 1,
                "count": 5000,
                "end": str(end_time),
                "style": "candles",
                "granularity": 900 # 900 seconds = 15 minute candles
            })
            
            await websocket.send(history_request)
            response = await websocket.recv()
            data = json.loads(response)
            
            if 'error' in data:
                print(f"Data Error: {data['error']['message']}")
                break
                
            if 'candles' in data and data['candles']:
                candles = data['candles']
                print(f"Received {len(candles)} candles.")
                oldest_epoch = candles[0]['epoch']
                all_candles = candles + all_candles # Keep chronological order
                
                # Set end_time for the next page to just before the oldest candle in this block
                end_time = oldest_epoch - 1
                
                if len(candles) < 5000:
                    print("Reached the beginning of available history.")
                    break
            else:
                break
        
        if all_candles:
            # Slice to exact target count
            all_candles = all_candles[-target_count:]
            print(f"Successfully collected {len(all_candles)} candles. Saving to CSV...")
            
            df = pd.DataFrame(all_candles)
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'epoch': 'Time'
            })
            df['Time'] = pd.to_datetime(df['Time'], unit='s')
            df.to_csv('deriv_historical_data.csv', index=False)
            print("Saved to 'deriv_historical_data.csv'!")

if __name__ == "__main__":
    asyncio.run(download_history())
