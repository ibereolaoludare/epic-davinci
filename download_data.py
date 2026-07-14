import pandas as pd
from datetime import datetime, timezone
import MetaTrader5 as mt5
from timezone_utils import to_utc

SYMBOL = "USDJPY"
TIMEFRAME = mt5.TIMEFRAME_H4
NUM_CANDLES = 50000

# Static BOJ/MoF Interventions
INTERVENTIONS = [
    {"date": datetime(2022, 9, 22, 8, 0, tzinfo=timezone.utc), "level": 145.90, "note": "MoF Yen buying"},
    {"date": datetime(2022, 10, 21, 14, 0, tzinfo=timezone.utc), "level": 151.90, "note": "MoF Yen buying"},
    {"date": datetime(2024, 4, 29, 4, 0, tzinfo=timezone.utc), "level": 160.20, "note": "MoF Yen buying (Golden Week)"},
    {"date": datetime(2024, 5, 1, 20, 0, tzinfo=timezone.utc), "level": 157.50, "note": "MoF Yen buying"},
    {"date": datetime(2026, 4, 30, 4, 0, tzinfo=timezone.utc), "level": 160.00, "note": "MoF Yen buying cycle"}
]

# Static Historical Rates for Fed and BOJ (Step-functions updated on announcement days)
# These would normally be fetched via an Economic Calendar API.
RATES = [
    {"date": datetime(2023, 7, 26, 18, 0, tzinfo=timezone.utc), "fed_rate": 5.50, "boj_rate": -0.10},
    {"date": datetime(2024, 3, 19, 3, 0, tzinfo=timezone.utc), "fed_rate": 5.50, "boj_rate": 0.10},
    {"date": datetime(2024, 7, 31, 18, 0, tzinfo=timezone.utc), "fed_rate": 5.50, "boj_rate": 0.25},
    {"date": datetime(2024, 9, 18, 18, 0, tzinfo=timezone.utc), "fed_rate": 5.00, "boj_rate": 0.25},
    {"date": datetime(2024, 11, 7, 19, 0, tzinfo=timezone.utc), "fed_rate": 4.75, "boj_rate": 0.25},
    {"date": datetime(2024, 12, 18, 19, 0, tzinfo=timezone.utc), "fed_rate": 4.50, "boj_rate": 0.25},
    # Verified 2026 conditions as per prompt:
    {"date": datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc), "fed_rate": 3.75, "boj_rate": 0.75} 
]

def fetch_mt5_data():
    if not mt5.initialize():
        print("MT5 initialization failed. Ensure MetaTrader 5 is open and logged in.")
        return None

    print(f"Requesting {NUM_CANDLES} H4 candles for {SYMBOL}...")
    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, NUM_CANDLES)
    mt5.shutdown()

    if rates is None or len(rates) == 0:
        print("Failed to retrieve data from MT5.")
        return None

    df = pd.DataFrame(rates)
    
    # MT5 returns time as integer POSIX timestamps. We must ensure they are UTC.
    # Note: MT5 server time depends on the broker. 
    # For a standard broker (UTC+2/UTC+3 DST), we would subtract the offset.
    # We will assume the user's MT5 server is UTC. If not, the user should subtract the offset here.
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df['time'] = df['time'].dt.tz_localize(timezone.utc)
    
    df = df.rename(columns={'time': 'Time', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'})
    df = df[['Time', 'Open', 'High', 'Low', 'Close', 'Volume']]
    
    return df

def save_macro_data():
    # Save interventions
    df_int = pd.DataFrame(INTERVENTIONS)
    df_int.to_csv("interventions.csv", index=False)
    
    # Save rates
    df_rates = pd.DataFrame(RATES)
    df_rates.to_csv("central_bank_rates.csv", index=False)
    print("Saved macro reference data (interventions.csv, central_bank_rates.csv).")

if __name__ == "__main__":
    df = fetch_mt5_data()
    if df is not None:
        df.to_csv(f"{SYMBOL}_H4.csv", index=False)
        print(f"Saved {len(df)} candles to {SYMBOL}_H4.csv")
    
    save_macro_data()
