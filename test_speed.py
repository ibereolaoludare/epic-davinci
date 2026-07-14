import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
import warnings
import os

warnings.filterwarnings("ignore")

def process_features(df_raw):
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
    feature_cols = ['Close', 'High', 'Low', 'Open', 'Log_Return', 'GARCH_Vol', 'ARIMA_Resid']
    return df[feature_cols].values

if not mt5.initialize():
    quit()

login = int(os.getenv("MT5_LOGIN", "32312791"))
password = os.getenv("MT5_PASSWORD")
server = os.getenv("MT5_SERVER", "Deriv-Demo")

if password:
    mt5.login(login, password=password, server=server)

rates = mt5.copy_rates_from_pos("Volatility 100 Index", mt5.TIMEFRAME_M15, 1, 5000)
df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')
df = df.rename(columns={'time': 'Time', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'})

start_time = time.time()
process_features(df)
end_time = time.time()

print(f"Time taken for 5000 rows: {end_time - start_time:.4f} seconds")
mt5.shutdown()
