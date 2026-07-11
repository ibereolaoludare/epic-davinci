import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import ta
import warnings
warnings.filterwarnings("ignore")

print("Loading historical data...")
df = pd.read_csv('deriv_historical_data.csv')

print("Calculating Standard Momentum Indicators (RSI, MACD, ATR, EMA)...")
# RSI
df['RSI_14'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi() / 100.0  # Normalized 0 to 1

# MACD
macd = ta.trend.MACD(close=df['Close'])
df['MACD'] = macd.macd()
df['MACD_Signal'] = macd.macd_signal()
df['MACD_Diff'] = macd.macd_diff()

# ATR
df['ATR'] = ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14).average_true_range()

# 50 EMA for distance calculation
df['EMA_50'] = ta.trend.EMAIndicator(close=df['Close'], window=50).ema_indicator()

print("Calculating Log Returns...")
df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
df = df.dropna().reset_index(drop=True)

print("Fitting GARCH(1,1) model for conditional volatility...")
returns_scaled = df['Log_Return'] * 100
am = arch_model(returns_scaled, vol='Garch', p=1, q=1)
res = am.fit(disp='off')
df['GARCH_Vol'] = res.conditional_volatility / 100

print("Fitting ARIMA(1,0,1) model...")
model = ARIMA(df['Log_Return'], order=(1, 0, 1))
model_fit = model.fit()
df['ARIMA_Resid'] = model_fit.resid

print("Normalizing Feature Space...")
# Distance from EMA (Stationary price proxy)
df['Dist_EMA_50'] = (df['Close'] - df['EMA_50']) / df['EMA_50']

# Standardize Volume
df['Volume_Z'] = (df['Volume'] - df['Volume'].rolling(50).mean()) / df['Volume'].rolling(50).std()

# Normalize MACD and ATR roughly to keep them bounded
df['MACD'] = df['MACD'] / df['ATR']
df['MACD_Signal'] = df['MACD_Signal'] / df['ATR']
df['MACD_Diff'] = df['MACD_Diff'] / df['ATR']
df['ATR_Norm'] = df['ATR'] / df['Close']

df = df.dropna().reset_index(drop=True)

# DROP RAW ABSOLUTE PRICES TO PREVENT NETWORK WEIGHT COLLAPSE
print("Dropping unscaled absolute prices...")
df = df.drop(columns=['Open', 'High', 'Low', 'Close', 'Volume', 'EMA_50'])

print("Saving advanced quantitative features to 'advanced_features.csv'...")
df.to_csv('advanced_features.csv', index=False)
print("Done! You now have institutional-grade normalized time series features.")
