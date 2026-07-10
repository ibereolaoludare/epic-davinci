import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import warnings
warnings.filterwarnings("ignore")

print("Loading historical data...")
df = pd.read_csv('deriv_historical_data.csv')

# 1. Log Returns (Stationarity)
print("Calculating Log Returns...")
df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
df = df.dropna().reset_index(drop=True)

# 2. GARCH Volatility
print("Fitting GARCH(1,1) model for conditional volatility...")
# We scale returns by 100 for better optimization in arch
returns_scaled = df['Log_Return'] * 100
am = arch_model(returns_scaled, vol='Garch', p=1, q=1)
res = am.fit(disp='off')
df['GARCH_Vol'] = res.conditional_volatility / 100 # scale back

# 3. ARIMA Residuals (Momentum proxy)
print("Fitting ARIMA(1,0,1) model...")
# Fitting ARIMA on the entire series for feature extraction 
model = ARIMA(df['Log_Return'], order=(1, 0, 1))
model_fit = model.fit()
df['ARIMA_Resid'] = model_fit.resid

print("Saving advanced quantitative features to 'advanced_features.csv'...")
df = df.dropna().reset_index(drop=True)
df.to_csv('advanced_features.csv', index=False)
print("Done! You now have institutional-grade time series features.")
