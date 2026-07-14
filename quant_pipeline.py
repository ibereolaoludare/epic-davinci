import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import ta
import warnings
import time
from multiprocessing import Pool, cpu_count
from pykalman import KalmanFilter
from hmmlearn import hmm
from feature_engineering import frac_diff_fast

warnings.filterwarnings("ignore")

def process_single_row(args):
    """
    Worker function to compute GARCH, ARIMA, and HMM for a single target index
    using the preceding 5,000 candles.
    """
    target_idx, log_returns = args
    # We take the 5000 candles up to and including target_idx
    window_returns = log_returns[target_idx - 5000 + 1 : target_idx + 1]
    
    # 1. GARCH
    returns_scaled = window_returns * 100
    try:
        am = arch_model(returns_scaled, vol='Garch', p=1, q=1)
        res = am.fit(disp='off')
        garch_vol = res.conditional_volatility[-1] / 100
    except:
        garch_vol = 0.0
    
    # 2. ARIMA
    try:
        model = ARIMA(returns_scaled / 100, order=(1, 0, 1))
        model_fit = model.fit()
        arima_resid = model_fit.resid[-1]
    except:
        arima_resid = 0.0
        
    # 3. HMM Regime Detection
    try:
        returns_for_hmm = (returns_scaled / 100.0).reshape(-1, 1)
        hmm_model = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=10)
        hmm_model.fit(returns_for_hmm)
        state_proba = hmm_model.predict_proba(returns_for_hmm)[-1]
        regime_0, regime_1, regime_2 = state_proba[0], state_proba[1], state_proba[2]
    except:
        regime_0, regime_1, regime_2 = 0.33, 0.33, 0.33
        
    return target_idx, garch_vol, arima_resid, regime_0, regime_1, regime_2

if __name__ == "__main__":
    print("Loading historical data...")
    df = pd.read_csv('deriv_historical_data.csv')

    print("1. Applying Kalman Filter (Online Smoothing, Zero Leakage)...")
    try:
        kf = KalmanFilter(transition_matrices=[1],
                          observation_matrices=[1],
                          initial_state_mean=df['Close'].iloc[0],
                          initial_state_covariance=1,
                          observation_covariance=1,
                          transition_covariance=0.01)
        state_means, _ = kf.filter(df['Close'].values)
        df['KF_Close'] = state_means.flatten()
    except:
        df['KF_Close'] = df['Close']

    print("2. Applying Fractional Differentiation (d=0.4)...")
    try:
        df['Frac_Diff'] = frac_diff_fast(df['Close'], d=0.4, window_size=50)
    except:
        df['Frac_Diff'] = df['Close'].pct_change()

    print("Calculating Standard Momentum Indicators (RSI, MACD, ATR, EMA) on Smoothed Price...")
    # RSI
    df['RSI_14'] = ta.momentum.RSIIndicator(close=df['KF_Close'], window=14).rsi() / 100.0

    # MACD
    macd = ta.trend.MACD(close=df['KF_Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Diff'] = macd.macd_diff()

    # ATR
    df['ATR'] = ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14).average_true_range()

    # 50 EMA for distance calculation
    df['EMA_50'] = ta.trend.EMAIndicator(close=df['KF_Close'], window=50).ema_indicator()

    print("Calculating Log Returns...")
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    df['GARCH_Vol'] = np.nan
    df['ARIMA_Resid'] = np.nan
    df['HMM_Regime_0'] = np.nan
    df['HMM_Regime_1'] = np.nan
    df['HMM_Regime_2'] = np.nan
    
    log_returns_array = df['Log_Return'].values
    
    # We start from index 5000
    start_idx = 5000
    end_idx = len(df)
    
    tasks = [(i, log_returns_array) for i in range(start_idx, end_idx)]
    
    print(f"Starting parallel multiprocessing for {len(tasks)} sliding windows across {cpu_count()} cores...")
    print("This will dynamically fit GARCH, ARIMA, and HMM ensuring ZERO LEAKAGE.")
    print("This may take 15-45 minutes depending on your CPU.")
    
    start_time = time.time()
    
    with Pool(processes=cpu_count()) as pool:
        completed = 0
        for result in pool.imap_unordered(process_single_row, tasks, chunksize=50):
            idx, g_vol, a_res, r0, r1, r2 = result
            df.at[idx, 'GARCH_Vol'] = g_vol
            df.at[idx, 'ARIMA_Resid'] = a_res
            df.at[idx, 'HMM_Regime_0'] = r0
            df.at[idx, 'HMM_Regime_1'] = r1
            df.at[idx, 'HMM_Regime_2'] = r2
            completed += 1
            if completed % 1000 == 0:
                elapsed = time.time() - start_time
                print(f"Processed {completed}/{len(tasks)} rows... ({elapsed:.1f}s elapsed)")

    print("Normalizing Feature Space...")
    # Distance from EMA (Stationary price proxy)
    df['Dist_EMA_50'] = (df['KF_Close'] - df['EMA_50']) / df['EMA_50']

    # Normalize MACD and ATR roughly to keep them bounded
    df['MACD'] = df['MACD'] / df['ATR']
    df['MACD_Signal'] = df['MACD_Signal'] / df['ATR']
    df['MACD_Diff'] = df['MACD_Diff'] / df['ATR']
    df['ATR_Norm'] = df['ATR'] / df['KF_Close']

    # Drop the first 5000 rows that don't have GARCH/ARIMA/HMM
    df = df.iloc[5000:].reset_index(drop=True)
    df = df.dropna().reset_index(drop=True)

    print("Saving strictly unbiased quantitative features to 'advanced_features_strict.csv'...")
    df.to_csv('advanced_features_strict.csv', index=False)
    print("Done! Pipeline successfully generated ADVANCED ZERO-LEAKAGE training data.")
