import pandas as pd
import numpy as np
import ta
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
from pykalman import KalmanFilter
from hmmlearn import hmm
import warnings

warnings.filterwarnings("ignore")

def get_weights_ffd(d, size):
    w = [1.]
    for k in range(1, size):
        w_ = -w[-1] / k * (d - k + 1)
        w.append(w_)
    return np.array(w)

def frac_diff_fast(series, d, window_size):
    w = get_weights_ffd(d, window_size)
    res = np.convolve(series.values, w, mode='valid')
    padded_res = np.concatenate([np.full(window_size - 1, np.nan), res])
    return pd.Series(padded_res, index=series.index)

def process_features(df_raw):
    """
    Advanced Quantitative Finance Feature Engineering Layer.
    Includes Fractional Differentiation, Kalman Filter Smoothing, and HMM Regime Detection.
    """
    df = df_raw.copy()
    
    # 1. Kalman Filter Smoothing (noise reduction on Price)
    try:
        kf = KalmanFilter(transition_matrices=[1],
                          observation_matrices=[1],
                          initial_state_mean=df['Close'].iloc[0],
                          initial_state_covariance=1,
                          observation_covariance=1,
                          transition_covariance=0.01)
        state_means, _ = kf.filter(df['Close'].values)
        df['KF_Close'] = state_means.flatten()
    except Exception as e:
        df['KF_Close'] = df['Close']
        print(f"Kalman Filter failed: {e}")

    # 2. Fractional Differentiation (stationary memory on Price)
    # d=0.4 is a common value in finance to achieve stationarity while retaining memory
    try:
        df['Frac_Diff'] = frac_diff_fast(df['Close'], d=0.4, window_size=50)
    except Exception as e:
        df['Frac_Diff'] = df['Close'].pct_change()
        print(f"Frac Diff failed: {e}")

    # Calculate indicators on the SMOOTHED price
    df['RSI_14'] = ta.momentum.RSIIndicator(close=df['KF_Close'], window=14).rsi() / 100.0
    
    macd = ta.trend.MACD(close=df['KF_Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Diff'] = macd.macd_diff()
    
    df['ATR'] = ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14).average_true_range()
    df['EMA_50'] = ta.trend.EMAIndicator(close=df['KF_Close'], window=50).ema_indicator()
    
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # We must not drop na yet to preserve indices
    returns_scaled = df['Log_Return'].dropna() * 100
    
    # GARCH Volatility
    try:
        am = arch_model(returns_scaled, vol='Garch', p=1, q=1)
        res = am.fit(disp='off')
        df['GARCH_Vol'] = np.nan
        df.iloc[1:, df.columns.get_loc('GARCH_Vol')] = res.conditional_volatility.values / 100
    except:
        df['GARCH_Vol'] = 0.0
    
    # ARIMA Residuals
    try:
        model = ARIMA(returns_scaled / 100, order=(1, 0, 1))
        model_fit = model.fit()
        df['ARIMA_Resid'] = np.nan
        df.iloc[1:, df.columns.get_loc('ARIMA_Resid')] = model_fit.resid.values
    except:
        df['ARIMA_Resid'] = 0.0
        
    # 3. Gaussian HMM Regime Detection
    try:
        # Train HMM on log returns
        returns_for_hmm = returns_scaled.values.reshape(-1, 1) / 100.0
        hmm_model = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=10)
        hmm_model.fit(returns_for_hmm)
        state_proba = hmm_model.predict_proba(returns_for_hmm)
        
        df['HMM_Regime_0'] = np.nan
        df['HMM_Regime_1'] = np.nan
        df['HMM_Regime_2'] = np.nan
        df.iloc[1:, df.columns.get_loc('HMM_Regime_0')] = state_proba[:, 0]
        df.iloc[1:, df.columns.get_loc('HMM_Regime_1')] = state_proba[:, 1]
        df.iloc[1:, df.columns.get_loc('HMM_Regime_2')] = state_proba[:, 2]
    except Exception as e:
        df['HMM_Regime_0'] = 0.33
        df['HMM_Regime_1'] = 0.33
        df['HMM_Regime_2'] = 0.33
        print(f"HMM failed: {e}")
        
    df['Dist_EMA_50'] = (df['KF_Close'] - df['EMA_50']) / df['EMA_50']
    df['MACD'] = df['MACD'] / df['ATR']
    df['MACD_Signal'] = df['MACD_Signal'] / df['ATR']
    df['MACD_Diff'] = df['MACD_Diff'] / df['ATR']
    df['ATR_Norm'] = df['ATR'] / df['KF_Close']
    
    df = df.dropna().reset_index(drop=True)
    
    # The exact 14 columns expected by the AI
    feature_cols = [
        'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Diff', 'ATR', 
        'Log_Return', 'GARCH_Vol', 'ARIMA_Resid', 'Dist_EMA_50', 'ATR_Norm',
        'Frac_Diff', 'HMM_Regime_0', 'HMM_Regime_1', 'HMM_Regime_2'
    ]
    return df[feature_cols].values
