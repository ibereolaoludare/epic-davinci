"""
Feature Engineering v2 — USD/JPY Adaptation
=========================================
Adds macroeconomic alternative data (Rate Differential, Intervention Proximity)
and adjusts for Forex weekend gaps and UTC normalization.
"""

import pandas as pd
import numpy as np
import ta
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
from pykalman import KalmanFilter
from hmmlearn import hmm
import warnings
import joblib
from datetime import timezone

warnings.filterwarnings("ignore")

FEATURE_COLS = [
    'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Diff', 'ATR',
    'Log_Return', 'GARCH_Vol', 'Dist_EMA_50', 'ATR_Norm',
    'Frac_Diff', 'HMM_Regime_0', 'HMM_Regime_1', 'HMM_Regime_2',
    'Hour_Sin', 'Hour_Cos', 'DayOfWeek_Sin', 'DayOfWeek_Cos',
    'BB_Width', 'Weekend_Gap', 'Rate_Diff', 'Intervention_Dist', 
    'In_Intervention_Zone', 'Days_To_Next_Release'
]

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

def load_macro_data():
    try:
        df_rates = pd.read_csv("central_bank_rates.csv", parse_dates=['date'])
        df_rates['date'] = pd.to_datetime(df_rates['date'], utc=True)
        df_rates = df_rates.sort_values('date')
        
        df_int = pd.read_csv("interventions.csv", parse_dates=['date'])
        df_int['date'] = pd.to_datetime(df_int['date'], utc=True)
        df_int = df_int.sort_values('date')
        return df_rates, df_int
    except FileNotFoundError:
        return pd.DataFrame(), pd.DataFrame()

def process_features_strict(df_raw, fit_end_idx=None):
    df = df_raw.copy()
    if fit_end_idx is None:
        fit_end_idx = len(df)

    df['Time'] = pd.to_datetime(df['Time'], utc=True)

    # 1. Kalman Filter
    try:
        kf = KalmanFilter(
            transition_matrices=[1], observation_matrices=[1],
            initial_state_mean=df['Close'].iloc[0], initial_state_covariance=1,
            observation_covariance=1, transition_covariance=0.01
        )
        kf = kf.em(df['Close'].values[:fit_end_idx], n_iter=5)
        state_means, _ = kf.filter(df['Close'].values)
        df['KF_Close'] = state_means.flatten()
    except Exception as e:
        df['KF_Close'] = df['Close']

    # 2. Fractional Diff
    try:
        df['Frac_Diff'] = frac_diff_fast(df['Close'], d=0.4, window_size=50)
    except:
        df['Frac_Diff'] = df['Close'].pct_change()

    # 3. Technicals
    df['RSI_14'] = ta.momentum.RSIIndicator(close=df['KF_Close'], window=14).rsi() / 100.0
    macd = ta.trend.MACD(close=df['KF_Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Diff'] = macd.macd_diff()
    df['ATR'] = ta.volatility.AverageTrueRange(
        high=df['High'], low=df['Low'], close=df['Close'], window=14
    ).average_true_range()
    df['EMA_50'] = ta.trend.EMAIndicator(close=df['KF_Close'], window=50).ema_indicator()
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))

    # 4. GARCH Volatility
    returns_all = df['Log_Return'].dropna() * 100
    returns_train = returns_all.iloc[:fit_end_idx - 1]
    try:
        am = arch_model(returns_train, vol='Garch', p=1, q=1)
        res = am.fit(disp='off')
        am_full = arch_model(returns_all, vol='Garch', p=1, q=1)
        res_full = am_full.fit(disp='off', starting_values=res.params.values)
        df['GARCH_Vol'] = np.nan
        df.iloc[1:len(returns_all)+1, df.columns.get_loc('GARCH_Vol')] = res_full.conditional_volatility.values / 100
    except:
        df['GARCH_Vol'] = df['Log_Return'].rolling(20).std()

    # 5. HMM
    try:
        returns_for_hmm_train = returns_train.values.reshape(-1, 1) / 100.0
        hmm_model = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=50, random_state=42)
        hmm_model.fit(returns_for_hmm_train)
        returns_for_hmm_all = returns_all.values.reshape(-1, 1) / 100.0
        state_proba = hmm_model.predict_proba(returns_for_hmm_all)
        df['HMM_Regime_0'] = np.nan
        df['HMM_Regime_1'] = np.nan
        df['HMM_Regime_2'] = np.nan
        df.iloc[1:len(returns_all)+1, df.columns.get_loc('HMM_Regime_0')] = state_proba[:, 0]
        df.iloc[1:len(returns_all)+1, df.columns.get_loc('HMM_Regime_1')] = state_proba[:, 1]
        df.iloc[1:len(returns_all)+1, df.columns.get_loc('HMM_Regime_2')] = state_proba[:, 2]
    except:
        df['HMM_Regime_0'], df['HMM_Regime_1'], df['HMM_Regime_2'] = 0.33, 0.33, 0.33

    # Derived
    df['Dist_EMA_50'] = (df['KF_Close'] - df['EMA_50']) / df['EMA_50']
    df['MACD'] = df['MACD'] / df['ATR']
    df['MACD_Signal'] = df['MACD_Signal'] / df['ATR']
    df['MACD_Diff'] = df['MACD_Diff'] / df['ATR']
    df['ATR_Norm'] = df['ATR'] / df['KF_Close']
    
    bb = ta.volatility.BollingerBands(close=df['KF_Close'], window=20, window_dev=2)
    df['BB_Width'] = bb.bollinger_wband()

    # Time/Cyclic
    hour = df['Time'].dt.hour + df['Time'].dt.minute / 60.0
    day_of_week = df['Time'].dt.dayofweek
    df['Hour_Sin'] = np.sin(2 * np.pi * hour / 24.0)
    df['Hour_Cos'] = np.cos(2 * np.pi * hour / 24.0)
    df['DayOfWeek_Sin'] = np.sin(2 * np.pi * day_of_week / 7.0)
    df['DayOfWeek_Cos'] = np.cos(2 * np.pi * day_of_week / 7.0)

    # Weekend Gap Flag
    time_diff_hours = df['Time'].diff().dt.total_seconds() / 3600
    df['Weekend_Gap'] = (time_diff_hours > 24).astype(int) # H4 standard gap is 4. Anything > 24 is a weekend gap.

    # ── MACRO FEATURES (Zero Leakage) ──
    df_rates, df_int = load_macro_data()
    
    # Forward fill rate diffs based strictly on past announcement dates
    if not df_rates.empty:
        df = df.sort_values('Time')
        # Merge asof ensures we only look backward for the latest rate decision
        df = pd.merge_asof(df, df_rates[['date', 'fed_rate', 'boj_rate']], left_on='Time', right_on='date', direction='backward')
        df['Rate_Diff'] = df['fed_rate'] - df['boj_rate']
        df = df.drop(columns=['date', 'fed_rate', 'boj_rate'])
        df['Rate_Diff'] = df['Rate_Diff'].fillna(0) # Fill prior to first known date
        
        # Days to next release (requires looking forward at dates, BUT this is scheduled public info)
        future_dates = df_rates['date'].values
        
        def get_days_to_next(t):
            t_naive = t.tz_localize(None) if t.tzinfo else t
            future = future_dates[future_dates > np.datetime64(t_naive)]
            if len(future) > 0:
                return (pd.Timestamp(future[0]) - t_naive).total_seconds() / 86400
            return 30.0 # Default if unknown
            
        df['Days_To_Next_Release'] = df['Time'].apply(get_days_to_next)
    else:
        df['Rate_Diff'] = 0.0
        df['Days_To_Next_Release'] = 0.0

    # Intervention proximity
    if not df_int.empty:
        # Distance to absolute known defense line
        df['Intervention_Dist'] = np.abs(df['Close'] - 160.0) / 160.0
        # Danger zone = within 0.5% of 160
        df['In_Intervention_Zone'] = (df['Intervention_Dist'] < 0.005).astype(int)
    else:
        df['Intervention_Dist'] = 0.0
        df['In_Intervention_Zone'] = 0

    df = df.dropna().reset_index(drop=True)
    return df

if __name__ == "__main__":
    print("Loading USDJPY H4 data...")
    try:
        df_raw = pd.read_csv("USDJPY_H4.csv")
        FIT_END = int(len(df_raw) * 0.4) # Adjust for forex data size (approx 10k out of 25k)
        
        print(f"Fitting GARCH/HMM on first {FIT_END} candles only (zero look-ahead)...")
        df_features = process_features_strict(df_raw, fit_end_idx=FIT_END)
        
        save_cols = ['Time', 'Open', 'High', 'Low', 'Close', 'KF_Close', 'EMA_50'] + FEATURE_COLS
        available = [c for c in save_cols if c in df_features.columns]
        df_features[available].to_csv("USDJPY_features_strict.csv", index=False)
        print(f"Saved USDJPY_features_strict.csv: {len(df_features)} rows")
    except FileNotFoundError:
        print("USDJPY_H4.csv not found. Please run download_data.py first.")
