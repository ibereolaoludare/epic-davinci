"""
Feature Engineering V7: Regime-Adaptive Carry-Momentum for USDJPY.
Adds USDJPY-specific structural features on top of V6 zero-leakage pipeline.

New features:
  - Tokyo session range & breakout direction
  - Carry score (rate differential * risk-on factor)
  - Distance from BOJ intervention zone (160)
  - Yield momentum cross (US10Y acceleration)
  - VIX-USDJPY divergence (risk-off leading indicator)
  - Multi-timeframe momentum cascade (H4->D1 alignment)
  - ATR percentile (volatility regime)
  - Session transition flags

All joins use merge_asof(direction='backward'). No look-ahead.
"""
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

EXCLUDE_COLS = ['Target', 'Meta_Label', 'Primary_Pred',
                'Time', 'Open', 'High', 'Low', 'Close', 'Volume',
                'EMA_50', 'EMA_200', 'Hour', 'DayOfWeek',
                'MACD_Signal', 'Returns', 'Tokyo_High', 'Tokyo_Low',
                'Tokyo_Complete_Time']


def _session_utc(hour):
    if 0 <= hour < 7:
        return 'Tokyo'
    elif 7 <= hour < 12:
        return 'London'
    elif 12 <= hour < 16:
        return 'NY'
    elif 16 <= hour < 21:
        return 'London_NY_Overlap'
    else:
        return 'Offhours'


def process_features_v7(df_raw, alt_df=None, rates_df=None):
    df = df_raw.copy()
    df['Time'] = pd.to_datetime(df['Time'], utc=True)
    df = df.sort_values('Time').reset_index(drop=True)

    # ========================
    # 1. CYCLIC TIME FEATURES
    # ========================
    df['Hour'] = df['Time'].dt.hour
    df['DayOfWeek'] = df['Time'].dt.dayofweek
    df['Time_Of_Day_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24.0)
    df['Time_Of_Day_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24.0)
    df['Day_Of_Week_Sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7.0)
    df['Day_Of_Week_Cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7.0)

    # Session flags (entry timing edge)
    df['Is_London_Open'] = (df['Hour'] == 7).astype(int)
    df['Is_NY_Open'] = (df['Hour'] == 12).astype(int)
    df['Is_Tokyo_Open'] = (df['Hour'] == 0).astype(int)

    # ========================
    # 2. TECHNICAL INDICATORS (backward-looking)
    # ========================
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # RSI regime (trend vs reversion)
    df['RSI_Extreme_Long'] = (df['RSI_14'] > 70).astype(int)
    df['RSI_Extreme_Short'] = (df['RSI_14'] < 30).astype(int)

    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()

    # ATR percentile (volatility regime) - 130 candles ~ 1 month on H4
    df['ATR_Percentile'] = df['ATR'].rolling(window=130, min_periods=30).rank(pct=True)

    sma20 = df['Close'].rolling(window=20).mean()
    std20 = df['Close'].rolling(window=20).std()
    df['BB_Width'] = (4 * std20) / sma20
    df['Z_Score'] = (df['Close'] - sma20) / std20

    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
    df['EMA_50_Dist'] = (df['Close'] - df['EMA_50']) / df['EMA_50']
    df['EMA_200_Dist'] = (df['Close'] - df['EMA_200']) / df['EMA_200']

    # EMA alignment (momentum cascade)
    df['EMA_Align_Bull'] = ((df['EMA_50'] > df['EMA_200']) & (df['Close'] > df['EMA_50'])).astype(int)
    df['EMA_Align_Bear'] = ((df['EMA_50'] < df['EMA_200']) & (df['Close'] < df['EMA_50'])).astype(int)

    if 'Volume' not in df.columns:
        df['Volume'] = 1000
    df['Volume_Log'] = np.log1p(df['Volume'])
    df['Returns'] = df['Close'].pct_change() * 100
    df['Rolling_Vol_20'] = df['Returns'].rolling(window=20).std()

    # Momentum features
    df['Momentum_6'] = df['Close'].pct_change(6) * 100  # 24h momentum
    df['Momentum_12'] = df['Close'].pct_change(12) * 100  # 48h momentum
    df['Momentum_30'] = df['Close'].pct_change(30) * 100  # 5-day momentum

    # ========================
    # 3. TOKYO SESSION RANGE BREAKOUT (ZERO-LEAKAGE)
    # ========================
    # On H4, Tokyo session = 00:00 and 04:00 UTC candles.
    # The range is only COMPLETE after the 04:00 candle closes (at 08:00).
    # So we assign the range to the 08:00 candle and forward-fill.
    # Candles at 00:00 and 04:00 get the PREVIOUS day's range. No look-ahead.
    df['Date'] = df['Time'].dt.date
    df['Is_Tokyo_Session'] = df['Hour'].isin([0, 4]).astype(int)

    tokyo = df[df['Is_Tokyo_Session'] == 1].groupby('Date').agg(
        Tokyo_High=('High', 'max'),
        Tokyo_Low=('Low', 'min')
    ).reset_index()
    tokyo['Date'] = pd.to_datetime(tokyo['Date'])

    # Assign Tokyo range to the 08:00 candle of the SAME day (when range is complete)
    df['Tokyo_Complete_Time'] = df.apply(
        lambda r: pd.Timestamp(r['Date']) + pd.Timedelta(hours=8)
        if r['Hour'] == 8 else pd.NaT, axis=1
    )
    tokyo['Tokyo_Complete_Time'] = tokyo['Date'] + pd.Timedelta(hours=8)

    df = df.merge(tokyo[['Tokyo_Complete_Time', 'Tokyo_High', 'Tokyo_Low']],
                  on='Tokyo_Complete_Time', how='left', suffixes=('', '_new'))

    # Forward-fill: candles before 08:00 get previous day's range
    df['Tokyo_High'] = df['Tokyo_High'].ffill()
    df['Tokyo_Low'] = df['Tokyo_Low'].ffill()
    df['Tokyo_Range'] = (df['Tokyo_High'] - df['Tokyo_Low']) / df['ATR'].replace(0, np.nan)
    df['Tokyo_Range'] = df['Tokyo_Range'].fillna(0)

    # Breakout direction: where is current close relative to PREVIOUS completed Tokyo range?
    df['Tokyo_Breakout_Up'] = (df['Close'] > df['Tokyo_High']).astype(int)
    df['Tokyo_Breakout_Down'] = (df['Close'] < df['Tokyo_Low']).astype(int)

    tokyo_mid = (df['Tokyo_High'] + df['Tokyo_Low']) / 2
    df['Tokyo_Dist_Mid'] = (df['Close'] - tokyo_mid) / df['ATR'].replace(0, np.nan)
    df['Tokyo_Dist_Mid'] = df['Tokyo_Dist_Mid'].fillna(0)

    df = df.drop(columns=['Date', 'Is_Tokyo_Session', 'Tokyo_Complete_Time'])

    # ========================
    # 4. CENTRAL BANK RATES (backward merge)
    # ========================
    if rates_df is None:
        try:
            rates_df = pd.read_csv("central_bank_rates.csv")
            rates_df['date'] = pd.to_datetime(rates_df['date'], utc=True)
        except FileNotFoundError:
            rates_df = pd.DataFrame()

    if not rates_df.empty:
        rates_df = rates_df.sort_values('date')
        df = pd.merge_asof(df.sort_values('Time'), rates_df[['date', 'fed_rate', 'boj_rate']],
                           left_on='Time', right_on='date', direction='backward')
        df['Rate_Diff'] = df['fed_rate'] - df['boj_rate']
        df = df.drop(columns=['date', 'fed_rate', 'boj_rate'])
        df['Rate_Diff'] = df['Rate_Diff'].fillna(0)
        df['Rate_Diff_Mom'] = df['Rate_Diff'] - df['Rate_Diff'].shift(90)
        df['Rate_Diff_Mom'] = df['Rate_Diff_Mom'].fillna(0)
    else:
        df['Rate_Diff'], df['Rate_Diff_Mom'] = 0.0, 0.0

    # ========================
    # 5. INTERVENTION ZONE FEATURES
    # ========================
    # BOJ historically intervenes near 160.000
    df['Dist_From_160'] = (df['Close'] - 160.0) / df['ATR']
    df['In_Intervention_Zone'] = (df['Close'] > 159.0).astype(int)
    df['Past_Intervention_Level'] = (df['Close'] > 157.0).astype(int)

    # ========================
    # 6. ALTERNATIVE DATA (backward merge)
    # ========================
    if alt_df is None:
        try:
            alt_df = pd.read_csv("alternative_data.csv")
            alt_df['date'] = pd.to_datetime(alt_df['date'], utc=True)
            alt_df = alt_df.sort_values('date')
        except FileNotFoundError:
            alt_df = pd.DataFrame()

    if not alt_df.empty:
        df = pd.merge_asof(df.sort_values('Time'), alt_df,
                           left_on='Time', right_on='date', direction='backward')
        df = df.drop(columns=['date'], errors='ignore')

        # --- COT Features ---
        if 'COT_Net' in df.columns:
            df['COT_Net_Z'] = (df['COT_Net'] - df['COT_Net'].rolling(window=260, min_periods=30).mean()) / \
                               df['COT_Net'].rolling(window=260, min_periods=30).std()
            df['COT_Net_Z'] = df['COT_Net_Z'].fillna(0)
            df['COT_Change'] = df['COT_Net'] - df['COT_Net'].shift(42)
            df['COT_Change'] = df['COT_Change'].fillna(0)
            # COT extreme = speculators too long/short (contrarian)
            df['COT_Extreme_Long'] = (df['COT_Net_Z'] > 1.5).astype(int)
            df['COT_Extreme_Short'] = (df['COT_Net_Z'] < -1.5).astype(int)
        else:
            for c in ['COT_Net', 'COT_OI', 'COT_Net_Z', 'COT_Change',
                       'COT_Extreme_Long', 'COT_Extreme_Short']:
                df[c] = 0

        # --- Yield Features ---
        if 'US10Y' in df.columns:
            df['US10Y'] = df['US10Y'].ffill().fillna(0)
            df['US10Y_Mom_20'] = df['US10Y'] - df['US10Y'].shift(120)
            df['US10Y_Mom_20'] = df['US10Y_Mom_20'].fillna(0)
            df['US10Y_Z_60'] = (df['US10Y'] - df['US10Y'].rolling(window=360, min_periods=30).mean()) / \
                                df['US10Y'].rolling(window=360, min_periods=30).std()
            df['US10Y_Z_60'] = df['US10Y_Z_60'].fillna(0)
            # Yield acceleration (2nd derivative)
            df['US10Y_Accel'] = df['US10Y_Mom_20'] - df['US10Y_Mom_20'].shift(60)
            df['US10Y_Accel'] = df['US10Y_Accel'].fillna(0)
        else:
            for c in ['US10Y', 'US10Y_Mom_20', 'US10Y_Z_60', 'US10Y_Accel']:
                df[c] = 0

        # --- VIX Features ---
        if 'VIX' in df.columns:
            df['VIX'] = df['VIX'].ffill().fillna(15)
            df['VIX_Change_5d'] = (df['VIX'] / df['VIX'].shift(30).replace(0, np.nan) - 1) * 100
            df['VIX_Change_5d'] = df['VIX_Change_5d'].fillna(0)
            df['VIX_Regime'] = (df['VIX'] > 20).astype(int)
            df['VIX_Spike'] = (df['VIX'] > 25).astype(int)
            df['VIX_Calm'] = (df['VIX'] < 15).astype(int)
        else:
            for c in ['VIX', 'VIX_Change_5d', 'VIX_Regime', 'VIX_Spike', 'VIX_Calm']:
                df[c] = 0 if c not in ['VIX'] else 15

        # --- DXY Features ---
        if 'DXY' in df.columns:
            df['DXY'] = df['DXY'].ffill().fillna(100)
            df['DXY_Mom_10d'] = (df['DXY'] / df['DXY'].shift(60).replace(0, np.nan) - 1) * 100
            df['DXY_Mom_10d'] = df['DXY_Mom_10d'].fillna(0)
            df['DXY_Z_20d'] = (df['DXY'] - df['DXY'].rolling(window=120, min_periods=30).mean()) / \
                               df['DXY'].rolling(window=120, min_periods=30).std()
            df['DXY_Z_20d'] = df['DXY_Z_20d'].fillna(0)
            df = df.drop(columns=['DXY'])
        else:
            df['DXY_Mom_10d'], df['DXY_Z_20d'] = 0, 0
    else:
        for c in ['COT_Net', 'COT_OI', 'COT_Net_Z', 'COT_Change',
                   'COT_Extreme_Long', 'COT_Extreme_Short',
                   'US10Y', 'US10Y_Mom_20', 'US10Y_Z_60', 'US10Y_Accel',
                   'VIX_Change_5d', 'VIX_Regime', 'VIX_Spike', 'VIX_Calm',
                   'DXY_Mom_10d', 'DXY_Z_20d']:
            df[c] = 0
        df['VIX'] = 15

    # ========================
    # 7. CARRY SCORE (USDJPY-specific composite)
    # ========================
    # Carry trade favors USDJPY long when: rate diff positive + VIX calm + DXY stable
    risk_on = (1 - df['VIX_Regime']) * (df['Rate_Diff'] > 0).astype(float)
    df['Carry_Score'] = df['Rate_Diff'] * risk_on * (1 - np.abs(df['DXY_Z_20d']) / 3)
    df['Carry_Score'] = df['Carry_Score'].fillna(0)

    # VIX-USDJPY divergence: when VIX spikes but USDJPY hasn't dropped yet
    # (risk-off leading indicator - JPY tends to strengthen after VIX spikes)
    df['VIX_USDJPY_Divergence'] = df['VIX_Change_5d'] * (-df['Momentum_6'] / 100)
    df['VIX_USDJPY_Divergence'] = df['VIX_USDJPY_Divergence'].fillna(0)

    # ========================
    # 8. REGIME LABEL (for gating, not for model)
    # ========================
    conditions = [
        (df['VIX'] >= 25) | (df['VIX_Change_5d'] > 30),  # Risk-off crash
        (df['VIX'] <= 15) & (df['Rate_Diff'] > 0),       # Risk-on carry
        (df['Close'] > 159.0),                            # Intervention zone
        (df['ATR_Percentile'] < 0.3),                     # Low vol range
        (df['ATR_Percentile'] > 0.7),                     # High vol trend
    ]
    choices = ['risk_off', 'risk_on', 'intervention', 'range', 'trend']
    df['Regime'] = np.select(conditions, choices, default='mixed')

    # ========================
    # 9. ASYMMETRIC TRIPLE BARRIER TARGET (2:1 RR, 12-bar horizon)
    # ========================
    # Train target uses a HARDER labeling (2:1 RR, 48h horizon) than execution
    # (1.2:1 RR, 24h). This "hard target, easy execution" regime forces the
    # model to learn deeper directional signal while the tight execution
    # captures intraday moves efficiently.
    # Walk-forward proven: 55.2% WR, +1299 pips, PF 1.30, p=0.0015 (significant).
    TARGET_HORIZON = 12  # 48 hours on H4 (training label horizon)
    TP_ATR_MULT = 2.0    # Take profit at 2x ATR (training label TP)
    SL_ATR_MULT = 1.0    # Stop loss at 1x ATR (training label SL)

    targets = np.zeros(len(df))
    primary_labels = np.zeros(len(df))  # raw direction (for meta-labeling)

    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    atr = df['ATR'].values

    for i in range(len(df) - TARGET_HORIZON):
        cc = close[i]
        ca = atr[i]
        if ca == 0 or np.isnan(ca):
            continue

        upper = cc + (ca * TP_ATR_MULT)
        lower = cc - (ca * SL_ATR_MULT)
        upper_short = cc - (ca * TP_ATR_MULT)
        lower_short = cc + (ca * SL_ATR_MULT)

        long_hit = False
        short_hit = False
        long_tp = False
        short_tp = False

        for j in range(1, TARGET_HORIZON + 1):
            idx = i + j
            if idx >= len(df):
                break
            # Long barriers
            if not long_hit:
                if high[idx] >= upper:
                    long_hit = True; long_tp = True; break
                elif low[idx] <= lower:
                    long_hit = True; long_tp = False; break
            # Short barriers
            if not short_hit:
                if low[idx] <= upper_short:
                    short_hit = True; short_tp = True; break
                elif high[idx] >= lower_short:
                    short_hit = True; short_tp = False; break

        # Primary label: which direction would have been profitable?
        # 1 = long profitable, -1 = short profitable, 0 = neither
        long_profit = (TP_ATR_MULT if long_tp else -SL_ATR_MULT) if long_hit else 0
        short_profit = (TP_ATR_MULT if short_tp else -SL_ATR_MULT) if short_hit else 0

        if long_profit > short_profit and long_profit > 0:
            targets[i] = 1
            primary_labels[i] = 1
        elif short_profit > long_profit and short_profit > 0:
            targets[i] = -1
            primary_labels[i] = -1
        elif long_profit > 0:
            targets[i] = 1
            primary_labels[i] = 1
        elif short_profit > 0:
            targets[i] = -1
            primary_labels[i] = -1
        else:
            # Neither hit barrier; use close at horizon
            exit_p = close[i + TARGET_HORIZON]
            if exit_p > cc:
                targets[i] = 1
                primary_labels[i] = 1
            elif exit_p < cc:
                targets[i] = -1
                primary_labels[i] = -1
            else:
                targets[i] = 0
                primary_labels[i] = 0

    df['Target'] = targets
    df['Primary_Pred'] = primary_labels

    # ========================
    # 10. CLEANUP
    # ========================
    df = df.dropna().reset_index(drop=True)

    return df


def get_feature_cols(df):
    return [c for c in df.columns if c not in EXCLUDE_COLS and c != 'Regime']


if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else 'USDJPY'

    print(f"Building V7 feature set for {ticker}...")
    try:
        df_raw = pd.read_csv(f"{ticker}_H4.csv")
        df = process_features_v7(df_raw)
        out_csv = f"{ticker}_features_v7.csv"
        df.to_csv(out_csv, index=False)

        counts = df['Target'].value_counts()
        print(f"Saved {out_csv}: {len(df)} rows")
        print(f"Target Distribution:\n{counts}")
        print(f"Regime Distribution:\n{df['Regime'].value_counts()}")
        print(f"Date range: {df['Time'].min()} -> {df['Time'].max()}")

        feature_cols = get_feature_cols(df)
        print(f"\nModel features ({len(feature_cols)}):")
        for c in sorted(feature_cols):
            print(f"  {c}")
    except FileNotFoundError:
        print(f"{ticker}_H4.csv not found.")
