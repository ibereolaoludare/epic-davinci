import pandas as pd
import numpy as np
from datetime import timezone

df = pd.read_csv('USDJPY_features_v5.csv').dropna()
df['Time'] = pd.to_datetime(df['Time'], utc=True)

total_len = len(df)
TRAIN_END = int(total_len * 0.70)
VAL_END = int(total_len * 0.85)

df_train = df.iloc[:TRAIN_END]
df_val   = df.iloc[TRAIN_END:VAL_END]
df_test  = df.iloc[VAL_END:]

print("=== DATE RANGES ===")
print(f"Train: {df_train['Time'].iloc[0]}  ->  {df_train['Time'].iloc[-1]}  ({len(df_train)} rows)")
print(f"Val:   {df_val['Time'].iloc[0]}  ->  {df_val['Time'].iloc[-1]}  ({len(df_val)} rows)")
print(f"Test:  {df_test['Time'].iloc[0]}  ->  {df_test['Time'].iloc[-1]}  ({len(df_test)} rows)")

# Check: Are test targets using future data that extends BEYOND the dataset?
TARGET_HORIZON = 12
last_valid_target_idx = len(df) - TARGET_HORIZON - 1
test_start_idx = VAL_END
test_end_idx = len(df) - 1
print(f"\n=== TARGET LABEL INTEGRITY ===")
print(f"Last row that can have a valid target label (idx): {last_valid_target_idx}")
print(f"Test set index range: {test_start_idx} -> {test_end_idx}")
n_invalid = max(0, test_end_idx - last_valid_target_idx)
print(f"Test rows with target=0 due to insufficient future horizon: {n_invalid}")
targets_test = df_test['Target'].values
print(f"Test target distribution: 1={np.sum(targets_test==1)}, -1={np.sum(targets_test==-1)}, 0={np.sum(targets_test==0)}")

# Check: Rate_Diff in test set
rates = pd.read_csv('central_bank_rates.csv')
rates['date'] = pd.to_datetime(rates['date'], utc=True)
test_start_date = df_test['Time'].iloc[0]
print(f"\n=== MACRO DATA INTEGRITY ===")
print(f"Test starts at: {test_start_date}")
print(f"Rate change dates:")
for _, row in rates.iterrows():
    marker = ""
    if row['date'] >= test_start_date:
        marker = " <-- FALLS IN TEST PERIOD (backward merge = rate is public at announcement)"
    print(f"  {row['date']}  fed={row['fed_rate']}  boj={row['boj_rate']}{marker}")

# Check: Event window dates in backtest
known_intervention_dates = [
    pd.Timestamp('2022-09-22 08:00:00', tz=timezone.utc),
    pd.Timestamp('2022-10-21 14:00:00', tz=timezone.utc),
    pd.Timestamp('2024-04-29 04:00:00', tz=timezone.utc),
    pd.Timestamp('2024-05-01 20:00:00', tz=timezone.utc),
    pd.Timestamp('2026-04-30 04:00:00', tz=timezone.utc)
]
print(f"\n=== EVENT WINDOW DATES vs TEST PERIOD ===")
for ev in known_intervention_dates:
    in_test = "YES" if ev >= test_start_date else "No"
    print(f"  {ev}  ->  In test: {in_test}")

# Count how many test candles fall in event windows
event_count = 0
for _, row in df_test.iterrows():
    t = row['Time']
    for ev in known_intervention_dates:
        if abs((t - ev).total_seconds()) <= 7 * 86400:
            event_count += 1
            break
print(f"Test candles in event windows: {event_count} / {len(df_test)}")

# Check: Is the backtest re-entering trades on overlapping horizons?
print(f"\n=== TRADE OVERLAP CHECK ===")
print(f"Backtest iterates i from 0 to {len(df_test) - TARGET_HORIZON}")
print(f"Each trade occupies candles i to i+{TARGET_HORIZON}")
print("WARNING: The backtest does NOT skip candles after entering a trade.")
print("This means overlapping trades are possible (trade at candle 5 and candle 6 simultaneously).")
print("In live trading you would only hold ONE position at a time.")
print("This is a SIMULATION FIDELITY issue, not a data leakage issue.")

# Check: Feature columns used
feature_cols = [c for c in df.columns if c not in ['Target', 'Time', 'Future_Close']]
print(f"\n=== FEATURE COLUMNS ({len(feature_cols)}) ===")
for c in feature_cols:
    print(f"  {c}")

# Check if Future_Close exists
print(f"\n=== DANGEROUS COLUMNS CHECK ===")
print(f"'Future_Close' in columns: {'Future_Close' in df.columns}")
print(f"'Days_To_Next_Release' in columns: {'Days_To_Next_Release' in df.columns}")
print(f"'GARCH_Vol' in columns: {'GARCH_Vol' in df.columns}")
print(f"'HMM_State' in columns: {'HMM_State' in df.columns}")
