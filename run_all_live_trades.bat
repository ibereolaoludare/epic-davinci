@echo off
title Live Trading Launcher (High-Carry Basket: USDJPY, GBPJPY, AUDUSD)

:: Set the project directory so it can be run from anywhere
set "PROJECT_DIR=C:\Users\ibere\documents\antigravity\epic-davinci"
cd /d "%PROJECT_DIR%"

echo Starting Live Trading for High-Carry Basket (USDJPY, GBPJPY, AUDUSD)...

:: Start each trading script in its own separate command window
start "Live Trading - USDJPY" cmd /k "python trade_live.py USDJPY --risk 0.02"
start "Live Trading - GBPJPY" cmd /k "python trade_live.py GBPJPY --risk 0.02"
start "Live Trading - AUDUSD" cmd /k "python trade_live.py AUDUSD --risk 0.02"

echo All 3 processes have been launched!
