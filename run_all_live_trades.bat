@echo off
title Live Trading Launcher

:: Set the project directory so it can be run from anywhere
set "PROJECT_DIR=C:\Users\ibere\documents\antigravity\epic-davinci"
cd /d "%PROJECT_DIR%"

echo Starting Live Trading for all pairs (USDJPY, EURUSD, GBPUSD)...

:: Start each trading script in its own separate command window
start "Live Trading - USDJPY" cmd /k "%ACTIVATE_CMD%python trade_live.py USDJPY --risk 0.02"
start "Live Trading - EURUSD" cmd /k "%ACTIVATE_CMD%python trade_live.py EURUSD --risk 0.02"
start "Live Trading - GBPUSD" cmd /k "%ACTIVATE_CMD%python trade_live.py GBPUSD --risk 0.02"

echo Processes have been launched!
