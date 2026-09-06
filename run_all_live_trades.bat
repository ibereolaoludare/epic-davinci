@echo off
title Live Trading Launcher (High-Carry Basket: USDJPY, GBPJPY, AUDUSD)

:: Navigate to script directory regardless of drive or user path
cd /d "%~dp0"

:: Auto-detect Python interpreter (prefer local virtual environment)
set "PY_EXE=python"
if exist "%~dp0venv\Scripts\python.exe" (
    set "PY_EXE=%~dp0venv\Scripts\python.exe"
) else if exist "%~dp0.venv\Scripts\python.exe" (
    set "PY_EXE=%~dp0.venv\Scripts\python.exe"
)

echo Using Python: %PY_EXE%
echo Checking dependencies...

:: Verify pandas is installed; install requirements if missing
"%PY_EXE%" -c "import pandas, catboost, MetaTrader5" >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Required dependencies missing. Installing from requirements.txt...
    if exist "%~dp0venv\Scripts\pip.exe" (
        "%~dp0venv\Scripts\pip.exe" install -r requirements.txt
    ) else (
        "%PY_EXE%" -m pip install -r requirements.txt
    )
)

echo Starting Live Trading for High-Carry Basket (USDJPY, GBPJPY, AUDUSD)...

:: Start each trading script in its own separate command window
start "Live Trading - USDJPY" cmd /k ""%PY_EXE%" trade_live.py USDJPY --risk 0.02"
start "Live Trading - GBPJPY" cmd /k ""%PY_EXE%" trade_live.py GBPJPY --risk 0.02"
start "Live Trading - AUDUSD" cmd /k ""%PY_EXE%" trade_live.py AUDUSD --risk 0.02"
start "Live Trading - XAUUSD" cmd /k ""%PY_EXE%" trade_live.py XAUUSD --risk 0.02"

echo All 4 processes have been launched!
