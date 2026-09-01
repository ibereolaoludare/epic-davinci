@echo off
title Epic DaVinci V7 — Master Live Portfolio Daemon

cd /d "%~dp0"

:: Auto-detect Python interpreter
set "PY_EXE=python"
if exist "%~dp0venv\Scripts\python.exe" (
    set "PY_EXE=%~dp0venv\Scripts\python.exe"
) else if exist "%~dp0.venv\Scripts\python.exe" (
    set "PY_EXE=%~dp0.venv\Scripts\python.exe"
)

echo Using Python: %PY_EXE%
echo Checking dependencies...

"%PY_EXE%" -c "import pandas, catboost, MetaTrader5" >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Required dependencies missing. Installing from requirements.txt...
    if exist "%~dp0venv\Scripts\pip.exe" (
        "%~dp0venv\Scripts\pip.exe" install -r requirements.txt
    ) else (
        "%PY_EXE%" -m pip install -r requirements.txt
    )
)

echo Starting Master Live Daemon for High-Carry Basket (USDJPY, GBPJPY, AUDUSD)...
"%PY_EXE%" trade_all_live.py --risk 0.02
pause
