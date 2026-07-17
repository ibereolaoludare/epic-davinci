@echo off
title Epic DaVinci Env Setup
echo =================================================
echo Setting up Python Virtual Environment (Python 3.12 CPU)
echo =================================================

:: Check if Python 3.12 is installed
py -3.12 -V >nul 2>&1
if errorlevel 1 (
    echo Python 3.12 was not found on this system.
    echo Downloading Python 3.12 Installer...
    
    :: Use PowerShell to download the official installer
    powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.12.4/python-3.12.4-amd64.exe' -OutFile 'py312.exe'"
    if errorlevel 1 (
        echo ERROR: Failed to download Python 3.12 installer.
        pause
        exit /b 1
    )
    
    echo Installing Python 3.12 silently...
    :: Start installer and wait for it to finish
    start /wait py312.exe /quiet PrependPath=1 AssociateFiles=1
    
    :: Clean up installer
    del py312.exe
    
    :: Re-check if installation was successful
    py -3.12 -V >nul 2>&1
    if errorlevel 1 (
        echo ERROR: Python 3.12 installation failed or is not recognized. Please restart the command prompt or install it manually.
        pause
        exit /b 1
    )
    echo Python 3.12 installed successfully!
) else (
    echo Python 3.12 is already installed.
)

:: Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment with Python 3.12...
    py -3.12 -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
) else (
    echo Virtual environment already exists. Skipping creation...
)

echo Installing dependencies from requirements.txt...
.\venv\Scripts\pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies.
    pause
    exit /b 1
)

echo =================================================
echo Setup complete! Virtual environment is ready.
echo =================================================
pause
