@echo off
title AI Traffic Monitor - AMD GPU
echoStarting AI Traffic Monitor...
echo ===================================================
echo  Monitor      : http://localhost:8501
echo  Stop         : Ctrl + C
echo ===================================================
echo.

:: Check if streamlit is installed
streamlit --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: 'streamlit' command not found.
    echo Please ensure Python and dependencies are installed.
    echo Try running: pip install -r requirements.txt
    pause
    exit /b
)

:: Run the app
streamlit run app.py

if %errorlevel% neq 0 (
    echo.
    echo Application crashed or stopped with error.
    pause
)
