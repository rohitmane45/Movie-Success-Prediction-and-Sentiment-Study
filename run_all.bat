@echo off
title Movie Success Prediction - Full Pipeline
echo.
echo ======================================================================
echo        MOVIE SUCCESS PREDICTION ^& SENTIMENT STUDY
echo        One-Click Full Setup ^& Run
echo ======================================================================
echo.

:: Navigate to the script's directory
cd /d "%~dp0"

:: ── Step 1: Check Python ──
echo [1/6] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.10+ and add to PATH.
    pause
    exit /b 1
)
echo       OK

:: ── Step 2: Create virtual environment if needed ──
echo [2/6] Setting up virtual environment...
if not exist ".venv\Scripts\python.exe" (
    echo       Creating .venv...
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
)
echo       OK

:: ── Step 3: Install dependencies ──
echo [3/6] Installing dependencies...
.venv\Scripts\pip.exe install -r requirements.txt --quiet
echo       OK

:: ── Step 4: Download NLTK data ──
echo [4/6] Downloading NLTK data...
.venv\Scripts\python.exe -c "import nltk; nltk.download('vader_lexicon', quiet=True); nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True); nltk.download('stopwords', quiet=True)"
echo       OK

:: ── Step 5: Train models on real TMDB data ──
echo [5/6] Training ML models on TMDB 5000 dataset...
echo       (This may take 1-2 minutes on first run)
echo.
.venv\Scripts\python.exe src/main.py --tmdb
echo.

:: ── Step 6: Launch Dashboard + API simultaneously ──
echo [6/6] Launching Dashboard ^& API...
echo.
echo       Dashboard: http://localhost:8501
echo       REST API:  http://localhost:8000/docs
echo.

:: Start API in background
start "FastAPI Server" /min .venv\Scripts\python.exe src/main.py --api

:: Start Dashboard (foreground - keeps window open)
.venv\Scripts\python.exe src/main.py --dashboard

:: If dashboard is closed, also stop the API
taskkill /FI "WINDOWTITLE eq FastAPI Server" /F >nul 2>&1
echo.
echo All services stopped.
pause
