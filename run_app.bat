@echo off
title Music Genre Classifier - GUI Application
color 0A

echo ========================================
echo   Music Genre Classification System
echo ========================================
echo.
echo Starting application...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.11 or later
    pause
    exit /b 1
)

echo [INFO] Using global Python environment
python --version

REM Check dependencies (optional - remove this if you don't want to check)
echo [INFO] Checking if required packages are installed...
python -c "import torch, librosa, customtkinter" >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Some required packages may be missing
    echo [INFO] Installing dependencies...
    pip install -q -r requirements.txt
)

echo.
echo ========================================
echo   Starting API Server (unified src/api.py)
echo ========================================
echo.

REM Start API server in background
start "Music Genre API" cmd /c "python -m uvicorn src.api:app --host 127.0.0.1 --port 8000"
timeout /t 3 /nobreak >nul

echo [INFO] API server started on http://127.0.0.1:8000
echo [INFO] API docs available at http://127.0.0.1:8000/docs
echo.

echo ========================================
echo   Starting GUI Application
echo ========================================
echo.
echo Close the GUI window to stop everything
echo.

REM Run the GUI application (blocks until closed)
python run_gui.py

REM Cleanup when GUI closes
echo.
echo ========================================
echo   Shutting down...
echo ========================================
echo.

REM Kill API server
echo [INFO] Stopping API server...
taskkill /F /FI "WINDOWTITLE eq Music Genre API*" >nul 2>&1

REM Kill any remaining related processes
taskkill /F /FI "WINDOWTITLE eq Music Genre Classifier*" >nul 2>&1
wmic process where "commandline like '%%uvicorn%%src.api:app%%' and name='python.exe'" delete >nul 2>&1

echo.
echo [INFO] Application closed successfully
echo.
pause
