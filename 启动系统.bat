@echo off
setlocal enabledelayedexpansion
title Predictive Maintenance System - Launcher
cd /d "%~dp0"

echo ============================================================
echo   Vibe Coding based Machine Condition Monitoring System
echo   Course: Intelligent Manufacturing Technology
echo ============================================================
echo.

REM ---------- check mode: environment only, no service start ----------
if /i "%~1"=="check" goto :check

REM ================= 1. Locate a working Python =================
set "PYEXE="

REM 1a) common python install dirs
for /d %%d in ("%USERPROFILE%\AppData\Roaming\uv\python\cpython-3.1*") do (
    if not defined PYEXE if exist "%%d\python.exe" set "PYEXE=%%d\python.exe"
)
for /d %%d in ("%LOCALAPPDATA%\Programs\Python\Python3*") do (
    if not defined PYEXE if exist "%%d\python.exe" set "PYEXE=%%d\python.exe"
)
for /d %%d in ("C:\Python3*") do (
    if not defined PYEXE if exist "%%d\python.exe" set "PYEXE=%%d\python.exe"
)
for /d %%d in ("%USERPROFILE%\anaconda3") do (
    if not defined PYEXE if exist "%%d\python.exe" set "PYEXE=%%d\python.exe"
)
for /d %%d in ("%LOCALAPPDATA%\miniconda3") do (
    if not defined PYEXE if exist "%%d\python.exe" set "PYEXE=%%d\python.exe"
)

REM 1b) fallback: `where python`, but verify it really runs (skip WindowsApps stub)
if not defined PYEXE (
    for /f "delims=" %%i in ('where python 2^>nul') do (
        if not defined PYEXE (
            "%%i" --version >nul 2>nul
            if not errorlevel 1 set "PYEXE=%%i"
        )
    )
)

if not defined PYEXE (
    echo [ERROR] No usable Python found.
    echo         Install Python 3.10+ from https://www.python.org/downloads/
    echo         and check "Add python.exe to PATH".
    goto :end
)
echo [1/6] Python OK: !PYEXE!

REM ================= 2. Virtual environment =================
set "VPY=.venv\Scripts\python.exe"
if not exist "!VPY!" (
    echo [2/6] First run: creating virtual environment...
    "!PYEXE!" -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        goto :end
    )
)
echo [2/6] Virtual environment OK

REM ================= 3. Dependencies (inside venv) =================
"!VPY!" -c "import fastapi, streamlit, torch, xgboost, sklearn, pandas, numpy" >nul 2>nul
if errorlevel 1 (
    echo [3/6] Installing dependencies into .venv, first run takes a few minutes...
    "!VPY!" -m pip install --default-timeout 120 -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
    if errorlevel 1 (
        echo [RETRY] Mirror install failed, retry with default source...
        "!VPY!" -m pip install --default-timeout 120 -r requirements.txt
        if errorlevel 1 (
            echo [ERROR] Dependency install failed. Check network and retry.
            goto :end
        )
    )
)
echo [3/6] Dependencies OK

REM ================= 4. Models (auto-train if missing) =================
if not exist "data\models\random_forest_diagnoser.joblib" (
    echo [4/6] Trained models not found, training now...
    "!VPY!" scripts\train_models.py
    if errorlevel 1 (
        echo [ERROR] Model training failed.
        goto :end
    )
)
echo [4/6] Models OK

REM ================= 5. Database =================
"!VPY!" scripts\init_db.py >nul
if errorlevel 1 (
    echo [ERROR] Database init failed.
    goto :end
)
echo [5/6] Database OK

REM ================= 6. Detect which services are already up =================
set "BACKEND_UP="
set "FRONTEND_UP="
netstat -ano | findstr ":8000 " | findstr "LISTENING" >nul && set "BACKEND_UP=1"
netstat -ano | findstr ":8501 " | findstr "LISTENING" >nul && set "FRONTEND_UP=1"

if defined BACKEND_UP goto :frontend_check
echo [7/7] Backend :8000 not running, starting...
start "PM-Backend" /min cmd /k "cd /d %~dp0 && .venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000"
echo         waiting for backend (model loading may take a few seconds)...
ping -n 8 127.0.0.1 >nul
:frontend_check
if defined FRONTEND_UP goto :open
echo [7/7] Frontend :8501 not running, starting...
start "PM-Frontend" /min cmd /k "cd /d %~dp0 && .venv\Scripts\python.exe -m streamlit run frontend/app.py --server.port 8501 --server.headless true"
echo         waiting for frontend...
ping -n 6 127.0.0.1 >nul

:open
echo Opening dashboard...
start "" http://localhost:8501
echo.
echo ============================================================
echo   System is running!
echo   Dashboard:   http://localhost:8501
echo   API Docs:    http://localhost:8000/docs
echo   To stop:     run "stop.bat"
echo ============================================================
goto :done

REM ---------- check mode ----------
:check
echo [check mode] Verifying environment only (no service start)...
set "PYEXE="
for /d %%d in ("%USERPROFILE%\AppData\Roaming\uv\python\cpython-3.1*") do (
    if not defined PYEXE if exist "%%d\python.exe" set "PYEXE=%%d\python.exe"
)
if not defined PYEXE (
    for /f "delims=" %%i in ('where python 2^>nul') do (
        if not defined PYEXE (
            "%%i" --version >nul 2>nul
            if not errorlevel 1 set "PYEXE=%%i"
        )
    )
)
if not defined PYEXE (
    echo [ERROR] No usable Python found.
    goto :end
)
echo [OK] Python: !PYEXE!
if not exist ".venv\Scripts\python.exe" (
    echo [INFO] Virtual env missing. Double-click this file to set up automatically.
    goto :end
)
echo [OK] Virtual environment
".venv\Scripts\python.exe" -c "import fastapi, streamlit, torch, xgboost, sklearn, pandas, numpy" >nul 2>nul
if errorlevel 1 (
    echo [INFO] Dependencies missing in .venv. Double-click this file to install them.
    goto :end
)
echo [OK] Dependencies
if exist "data\models\random_forest_diagnoser.joblib" (
    echo [OK] Models ready
) else (
    echo [INFO] Models missing, will auto-train on normal start
)
echo [OK] Environment check passed.
goto :done

:done
endlocal
exit /b 0

:end
echo.
echo Press any key to close this window...
pause >nul
endlocal
