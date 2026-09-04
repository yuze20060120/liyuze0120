@echo off
setlocal
title Predictive Maintenance System - Stopper
cd /d "%~dp0"

echo ============================================================
echo   Stopping the condition monitoring system services...
echo ============================================================
echo.

REM ---------- Kill processes on backend(8000) and frontend(8501) ----------
set "KILLED="
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8000 " ^| findstr "LISTENING"') do (
    taskkill /f /pid %%a >nul 2>nul && set "KILLED=1"
)
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8501 " ^| findstr "LISTENING"') do (
    taskkill /f /pid %%a >nul 2>nul && set "KILLED=1"
)

if defined KILLED (
    echo [DONE] System services stopped.
) else (
    echo [INFO] No running system services found.
)

echo.
echo Press any key to close this window...
pause >nul
endlocal
