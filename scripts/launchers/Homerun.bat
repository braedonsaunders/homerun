@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
if not defined WT_SESSION mode con cols=140 lines=45 >nul 2>nul
powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%..\infra\run.ps1" %*
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo Homerun exited with error code %EXIT_CODE%.
    echo Press any key to close this window.
    pause >nul
)

exit /b %EXIT_CODE%
