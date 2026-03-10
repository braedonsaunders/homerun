@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "RUN_SCRIPT=%SCRIPT_DIR%..\infra\run.ps1"
set "PS_EXE=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
if not exist "%PS_EXE%" set "PS_EXE=powershell"

if defined WT_SESSION if /I not "%HOMERUN_CONHOST_RELAUNCHED%"=="1" (
    set "HOMERUN_CONHOST_RELAUNCHED=1"
    start "Homerun" "%PS_EXE%" -NoProfile -ExecutionPolicy Bypass -File "%RUN_SCRIPT%" %*
    exit /b 0
)

if not defined WT_SESSION mode con cols=140 lines=45 >nul 2>nul
"%PS_EXE%" -NoProfile -ExecutionPolicy Bypass -File "%RUN_SCRIPT%" %*
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo Homerun exited with error code %EXIT_CODE%.
    echo Press any key to close this window.
    pause >nul
)

exit /b %EXIT_CODE%
