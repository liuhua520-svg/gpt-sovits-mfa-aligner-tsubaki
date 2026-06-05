@echo off
chcp 65001 >nul
setlocal

cd /d "%~dp0"

set "ENV_PY=%CD%\.mfa_env\python.exe"
set "PYTHONNOUSERSITE=1"

if not exist "%ENV_PY%" (
    echo Project env not found:
    echo %ENV_PY%
    echo Please run install_env.bat first.
    pause
    exit /b 1
)

echo ==============================
echo Using Python:
"%ENV_PY%" -c "import sys; print(sys.executable); print(sys.version)"
echo ==============================

"%ENV_PY%" backend\app.py
pause