@echo off
chcp 65001 >nul
setlocal

cd /d "%~dp0"

set "MFA_ENV_DIR=%CD%\.mfa_env"
set "ENV_PY=%CD%\.mfa_env\python.exe"
set "PYTHONNOUSERSITE=1"

if not exist "%ENV_PY%" (
    echo Project env not found:
    echo %ENV_PY%
    pause
    exit /b 1
)

"%ENV_PY%" backend\app.py
pause