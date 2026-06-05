@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

cd /d "%~dp0"

set "CONDA_BAT=%USERPROFILE%\miniconda3\condabin\conda.bat"
set "ENV_PREFIX=%CD%\.mfa_env"

if not exist "%CONDA_BAT%" (
    echo Conda cannot be found:
    echo %CONDA_BAT%
    pause
    exit /b 1
)

echo [1/5] Creating project env at %ENV_PREFIX%
call "%CONDA_BAT%" create -y -p "%ENV_PREFIX%" -c conda-forge python=3.10 pip montreal-forced-aligner kalpy
if errorlevel 1 (
    echo Environment creation failed.
    pause
    exit /b 1
)

echo [2/5] Installing Flask dependencies into the same env
call "%CONDA_BAT%" run -p "%ENV_PREFIX%" python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo pip upgrade failed.
    pause
    exit /b 1
)

call "%CONDA_BAT%" run -p "%ENV_PREFIX%" python -m pip install flask flask-cors textgrid
if errorlevel 1 (
    echo Flask install failed.
    pause
    exit /b 1
)

echo [3/5] Verifying MFA / _kalpy / Flask
call "%CONDA_BAT%" run -p "%ENV_PREFIX%" python -c "import montreal_forced_aligner, _kalpy, flask; print('ENV OK')"
if errorlevel 1 (
    echo Verification failed.
    pause
    exit /b 1
)

echo [4/5] Checking MFA version
call "%CONDA_BAT%" run -p "%ENV_PREFIX%" mfa version
if errorlevel 1 (
    echo MFA version check failed.
    pause
    exit /b 1
)

echo [5/5] Done.
pause