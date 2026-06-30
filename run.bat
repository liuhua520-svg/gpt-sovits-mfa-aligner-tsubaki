@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

cd /d "%~dp0"

echo ================================================
echo    SVS Lab Aligner + Qwen3-ASR + NeMo-FA 服务启动器
echo ================================================

set "MFA_PY=%CD%\.mfa_env\python.exe"
set "QWEN_PY=%CD%\.qwen3_env\Scripts\python.exe"
set "NEMO_PY=%CD%\.nemo_env\python.exe"

if not exist "%MFA_PY%" (
    echo [错误] .mfa_env\python.exe 未找到！
    pause
    exit /b 1
)

REM 计算总步数（主后端 + 已安装的可选后端），用于步骤编号显示
set /a TOTAL_STEPS=1
set "HAS_QWEN=0"
set "HAS_NEMO=0"
if exist "%QWEN_PY%" (
    set "HAS_QWEN=1"
    set /a TOTAL_STEPS+=1
)
if exist "%NEMO_PY%" (
    set "HAS_NEMO=1"
    set /a TOTAL_STEPS+=1
)

set /a STEP=1
set "WAIT_NEEDED=0"

if "%HAS_QWEN%"=="1" (
    echo [!STEP!/%TOTAL_STEPS%] 启动 Qwen3-ASR 推理服务（端口 5001）...
    start "Qwen3-ASR 服务" "%QWEN_PY%" "%CD%\qwen3_server.py"
    set /a STEP+=1
    set "WAIT_NEEDED=1"
) else (
    echo [警告] .qwen3_env\Scripts\python.exe 未找到，将跳过 Qwen3-ASR（该后端不可用）
)

if "%HAS_NEMO%"=="1" (
    echo [!STEP!/%TOTAL_STEPS%] 启动 NeMo Forced Aligner 服务（端口 5002）...
    start "NeMo Forced Aligner 服务" "%NEMO_PY%" "%CD%\nemo_server.py"
    set /a STEP+=1
    set "WAIT_NEEDED=1"
) else (
    echo [警告] .nemo_env\python.exe 未找到，将跳过 NeMo Forced Aligner（该后端不可用）
)

if "%WAIT_NEEDED%"=="1" (
    echo [等待 5 秒让后台服务完全启动...]
    timeout /t 5 /nobreak >nul
)

echo [!STEP!/%TOTAL_STEPS%] 启动主后端服务（端口 5000）...
"%MFA_PY%" backend\app.py

pause