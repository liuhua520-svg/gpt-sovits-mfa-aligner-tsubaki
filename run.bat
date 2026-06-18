@echo off
chcp 65001 >nul
setlocal

cd /d "%~dp0"

echo ================================================
echo    SVS Lab Aligner + Qwen3-ASR 服务启动器
echo ================================================

set "MFA_PY=%CD%\.mfa_env\python.exe"
set "QWEN_PY=%CD%\.qwen3_env\Scripts\python.exe"

if not exist "%MFA_PY%" (
    echo [错误] .mfa_env\python.exe 未找到！
    pause
    exit /b 1
)

if not exist "%QWEN_PY%" (
    echo [警告] .qwen3_env\Scripts\python.exe 未找到，将只启动主后端（Qwen3-ASR 不可用）
    goto :start_main
)

echo [1/2] 启动 Qwen3-ASR 推理服务（端口 5001）...
start "Qwen3-ASR 服务" "%QWEN_PY%" "%CD%\qwen3_server.py"

echo [等待 15 秒让服务完全启动...]
timeout /t 15 /nobreak >nul

:start_main
echo [2/2] 启动主后端服务（端口 5000）...
"%MFA_PY%" backend\app.py

pause