@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo ================================================
echo SVS Lab Aligner + Qwen3-ASR 服务启动器 (独立 GUI 环境)
echo ================================================

set "MFA_PY=%CD%\.mfa_env\python.exe"
set "QWEN_PY=%CD%\.qwen3_env\Scripts\python.exe"
set "GUI_PY=%CD%\qt_env\Scripts\python.exe"

if not exist "%MFA_PY%" (
    echo [错误] .mfa_env 未找到！
    pause
    exit /b 1
)

if not exist "%GUI_PY%" (
    echo [错误] qt_env 未找到！请先执行上面的创建命令
    pause
    exit /b 1
)

echo [1/3] 启动 Qwen3-ASR 服务 (5001)...
start "Qwen3-ASR" "%QWEN_PY%" "%CD%\qwen3_server.py"

timeout /t 5 /nobreak >nul

echo [2/3] 启动主后端服务 (5000)...
start "SVS Backend" "%MFA_PY%" "%CD%\backend\app.py"

timeout /t 10 /nobreak >nul

echo [3/3] 启动桌面 UI (Qt Material)...
"%GUI_PY%" qt_desktop\main.py

echo.
echo 所有服务已启动，按任意键关闭窗口...
pause