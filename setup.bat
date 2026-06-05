@echo off
REM GPT-SOVITS MFA Aligner 一键安装脚本 (Windows)
chcp 65001 >nul

echo.
echo ===============================================
echo   GPT-SOVITS MFA Aligner 安装程序
echo ===============================================
echo.

echo [1/4] 检查Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ 未安装Python，请先安装 Python 3.8+
    echo 下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo ✓ Python已安装

echo [2/4] 检查Node.js...
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ 未安装Node.js，请先安装 Node.js 16+
    echo 下载地址: https://nodejs.org/
    pause
    exit /b 1
)

echo ✓ Node.js已安装

echo [3/4] 安装Python依赖...
cd /d "%~dp0backend"
call pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ❌ Python依赖安装失败
    pause
    exit /b 1
)
cd /d "%~dp0"

echo ✓ Python依赖安装完成

echo [4/4] 安装前端依赖...
cd /d "%~dp0frontend"
call npm install
if %errorlevel% neq 0 (
    echo ❌ 前端依赖安装失败
    pause
    exit /b 1
)

call npm run build
if %errorlevel% neq 0 (
    echo ❌ 前端构建失败
    pause
    exit /b 1
)
cd /d "%~dp0"

echo ✓ 前端依赖安装完成

echo.
echo ===============================================
echo   ✅ 安装完成！
echo ===============================================
echo.
echo 下一步: 运行 run.bat 启动应用程序
echo.
pause