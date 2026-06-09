@echo off
REM GPT-SOVITS MFA Aligner 完整一键安装脚本 (Windows)
REM 功能: 自动创建MFA环境、安装所有依赖、构建前端、配置语言模型

chcp 65001 >nul
setlocal enabledelayedexpansion

cd /d "%~dp0"

set "VENV_DIR=%CD%\backend\venv"
set "CONDA_BAT=%USERPROFILE%\miniconda3\condabin\conda.bat"
set "ENV_PREFIX=%CD%\.mfa_env"

REM 定义MFA支持的语言
set "LANGUAGES=cmn eng jpn kor yue"
set "LANG_NAME_cmn=中文普通话"
set "LANG_NAME_eng=英语"
set "LANG_NAME_jpn=日语"
set "LANG_NAME_kor=韩语"
set "LANG_NAME_yue=粤语"

set "LANG_DEPS_cmn=pypinyin pycantonese"
set "LANG_DEPS_eng=soundfile"
set "LANG_DEPS_jpn=sudachipy sudachidict-core"
set "LANG_DEPS_kor=jamo"
set "LANG_DEPS_yue=pycantonese"

REM ─────────────────────────────────────────────────────────────────
REM 显示欢迎信息
REM ─────────────────────────────────────────────────────────────────

cls
echo.
echo ╔════════════════════════════════════════════════════════════════════════════╗
echo ║                                                                            ║
echo ║            🚀 GPT-SOVITS MFA Aligner 完整安装程序 (Windows)               ║
echo ║                                                                            ║
echo ║  本脚本将自动完成以下步骤:                                                ║
echo ║    ✓ 检查 Conda 环境                                                      ║
echo ║    ✓ 创建虚拟环境并安装 MFA 依赖                                         ║
echo ║    ✓ 安装 Flask 后端依赖                                                 ║
echo ║    ✓ 安装并构建 Vue 前端                                                 ║
echo ║    ✓ 交互式选择语言模型并下载                                            ║
echo ║                                                                            ║
echo ║  预计耗时: 15-30 分钟 (取决于网络和模型大小)                             ║
echo ║                                                                            ║
echo ╚════════════════════════════════════════════════════════════════════════════╝
echo.

pause

REM ─────────────────────────────────────────────────────────────────
REM Step 1: 检查 Conda
REM ─────────────────────────────────────────────────────────────────

cls
echo.
echo ════════════════════════════════════════════════════════════════════════════
echo Step 1/6: 检查 Conda 环境
echo ════════════════════════════════════════════════════════════════════════════
echo.

if not exist "%CONDA_BAT%" (
    echo [✗] Conda 未找到！
    echo 预期位置: %CONDA_BAT%
    echo.
    echo 请先安装 Miniconda3:
    echo   https://docs.conda.io/projects/miniconda/en/latest/
    echo.
    pause
    exit /b 1
)

echo [✓] Conda 已找到
echo.

REM ─────────────────────────────────────────────────────────────────
REM Step 2: 创建 Conda 环境
REM ─────────────────────────────────────────────────────────────────

cls
echo.
echo ════════════════════════════════════════════════════════════════════════════
echo Step 2/6: 创建 MFA 虚拟环境
echo ════════════════════════════════════════════════════════════════════════════
echo.
echo 环境位置: %ENV_PREFIX%
echo.

if exist "%ENV_PREFIX%" (
    echo [!] 发现已存在的环境
    set /p "CHOICE=是否删除并重新创建? (y/n): "
    if /i "!CHOICE!"=="y" (
        echo 正在删除旧环境...
        call "%CONDA_BAT%" env remove -y -p "%ENV_PREFIX%" >nul 2>&1
        timeout /t 2 /nobreak >nul
    ) else (
        echo [✓] 使用现有环境
        goto :skip_env_create
    )
)

echo 创建环境中... 请耐心等待（这可能需要几分钟）...
call "%CONDA_BAT%" create -y -p "%ENV_PREFIX%" -c conda-forge python=3.10 pip >nul 2>&1
if errorlevel 1 (
    echo [✗] 环境创建失败
    pause
    exit /b 1
)

:skip_env_create
echo [✓] MFA 环境已准备
echo.

REM ─────────────────────────────────────────────────────────────────
REM Step 3: 安装核心依赖
REM ─────────────────────────────────────────────────────────────────

cls
echo.
echo ════════════════════════════════════════════════════════════════════════════
echo Step 3/6: 安装核心依赖
echo ════════════════════════════════════════════════════════════════════════════
echo.

echo [*] 升级 pip/setuptools/wheel...
call "%CONDA_BAT%" run -p "%ENV_PREFIX%" python -m pip install --upgrade pip setuptools wheel >nul 2>&1
echo [✓] 已升级
echo.

echo [*] 安装 Montreal Forced Aligner (MFA)...
call "%CONDA_BAT%" run -p "%ENV_PREFIX%" python -m pip install montreal-forced-aligner kalpy >nul 2>&1
if errorlevel 1 (
    echo [✗] MFA/Kalpy 安装失败
    pause
    exit /b 1
)
echo [✓] MFA/Kalpy 已安装
echo.

echo [*] 安装 Flask 和后端依赖...
call "%CONDA_BAT%" run -p "%ENV_PREFIX%" python -m pip install flask flask-cors textgrid pypinyin pyyaml requests >nul 2>&1
if errorlevel 1 (
    echo [✗] Flask 和后端依赖安装失败
    pause
    exit /b 1
)
echo [✓] Flask 和后端依赖已安装
echo.

echo [*] 验证核心模块...
call "%CONDA_BAT%" run -p "%ENV_PREFIX%" python -c ^
    "import montreal_forced_aligner, _kalpy, flask, flask_cors, textgrid, pypinyin; print('验证成功')" >nul 2>&1
if errorlevel 1 (
    echo [✗] 核心模块验证失败
    pause
    exit /b 1
)
echo [✓] 所有核心模块已验证
echo.

REM ─────────────────────────────────────────────────────────────────
REM Step 4: 安装前端依赖
REM ─────────────────────────────────────────────────────────────────

cls
echo.
echo ════════════════════════════════════════════════════════════════════════════
echo Step 4/6: 安装前端依赖
echo ════════════════════════════════════════════════════════════════════════════
echo.

cd frontend

if not exist "package.json" (
    echo [✗] 未找到 frontend\package.json
    cd ..
    pause
    exit /b 1
)

echo [*] 安装 npm 包...
call npm install --legacy-peer-deps >nul 2>&1
if errorlevel 1 (
    echo [✗] npm 依赖安装失败
    cd ..
    pause
    exit /b 1
)
echo [✓] npm 依赖已安装
echo.

echo [*] 构建前端应用...
call npm run build >nul 2>&1
if errorlevel 1 (
    echo [✗] 前端构建失败
    cd ..
    pause
    exit /b 1
)
echo [✓] 前端已构建
echo.

cd ..

REM ─────────────────────────────────────────────────────────────────
REM Step 5: 语言模型配置
REM ─────────────────────────────────────────────────────────────────

cls
echo.
echo ════════════════════════════════════════════════════════════════════════════
echo Step 5/6: 语言模型配置
echo ════════════════════════════════════════════════════════════════════════════
echo.

echo 支持的语言:
echo   • cmn - %LANG_NAME_cmn%
echo   • eng - %LANG_NAME_eng%
echo   • jpn - %LANG_NAME_jpn%
echo   • kor - %LANG_NAME_kor%
echo   • yue - %LANG_NAME_yue%
echo.
echo 说明:
echo   输入 y     - 下载该语言的预训练模型
echo   输入 n     - 跳过该语言
echo   输入 all   - 下载所有剩余语言
echo.

set "INSTALL_ALL=false"
set "SELECTED_LANGS="

for %%L in (%LANGUAGES%) do (
    if "!INSTALL_ALL!"=="true" (
        set "CHOICE=y"
    ) else (
        set /p "CHOICE=下载 %%L - !LANG_NAME_%%L! ? (y/n/all): "
    )
    
    if /i "!CHOICE!"=="all" (
        echo [✓] 选择所有剩余语言
        set "INSTALL_ALL=true"
        set "SELECTED_LANGS=!SELECTED_LANGS! %%L"
    ) else if /i "!CHOICE!"=="y" (
        echo [✓] 选择 %%L
        set "SELECTED_LANGS=!SELECTED_LANGS! %%L"
    ) else (
        echo [!] 跳过 %%L
    )
)

echo.

if "!SELECTED_LANGS!"=="" (
    echo [!] 未选择任何语言模型
    echo [ℹ] 可以后续运行此脚本或在命令行手动下载
    echo.
) else (
    echo [ℹ] 选定语言: !SELECTED_LANGS!
    echo.
    
    REM 安装语言依赖和下载模型
    for %%L in (!SELECTED_LANGS!) do (
        echo [*] 处理语言: %%L
        
        REM 检查是否有额外依赖
        set "DEPS=!LANG_DEPS_%%L!"
        if not "!DEPS!"=="" (
            echo   • 安装 %%L 依赖: !DEPS!
            call "%CONDA_BAT%" run -p "%ENV_PREFIX%" python -m pip install !DEPS! >nul 2>&1
            if errorlevel 1 (
                echo   [!] %%L 依赖安装可能失败（继续）
            ) else (
                echo   [✓] %%L 依赖已安装
            )
        ) else (
            echo   [✓] %%L 无额外依赖
        )
        
        REM 下载模型
        echo   • 下载 %%L 预训练模型...
        call "%CONDA_BAT%" run -p "%ENV_PREFIX%" python << PYEOF
import sys
sys.path.insert(0, 'backend')
try:
    from mfa_utils import MFAChecker
    success, msg = MFAChecker.download_model('%%L')
    if success:
        print('[OK] %%L 模型已下载')
        sys.exit(0)
    else:
        print(f'[WARN] %%L 模型下载失败')
        sys.exit(1)
except Exception as e:
    print(f'[WARN] %%L 处理异常: {e}')
    sys.exit(1)
PYEOF
        if errorlevel 1 (
            echo   [!] %%L 模型下载可能失败（可稍后重试）
        ) else (
            echo   [✓] %%L 模型已下载
        )
    )
)

echo.

REM ─────────────────────────────────────────────────────────────────
REM Step 6: 完成
REM ─────────────────────────────────────────────────────────────────

cls
echo.
echo ════════════════════════════════════════════════════════════════════════════
echo Step 6/6: 安装完成！
echo ════════════════════════════════════════════════════════════════════════════
echo.

echo.
echo ╔════════════════════════════════════════════════════════════════════════════╗
echo ║                        ✅ 安装完成                                         ║
echo ╚════════════════════════════════════════════════════════════════════════════╝
echo.

echo 📍 下一步:
echo.
echo    启动应用程序:
echo    > run.bat
echo.
echo    或手动运行:
echo    > conda activate %ENV_PREFIX%
echo    > python backend/app.py
echo.

echo 📚 其他命令:
echo.
echo    开发模式启动:
echo    > run.bat --dev
echo.
echo    调试模式启动:
echo    > run.bat --debug
echo.
echo    重新运行安装:
echo    > setup.bat
echo.

echo 🔧 故障排除:
echo.
echo    激活虚拟环境:
echo    > conda activate %ENV_PREFIX%
echo.
echo    查看应用日志:
echo    > backend/logs/ 目录
echo.
echo    下载额外模型:
echo    > python backend/mfa_utils.py --download ko
echo.
echo    前端开发模式:
echo    > cd frontend
echo    > npm run dev
echo.

echo 📂 环境信息:
echo.
echo    虚拟环境: %ENV_PREFIX%
echo    Python 版本: 3.10
echo    MFA: 已安装
echo    Flask: 已安装
echo    前端: 已构建
echo.

echo ╔════════════════════════════════════════════════════════════════════════════╗
echo ║  ✨ 感谢使用 GPT-SOVITS MFA Aligner!                                       ║
echo ╚════════════════════════════════════════════════════════════════════════════╝
echo.

pause
exit /b 0
