@echo off
REM SVS Lab Aligner 完整一键安装脚本 (Windows)
chcp 65001 >nul
setlocal enabledelayedexpansion

cd /d "%~dp0"

set "ENV_PREFIX=%CD%\.mfa_env"
set "REQUIREMENTS_FILE=%CD%\backend\requirements.txt"

REM 定义MFA支持的语言
set "LANGUAGES=cmn eng jpn kor yue"
set "LANG_NAME_cmn=中文普通话"
set "LANG_NAME_eng=英语"
set "LANG_NAME_jpn=日语"
set "LANG_NAME_kor=韩语"
set "LANG_NAME_yue=粤语"

cls
echo.
echo ╔════════════════════════════════════════════════════════════════════════════╗
echo ║                                                                            ║
echo ║            🚀 SVS Lab Aligner 完整安装程序 (Windows)                       ║
echo ║                                                                            ║
echo ║  本脚本将自动完成以下步骤:                                                 ║
echo ║    ✓ 检查 Conda 和 Node.js 环境                                            ║
echo ║    ✓ 创建虚拟环境并根据 requirements.txt 安装所有依赖                      ║
echo ║    ✓ 安装并构建 Vue 前端                                                   ║
echo ║    ✓ 交互式选择语言模型并下载                                              ║
echo ║    ✓ (可选) 创建独立环境并安装 NeMo Forced Aligner                         ║
echo ║                                                                            ║
echo ║  预计耗时: 15-30 分钟 (取决于网络和模型大小)                               ║
echo ║                                                                            ║
echo ╚════════════════════════════════════════════════════════════════════════════╝
echo.
pause

REM ─────────────────────────────────────────────────────────────────
REM Step 1: 检查系统依赖 (Conda & Node)
REM ─────────────────────────────────────────────────────────────────
cls
echo.
echo ════════════════════════════════════════════════════════════════════════════
echo Step 1/6: 检查系统依赖
echo ════════════════════════════════════════════════════════════════════════════
echo.

REM 1.1 动态寻找 Conda
set "CONDA_BAT="
for %%p in (
    "%USERPROFILE%\miniconda3\condabin\conda.bat"
    "%USERPROFILE%\Anaconda3\condabin\conda.bat"
	"%USERPROFILE%\miniforge3\condabin\conda.bat"
    "%ALLUSERSPROFILE%\miniconda3\condabin\conda.bat"
    "%ALLUSERSPROFILE%\Anaconda3\condabin\conda.bat"
	"%ALLUSERSPROFILE%\miniforge3\condabin\conda.bat"
    "C:\ProgramData\Miniconda3\condabin\conda.bat"
    "C:\ProgramData\Anaconda3\condabin\conda.bat"
	"C:\ProgramData\Miniforge3\condabin\conda.bat"
	
) do (
    if exist "%%~p" (
        set "CONDA_BAT=%%~p"
        goto :conda_found
    )
)
REM 检查环境变量 PATH 中是否有 conda
for %%X in (conda.bat) do (set "CONDA_BAT=%%~$PATH:X")

:conda_found
if not defined CONDA_BAT (
    echo [✗] Conda 未找到！请确保已安装 Miniconda3 或 Miniforge。
    echo   Miniconda3 下载地址: https://docs.conda.io/projects/miniconda/en/latest/
	echo   Miniforge 下载地址: https://github.com/conda-forge/miniforge/releases
    pause
    exit /b 1
)
echo [✓] 发现 Conda: %CONDA_BAT%

REM 1.2 检查 Node.js
where node >nul 2>&1
if errorlevel 1 (
    echo [✗] Node.js 未找到！前端构建需要 Node.js 环境。
    echo   下载地址: https://nodejs.org/
    pause
    exit /b 1
)
echo [✓] 发现 Node.js

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
    ) else (
        echo [✓] 使用现有环境
        goto :skip_env_create
    )
)

echo 创建环境中... 请耐心等待（可能需要几分钟）...
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
REM Step 3: 安装依赖 (通过 requirements.txt)
REM ─────────────────────────────────────────────────────────────────
cls
echo.
echo ════════════════════════════════════════════════════════════════════════════
echo Step 3/6: 安装 Python 依赖
echo ════════════════════════════════════════════════════════════════════════════
echo.

if not exist "%REQUIREMENTS_FILE%" (
    echo [✗] 找不到 %REQUIREMENTS_FILE%
    pause
    exit /b 1
)

echo [*] 升级 pip/setuptools/wheel...
call "%CONDA_BAT%" run -p "%ENV_PREFIX%" python -m pip install --upgrade pip setuptools wheel >nul 2>&1

echo [*] 根据 requirements.txt 安装所有依赖 (包含 MFA 等核心包)...
echo   请耐心等待，这可能需要较长时间...
call "%CONDA_BAT%" run -p "%ENV_PREFIX%" python -m pip install -r "%REQUIREMENTS_FILE%"
if errorlevel 1 (
    echo [✗] 依赖安装失败，请检查上方报错信息。
    pause
    exit /b 1
)
echo [✓] 所有 Python 依赖已安装
echo.

REM ─────────────────────────────────────────────────────────────────
REM Step 4: 安装并构建前端
REM ─────────────────────────────────────────────────────────────────
cls
echo.
echo ════════════════════════════════════════════════════════════════════════════
echo Step 4/6: 安装并构建前端
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

echo [*] 构建前端应用...
call npm run build >nul 2>&1
if errorlevel 1 (
    echo [✗] 前端构建失败
    cd ..
    pause
    exit /b 1
)
echo [✓] 前端已构建
cd ..

REM ─────────────────────────────────────────────────────────────────
REM Step 5: 语言模型配置
REM ─────────────────────────────────────────────────────────────────
cls
echo.
echo ════════════════════════════════════════════════════════════════════════════
echo Step 5/6: 下载 MFA 语言模型
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
    echo [!] 未选择任何语言模型，可后续手动下载。
) else (
    echo [ℹ] 开始下载选定模型: !SELECTED_LANGS!
    echo.
    for %%L in (!SELECTED_LANGS!) do (
        echo [*] 下载 %%L 模型...
        call "%CONDA_BAT%" run -p "%ENV_PREFIX%" python -c "import sys; sys.path.insert(0, 'backend'); from mfa_utils import MFAChecker; success, msg = MFAChecker.download_model('%%L'); sys.exit(0 if success else 1)"
        if errorlevel 1 (
            echo   [!] %%L 模型下载失败，请检查网络后重试。
        ) else (
            echo   [✓] %%L 模型已下载
        )
    )
)

REM ─────────────────────────────────────────────────────────────────
REM Step 6: NeMo Forced Aligner 独立环境（可选）
REM ─────────────────────────────────────────────────────────────────
cls
echo.
echo ════════════════════════════════════════════════════════════════════════════
echo Step 6/6: NeMo Forced Aligner 独立环境 (可选)
echo ════════════════════════════════════════════════════════════════════════════
echo.
echo NeMo Forced Aligner 是一个可选的对齐后端 (NVIDIA CTC 强制对齐)。
echo 由于 nemo_toolkit 对 packaging/fsspec/omegaconf/hydra-core/lightning
echo 等核心依赖有严格版本限制，与主环境一起安装会产生依赖冲突，
echo 因此它需要运行在独立的环境里，作为一个本地服务 (端口 5002)
echo 供主后端通过 HTTP 调用。不安装也完全不影响 MFA / WhisperX / Qwen3 后端使用。
echo.

set "NEMO_ENV_PREFIX=%CD%\.nemo_env"
set /p "NEMO_CHOICE=是否现在创建独立环境并安装 NeMo Forced Aligner? (y/n): "

if /i "!NEMO_CHOICE!"=="y" (
    set "NEMO_NEED_CREATE=1"
    set "NEMO_CREATE_FAILED=0"

    if exist "%NEMO_ENV_PREFIX%" (
        echo [!] NeMo 环境已存在: %NEMO_ENV_PREFIX%
        set /p "NEMO_RECREATE=是否删除并重新创建? (y/n): "
        if /i "!NEMO_RECREATE!"=="y" (
            echo 正在删除旧 NeMo 环境...
            call "%CONDA_BAT%" env remove -y -p "%NEMO_ENV_PREFIX%" >nul 2>&1
        ) else (
            echo [✓] 使用现有 NeMo 环境
            set "NEMO_NEED_CREATE=0"
        )
    )

    if "!NEMO_NEED_CREATE!"=="1" (
        echo 创建 NeMo 独立环境中... 请耐心等待（可能需要几分钟）...
        call "%CONDA_BAT%" create -y -p "%NEMO_ENV_PREFIX%" -c conda-forge python=3.10 pip >nul 2>&1
        if errorlevel 1 (
            echo [✗] NeMo 环境创建失败，可稍后手动执行：
            echo     "%CONDA_BAT%" create -y -p "%NEMO_ENV_PREFIX%" -c conda-forge python=3.10 pip
            set "NEMO_CREATE_FAILED=1"
        )
    )

    if "!NEMO_CREATE_FAILED!"=="0" (
        echo [✓] NeMo 环境已准备
        echo [*] 在独立环境中安装 nemo_toolkit[asr]（体积较大，可能需要较长时间）...
        call "%CONDA_BAT%" run -p "%NEMO_ENV_PREFIX%" python -m pip install --upgrade pip setuptools wheel >nul 2>&1
        call "%CONDA_BAT%" run -p "%NEMO_ENV_PREFIX%" python -m pip install "nemo_toolkit[asr]>=2.7.0,<2.8.0" flask
        if errorlevel 1 (
            echo [✗] NeMo 依赖安装失败，可稍后手动执行：
            echo     "%CONDA_BAT%" run -p "%NEMO_ENV_PREFIX%" python -m pip install "nemo_toolkit[asr]>=2.7.0,<2.8.0" flask
        ) else (
            echo [✓] NeMo Forced Aligner 依赖已安装
            echo [ℹ] 首次启动 nemo_server.py 时会按所选语言自动下载模型权重（数百 MB ~ 1GB）
        )
    )
) else (
    echo [ℹ] 已跳过，可后续手动运行以下命令安装：
    echo     "%CONDA_BAT%" create -y -p "%CD%\.nemo_env" -c conda-forge python=3.10 pip
    echo     "%CONDA_BAT%" run -p "%CD%\.nemo_env" python -m pip install "nemo_toolkit[asr]>=2.7.0,<2.8.0" flask
)

REM ─────────────────────────────────────────────────────────────────
REM 结束
REM ─────────────────────────────────────────────────────────────────
cls
echo.
echo ╔════════════════════════════════════════════════════════════════════════════╗
echo ║                        ✅ 安装完全完成                                     ║
echo ╚════════════════════════════════════════════════════════════════════════════╝
echo.
echo 📍 下一步:
echo    双击运行 run.bat 即可启动应用程序。
echo.
pause
exit /b 0