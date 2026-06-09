#!/bin/bash
# GPT-SOVITS MFA Aligner 完整一键安装脚本 (Linux/Mac)
# 功能: 自动创建MFA环境、安装所有依赖、构建前端、配置语言模型

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# 脚本配置
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/backend/venv"
PYTHON_MIN_VERSION="3.8"
NODE_MIN_VERSION="16"

# MFA语言配置
declare -A LANGUAGE_MODELS=(
    ["cmn"]="中文普通话"
    ["eng"]="英语"
    ["jpn"]="日语"
    ["kor"]="韩语"
    ["yue"]="粤语"
)

declare -A LANGUAGE_DEPS=(
    ["cmn"]="pypinyin pycantonese"
    ["eng"]="soundfile"
    ["jpn"]="sudachipy sudachidict-core"
    ["kor"]="jamo"
    ["yue"]="pycantonese"
)

# 日志函数
log_section() {
    echo ""
    echo -e "${MAGENTA}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${MAGENTA}║${NC} $1"
    echo -e "${MAGENTA}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

log_step() {
    echo -e "${BLUE}➜${NC} $1"
}

log_ok() {
    echo -e "${GREEN}✓${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}!${NC} $1"
}

log_error() {
    echo -e "${RED}✗${NC} $1"
}

log_info() {
    echo -e "${CYAN}ℹ${NC} $1"
}

# 版本比较
version_ge() {
    [ "$(printf '%s\n' "$1" "$2" | sort -V | head -n 1)" = "$2" ]
}

# 命令检查
command_exists() {
    command -v "$1" &> /dev/null
}

# 用户确认
confirm() {
    local prompt="$1"
    local response
    while true; do
        read -p "$(echo -e "${BLUE}➜${NC} $prompt [y/n/all]: ")" -r response
        case "$response" in
            [yY]) return 0 ;;
            [nN]) return 1 ;;
            [aA][lL][lL]) return 2 ;;
            *) log_warn "请输入 y, n 或 all" ;;
        esac
    done
}

# ============================================================================
# 主程序
# ============================================================================

clear
cat << "EOF"

╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║            🚀 GPT-SOVITS MFA Aligner 完整安装程序                            ║
║                                                                              ║
║  本脚本将自动完成以下步骤:                                                  ║
║    ✓ 检查系统依赖 (Python, Node.js)                                        ║
║    ✓ 创建虚拟环境并安装 MFA 相关依赖                                       ║
║    ✓ 安装 Flask 后端依赖                                                   ║
║    ✓ 安装并构建 Vue 前端                                                   ║
║    ✓ 交互式选择语言模型并下载                                              ║
║                                                                              ║
║  预计耗时: 15-30 分钟 (取决于网络和模型大小)                               ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

EOF

read -p "按 Enter 键开始安装..." -r

# ─────────────────────────────────────────────────────────────────────────
# Step 1: 检查系统依赖
# ─────────────────────────────────────────────────────────────────────────

log_section "Step 1: 检查系统依赖"

log_step "检查 Python3..."
if ! command_exists python3; then
    log_error "未安装 Python3"
    log_info "请访问: https://www.python.org/downloads/"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
if ! version_ge "$PYTHON_VERSION" "$PYTHON_MIN_VERSION"; then
    log_error "Python 版本过低 (当前: $PYTHON_VERSION, 需要: ≥$PYTHON_MIN_VERSION)"
    exit 1
fi
log_ok "Python $PYTHON_VERSION"

log_step "检查 pip3..."
if ! command_exists pip3; then
    log_error "未安装 pip3"
    exit 1
fi
log_ok "pip $(pip3 --version | awk '{print $2}')"

log_step "检查 Node.js..."
if ! command_exists node; then
    log_error "未安装 Node.js"
    log_info "请访问: https://nodejs.org/"
    exit 1
fi

NODE_VERSION=$(node --version | sed 's/^v//')
if ! version_ge "$NODE_VERSION" "$NODE_MIN_VERSION"; then
    log_warn "Node.js 版本较低 (当前: $NODE_VERSION, 推荐: ≥$NODE_MIN_VERSION)"
fi
log_ok "Node.js $NODE_VERSION"
log_ok "npm $(npm --version)"

if command_exists git; then
    log_ok "git $(git --version | awk '{print $3}')"
else
    log_info "git 未安装 (可选，仅用于版本控制)"
fi

# ─────────────────────────────────────────────────────────────────────────
# Step 2: 创建虚拟环境
# ─────────────────────────────────────────────────────────────────────────

log_section "Step 2: 创建 Python 虚拟环境"

if [ -d "$VENV_DIR" ]; then
    log_warn "虚拟环境已存在: $VENV_DIR"
    read -p "是否删除并重新创建? [y/n]: " -r response
    if [[ "$response" =~ ^[yY]$ ]]; then
        log_step "删除旧虚拟环境..."
        rm -rf "$VENV_DIR"
        log_ok "已删除"
    else
        log_info "使用现有虚拟环境"
        goto_skip_venv=1
    fi
fi

if [ -z "$goto_skip_venv" ]; then
    log_step "创建虚拟环境..."
    python3 -m venv "$VENV_DIR"
    log_ok "虚拟环境已创建"
fi

log_step "激活虚拟环境..."
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
log_ok "虚拟环境已激活"

# ─────────────────────────────────────────────────────────────────────────
# Step 3: 安装 MFA 核心依赖
# ─────────────────────────────────────────────────────────────────────────

log_section "Step 3: 安装 MFA 核心依赖"

log_step "升级 pip/setuptools/wheel..."
pip install --upgrade pip setuptools wheel > /dev/null 2>&1
log_ok "已升级"

log_step "安装 Montreal Forced Aligner (MFA)..."
pip install montreal-forced-aligner kalpy > /dev/null 2>&1
if [ $? -ne 0 ]; then
    log_error "MFA/Kalpy 安装失败"
    exit 1
fi
log_ok "MFA/Kalpy 已安装"

log_step "安装 Flask 和后端依赖..."
pip install flask flask-cors textgrid pypinyin pyyaml requests > /dev/null 2>&1
if [ $? -ne 0 ]; then
    log_error "Flask 和后端依赖安装失败"
    exit 1
fi
log_ok "Flask 和后端依赖已安装"

log_step "验证核心模块..."
if ! python3 -c "import montreal_forced_aligner, _kalpy, flask, flask_cors, textgrid, pypinyin" 2>/dev/null; then
    log_error "核心模块验证失败"
    exit 1
fi
log_ok "所有核心模块已验证"

# ─────────────────────────────────────────────────────────────────────────
# Step 4: 安装前端依赖
# ─────────────────────────────────────────────────────────────────────────

log_section "Step 4: 安装前端依赖"

cd "$SCRIPT_DIR/frontend" || exit 1

if [ ! -f "package.json" ]; then
    log_error "未找到 frontend/package.json"
    exit 1
fi

log_step "安装 npm 包..."
npm install --legacy-peer-deps > /dev/null 2>&1
if [ $? -ne 0 ]; then
    log_error "npm 依赖安装失败"
    exit 1
fi
log_ok "npm 依赖已安装"

log_step "构建前端应用..."
npm run build > /dev/null 2>&1
if [ $? -ne 0 ]; then
    log_error "前端构建失败"
    exit 1
fi
log_ok "前端已构建"

cd "$SCRIPT_DIR" || exit 1

# ─────────────────────────────────────────────────────────────────────────
# Step 5: 语言模型配置
# ─────────────────────────────────────────────────────────────────────────

log_section "Step 5: 语言模型配置"

log_info "支持的语言:"
echo ""
for lang in cmn eng jpn kor yue; do
    lang_name="${LANGUAGE_MODELS[$lang]}"
    printf "  %-8s - %s\n" "$lang" "$lang_name"
done
echo ""

log_info "说明:"
echo "  • 输入 y     - 下载该语言的预训练模型"
echo "  • 输入 n     - 跳过该语言"
echo "  • 输入 all   - 下载所有剩余语言"
echo ""

source "$VENV_DIR/bin/activate" 2>/dev/null

INSTALL_ALL=false
SELECTED_LANGS=()

for lang in cmn eng jpn kor yue; do
    lang_name="${LANGUAGE_MODELS[$lang]}"
    
    if [ "$INSTALL_ALL" = true ]; then
        response=0
    else
        confirm "下载 $lang ($lang_name) 的预训练模型?"
        response=$?
    fi
    
    case $response in
        0) # yes
            log_ok "选择 $lang"
            SELECTED_LANGS+=("$lang")
            ;;
        2) # all
            log_ok "选择所有剩余语言"
            INSTALL_ALL=true
            SELECTED_LANGS+=("$lang")
            ;;
        1) # no
            log_warn "跳过 $lang"
            ;;
    esac
done

echo ""

if [ ${#SELECTED_LANGS[@]} -eq 0 ]; then
    log_warn "未选择任何语言模型"
    log_info "可以后续运行此脚本再次选择，或手动下载"
else
    log_section "Step 5.1: 安装语言特定依赖"
    
    for lang in "${SELECTED_LANGS[@]}"; do
        lang_name="${LANGUAGE_MODELS[$lang]}"
        deps="${LANGUAGE_DEPS[$lang]}"
        
        if [ -n "$deps" ]; then
            log_step "安装 $lang ($lang_name) 依赖: $deps"
            # shellcheck disable=SC2086
            pip install $deps > /dev/null 2>&1
            if [ $? -eq 0 ]; then
                log_ok "$lang 依赖已安装"
            else
                log_warn "$lang 依赖安装失败，继续..."
            fi
        else
            log_ok "$lang 无额外依赖"
        fi
    done
    
    log_section "Step 5.2: 下载 MFA 预训练模型"
    
    for lang in "${SELECTED_LANGS[@]}"; do
        lang_name="${LANGUAGE_MODELS[$lang]}"
        
        log_step "下载 $lang ($lang_name) 预训练模型..."
        log_info "这可能需要几分钟，请耐心等待..."
        
        python3 << PYEOF
import sys
sys.path.insert(0, '$SCRIPT_DIR/backend')
try:
    from mfa_utils import MFAChecker
    success, message = MFAChecker.download_model('$lang')
    if success:
        print(f'[OK] $lang 模型已下载')
        sys.exit(0)
    else:
        print(f'[WARN] $lang 模型下载失败: {message}')
        sys.exit(1)
except Exception as e:
    print(f'[WARN] $lang 处理异常: {e}')
    sys.exit(1)
PYEOF
        
        if [ $? -eq 0 ]; then
            log_ok "$lang 模型已下载"
        else
            log_warn "$lang 模型下载失败（可稍后重试）"
        fi
    done
fi

# ─────────────────────────────────────────────────────────────────────────
# 完成
# ─────────────────────────────────────────────────────────────────────────

log_section "✅ 安装完成"

cat << EOF

${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}

📍 下一步:

   启动应用程序:
   ${CYAN}./run.sh${NC}

   或手动运行:
   ${CYAN}source backend/venv/bin/activate${NC}
   ${CYAN}python backend/app.py${NC}

📚 其他命令:

   开发模式启动:
   ${CYAN}./run.sh --dev${NC}

   调试模式启动:
   ${CYAN}./run.sh --debug${NC}

   重新运行安装:
   ${CYAN}./setup.sh${NC}

🔧 故障排除:

   查看日志:
   ${CYAN}tail -f backend/logs/app.log${NC}

   激活虚拟环境:
   ${CYAN}source backend/venv/bin/activate${NC}

   下载额外模型:
   ${CYAN}python backend/mfa_utils.py --download ko${NC}

   前端开发模式:
   ${CYAN}cd frontend && npm run dev${NC}

${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}

📂 项目结构:

   ${SCRIPT_DIR}
   ├── backend/
   │   ├── venv/                    ← 虚拟环境
   │   ├── app.py                   ← 主应用
   │   ├── mfa_processor.py          ← MFA处理器
   │   ├── mfa_utils.py             ← MFA工具
   │   ├── phoneme_converter.py      ← 音素转换
   │   └── logs/
   ├── frontend/
   │   ├── dist/                    ← 构建输出
   │   ├── src/
   │   ├── package.json
   │   └── ...
   └── run.sh                        ← 启动脚本

✨ 感谢使用 GPT-SOVITS MFA Aligner!

EOF

echo ""
