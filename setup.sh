#!/bin/bash
# SVS Lab Aligner 完整一键安装脚本 (Linux/Mac)

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
REQ_FILE="$SCRIPT_DIR/backend/requirements.txt"
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

# 日志函数
log_section() {
    echo ""
    echo -e "${MAGENTA}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${MAGENTA}║${NC} $1"
    echo -e "${MAGENTA}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

log_step() { echo -e "${BLUE}➜${NC} $1"; }
log_ok() { echo -e "${GREEN}✓${NC} $1"; }
log_warn() { echo -e "${YELLOW}!${NC} $1"; }
log_error() { echo -e "${RED}✗${NC} $1"; }
log_info() { echo -e "${CYAN}ℹ${NC} $1"; }

version_ge() {
    [ "$(printf '%s\n' "$1" "$2" | sort -V | head -n 1)" = "$2" ]
}

command_exists() {
    command -v "$1" &> /dev/null
}

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

clear
cat << "EOF"

╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║            🚀 SVS Lab Aligner 完整安装程序 (Linux/Mac)                       ║
║                                                                              ║
║  本脚本将自动完成以下步骤:                                                   ║
║    ✓ 检查系统依赖 (Python, Node.js)                                          ║
║    ✓ 创建虚拟环境并安装 requirements.txt 依赖                                ║
║    ✓ 安装并构建 Vue 前端                                                     ║
║    ✓ 交互式选择语言模型并下载                                                ║
║                                                                              ║
║  预计耗时: 15-30 分钟 (取决于网络和模型大小)                                 ║
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
    log_error "未安装 Python3。请先安装 Python 3.8+"
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
    log_error "未安装 Node.js，前端构建需要 npm"
    exit 1
fi

NODE_VERSION=$(node --version | sed 's/^v//')
if ! version_ge "$NODE_VERSION" "$NODE_MIN_VERSION"; then
    log_warn "Node.js 版本较低 (当前: $NODE_VERSION, 推荐: ≥$NODE_MIN_VERSION)"
fi
log_ok "Node.js $NODE_VERSION"
log_ok "npm $(npm --version)"

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
source "$VENV_DIR/bin/activate"
log_ok "虚拟环境已激活"

# ─────────────────────────────────────────────────────────────────────────
# Step 3: 安装所有 Python 依赖
# ─────────────────────────────────────────────────────────────────────────
log_section "Step 3: 安装 Python 依赖"

if [ ! -f "$REQ_FILE" ]; then
    log_error "找不到 requirements.txt 文件: $REQ_FILE"
    exit 1
fi

log_step "升级 pip/setuptools/wheel..."
pip install --upgrade pip setuptools wheel > /dev/null 2>&1
log_ok "已升级"

log_step "根据 requirements.txt 安装核心依赖..."
log_info "这可能需要几分钟，取决于您的网络环境..."

if pip install -r "$REQ_FILE"; then
    log_ok "所有 Python 依赖已成功安装"
else
    log_error "依赖安装失败，请查看上方的报错信息"
    exit 1
fi

# ─────────────────────────────────────────────────────────────────────────
# Step 4: 安装前端依赖并构建
# ─────────────────────────────────────────────────────────────────────────
log_section "Step 4: 安装并构建前端"

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
log_section "Step 5: 下载 MFA 语言模型"

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
        0)
            log_ok "选择 $lang"
            SELECTED_LANGS+=("$lang")
            ;;
        2)
            log_ok "选择所有剩余语言"
            INSTALL_ALL=true
            SELECTED_LANGS+=("$lang")
            ;;
        1)
            log_warn "跳过 $lang"
            ;;
    esac
done

echo ""

if [ ${#SELECTED_LANGS[@]} -eq 0 ]; then
    log_warn "未选择任何语言模型"
    log_info "可后续手动下载。"
else
    for lang in "${SELECTED_LANGS[@]}"; do
        lang_name="${LANGUAGE_MODELS[$lang]}"
        log_step "下载 $lang ($lang_name) 预训练模型..."
        
        python3 -c "
import sys
sys.path.insert(0, '$SCRIPT_DIR/backend')
try:
    from mfa_utils import MFAChecker
    success, msg = MFAChecker.download_model('$lang')
    sys.exit(0 if success else 1)
except Exception:
    sys.exit(1)
"
        if [ $? -eq 0 ]; then
            log_ok "$lang 模型已下载"
        else
            log_warn "$lang 模型下载失败（请检查网络后重试）"
        fi
    done
fi

# ─────────────────────────────────────────────────────────────────────────
# Step 6: NeMo Forced Aligner 独立环境（可选）
# ─────────────────────────────────────────────────────────────────────────
log_section "Step 6: NeMo Forced Aligner 独立环境 (可选)"

log_info "NeMo Forced Aligner 是一个可选的对齐后端 (NVIDIA CTC 强制对齐)。"
log_info "由于 nemo_toolkit 对 packaging/fsspec/omegaconf/hydra-core/lightning"
log_info "等核心依赖有严格版本限制，与主环境一起安装会产生依赖冲突，"
log_info "因此它需要运行在独立的 Python 环境里，作为一个本地服务 (端口 5002)"
log_info "供主后端通过 HTTP 调用。不安装也完全不影响 MFA / WhisperX / Qwen3 后端使用。"
echo ""

NEMO_VENV_DIR="$SCRIPT_DIR/.nemo_env"

confirm "是否现在创建独立环境并安装 NeMo Forced Aligner?"
nemo_install_choice=$?

if [ $nemo_install_choice -eq 0 ] || [ $nemo_install_choice -eq 2 ]; then
    if [ -d "$NEMO_VENV_DIR" ]; then
        log_warn "NeMo 环境已存在: $NEMO_VENV_DIR"
        read -p "是否删除并重新创建? [y/n]: " -r nemo_response
        if [[ "$nemo_response" =~ ^[yY]$ ]]; then
            log_step "删除旧 NeMo 环境..."
            rm -rf "$NEMO_VENV_DIR"
            log_ok "已删除"
        else
            log_info "使用现有 NeMo 环境"
            skip_nemo_venv_create=1
        fi
    fi

    if [ -z "$skip_nemo_venv_create" ]; then
        log_step "创建 NeMo 独立虚拟环境..."
        python3 -m venv "$NEMO_VENV_DIR"
        log_ok "NeMo 虚拟环境已创建: $NEMO_VENV_DIR"
    fi

    log_step "在独立环境中安装 nemo_toolkit[asr]（体积较大，可能需要较长时间）..."
    if "$NEMO_VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel > /dev/null 2>&1 && \
       "$NEMO_VENV_DIR/bin/python" -m pip install "nemo_toolkit[asr]>=2.7.0,<2.8.0" flask; then
        log_ok "NeMo Forced Aligner 依赖已安装"
        log_info "首次启动 nemo_server.py 时会按所选语言自动下载模型权重（数百 MB ~ 1GB）"
    else
        log_warn "NeMo 依赖安装失败，可稍后手动执行："
        log_warn "  $NEMO_VENV_DIR/bin/python -m pip install \"nemo_toolkit[asr]>=2.7.0,<2.8.0\" flask"
    fi
else
    log_info "已跳过，可后续手动运行以下命令安装："
    log_info "  python3 -m venv .nemo_env"
    log_info "  .nemo_env/bin/python -m pip install \"nemo_toolkit[asr]>=2.7.0,<2.8.0\" flask"
fi

# ─────────────────────────────────────────────────────────────────────────
# 完成
# ─────────────────────────────────────────────────────────────────────────
log_section "✅ 安装完全完成"

cat << EOF

${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}

📍 下一步:
   启动应用程序:
   ${CYAN}./run.sh${NC}

🔧 故障排除:
   查看日志: ${CYAN}tail -f backend/logs/app.log${NC}

${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}

EOF

echo ""