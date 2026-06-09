#!/bin/bash
# 启动 GPT-SOVITS MFA Aligner (Linux/Mac)
# 支持: 开发模式、调试模式、环境配置

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
FRONTEND_DIR="$SCRIPT_DIR/frontend"

# 颜色定义
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# 日志函数
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_ok() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 默认配置
MODE="production"
PYTHON_CMD="python3"
PORT="5000"
HOST="127.0.0.1"
DEBUG=false

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --dev)
            MODE="development"
            DEBUG=true
            log_info "开发模式已启用"
            shift
            ;;
        --debug)
            DEBUG=true
            log_info "调试模式已启用"
            shift
            ;;
        --host)
            HOST="$2"
            log_info "主机设置为: $HOST"
            shift 2
            ;;
        --port)
            PORT="$2"
            log_info "端口设置为: $PORT"
            shift 2
            ;;
        --help)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --dev              开发模式（热重载、更详细日志）"
            echo "  --debug            调试模式（Flask 调试器）"
            echo "  --host HOST        监听地址（默认: 127.0.0.1）"
            echo "  --port PORT        监听端口（默认: 5000）"
            echo "  --help             显示此帮助信息"
            echo ""
            echo "示例:"
            echo "  $0                 # 生产模式"
            echo "  $0 --dev           # 开发模式"
            echo "  $0 --host 0.0.0.0 --port 8080  # 公网访问、8080 端口"
            echo ""
            exit 0
            ;;
        *)
            log_warn "未知选项: $1"
            shift
            ;;
    esac
done

echo ""
echo "========================================"
echo "  GPT-SOVITS MFA Aligner"
echo "========================================"
echo ""

# 检查必要的目录和文件
log_info "检查项目结构..."

if [ ! -d "$BACKEND_DIR" ]; then
    log_error "未找到后端目录: $BACKEND_DIR"
    exit 1
fi

if [ ! -f "$BACKEND_DIR/app.py" ]; then
    log_error "未找到后端主文件: $BACKEND_DIR/app.py"
    exit 1
fi

if [ ! -d "$FRONTEND_DIR" ]; then
    log_error "未找到前端目录: $FRONTEND_DIR"
    exit 1
fi

log_ok "项目结构验证成功"

# 检查Python环境
log_info "检查 Python 环境..."

# 优先使用项目虚拟环境
if [ -f "$BACKEND_DIR/venv/bin/python" ]; then
    PYTHON_CMD="$BACKEND_DIR/venv/bin/python"
    log_ok "使用项目虚拟环境"
elif [ -f "$BACKEND_DIR/venv/bin/python3" ]; then
    PYTHON_CMD="$BACKEND_DIR/venv/bin/python3"
    log_ok "使用项目虚拟环境"
else
    log_warn "未找到项目虚拟环境，使用系统 Python"
fi

# 验证Python版本
PYTHON_VERSION=$("$PYTHON_CMD" --version 2>&1 | awk '{print $2}')
log_ok "Python 版本: $PYTHON_VERSION"

# 检查依赖
log_info "验证依赖..."
if ! "$PYTHON_CMD" -c "import flask, flask_cors" 2>/dev/null; then
    log_error "缺少必要的 Python 依赖，请先运行 setup.sh"
    exit 1
fi
log_ok "依赖检查完成"

# 创建日志目录
if [ ! -d "$BACKEND_DIR/logs" ]; then
    mkdir -p "$BACKEND_DIR/logs"
    log_ok "日志目录已创建"
fi

echo ""
echo "========================================"
echo "  启动配置"
echo "========================================"
echo "模式:         $MODE"
echo "主机:         $HOST"
echo "端口:         $PORT"
echo "调试:         $([ "$DEBUG" = true ] && echo '启用' || echo '禁用')"
echo ""

# 设置环境变量
export FLASK_APP="$BACKEND_DIR/app.py"
export FLASK_ENV="$( [ "$MODE" = "development" ] && echo "development" || echo "production" )"
export FLASK_DEBUG=$([ "$DEBUG" = true ] && echo "1" || echo "0")
export MFA_FRONT_HOST="$HOST"
export MFA_FRONT_PORT="$PORT"

# 检查GitHub Token
if [ -n "$MFA_GITHUB_TOKEN" ]; then
    log_ok "GitHub Token 已配置（用于 API 速率限制）"
    export MFA_GITHUB_TOKEN
fi

log_info "启动后端服务..."
echo ""

# 启动Flask应用
cd "$BACKEND_DIR"

if [ "$DEBUG" = true ]; then
    # 调试模式：使用Flask开发服务器
    "$PYTHON_CMD" app.py \
        --host "$HOST" \
        --port "$PORT" \
        --debug
else
    # 生产模式：使用gunicorn（如果可用）或Flask服务器
    if "$PYTHON_CMD" -c "import gunicorn" 2>/dev/null; then
        log_info "使用 gunicorn 运行（推荐用于生产环境）"
        gunicorn \
            --bind "$HOST:$PORT" \
            --workers 4 \
            --timeout 120 \
            --access-logfile "$BACKEND_DIR/logs/access.log" \
            --error-logfile "$BACKEND_DIR/logs/error.log" \
            --log-level info \
            app:app
    else
        log_warn "未安装 gunicorn，使用 Flask 开发服务器"
        log_warn "生产环境建议: pip install gunicorn"
        "$PYTHON_CMD" app.py \
            --host "$HOST" \
            --port "$PORT"
    fi
fi
