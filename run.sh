#!/bin/bash

# 设置终端编码为 UTF-8
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

# 切换到当前脚本所在的绝对路径
cd "$(dirname "$(readlink -f "$0")")"

echo "================================================"
echo "    SVS Lab Aligner + Qwen3-ASR + NeMo-FA 服务启动器"
echo "================================================"

# Linux 环境下虚拟环境的 Python 路径在 bin 目录下
MFA_PY="$(pwd)/.mfa_env/bin/python"
QWEN_PY="$(pwd)/.qwen3_env/bin/python"
NEMO_PY="$(pwd)/.nemo_env/bin/python"

# 记录后台服务的 PID，主服务退出时统一清理，避免留下孤儿进程
BG_PIDS=()
cleanup() {
    if [ ${#BG_PIDS[@]} -gt 0 ]; then
        echo ""
        echo "[清理] 正在停止后台服务 (PID: ${BG_PIDS[*]})..."
        kill "${BG_PIDS[@]}" 2>/dev/null
    fi
}
trap cleanup EXIT INT TERM

# 检查主后端环境
if [ ! -f "$MFA_PY" ]; then
    echo "[错误] .mfa_env/bin/python 未找到！"
    read -p "按回车键退出..."
    exit 1
fi

STEP=1
TOTAL_STEPS=1
[ -f "$QWEN_PY" ] && TOTAL_STEPS=$((TOTAL_STEPS + 1))
[ -f "$NEMO_PY" ] && TOTAL_STEPS=$((TOTAL_STEPS + 1))

# 检查并启动 Qwen3 后端
if [ -f "$QWEN_PY" ]; then
    echo "[$STEP/$TOTAL_STEPS] 启动 Qwen3-ASR 推理服务（端口 5001）..."
    # 使用 & 让服务在后台运行，并将日志重定向到文件，避免干扰主控制台
    "$QWEN_PY" "$(pwd)/qwen3_server.py" > qwen3_server.log 2>&1 &
    BG_PIDS+=($!)
    STEP=$((STEP + 1))
else
    echo "[警告] .qwen3_env/bin/python 未找到，将跳过 Qwen3-ASR（该后端不可用）"
fi

# 检查并启动 NeMo Forced Aligner 后端
if [ -f "$NEMO_PY" ]; then
    echo "[$STEP/$TOTAL_STEPS] 启动 NeMo Forced Aligner 服务（端口 5002）..."
    "$NEMO_PY" "$(pwd)/nemo_server.py" > nemo_server.log 2>&1 &
    BG_PIDS+=($!)
    STEP=$((STEP + 1))
else
    echo "[警告] .nemo_env/bin/python 未找到，将跳过 NeMo Forced Aligner（该后端不可用）"
fi

if [ ${#BG_PIDS[@]} -gt 0 ]; then
    echo "[等待 5 秒让后台服务完全启动...]"
    sleep 5
fi

echo "[$STEP/$TOTAL_STEPS] 启动主后端服务（端口 5000）..."
# 前台运行主服务；Ctrl+C 或主服务退出都会触发上面注册的 cleanup
"$MFA_PY" backend/app.py

# 主服务退出后的提示
read -p "服务已停止，按回车键退出..."