#!/bin/bash
# GPT-SOVITS MFA Aligner 一键安装脚本 (Linux/Mac)

echo ""
echo "==============================================="
echo "  GPT-SOVITS MFA Aligner 安装程序"
echo "==============================================="
echo ""

echo "[1/4] 检查Python..."
if ! command -v python3 &> /dev/null; then
    echo "❌ 未安装Python，请先安装 Python 3.8+"
    exit 1
fi
echo "✓ Python已安装"

echo "[2/4] 检查Node.js..."
if ! command -v node &> /dev/null; then
    echo "❌ 未安装Node.js，请先安装 Node.js 16+"
    exit 1
fi
echo "✓ Node.js已安装"

echo "[3/4] 安装Python依赖..."
cd backend
pip install -r requirements.txt || { echo "❌ Python依赖安装失败"; exit 1; }
cd ..
echo "✓ Python依赖安装完成"

echo "[4/4] 安装前端依赖..."
cd frontend
npm install || { echo "❌ 前端依赖安装失败"; exit 1; }
npm run build || { echo "❌ 前端构建失败"; exit 1; }
cd ..
echo "✓ 前端依赖安装完成"

echo ""
echo "==============================================="
echo "  ✅ 安装完成！"
echo "==============================================="
echo ""
echo "下一步: 运行 ./run.sh 启动应用程序"
echo ""