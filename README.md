# GPT-SOVITS MFA 音频标注独立应用

一个完全独立的Web应用程序，用于自动标注GPT-SOVITS生成的语音文件。

## 🚀 快速开始

### Windows

\`\`\`batch
setup.bat
run.bat
\`\`\`

### Linux / Mac

\`\`\`bash
chmod +x setup.sh run.sh
./setup.sh
./run.sh
\`\`\`

应用将在 http://localhost:5000 打开

## ✨ 功能

- 支持多种音频格式 (WAV/MP3/FLAC)
- 自动时间对齐
- LAB格式导出
- 多语言支持
- Web界面友好

## 📋 系统要求

- Python 3.8+
- Node.js 16+
- 4GB RAM
- 1GB 磁盘空间

## 🔧 安装MFA

\`\`\`bash
# 安装MFA
pip install montreal-forced-aligner

# 下载语言模型
mfa model download acoustic cmn  # 中文
mfa model download acoustic eng  # 英语
mfa model download acoustic jpn  # 日语
\`\`\`

## 📖 使用流程

1. 在GPT-SOVITS中生成语音
2. 打开本应用
3. 上传WAV文件和输入文本
4. 选择对应语言
5. 点击开始标注
6. 下载生成的LAB文件

## 📝 许可证

MIT License