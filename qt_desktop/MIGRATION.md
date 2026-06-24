# Qt Widgets → Qt Quick 迁移指南

## 概述

本目录是原 `gui/` 目录的 **Qt Quick (QML) 重写版本**。  
Python 后端逻辑（`api_client.py`、`i18n.py`、`locales/`）**完全不变**，只替换了视图层。

---

## 文件结构对比

```
旧版 (Qt Widgets)              新版 (Qt Quick)
─────────────────────────────  ──────────────────────────────────
main.py                    →   main.py           (重写为 QML 引擎入口)
main_window.py             →   qml/main.qml      (根窗口)
main_window.ui             →   (已废弃，由 QML 替代)
processor_widget.py        →   qml/ProcessorWidget.qml
system_status_widget.py    →   qml/SystemStatusWidget.qml
api_client.py              →   api_client.py     (未改动)
i18n.py                    →   i18n.py           (未改动)
requirements_gui.txt       →   requirements_gui.txt (移除 qt-material)
(新增) bridge.py           →   Python ↔ QML 桥接层
```

### QML 组件清单

| 文件 | 作用 |
|------|------|
| `qml/main.qml` | 根 ApplicationWindow、顶层布局 |
| `qml/AppHeader.qml` | 标题栏（标题、语言选择、就绪徽章）|
| `qml/AppFooter.qml` | 底部版权栏 |
| `qml/BackendBar.qml` | 后端 URL 输入栏 |
| `qml/ProcessorWidget.qml` | 主处理表单（拖放、参数、进度、结果）|
| `qml/SystemStatusWidget.qml` | 系统状态面板 |
| `qml/FileDropZone.qml` | 文件拖放 / 浏览区域 |
| `qml/WarningBanner.qml` | 警告 / 错误横幅 |
| `qml/StatusBadge.qml` | 绿/红状态徽章 |
| `qml/CollapsibleSection.qml` | 可折叠内容区 |
| `qml/FormRow.qml` | 表单行（左标签 + 右内容）|
| `qml/RowLabel.qml` | 高级设置行（窄标签 + 右内容）|

---

## 安装与运行

```bash
# 1. 安装依赖（移除了 qt-material，Material 风格内置于 PySide6）
pip install -r requirements_gui.txt

# 2. 确保 locales/ 目录和后端 Flask 服务在同一父目录
#    目录结构示例：
#    project_root/
#    ├── svs_qt_quick/    ← 本目录
#    │   ├── main.py
#    │   ├── bridge.py
#    │   ├── api_client.py
#    │   ├── i18n.py
#    │   ├── locales/     ← 从原项目复制过来
#    │   ├── qml/
#    │   └── requirements_gui.txt
#    └── backend/         ← Flask 后端

# 3. 运行
python main.py
python main.py --lang en
python main.py --backend http://192.168.1.100:5000
python main.py --debug
```

---

## 架构说明

### Python ↔ QML 数据流

```
Python                        QML
──────────────────────────    ──────────────────────────────
I18nBridge.t(key) → str   →  i18n.t("key")
I18nBridge.languageChanged →  Connections { onLanguageChanged }
StatusBridge.mfaInstalled  →  status.mfaInstalled  (Property binding)
ApiBridge.checkStatus()    ←  api.checkStatus()     (QML calls Slot)
ApiBridge.jobCompleted     →  onJobCompleted(result)(Signal)
ApiBridge.startJob(...)    ←  api.startJob(...)     (QML calls Slot)
```

### 三个全局上下文对象

| QML 名称 | Python 类 | 作用 |
|----------|-----------|------|
| `i18n` | `I18nBridge` | 国际化翻译、语言切换 |
| `api`  | `ApiBridge`  | 所有 HTTP 请求、作业轮询 |
| `status` | `StatusBridge` | 系统状态状态机（MFA、模型等）|

---

## 与原版的主要差异

| 方面 | 旧版 (Qt Widgets) | 新版 (Qt Quick) |
|------|-------------------|-----------------|
| 主题 | `qt_material` 外部库 | Qt Material 内置风格 |
| 布局 | `QVBoxLayout` / `QHBoxLayout` | `ColumnLayout` / `RowLayout` / `Column` |
| 文件对话框 | `QFileDialog` | `QtQuick.Dialogs.FileDialog` |
| 拖放 | `QDropEvent` | QML `DropArea` |
| 国际化刷新 | `Signal.connect` | `Connections { onLanguageChanged }` |
| 状态管理 | Python 属性 | QML 属性绑定 + Python Property |
| 进度轮询 | `QTimer` in widget | `QTimer` in `ApiBridge` + QML signal |
| 剪贴板 | `QClipboard` | `TextArea.copy()` (built-in) |

---

## 常见问题

**Q: 运行时提示 `QML module not found`**  
A: 确认 PySide6 版本 ≥ 6.6，然后：`pip install --upgrade PySide6`

**Q: 语言切换后部分文字未更新**  
A: 检查对应 QML 文件是否有 `Connections { target: i18n; function onLanguageChanged() { ... } }`

**Q: 文件拖放无效**  
A: 某些平台需要在终端以非管理员权限运行；拖放支持依赖 `DropArea` 组件，确保 QML 文件完整。

**Q: 如何添加新的翻译键？**  
A: 在 `locales/` 各语言文件中添加键值对，Python 侧无需修改（`I18nBridge.t()` 自动查找）。
