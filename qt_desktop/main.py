# main.py — Qt Quick entry point
"""
SVS Lab Aligner — Qt Quick Edition
====================================
Run:
    python main.py
    python main.py --lang en
    python main.py --backend http://127.0.0.1:5000
    python main.py --debug
"""
from __future__ import annotations

import sys
import os
import argparse
import logging
from pathlib import Path

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _check_deps() -> list[str]:
    missing = []
    for pkg, name in [("PySide6", "PySide6"), ("requests", "requests")]:
        try:
            __import__(name)
        except ImportError:
            missing.append(pkg)
    return missing


def main():
    parser = argparse.ArgumentParser(description="SVS Lab Aligner GUI (Qt Quick)")
    parser.add_argument("--lang",    default="zh_cn",
                        choices=["zh_cn", "zh_tw", "ja", "ko", "en"])
    parser.add_argument("--backend", default="http://127.0.0.1:5000")
    parser.add_argument("--debug",   action="store_true")
    args = parser.parse_args()

    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    missing = _check_deps()
    if missing:
        print("Missing packages:", ", ".join(missing))
        print("Install: pip install", " ".join(missing))
        sys.exit(1)

    from PySide6.QtGui import QGuiApplication
    from PySide6.QtQml import QQmlApplicationEngine
    from PySide6.QtCore import QUrl, Qt
    from PySide6.QtQuickControls2 import QQuickStyle

    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    # Use Material style for all Qt Quick Controls
    QQuickStyle.setStyle("Material")

    app = QGuiApplication(sys.argv)
    app.setApplicationName("SVS Lab Aligner")
    app.setApplicationDisplayName("SVS Lab Aligner")
    app.setOrganizationName("SVSTools")
    app.setOrganizationDomain("github.com/liuhua520-svg")

    # ── bridge objects ────────────────────────────────────────────
    from i18n import I18n
    from api_client import ApiClient
    from bridge import I18nBridge, StatusBridge, ApiBridge

    i18n_obj    = I18n(args.lang)
    api_obj     = ApiClient(args.backend)
    status_obj  = StatusBridge()
    i18n_bridge = I18nBridge(i18n_obj)
    api_bridge  = ApiBridge(api_obj, status_obj)

    # ── QML engine ────────────────────────────────────────────────
    engine = QQmlApplicationEngine()

    # Register Python objects as global QML context properties
    ctx = engine.rootContext()
    ctx.setContextProperty("i18n",   i18n_bridge)
    ctx.setContextProperty("api",    api_bridge)
    ctx.setContextProperty("status", status_obj)

    # Let QML files in the qml/ dir resolve each other by type name
    qml_dir  = Path(__file__).parent / "qml"
    main_qml = qml_dir / "main.qml"
    engine.addImportPath(str(qml_dir))
    engine.load(QUrl.fromLocalFile(str(main_qml)))

    if not engine.rootObjects():
        logging.critical("QML engine failed to load %s", main_qml)
        sys.exit(-1)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
