# main.py — application entry point
"""
SVS Lab Aligner — Qt Material Edition
======================================
Run:
    python main.py
    python main.py --lang en          # start in English
    python main.py --lang zh_cn       # start in Simplified Chinese (default)
    python main.py --backend http://127.0.0.1:5000
"""
from __future__ import annotations

import sys
import os
import argparse
import logging

# ── ensure project root is importable ────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── early dependency check ────────────────────────────────────────────────────
def _check_deps() -> list[str]:
    missing = []
    for pkg, import_name in [
        ("PySide6",     "PySide6"),
        ("qt_material", "qt_material"),
        ("requests",    "requests"),
    ]:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg)
    return missing


def _show_missing_deps(missing: list[str]):
    """Fallback error dialog when even PySide6 is present but others are missing."""
    from PySide6.QtWidgets import QApplication, QMessageBox
    app = QApplication(sys.argv)
    msg = QMessageBox()
    msg.setWindowTitle("Missing Dependencies")
    msg.setIcon(QMessageBox.Critical)
    cmd = "pip install " + " ".join(missing)
    msg.setText(
        f"The following packages are missing:\n\n"
        + "\n".join(f"  • {p}" for p in missing)
        + f"\n\nPlease install them:\n\n    {cmd}"
    )
    msg.exec()
    sys.exit(1)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    # parse CLI
    parser = argparse.ArgumentParser(description="SVS Lab Aligner GUI")
    parser.add_argument("--lang",    default="zh_cn",
                        choices=["zh_cn","zh_tw","ja","ko","en"],
                        help="UI language at startup")
    parser.add_argument("--backend", default="http://127.0.0.1:5000",
                        help="Flask backend base URL")
    parser.add_argument("--theme",   default="dark_teal.xml",
                        help="Qt Material theme file")
    parser.add_argument("--debug",   action="store_true",
                        help="Enable verbose logging")
    args = parser.parse_args()

    # logging
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # dependency check (after arg-parse so --help still works without Qt)
    missing = _check_deps()
    if missing:
        try:
            _show_missing_deps(missing)
        except Exception:
            print("Missing packages:", ", ".join(missing))
            print("Install: pip install", " ".join(missing))
            sys.exit(1)

    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QIcon

    # ── HiDPI
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("SVS Lab Aligner")
    app.setApplicationDisplayName("SVS Lab Aligner")
    app.setOrganizationName("SVSTools")
    app.setOrganizationDomain("github.com/liuhua520-svg")

    # ── Qt Material theme
    try:
        from qt_material import apply_stylesheet
        extra = {
            "density_scale":  "-1",
            "font_family":    _best_font(),
            "primaryColor":       "#80cbc4",
            "primaryLightColor":  "#b2dfdb",
            "secondaryColor":     "#b39ddb",
            "secondaryLightColor":"#d1c4e9",
        }
        apply_stylesheet(app, theme=args.theme, extra=extra)
    except Exception as exc:
        logging.warning("qt_material not available (%s) — using plain style", exc)

    # ── override a few globals after qt_material
    app.setStyleSheet(app.styleSheet() + _EXTRA_CSS)

    # ── i18n
    from i18n import I18n
    i18n = I18n(args.lang)

    # ── main window
    from main_window import MainWindow
    win = MainWindow(i18n)

    # apply backend URL from CLI
    if args.backend != "http://127.0.0.1:5000":
        win._api.base_url = args.backend

    win.show()
    sys.exit(app.exec())


def _best_font() -> str:
    """Pick the best available CJK-capable font for the current OS."""
    from PySide6.QtGui import QFontDatabase
    db = QFontDatabase()
    families = db.families()
    candidates = [
        # Windows
        "Microsoft YaHei UI",
        "Yu Gothic UI",
        "Malgun Gothic",
        # macOS
        "PingFang SC",
        "Hiragino Sans",
        "Apple SD Gothic Neo",
        # Linux
        "Noto Sans CJK SC",
        "WenQuanYi Micro Hei",
        "Source Han Sans CN",
        # Fallbacks
        "Segoe UI",
        "Arial",
    ]
    for f in candidates:
        if f in families:
            return f
    return ""   # Qt default


# ── extra CSS on top of qt_material ──────────────────────────────────────────
_EXTRA_CSS = """
/* ── global ── */
QMainWindow, QDialog, QWidget      { background:#141624; }
QLabel                             { background:transparent; }
QToolTip                           { background:#1e2030; color:#eceff1;
                                     border:1px solid #3a3f5c; }

/* ── scroll bars ── */
QScrollBar:vertical                { background:#141624; width:8px; margin:0; }
QScrollBar::handle:vertical        { background:#3a3f5c; border-radius:4px; min-height:20px; }
QScrollBar::handle:vertical:hover  { background:#546e7a; }
QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical      { height:0; }
QScrollBar:horizontal              { background:#141624; height:8px; margin:0; }
QScrollBar::handle:horizontal      { background:#3a3f5c; border-radius:4px; min-width:20px; }

/* ── inputs ── */
QLineEdit, QTextEdit, QPlainTextEdit {
    background:#1a1d2e; color:#eceff1;
    border:1px solid #3a3f5c; border-radius:4px;
    selection-background-color:#37474f;
}
QLineEdit:focus, QTextEdit:focus   { border-color:#80cbc4; }

QSpinBox, QDoubleSpinBox           { background:#1a1d2e; color:#eceff1;
                                     border:1px solid #3a3f5c; border-radius:4px;
                                     padding:2px 4px; }
QSpinBox:focus, QDoubleSpinBox:focus { border-color:#80cbc4; }
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
    background:#263238; border:none; width:16px; }

/* ── buttons ── */
QPushButton                        { background:#1e2030; color:#eceff1;
                                     border:1px solid #3a3f5c; border-radius:5px;
                                     padding:5px 14px; }
QPushButton:hover                  { background:#263238; border-color:#80cbc4; }
QPushButton:pressed                { background:#37474f; }
QPushButton:disabled               { background:#141624; color:#455a64;
                                     border-color:#263238; }
QPushButton:flat                   { background:transparent; border:none; }

/* ── radio / check ── */
QRadioButton, QCheckBox            { color:#eceff1; spacing:6px; }
QRadioButton::indicator            { width:14px; height:14px; }
QRadioButton::indicator:checked    { background:#80cbc4; border-radius:7px;
                                     border:2px solid #80cbc4; }
QRadioButton::indicator:unchecked  { background:#1e2030; border-radius:7px;
                                     border:2px solid #607d8b; }
QCheckBox::indicator               { width:14px; height:14px; border-radius:3px; }
QCheckBox::indicator:checked       { background:#80cbc4; border:2px solid #80cbc4; }
QCheckBox::indicator:unchecked     { background:#1e2030; border:2px solid #607d8b; }

/* ── combo box ── */
QComboBox                          { background:#1e2030; color:#eceff1;
                                     border:1px solid #3a3f5c; border-radius:4px;
                                     padding:3px 8px; }
QComboBox:hover                    { border-color:#80cbc4; }
QComboBox::drop-down               { border:none; width:20px; }
QComboBox QAbstractItemView        { background:#1e2030; color:#eceff1;
                                     border:1px solid #3a3f5c;
                                     selection-background-color:#263238; }

/* ── tabs ── */
QTabWidget::pane                   { border:1px solid #3a3f5c; border-radius:4px;
                                     background:#1a1d2e; }
QTabBar::tab                       { background:#141624; color:#b0bec5;
                                     padding:6px 16px; border:none; }
QTabBar::tab:selected              { color:#80cbc4;
                                     border-bottom:2px solid #80cbc4; }
QTabBar::tab:hover                 { color:#eceff1; }

/* ── table ── */
QTableWidget                       { background:#141624; color:#eceff1;
                                     gridline-color:#263238;
                                     border:1px solid #3a3f5c; }
QHeaderView::section               { background:#1e2030; color:#80cbc4;
                                     border:none; padding:4px; }
QTableWidget::item:selected        { background:#263238; }

/* ── progress bar ── */
QProgressBar                       { background:#1e2030; border:1px solid #3a3f5c;
                                     border-radius:5px; text-align:center;
                                     color:#eceff1; font-size:11px; }
QProgressBar::chunk                { background:qlineargradient(
                                       x1:0,y1:0,x2:1,y2:0,
                                       stop:0 #00897b, stop:1 #00acc1);
                                     border-radius:5px; }

/* ── group box ── */
QGroupBox                          { color:#80cbc4; border:1px solid #3a3f5c;
                                     border-radius:6px; margin-top:8px;
                                     padding-top:8px; }
QGroupBox::title                   { subcontrol-origin:margin;
                                     subcontrol-position:top left;
                                     padding:0 6px; color:#80cbc4; }

/* ── tool button (collapsible) ── */
QToolButton                        { background:transparent; border:none;
                                     color:#80cbc4; }
QToolButton:hover                  { color:#b2dfdb; }

/* ── status bar ── */
QStatusBar                         { background:#0d1117; color:#484f58;
                                     font-size:10px; }
"""


if __name__ == "__main__":
    main()
