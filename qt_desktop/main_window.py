# main_window.py — application shell
from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QFont, QDesktopServices, QIcon, QPixmap, QPainter, QColor
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QFrame, QScrollArea, QSizePolicy,
    QStatusBar, QLineEdit,
)

from i18n import I18n
from api_client import ApiClient
from processor_widget import ProcessorWidget
from system_status_widget import SystemStatusWidget


# ── warning banner ────────────────────────────────────────────────────────────

class _WarningBanner(QFrame):
    def __init__(self, kind: str = "error", parent=None):
        super().__init__(parent)
        colors = {
            "error":   ("#4a1010", "#ef9a9a", "#b71c1c"),
            "warning": ("#3a2800", "#ffe082", "#f57f17"),
        }
        bg, fg, border = colors.get(kind, colors["error"])
        self.setStyleSheet(
            f"QFrame {{ background:{bg}; border:1px solid {border};"
            f" border-radius:6px; padding:4px; }}"
        )
        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 8)

        self._lbl = QLabel("")
        self._lbl.setWordWrap(True)
        self._lbl.setStyleSheet(f"color:{fg}; border:none; background:transparent;")
        lay.addWidget(self._lbl)

    def set_text(self, text: str):
        self._lbl.setText(text)

    def refresh_lang(self, text: str):
        self.set_text(text)


# ── header ────────────────────────────────────────────────────────────────────

class _Header(QFrame):
    language_selected = __import__("PySide6.QtCore", fromlist=["Signal"]).Signal(str)

    def __init__(self, i18n: I18n, parent=None):
        super().__init__(parent)
        self._i18n = i18n
        self.setObjectName("appHeader")
        self.setStyleSheet(
            "#appHeader { background:qlineargradient("
            "x1:0,y1:0,x2:1,y2:0,"
            "stop:0 #1a1040, stop:1 #0d1b2a);"
            " border-bottom:1px solid #3a3f5c; }"
        )
        self.setFixedHeight(64)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(20, 0, 20, 0)
        lay.setSpacing(12)

        # ── left: title block
        title_col = QVBoxLayout()
        title_col.setSpacing(0)
        self._title = QLabel(i18n.t("app_title"))
        self._title.setFont(QFont("", 16, QFont.Bold))
        self._title.setStyleSheet(
            "QLabel { background:transparent; color:qlineargradient("
            "x1:0,y1:0,x2:1,y2:0,stop:0 #80cbc4,stop:1 #b39ddb); }"
        )
        self._subtitle = QLabel(i18n.t("app_subtitle"))
        self._subtitle.setStyleSheet(
            "QLabel { background:transparent; color:#607d8b; font-size:10px; }"
        )
        title_col.addWidget(self._title)
        title_col.addWidget(self._subtitle)
        lay.addLayout(title_col)
        lay.addStretch()

        # ── language selector
        lang_row = QHBoxLayout()
        lang_row.setSpacing(6)
        self._lang_icon = QLabel("🌐")
        self._lang_icon.setStyleSheet("font-size:14px; background:transparent;")
        self._lang_combo = QComboBox()
        self._lang_combo.setFixedWidth(120)
        self._lang_combo.setStyleSheet(
            "QComboBox { background:#1e2030; color:#b0bec5;"
            " border:1px solid #3a3f5c; border-radius:4px; padding:2px 6px; }"
            "QComboBox::drop-down { border:none; }"
        )
        for code in i18n.LANGUAGE_CODES:
            self._lang_combo.addItem(i18n.DISPLAY_NAMES[code], code)
        # select current
        idx = i18n.LANGUAGE_CODES.index(i18n.current)
        self._lang_combo.setCurrentIndex(idx)
        self._lang_combo.currentIndexChanged.connect(self._on_lang_changed)

        lang_row.addWidget(self._lang_icon)
        lang_row.addWidget(self._lang_combo)
        lay.addLayout(lang_row)

        # ── system-ready badge
        self._ready_badge = QLabel(i18n.t("system_not_ready"))
        self._ready_badge.setStyleSheet(
            "QLabel { background:#b71c1c; color:#fff; border-radius:4px;"
            " padding:3px 10px; font-size:11px; font-weight:bold; }"
        )
        lay.addWidget(self._ready_badge)

        i18n.language_changed.connect(self._retranslate)

    def set_ready(self, ready: bool):
        t = self._i18n.t
        txt   = t("system_ready") if ready else t("system_not_ready")
        color = "#1b5e20" if ready else "#b71c1c"
        self._ready_badge.setText(txt)
        self._ready_badge.setStyleSheet(
            f"QLabel {{ background:{color}; color:#fff; border-radius:4px;"
            f" padding:3px 10px; font-size:11px; font-weight:bold; }}"
        )

    def _on_lang_changed(self, idx: int):
        code = self._lang_combo.itemData(idx)
        self._i18n.set_language(code)

    def _retranslate(self):
        t = self._i18n.t
        self._title.setText(t("app_title"))
        self._subtitle.setText(t("app_subtitle"))
        # refresh combo labels for current language's own display names
        for i, code in enumerate(self._i18n.LANGUAGE_CODES):
            self._lang_combo.setItemText(i, self._i18n.DISPLAY_NAMES[code])


# ── footer ────────────────────────────────────────────────────────────────────

class _Footer(QFrame):
    def __init__(self, i18n: I18n, parent=None):
        super().__init__(parent)
        self._i18n = i18n
        self.setObjectName("appFooter")
        self.setStyleSheet(
            "#appFooter { background:#0d1117; border-top:1px solid #21262d; }"
        )
        lay = QHBoxLayout(self)
        lay.setContentsMargins(20, 8, 20, 8)

        self._lbl = QLabel(i18n.t("footer_text"))
        self._lbl.setStyleSheet("color:#484f58; font-size:10px; background:transparent;")
        lay.addWidget(self._lbl)
        lay.addStretch()

        gh = QPushButton("📚 GitHub")
        gh.setFlat(True)
        gh.setStyleSheet("QPushButton { color:#58a6ff; border:none; font-size:10px; } "
                         "QPushButton:hover { color:#79c0ff; }")
        gh.setCursor(Qt.PointingHandCursor)
        gh.clicked.connect(lambda: QDesktopServices.openUrl(
            QUrl("https://github.com/liuhua520-svg/gpt-sovits-mfa-aligner-tsubaki")
        ))
        lay.addWidget(gh)

        i18n.language_changed.connect(self._retranslate)

    def _retranslate(self):
        self._lbl.setText(self._i18n.t("footer_text"))


# ── backend URL bar ───────────────────────────────────────────────────────────

class _BackendBar(QFrame):
    def __init__(self, i18n: I18n, api: ApiClient, parent=None):
        super().__init__(parent)
        self._i18n = i18n
        self._api  = api
        self.setObjectName("backendBar")
        self.setStyleSheet(
            "#backendBar { background:#141624; border-bottom:1px solid #3a3f5c; }"
        )

        lay = QHBoxLayout(self)
        lay.setContentsMargins(16, 4, 16, 4)
        lay.setSpacing(8)

        self._url_lbl = QLabel(i18n.t("backend_url_label"))
        self._url_lbl.setStyleSheet("color:#607d8b; font-size:11px;")
        lay.addWidget(self._url_lbl)

        self._url_edit = QLineEdit(api.base_url)
        self._url_edit.setFixedWidth(220)
        self._url_edit.setStyleSheet(
            "QLineEdit { background:#1e2030; color:#eceff1;"
            " border:1px solid #3a3f5c; border-radius:3px;"
            " padding:2px 6px; font-size:11px; }"
        )
        lay.addWidget(self._url_edit)

        self._conn_btn = QPushButton(i18n.t("connect"))
        self._conn_btn.setFixedWidth(70)
        self._conn_btn.setStyleSheet(
            "QPushButton { background:#00695c; color:#fff; border-radius:3px; }"
            "QPushButton:hover { background:#00897b; }"
        )
        self._conn_btn.clicked.connect(self._on_connect)
        lay.addWidget(self._conn_btn)
        lay.addStretch()

        i18n.language_changed.connect(self._retranslate)

    url_changed = __import__("PySide6.QtCore", fromlist=["Signal"]).Signal(str)

    def _on_connect(self):
        new_url = self._url_edit.text().strip().rstrip("/")
        if new_url:
            self._api.base_url = new_url
            self.url_changed.emit(new_url)

    def _retranslate(self):
        self._url_lbl.setText(self._i18n.t("backend_url_label"))
        self._conn_btn.setText(self._i18n.t("connect"))


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ══════════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self, i18n: I18n, parent=None):
        super().__init__(parent)
        self._i18n = i18n
        self._api  = ApiClient("http://127.0.0.1:5000")

        self.setWindowTitle("SVS Lab Aligner")
        self.resize(900, 820)
        self.setMinimumSize(700, 600)
        self.setStyleSheet(
            "QMainWindow, QWidget { background:#141624; color:#eceff1; }"
            "QLabel { background:transparent; }"
            "QScrollBar:vertical { background:#1e2030; width:8px; }"
            "QScrollBar::handle:vertical { background:#3a3f5c; border-radius:4px; }"
            "QPushButton { background:#1e2030; color:#eceff1;"
            " border:1px solid #3a3f5c; border-radius:4px; padding:5px 12px; }"
            "QPushButton:hover { background:#263238; }"
            "QComboBox QAbstractItemView { background:#1e2030; color:#eceff1; }"
            "QRadioButton { color:#eceff1; spacing:6px; }"
            "QCheckBox  { color:#eceff1; spacing:6px; }"
            "QSpinBox, QDoubleSpinBox { background:#1e2030; color:#eceff1;"
            " border:1px solid #3a3f5c; border-radius:3px; padding:2px; }"
            "QTabWidget::pane { border:1px solid #3a3f5c; border-radius:4px; }"
            "QProgressBar { background:#1e2030; border:1px solid #3a3f5c;"
            " border-radius:4px; text-align:center; }"
            "QProgressBar::chunk { background:#00897b; border-radius:4px; }"
        )

        # ── central widget
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── header
        self._header = _Header(i18n)
        root.addWidget(self._header)

        # ── backend bar
        self._backend_bar = _BackendBar(i18n, self._api)
        self._backend_bar.url_changed.connect(self._on_url_changed)
        root.addWidget(self._backend_bar)

        # ── scrollable content area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border:none; }")
        root.addWidget(scroll, 1)

        content = QWidget()
        scroll.setWidget(content)
        col = QVBoxLayout(content)
        col.setContentsMargins(20, 16, 20, 16)
        col.setSpacing(16)

        # ── processor widget
        self._processor = ProcessorWidget(i18n, self._api)
        self._processor.status_changed.connect(self._on_status_changed)
        col.addWidget(self._processor)

        # ── system status widget
        self._status_panel = SystemStatusWidget(i18n)
        self._status_panel.download_model_requested.connect(self._on_download_model)
        col.addWidget(self._status_panel)

        # ── warning banners (hidden by default)
        self._warn_mfa = _WarningBanner("error")
        self._warn_mfa.setVisible(False)
        col.addWidget(self._warn_mfa)

        self._warn_models = _WarningBanner("warning")
        self._warn_models.setVisible(False)
        col.addWidget(self._warn_models)

        col.addStretch()

        # ── footer
        self._footer = _Footer(i18n)
        root.addWidget(self._footer)

        # ── status bar
        self.statusBar().setStyleSheet(
            "QStatusBar { background:#0d1117; color:#607d8b; font-size:10px; }"
        )
        self.statusBar().showMessage("SVS Lab Aligner  |  http://127.0.0.1:5000")

        # ── initial status check
        i18n.language_changed.connect(self._retranslate)
        QTimer.singleShot(500, self._check_status)

    # ── backend communication ─────────────────────────────────────
    def _check_status(self):
        self._status_worker = self._api.status()
        self._status_worker.succeeded.connect(self._on_status_fetched)
        self._status_worker.failed.connect(self._on_status_fail)
        self._status_worker.start()

    def _on_status_fetched(self, data: dict):
        pipeline = data.get("pipeline", {})
        aligner  = data.get("aligner", {})
        self._processor.update_system_status(pipeline, aligner)
        self._status_panel.update_status(pipeline, aligner)
        self._update_warnings(pipeline)
        self.statusBar().showMessage(
            f"SVS Lab Aligner  |  {self._api.base_url}  |  "
            + self._i18n.t("msg_status_refreshed")
        )

    def _on_status_fail(self, err: str):
        self.statusBar().showMessage(
            self._i18n.t("msg_backend_unreachable") + f":  {err}"
        )

    def _on_url_changed(self, url: str):
        self.statusBar().showMessage(f"Backend → {url}")
        self._check_status()

    def _on_status_changed(self, sys_status: dict):
        mfa_ok = bool(sys_status.get("mfa", {}).get("installed"))
        self._header.set_ready(mfa_ok)

    def _update_warnings(self, pipeline_data: dict):
        t = self._i18n.t
        status = pipeline_data.get("status", {})
        mfa = status.get("mfa", {})
        installed = bool(mfa.get("installed"))

        if not installed:
            self._warn_mfa.set_text(
                f"<b>{t('warn_mfa_not_installed')}</b><br>"
                f"<code>{t('warn_mfa_install_cmd')}</code><br>"
                f"{t('warn_install_hint')}"
            )
            self._warn_mfa.setVisible(True)
            self._warn_models.setVisible(False)
        else:
            self._warn_mfa.setVisible(False)
            models = mfa.get("models", {})
            all_ok = all(models.get(c, False) for c in
                         ["cmn","eng","jpn","kor","yue"])
            if not all_ok:
                self._warn_models.set_text(
                    f"<b>{t('warn_not_ready_title')}</b><br>"
                    f"{t('warn_not_ready_msg')}"
                )
                self._warn_models.setVisible(True)
            else:
                self._warn_models.setVisible(False)

    def _on_download_model(self, lang: str):
        self._status_panel.set_model_downloading(lang, True)
        worker = self._api.download_model(lang)
        worker.succeeded.connect(lambda d: self._on_model_downloaded(lang, d))
        worker.failed.connect(lambda e: self._on_model_fail(lang, e))
        worker.start()
        # keep worker alive
        setattr(self, f"_dl_{lang}", worker)

    def _on_model_downloaded(self, lang: str, _data: dict):
        self._status_panel.set_model_downloading(lang, False)
        self.statusBar().showMessage(
            self._i18n.t("msg_model_ok", lang=lang)
        )
        QTimer.singleShot(1000, self._check_status)

    def _on_model_fail(self, lang: str, err: str):
        self._status_panel.set_model_downloading(lang, False)
        self.statusBar().showMessage(
            self._i18n.t("msg_model_fail", error=err)
        )

    # ── i18n refresh ──────────────────────────────────────────────
    def _retranslate(self):
        self.statusBar().showMessage(
            f"SVS Lab Aligner  |  {self._api.base_url}"
        )
