# system_status_widget.py — bottom status panel
from __future__ import annotations
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QFrame, QSizePolicy,
)
from PySide6.QtGui import QFont

from i18n import I18n

# ── small status badge ────────────────────────────────────────────────────────

class _Badge(QLabel):
    def __init__(self, text: str = "", ok: bool = True, parent=None):
        super().__init__(text, parent)
        self.setOk(ok)
        self.setFixedHeight(22)

    def setOk(self, ok: bool):
        color = "#4caf50" if ok else "#f44336"
        self.setStyleSheet(
            f"QLabel {{ background:{color}; color:#fff; border-radius:4px;"
            f" padding:1px 8px; font-size:11px; font-weight:bold; }}"
        )


# ── single model row ──────────────────────────────────────────────────────────

class _ModelRow(QWidget):
    download_requested = Signal(str)   # emits lang code

    def __init__(self, lang_code: str, display_name: str,
                 downloaded: bool, i18n: I18n, parent=None):
        super().__init__(parent)
        self._lang = lang_code
        self._i18n = i18n

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 2)

        self._badge = _Badge(display_name, ok=downloaded)
        lay.addWidget(self._badge)

        self._btn = QPushButton(i18n.t("download_btn"))
        self._btn.setFixedWidth(90)
        self._btn.setCursor(Qt.PointingHandCursor)
        self._btn.setVisible(not downloaded)
        self._btn.clicked.connect(lambda: self.download_requested.emit(lang_code))
        lay.addWidget(self._btn)
        lay.addStretch()

    def refresh_lang(self):
        self._btn.setText(self._i18n.t("download_btn"))

    def set_downloading(self, downloading: bool):
        self._btn.setEnabled(not downloading)
        self._btn.setText(
            self._i18n.t("downloading_btn") if downloading
            else self._i18n.t("download_btn")
        )

    def set_downloaded(self, downloaded: bool):
        self._badge.setOk(downloaded)
        self._btn.setVisible(not downloaded)


# ── main widget ───────────────────────────────────────────────────────────────

class SystemStatusWidget(QWidget):
    download_model_requested = Signal(str)

    def __init__(self, i18n: I18n, parent=None):
        super().__init__(parent)
        self._i18n = i18n
        self._model_rows: dict[str, _ModelRow] = {}
        self._module_labels: dict[str, QLabel] = {}
        self._backend_labels: dict[str, QLabel] = {}
        self._setup_ui()
        i18n.language_changed.connect(self._retranslate)

    # ── build UI ──────────────────────────────────────────────────
    def _setup_ui(self):
        t = self._i18n.t

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        card = QFrame()
        card.setObjectName("statusCard")
        card.setStyleSheet(
            "#statusCard { background:#1e2030; border-radius:8px;"
            " border:1px solid #3a3f5c; }"
        )
        outer.addWidget(card)

        lay = QVBoxLayout(card)
        lay.setContentsMargins(16, 12, 16, 14)
        lay.setSpacing(12)

        # ── title
        self._title_lbl = QLabel(t("system_status"))
        self._title_lbl.setFont(QFont("", 13, QFont.Bold))
        self._title_lbl.setStyleSheet("color:#80cbc4;")
        lay.addWidget(self._title_lbl)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color:#3a3f5c;")
        lay.addWidget(sep)

        grid = QGridLayout()
        grid.setHorizontalSpacing(24)
        grid.setVerticalSpacing(10)
        row = 0

        # ── MFA status
        self._mfa_status_lbl = QLabel(t("mfa_status"))
        self._mfa_status_lbl.setStyleSheet("color:#b0bec5; font-weight:bold;")
        self._mfa_badge = _Badge(t("mfa_not_installed"), ok=False)
        grid.addWidget(self._mfa_status_lbl, row, 0, Qt.AlignTop)
        grid.addWidget(self._mfa_badge, row, 1, Qt.AlignLeft | Qt.AlignTop)
        row += 1

        self._version_lbl = QLabel(t("mfa_version"))
        self._version_lbl.setStyleSheet("color:#b0bec5;")
        self._version_val = QLabel("—")
        self._version_val.setStyleSheet("color:#eceff1;")
        grid.addWidget(self._version_lbl, row, 0, Qt.AlignTop)
        grid.addWidget(self._version_val, row, 1, Qt.AlignTop)
        row += 1

        # ── Language models
        self._lang_models_lbl = QLabel(t("language_models"))
        self._lang_models_lbl.setStyleSheet("color:#b0bec5; font-weight:bold;")
        grid.addWidget(self._lang_models_lbl, row, 0, Qt.AlignTop)

        models_col = QVBoxLayout()
        models_col.setSpacing(4)
        for code, key in [("cmn","model_cmn"), ("eng","model_eng"),
                          ("jpn","model_jpn"), ("kor","model_kor"),
                          ("yue","model_yue")]:
            mr = _ModelRow(code, t(key), False, self._i18n)
            mr.download_requested.connect(self._on_download_requested)
            self._model_rows[code] = mr
            models_col.addWidget(mr)

        models_wrap = QWidget()
        models_wrap.setLayout(models_col)
        grid.addWidget(models_wrap, row, 1, Qt.AlignTop)
        row += 1

        # ── Processing modules
        self._proc_lbl = QLabel(t("processing_modules"))
        self._proc_lbl.setStyleSheet("color:#b0bec5; font-weight:bold;")
        grid.addWidget(self._proc_lbl, row, 0, Qt.AlignTop)

        proc_col = QVBoxLayout()
        proc_col.setSpacing(4)
        for key in ["PyWORLD (DIO/Harvest)", "CREPE", "RMVPE"]:
            lbl = _Badge(key, ok=False)
            self._module_labels[key] = lbl
            proc_col.addWidget(lbl)

        proc_wrap = QWidget()
        proc_wrap.setLayout(proc_col)
        grid.addWidget(proc_wrap, row, 1, Qt.AlignTop)
        row += 1

        # ── Alt backends
        self._alt_lbl = QLabel(t("alt_backends_section"))
        self._alt_lbl.setStyleSheet("color:#b0bec5; font-weight:bold;")
        grid.addWidget(self._alt_lbl, row, 0, Qt.AlignTop)

        alt_col = QVBoxLayout()
        alt_col.setSpacing(4)
        for key, display in [("whisperx","WhisperX"),
                              ("qwen3_asr","Qwen3-ASR-1.7B"),
                              ("qwen3_aligner","Qwen3-FA-0.6B")]:
            lbl = _Badge(display, ok=False)
            self._backend_labels[key] = lbl
            alt_col.addWidget(lbl)

        alt_wrap = QWidget()
        alt_wrap.setLayout(alt_col)
        grid.addWidget(alt_wrap, row, 1, Qt.AlignTop)
        row += 1

        lay.addLayout(grid)

        # ── model cache path
        self._cache_lbl = QLabel(t("model_cache_dir"))
        self._cache_lbl.setStyleSheet("color:#b0bec5; font-size:11px;")
        self._cache_val = QLabel("")
        self._cache_val.setStyleSheet("color:#80cbc4; font-size:11px;")
        self._cache_val.setWordWrap(True)

        cache_row = QHBoxLayout()
        cache_row.setContentsMargins(0, 0, 0, 0)
        cache_row.addWidget(self._cache_lbl)
        cache_row.addWidget(self._cache_val)
        cache_row.addStretch()
        lay.addLayout(cache_row)

    # ── public: update from API data ──────────────────────────────
    def update_status(self, pipeline_data: dict, aligner_data: dict):
        t = self._i18n.t
        status = pipeline_data.get("status", {})

        # MFA
        mfa = status.get("mfa", {})
        installed = bool(mfa.get("installed"))
        self._mfa_badge.setText(
            t("mfa_installed") if installed else t("mfa_not_installed")
        )
        self._mfa_badge.setOk(installed)
        self._version_val.setText(mfa.get("version") or "—")

        # Language models
        models = mfa.get("models", {})
        for code, row in self._model_rows.items():
            row.set_downloaded(bool(models.get(code, False)))

        # Processing modules
        audio = status.get("audio_processing", {})
        pw_ok = bool(audio.get("pyworld_available"))
        self._module_labels["PyWORLD (DIO/Harvest)"].setOk(pw_ok)

        f0_backends = audio.get("f0_backends", {})
        crepe_ok = bool((f0_backends.get("crepe") or {}).get("available"))
        rmvpe_ok = bool((f0_backends.get("rmvpe") or {}).get("available"))
        self._module_labels["CREPE"].setOk(crepe_ok)
        self._module_labels["RMVPE"].setOk(rmvpe_ok)

        # Alt backends
        alt = aligner_data.get("backends", {})
        for key in ["whisperx", "qwen3_asr", "qwen3_aligner"]:
            ok = bool((alt.get(key) or {}).get("available"))
            if key in self._backend_labels:
                self._backend_labels[key].setOk(ok)

        # cache dir
        cache = (alt.get("models_dir") or
                 status.get("alt_aligners", {}).get("models_dir", ""))
        self._cache_val.setText(str(cache) if cache else "")
        self._cache_val.setVisible(bool(cache))
        self._cache_lbl.setVisible(bool(cache))

    def set_model_downloading(self, lang: str, downloading: bool):
        if lang in self._model_rows:
            self._model_rows[lang].set_downloading(downloading)

    def update_model_names(self):
        """Refresh model display names after language change."""
        t = self._i18n.t
        for code, key in [("cmn","model_cmn"), ("eng","model_eng"),
                          ("jpn","model_jpn"), ("kor","model_kor"),
                          ("yue","model_yue")]:
            if code in self._model_rows:
                self._model_rows[code].refresh_lang()

    # ── signals ───────────────────────────────────────────────────
    def _on_download_requested(self, lang: str):
        self.download_model_requested.emit(lang)

    # ── i18n refresh ──────────────────────────────────────────────
    def _retranslate(self):
        t = self._i18n.t
        self._title_lbl.setText(t("system_status"))
        self._mfa_status_lbl.setText(t("mfa_status"))
        self._version_lbl.setText(t("mfa_version"))
        self._lang_models_lbl.setText(t("language_models"))
        self._proc_lbl.setText(t("processing_modules"))
        self._alt_lbl.setText(t("alt_backends_section"))
        self._cache_lbl.setText(t("model_cache_dir"))
        self.update_model_names()
