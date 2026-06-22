# processor_widget.py — main processing form
from __future__ import annotations

import os
import tempfile
import logging
from pathlib import Path
from typing import Optional

from PySide6.QtCore import (
    Qt, Signal, QTimer, QUrl, QMimeData,
)
from PySide6.QtGui import QFont, QDragEnterEvent, QDropEvent, QClipboard
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QLabel, QPushButton, QRadioButton, QButtonGroup, QComboBox,
    QTextEdit, QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox,
    QProgressBar, QFrame, QTabWidget, QTableWidget, QTableWidgetItem,
    QSizePolicy, QScrollArea, QFileDialog, QMessageBox, QApplication,
    QGroupBox, QToolButton,
)

from i18n import I18n
from api_client import ApiClient, ProcessWorker, JobPollWorker, FileDownloadWorker

logger = logging.getLogger(__name__)

_NOTE_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

def _midi_to_name(n: int) -> str:
    return f"{_NOTE_NAMES[n % 12]}{n // 12 - 1}"

def _fmt_ms(ms: int) -> str:
    s = ms // 1000
    if s < 60:     return f"{s}s"
    if s < 3600:   return f"{s//60}m {s%60}s"
    return f"{s//3600}h {(s%3600)//60}m {s%60}s"

def _fmt_size(b: int) -> str:
    for unit in ("Bytes","KB","MB","GB"):
        if b < 1024: return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} TB"


# ── reusable drop area ────────────────────────────────────────────────────────

class DropArea(QFrame):
    file_selected = Signal(str)   # full path
    cleared       = Signal()

    def __init__(self, accept_exts: list[str], hint: str,
                 tip: str = "", parent=None):
        super().__init__(parent)
        self._exts = [e.lower().lstrip(".") for e in accept_exts]
        self._path: Optional[str] = None

        self.setAcceptDrops(True)
        self.setMinimumHeight(80)
        self.setCursor(Qt.PointingHandCursor)
        self._apply_style(idle=True)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(4)

        icon_row = QHBoxLayout()
        self._icon_lbl = QLabel("📂")
        self._icon_lbl.setFont(QFont("", 20))
        icon_row.addStretch()
        icon_row.addWidget(self._icon_lbl)
        icon_row.addStretch()
        lay.addLayout(icon_row)

        self._hint_lbl = QLabel(hint)
        self._hint_lbl.setAlignment(Qt.AlignCenter)
        self._hint_lbl.setStyleSheet("color:#b0bec5; font-size:12px;")
        lay.addWidget(self._hint_lbl)

        if tip:
            self._tip_lbl = QLabel(tip)
            self._tip_lbl.setAlignment(Qt.AlignCenter)
            self._tip_lbl.setStyleSheet("color:#607d8b; font-size:10px;")
            lay.addWidget(self._tip_lbl)

        btn_row = QHBoxLayout()
        self._browse_btn = QPushButton("Browse…")
        self._browse_btn.setFixedWidth(90)
        self._browse_btn.clicked.connect(self._browse)
        self._clear_btn  = QPushButton("✕")
        self._clear_btn.setFixedWidth(36)
        self._clear_btn.setVisible(False)
        self._clear_btn.clicked.connect(self._clear)
        btn_row.addStretch()
        btn_row.addWidget(self._browse_btn)
        btn_row.addWidget(self._clear_btn)
        btn_row.addStretch()
        lay.addLayout(btn_row)

        self._file_lbl = QLabel("")
        self._file_lbl.setAlignment(Qt.AlignCenter)
        self._file_lbl.setWordWrap(True)
        self._file_lbl.setStyleSheet("color:#4caf50; font-size:11px;")
        self._file_lbl.setVisible(False)
        lay.addWidget(self._file_lbl)

    def _apply_style(self, idle: bool):
        if idle:
            self.setStyleSheet(
                "DropArea { border:2px dashed #3a3f5c; border-radius:8px;"
                " background:#1a1d2e; } "
                "DropArea:hover { border-color:#80cbc4; }"
            )
        else:
            self.setStyleSheet(
                "DropArea { border:2px solid #4caf50; border-radius:8px;"
                " background:#1e2a1e; }"
            )

    def _browse(self):
        exts = " ".join(f"*.{e}" for e in self._exts)
        path, _ = QFileDialog.getOpenFileName(
            self, "Select file", "", f"Files ({exts})"
        )
        if path:
            self._set_path(path)

    def _clear(self):
        self._path = None
        self._apply_style(idle=True)
        self._file_lbl.setVisible(False)
        self._clear_btn.setVisible(False)
        self._icon_lbl.setText("📂")
        self.cleared.emit()

    def _set_path(self, path: str):
        ext = Path(path).suffix.lstrip(".").lower()
        if ext not in self._exts:
            return
        self._path = path
        sz = Path(path).stat().st_size
        self._file_lbl.setText(f"✓  {Path(path).name}  ({_fmt_size(sz)})")
        self._file_lbl.setVisible(True)
        self._clear_btn.setVisible(True)
        self._icon_lbl.setText("✅")
        self._apply_style(idle=False)
        self.file_selected.emit(path)

    @property
    def path(self) -> Optional[str]:
        return self._path

    def set_texts(self, hint: str, browse: str, clear_tip: str = "✕"):
        self._hint_lbl.setText(hint)
        self._browse_btn.setText(browse)
        self._clear_btn.setText(clear_tip)

    # drag & drop
    def dragEnterEvent(self, e: QDragEnterEvent):
        if e.mimeData().hasUrls():
            urls = e.mimeData().urls()
            if urls and Path(urls[0].toLocalFile()).suffix.lstrip(".").lower() in self._exts:
                e.acceptProposedAction()
                return
        e.ignore()

    def dropEvent(self, e: QDropEvent):
        urls = e.mimeData().urls()
        if urls:
            self._set_path(urls[0].toLocalFile())
        e.acceptProposedAction()

    def mousePressEvent(self, e):
        self._browse()


# ── collapsible advanced section ──────────────────────────────────────────────

class _CollapsibleBox(QWidget):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self._toggle = QToolButton()
        self._toggle.setCheckable(True)
        self._toggle.setChecked(False)
        self._toggle.setArrowType(Qt.RightArrow)
        self._toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._toggle.setText(title)
        self._toggle.setStyleSheet(
            "QToolButton { border:none; color:#80cbc4; font-weight:bold; "
            "font-size:12px; padding:4px; }"
        )
        self._toggle.setFixedHeight(30)
        self._toggle.clicked.connect(self._on_toggle)

        self._content = QWidget()
        self._content.setVisible(False)
        self._content.setObjectName("advContent")
        self._content.setStyleSheet(
            "#advContent { border:1px solid #3a3f5c; border-radius:6px;"
            " background:#1a1d2e; padding:8px; }"
        )

        vlay = QVBoxLayout(self)
        vlay.setContentsMargins(0, 0, 0, 0)
        vlay.setSpacing(0)
        vlay.addWidget(self._toggle)
        vlay.addWidget(self._content)

    def _on_toggle(self, checked: bool):
        self._toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        self._content.setVisible(checked)

    def set_title(self, t: str):
        self._toggle.setText(t)

    def set_content_layout(self, layout):
        self._content.setLayout(layout)

    @property
    def is_open(self) -> bool:
        return self._toggle.isChecked()


# ── status badge ──────────────────────────────────────────────────────────────

class _StatusBadge(QLabel):
    def __init__(self, text="", ok=True, parent=None):
        super().__init__(text, parent)
        self._update(ok)

    def _update(self, ok: bool):
        color = "#4caf50" if ok else "#f44336"
        self.setStyleSheet(
            f"QLabel {{ background:{color}; color:#fff; border-radius:3px;"
            f" padding:1px 6px; font-size:11px; }}"
        )

    def set_ok(self, ok: bool, text: str = ""):
        if text:
            self.setText(text)
        self._update(ok)


# ── small section separator ───────────────────────────────────────────────────
def _sep() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.HLine)
    f.setStyleSheet("color:#3a3f5c;")
    return f

def _section_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet("color:#80cbc4; font-size:11px; font-weight:bold; "
                      "padding:2px 0;")
    return lbl


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN WIDGET
# ─────────────────────────────────────────────────────────────────────────────

class ProcessorWidget(QScrollArea):
    status_changed = Signal(dict)   # bubbles up to MainWindow

    def __init__(self, i18n: I18n, api: ApiClient, parent=None):
        super().__init__(parent)
        self._i18n = i18n
        self._api  = api

        # ── state ────────────────────────────────────────────────
        self._mode    = "mfa-only"     # 'mfa-only' | 'full' | 'project-only'
        self._backend = "mfa"          # 'mfa'|'whisperx'|'qwen3_asr'|'qwen3_aligner'
        self._sys_status: dict  = {}
        self._aln_status: dict  = {}
        self._result: dict      = {}
        self._job_id: str       = ""
        self._midi_bpm: float   = 120.0
        self._downloading_langs: set[str] = set()

        # active workers (keep refs to avoid GC)
        self._process_worker: Optional[ProcessWorker]     = None
        self._poll_worker:    Optional[JobPollWorker]     = None
        self._dl_worker:      Optional[FileDownloadWorker]= None

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(1500)
        self._poll_timer.timeout.connect(self._tick_poll)

        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setStyleSheet("QScrollArea { border:none; }")

        inner = QWidget()
        self.setWidget(inner)
        self._main_lay = QVBoxLayout(inner)
        self._main_lay.setContentsMargins(0, 0, 0, 0)
        self._main_lay.setSpacing(14)

        self._build_ui()
        i18n.language_changed.connect(self._retranslate)
        self._update_visibility()

    # ══════════════════════════════════════════════════════════════
    #  UI CONSTRUCTION
    # ══════════════════════════════════════════════════════════════
    def _build_ui(self):
        t = self._i18n.t
        lay = self._main_lay

        # ── card frame
        card = QFrame()
        card.setObjectName("procCard")
        card.setStyleSheet(
            "#procCard { background:#1e2030; border-radius:10px;"
            " border:1px solid #3a3f5c; }"
        )
        lay.addWidget(card)

        form_lay = QVBoxLayout(card)
        form_lay.setContentsMargins(20, 16, 20, 20)
        form_lay.setSpacing(14)

        # ── card header
        hdr = QHBoxLayout()
        self._card_title = QLabel(t("single_file_processing"))
        self._card_title.setFont(QFont("", 13, QFont.Bold))
        self._card_title.setStyleSheet("color:#eceff1;")
        hdr.addWidget(self._card_title)
        hdr.addStretch()

        self._gh_btn = QPushButton(t("github_link"))
        self._gh_btn.setCursor(Qt.PointingHandCursor)
        self._gh_btn.setStyleSheet(
            "QPushButton { background:transparent; color:#80cbc4; border:none; }"
            "QPushButton:hover { color:#b2dfdb; }"
        )
        self._gh_btn.clicked.connect(self._open_github)
        hdr.addWidget(self._gh_btn)

        self._refresh_btn = QPushButton(t("check_status"))
        self._refresh_btn.setCursor(Qt.PointingHandCursor)
        self._refresh_btn.setStyleSheet(
            "QPushButton { background:transparent; color:#80cbc4; border:none; }"
            "QPushButton:hover { color:#b2dfdb; }"
        )
        self._refresh_btn.clicked.connect(self._do_refresh_status)
        hdr.addWidget(self._refresh_btn)
        form_lay.addLayout(hdr)
        form_lay.addWidget(_sep())

        fl = QFormLayout()
        fl.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        fl.setHorizontalSpacing(16)
        fl.setVerticalSpacing(12)

        # ── 1. Audio file
        self._audio_lbl = QLabel(t("audio_file"))
        self._audio_drop = DropArea(
            ["wav","mp3","flac","m4a","aac"],
            t("drop_audio_hint"), t("audio_formats_tip")
        )
        self._audio_drop.file_selected.connect(self._on_audio_selected)
        self._audio_drop.cleared.connect(lambda: None)
        fl.addRow(self._audio_lbl, self._audio_drop)

        # ── 2. Aligner backend (hidden in project-only)
        self._backend_lbl = QLabel(t("aligner_backend"))
        self._backend_frame = self._build_backend_selector()
        fl.addRow(self._backend_lbl, self._backend_frame)

        # ── 3. Aligner device (hidden for MFA + project-only)
        self._device_lbl = QLabel(t("aligner_device"))
        self._device_frame = self._build_device_selector()
        fl.addRow(self._device_lbl, self._device_frame)

        # ── 4. LAB/MIDI (only in project-only)
        self._labmidi_lbl = QLabel(t("lab_midi_file"))
        self._labmidi_drop = DropArea(
            ["lab","mid","midi"],
            t("drop_lab_hint"), t("lab_midi_tip")
        )
        self._labmidi_drop.file_selected.connect(self._on_labmidi_selected)
        self._labmidi_drop.cleared.connect(self._on_labmidi_cleared)
        fl.addRow(self._labmidi_lbl, self._labmidi_drop)

        # ── 5. Text input (hidden in project-only)
        self._text_lbl = QLabel(t("input_text"))
        self._text_edit = QTextEdit()
        self._text_edit.setFixedHeight(90)
        self._text_edit.setPlaceholderText(t("text_placeholder_required"))
        self._text_edit.setStyleSheet(
            "QTextEdit { background:#141624; color:#eceff1;"
            " border:1px solid #3a3f5c; border-radius:4px; }"
        )
        self._text_edit.textChanged.connect(self._on_text_changed)
        self._char_lbl = QLabel(t("char_count", count=0))
        self._char_lbl.setStyleSheet("color:#607d8b; font-size:10px;")
        text_col = QVBoxLayout()
        text_col.setContentsMargins(0,0,0,0)
        text_col.addWidget(self._text_edit)
        text_col.addWidget(self._char_lbl)
        text_wrap = QWidget(); text_wrap.setLayout(text_col)
        fl.addRow(self._text_lbl, text_wrap)

        # ── 6. Language
        self._lang_lbl  = QLabel(t("language_select"))
        self._lang_combo = QComboBox()
        for code, display in [("cmn","lang_cmn"),("eng","lang_eng"),
                               ("jpn","lang_jpn"),("kor","lang_kor"),
                               ("yue","lang_yue")]:
            self._lang_combo.addItem(t(display), code)
        self._lang_combo.setFixedWidth(200)
        fl.addRow(self._lang_lbl, self._lang_combo)

        # ── 7. Processing mode
        self._mode_lbl = QLabel(t("processing_mode"))
        self._mode_frame, self._mode_grp = self._build_radio_group([
            ("mfa-only",      t("mode_mfa_only")),
            ("full",          t("mode_full")),
            ("project-only",  t("mode_project_only")),
        ])
        self._mode_grp.buttonClicked.connect(self._on_mode_changed)
        self._mode_desc = QLabel(t("mode_mfa_only_desc"))
        self._mode_desc.setStyleSheet("color:#607d8b; font-size:10px;")
        mode_col = QVBoxLayout()
        mode_col.setContentsMargins(0,0,0,0)
        mode_col.addWidget(self._mode_frame)
        mode_col.addWidget(self._mode_desc)
        mode_wrap = QWidget(); mode_wrap.setLayout(mode_col)
        fl.addRow(self._mode_lbl, mode_wrap)

        # ── 8. Output format (hidden in mfa-only)
        self._fmt_lbl   = QLabel(t("output_format"))
        self._fmt_combo = QComboBox()
        self._fmt_combo.addItem(t("format_sv"),   "sv")
        self._fmt_combo.addItem(t("format_utau"), "utau")
        self._fmt_combo.setFixedWidth(280)
        fl.addRow(self._fmt_lbl, self._fmt_combo)

        # ── 9. Phoneme conversion (only project-only)
        self._phoneme_lbl = QLabel(t("phoneme_conversion"))
        self._phoneme_frame, self._phoneme_grp = self._build_radio_group([
            ("none",     t("phoneme_none")),
            ("merge",    t("phoneme_merge")),
            ("hiragana", t("phoneme_hiragana")),
            ("katakana", t("phoneme_katakana")),
        ])
        self._phoneme_grp.buttonClicked.connect(self._on_phoneme_changed)
        self._phoneme_desc = QLabel(t("phoneme_none_desc"))
        self._phoneme_desc.setStyleSheet("color:#607d8b; font-size:10px;")
        phoneme_col = QVBoxLayout()
        phoneme_col.setContentsMargins(0,0,0,0)
        phoneme_col.addWidget(self._phoneme_frame)
        phoneme_col.addWidget(self._phoneme_desc)
        phoneme_wrap = QWidget(); phoneme_wrap.setLayout(phoneme_col)
        fl.addRow(self._phoneme_lbl, phoneme_wrap)

        # ── 10. Track name (hidden in mfa-only)
        self._title_lbl  = QLabel(t("track_name"))
        self._title_edit = QLineEdit("Project")
        self._title_edit.setPlaceholderText(t("track_name_placeholder"))
        self._title_edit.setFixedWidth(260)
        fl.addRow(self._title_lbl, self._title_edit)

        form_lay.addLayout(fl)

        # ── 11. Advanced settings
        self._adv_box = _CollapsibleBox(t("advanced_settings"))
        self._build_advanced_content()
        form_lay.addWidget(self._adv_box)

        # ── MIDI info banner
        self._midi_banner = QLabel(t("midi_loaded_banner"))
        self._midi_banner.setWordWrap(True)
        self._midi_banner.setStyleSheet(
            "QLabel { background:#1a2a1a; color:#81c784;"
            " border:1px solid #4caf50; border-radius:4px; padding:6px; }"
        )
        self._midi_banner.setVisible(False)
        form_lay.addWidget(self._midi_banner)

        # ── 12. Action buttons + progress
        btn_row = QHBoxLayout()
        self._start_btn = QPushButton(t("start_processing"))
        self._start_btn.setFixedHeight(40)
        self._start_btn.setFont(QFont("", 12, QFont.Bold))
        self._start_btn.setCursor(Qt.PointingHandCursor)
        self._start_btn.setStyleSheet(
            "QPushButton { background:#00897b; color:#fff; border-radius:6px; }"
            "QPushButton:hover { background:#00acc1; }"
            "QPushButton:disabled { background:#37474f; color:#78909c; }"
        )
        self._start_btn.clicked.connect(self._do_process)

        self._reset_btn = QPushButton(t("reset"))
        self._reset_btn.setFixedHeight(40)
        self._reset_btn.setCursor(Qt.PointingHandCursor)
        self._reset_btn.clicked.connect(self._do_reset)

        self._not_ready_lbl = QLabel(t("system_not_ready_hint"))
        self._not_ready_lbl.setStyleSheet("color:#f44336; font-size:10px;")
        self._not_ready_lbl.setVisible(False)

        btn_row.addWidget(self._start_btn)
        btn_row.addWidget(self._reset_btn)
        btn_row.addWidget(self._not_ready_lbl)
        btn_row.addStretch()
        form_lay.addLayout(btn_row)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setVisible(False)
        form_lay.addWidget(self._progress)

        # ── 13. Result
        self._result_frame = self._build_result_frame()
        form_lay.addWidget(self._result_frame)

        # ── 14. Error
        self._err_lbl = QLabel("")
        self._err_lbl.setWordWrap(True)
        self._err_lbl.setStyleSheet(
            "QLabel { background:#4a1010; color:#ef9a9a; border-radius:4px;"
            " padding:8px; border:1px solid #b71c1c; }"
        )
        self._err_lbl.setVisible(False)
        form_lay.addWidget(self._err_lbl)

        lay.addStretch()

    # ── radio-button group helper ─────────────────────────────────
    def _build_radio_group(self, items: list[tuple[str, str]]) -> tuple[QWidget, QButtonGroup]:
        grp  = QButtonGroup()
        wrap = QWidget()
        row  = QHBoxLayout(wrap)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(12)
        for i, (val, label) in enumerate(items):
            rb = QRadioButton(label)
            rb.setProperty("value", val)
            if i == 0:
                rb.setChecked(True)
            grp.addButton(rb, i)
            row.addWidget(rb)
        row.addStretch()
        return wrap, grp

    def _grp_value(self, grp: QButtonGroup) -> str:
        btn = grp.checkedButton()
        return btn.property("value") if btn else ""

    def _grp_set(self, grp: QButtonGroup, val: str):
        for btn in grp.buttons():
            if btn.property("value") == val:
                btn.setChecked(True)
                return

    # ── backend selector ──────────────────────────────────────────
    def _build_backend_selector(self) -> QWidget:
        t = self._i18n.t
        wrap = QWidget()
        col  = QVBoxLayout(wrap)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(4)

        self._backend_grp = QButtonGroup()
        backends = [
            ("mfa",            "MFA"),
            ("whisperx",       "WhisperX"),
            ("qwen3_asr",      "Qwen3-ASR"),
            ("qwen3_aligner",  "Qwen3-FA"),
        ]
        self._backend_badges: dict[str, _StatusBadge] = {}
        self._backend_rb: dict[str, QRadioButton]     = {}

        for i, (val, label) in enumerate(backends):
            row = QHBoxLayout()
            row.setSpacing(6)
            rb = QRadioButton(label)
            rb.setProperty("value", val)
            if i == 0:
                rb.setChecked(True)
            self._backend_grp.addButton(rb, i)
            self._backend_rb[val] = rb

            badge = _StatusBadge("✗", ok=False)
            badge.setFixedWidth(24)
            self._backend_badges[val] = badge

            row.addWidget(rb)
            row.addWidget(badge)
            row.addStretch()
            col.addLayout(row)

        self._backend_desc = QLabel(t("backend_mfa_desc"))
        self._backend_desc.setWordWrap(True)
        self._backend_desc.setStyleSheet("color:#607d8b; font-size:10px;")
        col.addWidget(self._backend_desc)

        self._backend_grp.buttonClicked.connect(self._on_backend_changed)
        return wrap

    # ── aligner device selector ───────────────────────────────────
    def _build_device_selector(self) -> QWidget:
        t = self._i18n.t
        wrap = QWidget()
        col  = QVBoxLayout(wrap)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(4)

        items = [("auto",t("device_auto")),("cpu",t("device_cpu")),("cuda",t("device_cuda"))]
        self._dev_frame, self._dev_grp = self._build_radio_group(items)
        col.addWidget(self._dev_frame)

        self._dev_desc = QLabel(t("auto_device_hint"))
        self._dev_desc.setWordWrap(True)
        self._dev_desc.setStyleSheet("color:#607d8b; font-size:10px;")
        col.addWidget(self._dev_desc)

        self._dev_grp.buttonClicked.connect(self._on_device_changed)
        return wrap

    # ── advanced settings content ─────────────────────────────────
    def _build_advanced_content(self):
        t = self._i18n.t
        adv = QGridLayout()
        adv.setHorizontalSpacing(16)
        adv.setVerticalSpacing(10)
        r = 0

        # BPM
        self._bpm_lbl = QLabel(t("bpm"))
        self._bpm_lbl.setStyleSheet("color:#b0bec5;")
        self._bpm_spin = QSpinBox()
        self._bpm_spin.setRange(20, 300)
        self._bpm_spin.setValue(120)
        self._bpm_spin.setFixedWidth(90)
        adv.addWidget(self._bpm_lbl,  r, 0, Qt.AlignRight)
        adv.addWidget(self._bpm_spin, r, 1)
        r += 1

        # Base pitch
        self._pitch_lbl  = QLabel(t("base_pitch"))
        self._pitch_lbl.setStyleSheet("color:#b0bec5;")
        pitch_row = QHBoxLayout()
        self._pitch_spin = QSpinBox()
        self._pitch_spin.setRange(12, 108)
        self._pitch_spin.setValue(60)
        self._pitch_spin.setFixedWidth(70)
        self._pitch_note = QLabel(_midi_to_name(60))
        self._pitch_note.setStyleSheet("color:#80cbc4; font-weight:bold;")
        self._pitch_spin.valueChanged.connect(
            lambda v: self._pitch_note.setText(_midi_to_name(v))
        )
        pitch_row.addWidget(self._pitch_spin)
        pitch_row.addWidget(self._pitch_note)
        pitch_row.addStretch()
        adv.addWidget(self._pitch_lbl, r, 0, Qt.AlignRight)
        adv.addLayout(pitch_row, r, 1)
        r += 1

        adv.addWidget(_section_label(t("pitch_control")), r, 0, 1, 2)
        r += 1

        # Auto note pitch
        self._auto_pitch_lbl = QLabel(t("auto_note_pitch"))
        self._auto_pitch_lbl.setStyleSheet("color:#b0bec5;")
        self._auto_pitch_chk = QCheckBox(t("auto_note_pitch_on"))
        self._auto_pitch_chk.setChecked(True)
        adv.addWidget(self._auto_pitch_lbl, r, 0, Qt.AlignRight)
        adv.addWidget(self._auto_pitch_chk, r, 1)
        r += 1

        # Export pitch line
        self._exp_pitch_lbl = QLabel(t("export_pitch_line"))
        self._exp_pitch_lbl.setStyleSheet("color:#b0bec5;")
        self._exp_pitch_chk = QCheckBox(t("export_pitch_line_on"))
        self._exp_pitch_chk.setChecked(True)
        adv.addWidget(self._exp_pitch_lbl, r, 0, Qt.AlignRight)
        adv.addWidget(self._exp_pitch_chk, r, 1)
        r += 1

        adv.addWidget(_section_label(t("f0_method_section")), r, 0, 1, 2)
        r += 1

        # F0 method
        self._f0_method_lbl = QLabel(t("f0_method"))
        self._f0_method_lbl.setStyleSheet("color:#b0bec5;")
        items = [("dio",t("f0_dio")),("harvest",t("f0_harvest")),
                 ("crepe",t("f0_crepe")),("rmvpe",t("f0_rmvpe"))]
        self._f0_frame, self._f0_grp = self._build_radio_group(items)
        adv.addWidget(self._f0_method_lbl, r, 0, Qt.AlignRight | Qt.AlignTop)
        adv.addWidget(self._f0_frame, r, 1)
        self._f0_grp.buttonClicked.connect(self._on_f0_changed)
        r += 1

        # CREPE model
        self._crepe_lbl = QLabel(t("crepe_model_size"))
        self._crepe_lbl.setStyleSheet("color:#b0bec5;")
        c_items = [("full",t("crepe_full")),("tiny",t("crepe_tiny"))]
        self._crepe_frame, self._crepe_grp = self._build_radio_group(c_items)
        adv.addWidget(self._crepe_lbl,  r, 0, Qt.AlignRight)
        adv.addWidget(self._crepe_frame, r, 1)
        r += 1

        # F0 device
        self._f0dev_lbl = QLabel(t("f0_device"))
        self._f0dev_lbl.setStyleSheet("color:#b0bec5;")
        d_items = [("auto",t("device_auto")),("cpu",t("device_cpu")),
                   ("cuda",t("device_cuda"))]
        self._f0dev_frame, self._f0dev_grp = self._build_radio_group(d_items)
        adv.addWidget(self._f0dev_lbl,   r, 0, Qt.AlignRight)
        adv.addWidget(self._f0dev_frame, r, 1)
        r += 1

        # Precision
        self._prec_lbl = QLabel(t("precision"))
        self._prec_lbl.setStyleSheet("color:#b0bec5;")
        p_items = [("single",t("precision_single")),("double",t("precision_double"))]
        self._prec_frame, self._prec_grp = self._build_radio_group(p_items)
        adv.addWidget(self._prec_lbl,   r, 0, Qt.AlignRight)
        adv.addWidget(self._prec_frame, r, 1)
        r += 1

        # F0 smooth
        self._smooth_lbl = QLabel(t("f0_smooth"))
        self._smooth_lbl.setStyleSheet("color:#b0bec5;")
        self._smooth_chk = QCheckBox(t("f0_smooth"))
        self._smooth_chk.setChecked(True)
        adv.addWidget(self._smooth_lbl, r, 0, Qt.AlignRight)
        adv.addWidget(self._smooth_chk, r, 1)
        r += 1

        # Smooth window
        self._win_lbl  = QLabel(t("f0_smooth_window"))
        self._win_lbl.setStyleSheet("color:#b0bec5;")
        win_row = QHBoxLayout()
        self._win_spin = QSpinBox()
        self._win_spin.setRange(1, 21)
        self._win_spin.setValue(5)
        self._win_spin.setSingleStep(2)
        self._win_spin.setFixedWidth(70)
        win_tip = QLabel(t("smooth_tip"))
        win_tip.setStyleSheet("color:#607d8b; font-size:10px;")
        win_row.addWidget(self._win_spin)
        win_row.addWidget(win_tip)
        win_row.addStretch()
        adv.addWidget(self._win_lbl, r, 0, Qt.AlignRight)
        adv.addLayout(win_row, r, 1)
        r += 1

        # F0 floor
        self._floor_lbl  = QLabel(t("f0_floor_hz"))
        self._floor_lbl.setStyleSheet("color:#b0bec5;")
        self._floor_spin = QSpinBox()
        self._floor_spin.setRange(40, 200)
        self._floor_spin.setValue(71)
        self._floor_spin.setSuffix(" Hz")
        self._floor_spin.setFixedWidth(100)
        adv.addWidget(self._floor_lbl,  r, 0, Qt.AlignRight)
        adv.addWidget(self._floor_spin, r, 1)
        r += 1

        # F0 ceil
        self._ceil_lbl  = QLabel(t("f0_ceil_hz"))
        self._ceil_lbl.setStyleSheet("color:#b0bec5;")
        self._ceil_spin = QSpinBox()
        self._ceil_spin.setRange(300, 2000)
        self._ceil_spin.setValue(800)
        self._ceil_spin.setSuffix(" Hz")
        self._ceil_spin.setFixedWidth(100)
        f0_range_tip = QLabel(t("f0_range_tip"))
        f0_range_tip.setStyleSheet("color:#607d8b; font-size:10px;")
        ceil_col = QVBoxLayout()
        ceil_col.addWidget(self._ceil_spin)
        ceil_col.addWidget(f0_range_tip)
        adv.addWidget(self._ceil_lbl,  r, 0, Qt.AlignRight | Qt.AlignTop)
        adv.addLayout(ceil_col,        r, 1)

        self._adv_box.set_content_layout(adv)

    # ── result section ────────────────────────────────────────────
    def _build_result_frame(self) -> QWidget:
        t = self._i18n.t
        frame = QWidget()
        frame.setVisible(False)
        col = QVBoxLayout(frame)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(10)

        self._result_title = QLabel(t("result_title"))
        self._result_title.setFont(QFont("", 12, QFont.Bold))
        self._result_title.setStyleSheet("color:#4caf50;")
        col.addWidget(self._result_title)

        # Info row
        info_row = QHBoxLayout()
        self._res_time = QLabel("")
        self._res_time.setStyleSheet("color:#b0bec5; font-size:11px;")
        self._res_segs = QLabel("")
        self._res_segs.setStyleSheet("color:#b0bec5; font-size:11px;")
        info_row.addWidget(self._res_time)
        info_row.addSpacing(20)
        info_row.addWidget(self._res_segs)
        info_row.addStretch()
        col.addLayout(info_row)

        # Tab widget
        self._tabs = QTabWidget()
        self._tabs.setStyleSheet(
            "QTabBar::tab { padding:6px 14px; color:#b0bec5; }"
            "QTabBar::tab:selected { color:#80cbc4; border-bottom:2px solid #80cbc4; }"
        )

        # LAB tab
        self._lab_tab = QWidget()
        lt = QVBoxLayout(self._lab_tab)
        self._lab_text = QTextEdit()
        self._lab_text.setReadOnly(True)
        self._lab_text.setFont(QFont("Consolas,Courier New", 10))
        self._lab_text.setStyleSheet(
            "QTextEdit { background:#141624; color:#a5d6a7;"
            " border:1px solid #3a3f5c; border-radius:4px; }"
        )
        self._lab_text.setFixedHeight(200)
        lt.addWidget(self._lab_text)
        lab_btns = QHBoxLayout()
        self._copy_lab_btn = QPushButton(t("copy_lab"))
        self._copy_lab_btn.clicked.connect(self._do_copy_lab)
        self._dl_lab_btn   = QPushButton(t("download_lab"))
        self._dl_lab_btn.clicked.connect(self._do_download_lab)
        lab_btns.addWidget(self._copy_lab_btn)
        lab_btns.addWidget(self._dl_lab_btn)
        lab_btns.addStretch()
        lt.addLayout(lab_btns)

        # File info tab
        self._info_tab = QWidget()
        it = QVBoxLayout(self._info_tab)
        self._info_text = QTextEdit()
        self._info_text.setReadOnly(True)
        self._info_text.setFixedHeight(200)
        it.addWidget(self._info_text)

        # Details tab
        self._details_tab = QWidget()
        dt = QVBoxLayout(self._details_tab)
        self._details_table = QTableWidget(3, 3)
        self._details_table.setHorizontalHeaderLabels([
            t("col_stage"), t("col_status"), t("col_details")
        ])
        self._details_table.setFixedHeight(140)
        self._details_table.horizontalHeader().setStretchLastSection(True)
        self._details_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._details_table.setStyleSheet(
            "QTableWidget { background:#141624; color:#eceff1;"
            " border:1px solid #3a3f5c; gridline-color:#3a3f5c; }"
            "QHeaderView::section { background:#1e2030; color:#80cbc4; }"
        )
        dt.addWidget(self._details_table)

        self._tabs.addTab(self._lab_tab,     t("tab_lab_content"))
        self._tabs.addTab(self._info_tab,    t("tab_file_info"))
        self._tabs.addTab(self._details_tab, t("tab_details"))
        col.addWidget(self._tabs)

        # Action buttons
        act_row = QHBoxLayout()
        self._dl_proj_btn = QPushButton(t("download_project"))
        self._dl_proj_btn.setVisible(False)
        self._dl_proj_btn.clicked.connect(self._do_download_project)
        self._next_btn = QPushButton(t("process_next"))
        self._next_btn.clicked.connect(self._do_reset)
        act_row.addWidget(self._dl_proj_btn)
        act_row.addWidget(self._next_btn)
        act_row.addStretch()
        col.addLayout(act_row)

        return frame

    # ══════════════════════════════════════════════════════════════
    #  VISIBILITY & STATE
    # ══════════════════════════════════════════════════════════════
    def _update_visibility(self):
        is_proj  = (self._mode == "project-only")
        is_full  = (self._mode == "full")
        is_mfa   = (self._backend == "mfa")
        show_dev = (not is_mfa) and (not is_proj)

        self._backend_lbl.setVisible(not is_proj)
        self._backend_frame.setVisible(not is_proj)
        self._device_lbl.setVisible(show_dev)
        self._device_frame.setVisible(show_dev)
        self._labmidi_lbl.setVisible(is_proj)
        self._labmidi_drop.setVisible(is_proj)
        self._text_lbl.setVisible(not is_proj)
        self._text_edit.parent().setVisible(not is_proj)
        self._lang_lbl.setVisible(not is_proj)
        self._lang_combo.setVisible(not is_proj)
        self._fmt_lbl.setVisible(not (self._mode == "mfa-only"))
        self._fmt_combo.setVisible(not (self._mode == "mfa-only"))
        self._phoneme_lbl.setVisible(is_proj)
        self._phoneme_frame.parent().setVisible(is_proj)
        self._title_lbl.setVisible(not (self._mode == "mfa-only"))
        self._title_edit.setVisible(not (self._mode == "mfa-only"))
        self._adv_box.setVisible(not (self._mode == "mfa-only"))
        self._update_start_enabled()

    def _update_start_enabled(self):
        is_proj  = (self._mode == "project-only")
        is_mfa   = (self._backend == "mfa")
        has_audio = bool(self._audio_drop.path)
        has_labmidi = bool(self._labmidi_drop.path)
        has_text = bool(self._text_edit.toPlainText().strip())
        text_optional = self._backend in ("whisperx","qwen3_asr")
        backend_ready = self._backend_ready()

        if is_proj:
            ok = has_audio and has_labmidi
        else:
            ok = has_audio and (has_text or text_optional) and backend_ready

        self._start_btn.setEnabled(ok)
        self._not_ready_lbl.setVisible(not backend_ready and not is_proj)

    def _backend_ready(self) -> bool:
        if self._backend == "mfa":
            mfa = self._sys_status.get("mfa", {})
            if not mfa.get("installed"):
                return False
            lang = self._lang_combo.currentData() or "cmn"
            return bool((mfa.get("models") or {}).get(lang, False))
        info = self._aln_status.get(self._backend, {})
        return bool(info.get("available"))

    # ── backend badge updates ──────────────────────────────────────
    def _update_backend_badges(self):
        mfa_ok = bool(self._sys_status.get("mfa", {}).get("installed"))
        self._backend_badges["mfa"].set_ok(mfa_ok, "✓" if mfa_ok else "✗")
        for key in ["whisperx","qwen3_asr","qwen3_aligner"]:
            ok = bool((self._aln_status.get(key) or {}).get("available"))
            self._backend_badges[key].set_ok(ok, "✓" if ok else "✗")

    # ── MIDI helpers ──────────────────────────────────────────────
    def _parse_midi_bpm(self, path: str) -> float:
        """Quick BPM extraction from MIDI without external libs."""
        try:
            with open(path, "rb") as fh:
                data = fh.read(2048)
            for i in range(len(data) - 5):
                if data[i] == 0xFF and data[i+1] == 0x51 and data[i+2] == 0x03:
                    us = (data[i+3] << 16) | (data[i+4] << 8) | data[i+5]
                    if us > 0:
                        return round(60_000_000 / us, 1)
        except Exception:
            pass
        return 120.0

    # ══════════════════════════════════════════════════════════════
    #  EVENT HANDLERS
    # ══════════════════════════════════════════════════════════════
    def _on_audio_selected(self, path: str):
        sz = Path(path).stat().st_size
        if sz > 512 * 1024 * 1024:
            self._show_err(self._i18n.t("msg_audio_too_large"))
        self._update_start_enabled()

    def _on_labmidi_selected(self, path: str):
        ext = Path(path).suffix.lower().lstrip(".")
        if ext in ("mid","midi"):
            self._midi_bpm = self._parse_midi_bpm(path)
            self._bpm_spin.setValue(int(self._midi_bpm))
            self._bpm_spin.setEnabled(False)
            self._pitch_spin.setEnabled(False)
            self._auto_pitch_chk.setEnabled(False)
            self._midi_banner.setVisible(True)
        self._update_start_enabled()

    def _on_labmidi_cleared(self):
        self._bpm_spin.setEnabled(True)
        self._pitch_spin.setEnabled(True)
        self._auto_pitch_chk.setEnabled(True)
        self._midi_banner.setVisible(False)
        self._update_start_enabled()

    def _on_text_changed(self):
        n = len(self._text_edit.toPlainText())
        self._char_lbl.setText(self._i18n.t("char_count", count=n))
        self._update_start_enabled()

    def _on_mode_changed(self, btn):
        self._mode = btn.property("value")
        descs = {
            "mfa-only":     "mode_mfa_only_desc",
            "full":         "mode_full_desc",
            "project-only": "mode_project_only_desc",
        }
        self._mode_desc.setText(self._i18n.t(descs.get(self._mode,"")))
        self._update_visibility()

    def _on_backend_changed(self, btn):
        self._backend = btn.property("value")
        descs = {
            "mfa":           "backend_mfa_desc",
            "whisperx":      "backend_whisperx_desc",
            "qwen3_asr":     "backend_qwen3_asr_desc",
            "qwen3_aligner": "backend_qwen3_aligner_desc",
        }
        self._backend_desc.setText(self._i18n.t(descs.get(self._backend,"")))
        text_opt = self._backend in ("whisperx","qwen3_asr")
        self._text_edit.setPlaceholderText(
            self._i18n.t("text_placeholder_optional" if text_opt
                         else "text_placeholder_required")
        )
        self._update_visibility()

    def _on_device_changed(self, btn):
        val = btn.property("value")
        descs = {
            "auto": "auto_device_hint",
            "cpu":  "cpu_mode_hint",
            "cuda": "whisperx_gpu_hint" if self._backend == "whisperx"
                    else "qwen3_gpu_hint",
        }
        self._dev_desc.setText(self._i18n.t(descs.get(val,"")))

    def _on_phoneme_changed(self, btn):
        val = btn.property("value")
        descs = {
            "none":     "phoneme_none_desc",
            "merge":    "phoneme_merge_desc",
            "hiragana": "phoneme_hiragana_desc",
            "katakana": "phoneme_katakana_desc",
        }
        self._phoneme_desc.setText(self._i18n.t(descs.get(val,"")))

    def _on_f0_changed(self, btn):
        val = btn.property("value")
        show = val in ("crepe","rmvpe")
        self._crepe_lbl.setVisible(show)
        self._crepe_frame.setVisible(show)
        self._f0dev_lbl.setVisible(show)
        self._f0dev_frame.setVisible(show)

    # ══════════════════════════════════════════════════════════════
    #  PROCESSING
    # ══════════════════════════════════════════════════════════════
    def _collect_params(self) -> dict:
        return {
            "text":            self._text_edit.toPlainText().strip(),
            "language":        self._lang_combo.currentData() or "cmn",
            "aligner_backend": self._backend,
            "aligner_device":  self._grp_value(self._dev_grp),
            "output_format":   self._fmt_combo.currentData() or "sv",
            "project_title":   self._title_edit.text().strip() or "Project",
            "bpm":             self._bpm_spin.value(),
            "base_pitch":      self._pitch_spin.value(),
            "auto_note_pitch": self._auto_pitch_chk.isChecked(),
            "export_pitch_line": self._exp_pitch_chk.isChecked(),
            "f0_method":       self._grp_value(self._f0_grp),
            "crepe_model":     self._grp_value(self._crepe_grp),
            "f0_device":       self._grp_value(self._f0dev_grp),
            "precision":       self._grp_value(self._prec_grp),
            "f0_smooth":       self._smooth_chk.isChecked(),
            "f0_smooth_window": self._win_spin.value(),
            "f0_floor":        self._floor_spin.value(),
            "f0_ceil":         self._ceil_spin.value(),
            "phoneme_mode":    self._grp_value(self._phoneme_grp),
        }

    def _do_process(self):
        t = self._i18n.t
        audio = self._audio_drop.path
        if not audio:
            self._show_err(t("msg_select_audio")); return

        lab   = None
        midi  = None
        if self._mode == "project-only":
            p = self._labmidi_drop.path
            if not p:
                self._show_err(t("msg_select_lab_midi")); return
            ext = Path(p).suffix.lower().lstrip(".")
            if ext == "lab":   lab  = p
            elif ext in ("mid","midi"): midi = p
        else:
            txt = self._text_edit.toPlainText().strip()
            if not txt and self._backend not in ("whisperx","qwen3_asr"):
                self._show_err(t("msg_enter_text")); return
            if not self._backend_ready():
                self._show_err(t("msg_backend_not_ready")); return

        self._set_processing(True)
        self._reset_stages()
        params = self._collect_params()

        self._process_worker = self._api.process(
            self._mode, audio, lab, midi, params
        )
        self._process_worker.succeeded.connect(self._on_submit_ok)
        self._process_worker.failed.connect(self._on_submit_fail)
        self._process_worker.start()

    def _on_submit_ok(self, data: dict):
        job_id = data.get("job_id")
        if job_id:
            self._job_id = job_id
            self._poll_timer.start()
        else:
            # synchronous result (rare)
            self._on_job_done(data)

    def _on_submit_fail(self, err: str):
        self._set_processing(False)
        self._show_err(err)

    def _tick_poll(self):
        if not self._job_id:
            self._poll_timer.stop()
            return
        self._poll_worker = self._api.poll_job(self._job_id)
        self._poll_worker.succeeded.connect(self._on_poll_ok)
        self._poll_worker.failed.connect(self._on_poll_fail)
        self._poll_worker.start()

    def _on_poll_ok(self, data: dict):
        job = data.get("job", {})
        status = job.get("status","")
        if status == "done":
            self._poll_timer.stop()
            self._job_id = ""
            self._on_job_done(job.get("result", {}))
        elif status == "failed":
            self._poll_timer.stop()
            self._job_id = ""
            self._set_processing(False)
            self._show_err(job.get("error","Processing failed"))
        else:
            # still running, bump progress
            cur = self._progress.value()
            if cur < 85:
                self._progress.setValue(cur + 2)

    def _on_poll_fail(self, err: str):
        # network blip — keep polling
        logger.warning("Poll error (will retry): %s", err)

    def _on_job_done(self, result: dict):
        self._set_processing(False)
        self._progress.setValue(100)
        self._result = result

        lab_content  = result.get("lab_content","")
        project_path = (result.get("project_path") or
                        result.get("output_path",""))
        lab_path     = result.get("lab_path","")
        proc_time    = result.get("processing_time", 0)
        segments     = result.get("segments", 0)
        cfg          = result.get("config", {})
        fmt          = result.get("project_format","sv")

        t = self._i18n.t

        # update details stages
        if self._mode == "mfa-only":
            self._set_stage(0, "done",    f"{segments} segments")
            self._set_stage(1, "skipped", t("status_skipped"))
            self._set_stage(2, "skipped", t("status_skipped"))
        elif self._mode == "project-only":
            self._set_stage(0, "skipped", t("status_skipped"))
            self._set_stage(1, "done",    "F0 done")
            self._set_stage(2, "done",    Path(project_path).name if project_path else "")
        else:
            self._set_stage(0, "done", "alignment done")
            self._set_stage(1, "done", "F0 done")
            self._set_stage(2, "done", Path(project_path).name if project_path else "")

        # LAB tab
        self._lab_text.setPlainText(lab_content)

        # File info tab
        lines = []
        if lab_path:
            lines.append(f"{t('lab_path_label')}  {lab_path}")
        if project_path:
            lines.append(f"{t('project_path_label')}  {project_path}")
            fmt_str = t("sv_studio") if fmt == "sv" else t("openutau")
            lines.append(f"{t('output_format_label')}  {fmt_str}")
        if cfg:
            lines.append("")
            lines.append(t("config_label"))
            lines.append(t("cfg_bpm", v=cfg.get("bpm",120)))
            bp = cfg.get("base_pitch",60)
            lines.append(t("cfg_base_pitch", note=_midi_to_name(bp), midi=bp))
            on_off = lambda v: t("state_on") if v else t("state_off")
            lines.append(t("cfg_auto_pitch",   state=on_off(cfg.get("auto_note_pitch"))))
            lines.append(t("cfg_export_pitch", state=on_off(cfg.get("export_pitch_line"))))
            lines.append(t("cfg_f0_method",    method=(cfg.get("f0_method","?").upper())))
            lines.append(t("cfg_device",       device=cfg.get("f0_device","auto")))
            prec = t("precision_double") if cfg.get("use_double_precision") else t("precision_single")
            lines.append(t("cfg_precision",    prec=prec))
        self._info_text.setPlainText("\n".join(lines))

        # summary
        self._res_time.setText(t("processing_time", time=_fmt_ms(proc_time)))
        if segments:
            self._res_segs.setText(t("segments_label", count=segments))

        # show download project button
        self._dl_proj_btn.setVisible(bool(project_path))
        self._result_frame.setVisible(True)
        self._err_lbl.setVisible(False)

    # ── stage helpers ──────────────────────────────────────────────
    def _reset_stages(self):
        t = self._i18n.t
        stages = [t("stage_alignment"), t("stage_f0"), t("stage_project")]
        for i, stage in enumerate(stages):
            for col, text in enumerate([stage, t("status_waiting"), "—"]):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter if col == 1 else Qt.AlignLeft)
                self._details_table.setItem(i, col, item)

    def _set_stage(self, row: int, status: str, detail: str):
        t = self._i18n.t
        status_map = {
            "done":    ("status_done",    "#4caf50"),
            "running": ("status_running", "#ff9800"),
            "skipped": ("status_skipped", "#9e9e9e"),
            "failed":  ("status_failed",  "#f44336"),
            "waiting": ("status_waiting", "#607d8b"),
        }
        key, color = status_map.get(status, ("status_waiting","#607d8b"))
        item = QTableWidgetItem(t(key))
        item.setForeground(Qt.white)
        item.setBackground(__import__("PySide6.QtGui",fromlist=["QColor"]).QColor(color))
        item.setTextAlignment(Qt.AlignCenter)
        self._details_table.setItem(row, 1, item)
        self._details_table.setItem(row, 2, QTableWidgetItem(detail))

    # ── UI helpers ────────────────────────────────────────────────
    def _set_processing(self, on: bool):
        t = self._i18n.t
        self._start_btn.setEnabled(not on)
        self._reset_btn.setEnabled(not on)
        self._progress.setVisible(on)
        if on:
            self._progress.setValue(10)
            self._result_frame.setVisible(False)
            self._err_lbl.setVisible(False)
            self._start_btn.setText(t("processing_btn", percent=0))
        else:
            self._start_btn.setText(t("start_processing"))
            self._update_start_enabled()

    def _show_err(self, msg: str):
        self._err_lbl.setText(f"❌  {msg}")
        self._err_lbl.setVisible(True)

    def _do_reset(self):
        self._poll_timer.stop()
        self._job_id = ""
        self._audio_drop._clear()
        self._labmidi_drop._clear()
        self._text_edit.clear()
        self._title_edit.setText("Project")
        self._result_frame.setVisible(False)
        self._err_lbl.setVisible(False)
        self._progress.setVisible(False)
        self._progress.setValue(0)
        self._midi_banner.setVisible(False)
        self._bpm_spin.setEnabled(True)
        self._pitch_spin.setEnabled(True)
        self._auto_pitch_chk.setEnabled(True)
        self._set_processing(False)

    def _do_copy_lab(self):
        t = self._i18n.t
        txt = self._lab_text.toPlainText()
        if not txt:
            self._show_err(t("msg_no_lab_content")); return
        QApplication.clipboard().setText(txt)

    def _do_download_lab(self):
        t = self._i18n.t
        txt = self._lab_text.toPlainText()
        if not txt:
            self._show_err(t("msg_no_lab_content")); return
        stem = "alignment"
        pp = (self._result.get("project_path") or
              self._result.get("output_path",""))
        if pp:
            stem = Path(pp).stem
        elif self._result.get("lab_path"):
            stem = Path(self._result["lab_path"]).stem
        elif self._audio_drop.path:
            stem = Path(self._audio_drop.path).stem
        save, _ = QFileDialog.getSaveFileName(
            self, "Save LAB", f"{stem}.lab", "LAB files (*.lab)"
        )
        if save:
            Path(save).write_text(txt, encoding="utf-8")

    def _do_download_project(self):
        result = self._result
        pp = result.get("project_path") or result.get("output_path","")
        if not pp:
            return
        filename = Path(pp).name
        save, _ = QFileDialog.getSaveFileName(
            self, "Save Project", filename,
            "Project files (*.svp *.ustx)"
        )
        if not save:
            return
        self._dl_proj_btn.setEnabled(False)
        self._dl_worker = self._api.download_file(filename, save)
        self._dl_worker.succeeded.connect(lambda _: self._dl_proj_btn.setEnabled(True))
        self._dl_worker.failed.connect(lambda e: (
            self._show_err(e),
            self._dl_proj_btn.setEnabled(True)
        ))
        self._dl_worker.start()

    # ── public: update from MainWindow ────────────────────────────
    def update_system_status(self, pipeline_data: dict, aligner_data: dict):
        self._sys_status = pipeline_data.get("status", {})
        backends = aligner_data.get("backends", {})
        self._aln_status = {k: v for k, v in backends.items() if k != "mfa"}
        self._update_backend_badges()
        self._update_start_enabled()
        self.status_changed.emit(self._sys_status)

    def _do_refresh_status(self):
        from api_client import StatusWorker
        w = self._api.status()
        w.succeeded.connect(lambda d: self.update_system_status(d["pipeline"],d["aligner"]))
        w.failed.connect(lambda e: self._show_err(
            self._i18n.t("msg_backend_unreachable") + f": {e}"
        ))
        w.start()

    def _open_github(self):
        from PySide6.QtGui import QDesktopServices
        QDesktopServices.openUrl(QUrl("https://github.com/liuhua520-svg/gpt-sovits-mfa-aligner-tsubaki"))

    # ── i18n refresh ──────────────────────────────────────────────
    def _retranslate(self):
        t = self._i18n.t
        self._card_title.setText(t("single_file_processing"))
        self._gh_btn.setText(t("github_link"))
        self._refresh_btn.setText(t("check_status"))
        self._audio_lbl.setText(t("audio_file"))
        self._audio_drop.set_texts(t("drop_audio_hint"), t("browse"))
        self._backend_lbl.setText(t("aligner_backend"))
        self._device_lbl.setText(t("aligner_device"))
        self._labmidi_lbl.setText(t("lab_midi_file"))
        self._labmidi_drop.set_texts(t("drop_lab_hint"), t("browse"))
        self._text_lbl.setText(t("input_text"))
        self._lang_lbl.setText(t("language_select"))
        self._mode_lbl.setText(t("processing_mode"))
        self._fmt_lbl.setText(t("output_format"))
        self._phoneme_lbl.setText(t("phoneme_conversion"))
        self._title_lbl.setText(t("track_name"))
        self._title_edit.setPlaceholderText(t("track_name_placeholder"))
        self._adv_box.set_title(t("advanced_settings"))
        self._start_btn.setText(t("start_processing"))
        self._reset_btn.setText(t("reset"))
        self._not_ready_lbl.setText(t("system_not_ready_hint"))
        self._result_title.setText(t("result_title"))
        self._copy_lab_btn.setText(t("copy_lab"))
        self._dl_lab_btn.setText(t("download_lab"))
        self._dl_proj_btn.setText(t("download_project"))
        self._next_btn.setText(t("process_next"))
        self._midi_banner.setText(t("midi_loaded_banner"))
        self._tabs.setTabText(0, t("tab_lab_content"))
        self._tabs.setTabText(1, t("tab_file_info"))
        self._tabs.setTabText(2, t("tab_details"))

        # rebuild language combo labels
        lang_keys = [("cmn","lang_cmn"),("eng","lang_eng"),
                     ("jpn","lang_jpn"),("kor","lang_kor"),("yue","lang_yue")]
        for i, (_, key) in enumerate(lang_keys):
            self._lang_combo.setItemText(i, t(key))

        # rebuild output format labels
        self._fmt_combo.setItemText(0, t("format_sv"))
        self._fmt_combo.setItemText(1, t("format_utau"))

        self._reset_stages()
