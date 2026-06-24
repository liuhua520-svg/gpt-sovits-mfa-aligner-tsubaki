# bridge.py — Python-QML bridge objects
# All state that QML needs is exposed here via Property / Signal / Slot.
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, Property, Signal, Slot, QTimer

from i18n import I18n
from api_client import ApiClient

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  I18nBridge
# ─────────────────────────────────────────────────────────────────────────────

class I18nBridge(QObject):
    """Wraps I18n for QML.  All UI text goes through i18n.t(key)."""

    languageChanged = Signal()

    def __init__(self, i18n: I18n, parent=None):
        super().__init__(parent)
        self._i18n = i18n
        i18n.language_changed.connect(self.languageChanged)

    # ── translate ──────────────────────────────────────────────────
    @Slot(str, result=str)
    def t(self, key: str) -> str:
        return self._i18n.t(key)

    @Slot(str, "QVariantMap", result=str)
    def tf(self, key: str, kwargs: dict) -> str:
        """Translate with keyword-format args (e.g. tf("msg_ok", {"lang": "cmn"}))."""
        return self._i18n.t(key, **kwargs)

    # ── language switching ─────────────────────────────────────────
    @Slot(str)
    def setLanguage(self, code: str):
        self._i18n.set_language(code)

    @Property(str, notify=languageChanged)
    def currentLanguage(self):
        return self._i18n.current

    # ── combo-box data (constant, so QML can build the selector) ──
    @Property(list, constant=True)
    def languageCodes(self):
        return list(self._i18n.LANGUAGE_CODES)

    @Property(list, constant=True)
    def displayNames(self):
        return [self._i18n.DISPLAY_NAMES[c] for c in self._i18n.LANGUAGE_CODES]


# ─────────────────────────────────────────────────────────────────────────────
#  StatusBridge
# ─────────────────────────────────────────────────────────────────────────────

class StatusBridge(QObject):
    """Holds parsed system-status state; QML binds to its properties."""

    updated = Signal()

    _LANGS = ["cmn", "eng", "jpn", "kor", "yue"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mfa_installed  = False
        self._mfa_version    = "—"
        self._models         = {c: False for c in self._LANGS}
        self._pyworld        = False
        self._crepe          = False
        self._rmvpe          = False
        self._whisperx       = False
        self._qwen3_asr      = False
        self._qwen3_aligner  = False
        self._cache_dir      = ""

    def apply(self, pipeline: dict, aligner: dict):
        st  = pipeline.get("status", {})
        mfa = st.get("mfa", {})

        self._mfa_installed = bool(mfa.get("installed"))
        self._mfa_version   = mfa.get("version") or "—"

        mdls = mfa.get("models", {})
        for c in self._LANGS:
            self._models[c] = bool(mdls.get(c, False))

        audio = st.get("audio_processing", {})
        self._pyworld = bool(audio.get("pyworld_available"))
        f0 = audio.get("f0_backends", {})
        self._crepe = bool((f0.get("crepe") or {}).get("available"))
        self._rmvpe = bool((f0.get("rmvpe") or {}).get("available"))

        alt = aligner.get("backends", {})
        self._whisperx      = bool((alt.get("whisperx")       or {}).get("available"))
        self._qwen3_asr     = bool((alt.get("qwen3_asr")      or {}).get("available"))
        self._qwen3_aligner = bool((alt.get("qwen3_aligner")  or {}).get("available"))
        self._cache_dir     = str(
            alt.get("models_dir") or
            st.get("alt_aligners", {}).get("models_dir", "")
        )
        self.updated.emit()

    # ── properties ─────────────────────────────────────────────────
    @Property(bool, notify=updated)
    def mfaInstalled(self): return self._mfa_installed

    @Property(str, notify=updated)
    def mfaVersion(self): return self._mfa_version

    @Property("QVariantMap", notify=updated)
    def models(self): return dict(self._models)

    @Property(bool, notify=updated)
    def pyworldAvailable(self): return self._pyworld

    @Property(bool, notify=updated)
    def crepeAvailable(self): return self._crepe

    @Property(bool, notify=updated)
    def rmvpeAvailable(self): return self._rmvpe

    @Property(bool, notify=updated)
    def whisperxAvailable(self): return self._whisperx

    @Property(bool, notify=updated)
    def qwen3AsrAvailable(self): return self._qwen3_asr

    @Property(bool, notify=updated)
    def qwen3AlignerAvailable(self): return self._qwen3_aligner

    @Property(str, notify=updated)
    def cacheDir(self): return self._cache_dir

    # ── convenience slots ──────────────────────────────────────────
    @Slot(str, result=bool)
    def modelAvailable(self, lang: str) -> bool:
        return self._models.get(lang, False)

    @Slot(str, result=bool)
    def backendAvailable(self, key: str) -> bool:
        return {
            "mfa":           self._mfa_installed,
            "whisperx":      self._whisperx,
            "qwen3_asr":     self._qwen3_asr,
            "qwen3_aligner": self._qwen3_aligner,
        }.get(key, False)


# ─────────────────────────────────────────────────────────────────────────────
#  ApiBridge
# ─────────────────────────────────────────────────────────────────────────────

class ApiBridge(QObject):
    """Wraps ApiClient; all network operations run on background threads."""

    # ── signals emitted to QML ─────────────────────────────────────
    baseUrlChanged     = Signal(str)
    statusFetched      = Signal()
    statusFailed       = Signal(str)

    modelDownloading   = Signal(str, bool)     # lang, is_downloading
    modelDownloaded    = Signal(str)
    modelFailed        = Signal(str, str)      # lang, error

    jobProgress        = Signal(int, str)      # percent, message
    jobCompleted       = Signal("QVariantMap")
    jobFailed          = Signal(str)

    fileDownloaded     = Signal(str, str)      # ctx_key, saved_path
    fileDownloadFailed = Signal(str, str)      # ctx_key, error

    def __init__(self, api: ApiClient, status: StatusBridge, parent=None):
        super().__init__(parent)
        self._api    = api
        self._status = status
        self._workers: dict = {}      # refs kept to prevent GC
        self._job_id = ""
        self._fake_progress = 10

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(1500)
        self._poll_timer.timeout.connect(self._tick_poll)

    # ── base URL ───────────────────────────────────────────────────
    @Property(str, notify=baseUrlChanged)
    def baseUrl(self): return self._api.base_url

    @Slot(str)
    def setBaseUrl(self, url: str):
        self._api.base_url = url.rstrip("/")
        self.baseUrlChanged.emit(self._api.base_url)

    # ── system status ──────────────────────────────────────────────
    @Slot()
    def checkStatus(self):
        w = self._api.status()
        w.succeeded.connect(self._on_status_ok)
        w.failed.connect(self.statusFailed)
        w.start()
        self._workers["status"] = w

    def _on_status_ok(self, data: dict):
        self._status.apply(data.get("pipeline", {}), data.get("aligner", {}))
        self.statusFetched.emit()

    # ── model download ─────────────────────────────────────────────
    @Slot(str)
    def downloadModel(self, lang: str):
        self.modelDownloading.emit(lang, True)
        w = self._api.download_model(lang)
        w.succeeded.connect(lambda _: self._on_model_ok(lang))
        w.failed.connect(lambda e: self._on_model_fail(lang, e))
        w.start()
        self._workers[f"dl_{lang}"] = w

    def _on_model_ok(self, lang: str):
        self.modelDownloading.emit(lang, False)
        self.modelDownloaded.emit(lang)
        QTimer.singleShot(800, self.checkStatus)

    def _on_model_fail(self, lang: str, err: str):
        self.modelDownloading.emit(lang, False)
        self.modelFailed.emit(lang, err)

    # ── processing job ─────────────────────────────────────────────
    @Slot(str, str, str, str, str, "QVariantMap")
    def startJob(self, mode: str, audio: str, lab: str,
                 midi: str, language: str, params: dict):
        p = dict(params)
        p["language"] = language
        w = self._api.process(
            mode,
            audio or None,
            lab   or None,
            midi  or None,
            p,
        )
        w.succeeded.connect(self._on_submit_ok)
        w.failed.connect(self.jobFailed)
        w.start()
        self._workers["proc"] = w
        self._fake_progress   = 10

    def _on_submit_ok(self, data: dict):
        job_id = data.get("job_id")
        if job_id:
            self._job_id = job_id
            self._poll_timer.start()
        else:
            self.jobCompleted.emit(data)

    def _tick_poll(self):
        if not self._job_id:
            self._poll_timer.stop()
            return
        # advance fake progress while waiting
        self._fake_progress = min(self._fake_progress + 3, 90)
        self.jobProgress.emit(self._fake_progress, "")

        w = self._api.poll_job(self._job_id)
        w.succeeded.connect(self._on_poll_ok)
        w.failed.connect(lambda e: logger.warning("poll blip: %s", e))
        w.start()
        self._workers["poll"] = w

    def _on_poll_ok(self, data: dict):
        job    = data.get("job", {})
        status = job.get("status", "")
        if status == "done":
            self._poll_timer.stop()
            self._job_id = ""
            self.jobProgress.emit(100, "")
            self.jobCompleted.emit(job.get("result", {}))
        elif status == "failed":
            self._poll_timer.stop()
            self._job_id = ""
            self.jobFailed.emit(job.get("error", "Processing failed"))

    # ── file download ──────────────────────────────────────────────
    @Slot(str, str, str)
    def downloadFile(self, filename: str, save_path: str, ctx_key: str):
        w = self._api.download_file(filename, save_path)
        w.succeeded.connect(lambda d: self.fileDownloaded.emit(ctx_key, d.get("path", "")))
        w.failed.connect(lambda e: self.fileDownloadFailed.emit(ctx_key, e))
        w.start()
        self._workers[f"file_{ctx_key}"] = w

    # ── utility slots called from QML ─────────────────────────────
    @Slot(str, result=str)
    def fileSizeStr(self, path: str) -> str:
        try:
            b = Path(path).stat().st_size
            for unit in ("Bytes", "KB", "MB", "GB"):
                if b < 1024:
                    return f"{b:.1f} {unit}"
                b /= 1024
            return f"{b:.1f} TB"
        except Exception:
            return ""

    @Slot(str, result=float)
    def parseMidiBpm(self, path: str) -> float:
        """Extract tempo from MIDI header without external libs."""
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

    @Slot(int, result=str)
    def midiToNoteName(self, midi: int) -> str:
        names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
        return f"{names[midi % 12]}{midi // 12 - 1}"

    @Slot(str, str, result=bool)
    def writeTextFile(self, path: str, content: str) -> bool:
        try:
            Path(path).write_text(content, encoding="utf-8")
            return True
        except Exception as e:
            logger.error("writeTextFile: %s", e)
            return False
