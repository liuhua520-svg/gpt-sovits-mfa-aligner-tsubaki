# api_client.py — non-blocking HTTP client for the Flask backend
from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, QThread, Signal

logger = logging.getLogger(__name__)

# ── worker base ───────────────────────────────────────────────────────────────

class _Worker(QThread):
    """Base worker that runs a task on a background thread."""
    succeeded = Signal(dict)
    failed    = Signal(str)

    def __init__(self, base_url: str, parent=None):
        super().__init__(parent)
        self._base = base_url.rstrip("/")

    def _get(self, path: str, timeout: int = 15) -> dict:
        import requests
        r = requests.get(f"{self._base}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json()

    def _post_json(self, path: str, payload: dict, timeout: int = 30) -> dict:
        import requests
        r = requests.post(f"{self._base}{path}", json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()

    def _post_multipart(self, path: str,
                        files: dict, data: dict,
                        timeout: int = 300) -> dict:
        import requests
        r = requests.post(f"{self._base}{path}",
                          files=files, data=data, timeout=timeout)
        r.raise_for_status()
        return r.json()


# ── concrete workers ──────────────────────────────────────────────────────────

class StatusWorker(_Worker):
    """GET /api/pipeline/status  +  GET /api/aligner/status

    【修复】check_mfa_installed() 在冷启动时会调用子进程，耗时可达 30+ 秒。
    将这两条请求的超时从默认 15 s 提高到 120 s，确保首次连接不会因超时
    被误判为"无法连接到后端"。后续调用命中 TTL 缓存后，响应会在 < 1 s 内返回。
    """

    # 状态接口超时：首次冷启动需要等待 MFA 子进程，给够 120 s
    _STATUS_TIMEOUT = 120

    def run(self):
        try:
            pipeline = self._get("/api/pipeline/status", timeout=self._STATUS_TIMEOUT)
            aligner  = self._get("/api/aligner/status",  timeout=self._STATUS_TIMEOUT)
            self.succeeded.emit({"pipeline": pipeline, "aligner": aligner})
        except Exception as exc:
            self.failed.emit(str(exc))


class DownloadModelWorker(_Worker):
    """POST /api/mfa/download-model/<lang>"""

    def __init__(self, base_url: str, lang: str, parent=None):
        super().__init__(base_url, parent)
        self._lang = lang

    def run(self):
        try:
            data = self._post_json(f"/api/mfa/download-model/{self._lang}", {})
            self.succeeded.emit(data)
        except Exception as exc:
            self.failed.emit(str(exc))


class JobPollWorker(_Worker):
    """GET /api/pipeline/job/<job_id> — single poll tick."""

    def __init__(self, base_url: str, job_id: str, parent=None):
        super().__init__(base_url, parent)
        self._job_id = job_id

    def run(self):
        try:
            data = self._get(f"/api/pipeline/job/{self._job_id}")
            self.succeeded.emit(data)
        except Exception as exc:
            self.failed.emit(str(exc))


class ProcessWorker(_Worker):
    """
    Submit a processing job (mfa-only, full, or project-only) and
    immediately return the job_id.  Polling is done separately by
    the UI layer using JobPollWorker.
    """

    def __init__(self, base_url: str, mode: str,
                 audio_path: Optional[str],
                 lab_path: Optional[str],
                 midi_path: Optional[str],
                 params: dict,
                 parent=None):
        super().__init__(base_url, parent)
        self._mode       = mode
        self._audio_path = audio_path
        self._lab_path   = lab_path
        self._midi_path  = midi_path
        self._params     = params

    # helper: open a file for multipart upload
    @staticmethod
    def _fopen(path: str):
        return open(path, "rb")

    def run(self):
        try:
            if self._mode == "project-only":
                self._submit_project_only()
            elif self._mode == "full":
                self._submit_full()
            else:
                self._submit_mfa_only()
        except Exception as exc:
            logger.exception("ProcessWorker error")
            self.failed.emit(str(exc))

    def _submit_mfa_only(self):
        files = {
            "audio_file": (Path(self._audio_path).name,
                           self._fopen(self._audio_path)),
        }
        data = {
            "text":             self._params.get("text", ""),
            "language":         self._params.get("language", "cmn"),
            "aligner_backend":  self._params.get("aligner_backend", "mfa"),
            "aligner_device":   self._params.get("aligner_device", "auto"),
            "f0_device":        self._params.get("aligner_device", "auto"),
        }
        result = self._post_multipart("/api/pipeline/mfa-only", files, data)
        self.succeeded.emit(result)

    def _submit_full(self):
        files = {"audio_file": (Path(self._audio_path).name,
                                self._fopen(self._audio_path))}
        p = self._params
        data = {
            "text":              p.get("text", ""),
            "language":          p.get("language", "cmn"),
            "aligner_backend":   p.get("aligner_backend", "mfa"),
            "aligner_device":    p.get("aligner_device", "auto"),
            "f0_device":         p.get("f0_device", "auto"),
            "format":            p.get("output_format", "sv"),
            "title":             p.get("project_title", "Project"),
            "bpm":               str(p.get("bpm", 120)),
            "base_pitch":        str(p.get("base_pitch", 60)),
            "f0_method":         p.get("f0_method", "dio"),
            "crepe_model":       p.get("crepe_model", "full"),
            "f0_smooth":         str(p.get("f0_smooth", True)).lower(),
            "f0_smooth_window":  str(p.get("f0_smooth_window", 5)),
            "precision":         p.get("precision", "single"),
            "f0_floor":          str(p.get("f0_floor", 71.0)),
            "f0_ceil":           str(p.get("f0_ceil", 800.0)),
            "auto_note_pitch":   str(p.get("auto_note_pitch", True)).lower(),
            "export_pitch_line": str(p.get("export_pitch_line", True)).lower(),
        }
        result = self._post_multipart("/api/pipeline/full", files, data)
        self.succeeded.emit(result)

    def _submit_project_only(self):
        files: dict = {
            "wav_file": (Path(self._audio_path).name,
                         self._fopen(self._audio_path))
        }
        p = self._params

        if self._lab_path:
            files["lab_file"] = (Path(self._lab_path).name,
                                 self._fopen(self._lab_path))
        elif self._midi_path:
            files["midi_file"] = (Path(self._midi_path).name,
                                  self._fopen(self._midi_path))

        data = {
            "format":            p.get("output_format", "sv"),
            "title":             p.get("project_title", "Project"),
            "bpm":               str(p.get("bpm", 120)),
            "base_pitch":        str(p.get("base_pitch", 60)),
            "f0_method":         p.get("f0_method", "dio"),
            "crepe_model":       p.get("crepe_model", "full"),
            "f0_smooth":         str(p.get("f0_smooth", True)).lower(),
            "f0_smooth_window":  str(p.get("f0_smooth_window", 5)),
            "precision":         p.get("precision", "single"),
            "f0_floor":          str(p.get("f0_floor", 71.0)),
            "f0_ceil":           str(p.get("f0_ceil", 800.0)),
            "auto_note_pitch":   str(p.get("auto_note_pitch", True)).lower(),
            "export_pitch_line": str(p.get("export_pitch_line", True)).lower(),
            "phoneme_mode":      p.get("phoneme_mode", "none"),
            "f0_device":         p.get("f0_device", "auto"),
        }
        result = self._post_multipart("/api/pipeline/project-only", files, data)
        self.succeeded.emit(result)


class FileDownloadWorker(_Worker):
    """
    GET /api/work-dir/download/<filename>  — saves the file to *save_path*.
    Emits succeeded({'path': save_path}) on success.
    """

    def __init__(self, base_url: str, filename: str,
                 save_path: str, parent=None):
        super().__init__(base_url, parent)
        self._filename  = filename
        self._save_path = save_path

    def run(self):
        try:
            import requests
            url = f"{self._base}/api/work-dir/download/{self._filename}"
            r = requests.get(url, timeout=120, stream=True)
            r.raise_for_status()
            with open(self._save_path, "wb") as fh:
                for chunk in r.iter_content(chunk_size=65536):
                    fh.write(chunk)
            self.succeeded.emit({"path": self._save_path})
        except Exception as exc:
            self.failed.emit(str(exc))


# ── convenience façade ────────────────────────────────────────────────────────

class ApiClient(QObject):
    """
    Thin façade that creates worker threads on demand.
    The caller owns each worker and must keep a reference alive
    until it emits a terminal signal.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:5000", parent=None):
        super().__init__(parent)
        self.base_url = base_url

    # ── factory helpers ───────────────────────────────────────────
    def status(self) -> StatusWorker:
        return StatusWorker(self.base_url, self)

    def download_model(self, lang: str) -> DownloadModelWorker:
        return DownloadModelWorker(self.base_url, lang, self)

    def poll_job(self, job_id: str) -> JobPollWorker:
        return JobPollWorker(self.base_url, job_id, self)

    def process(self, mode: str,
                audio_path: Optional[str],
                lab_path:   Optional[str],
                midi_path:  Optional[str],
                params: dict) -> ProcessWorker:
        return ProcessWorker(self.base_url, mode,
                             audio_path, lab_path, midi_path,
                             params, self)

    def download_file(self, filename: str, save_path: str) -> FileDownloadWorker:
        return FileDownloadWorker(self.base_url, filename, save_path, self)
