# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time
from importlib.metadata import PackageNotFoundError, version as pkg_version
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

class MFAChecker:
    # 【稳健性修复】缓存最近一次成功的 "mfa version" 检测结果。
    # 同一台机器上跑对齐任务时（不管是 MFA 还是 Qwen3），CPU/磁盘 IO 被占满，
    # 会导致冷启动 import montreal_forced_aligner + kalpy 的子进程偶尔超过
    # 超时时间，从而把"系统繁忙"误判成"MFA 未安装"——这正是
    # "有时检测不到 MFA，刷新页面才恢复正常" 的根因（刷新只是又重试了一次，
    # 刚好赶上系统不那么忙）。
    _status_cache_lock = threading.Lock()
    _last_good_mfa_check: Optional[Tuple[bool, str, float]] = None  # (ok, msg, timestamp)
    _MFA_CHECK_CACHE_TTL = 120.0  # 秒：在这个窗口内允许复用上一次的成功结果

    # MFA 3.3.9 模型映射：语言代码 -> {"dictionary": ..., "acoustic": ...}
    LANGUAGE_MODELS: Dict[str, Dict[str, str]] = {
        "cmn": {
            "dictionary": "mandarin_china_mfa",
            "acoustic": "mandarin_mfa",
        },
        "zh": {
            "dictionary": "mandarin_china_mfa",
            "acoustic": "mandarin_mfa",
        },
        "eng": {
            "dictionary": "english_us_mfa",
            "acoustic": "english_mfa",
        },
        "en": {
            "dictionary": "english_us_mfa",
            "acoustic": "english_mfa",
        },
        "jpn": {
            "dictionary": "japanese_mfa",
            "acoustic": "japanese_mfa",
        },
        "ja": {
            "dictionary": "japanese_mfa",
            "acoustic": "japanese_mfa",
        },
        "kor": {
            "dictionary": "korean_mfa",
            "acoustic": "korean_mfa",
        },
        "ko": {
            "dictionary": "korean_mfa",
            "acoustic": "korean_mfa",
        },
        "yue": {
            "dictionary": "mandarin_china_mfa",
            "acoustic": "mandarin_mfa",
        }
    }

    # ===== 输出格式映射：语言 → 转换目标 =====
    PHONEME_OUTPUT_FORMAT: Dict[str, str] = {
        'en': 'arpabet',     # English → ARPABET
        'eng': 'arpabet',
        'ja': 'romaji',      # Japanese → ROMAJI
        'jpn': 'romaji',
        'zh': 'pinyin',      # Chinese → Pinyin (no conversion needed)
        'cmn': 'pinyin',
        'yue': 'jyutping',   # Cantonese → Jyutping
        'ko': 'hangul',      # Korean → Hangul Jamo (no conversion needed)
        'kor': 'hangul',
    }

    @staticmethod
    def project_root() -> Path:
        return Path(__file__).resolve().parent.parent

    @staticmethod
    def env_dir() -> Path:
        env_dir = os.environ.get("MFA_ENV_DIR")
        if env_dir and Path(env_dir).exists():
            return Path(env_dir)

        local_prefix = MFAChecker.project_root() / ".mfa_env"
        if local_prefix.exists():
            return local_prefix

        return Path(sys.prefix)

    @staticmethod
    def env_python() -> Path:
        p = MFAChecker.env_dir() / "python.exe"
        if p.exists():
            return p
        return Path(sys.executable)

    @staticmethod
    def mfa_root_dir() -> Path:
        """
        统一 MFA 根目录，优先读取 MFA_ROOT_DIR。
        这样可以兼容你把模型放到 E 盘的情况。
        """
        root = os.environ.get("MFA_ROOT_DIR")
        if root and Path(root).exists():
            return Path(root)
        return Path.home() / "Documents" / "MFA"

    @staticmethod
    def resolve_mfa_exe() -> Optional[Path]:
        candidates = [
            os.environ.get("MFA_EXE"),
            str(MFAChecker.env_dir() / "Scripts" / "mfa.exe"),
            str(MFAChecker.env_dir() / "Scripts" / "mfa"),
        ]
        for c in candidates:
            if c and Path(c).exists():
                return Path(c)
        return None

    @staticmethod
    def check_kalpy() -> Tuple[bool, str]:
        try:
            import _kalpy  # noqa: F401
            return True, "OK"
        except Exception as e:
            return False, str(e)

    @staticmethod
    def _cache_mfa_result(ok: bool, msg: str) -> None:
        """将检测结果写入 TTL 缓存（仅缓存成功结果）。"""
        if ok:
            with MFAChecker._status_cache_lock:
                MFAChecker._last_good_mfa_check = (ok, msg, time.monotonic())

    @staticmethod
    def check_mfa_installed() -> Tuple[bool, str]:
        """
        检查 MFA 是否可用，并尽量返回真实版本号。

        检测顺序（速度从快到慢）：
          1. TTL 缓存 — 120 s 内复用上一次的成功结果（0 ms）
          2. 同进程 importlib.metadata — MFA 在同一 venv 时立即返回（~1 ms）
          3. 子进程元数据查询 — MFA 在独立 venv 时调用（~5–30 s，有超时保护）
          4. 子进程 CLI 版本查询 — 最后的兜底手段

        任何一次探测成功，都缓存结果并返回实际版本字符串。
        """
        # ── 1. TTL 缓存：120 s 内直接复用成功结果 ────────────────────────────
        with MFAChecker._status_cache_lock:
            cached = MFAChecker._last_good_mfa_check
        if cached is not None:
            ok, msg, ts = cached
            if ok and (time.monotonic() - ts) < MFAChecker._MFA_CHECK_CACHE_TTL:
                logger.debug("check_mfa_installed: 命中 TTL 缓存 (%s)", msg)
                return ok, msg

        # ── 2. 同进程元数据查询（最快，无子进程开销）────────────────────────
        # Flask 与 MFA 运行在同一 venv 时，这里直接返回，整个函数开销 < 1 ms。
        # pkg_version 已在文件顶部 import，此处是第一次实际调用它。
        try:
            v = pkg_version("montreal-forced-aligner")
            if v:
                logger.info("check_mfa_installed: 同进程检测成功，版本 %s", v)
                MFAChecker._cache_mfa_result(True, v)
                return True, v
        except PackageNotFoundError:
            # MFA 不在当前 Python 环境，跌落到子进程探测
            logger.debug("check_mfa_installed: 当前进程未安装 MFA，尝试独立 venv")
        except Exception as e:
            logger.debug("check_mfa_installed: 同进程 pkg_version 异常: %s", e)

        # ── 3 & 4. 子进程探测（MFA 在独立 venv 时才走到这里）────────────────
        py = MFAChecker.env_python()

        def _normalize_version_text(text: str) -> str:
            text = (text or "").strip()
            if not text:
                return ""
            # 取最后一行，避免前面带欢迎信息/警告
            return text.splitlines()[-1].strip()

        probes = [
            # 3) 子进程内读包元数据，比 CLI 启动更轻量
            [
                str(py),
                "-c",
                (
                    "from importlib.metadata import version; "
                    "print(version('montreal-forced-aligner'))"
                ),
            ],
            # 4) CLI 版本命令（最重，作为最终兜底）
            [str(py), "-m", "montreal_forced_aligner.command_line.mfa", "version"],
        ]

        last_msg = ""

        for cmd in probes:
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )

                stdout = (result.stdout or "").strip()
                stderr = (result.stderr or "").strip()

                if result.returncode == 0:
                    version_text = _normalize_version_text(stdout)
                    if version_text:
                        MFAChecker._cache_mfa_result(True, version_text)
                        return True, version_text

                    if stderr:
                        # 有些环境版本号会被打到 stderr，顺手兼容一下
                        version_text = _normalize_version_text(stderr)
                        if version_text:
                            MFAChecker._cache_mfa_result(True, version_text)
                            return True, version_text

                    MFAChecker._cache_mfa_result(True, "unknown")
                    return True, "unknown"

                last_msg = stderr or stdout or f"returncode={result.returncode}"

            except subprocess.TimeoutExpired:
                last_msg = "版本检测超时"
                continue
            except Exception as e:
                last_msg = str(e)

        return False, last_msg or "MFA not detected"

    @staticmethod
    def check_model_downloaded(model_name: str, model_type: str = "acoustic") -> bool:
        """
        改成统一从 MFA_ROOT_DIR / Documents/MFA 找模型。
        """
        mfa_cache_home = MFAChecker.mfa_root_dir()

        inspect_path = mfa_cache_home / "inspect" / model_name
        if inspect_path.exists():
            logger.info(f"✓ 找到模型 {model_name} 在 inspect 路径: {inspect_path}")
            return True

        old_path = mfa_cache_home / "pretrained_models" / model_type / f"{model_name}.zip"
        if old_path.exists():
            logger.info(f"✓ 找到模型 {model_name} 在 pretrained_models 路径: {old_path}")
            return True

        dict_path = None
        if model_type == "dictionary":
            dict_path = mfa_cache_home / "pretrained_models" / model_type / f"{model_name}.dict"
            if dict_path.exists():
                logger.info(f"✓ 找到模型 {model_name} 在 pretrained_models 路径: {dict_path}")
                return True

        logger.warning(f"✗ 未找到模型 {model_name} (类型: {model_type})")
        logger.warning(f"  检查位置 1: {inspect_path}")
        logger.warning(f"  检查位置 2: {old_path}")
        if dict_path:
            logger.warning(f"  检查位置 3: {dict_path}")

        return False

    @staticmethod
    def get_status() -> Dict[str, object]:
        kalpy_ok, kalpy_msg = MFAChecker.check_kalpy()
        mfa_ok, mfa_msg = MFAChecker.check_mfa_installed()

        models_status = {}
        if mfa_ok:
            primary_langs = ["cmn", "eng", "jpn", "kor", "yue"]
            for lang_code in primary_langs:
                models = MFAChecker.LANGUAGE_MODELS.get(lang_code)
                if not models:
                    continue

                dict_model = models["dictionary"]
                acoustic_model = models["acoustic"]

                logger.info(f"检查 {lang_code}: dictionary={dict_model}, acoustic={acoustic_model}")
                dict_ok = MFAChecker.check_model_downloaded(dict_model, "dictionary")
                acoustic_ok = MFAChecker.check_model_downloaded(acoustic_model, "acoustic")
                models_status[lang_code] = dict_ok and acoustic_ok

                logger.info(f"  {lang_code}: dict={dict_ok}, acoustic={acoustic_ok}, combined={models_status[lang_code]}")

        return {
            # 这里建议把"安装"和"可运行"分开
            "installed": bool(mfa_ok),
            "ready": bool(kalpy_ok and mfa_ok),
            "version": mfa_msg if mfa_ok else "",
            "kalpy": kalpy_ok,
            "kalpy_message": kalpy_msg,
            "mfa": mfa_ok,
            "mfa_message": mfa_msg,
            "mfa_version": mfa_msg if mfa_ok else "",
            "models": {
                "cmn": models_status.get("cmn", False),
                "eng": models_status.get("eng", False),
                "jpn": models_status.get("jpn", False),
                "kor": models_status.get("kor", False),
                "yue": models_status.get("yue", False),
            }
        }

    @staticmethod
    def download_model(language: str) -> Tuple[bool, str]:
        models = MFAChecker.LANGUAGE_MODELS.get(language)
        if not models:
            return False, f"Unknown language: {language}"
        
        dict_model = models["dictionary"]
        acoustic_model = models["acoustic"]
        py = MFAChecker.env_python()
        
        results = []
        
        # 下载 Dictionary
        cmd_dict = [
            str(py),
            "-m",
            "montreal_forced_aligner.command_line.mfa",
            "model",
            "download",
            "dictionary",
            dict_model,
        ]
        try:
            result = subprocess.run(cmd_dict, capture_output=True, text=True, timeout=600)
            if result.returncode == 0:
                results.append(f"Dictionary {dict_model} downloaded")
                # 模型下载成功后，让缓存自然过期以触发重新检测
                with MFAChecker._status_cache_lock:
                    MFAChecker._last_good_mfa_check = None
            else:
                return False, f"Dictionary download failed: {result.stderr}"
        except Exception as e:
            return False, f"Dictionary download error: {str(e)}"
        
        # 下载 Acoustic Model
        cmd_acoustic = [
            str(py),
            "-m",
            "montreal_forced_aligner.command_line.mfa",
            "model",
            "download",
            "acoustic",
            acoustic_model,
        ]
        try:
            result = subprocess.run(cmd_acoustic, capture_output=True, text=True, timeout=600)
            if result.returncode == 0:
                results.append(f"Acoustic {acoustic_model} downloaded")
            else:
                return False, f"Acoustic download failed: {result.stderr}"
        except Exception as e:
            return False, f"Acoustic download error: {str(e)}"
        
        return True, " + ".join(results)
