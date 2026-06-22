# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sys
import time
import subprocess
import logging
import threading
from pathlib import Path
from typing import Dict, Tuple, Optional

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
    def resolve_mfa_exe() -> Optional[Path]:
        # 只认项目环境里的 mfa.exe，避免误命中 base 环境
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
    def _run_mfa_version_once(py: Path, timeout: float) -> Tuple[bool, str]:
        cmd = [
            str(py),
            "-m",
            "montreal_forced_aligner.command_line.mfa",
            "version",
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return True, result.stdout.strip() or "OK"
        msg = result.stderr.strip() or result.stdout.strip() or "mfa version failed"
        return False, msg

    @staticmethod
    def check_mfa_installed(use_cache: bool = True) -> Tuple[bool, str]:
        """
        用项目环境自己的 python 去调用 MFA，彻底避开 PATH / base conda。

        【稳健性修复】单次 subprocess 偶尔会因为系统正在跑对齐任务、磁盘/CPU
        被占满，导致冷启动 import montreal_forced_aligner + kalpy 耗时暴涨而
        超时，这不代表 MFA 真的没装。这里做三件事：
          1. 超时从 30s 放宽到 45s；
          2. 失败/超时后立刻重试一次（间隔 1s）；
          3. 两次都失败时，如果 _MFA_CHECK_CACHE_TTL 秒内有过一次成功记录，
             直接复用该结果，而不是把"繁忙导致的超时"误判成"未安装"。
        """
        py = MFAChecker.env_python()
        last_err = ""

        for attempt in range(2):
            try:
                ok, msg = MFAChecker._run_mfa_version_once(py, timeout=45)
                if ok:
                    with MFAChecker._status_cache_lock:
                        MFAChecker._last_good_mfa_check = (True, msg, time.time())
                    return True, msg
                last_err = msg
            except subprocess.TimeoutExpired as e:
                last_err = f"mfa version 检测超时（第 {attempt + 1} 次，{e.timeout}s）：系统可能正繁忙"
                logger.warning(last_err)
            except Exception as e:
                last_err = str(e)

            if attempt == 0:
                time.sleep(1.0)

        if use_cache:
            with MFAChecker._status_cache_lock:
                cached = MFAChecker._last_good_mfa_check
            if cached:
                ok, msg, ts = cached
                age = time.time() - ts
                if age < MFAChecker._MFA_CHECK_CACHE_TTL:
                    logger.warning(
                        f"check_mfa_installed 两次实时检测均失败（{last_err}），"
                        f"复用 {age:.0f}s 前的成功结果，避免误报'未安装'"
                    )
                    return True, msg

        logger.warning(f"check_mfa_installed 失败: {last_err}")
        return False, last_err

    @staticmethod
    def check_model_downloaded(model_name: str, model_type: str = "acoustic") -> bool:
        """检查指定模型是否已下载"""
        # 直接检查 MFA 的模型缓存目录
        mfa_cache_home = Path.home() / "Documents" / "MFA" / "models"
        
        # 新结构：models/inspect/{model_name}/
        inspect_path = mfa_cache_home / "inspect" / model_name
        if inspect_path.exists():
            logger.info(f"✓ 找到模型 {model_name} 在 inspect 路径: {inspect_path}")
            return True
        
        # 旧结构：pretrained_models/{type}/{model_name}.zip
        old_path = mfa_cache_home.parent / "pretrained_models" / model_type / f"{model_name}.zip"
        if old_path.exists():
            logger.info(f"✓ 找到模型 {model_name} 在 pretrained_models 路径: {old_path}")
            return True
        
        # 字典文件特殊处理：检查 .dict 文件
        dict_path = None
        if model_type == "dictionary":
            dict_path = mfa_cache_home.parent / "pretrained_models" / model_type / f"{model_name}.dict"
            if dict_path.exists():
                logger.info(f"✓ 找到模型 {model_name} 在 pretrained_models 路径: {dict_path}")
                return True
        
        logger.warning(f"✗ 未找到模型 {model_name} (类型: {model_type})")
        logger.warning(f"  检查位置 1: {inspect_path}")
        logger.warning(f"  检查位置 2: {old_path}")
        if dict_path:
            logger.warning(f"  检查位置 3: {dict_path}")
        
        # 备用方案：尝试用 mfa model inspect 再检查一次
        py = MFAChecker.env_python()
        cmd = [
            str(py),
            "-m",
            "montreal_forced_aligner.command_line.mfa",
            "model",
            "inspect",
            model_type,
            model_name,
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=45,
            )
            if result.returncode == 0:
                logger.info(f"✓ mfa model inspect 验证了 {model_name}")
                return True
            logger.warning(f"✗ mfa model inspect 验证失败 {model_name}: {result.stderr[:100]}")
            return False
        except Exception as e:
            logger.error(f"✗ mfa model inspect 异常: {e}")
            return False

    @staticmethod
    def get_status() -> Dict[str, object]:
        kalpy_ok, kalpy_msg = MFAChecker.check_kalpy()
        mfa_ok, mfa_msg = MFAChecker.check_mfa_installed()

        # 【修复】语言模型下载状态主要靠文件系统路径检测（check_model_downloaded
        # 内部优先走 Path.exists()，快且可靠），不应该依赖本次 "mfa version"
        # 子进程是否刚好成功。之前用 `if mfa_ok:` 把整个循环包起来，导致一次
        # 瞬时超时就会把全部 5 个语言的 ✓ 一起拖成 ✗（看起来像"系统全崩了"），
        # 实际上模型文件一直都在磁盘上，跟这次 version 检测有没有超时无关。
        # 这里始终执行检查，与 mfa_ok 解耦。
        models_status = {}
        # 只检查前端使用的主要语言代码（避免重复检查）
        primary_langs = ["cmn", "eng", "jpn", "kor", "yue"]

        for lang_code in primary_langs:
            models = MFAChecker.LANGUAGE_MODELS.get(lang_code)
            if not models:
                continue

            # 检查 Dictionary 和 Acoustic 都已下载
            dict_model = models["dictionary"]
            acoustic_model = models["acoustic"]

            logger.info(f"检查 {lang_code}: dictionary={dict_model}, acoustic={acoustic_model}")
            dict_ok = MFAChecker.check_model_downloaded(dict_model, "dictionary")
            acoustic_ok = MFAChecker.check_model_downloaded(acoustic_model, "acoustic")

            models_status[lang_code] = dict_ok and acoustic_ok
            logger.info(f"  {lang_code}: dict={dict_ok}, acoustic={acoustic_ok}, combined={models_status[lang_code]}")

        return {
            "installed": bool(kalpy_ok and mfa_ok),
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
