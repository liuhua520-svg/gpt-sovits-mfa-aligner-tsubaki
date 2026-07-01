# -*- coding: utf-8 -*-
"""
应用级设置管理：模型自动更新开关（HF_HUB_OFFLINE）与镜像站下载配置（HF_ENDPOINT）。

设计说明
────────
这两个环境变量分别由三个各自独立的进程消费：
  - app.py（经由其在模块顶层 import 的 alt_aligners.py）
  - qwen3_server.py（独立 venv 子服务，端口 5001）
  - nemo_server.py（独立 venv 子服务，端口 5002）

三者都会在各自最早的时机（import huggingface_hub / transformers /
qwen_asr / nemo 之前）调用本模块的 apply_env_from_settings()，从同一份
JSON 配置文件读取设置并写入 os.environ。这样设置页面只需要写一次文件，
三个进程各自重启后即可生效——环境变量只在"进程启动时"读取一次，
所以修改设置后必须重启对应进程（尤其是 Qwen3 / NeMo 两个微服务）才能
让新配置真正生效，这一点在设置页面的提示文案里需要向用户说清楚。
"""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent
SETTINGS_PATH = PROJECT_DIR / "backend" / "app_settings.json"

_lock = threading.RLock()

DEFAULT_SETTINGS: Dict[str, object] = {
    # False → HF_HUB_OFFLINE=1（默认禁用自动联网更新，行为与改造前一致，最省心）
    # True  → HF_HUB_OFFLINE=0（允许 huggingface_hub 联网检查/下载模型）
    "auto_update_models": False,
    "use_mirror": False,
    "mirror_url": "https://hf-mirror.com/",
}


def load_settings() -> Dict[str, object]:
    with _lock:
        if SETTINGS_PATH.exists():
            try:
                data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
                merged = dict(DEFAULT_SETTINGS)
                if isinstance(data, dict):
                    merged.update({k: v for k, v in data.items() if k in DEFAULT_SETTINGS})
                return merged
            except Exception as e:
                logger.error("读取设置文件失败（%s），使用默认设置: %s", SETTINGS_PATH, e)
                return dict(DEFAULT_SETTINGS)
        return dict(DEFAULT_SETTINGS)


def save_settings(new_settings: Dict[str, object]) -> Dict[str, object]:
    with _lock:
        current = load_settings()
        current.update({k: v for k, v in (new_settings or {}).items() if k in DEFAULT_SETTINGS})

        # 基本校验，避免脏数据写入
        current["auto_update_models"] = bool(current.get("auto_update_models"))
        current["use_mirror"] = bool(current.get("use_mirror"))
        mirror_url = str(current.get("mirror_url") or "").strip()
        current["mirror_url"] = mirror_url or DEFAULT_SETTINGS["mirror_url"]

        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = SETTINGS_PATH.with_suffix(".json.tmp")
        tmp_path.write_text(
            json.dumps(current, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp_path.replace(SETTINGS_PATH)
        return current


def apply_env_from_settings() -> Dict[str, object]:
    """
    在进程启动早期调用：把设置文件内容映射为环境变量。

    必须在 import huggingface_hub / transformers / qwen_asr / nemo_toolkit
    之前调用，这些库只在 import 时读取一次 HF_HUB_OFFLINE / HF_ENDPOINT，
    之后修改 os.environ 不会再生效（需要重启进程）。
    """
    settings = load_settings()

    os.environ["HF_HUB_OFFLINE"] = "0" if settings.get("auto_update_models") else "1"

    if settings.get("use_mirror") and settings.get("mirror_url"):
        os.environ["HF_ENDPOINT"] = str(settings["mirror_url"]).rstrip("/") + "/"
    else:
        os.environ.pop("HF_ENDPOINT", None)

    logger.info(
        "已应用模型下载设置: HF_HUB_OFFLINE=%s, HF_ENDPOINT=%s",
        os.environ.get("HF_HUB_OFFLINE"),
        os.environ.get("HF_ENDPOINT", "(未设置，使用官方 huggingface.co)"),
    )
    return settings
