# -*- coding: utf-8 -*-
"""
用户自定义"单词 → 音素"词典管理模块。

支持两套彼此独立的词典（不支持 OpenUTAU）：
  - "synthesizerv"：ARPABET 记号（与 word_to_arpabet() / SVP phonemes 字段一致，
                    小写、无重音数字，例如 "hh ah l ow"）
  - "vocaloid"    ：VOCALOID4 音素记号（与 arpabet_to_vocaloid4() 的输出一致，
                    例如 "h @ l ou"）

词典条目在这两套记号之间不自动互转——用户在词典页面选择要维护的来源后，
录入的音素字符串会被原样存储、原样使用。若用户只维护了一侧词典，
另一侧在命中前会自动回退到软件默认的转换流程（MFA 词典 / g2p_en /
arpabet_to_vocaloid4），不会报错。

数据以单个 JSON 文件持久化，进程内用一把锁保护，并做了一层内存缓存，
避免每次查词都读盘。
"""
from __future__ import annotations

import csv
import io
import json
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent
DICT_STORE_PATH = PROJECT_DIR / "backend" / "user_dictionary.json"

# 支持的词典来源。"default" 不是一个真实存储的来源，
# 而是"不使用用户词典，走软件默认转换流程"的哨兵值，调用方直接传入即可。
VALID_SOURCES: Tuple[str, ...] = ("synthesizerv", "vocaloid")
SOURCE_DEFAULT = "default"

_lock = threading.RLock()
_cache: Optional[Dict[str, Dict[str, str]]] = None


def _empty_store() -> Dict[str, Dict[str, str]]:
    return {src: {} for src in VALID_SOURCES}


def _load() -> Dict[str, Dict[str, str]]:
    """加载词典（带内存缓存）。持有 _lock 时调用。"""
    global _cache
    if _cache is not None:
        return _cache

    if DICT_STORE_PATH.exists():
        try:
            raw = json.loads(DICT_STORE_PATH.read_text(encoding="utf-8"))
            store = _empty_store()
            if isinstance(raw, dict):
                for src in VALID_SOURCES:
                    entries = raw.get(src, {})
                    if isinstance(entries, dict):
                        store[src] = {
                            str(k).strip().upper(): str(v).strip()
                            for k, v in entries.items()
                            if str(k).strip() and str(v).strip()
                        }
            _cache = store
        except Exception as e:
            logger.error("加载用户词典失败（%s），本次以空词典启动: %s", DICT_STORE_PATH, e)
            _cache = _empty_store()
    else:
        _cache = _empty_store()

    return _cache


def _save(store: Dict[str, Dict[str, str]]) -> None:
    """持有 _lock 时调用。写盘后刷新缓存。"""
    global _cache
    DICT_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = DICT_STORE_PATH.with_suffix(".json.tmp")
    tmp_path.write_text(
        json.dumps(store, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tmp_path.replace(DICT_STORE_PATH)  # 原子替换，避免写到一半崩溃损坏文件
    _cache = store


def _normalize_source(source: str) -> str:
    src = (source or "").strip().lower()
    if src not in VALID_SOURCES:
        raise ValueError(
            f"不支持的词典来源: {source!r}（仅支持 {', '.join(VALID_SOURCES)}，"
            "不支持 OpenUTAU）"
        )
    return src


def lookup_word(word: str, source: str) -> Optional[str]:
    """
    在指定来源词典中查找单词的音素映射（原始字符串，未 split）。

    source 若不是合法的词典来源（例如调用方直接传入 "default"，
    表示用户在高级设置里选择了"使用软件默认值"），直接返回 None，
    调用方据此回退到软件默认转换流程——这里不对 "default" 抛异常，
    因为它是一个合法的"不查词典"输入，而非错误。
    """
    if not word:
        return None
    src = (source or "").strip().lower()
    if src not in VALID_SOURCES:
        return None
    with _lock:
        store = _load()
        return store.get(src, {}).get(word.strip().upper())


def list_entries(source: str) -> Dict[str, str]:
    src = _normalize_source(source)
    with _lock:
        store = _load()
        return dict(store.get(src, {}))


def upsert_entry(source: str, word: str, phonemes: str) -> None:
    src = _normalize_source(source)
    word = (word or "").strip().upper()
    phonemes = (phonemes or "").strip()
    if not word:
        raise ValueError("单词不能为空")
    if not phonemes:
        raise ValueError("音素不能为空")

    with _lock:
        store = {k: dict(v) for k, v in _load().items()}
        store.setdefault(src, {})[word] = phonemes
        _save(store)


def delete_entry(source: str, word: str) -> bool:
    src = _normalize_source(source)
    word = (word or "").strip().upper()

    with _lock:
        store = {k: dict(v) for k, v in _load().items()}
        existed = word in store.get(src, {})
        if existed:
            del store[src][word]
            _save(store)
        return existed


def bulk_import(source: str, entries: Dict[str, str], overwrite: bool = True) -> Tuple[int, int]:
    """
    批量导入词条。

    Returns
    -------
    (added, updated)
    """
    src = _normalize_source(source)

    with _lock:
        store = {k: dict(v) for k, v in _load().items()}
        target = store.setdefault(src, {})
        added, updated = 0, 0

        for raw_word, raw_phones in (entries or {}).items():
            word = (raw_word or "").strip().upper()
            phones = (raw_phones or "").strip()
            if not word or not phones:
                continue
            if word in target:
                if overwrite:
                    target[word] = phones
                    updated += 1
            else:
                target[word] = phones
                added += 1

        _save(store)
        return added, updated


def export_json(source: Optional[str] = None) -> Dict[str, Dict[str, str]]:
    with _lock:
        store = _load()
        if source:
            src = _normalize_source(source)
            return {src: dict(store.get(src, {}))}
        return {k: dict(v) for k, v in store.items()}


def export_csv(source: str) -> str:
    src = _normalize_source(source)
    entries = list_entries(src)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["word", "phonemes"])
    for word in sorted(entries.keys()):
        writer.writerow([word, entries[word]])
    return buf.getvalue()


def import_csv_text(source: str, csv_text: str, overwrite: bool = True) -> Tuple[int, int]:
    """
    解析 "word,phonemes" 两列 CSV（首行可以是表头 word,phonemes，会被自动跳过）。
    """
    src = _normalize_source(source)
    rows = list(csv.reader(io.StringIO(csv_text)))
    if not rows:
        return 0, 0

    start_idx = 0
    header = [c.strip().lower() for c in rows[0][:2]]
    if header == ["word", "phonemes"]:
        start_idx = 1

    entries: Dict[str, str] = {}
    for row in rows[start_idx:]:
        if len(row) < 2:
            continue
        entries[row[0]] = row[1]

    return bulk_import(src, entries, overwrite=overwrite)
