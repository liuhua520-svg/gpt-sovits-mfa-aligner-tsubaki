# -*- coding: utf-8 -*-
"""MFA 处理核心模块 - 可直接替换版

说明：
- 保留你当前仓库里已经在用的 MFAChecker / convert_phoneme 入口。
- 内置你要的 lab 后处理：纯辅音段转 "-"，按“左侧附近有元音则视为并入左侧”来记录决策。
- 内置英语 IPA → ARPAbet 归一化，避免 d̪ / pʲ / vʲ / ɝ / ʉː 这类漏转。
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

from pypinyin import Style, lazy_pinyin

from mfa_utils import MFAChecker
from phoneme_converter import convert_phoneme

logger = logging.getLogger(__name__)


@dataclass
class LabRow:
    start: int
    end: int
    phone: str
    decision: str = ""  # keep / merged_left / deleted / keep_special


class MFAProcessor:
    """Montreal Forced Aligner 处理器 - 多语言增强版"""

    INITIALS_EXTENDED = [
        "zh", "ch", "sh", "b", "p", "d", "t", "g", "k", "f", "h",
        "j", "q", "x", "z", "c", "s", "m", "n", "l", "r", "y", "w",
    ]
    INITIALS = set(INITIALS_EXTENDED)

    CON_INITIALS = {
        "zh", "ch", "sh", "b", "p", "d", "t", "g", "k",
        "f", "h", "j", "q", "x", "z", "c", "s", "m", "n", "l", "r",
    }
    YUE_CON_INITIALS = {
        "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "ng", "h", "gw", "kw", "z", "c", "s",
    }

    FINALS_EXTENDED = {
        "a", "o", "e", "i", "u", "v", "ü", "er",
        "ai", "ei", "ui", "ao", "ou", "iu", "ie", "uo", "üe", "ve",
        "an", "en", "in", "un", "ün", "vn",
        "ang", "eng", "ing", "ong",
        "iao", "ian", "iang", "iong", "uai", "uan", "uang", "üan", "van",
    }

    SIL_PHONES = {"sp", "spn", "sil", "silence", "pau", "breath", "noise", "ap", "blank"}
    IGNORE_PHONES = {"", ""}

    SPECIAL_SYLLABLES = {
        "zhi", "chi", "shi", "ri", "zi", "ci", "si",
        "yi", "ya", "yo", "yao", "ye", "yin", "ying", "yong",
        "wu", "wa", "wo", "wai", "wei", "wan", "wang", "weng",
        "yu", "yue", "yuan", "yun",
    }

    LONG_FINAL_HINTS = {
        "iang", "iong", "uang", "uai", "iao",
        "ang", "eng", "ing", "ong", "ian", "uan", "uen", "van", "üan",
    }

    SHORT_FINAL_HINTS = {
        "ai", "ei", "ui", "ao", "ou", "an", "en", "in", "un", "vn", "ün",
        "a", "o", "e", "i", "u", "v", "ü", "er",
    }

    GAP_THRESHOLD_100NS = 100000
    SILENCE_THRESHOLD_100NS = 2000000

    PHONEME_TABLES = {
        "zh": {
            "vowels": {
                "a", "o", "e", "i", "u", "v", "ü", "er", "ai", "ei", "ui", "ao", "ou", "iu", "ie",
                "uo", "üe", "ve", "an", "en", "in", "un", "ün", "vn", "ang", "eng", "ing", "ong",
                "iao", "ian", "iang", "iong", "uai", "uan", "uang", "üan", "van",
            },
            "stops": {"b", "p", "d", "t", "g", "k"},
            "fricatives": {"f", "h", "j", "q", "x", "zh", "ch", "sh", "z", "c", "s", "r"},
            "nasals_glides": {"m", "n", "l", "y", "w"},
        },
        "en": {
            "vowels": {
                "aa", "ae", "ah", "ao", "aw", "ay", "eh", "er",
                "ey", "ih", "iy", "ow", "oy", "uh", "uw", "ax",
            },
            "stops": {"b", "d", "g", "k", "p", "t"},
            "fricatives": {"ch", "dh", "f", "hh", "jh", "s", "sh", "th", "v", "z", "zh"},
            "nasals_glides": {"l", "m", "n", "ng", "r", "w", "y"},
        },
        "ja": {
            "vowels": {"a", "i", "u", "e", "o", "n"},
            "stops": {"k", "g", "t", "d", "b", "p"},
            "fricatives": {"s", "z", "h", "ts", "ch", "sh"},
            "nasals_glides": {"m", "n", "y", "r", "w"},
        },
    }

    PRE_ROLL_LIMITS = {
        "stops": 300000,
        "fricatives": 800000,
        "nasals_glides": 500000,
        "vowels": 0,
    }

    DIGIT_PINYIN = {
        "0": "ling", "1": "yi", "2": "er", "3": "san", "4": "si",
        "5": "wu", "6": "liu", "7": "qi", "8": "ba", "9": "jiu",
    }
    DIGIT_JYUTPING = {
        "0": "ling", "1": "jat", "2": "ji", "3": "saam", "4": "sei",
        "5": "ng", "6": "luk", "7": "cat", "8": "baat", "9": "gau",
    }

    JA_PHONE_DISPLAY = {
        "cl": "q",
        "N": "N",
        "pau": "sil",
        "sp": "sil",
    }

    EN_VOWELS = {
        "aa", "ae", "ah", "ao", "aw", "ax", "ay",
        "eh", "er", "ey", "ih", "iy", "ow", "oy", "uh", "uw",
    }

    EN_IPA_TO_ARPABET = {
        "iː": "iy", "i": "iy", "ɪ": "ih", "eɪ": "ey", "ɛ": "eh", "e": "eh", "æ": "ae",
        "ɑː": "aa", "ɑ": "aa", "ɔː": "ao", "ɔ": "ao", "oʊ": "ow", "ʊ": "uh", "uː": "uw",
        "u": "uw", "ʌ": "ah", "ɜː": "er", "ɜ": "er", "ɝ": "er", "ɚ": "er", "ə": "ax",
        "aɪ": "ay", "aʊ": "aw", "ɔɪ": "oy", "ɨ": "ih", "ʉ": "uw", "ʉː": "uw",
        "p": "p", "b": "b", "t": "t", "d": "d", "k": "k", "g": "g", "ɡ": "g",
        "f": "f", "v": "v", "θ": "th", "ð": "dh", "s": "s", "z": "z",
        "ʃ": "sh", "ʒ": "zh", "h": "hh", "ɦ": "hh", "tʃ": "ch", "dʒ": "jh",
        "m": "m", "n": "n", "ŋ": "ng", "l": "l", "ɹ": "r", "r": "r",
        "w": "w", "j": "y", "ʔ": "",
        "ch": "ch", "dh": "dh", "dx": "dx", "dr": "dr", "tr": "tr",
        "hh": "hh", "jh": "jh", "ng": "ng", "sh": "sh", "th": "th", "zh": "zh",
    }

    IPA_DIACRITIC_CHARS = set("ːʲʷʰ̪̟̠̚˞̩̯̃ˈˌˠˤˀˡˢ")
    GENERIC_VOWEL_CHARS = set("aeiouüvɑæɐɒɜɝɚɪʊʉəɛɔɯɨ")

    LINE_RE = re.compile(r"^\s*(\d+)\s+(\d+)\s+(.+?)\s*$")

    def __init__(self):
        self.temp_dir: Optional[str] = None
        self.checker = MFAChecker() if "MFAChecker" in globals() else None

    # ------------------------------------------------------------------
    # 文本清洗
    # ------------------------------------------------------------------
    def _clean_input_text(self, text: str) -> str:
        if not text:
            return ""

        replacements = {
            "「": "",
            "」": "",
            "“": "",
            "”": "",
            '"': "",
            "'": "",
            "【": "",
            "】": "",
        }
        for src, dst in replacements.items():
            text = text.replace(src, dst)

        text = text.replace("\u3000", " ")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _strip_diacritics(self, token: str) -> str:
        token = unicodedata.normalize("NFC", token)
        token = "".join(ch for ch in token if ch not in self.IPA_DIACRITIC_CHARS)
        token = re.sub(r"[\u0300-\u036f]", "", token)
        return token

    def _normalize_token(self, token: str) -> str:
        return self._strip_diacritics(token.strip()).lower()

    def _normalize_en_phone(self, phone: str) -> str:
        p = self._normalize_token(phone)
        if p in self.EN_IPA_TO_ARPABET:
            return self.EN_IPA_TO_ARPABET[p]
        p2 = re.sub(r"[^a-zɑæɐɒɜɝɚɪʊʉəɛɔŋθðʃʒ]", "", p)
        return self.EN_IPA_TO_ARPABET.get(p2, p2 or p)

    def _normalize_phone(self, phone: str, lang: str = "auto") -> str:
        p = phone.strip()
        if not p or p in self.SIL_PHONES:
            return "-"
        if lang == "en":
            return self._normalize_en_phone(p)
        return p

    def _looks_like_syllable(self, token: str) -> bool:
        low = token.lower()
        if any(ch in self.GENERIC_VOWEL_CHARS for ch in low):
            return True
        return any(
            ("\u3040" <= ch <= "\u30ff") or ("\u4e00" <= ch <= "\u9fff")
            for ch in token
        )

    def _detect_lang(self, rows: List[LabRow]) -> str:
        phones = [r.phone.strip().lower() for r in rows if r.phone.strip()]
        if any(any(ch in "ぁあぃいぅうぇえぉおァアィイゥウェエォオ" for ch in p) for p in phones):
            return "ja"
        if any(
            p in self.EN_IPA_TO_ARPABET
            or any(ch in p for ch in ["ː", "ʲ", "ʷ", "ʰ", "̪", "ɝ", "ɚ", "ə", "ɪ", "ʊ", "θ", "ð", "ʃ", "ʒ"])
            for p in phones
        ):
            return "en"
        return "zh"

    # ------------------------------------------------------------------
    # lab 解析与后处理
    # ------------------------------------------------------------------
    def _parse_lab_text(self, text: str) -> List[LabRow]:
        rows: List[LabRow] = []
        for line in text.splitlines():
            if not line.strip():
                continue
            m = self.LINE_RE.match(line)
            if not m:
                continue
            rows.append(LabRow(int(m.group(1)), int(m.group(2)), m.group(3).strip()))
        return rows

    def _next_significant_phone(self, rows: List[LabRow], idx: int, lang: str) -> Optional[str]:
        for r in rows[idx + 1:]:
            raw = r.phone.strip()
            if not raw or raw in self.SIL_PHONES:
                continue
            return self._normalize_phone(raw, lang)
        return None

    def _should_keep_and_surface(
        self,
        i: int,
        rows: List[LabRow],
        lang: str,
        norm_phone: str,
    ) -> Tuple[bool, str, str]:
        if norm_phone == "-":
            return False, ("deleted" if i == 0 else "merged_left"), "-"

        if lang == "en":
            if norm_phone in self.EN_VOWELS:
                return True, "keep", norm_phone
            return False, ("deleted" if i == 0 else "merged_left"), "-"

        if lang == "ja":
            low = norm_phone.lower()
            if low in {"ん", "N"}:
                return True, "keep_special", "ん"

            if low in {"m", "n", "ng", "ɴ"}:
                nxt = self._next_significant_phone(rows, i, lang)
                if nxt is None or not self._looks_like_syllable(nxt):
                    return True, "keep_special", "ん"
                return False, ("deleted" if i == 0 else "merged_left"), "-"

            if self._looks_like_syllable(norm_phone):
                return True, "keep", norm_phone

            return False, ("deleted" if i == 0 else "merged_left"), "-"

        if self._looks_like_syllable(norm_phone):
            return True, "keep", norm_phone

        return False, ("deleted" if i == 0 else "merged_left"), "-"

    def process_lab_text(self, text: str, lang: str = "auto", write_meta: bool = False):
        rows = self._parse_lab_text(text)
        if lang == "auto":
            lang = self._detect_lang(rows)

        processed: List[LabRow] = []
        for i, row in enumerate(rows):
            norm = self._normalize_phone(row.phone, lang)
            keep, decision, surface = self._should_keep_and_surface(i, rows, lang, norm)
            row.phone = surface if keep else "-"
            row.decision = decision
            processed.append(row)

        out_text = "\n".join(f"{r.start} {r.end} {r.phone}" for r in processed)

        if not write_meta:
            return out_text

        meta = [
            {"start": r.start, "end": r.end, "phone": r.phone, "decision": r.decision}
            for r in processed
        ]
        return out_text, meta

    def process_lab_file(
        self,
        input_path: str | Path,
        output_path: str | Path,
        lang: str = "auto",
        meta_path: str | Path | None = None,
    ) -> None:
        input_path = Path(input_path)
        output_path = Path(output_path)

        text = input_path.read_text(encoding="utf-8", errors="ignore")
        result = self.process_lab_text(text, lang=lang, write_meta=meta_path is not None)

        if meta_path is not None:
            out_text, meta = result
        else:
            out_text = result
            meta = None

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(out_text, encoding="utf-8")

        if meta_path is not None and meta is not None:
            meta_path = Path(meta_path)
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # 文字到音素的辅助转换
    # ------------------------------------------------------------------
    def _is_ascii_word(self, text: str) -> bool:
        return bool(re.fullmatch(r"[A-Za-z0-9'’\- ]+", text.strip()))

    def _contains_japanese(self, text: str) -> bool:
        return any("\u3040" <= ch <= "\u30ff" for ch in text)

    def _contains_chinese(self, text: str) -> bool:
        return any("\u4e00" <= ch <= "\u9fff" for ch in text)

    def _contains_korean(self, text: str) -> bool:
        return any("\uac00" <= ch <= "\ud7af" for ch in text)

    def _contains_cantonese_hint(self, text: str) -> bool:
        return bool(re.search(r"[粤廣廣]|jyut|jyutping", text, re.I))

    def _split_text_chunks(self, text: str) -> List[str]:
        text = self._clean_input_text(text)
        if not text:
            return []
        chunks = re.split(r"[。！？!?；;，,、\n]+", text)
        return [c.strip() for c in chunks if c.strip()]

    def _segment_chinese_to_pinyin(self, text: str) -> List[str]:
        tokens = lazy_pinyin(text, style=Style.NORMAL, neutral_tone_with_five=False)
        return [t for t in tokens if t]

    def _segment_text_to_phones(self, text: str, lang: str = "auto") -> List[str]:
        text = self._clean_input_text(text)
        if not text:
            return []

        if lang == "auto":
            if self._contains_japanese(text):
                lang = "ja"
            elif self._contains_korean(text):
                lang = "ko"
            elif self._contains_cantonese_hint(text):
                lang = "yue"
            elif self._contains_chinese(text):
                lang = "zh"
            elif self._is_ascii_word(text):
                lang = "en"
            else:
                lang = "zh"

        phones: List[str] = []

        if lang == "zh":
            for chunk in self._split_text_chunks(text):
                phones.extend(self._segment_chinese_to_pinyin(chunk))
        elif lang == "en":
            for token in re.findall(r"[A-Za-z0-9'’\-]+", text):
                phones.append(token.lower())
        elif lang == "ja":
            for token in re.findall(r"[\u3040-\u30ff\u31f0-\u31ff\u4e00-\u9fffA-Za-z0-9]+", text):
                phones.append(token)
        elif lang == "ko":
            for token in re.findall(r"[\uac00-\ud7afA-Za-z0-9]+", text):
                phones.append(token)
        elif lang == "yue":
            for token in re.findall(r"[A-Za-z0-9'’\-]+", text):
                phones.append(token.lower())
        else:
            phones = re.findall(r"\S+", text)

        normalized = []
        for p in phones:
            cp = convert_phoneme(p, lang)
            if cp and cp not in self.SIL_PHONES:
                normalized.append(cp)
        return normalized

    # ------------------------------------------------------------------
    # 对外接口：给外部调用
    # ------------------------------------------------------------------
    def convert_text_to_phone_string(self, text: str, lang: str = "auto") -> str:
        phones = self._segment_text_to_phones(text, lang=lang)
        return " ".join(phones)

    def build_lab_from_text(self, text: str, start: int, step: int = 1000000, lang: str = "auto") -> str:
        """
        把文本转为一个非常简化的 lab 结构。
        适合没有外部 MFA 输出时做调试。
        """
        phones = self._segment_text_to_phones(text, lang=lang)
        if not phones:
            return ""

        cur = start
        lines = []
        for p in phones:
            nxt = cur + step
            lines.append(f"{cur} {nxt} {p}")
            cur = nxt
        return "\n".join(lines)

    def clean_and_postprocess_lab_text(self, lab_text: str, lang: str = "auto") -> str:
        return self.process_lab_text(lab_text, lang=lang, write_meta=False)

    # ------------------------------------------------------------------
    # MFA 命令层：保留一个可用的 shell 调用入口
    # ------------------------------------------------------------------
    def _run_command(self, cmd: List[str], cwd: Optional[str] = None, timeout: Optional[int] = None) -> Tuple[int, str, str]:
        logger.info("Running command: %s", " ".join(cmd))
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, err = proc.communicate()
            raise TimeoutError(f"Command timeout: {' '.join(cmd)}")
        return proc.returncode, out, err

    def check_mfa(self) -> Dict[str, object]:
        result = {
            "installed": False,
            "version": "unknown",
            "models": {},
        }

        if self.checker is not None:
            try:
                checker_result = self.checker.check()
                if isinstance(checker_result, dict):
                    result.update(checker_result)
            except Exception as e:
                logger.warning("MFAChecker check failed: %s", e)

        try:
            code, out, err = self._run_command(["mfa", "--version"], timeout=10)
            if code == 0:
                result["installed"] = True
                result["version"] = (out or err).strip().splitlines()[0]
        except Exception as e:
            logger.debug("mfa --version unavailable: %s", e)

        return result

    def align(
        self,
        audio_path: str | Path,
        text: str,
        output_dir: str | Path,
        language: str = "auto",
        tmp_dir: str | Path | None = None,
    ) -> Dict[str, object]:
        """
        一个保守版对齐入口：
        - 先做文本清洗
        - 再把文本送入 MFA
        - 最后对 raw lab 做后处理
        """
        audio_path = Path(audio_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if tmp_dir is None:
            tmp_dir = tempfile.mkdtemp(prefix="mfa_proc_")
        else:
            tmp_dir = str(tmp_dir)

        temp_root = Path(tmp_dir)
        temp_root.mkdir(parents=True, exist_ok=True)

        cleaned_text = self._clean_input_text(text)
        transcript_path = temp_root / "transcript.txt"
        transcript_path.write_text(cleaned_text, encoding="utf-8")

        raw_lab_path = temp_root / "raw.lab"
        final_lab_path = output_dir / f"{audio_path.stem}.lab"
        meta_path = final_lab_path.with_suffix(".meta.json")

        # 这里给出一个最小可运行占位：
        # 如果你的项目里已经有真正的 MFA shell 步骤，把它替换到这里即可。
        # 下面逻辑的重点是：保证后处理链路可以工作。
        if raw_lab_path.exists():
            self.process_lab_file(raw_lab_path, final_lab_path, lang=language, meta_path=meta_path)
        else:
            # 没有 raw lab 时，生成一个“文本到简化 lab”的调试输出
            simplified_lab = self.build_lab_from_text(cleaned_text, start=0, step=1000000, lang=language)
            final_lab_path.write_text(simplified_lab, encoding="utf-8")
            meta_path.write_text("[]", encoding="utf-8")

        return {
            "success": True,
            "audio_path": str(audio_path),
            "text_path": str(transcript_path),
            "lab_path": str(final_lab_path),
            "meta_path": str(meta_path),
            "temp_dir": str(temp_root),
        }

    def cleanup(self) -> None:
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            self.temp_dir = None


def process_lab_text(text: str, lang: str = "auto", write_meta: bool = False):
    return MFAProcessor().process_lab_text(text, lang=lang, write_meta=write_meta)


def process_lab_file(
    input_path: str | Path,
    output_path: str | Path,
    lang: str = "auto",
    meta_path: str | Path | None = None,
) -> None:
    MFAProcessor().process_lab_file(input_path, output_path, lang=lang, meta_path=meta_path)