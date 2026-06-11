# -*- coding: utf-8 -*-
"""
MFA 处理核心模块 - 多语言增强版 v9.4
新增：长音频 RMS/百分位静音分割预处理，解决长音频 MFA 对齐质量下降问题。
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
import shutil
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from pypinyin import lazy_pinyin, Style
from mfa_utils import MFAChecker
from phoneme_converter import convert_phoneme, build_ja_hiragana_lab, merge_lab_silence

logger = logging.getLogger(__name__)

# ── Module-level cache: SudachiPy dict loads in ~4 min on first call.
# Caching it here means subsequent requests reuse the same object. ──────────
_ja_tokenizer: Optional[object] = None


class MFAProcessor:
    """Montreal Forced Aligner 处理器 - 多语言增强版 v9.4"""

    INITIALS_EXTENDED = [
        "zh", "ch", "sh",
        "b", "p", "d", "t", "g", "k",
        "f", "h", "j", "q", "x", "z", "c", "s",
        "m", "n", "l", "r",
        "y", "w",
    ]
    INITIALS = set(INITIALS_EXTENDED)

    CON_INITIALS = {
        "zh", "ch", "sh",
        "b", "p", "d", "t", "g", "k",
        "f", "h", "j", "q", "x", "z", "c", "s",
        "m", "n", "l", "r",
    }

    YUE_CON_INITIALS = {
        "b", "p", "m", "f",
        "d", "t", "n", "l",
        "g", "k", "ng", "h",
        "gw", "kw",
        "z", "c", "s",
    }

    FINALS_EXTENDED = {
        "a", "o", "e", "i", "u", "v", "ü", "er",
        "ai", "ei", "ui", "ao", "ou", "iu", "ie", "uo", "üe", "ve",
        "an", "en", "in", "un", "ün", "vn",
        "ang", "eng", "ing", "ong",
        "iao", "ian", "iang", "iong", "uai", "uan", "uang", "üan", "van",
    }

    SIL_PHONES = {"sp", "spn", "sil", "silence", "pau", "breath", "noise", "ap", "blank"}
    IGNORE_PHONES = {"", "<eps>"}

    SPECIAL_SYLLABLES = {
        "zhi", "chi", "shi", "ri", "zi", "ci", "si",
        "yi", "ya", "yo", "yao", "ye", "yin", "ying", "yong",
        "wu", "wa", "wo", "wai", "wei", "wan", "wang", "weng",
        "yu", "yue", "yuan", "yun",
    }

    LONG_FINAL_HINTS = {
        "iang", "iong", "uang", "uai", "iao",
        "ang", "eng", "ing", "ong",
        "ian", "uan", "uen", "van", "üan",
    }
    SHORT_FINAL_HINTS = {
        "ai", "ei", "ui", "ao", "ou",
        "an", "en", "in", "un", "vn", "ün",
        "a", "o", "e", "i", "u", "v", "ü", "er",
    }

    GAP_THRESHOLD_100NS = 100000
    SILENCE_THRESHOLD_100NS = 2000000

    PHONEME_TABLES = {
        'zh': {
            'vowels': {
                'a', 'o', 'e', 'i', 'u', 'v', 'ü', 'er',
                'ai', 'ei', 'ui', 'ao', 'ou', 'iu', 'ie', 'uo', 'üe', 've',
                'an', 'en', 'in', 'un', 'ün', 'vn',
                'ang', 'eng', 'ing', 'ong',
                'iao', 'ian', 'iang', 'iong', 'uai', 'uan', 'uang', 'üan', 'van'
            },
            'stops': {'b', 'p', 'd', 't', 'g', 'k'},
            'fricatives': {'f', 'h', 'j', 'q', 'x', 'zh', 'ch', 'sh', 'z', 'c', 's', 'r'},
            'nasals_glides': {'m', 'n', 'l', 'y', 'w'}
        },
        'en': {
            'vowels': {
                'aa', 'ae', 'ah', 'ao', 'aw', 'ay', 'eh', 'er', 'ey', 'ih', 'iy',
                'ow', 'oy', 'uh', 'uw'
            },
            'stops': {'b', 'd', 'g', 'k', 'p', 't'},
            'fricatives': {'ch', 'dh', 'f', 'hh', 'jh', 's', 'sh', 'th', 'v', 'z', 'zh'},
            'nasals_glides': {'l', 'm', 'n', 'ng', 'r', 'w', 'y'}
        },
        'ja': {
            'vowels': {'a', 'i', 'u', 'e', 'o', 'n'},
            'stops': {'k', 'g', 't', 'd', 'b', 'p'},
            'fricatives': {'s', 'z', 'h', 'ts', 'ch', 'sh'},
            'nasals_glides': {'m', 'n', 'y', 'r', 'w'}
        }
    }

    PRE_ROLL_LIMITS = {
        'stops': 300000,
        'fricatives': 800000,
        'nasals_glides': 500000,
        'vowels': 0
    }

    DIGIT_PINYIN = {
        '0': 'ling', '1': 'yi',  '2': 'er',  '3': 'san', '4': 'si',
        '5': 'wu',   '6': 'liu', '7': 'qi',  '8': 'ba',  '9': 'jiu',
    }

    DIGIT_JYUTPING = {
        '0': 'ling', '1': 'jat', '2': 'ji',   '3': 'saam', '4': 'sei',
        '5': 'ng',   '6': 'luk', '7': 'cat',  '8': 'baat', '9': 'gau',
    }

    JA_PHONE_DISPLAY = {
        'cl':  'q',
        'N':   'N',
        'pau': 'sil',
        'sp':  'sil',
    }

    # ── 长音频分割阈值 ────────────────────────────────────────────────────────
    # 超过此时长的音频将预先在静音处分割，分段送入 MFA 以保证对齐质量。
    LONG_AUDIO_THRESHOLD_SEC: float = 25.0
    # 每个分段的最大时长；超过则强制截断。
    MAX_SEGMENT_SEC: float = 27.0
    # 用于 RMS 静音检测的滑窗（ms）和步长（ms）
    RMS_WINDOW_MS: int = 20
    RMS_HOP_MS: int = 10
    # 百分位阈值：低于该能量百分位的帧视为静音
    RMS_PERCENTILE: float = 15.0
    # 连续静音至少多少 ms 才视为可分割点
    MIN_SILENCE_MS: int = 200

    def __init__(self):
        self.temp_dir: Optional[str] = None

    # =====================================================================
    # 文本标点符号规范化清洗
    # =====================================================================
    def _clean_input_text(self, text: str) -> str:
        """
        清洗文本标点符号：
        1. 「」、""、"" 删除（替换为空字符串）
        2. （）、《》、()、＜＞、<>、【】 视为逗号（替换为 ","）
        """
        if not text:
            return ""
        ignore_pattern = r'[「」"""]'
        text = re.sub(ignore_pattern, "", text)
        comma_pattern = r'[（）《》()＜＞<>【】]'
        text = re.sub(comma_pattern, ",", text)
        return text

    # =====================================================================
    # 多语言环境检查
    # =====================================================================
    def _check_zh_environment(self) -> Tuple[bool, str]:
        try:
            from pypinyin import lazy_pinyin, Style
            lazy_pinyin("测试", style=Style.NORMAL)
            return True, "✓ 中文环境就绪（pypinyin已安装）"
        except ImportError:
            return False, "❌ 缺失中文支持：pip install pypinyin"
        except Exception as e:
            return False, f"❌ 中文环境错误：{str(e)}"

    def _check_en_environment(self) -> Tuple[bool, str]:
        try:
            mfa_ok, mfa_msg = MFAChecker.check_mfa_installed()
            if mfa_ok:
                return True, "✓ 英语环境就绪（MFA已安装）"
            else:
                return False, f"❌ MFA未安装：{mfa_msg}"
        except Exception as e:
            return False, f"❌ 英语环境错误：{str(e)}"

    def _check_ja_environment(self) -> Tuple[bool, str]:
        """检查日语环境（首次调用时加载字典，后续直接复用缓存，避免每次重复 ~4 min 加载）"""
        global _ja_tokenizer
        try:
            from sudachipy import Dictionary
            if _ja_tokenizer is None:
                _ja_tokenizer = Dictionary().create()
            test_result = _ja_tokenizer.tokenize("テスト")
            if test_result and len(test_result) > 0:
                return True, "✓ 日语环境就绪（sudachipy + 字典已就绪）"
            else:
                return False, "❌ 日语字典初始化失败"
        except ImportError:
            return False, "❌ 缺失日语支持（sudachipy未安装）\n   请运行：pip install sudachipy sudachidict-core"
        except Exception as e:
            error_msg = str(e)
            if "dictionary" in error_msg.lower():
                return False, "❌ 日语字典资源不可用\n   请重新安装：pip install --force-reinstall sudachidict-core"
            else:
                return False, f"❌ 日语环境错误：{error_msg}"

    def _check_ko_environment(self) -> Tuple[bool, str]:
        try:
            mfa_ok, mfa_msg = MFAChecker.check_mfa_installed()
            if not mfa_ok:
                return False, f"❌ MFA未安装：{mfa_msg}"
            try:
                import jamo
                return True, "✓ 韩语环境就绪（MFA + jamo已安装）"
            except ImportError:
                return False, "❌ 缺失韩语支持（jamo）\n   请运行：pip install jamo"
        except Exception as e:
            return False, f"❌ 韩语环境错误：{str(e)}"

    def _check_yue_environment(self) -> Tuple[bool, str]:
        try:
            import pycantonese
            return True, "✓ 粤语环境就绪（pycantonese已安装）"
        except ImportError:
            return False, "❌ 缺失粤语支持（pycantonese）\n   请运行：pip install pycantonese"
        except Exception as e:
            return False, f"❌ 粤语环境错误：{str(e)}"

    def _check_language_environment(self, lang: str) -> Tuple[bool, str]:
        logger.info(f"\n{'='*60}\n检查语言环境：{lang.upper()}\n{'='*60}")
        if lang in ('zh', 'cmn'):
            ok, msg = self._check_zh_environment()
        elif lang == 'en':
            ok, msg = self._check_en_environment()
        elif lang == 'ja':
            ok, msg = self._check_ja_environment()
        elif lang == 'ko':
            ok, msg = self._check_ko_environment()
        elif lang == 'yue':
            ok, msg = self._check_yue_environment()
        else:
            logger.warning(f"未知语言代码：{lang}")
            return True, f"未知语言：{lang}（跳过环境检查）"
        logger.info(msg)
        return ok, msg

    # =====================================================================
    # 文本处理
    # =====================================================================
    def _text_to_pinyin_notone(self, text: str) -> str:
        phones = lazy_pinyin(text, style=Style.NORMAL, errors="ignore")
        phones = [p.strip().lower() for p in phones if p and p.strip()]
        return " ".join(phones)

    def _text_to_jyutping(self, text: str) -> str:
        try:
            import pycantonese
            result = pycantonese.characters_to_jyutping(text)
            syls: List[str] = []
            for _char, jyut in result:
                if jyut:
                    jyut_no_tone = re.sub(r'[0-9]+$', '', jyut).strip().lower()
                    if jyut_no_tone:
                        syls.append(jyut_no_tone)
            return " ".join(syls)
        except (ImportError, Exception) as e:
            logger.warning(f"粤语转换失败: {e}")
            return ""

    def _normalize_no_tone_pinyin(self, text: str) -> List[str]:
        if not text:
            return []
        tokens: List[str] = []
        for part in text.split():
            p = part.strip().lower()
            if p:
                p = re.sub(r"[^a-züv]+", "", p)
                if p:
                    tokens.append(p)
        return tokens

    def _normalize_jyutping(self, text: str) -> List[str]:
        if not text:
            return []
        tokens: List[str] = []
        for part in text.split():
            p = part.strip().lower()
            if p:
                p = re.sub(r"[^a-z]+", "", p)
                if p:
                    tokens.append(p)
        return tokens

    def _segment_to_syllables(self, segment: str) -> List[str]:
        return self._normalize_no_tone_pinyin(self._text_to_pinyin_notone(segment))

    def _segment_to_jyutping(self, segment: str) -> List[str]:
        return self._normalize_jyutping(self._text_to_jyutping(segment))

    # =====================================================================
    # 语言/词语类型检测
    # =====================================================================
    def _is_english_word(self, word: str) -> bool:
        return bool(re.match(r"^[a-zA-Z''\-]+$", word.strip()))

    def _is_digit_char(self, word: str) -> bool:
        return word.strip() in '0123456789'

    # =====================================================================
    # Phone 清洗
    # =====================================================================
    def _clean_phone(self, phone: str) -> str:
        if not phone:
            return ""
        phone = phone.strip()
        tone_marks = ["˥", "˧", "˩", "˨", "˦", "˧˥", "˥˩", "˩˧", "˨˩", "˥˧", "˧˨", "˨˧"]
        for mark in tone_marks:
            phone = phone.replace(mark, "")
        return phone.strip().lower()

    def _is_silence_phone(self, phone: str) -> bool:
        return (phone or "").strip().lower() in self.SIL_PHONES

    def _clean_phone_token(self, phone: str) -> str:
        phone = (phone or "").strip().lower().replace("ü", "v")
        phone = re.sub(r"[^a-zA-Zv]+", "", phone)
        return phone.strip()

    def _extract_phone_items(self, tier) -> List[Tuple[int, int, str]]:
        items: List[Tuple[int, int, str]] = []
        for interval in tier:
            mark = self._clean_phone((interval.mark or "").strip())
            if mark in self.IGNORE_PHONES:
                continue
            start = int(interval.minTime * 10000000)
            end = int(interval.maxTime * 10000000)
            items.append((start, end, mark))
        items.sort(key=lambda x: (x[0], x[1]))
        return items

    def _get_phones_for_word(
        self,
        word_start: int,
        word_end: int,
        phone_items: List[Tuple[int, int, str]]
    ) -> List[Tuple[int, int, str]]:
        result = [
            (s, e, p) for s, e, p in phone_items
            if s >= word_start and s < word_end
            and p not in self.SIL_PHONES
            and p not in self.IGNORE_PHONES
        ]
        result.sort(key=lambda x: x[0])
        return result

    def _get_arpabet_entries(
        self,
        word_start: int,
        word_end: int,
        phone_items: List[Tuple[int, int, str]]
    ) -> List[Tuple[int, int, str]]:
        word_phones = self._get_phones_for_word(word_start, word_end, phone_items)
        entries: List[Tuple[int, int, str]] = []
        for s, e, p in word_phones:
            arp = convert_phoneme(p, 'en')
            if arp and arp not in self.SIL_PHONES:
                entries.append((s, e, arp))
        return entries

    def _get_romaji_entries(
        self,
        word_start: int,
        word_end: int,
        phone_items: List[Tuple[int, int, str]]
    ) -> List[Tuple[int, int, str]]:
        word_phones = self._get_phones_for_word(word_start, word_end, phone_items)
        entries: List[Tuple[int, int, str]] = []
        for s, e, p in word_phones:
            romaji = convert_phoneme(p, 'ja')
            if romaji is None:
                logger.debug(f"  跳过静音: {p}")
                continue
            elif romaji == "":
                logger.debug(f"  省略符号: {p}")
                continue
            elif romaji not in self.SIL_PHONES:
                entries.append((s, e, romaji))
                logger.debug(f"  转换: {p} → {romaji}")
        logger.debug(f"提取 Romaji 条目: {len(entries)} 个 from [{word_start}, {word_end}]")
        return entries

    def _get_jamo_entries(
        self,
        word_start: int,
        word_end: int,
        phone_items: List[Tuple[int, int, str]]
    ) -> List[Tuple[int, int, str]]:
        word_phones = self._get_phones_for_word(word_start, word_end, phone_items)
        entries: List[Tuple[int, int, str]] = []
        for s, e, p in word_phones:
            jamo = convert_phoneme(p, 'ko')
            if jamo is None:
                logger.debug(f"  跳过静音: {p}")
                continue
            elif jamo == "":
                logger.debug(f"  省略符号: {p}")
                continue
            elif jamo not in self.SIL_PHONES:
                entries.append((s, e, jamo))
                logger.debug(f"  转换: {p} → {jamo}")
        logger.debug(f"提取 Jamo 条目: {len(entries)} 个 from [{word_start}, {word_end}]")
        return entries

    # =====================================================================
    # 音节工具
    # =====================================================================
    def _syllable_weight(self, syl: str) -> float:
        syl = (syl or "").strip().lower()
        if not syl:
            return 1.0
        weight = 1.0 + min(len(syl), 8) * 0.10
        if syl in self.SPECIAL_SYLLABLES:
            weight += 0.10
        if any(syl.endswith(hint) for hint in self.LONG_FINAL_HINTS):
            weight += 0.20
        elif any(syl.endswith(hint) for hint in self.SHORT_FINAL_HINTS):
            weight += 0.08
        if syl.startswith(("zh", "ch", "sh")):
            weight += 0.12
        else:
            for ini in self.INITIALS_EXTENDED[:6]:
                if syl.startswith(ini):
                    weight += 0.08
                    break
        return max(weight, 0.75)

    def _split_pinyin_syllable(self, syl: str) -> Tuple[str, str]:
        syl = (syl or "").strip().lower().replace("ü", "v")
        if not syl:
            return "", ""
        for ini in self.INITIALS_EXTENDED:
            if syl.startswith(ini):
                final = syl[len(ini):]
                return ini, final
        return "", syl

    def _get_syllable_nucleus(self, syl: str) -> str:
        syl = (syl or "").strip().lower().replace("ü", "v")
        if not syl:
            return ""
        _, final = self._split_pinyin_syllable(syl)
        if not final:
            return ""
        for ch in final:
            if ch in "aeiouv":
                return ch
        return final[0] if final else ""

    # =====================================================================
    # con 标记逻辑
    # =====================================================================
    def _has_con_onset(self, syl: str, lang: str = 'zh') -> bool:
        syl = (syl or "").strip().lower().replace("ü", "v")
        if not syl:
            return False
        if lang in ('zh', 'cmn'):
            for onset in sorted(self.CON_INITIALS, key=len, reverse=True):
                if syl.startswith(onset):
                    return True
            return False
        if lang == 'yue':
            for onset in sorted(self.YUE_CON_INITIALS, key=len, reverse=True):
                if syl.startswith(onset) and len(syl) > len(onset):
                    return True
            return False
        return False

    def _get_con_boundary(
        self,
        syl_start: int,
        syl_end: int,
        phone_items: List[Tuple[int, int, str]]
    ) -> int:
        if phone_items:
            syl_phones = [
                (s, e, p) for s, e, p in phone_items
                if s >= syl_start and s < syl_end
                and p not in self.SIL_PHONES
                and p not in self.IGNORE_PHONES
            ]
            if syl_phones:
                syl_phones.sort(key=lambda x: x[0])
                first_phone_end = syl_phones[0][1]
                syl_duration = syl_end - syl_start
                max_con_end = syl_start + syl_duration // 2
                con_boundary = min(first_phone_end, max_con_end)
                if con_boundary - syl_start < 100000:
                    con_boundary = syl_start + 100000
                if con_boundary >= syl_end - 100000:
                    con_boundary = syl_end - 100000
                return con_boundary
        syl_duration = syl_end - syl_start
        con_duration = max(int(syl_duration * 0.15), 100000)
        con_duration = min(con_duration, int(syl_duration * 0.40))
        return syl_start + con_duration

    def _make_con_entries(
        self,
        syl_start: int,
        syl_end: int,
        syl: str,
        phone_items: List[Tuple[int, int, str]],
        lang: str = 'zh'
    ) -> List[Tuple[int, int, str]]:
        if self._has_con_onset(syl, lang):
            con_boundary = self._get_con_boundary(syl_start, syl_end, phone_items)
            return [
                (syl_start, con_boundary, "-"),
                (con_boundary, syl_end, syl),
            ]
        else:
            return [(syl_start, syl_end, syl)]

    def _syllable_anchor_candidates(self, syl: str) -> List[str]:
        syl = (syl or "").strip().lower().replace("ü", "v")
        if not syl:
            return []
        onset, rhyme = self._split_pinyin_syllable(syl)
        nucleus = self._get_syllable_nucleus(syl)
        cands: List[str] = []

        def push(token: str):
            token = (token or "").strip().lower().replace("ü", "v")
            if token and token not in cands:
                cands.append(token)

        push(syl)
        push(rhyme)
        push(nucleus)
        if onset:
            push(onset)
        push(rhyme.replace("v", "u"))
        push(rhyme.replace("v", "i"))
        push(nucleus.replace("v", "u"))
        push(nucleus.replace("v", "i"))
        cleaned = re.sub(r"[^a-z]+", "", syl).lower()
        push(cleaned)
        return [c for c in cands if c]

    def _phone_matches_syllable(self, phone: str, syl: str) -> bool:
        phone = self._clean_phone_token(phone)
        if not phone:
            return False
        return phone in set(self._syllable_anchor_candidates(syl))

    # =====================================================================
    # 音节分配
    # =====================================================================
    def _distribute_syllables_in_word(
        self,
        word_start: int,
        word_end: int,
        syllables: List[str],
        phone_tier=None,
        lang: str = 'zh'
    ) -> List[Tuple[int, int, str]]:
        if not syllables:
            return []
        if len(syllables) == 1:
            return [(word_start, word_end, syllables[0])]
        return self._distribute_syllables_by_weight(word_start, word_end, syllables)

    def _distribute_syllables_by_weight(
        self,
        word_start: int,
        word_end: int,
        syllables: List[str]
    ) -> List[Tuple[int, int, str]]:
        if not syllables:
            return []
        weights = [self._syllable_weight(s) for s in syllables]
        total_weight = sum(weights) or 1.0
        word_duration = word_end - word_start
        result: List[Tuple[int, int, str]] = []
        current_time = word_start
        for i, syl in enumerate(syllables):
            if i == len(syllables) - 1:
                syl_end = word_end
            else:
                syl_duration = int(round(word_duration * (weights[i] / total_weight)))
                syl_end = current_time + syl_duration
            if syl_end - current_time < 60000:
                syl_end = current_time + 60000
            if i < len(syllables) - 1 and syl_end > word_end:
                syl_end = word_end
            result.append((current_time, syl_end, syl))
            current_time = syl_end
        return result

    # =====================================================================
    # 各语言 Word Tier 处理
    # =====================================================================
    def _process_zh_words(
        self,
        word_tier,
        phone_items: List[Tuple[int, int, str]],
        text: str
    ) -> List[str]:
        target_syls = self._normalize_no_tone_pinyin(self._text_to_pinyin_notone(text))
        lines: List[str] = []
        syl_index = 0

        for interval in word_tier:
            mark = getattr(interval, "mark", getattr(interval, "text", ""))
            mark = (mark or "").strip()
            start = int(interval.minTime * 10000000)
            end = int(interval.maxTime * 10000000)

            if not mark or mark in self.IGNORE_PHONES:
                continue
            if mark in self.SIL_PHONES or mark in ("sp", "spn"):
                lines.append(f"{start} {end} sil")
                continue
            if self._is_english_word(mark):
                entries = self._get_arpabet_entries(start, end, phone_items)
                if entries:
                    for s, e, p in entries:
                        lines.append(f"{s} {e} {p}")
                else:
                    lines.append(f"{start} {end} {mark.lower()}")
                continue
            if self._is_digit_char(mark):
                syl = self.DIGIT_PINYIN.get(mark.strip(), mark)
                for es, ee, el in self._make_con_entries(start, end, syl, phone_items, 'zh'):
                    lines.append(f"{es} {ee} {el}")
                syl_index = min(syl_index + 1, len(target_syls))
                continue
            clean_mark = re.sub(r"[^\w\u4e00-\u9fa5]+", "", mark)
            if not clean_mark:
                continue
            mark_syls = self._segment_to_syllables(clean_mark)
            syl_count = len(mark_syls)
            if syl_count == 0:
                continue
            current_syls = target_syls[syl_index: syl_index + syl_count]
            syl_index += syl_count
            if not current_syls:
                current_syls = mark_syls
            syl_lines = self._distribute_syllables_in_word(start, end, current_syls, None, 'zh')
            for s, e, syl in syl_lines:
                for es, ee, el in self._make_con_entries(s, e, syl, phone_items, 'zh'):
                    lines.append(f"{es} {ee} {el}")

        return lines

    def _process_en_words(
        self,
        word_tier,
        phone_items: List[Tuple[int, int, str]],
        text: str
    ) -> List[str]:
        lines: List[str] = []
        for interval in word_tier:
            mark = getattr(interval, "mark", getattr(interval, "text", ""))
            mark = (mark or "").strip()
            start = int(interval.minTime * 10000000)
            end = int(interval.maxTime * 10000000)
            if not mark or mark in self.IGNORE_PHONES:
                continue
            if mark in self.SIL_PHONES or mark in ("sp", "spn"):
                lines.append(f"{start} {end} sil")
                continue
            entries = self._get_arpabet_entries(start, end, phone_items)
            if entries:
                for s, e, p in entries:
                    lines.append(f"{s} {e} {p}")
            else:
                lines.append(f"{start} {end} {mark.lower()}")
        return lines

    def _process_ja_words(
        self,
        word_tier,
        phone_items: List[Tuple[int, int, str]],
        text: str
    ) -> List[str]:
        lines: List[str] = []
        for interval in word_tier:
            mark = getattr(interval, "mark", getattr(interval, "text", ""))
            mark = (mark or "").strip()
            start = int(interval.minTime * 10000000)
            end = int(interval.maxTime * 10000000)
            if not mark or mark in self.IGNORE_PHONES:
                continue
            if mark in self.SIL_PHONES or mark in ("sp", "spn"):
                lines.append(f"{start} {end} sil")
                continue
            if self._is_english_word(mark):
                entries = self._get_arpabet_entries(start, end, phone_items)
                if entries:
                    for s, e, p in entries:
                        lines.append(f"{s} {e} {p}")
                else:
                    lines.append(f"{start} {end} {mark.lower()}")
                continue
            entries = self._get_romaji_entries(start, end, phone_items)
            if entries:
                for s, e, p in entries:
                    lines.append(f"{s} {e} {p}")
            else:
                logger.warning(f"日语 word '{mark}' 无法从 Phone Tier 获取音素，跳过")
        return lines

    def _process_ko_words(
        self,
        word_tier,
        phone_items: List[Tuple[int, int, str]],
        text: str
    ) -> List[str]:
        lines: List[str] = []
        for interval in word_tier:
            mark = getattr(interval, "mark", getattr(interval, "text", ""))
            mark = (mark or "").strip()
            start = int(interval.minTime * 10000000)
            end = int(interval.maxTime * 10000000)
            if not mark or mark in self.IGNORE_PHONES:
                continue
            if mark in self.SIL_PHONES or mark in ("sp", "spn"):
                lines.append(f"{start} {end} sil")
                continue
            if self._is_english_word(mark):
                entries = self._get_arpabet_entries(start, end, phone_items)
                if entries:
                    for s, e, p in entries:
                        lines.append(f"{s} {e} {p}")
                else:
                    lines.append(f"{start} {end} {mark.lower()}")
                continue
            if self._is_korean_text(mark):
                syllable_entries = self._decompose_korean_syllable_with_onset(
                    start, end, mark, phone_items=phone_items
                )
                if syllable_entries:
                    for s, e, p in syllable_entries:
                        lines.append(f"{s} {e} {p}")
                else:
                    logger.warning(f"无法处理韩语 word '{mark}'，跳过")
            else:
                logger.warning(f"非韩文 word '{mark}'，跳过")
        return lines

    def _is_korean_text(self, text: str) -> bool:
        if not text:
            return False
        for char in text:
            code = ord(char)
            if (0xAC00 <= code <= 0xD7A3
                    or 0x3130 <= code <= 0x318F
                    or 0x1100 <= code <= 0x11FF):
                return True
        return False

    def _get_korean_initial_consonant(self, char: str) -> Optional[str]:
        if not char:
            return None
        code = ord(char)
        if 0xAC00 <= code <= 0xD7A3:
            offset = code - 0xAC00
            initial_idx = offset // 588
            if initial_idx == 12:
                return None
            return "has_initial"
        if 0x3131 <= code <= 0x318E:
            if 0x3131 <= code <= 0x3164:
                return "has_initial"
            else:
                return None
        if 0x1100 <= code <= 0x11FF:
            if 0x1100 <= code <= 0x1114:
                return "has_initial"
            else:
                return None
        return None

    def _decompose_korean_syllable_with_onset(
        self,
        word_start: int,
        word_end: int,
        korean_text: str,
        phone_items: Optional[List[Tuple[int, int, str]]] = None,
    ) -> List[Tuple[int, int, str]]:
        try:
            import jamo
            entries: List[Tuple[int, int, str]] = []
            if not korean_text:
                return entries
            korean_chars: List[Tuple[str, bool]] = []
            for char in korean_text:
                if not char or char == ' ':
                    continue
                has_initial = False
                try:
                    initial_marker = self._get_korean_initial_consonant(char)
                    if initial_marker == "has_initial":
                        has_initial = True
                except Exception as jamo_err:
                    logger.debug(f"Jamo 初声检测失败 {char}: {jamo_err}")
                korean_chars.append((char, has_initial))
            if not korean_chars:
                return entries
            n_chars = len(korean_chars)
            syllable_time_ranges: List[Tuple[int, int]] = []
            if phone_items:
                word_phones = [
                    (s, e, p) for s, e, p in phone_items
                    if s >= word_start and s < word_end
                    and p not in self.SIL_PHONES
                    and p not in self.IGNORE_PHONES
                ]
                word_phones.sort(key=lambda x: x[0])
                if len(word_phones) == n_chars:
                    syllable_time_ranges = [(s, e) for s, e, _ in word_phones]
                elif len(word_phones) > n_chars:
                    jamo_counts = []
                    for char, _ in korean_chars:
                        try:
                            dec = jamo.decompose(char)
                            jamo_counts.append(max(len(dec), 1))
                        except Exception:
                            jamo_counts.append(2)
                    idx = 0
                    for i, (char, _) in enumerate(korean_chars):
                        n = jamo_counts[i]
                        chunk = word_phones[idx: idx + n]
                        if chunk:
                            syl_start = chunk[0][0]
                            syl_end = (
                                word_end if i == n_chars - 1
                                else (chunk[-1][1] if len(chunk) == n
                                      else word_phones[min(idx + n, len(word_phones) - 1)][0])
                            )
                            syllable_time_ranges.append((syl_start, syl_end))
                        idx += n
            if len(syllable_time_ranges) != n_chars:
                word_duration = word_end - word_start
                total_units = sum(2 if hi else 1 for _, hi in korean_chars)
                unit_dur = max(word_duration // max(total_units, 1), 100000)
                cur = word_start
                syllable_time_ranges = []
                for i, (char, has_init) in enumerate(korean_chars):
                    units = 2 if has_init else 1
                    syl_end = (
                        word_end if i == n_chars - 1
                        else min(cur + units * unit_dur, word_end)
                    )
                    syllable_time_ranges.append((cur, syl_end))
                    cur = syl_end
            for i, ((char, has_initial), (syl_start, syl_end)) in enumerate(
                zip(korean_chars, syllable_time_ranges)
            ):
                syl_dur = max(syl_end - syl_start, 200000)
                if has_initial:
                    dash_dur = max(syl_dur // 3, 60000)
                    dash_end = min(syl_start + dash_dur, syl_end - 60000)
                    entries.append((syl_start, dash_end, "-"))
                    entries.append((dash_end, syl_end, char))
                else:
                    entries.append((syl_start, syl_end, char))
            if entries and entries[-1][1] < word_end:
                s, e, p = entries[-1]
                entries[-1] = (s, word_end, p)
            return entries
        except ImportError:
            logger.error("jamo 库未安装，无法分解韩文字符")
            return []
        except Exception as e:
            logger.error(f"韩文分解失败: {e}", exc_info=True)
            return []

    def _process_yue_words(
        self,
        word_tier,
        phone_items: List[Tuple[int, int, str]],
        text: str
    ) -> List[str]:
        target_syls = self._normalize_jyutping(self._text_to_jyutping(text))
        lines: List[str] = []
        syl_index = 0
        for interval in word_tier:
            mark = getattr(interval, "mark", getattr(interval, "text", ""))
            mark = (mark or "").strip()
            start = int(interval.minTime * 10000000)
            end = int(interval.maxTime * 10000000)
            if not mark or mark in self.IGNORE_PHONES:
                continue
            if mark in self.SIL_PHONES or mark in ("sp", "spn"):
                lines.append(f"{start} {end} sil")
                continue
            if self._is_english_word(mark):
                entries = self._get_arpabet_entries(start, end, phone_items)
                if entries:
                    for s, e, p in entries:
                        lines.append(f"{s} {e} {p}")
                else:
                    lines.append(f"{start} {end} {mark.lower()}")
                continue
            if self._is_digit_char(mark):
                syl = self.DIGIT_JYUTPING.get(mark.strip(), mark)
                for es, ee, el in self._make_con_entries(start, end, syl, phone_items, 'yue'):
                    lines.append(f"{es} {ee} {el}")
                syl_index = min(syl_index + 1, len(target_syls))
                continue
            clean_mark = re.sub(r"[^\u4e00-\u9fa5\u3400-\u4dbf\uf900-\ufaff\u3000-\u303f]+", "", mark)
            syl_count = len(clean_mark)
            if syl_count == 0:
                word_phones = self._get_phones_for_word(start, end, phone_items)
                for s, e, p in word_phones:
                    if p not in self.SIL_PHONES:
                        lines.append(f"{s} {e} {p}")
                continue
            current_syls = target_syls[syl_index: syl_index + syl_count]
            syl_index += syl_count
            if not current_syls:
                word_phones = self._get_phones_for_word(start, end, phone_items)
                for s, e, p in word_phones:
                    if p not in self.SIL_PHONES:
                        lines.append(f"{s} {e} {p}")
                continue
            syl_lines = self._distribute_syllables_in_word(start, end, current_syls, None, 'yue')
            for s, e, syl in syl_lines:
                for es, ee, el in self._make_con_entries(s, e, syl, phone_items, 'yue'):
                    lines.append(f"{es} {ee} {el}")
        return lines

    # =====================================================================
    # TextGrid 处理
    # =====================================================================
    def _textgrid_to_lab_word_tier_primary(
        self,
        textgrid_path: str,
        text: str,
        lang: str = 'zh'
    ) -> str:
        """Word Tier 对齐主逻辑（含 LAB 后处理）"""
        try:
            from textgrid import TextGrid
            tg = TextGrid.fromFile(textgrid_path)
            word_tier = None
            phone_tier = None
            for candidate in tg:
                tier_name = getattr(candidate, "name", "").lower()
                if "word" in tier_name:
                    word_tier = candidate
                elif "phone" in tier_name or "phoneme" in tier_name:
                    phone_tier = candidate
            if word_tier is None:
                logger.error("未找到 Word Tier")
                return ""
            phone_items: List[Tuple[int, int, str]] = []
            if phone_tier is not None:
                phone_items = self._extract_phone_items(phone_tier)
            if lang in ('zh', 'cmn'):
                lines = self._process_zh_words(word_tier, phone_items, text)
            elif lang == 'en':
                lines = self._process_en_words(word_tier, phone_items, text)
            elif lang == 'ja':
                lines = self._process_ja_words(word_tier, phone_items, text)
            elif lang == 'ko':
                lines = self._process_ko_words(word_tier, phone_items, text)
            elif lang == 'yue':
                lines = self._process_yue_words(word_tier, phone_items, text)
            else:
                lines = self._process_en_words(word_tier, phone_items, text)
            lines = self._apply_lab_postprocess(lines, lang)
            return "\n".join(lines)
        except ImportError:
            return self._parse_textgrid_manual(textgrid_path, text, lang)
        except Exception as e:
            logger.error(f"对齐失败: {e}", exc_info=True)
            return ""

    # =====================================================================
    # LAB 后处理辅助方法
    # =====================================================================
    @staticmethod
    def _parse_lab_lines(lines: List[str]) -> List[Tuple[int, int, str]]:
        entries: List[Tuple[int, int, str]] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                entries.append((int(parts[0]), int(parts[1]), parts[2]))
            except (ValueError, IndexError):
                continue
        return entries

    def _apply_lab_postprocess(
        self,
        lines: List[str],
        lang: str,
    ) -> List[str]:
        """
        LAB 后处理流水线：
          • 日语 (ja)  ： romaji 序列 → hiragana + '-' (build_ja_hiragana_lab)
                           → 合并 '-' 标记 (merge_lab_silence)
          • 中/英/韩/粤 ： 直接合并 '-' 标记 (merge_lab_silence)
        """
        entries = self._parse_lab_lines(lines)
        if lang in ('ja', 'jpn'):
            entries = build_ja_hiragana_lab(entries)
        merged = merge_lab_silence(entries)
        return [f"{s} {e} {p}" for s, e, p in merged]

    def _parse_textgrid_manual(
        self,
        textgrid_path: str,
        text: str,
        lang: str = 'zh'
    ) -> str:
        try:
            with open(textgrid_path, "r", encoding="utf-8") as f:
                content = f.read()
            tiers = content.split("item [")
            word_tier_content = ""
            phone_tier_content = ""
            for tier in tiers:
                if 'name = "words"' in tier or 'name = "word"' in tier:
                    word_tier_content = tier
                elif 'name = "phones"' in tier or 'name = "phone"' in tier:
                    phone_tier_content = tier
            if not word_tier_content:
                logger.error("未找到 Word Tier")
                return ""
            pattern = r"xmin = ([\d.]+)\s+xmax = ([\d.]+)\s+text = \"(.*?)\""
            matches = re.findall(pattern, word_tier_content, re.DOTALL)

            class MockInterval:
                def __init__(self, start, end, mark):
                    self.minTime = float(start)
                    self.maxTime = float(end)
                    self.mark = mark

            word_tier = [MockInterval(m[0], m[1], m[2]) for m in matches]
            phone_items: List[Tuple[int, int, str]] = []
            if phone_tier_content:
                p_matches = re.findall(pattern, phone_tier_content, re.DOTALL)
                phone_tier = [MockInterval(m[0], m[1], m[2]) for m in p_matches]
                phone_items = self._extract_phone_items(phone_tier)
            if lang in ('zh', 'cmn'):
                lines = self._process_zh_words(word_tier, phone_items, text)
            elif lang == 'en':
                lines = self._process_en_words(word_tier, phone_items, text)
            elif lang == 'ja':
                lines = self._process_ja_words(word_tier, phone_items, text)
            elif lang == 'ko':
                lines = self._process_ko_words(word_tier, phone_items, text)
            elif lang == 'yue':
                lines = self._process_yue_words(word_tier, phone_items, text)
            else:
                lines = self._process_en_words(word_tier, phone_items, text)
            lines = self._apply_lab_postprocess(lines, lang)
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"手工解析失败: {e}", exc_info=True)
            return ""

    def _textgrid_to_lab(self, textgrid_path: str, text: str, lang: str = 'zh') -> str:
        return self._textgrid_to_lab_word_tier_primary(textgrid_path, text, lang)

    def _get_audio_duration(self, audio_path: str) -> int:
        try:
            import soundfile as sf
            data, sr = sf.read(audio_path)
            duration_seconds = len(data) / sr
            return int(duration_seconds * 10000000)
        except (ImportError, Exception):
            try:
                import wave
                with wave.open(audio_path, "rb") as wav_file:
                    n_frames = wav_file.getnframes()
                    framerate = wav_file.getframerate()
                    duration_seconds = n_frames / framerate
                    return int(duration_seconds * 10000000)
            except Exception as e:
                logger.error(f"获取音频时长失败: {e}")
                return 0

    # =====================================================================
    # ★ 新增：长音频 RMS/百分位静音分割
    # =====================================================================

    def _compute_rms_frames(
        self,
        audio_data: "np.ndarray",
        sr: int,
    ) -> Tuple["np.ndarray", "np.ndarray"]:
        """
        在滑窗上计算 RMS 能量帧。
        窗宽 / 步长由类常量 RMS_WINDOW_MS / RMS_HOP_MS 决定。
        返回 (rms_array [float32], frame_times_seconds [float32])。
        """
        import numpy as np

        win = max(1, int(sr * self.RMS_WINDOW_MS / 1000))
        hop = max(1, int(sr * self.RMS_HOP_MS / 1000))
        n = max(1, (len(audio_data) - win) // hop + 1)

        rms = np.zeros(n, dtype=np.float32)
        for i in range(n):
            chunk = audio_data[i * hop: i * hop + win]
            if len(chunk) > 0:
                rms[i] = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))

        times = np.arange(n, dtype=np.float32) * hop / sr
        return rms, times

    def _find_silence_split_points(
        self,
        rms: "np.ndarray",
        times: "np.ndarray",
        total_sec: float,
        max_segment_sec: float,
    ) -> List[float]:
        """
        用 RMS 百分位阈值检测静音，返回分割时刻列表（含 0.0 和 total_sec）。

        算法：
        1. 以 RMS_PERCENTILE 百分位（仅统计非零帧）为静音阈值。
        2. 连续静音超过 MIN_SILENCE_MS 的区间取中点作为候选分割点。
        3. 贪心选取候选点，使相邻分割点距离 ≥ max_segment_sec × 0.6。
        4. 若仍存在超长片段，强制等分补充分割点。
        """
        import numpy as np

        nonzero = rms[rms > 1e-10]
        if len(nonzero) < 10:
            logger.warning("音频几乎全为静音，跳过分割")
            return [0.0, total_sec]

        threshold = float(np.percentile(nonzero, self.RMS_PERCENTILE))
        is_silent = (rms <= threshold)
        min_frames = max(2, int(self.MIN_SILENCE_MS / self.RMS_HOP_MS))

        # 收集静音区间中点
        candidates: List[float] = []
        run_start: Optional[int] = None
        for i, silent in enumerate(is_silent):
            if silent and run_start is None:
                run_start = i
            elif not silent and run_start is not None:
                run_len = i - run_start
                if run_len >= min_frames:
                    mid_idx = (run_start + i) // 2
                    candidates.append(float(times[min(mid_idx, len(times) - 1)]))
                run_start = None
        if run_start is not None:
            run_len = len(is_silent) - run_start
            if run_len >= min_frames:
                mid_idx = (run_start + len(is_silent)) // 2
                candidates.append(float(times[min(mid_idx, len(times) - 1)]))

        # 贪心选取：间距至少 max_segment_sec × 0.6
        min_gap = max_segment_sec * 0.6
        chosen: List[float] = []
        last = 0.0
        for cand in candidates:
            if cand - last >= min_gap:
                chosen.append(cand)
                last = cand

        # 构造最终分割点列表，强制补充超长片段的等分点
        split_points: List[float] = [0.0]
        for sp in chosen:
            # 若上一个点到 sp 之间已经超过 max_segment_sec，先等分填充
            while sp - split_points[-1] > max_segment_sec:
                split_points.append(split_points[-1] + max_segment_sec)
            split_points.append(sp)

        # 处理尾段
        while total_sec - split_points[-1] > max_segment_sec:
            split_points.append(split_points[-1] + max_segment_sec)
        split_points.append(total_sec)

        return split_points

    def _split_audio_by_rms_silence(
        self,
        audio_path: str,
        max_segment_sec: Optional[float] = None,
    ) -> List[Tuple[str, float, float]]:
        """
        在 temp_dir 中以 RMS/百分位静音检测对音频文件预分割。

        返回 [(segment_wav_path, start_sec, end_sec), ...]。
        如果无需分割（时长 ≤ LONG_AUDIO_THRESHOLD_SEC 或库不可用），
        返回 [(audio_path, 0.0, total_sec)]。
        """
        if max_segment_sec is None:
            max_segment_sec = self.MAX_SEGMENT_SEC

        try:
            import numpy as np
            import soundfile as sf
        except ImportError as e:
            logger.warning(f"soundfile/numpy 不可用，跳过预分割: {e}")
            return [(audio_path, 0.0, self._get_audio_duration(audio_path) / 10_000_000)]

        try:
            data, sr = sf.read(audio_path, dtype="float32")
        except Exception as e:
            logger.error(f"读取音频失败: {e}")
            return [(audio_path, 0.0, self._get_audio_duration(audio_path) / 10_000_000)]

        # 多声道 → 单声道
        if data.ndim > 1:
            data = data.mean(axis=1)

        total_sec = len(data) / sr
        logger.info(f"音频时长 {total_sec:.1f}s，开始 RMS 静音分割（最大分段={max_segment_sec}s）")

        rms, times = self._compute_rms_frames(data, sr)
        split_points = self._find_silence_split_points(rms, times, total_sec, max_segment_sec)

        if len(split_points) <= 2:
            # 只有 [0, total]，无有效切割点
            logger.info("未找到足够静音点，保持单一片段")
            return [(audio_path, 0.0, total_sec)]

        seg_dir = self.temp_dir or os.path.dirname(audio_path)
        base = Path(audio_path).stem
        segments: List[Tuple[str, float, float]] = []

        for i in range(len(split_points) - 1):
            seg_start = split_points[i]
            seg_end = split_points[i + 1]
            seg_dur = seg_end - seg_start

            if seg_dur < 0.5:
                logger.debug(f"  片段 {i} 过短 ({seg_dur:.3f}s)，跳过")
                continue

            start_sample = int(seg_start * sr)
            end_sample = min(int(seg_end * sr), len(data))
            seg_data = data[start_sample:end_sample]

            seg_path = os.path.join(seg_dir, f"{base}_seg{i:03d}.wav")
            try:
                sf.write(seg_path, seg_data, sr)
            except Exception as e:
                logger.error(f"  写分段 WAV 失败: {e}")
                continue

            segments.append((seg_path, seg_start, seg_end))
            logger.info(f"  片段 {i}: [{seg_start:.2f}s, {seg_end:.2f}s] → {seg_path}")

        return segments if segments else [(audio_path, 0.0, total_sec)]

    # =====================================================================
    # ★ 新增：文本按分段时长比例切分
    # =====================================================================

    def _count_syllable_weight(self, text: str, lang: str) -> int:
        """
        估计文本的音节权重，用于按比例分配文本到各音频分段。
        CJK 语言：汉字数 + 英文词数。英语：英文词数。韩语：音节字符数。
        """
        if lang in ('zh', 'cmn', 'yue', 'ja', 'jpn'):
            cjk = sum(
                1 for c in text
                if '\u3000' <= c <= '\u9fff'
                or '\uf900' <= c <= '\ufaff'
                or '\u3040' <= c <= '\u30ff'
            )
            en_words = len(re.findall(r'[a-zA-Z]+', text))
            return max(cjk + en_words, 1)
        elif lang in ('ko', 'kor'):
            ko_chars = sum(1 for c in text if '\uac00' <= c <= '\ud7a3')
            return max(ko_chars, 1)
        else:  # en
            return max(len(re.findall(r'[a-zA-Z]+', text)), 1)

    def _split_text_at_clause_boundaries(self, text: str, lang: str) -> List[str]:
        """
        在句/短语边界处分割文本，返回子句列表（每个子句保留末尾标点）。

        CJK：
          • 一级分割：在 。！？；\n 处（完整句子）
          • 二级分割：若某子句超过 15 字，再在 ，、 处进一步切分
            （中文长句往往只有逗号，不进行二级分割会导致一个子句横跨多个音频分段，
              进而造成文本-音频不匹配，触发 MFA NoAlignmentsError）
        英语：在 .!? 后接空格处分割。
        """
        text = text.strip()
        if not text:
            return []

        if lang in ('zh', 'cmn', 'yue', 'ja', 'jpn', 'ko', 'kor'):
            # 一级分割：句末标点
            primary = re.split(r'(?<=[。！？；\n])', text)
            clauses: List[str] = []
            for part in primary:
                part = part.strip()
                if not part:
                    continue
                # 二级分割：长句在逗号/顿号处继续拆分（提升文本-时间对齐精度）
                if len(part) > 15:
                    sub_parts = re.split(r'(?<=[，、])', part)
                    clauses.extend([s.strip() for s in sub_parts if s.strip()])
                else:
                    clauses.append(part)
        else:
            parts = re.split(r'(?<=[.!?])\s+', text)
            clauses = [p.strip() for p in parts if p.strip()]

        return clauses

    def _split_text_for_segments(
        self,
        text: str,
        lang: str,
        segment_durations: List[float],
    ) -> List[str]:
        """
        将文本分配到各音频分段。

        策略：
        1. 先按句/短语边界拆分文本为子句。
        2. 按各分段时长占总时长的比例，计算每个分段的目标音节权重。
        3. 贪心地将子句打包到对应分段，使实际权重尽量接近目标值。

        若子句数量不足以覆盖所有分段，则对最长子句按字符比例强制截断补充。
        最后一个分段获取所有剩余文本。
        """
        n_segs = len(segment_durations)

        if n_segs <= 1 or not text.strip():
            return [text] * n_segs

        total_dur = sum(segment_durations)
        if total_dur <= 0:
            return [text] + [""] * (n_segs - 1)

        clauses = self._split_text_at_clause_boundaries(text, lang)
        if not clauses:
            return [text] + [""] * (n_segs - 1)

        # 每个子句的音节权重
        clause_weights = [self._count_syllable_weight(c, lang) for c in clauses]
        total_weight = sum(clause_weights) or 1

        # 每个分段的目标音节权重
        target_weights = [
            total_weight * (d / total_dur) for d in segment_durations
        ]

        seg_texts: List[str] = [""] * n_segs
        clause_idx = 0

        for seg_i in range(n_segs):
            if clause_idx >= len(clauses):
                break  # 文本已分配完毕
            if seg_i == n_segs - 1:
                # 最后一个分段：消费全部剩余子句
                seg_texts[seg_i] = "".join(clauses[clause_idx:])
                clause_idx = len(clauses)
                break

            target = target_weights[seg_i]
            accumulated = 0.0
            while clause_idx < len(clauses):
                w = clause_weights[clause_idx]
                # 若还没有内容，或加入后不超过目标的 130%，则纳入
                if accumulated == 0 or accumulated + w <= target * 1.3:
                    seg_texts[seg_i] += clauses[clause_idx]
                    accumulated += w
                    clause_idx += 1
                else:
                    break

        # 保险：若有剩余子句（贪心提前结束），追加到最后一个分段
        if clause_idx < len(clauses):
            seg_texts[-1] += "".join(clauses[clause_idx:])

        # 确保所有分段非空（若某分段为空，向相邻分段借一个子句）
        for i in range(n_segs):
            if not seg_texts[i].strip() and i < n_segs - 1 and seg_texts[i + 1].strip():
                # 从下一分段借第一个字符（最小单位）
                borrow = seg_texts[i + 1][:1]
                seg_texts[i] = borrow
                seg_texts[i + 1] = seg_texts[i + 1][1:]

        return seg_texts

    # =====================================================================
    # ★ 新增：提取 MFA 对齐逻辑为独立方法，支持复用
    # =====================================================================

    def _prepare_text_for_mfa(self, text: str, lang: str) -> str:
        """
        送入 MFA 语料库前对文本做最小化清洗：
        • 去除标点符号（MFA 字典中缺少的标点会导致 NoAlignmentsError）
        • 保留语言本身的字符集和空格
        • 折叠连续空白

        注意：不改变字符顺序，保证与 TextGrid 对齐结果的对应关系。
        """
        if not text:
            return ""

        if lang in ('zh', 'cmn'):
            # 保留：CJK 基本/扩展，拉丁字母，数字，空格
            clean = re.sub(
                r'[^\u4e00-\u9fa5\u3400-\u4dbf\uf900-\ufaff a-zA-Z0-9]', ' ', text
            )
        elif lang == 'yue':
            clean = re.sub(
                r'[^\u4e00-\u9fa5\u3400-\u4dbf\uf900-\ufaff a-zA-Z0-9]', ' ', text
            )
        elif lang in ('ja', 'jpn'):
            # 保留：平假名、片假名、CJK、拉丁、数字
            clean = re.sub(
                r'[^\u3040-\u30ff\u4e00-\u9fa5\uff66-\uff9f a-zA-Z0-9]', ' ', text
            )
        elif lang in ('ko', 'kor'):
            # 保留：韩文音节、Jamo 字母、拉丁、数字
            clean = re.sub(
                r'[^\uac00-\ud7a3\u3130-\u318f\u1100-\u11ff a-zA-Z0-9]', ' ', text
            )
        else:
            # 英语：保留字母、数字、撇号
            clean = re.sub(r"[^a-zA-Z0-9'\- ]", ' ', text)

        clean = re.sub(r'\s+', ' ', clean).strip()
        logger.debug(f"[_prepare_text_for_mfa] lang={lang} 清洗后长度: {len(clean)}")
        return clean

    def _run_mfa_align(
        self,
        corpus_dir: str,
        output_dir: str,
        lang: str,
        timeout_seconds: int = 300,
        mfa_temp_dir: Optional[str] = None,
        beam: int = 100,
        retry_beam: int = 400,
    ) -> Optional[str]:
        """
        在 corpus_dir 中执行 MFA align，输出到 output_dir。
        成功返回 TextGrid 路径；失败返回 None。

        mfa_temp_dir：为此次对齐提供专属的 MFA 内部工作目录。
        ★ 关键：MFA 3.x 默认把所有运行的 SQLite 数据库、模型缓存等都写入
          ~/Documents/MFA/，多次顺序调用时会互相踩踏导致 code=1 失败。
          通过 --temp_directory 让每次对齐使用独立目录可彻底避免此问题。

        beam / retry_beam：MFA 默认 beam=10, retry_beam=40，对于困难语音
          （带噪、停顿、长句）容易触发 NoAlignmentsError。
          推荐值：beam=100, retry_beam=400（MFA 官方建议）。
          严重困难时可升至 beam=200, retry_beam=800。
        """
        models = MFAChecker.LANGUAGE_MODELS.get(lang, {})
        dict_model = models.get("dictionary", lang)
        acoustic_model = models.get("acoustic", lang)
        py = MFAChecker.env_python()

        cmd = [
            str(py), "-m", "montreal_forced_aligner.command_line.mfa",
            "align", corpus_dir, dict_model, acoustic_model,
            output_dir, "--clean", "--single_speaker",
            # ★ 修复 NoAlignmentsError：扩大搜索束（MFA 官方建议值）
            "--beam", str(beam), "--retry_beam", str(retry_beam),
        ]

        # ★ 核心修复：每次对齐使用独立的 MFA 临时目录，避免多段顺序对齐时
        #   共享 ~/Documents/MFA/ 导致 SQLite 数据库/模型缓存冲突（code=1）
        if mfa_temp_dir is not None:
            os.makedirs(mfa_temp_dir, exist_ok=True)
            cmd += ["--temp_directory", mfa_temp_dir]
            logger.debug(f"MFA temp_directory: {mfa_temp_dir}")

        env = os.environ.copy()
        mfa_env_dir = MFAChecker.env_dir()
        env["CONDA_PREFIX"] = str(mfa_env_dir)
        lib_bin = mfa_env_dir / "Library" / "bin"
        if lib_bin.exists():
            env["PATH"] = str(lib_bin) + os.pathsep + env.get("PATH", "")
        lib_dir = mfa_env_dir / "lib"
        if lib_dir.exists():
            env["LD_LIBRARY_PATH"] = str(lib_dir) + os.pathsep + env.get("LD_LIBRARY_PATH", "")

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=timeout_seconds, env=env
            )
        except subprocess.TimeoutExpired:
            logger.error(f"MFA align 超时 ({timeout_seconds}s)")
            return None
        except Exception as e:
            logger.error(f"MFA align 异常: {e}")
            return None

        if result.returncode != 0:
            # 记录完整 stderr + stdout 以便诊断（不截断）
            full_diag = ((result.stderr or "") + (result.stdout or "")).strip()
            logger.error(
                f"MFA align 失败 (code={result.returncode}):\n"
                f"{'=' * 60}\n{full_diag}\n{'=' * 60}"
            )
            return None

        textgrid_files = list(Path(output_dir).glob("**/*.TextGrid"))
        if not textgrid_files:
            logger.error("MFA 未生成 TextGrid")
            return None

        return str(textgrid_files[0])

    # =====================================================================
    # ★ 新增：长音频分段处理流程
    # =====================================================================

    @staticmethod
    def _merge_segment_lab_entries(
        all_entries: List[Tuple[int, int, str]]
    ) -> List[Tuple[int, int, str]]:
        """
        合并各分段 LAB 条目：
        1. 按起始时间排序。
        2. 填补分段边界处的微小时间空隙（< 50 ms = 500000 单位）。
        3. 合并相邻的连续 sil 条目（避免多余的静音段叠加）。
        """
        if not all_entries:
            return []

        all_entries.sort(key=lambda x: x[0])
        result: List[Tuple[int, int, str]] = []

        for s, e, p in all_entries:
            if not result:
                result.append((s, e, p))
                continue

            prev_s, prev_e, prev_p = result[-1]

            # 填补微小空隙（< 50ms）
            gap = s - prev_e
            if 0 < gap < 500_000:
                result[-1] = (prev_s, s, prev_p)
                prev_e = s

            # 合并连续静音
            if p == 'sil' and prev_p == 'sil':
                result[-1] = (prev_s, max(prev_e, e), 'sil')
            else:
                result.append((s, e, p))

        return result

    def _process_with_segmentation(
        self,
        audio_path: str,
        text: str,
        lang: str,
        text_for_mfa: str,
        phoneme_text: str,
        start_time: float,
    ) -> Dict:
        """
        长音频处理路径：
        ① RMS 静音检测 → 切分为若干 ≤ MAX_SEGMENT_SEC 的 WAV 片段
        ② 按时长比例将文本分配到各片段
        ③ 对每个片段独立运行 MFA align → 解析 TextGrid → 转 LAB
        ④ 将各片段 LAB 时间戳加上片段起始偏移量后合并
        """
        logger.info(f"[长音频模式] 启动分段对齐 (阈值={self.LONG_AUDIO_THRESHOLD_SEC}s)")

        # ① 切分音频
        segments = self._split_audio_by_rms_silence(audio_path)

        if len(segments) <= 1:
            logger.info("分割结果为单一片段，回退到直接处理模式")
            return self._process_direct(
                audio_path, text, lang, text_for_mfa, phoneme_text, start_time
            )

        logger.info(f"共 {len(segments)} 个片段")

        # ② 分配文本
        durations = [end - start for _, start, end in segments]
        seg_texts = self._split_text_for_segments(text_for_mfa, lang, durations)

        logger.info("文本分配结果：")
        for i, (seg_text, (_, t0, t1)) in enumerate(zip(seg_texts, segments)):
            preview = seg_text[:40].replace('\n', ' ')
            logger.info(f"  片段{i+1} [{t0:.2f}s-{t1:.2f}s] text='{preview}...'")

        # ③ 逐片段 MFA 对齐
        all_entries: List[Tuple[int, int, str]] = []
        success_count = 0

        for i, ((seg_path, seg_start_sec, seg_end_sec), seg_text) in enumerate(
            zip(segments, seg_texts)
        ):
            seg_dur = seg_end_sec - seg_start_sec
            if not seg_text.strip():
                logger.warning(f"  片段{i+1} 文本为空，跳过")
                continue

            logger.info(f"  ▶ 片段{i+1}/{len(segments)} [{seg_start_sec:.2f}s-{seg_end_sec:.2f}s]")

            seg_corpus = os.path.join(self.temp_dir, f"corpus_seg{i:03d}")
            seg_output = os.path.join(self.temp_dir, f"aligned_seg{i:03d}")
            os.makedirs(seg_corpus, exist_ok=True)
            os.makedirs(seg_output, exist_ok=True)

            seg_basename = f"seg{i:03d}"
            seg_wav_dest = os.path.join(seg_corpus, f"{seg_basename}.wav")
            try:
                shutil.copy2(seg_path, seg_wav_dest)
            except Exception as e:
                logger.error(f"  复制分段 WAV 失败: {e}")
                continue

            seg_txt_path = os.path.join(seg_corpus, f"{seg_basename}.txt")
            # ★ 清洗后写入：去除标点，防止 MFA 字典未收录的标点触发 NoAlignmentsError
            cleaned_seg_text = self._prepare_text_for_mfa(seg_text, lang)
            if not cleaned_seg_text:
                logger.warning(f"  片段{i+1} 清洗后文本为空，跳过")
                continue
            with open(seg_txt_path, "w", encoding="utf-8") as f:
                f.write(cleaned_seg_text)

            # 超时时间：按片段时长动态计算，最少 90s
            seg_timeout = max(90, int(seg_dur * 25))
            if lang in ('ja', 'ko'):
                seg_timeout = max(120, int(seg_dur * 40))

            # ★ 每段使用独立的 MFA 内部工作目录，彻底避免多段顺序对齐时
            #   共享 ~/Documents/MFA/ 导致 SQLite 数据库/模型缓存冲突（code=1）
            seg_mfa_temp = os.path.join(self.temp_dir, f"mfa_work_{i:03d}")
            tg_path = self._run_mfa_align(
                seg_corpus, seg_output, lang, seg_timeout, seg_mfa_temp,
                beam=100, retry_beam=400,
            )

            # ★ 失败时自动用超大束重试一次（beam=200 / retry_beam=800）
            if tg_path is None:
                logger.warning(f"  片段{i+1} 首次对齐失败，使用超大束 (beam=200) 重试...")
                seg_output_retry = os.path.join(self.temp_dir, f"aligned_seg{i:03d}_retry")
                seg_mfa_temp_retry = os.path.join(self.temp_dir, f"mfa_work_{i:03d}_retry")
                os.makedirs(seg_output_retry, exist_ok=True)
                tg_path = self._run_mfa_align(
                    seg_corpus, seg_output_retry, lang,
                    seg_timeout + 60, seg_mfa_temp_retry,
                    beam=200, retry_beam=800,
                )

            if tg_path is None:
                logger.warning(f"  片段{i+1} 对齐失败（两次尝试均失败），跳过")
                continue

            # 解析 TextGrid → LAB 行
            seg_lab = self._textgrid_to_lab(tg_path, seg_text, lang=lang)
            if not seg_lab:
                logger.warning(f"  片段{i+1} LAB 为空，跳过")
                continue

            # ④ 应用时间偏移
            offset_100ns = int(seg_start_sec * 10_000_000)
            for line in seg_lab.splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        entry_start = int(parts[0]) + offset_100ns
                        entry_end = int(parts[1]) + offset_100ns
                        label = parts[2]
                        all_entries.append((entry_start, entry_end, label))
                    except ValueError:
                        continue

            success_count += 1

        if not all_entries:
            logger.error("所有片段对齐均失败，回退到直接处理模式")
            return self._process_direct(
                audio_path, text, lang, text_for_mfa, phoneme_text, start_time
            )

        # 合并，整理边界
        merged_entries = self._merge_segment_lab_entries(all_entries)
        lab_content = "\n".join(f"{s} {e} {p}" for s, e, p in merged_entries)

        processing_time = int((time.time() - start_time) * 1000)
        logger.info(
            f"[长音频模式] 完成：{success_count}/{len(segments)} 个片段成功，"
            f"共 {len(merged_entries)} 条 LAB 条目，耗时 {processing_time}ms"
        )

        return {
            "success": True,
            "raw_text": text,
            "phoneme_text": phoneme_text,
            "lab_content": lab_content,
            "audio_duration": self._get_audio_duration(audio_path),
            "processing_time": processing_time,
            "segments_count": len(segments),
            "segments_success": success_count,
        }

    def _process_direct(
        self,
        audio_path: str,
        text: str,
        lang: str,
        text_for_mfa: str,
        phoneme_text: str,
        start_time: float,
    ) -> Dict:
        """
        直接（不分段）处理路径：将整段音频送入 MFA。
        原 process() 方法的核心对齐逻辑已提取至此。
        """
        corpus_dir = os.path.join(self.temp_dir, "corpus")
        os.makedirs(corpus_dir, exist_ok=True)
        basename = Path(audio_path).stem
        audio_dest = os.path.join(corpus_dir, f"{basename}.wav")
        shutil.copy2(audio_path, audio_dest)
        text_file = os.path.join(corpus_dir, f"{basename}.txt")
        # ★ 清洗后写入：去除标点防止 MFA NoAlignmentsError
        cleaned_text_for_mfa = self._prepare_text_for_mfa(text_for_mfa, lang)
        with open(text_file, "w", encoding="utf-8") as f:
            f.write(cleaned_text_for_mfa or text_for_mfa)  # fallback 到原始文本

        output_dir = os.path.join(self.temp_dir, "aligned")
        os.makedirs(output_dir, exist_ok=True)

        # 超时：短音频 300s；日/韩 600s；超长音频按时长等比
        audio_duration_sec = self._get_audio_duration(audio_path) / 10_000_000
        timeout_seconds = max(300, int(audio_duration_sec * 15))
        if lang in ('ja', 'ko'):
            timeout_seconds = max(600, int(audio_duration_sec * 25))

        # 直接模式也使用独立 temp dir，防止与分段模式的遗留状态冲突
        direct_mfa_temp = os.path.join(self.temp_dir, "mfa_work_direct")
        tg_path = self._run_mfa_align(
            corpus_dir, output_dir, lang, timeout_seconds, direct_mfa_temp,
            beam=100, retry_beam=400,
        )

        if tg_path is None:
            return {
                "success": False, "error": "MFA align 失败或未生成 TextGrid",
                "processing_time": int((time.time() - start_time) * 1000)
            }

        lab_content = self._textgrid_to_lab(tg_path, text, lang=lang)
        if not lab_content:
            return {
                "success": False, "error": "LAB 内容为空",
                "processing_time": int((time.time() - start_time) * 1000)
            }

        return {
            "success": True,
            "raw_text": text,
            "phoneme_text": phoneme_text,
            "lab_content": lab_content,
            "textgrid_path": tg_path,
            "audio_duration": self._get_audio_duration(audio_path),
            "processing_time": int((time.time() - start_time) * 1000),
        }

    # =====================================================================
    # 主流程
    # =====================================================================
    def process(self, audio_file, text: str, language: str = "cmn") -> Dict:
        """
        处理单个音频文件。

        流程：
        1. 清洗文本标点
        2. 检查语言环境
        3. 保存上传音频
        4. 若音频 > LONG_AUDIO_THRESHOLD_SEC → 分段对齐（RMS 静音切割）
           否则 → 直接整段对齐
        """
        start_time = time.time()
        try:
            self.temp_dir = tempfile.mkdtemp(prefix="mfa_")
            raw_lang = (language or "cmn").lower().strip()

            # ★ 优先清洗输入文本（统一标点规范化）
            text = self._clean_input_text(text)

            if raw_lang in ("cmn", "zh", "zh-cn", "mandarin"):
                lang = "zh"
                text_for_mfa = text.strip()
                phoneme_text = self._text_to_pinyin_notone(text)
            elif raw_lang in ("yue", "zh-yue", "cantonese"):
                lang = "yue"
                text_for_mfa = text.strip()
                phoneme_text = self._text_to_jyutping(text) or text.strip()
            elif raw_lang in ("en", "english", "eng"):
                lang = "en"
                text_for_mfa = text.strip()
                phoneme_text = text.strip()
            elif raw_lang in ("ja", "japanese", "jpn"):
                lang = "ja"
                text_for_mfa = text.strip()
                phoneme_text = text.strip()
            elif raw_lang in ("ko", "korean", "kor"):
                lang = "ko"
                text_for_mfa = text.strip()
                phoneme_text = text.strip()
            else:
                lang = raw_lang
                text_for_mfa = text.strip()
                phoneme_text = text.strip()

            env_ok, env_msg = self._check_language_environment(lang)
            if not env_ok:
                logger.error(env_msg)
                return {
                    "success": False, "error": env_msg,
                    "processing_time": int((time.time() - start_time) * 1000)
                }

            # 保存上传的音频文件
            audio_path = os.path.join(self.temp_dir, audio_file.filename)
            audio_file.save(audio_path)

            audio_duration_100ns = self._get_audio_duration(audio_path)
            audio_duration_sec = audio_duration_100ns / 10_000_000

            logger.info(
                f"音频时长: {audio_duration_sec:.1f}s | 语言: {lang} | "
                f"长音频阈值: {self.LONG_AUDIO_THRESHOLD_SEC}s"
            )

            # ★ 根据音频时长选择处理路径
            if audio_duration_sec > self.LONG_AUDIO_THRESHOLD_SEC:
                return self._process_with_segmentation(
                    audio_path, text, lang, text_for_mfa, phoneme_text, start_time
                )
            else:
                return self._process_direct(
                    audio_path, text, lang, text_for_mfa, phoneme_text, start_time
                )

        except subprocess.TimeoutExpired:
            return {
                "success": False, "error": "MFA 超时",
                "processing_time": int((time.time() - start_time) * 1000)
            }
        except FileNotFoundError:
            return {
                "success": False, "error": "MFA 命令不存在",
                "processing_time": int((time.time() - start_time) * 1000)
            }
        except Exception as e:
            logger.error(f"处理错误: {e}", exc_info=True)
            return {
                "success": False, "error": str(e),
                "processing_time": int((time.time() - start_time) * 1000)
            }
        finally:
            self._cleanup()

    def _cleanup(self):
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except Exception as e:
                logger.warning(f"清理失败: {e}")
