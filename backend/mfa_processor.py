# -*- coding: utf-8 -*-
"""
MFA 处理核心模块 - 多语言增强版 v9.3
特点：
1. 多语言支持：中文、英语、日语、粤语、韩语
2. IPA→Romaji/ARPABET/Hangul Jamo 自动转换
3. Word Tier 优先对齐
4. Phone Tier 精确phoneme处理
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
from typing import Dict, List, Tuple, Optional, Set

from pypinyin import lazy_pinyin, Style
from mfa_utils import MFAChecker
from phoneme_converter import convert_phoneme

logger = logging.getLogger(__name__)


class MFAProcessor:
    """Montreal Forced Aligner 处理器 - 多语言增强版 v9.3"""

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

    def __init__(self):
        self.temp_dir: Optional[str] = None

    # =====================================================================
    # 多语言环境检查
    # =====================================================================
    def _check_zh_environment(self) -> Tuple[bool, str]:
        """检查中文（普通话）环境"""
        try:
            from pypinyin import lazy_pinyin, Style
            lazy_pinyin("测试", style=Style.NORMAL)
            return True, "✓ 中文环境就绪（pypinyin已安装）"
        except ImportError:
            return False, "❌ 缺失中文支持：pip install pypinyin"
        except Exception as e:
            return False, f"❌ 中文��境错误：{str(e)}"

    def _check_en_environment(self) -> Tuple[bool, str]:
        """检查英语环境"""
        try:
            mfa_ok, mfa_msg = MFAChecker.check_mfa_installed()
            if mfa_ok:
                return True, "✓ 英语环境就绪（MFA已安装）"
            else:
                return False, f"❌ MFA未安装：{mfa_msg}"
        except Exception as e:
            return False, f"❌ 英语环境错误：{str(e)}"

    def _check_ja_environment(self) -> Tuple[bool, str]:
        """检查日语环境"""
        try:
            from sudachipy import Dictionary
            tokenizer = Dictionary().create()
            test_result = tokenizer.tokenize("テスト")
            if test_result and len(test_result) > 0:
                return True, "✓ 日语环境就绪（sudachipy + 字典已就绪）"
            else:
                return False, "❌ 日语字典初始化失败"
        except ImportError as e:
            return False, "❌ 缺失日语支持（sudachipy未安装）\n   请运行：pip install sudachipy sudachidict-core"
        except Exception as e:
            error_msg = str(e)
            if "dictionary" in error_msg.lower():
                return False, "❌ 日语字典资源不可用\n   请重新安装：pip install --force-reinstall sudachidict-core"
            else:
                return False, f"❌ 日语环境错误：{error_msg}"

    def _check_ko_environment(self) -> Tuple[bool, str]:
        """检查韩语环境"""
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
        """检查粤语环境"""
        try:
            import pycantonese
            return True, "✓ 粤语环境就绪（pycantonese已安装）"
        except ImportError:
            return False, "❌ 缺失粤语支持（pycantonese）\n   请运行：pip install pycantonese"
        except Exception as e:
            return False, f"❌ 粤语环境错误：{str(e)}"

    def _check_language_environment(self, lang: str) -> Tuple[bool, str]:
        """统一的语言环境检查"""
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
        """中文文本转无声调拼音"""
        phones = lazy_pinyin(text, style=Style.NORMAL, errors="ignore")
        phones = [p.strip().lower() for p in phones if p and p.strip()]
        return " ".join(phones)

    def _text_to_jyutping(self, text: str) -> str:
        """粤语文本转粤拼（无声调）"""
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
        """无声调拼音文本转音节列表"""
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
        """粤拼文本转音节列表"""
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
        """文本片段 -> 拼音音节列表"""
        return self._normalize_no_tone_pinyin(self._text_to_pinyin_notone(segment))

    def _segment_to_jyutping(self, segment: str) -> List[str]:
        """文本片段 -> 粤拼音节列表"""
        return self._normalize_jyutping(self._text_to_jyutping(segment))

    # =====================================================================
    # 语言/词语类型检测
    # =====================================================================
    def _is_english_word(self, word: str) -> bool:
        """检测是否为英语单词"""
        return bool(re.match(r"^[a-zA-Z''\-]+$", word.strip()))

    def _is_digit_char(self, word: str) -> bool:
        """检测是否为单个数字字符"""
        return word.strip() in '0123456789'

    # =====================================================================
    # Phone 清洗
    # =====================================================================
    def _clean_phone(self, phone: str) -> str:
        """去掉 IPA 声调符号"""
        if not phone:
            return ""
        phone = phone.strip()
        tone_marks = ["˥", "˧", "˩", "˨", "˦", "˧˥", "˥˩", "˩˧", "˨˩", "˥˧", "˧˨", "˨˧"]
        for mark in tone_marks:
            phone = phone.replace(mark, "")
        return phone.strip().lower()

    def _is_silence_phone(self, phone: str) -> bool:
        """判断是否为静音标记"""
        return (phone or "").strip().lower() in self.SIL_PHONES

    def _clean_phone_token(self, phone: str) -> str:
        """宽松的 phone 归一化"""
        phone = (phone or "").strip().lower().replace("ü", "v")
        phone = re.sub(r"[^a-zA-Zv]+", "", phone)
        return phone.strip()

    def _extract_phone_items(self, tier) -> List[Tuple[int, int, str]]:
        """从 TextGrid tier 中提取 phone items"""
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
        """提取属于某个 word 时间段内的 phone 条目"""
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
        """提取 ARPABET 音素条目"""
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
        """提取 ROMAJI 音素条目（IPA→Romaji 自动转换）"""
        word_phones = self._get_phones_for_word(word_start, word_end, phone_items)
        entries: List[Tuple[int, int, str]] = []
        
        for s, e, p in word_phones:
            # ★ 关键：使用 convert_phoneme 进行 IPA → Romaji 转换
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
        """提取 Hangul Jamo 音素条目（IPA→Jamo 自动转换）"""
        word_phones = self._get_phones_for_word(word_start, word_end, phone_items)
        entries: List[Tuple[int, int, str]] = []
        
        for s, e, p in word_phones:
            # ★ 关键：使用 convert_phoneme 进行 IPA → Hangul Jamo 转换
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
        """计算音节权重"""
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
        """拆分拼音音节为声母和韵母"""
        syl = (syl or "").strip().lower().replace("ü", "v")
        if not syl:
            return "", ""
        for ini in self.INITIALS_EXTENDED:
            if syl.startswith(ini):
                final = syl[len(ini):]
                return ini, final
        return "", syl

    def _get_syllable_nucleus(self, syl: str) -> str:
        """提取拼音音节的主元音"""
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
        """判断音节是否有辅音声母"""
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
        """获取 con 标记的边界"""
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
        fallback = syl_start + con_duration
        return fallback

    def _make_con_entries(
        self,
        syl_start: int,
        syl_end: int,
        syl: str,
        phone_items: List[Tuple[int, int, str]],
        lang: str = 'zh'
    ) -> List[Tuple[int, int, str]]:
        """生成单个音节的 LAB 条目"""
        if self._has_con_onset(syl, lang):
            con_boundary = self._get_con_boundary(syl_start, syl_end, phone_items)
            return [
                (syl_start, con_boundary, "-"),
                (con_boundary, syl_end, syl),
            ]
        else:
            return [(syl_start, syl_end, syl)]

    def _syllable_anchor_candidates(self, syl: str) -> List[str]:
        """生成音节的可匹配候选"""
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
        """判断 phone 是否匹配拼音音节"""
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
        """在 Word 内分配音节"""
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
        """基于权重的比例分配"""
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
        """处理中文 Word Tier → 拼音 + con 标记"""
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
        """处理英语 Word Tier → ARPABET"""
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
        """处理日语 Word Tier → ROMAJI（IPA→Romaji 自动转换）"""
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
        """处理韩语 Word Tier → 完整韩文字符 + "-" 初声标记

        输出格式：保持完整韩文字符，对有初声的字符添加 "-" 标注

        例如："도와드릴까요"
        17900000 18583333 -   (도的初声标记)
        18583333 19266666 도  (完整字)
        19266666 20050000 와  (无初声，直接字)
        20050000 20733333 -   (드的初声标记)
        20733333 21416666 드  (完整字)
        21416666 22100000 릴  (完整字)
        22100000 22783333 -   (까의初声标记)
        22783333 23466666 까  (完整字)
        23466666 24150000 요  (无初声，直接字)
        """
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

            # ★ 韩文处理：检查是否为韩文并分解
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
        """检测文本是否为韩文（包括完成型音节和 Jamo 字母）

        覆盖范围：
        - AC00–D7A3  完成型 Hangul syllables（도, 와, 릴 …）
        - 3130–318F  Compatibility Jamo（ㄷ, ㅏ, ㄹ …）
        - 1100–11FF  Hangul Jamo（ᄃ, ᅩ … Unicode Jamo）
        """
        if not text:
            return False
        for char in text:
            code = ord(char)
            if (0xAC00 <= code <= 0xD7A3   # composed syllables
                    or 0x3130 <= code <= 0x318F   # compatibility Jamo
                    or 0x1100 <= code <= 0x11FF):  # Hangul Jamo
                return True
        return False

    def _get_korean_initial_consonant(self, char: str) -> Optional[str]:
        """
        从韩文字符提取初声（初声）。

        韩文字符编码方式（完成型音节）：
        code = 0xAC00 + (initial × 588) + (medial × 28) + final

        初声索引：
        0=ㄱ, 1=ㄲ, 2=ㄴ, 3=ㄷ, 4=ㄸ, 5=ㄹ, 6=ㅁ, 7=ㅂ, 8=ㅃ, 9=ㅄ, 10=ㅅ,
        11=ㅆ, 12=ㅇ(零初声), 13=ㅈ, 14=ㅉ, 15=ㅊ, 16=ㅋ, 17=ㅌ, 18=ㅍ, 19=ㅎ

        仅当 initial != 12（ㅇ）时，才认为有初声。

        Jamo 兼容字符 (3131-318E) 已包含初声字母本身。
        """
        if not char:
            return None

        code = ord(char)

        # Case 1: 完成型音节 (AC00-D7A3)
        if 0xAC00 <= code <= 0xD7A3:
            # 计算初声索引
            offset = code - 0xAC00
            initial_idx = offset // 588

            # initial_idx == 12 是 ㅇ(零初声)，没有初声
            if initial_idx == 12:
                return None

            return "has_initial"  # 有初声标记

        # Case 2: Jamo 兼容字符 (3131-318E)
        # 这些是独立的音节字母，如 ㄱ, ㄴ, ㅏ, ㅠ 等
        if 0x3131 <= code <= 0x318E:
            # 3131-3164 是初声辅音
            # 3165-318E 是中声元音或其他
            if 0x3131 <= code <= 0x3164:
                # 这是初声字母本身
                return "has_initial"
            else:
                return None

        # Case 3: Unicode Hangul Jamo (1100-11FF)
        if 0x1100 <= code <= 0x11FF:
            # 1100-1112 是初声，1113-1114 是初声扩展
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
        """
        分解韩文字符：完整字符 + "-" 초성标记

        核心规则：
        - 有初声（初声 ≠ ㅇ）→ 输出 "-"(初声标记) + 完整字符
        - 无初声或初声 = ㅇ    → 直接输出完整字符

        时间分配策略：
        1. 若 phone_items 含本 word 时间段内的完整音节 token，用其时间戳
        2. 否则回退到等比均分

        ★ 改进逻辑：
        - 更准确的 Jamo 分解（使用 get_korean_initial_consonant）
        - 更鲁棒的时间段聚合
        - 完整字符作为主要输出单位
        """
        try:
            import jamo

            entries: List[Tuple[int, int, str]] = []
            if not korean_text:
                return entries

            # ── 1. 提取韩文字符及其初声信息 ──────────────────────────
            korean_chars: List[Tuple[str, bool]] = []  # (char, has_initial)
            for char in korean_text:
                if not char or char == ' ':
                    continue

                has_initial = False
                try:
                    # 使用改进的初声检测方法
                    initial_marker = self._get_korean_initial_consonant(char)
                    if initial_marker == "has_initial":
                        has_initial = True
                except Exception as jamo_err:
                    logger.debug(f"Jamo 初声检测失败 {char}: {jamo_err}")

                korean_chars.append((char, has_initial))

            if not korean_chars:
                logger.debug(f"韩文 '{korean_text}': 无有效字符")
                return entries

            n_chars = len(korean_chars)
            logger.debug(f"韩文 '{korean_text}': {n_chars} 个字符，初声分布：{[hi for _, hi in korean_chars]}")

            # ── 2. 尝试从 phone_items 中抽取本 word 内的音节时间段 ──
            syllable_time_ranges: List[Tuple[int, int]] = []
            if phone_items:
                word_phones = [
                    (s, e, p) for s, e, p in phone_items
                    if s >= word_start and s < word_end
                    and p not in self.SIL_PHONES
                    and p not in self.IGNORE_PHONES
                ]
                word_phones.sort(key=lambda x: x[0])

                # Case A: phone tier 中的条目数恰好等于音节数
                #         （MFA 直接输出整字）
                if len(word_phones) == n_chars:
                    syllable_time_ranges = [(s, e) for s, e, _ in word_phones]
                    logger.debug(
                        f"韩文 '{korean_text}': 使用完整 phone_items 时间段 "
                        f"({len(syllable_time_ranges)} 个)"
                    )

                # Case B: phone tier 中的条目是 Jamo 级别
                elif len(word_phones) > n_chars:
                    # 估算每个音节占几个 Jamo phone
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
                            # 最后一个音节一直到 word_end
                            syl_end = (
                                word_end if i == n_chars - 1
                                else (chunk[-1][1] if len(chunk) == n 
                                      else word_phones[min(idx + n, len(word_phones) - 1)][0])
                            )
                            syllable_time_ranges.append((syl_start, syl_end))
                        idx += n

                    logger.debug(
                        f"韩文 '{korean_text}': 从 Jamo phone_items 聚合 "
                        f"{len(syllable_time_ranges)} 个音节时间段"
                    )

            # ── 3. 回退：等比均分 ──────────────────────────────────────
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

                logger.debug(
                    f"韩文 '{korean_text}': 等比均分回退，{n_chars} 个音节"
                )

            # ── 4. 生成 LAB 条目 ──────────────────────────────────────
            for i, ((char, has_initial), (syl_start, syl_end)) in enumerate(
                zip(korean_chars, syllable_time_ranges)
            ):
                syl_dur = max(syl_end - syl_start, 200000)

                if has_initial:
                    # 有初声："-"(初声) + 完整字符
                    # 时间分配："-" 占 1/3，字占 2/3（最小各 60000）
                    dash_dur = max(syl_dur // 3, 60000)
                    dash_end = min(syl_start + dash_dur, syl_end - 60000)

                    entries.append((syl_start, dash_end, "-"))
                    logger.debug(f"  [{syl_start}-{dash_end}] - (초성)")

                    entries.append((dash_end, syl_end, char))
                    logger.debug(f"  [{dash_end}-{syl_end}] {char} (자)")
                else:
                    # 无初声：直接输出完整字符
                    entries.append((syl_start, syl_end, char))
                    logger.debug(f"  [{syl_start}-{syl_end}] {char} (자)")

            # 确保最后一个条目到达 word_end
            if entries and entries[-1][1] < word_end:
                s, e, p = entries[-1]
                entries[-1] = (s, word_end, p)

            return entries

        except ImportError:
            logger.error("jamo 库未安装，无法分解韩文字符")
            logger.error("请运行：pip install jamo")
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
        """处理粤语 Word Tier → 粤拼 + con 标记"""
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
        """Word Tier 对齐主逻辑"""
        try:
            from textgrid import TextGrid
            tg = TextGrid.fromFile(textgrid_path)
            logger.info(f"加载 TextGrid: {len(tg)} tiers")
            
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
            
            logger.info(f"✓ Word Tier 模式启动（多语言 + IPA转换）")
            
            phone_items: List[Tuple[int, int, str]] = []
            if phone_tier is not None:
                phone_items = self._extract_phone_items(phone_tier)
                logger.info(f"✓ Phone Tier: {len(phone_items)} 个音素条目")
            else:
                logger.warning("未找到 Phone Tier")
            
            if lang in ('zh', 'cmn'):
                logger.info("✓ 语言模式：中文普通话")
                lines = self._process_zh_words(word_tier, phone_items, text)
            elif lang == 'en':
                logger.info("✓ 语言模式：英语（ARPABET）")
                lines = self._process_en_words(word_tier, phone_items, text)
            elif lang == 'ja':
                logger.info("✓ 语言模式：日语（ROMAJI，IPA→Romaji 自动转换）")
                lines = self._process_ja_words(word_tier, phone_items, text)
            elif lang == 'ko':
                logger.info("✓ 语言模式：韩语（Hangul Jamo，初声 - 标记）★ ENHANCED!")
                lines = self._process_ko_words(word_tier, phone_items, text)
            elif lang == 'yue':
                logger.info("✓ 语言模式：粤语（粤拼）")
                lines = self._process_yue_words(word_tier, phone_items, text)
            else:
                logger.warning(f"未知语言 '{lang}'，使用英语模式")
                lines = self._process_en_words(word_tier, phone_items, text)
            
            result = "\n".join(lines)
            logger.info(f"✓ 对齐完成（{len(lines)} 行）")
            return result
            
        except ImportError:
            logger.warning("textgrid 未安装，启用手工解析")
            return self._parse_textgrid_manual(textgrid_path, text, lang)
        except Exception as e:
            logger.error(f"对齐失败: {e}", exc_info=True)
            return ""

    def _parse_textgrid_manual(
        self,
        textgrid_path: str,
        text: str,
        lang: str = 'zh'
    ) -> str:
        """手工解析 TextGrid"""
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
                logger.info(f"手工解析：Word Tier {len(word_tier)} 项，Phone Tier {len(phone_items)} 个")
            else:
                logger.info(f"手工解析：Word Tier {len(word_tier)} 项，无 Phone Tier")
            
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
            
            return "\n".join(lines)
            
        except Exception as e:
            logger.error(f"手工解析失败: {e}", exc_info=True)
            return ""

    def _textgrid_to_lab(self, textgrid_path: str, text: str, lang: str = 'zh') -> str:
        """主入口"""
        return self._textgrid_to_lab_word_tier_primary(textgrid_path, text, lang)

    def _get_audio_duration(self, audio_path: str) -> int:
        """获取音频时长（单位：100ns）"""
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
    # 主流程
    # =====================================================================
    def process(self, audio_file, text: str, language: str = "cmn") -> Dict:
        """处理单个音频文件"""
        start_time = time.time()
        try:
            self.temp_dir = tempfile.mkdtemp(prefix="mfa_")
            raw_lang = (language or "cmn").lower().strip()

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

            audio_path = os.path.join(self.temp_dir, audio_file.filename)
            audio_file.save(audio_path)
            audio_duration = self._get_audio_duration(audio_path)

            corpus_dir = os.path.join(self.temp_dir, "corpus")
            os.makedirs(corpus_dir, exist_ok=True)
            basename = Path(audio_path).stem
            audio_dest = os.path.join(corpus_dir, f"{basename}.wav")
            shutil.copy2(audio_path, audio_dest)
            text_file = os.path.join(corpus_dir, f"{basename}.txt")
            with open(text_file, "w", encoding="utf-8") as f:
                f.write(text_for_mfa)

            output_dir = os.path.join(self.temp_dir, "aligned")
            os.makedirs(output_dir, exist_ok=True)
            models = MFAChecker.LANGUAGE_MODELS.get(lang, {})
            dict_model = models.get("dictionary", lang)
            acoustic_model = models.get("acoustic", lang)
            py = MFAChecker.env_python()
            cmd = [
                str(py), "-m", "montreal_forced_aligner.command_line.mfa",
                "align", corpus_dir, dict_model, acoustic_model,
                output_dir, "--clean", "--single_speaker"
            ]
            
            timeout_seconds = 600 if lang in ('ja', 'ko') else 300
            
            env = os.environ.copy()
            mfa_env_dir = MFAChecker.env_dir()
            env["CONDA_PREFIX"] = str(mfa_env_dir)
            lib_bin = mfa_env_dir / "Library" / "bin"
            if lib_bin.exists():
                env["PATH"] = str(lib_bin) + os.pathsep + env.get("PATH", "")
            lib_dir = mfa_env_dir / "lib"
            if lib_dir.exists():
                env["LD_LIBRARY_PATH"] = str(lib_dir) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
            
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout_seconds, env=env
            )
            
            if result.returncode != 0:
                error_msg = result.stderr if result.stderr else "MFA 失败"
                logger.error(f"MFA: {error_msg}")
                return {
                    "success": False, "error": error_msg,
                    "processing_time": int((time.time() - start_time) * 1000)
                }

            textgrid_files = list(Path(output_dir).glob("**/*.TextGrid"))
            if not textgrid_files:
                return {
                    "success": False, "error": "MFA 未生成 TextGrid",
                    "processing_time": int((time.time() - start_time) * 1000)
                }
            textgrid_path = str(textgrid_files[0])

            lab_content = self._textgrid_to_lab(textgrid_path, text, lang=lang)

            if not lab_content:
                return {
                    "success": False, "error": "LAB 内容为空",
                    "processing_time": int((time.time() - start_time) * 1000)
                }

            processing_time = int((time.time() - start_time) * 1000)

            return {
                "success": True,
                "raw_text": text,
                "phoneme_text": phoneme_text,
                "lab_content": lab_content,
                "textgrid_path": textgrid_path,
                "audio_duration": audio_duration,
                "processing_time": processing_time
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False, "error": "MFA 超时",
                "processing_time": int((time.time() - start_time) * 1000)
            }
        except FileNotFoundError as e:
            return {
                "success": False, "error": "MFA 命令不存在",
                "processing_time": int((time.time() - start_time) * 1000)
            }
        except Exception as e:
            logger.error(f"错误: {e}", exc_info=True)
            return {
                "success": False, "error": str(e),
                "processing_time": int((time.time() - start_time) * 1000)
            }
        finally:
            self._cleanup()

    def _cleanup(self):
        """清理临时文件"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except Exception as e:
                logger.warning(f"清理失败: {e}")
