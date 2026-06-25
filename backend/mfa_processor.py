# -*- coding: utf-8 -*-
"""
MFA 处理核心模块 - 多语言增强版 v9.3 (已包含特定标点符号清洗规则)
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
from phoneme_converter import (
    convert_phoneme,
    build_ja_hiragana_lab,
    merge_lab_silence,
    word_to_arpabet,
    distribute_arpabet_phones,
    katakana_to_romaji_moras,
    hiragana_to_katakana,
)

logger = logging.getLogger(__name__)

# ── Module-level cache: SudachiPy dict loads in ~4 min on first call.
# Caching it here means subsequent requests reuse the same object. ──────────
_ja_tokenizer: Optional[object] = None


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
    # 新增：文本标点符号规范化清洗
    # =====================================================================
    def _clean_input_text(self, text: str) -> str:
        """
        文本清洗：
        1) 删除引号、书名号、括号类符号
        2) 删除冒号、分号、顿号
        3) 统一句末标点，避免句子边界被奇怪符号干扰
        """
        if not text:
            return ""

        # 删除引号
        text = re.sub(r'[「」"“”]', '', text)

        # 删除括号 / 书名号
        text = re.sub(r'[（）()《》＜＞<>【】]', '', text)

        # 删除会干扰对齐的标点
        text = re.sub(r'[：；:;、]', '', text)

        # 可选：把感叹号和问号统一成句号，减少分段歧义
        text = re.sub(r'[!?！？]', '。', text)

        # 合并多余空白
        text = re.sub(r'\s+', ' ', text).strip()

        return text

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
            return False, f"❌ 中文环境错误：{str(e)}"

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
        """检查日语环境（首次调用时加载字典，后续直接复用缓存，避免每次重复 ~4 min 加载）"""
        global _ja_tokenizer
        try:
            from sudachipy import Dictionary
            if _ja_tokenizer is None:
                _ja_tokenizer = Dictionary().create()
            # 轻量冒烟测试，复用已缓存的 tokenizer
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
    # 日语：英语 → 片假名（MFA 对齐前预处理 + 兜底转换）
    # =====================================================================
    # 背景：MFA 的 japanese_mfa 词典只收录假名/汉字词条，完全不认识拉丁字母
    # 拼写的英语单词——送进去只会被判定成 OOV（spn），拿不到任何音素级时间
    # 戳。但日语外来语本身在词典里就是以片假名形式收录的（例如 "love" 对应
    # 词条其实是「ラブ」），所以思路是：在文本送入 MFA 之前，先用 sudachipy
    # 把能识别的英语单词换成它的片假名读音，MFA 就能像处理普通外来语一样正
    # 常给出对齐结果；万一个别词 sudachi 也不认识（生僻词/专有名词），则在
    # 下面 _process_ja_words() 里用同一套转换逻辑做最后兜底，至少能输出合
    # 理的假名近似，而不是把 ARPABET 音素硬塞进只认日语罗马音的
    # build_ja_hiragana_lab()。

    @staticmethod
    def _is_full_katakana(s: str) -> bool:
        """字符串是否整体由片假名字符组成（含长音符ー、促音ッ等片假名区块字符）"""
        if not s:
            return False
        return all("\u30A0" <= ch <= "\u30FF" for ch in s)

    def _valid_katakana_reading(self, surface: str, reading: Optional[str]) -> Optional[str]:
        """
        校验 sudachipy 给出的 reading_form() 是否是"真正转换成功"的片假名读音。

        sudachi 对完全不认识的词通常会原样返回 surface 本身，或者读音里混杂
        非片假名字符——这两种情况都说明 sudachi 没能给出有效转换，不应当替
        换，否则会把英语原文误判成转换成功，产生错误读音。
        """
        reading = (reading or "").strip()
        if not reading or reading == surface:
            return None
        if not self._is_full_katakana(reading):
            return None
        return reading

    def _normalize_japanese_text_for_mfa(self, text: str) -> str:
        """
        MFA 对齐前的日语文本预处理：把文本中夹杂的英语单词替换成 sudachipy
        给出的片假名读音，写入 MFA 语料 txt 之前调用。

        只替换 sudachi 词典里确实有片假名读音的词；sudachi 不认识的生僻词/
        专有名词原样保留英语，交给 _process_ja_words() 里的兜底逻辑处理
        （那样至少在日志里能看到，而不是被静默替换成错误读音）。
        """
        global _ja_tokenizer
        if not text:
            return text

        try:
            from sudachipy import Dictionary
            if _ja_tokenizer is None:
                _ja_tokenizer = Dictionary().create()
            morphemes = _ja_tokenizer.tokenize(text)
        except Exception as e:
            logger.warning(
                f"[ja] sudachipy 分词失败，跳过英语→片假名预处理（文本中的英语单词"
                f"送入 MFA 可能直接判定为 OOV）：{e}"
            )
            return text

        pieces: List[str] = []
        substitutions: List[Tuple[str, str]] = []
        for m in morphemes:
            surface = m.surface()
            if self._is_english_word(surface):
                try:
                    reading = m.reading_form()
                except Exception:
                    reading = None
                katakana = self._valid_katakana_reading(surface, reading)
                if katakana:
                    pieces.append(katakana)
                    substitutions.append((surface, katakana))
                    continue
            pieces.append(surface)

        if substitutions:
            logger.info(
                f"[ja] MFA 预处理：{len(substitutions)} 个英语单词已转换为片假名读音"
                f"以便对齐：{substitutions}"
            )

        return "".join(pieces)

    def _get_katakana_reading_for_word(self, word: str) -> Optional[str]:
        """
        单词级兜底：用 sudachipy 查询一个孤立英语单词（已在 Word Tier 里，
        说明 _normalize_japanese_text_for_mfa() 没能在预处理阶段替换掉它，
        或调用方根本没经过那一步，例如复用本类的替代对齐后端）的片假名读音。
        """
        global _ja_tokenizer
        word = (word or "").strip()
        if not word:
            return None
        try:
            from sudachipy import Dictionary
            if _ja_tokenizer is None:
                _ja_tokenizer = Dictionary().create()
            morphemes = _ja_tokenizer.tokenize(word)
            reading = "".join((m.reading_form() or "") for m in morphemes)
        except Exception as e:
            logger.debug(f"[ja] sudachipy 片假名读音查询失败 '{word}': {e}")
            return None
        return self._valid_katakana_reading(word, reading)

    def _distribute_katakana_mora_phones(
        self,
        word_start: int,
        word_end: int,
        moras: List[Tuple[str, str]],
    ) -> List[Tuple[int, int, str]]:
        """
        把 katakana_to_romaji_moras() 给出的 (辅音, 元音) 二元组列表，展开成
        build_ja_hiragana_lab() 能直接消费的单音素条目序列，并把该词已有的
        时间跨度按音素个数等分。

        日语各音素之间的时长差异远小于英语 ARPABET（不像英语有明显的长元音/
        短辅音之分），等分是这里足够合理的近似；这是 MFA Phone Tier 完全没
        有该词真实音素时的最后一道兜底——只要 sudachi 能给出片假名读音，就
        能保证最终 LAB 至少是合理的平假名序列，而不是 ARPABET 乱码或被 '-'
        吞掉。
        """
        flat: List[str] = []
        for cons, vow in moras:
            if cons:
                flat.append(cons)
            if vow:
                flat.append(vow)

        if not flat:
            return []
        if len(flat) == 1:
            return [(word_start, word_end, flat[0])]

        duration = word_end - word_start
        n = len(flat)
        result: List[Tuple[int, int, str]] = []
        cursor = word_start
        for i, p in enumerate(flat):
            if i == n - 1:
                seg_end = word_end
            else:
                seg_end = word_start + int(round(duration * (i + 1) / n))
                seg_end = max(seg_end, cursor + 1)
            result.append((cursor, seg_end, p))
            cursor = seg_end

        return result

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
        text: str,
        english_word_align: bool = False,
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
                if english_word_align:
                    # 英语单词级对齐模式：直接输出单词，不做音素拆分
                    lines.append(f"{start} {end} {mark.lower()}")
                else:
                    entries = self._get_arpabet_entries(start, end, phone_items)
                    if entries:
                        for s, e, p in entries:
                            lines.append(f"{s} {e} {p}")
                    else:
                        # MFA Phone Tier 为空（CJK 声学模型不输出 ARPABET，或替代
                        # 对齐后端 phone_items=[]）：走 G2P 获取音素序列，再按权重
                        # 比例把词的时间跨度分配给各个 ARPABET 音素。
                        g2p_phones = word_to_arpabet(mark)
                        if g2p_phones:
                            for s, e, p in distribute_arpabet_phones(start, end, g2p_phones):
                                lines.append(f"{s} {e} {p}")
                        else:
                            logger.warning(
                                f"[zh] 英语词 '{mark}' 无法获取音素（MFA Phone Tier 为空，"
                                "G2P 词典 / g2p_en 均未命中），按整词输出。"
                            )
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
        text: str,
        english_word_align: bool = False,
    ) -> List[str]:
        """处理英语 Word Tier → ARPABET（或单词级，取决于 english_word_align）"""
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

            # 英语单词级对齐模式：直接输出单词，跳过音素拆分
            if english_word_align:
                lines.append(f"{start} {end} {mark.lower()}")
                continue

            entries = self._get_arpabet_entries(start, end, phone_items)
            if entries:
                for s, e, p in entries:
                    lines.append(f"{s} {e} {p}")
                continue

            # phone_items 里没有这个词的真实 MFA 音素（最常见原因：调用方
            # 是 WhisperX/Qwen3 等替代对齐后端，根本没有逐音素 TextGrid，
            # AltAlignerBase._word_entries_to_lab() 永远传入空的
            # phone_items 列表）。改用 G2P 把这个词转换为 ARPABET 音素
            # 序列，再按音素类型权重把该词已有的时间跨度比例分配给各
            # 个音素——而不是像旧版本那样把整个单词原样当成一个"音素"
            # 塞进 LAB（这正是 WhisperXAligner 英语输出始终停留在词级、
            # 从未真正到达音素级的根本原因）。
            g2p_phones = word_to_arpabet(mark)
            if g2p_phones:
                for s, e, p in distribute_arpabet_phones(start, end, g2p_phones):
                    lines.append(f"{s} {e} {p}")
                continue

            # G2P 也失败（词典和 g2p_en 都未命中，或均未安装）：保留旧
            # 行为作为最终兜底，至少不丢时间轴；同时给出明确告警，
            # 不会静默退化成"看起来正常但其实是整词"的输出。
            logger.warning(
                f"[MFA] 英语词 '{mark}' 无法获取音素（MFA Phone Tier 为空，"
                "G2P 词典 / g2p_en 均未命中），按整词输出。如需音素级 "
                "ARPABET，请确认已下载 MFA english_us_mfa 词典，或执行 "
                "pip install g2p_en"
            )
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
                
            # 纯日文假名直通，避免 しゅ 这类拗音被后面的英语/兜底逻辑吞掉
            if re.fullmatch(r"[\u3040-\u309F\u30A0-\u30FFー]+", mark):
                katakana = hiragana_to_katakana(mark)
                moras = katakana_to_romaji_moras(katakana)
                mora_entries = self._distribute_katakana_mora_phones(start, end, moras)
                if mora_entries:
                    for s, e, p in mora_entries:
                        lines.append(f"{s} {e} {p}")
                    continue
                
            if self._is_english_word(mark):
                # ① 首选：sudachi 片假名读音 → romaji 音素序列。
                # 正常情况下文本已经在 process() 里被 _normalize_japanese_text_
                # for_mfa() 预处理过，这里基本不会再遇到英语单词；会走到这条
                # 分支通常是：(a) sudachi 在预处理阶段没认出这个词，或 (b) 调
                # 用方是复用本类逻辑的替代对齐后端（WhisperX / Qwen3 等），从
                # 未经过那一步预处理。两种情况都先再查一次片假名读音——只要
                # sudachi 认得，就能展开成 build_ja_hiragana_lab() 能正确处理
                # 的日语罗马音，而不是 ARPABET。
                katakana = self._get_katakana_reading_for_word(mark)
                if katakana:
                    moras = katakana_to_romaji_moras(katakana)
                    mora_entries = self._distribute_katakana_mora_phones(start, end, moras)
                    if mora_entries:
                        for s, e, p in mora_entries:
                            lines.append(f"{s} {e} {p}")
                        continue

                # ② sudachi 也无法识别（生僻词/专有名词）：保留旧的 ARPABET
                # 兜底，至少不丢时间轴；但明确告警——音素体系不是日语罗马音，
                # build_ja_hiragana_lab() 无法正确转换，对应位置在最终 LAB
                # 里可能变成 '-' 占位或被吞掉，建议检查该词。
                entries = self._get_arpabet_entries(start, end, phone_items)
                if entries:
                    for s, e, p in entries:
                        lines.append(f"{s} {e} {p}")
                else:
                    g2p_phones = word_to_arpabet(mark)
                    if g2p_phones:
                        for s, e, p in distribute_arpabet_phones(start, end, g2p_phones):
                            lines.append(f"{s} {e} {p}")
                    else:
                        logger.warning(
                            f"[ja] 英语词 '{mark}' 无法获取音素（MFA Phone Tier 为空，"
                            "G2P 词典 / g2p_en 均未命中），按整词输出。"
                        )
                        lines.append(f"{start} {end} {mark.lower()}")
                logger.warning(
                    f"[ja] 英语词 '{mark}' sudachipy 未能给出片假名读音，已退回 "
                    "ARPABET 兜底；由于音素体系不是日语罗马音，最终 LAB 中对应"
                    "位置可能不是有效假名，建议检查该词是否为生僻外来语/专有名词。"
                )
                continue

            entries = self._get_romaji_entries(start, end, phone_items)
            if entries:
                for s, e, p in entries:
                    lines.append(f"{s} {e} {p}")
            else:
                # ★ MFA Phone Tier 为空（该词 OOV / 字典未收录，MFA 标注为 spn）。
                # 这是拗音（しゅ / ちょ 等）和生僻词最常见的失败原因：
                # MFA japanese_mfa 词典以「词」为粒度收录条目，孤立假名音节
                # （作为歌词出现时很常见）若不在词典中，Phone Tier 就会给 spn，
                # 导致 _get_romaji_entries 过滤掉 spn 后返回空列表，整个字/词
                # 的时间段从 LAB 输出里彻底消失（正是"しゅ 消失只剩か"的根因）。
                #
                # 回退策略：把 Word Tier 里的假名文字本身当作发音依据——
                #   平假名 → 片假名（codepoint 平移）
                #   → katakana_to_romaji_moras → 罗马音音素 (辅音, 元音) 对列表
                #   → _distribute_katakana_mora_phones → 等分该词时间段的各音素条目
                #
                # 时间精度：音素级时间戳是等分近似（MFA 给不出真实的音素级对齐），
                # 词级时间段本身来自 MFA 的词级强制对齐，因此词边界是准确的；
                # 这比把整个词时间段丢弃（旧行为）要好得多。
                kata = hiragana_to_katakana(mark)
                moras = katakana_to_romaji_moras(kata)
                mora_entries = self._distribute_katakana_mora_phones(start, end, moras)
                if mora_entries:
                    for s, e, p in mora_entries:
                        lines.append(f"{s} {e} {p}")
                    logger.info(
                        f"[ja] OOV 回退: '{mark}' (MFA Phone Tier 为 spn/空) → "
                        f"由假名文字自身生成 {len(mora_entries)} 个罗马音音素条目，"
                        f"词级时间段准确，音素级时间为等分近似。"
                    )
                else:
                    # mark 含非假名字符（汉字/符号），katakana_to_romaji_moras
                    # 无法解析 → 只能放弃，保持旧行为。
                    logger.warning(
                        f"[ja] word '{mark}' 无法从 Phone Tier 获取音素，"
                        f"且假名→罗马音回退也失败（mark 含非假名字符？），跳过。"
                    )

        return lines

    def _process_ko_words(
        self,
        word_tier,
        phone_items: List[Tuple[int, int, str]],
        text: str,
        english_word_align: bool = False,
    ) -> List[str]:
        """处理韩语 Word Tier"""
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
                if english_word_align:
                    # 英语单词级对齐模式：直接输出单词，不做音素拆分
                    lines.append(f"{start} {end} {mark.lower()}")
                else:
                    entries = self._get_arpabet_entries(start, end, phone_items)
                    if entries:
                        for s, e, p in entries:
                            lines.append(f"{s} {e} {p}")
                    else:
                        g2p_phones = word_to_arpabet(mark)
                        if g2p_phones:
                            for s, e, p in distribute_arpabet_phones(start, end, g2p_phones):
                                lines.append(f"{s} {e} {p}")
                        else:
                            logger.warning(
                                f"[ko] 英语词 '{mark}' 无法获取音素（MFA Phone Tier 为空，"
                                "G2P 词典 / g2p_en 均未命中），按整词输出。"
                            )
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
        """检测文本是否为韩文"""
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
        """从韩文字符提取初声"""
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
        """分解韩文字符"""
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
        text: str,
        english_word_align: bool = False,
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
                if english_word_align:
                    # 英语单词级对齐模式：直接输出单词，不做音素拆分
                    lines.append(f"{start} {end} {mark.lower()}")
                else:
                    entries = self._get_arpabet_entries(start, end, phone_items)
                    if entries:
                        for s, e, p in entries:
                            lines.append(f"{s} {e} {p}")
                    else:
                        g2p_phones = word_to_arpabet(mark)
                        if g2p_phones:
                            for s, e, p in distribute_arpabet_phones(start, end, g2p_phones):
                                lines.append(f"{s} {e} {p}")
                        else:
                            logger.warning(
                                f"[yue] 英语词 '{mark}' 无法获取音素（MFA Phone Tier 为空，"
                                "G2P 词典 / g2p_en 均未命中），按整词输出。"
                            )
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
        lang: str = 'zh',
        english_word_align: bool = False,
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
                lines = self._process_zh_words(word_tier, phone_items, text, english_word_align=english_word_align)
            elif lang == 'en':
                lines = self._process_en_words(word_tier, phone_items, text, english_word_align=english_word_align)
            elif lang == 'ja':
                lines = self._process_ja_words(word_tier, phone_items, text)
            elif lang == 'ko':
                lines = self._process_ko_words(word_tier, phone_items, text, english_word_align=english_word_align)
            elif lang == 'yue':
                lines = self._process_yue_words(word_tier, phone_items, text, english_word_align=english_word_align)
            else:
                lines = self._process_en_words(word_tier, phone_items, text, english_word_align=english_word_align)
            
            # ★ 后处理：日语 romaji → hiragana；所有语言合并 '-' 标记
            lines = self._apply_lab_postprocess(lines, lang)
            return "\n".join(lines)
            
        except ImportError:
            return self._parse_textgrid_manual(textgrid_path, text, lang, english_word_align=english_word_align)
        except Exception as e:
            logger.error(f"对齐失败: {e}", exc_info=True)
            return ""

    # =====================================================================
    # LAB 后处理辅助方法
    # =====================================================================
    @staticmethod
    def _parse_lab_lines(lines: List[str]) -> List[Tuple[int, int, str]]:
        """将 LAB 行列表解析为 (start_100ns, end_100ns, label) 列表。"""
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

        规则说明
        --------
        '-' 在有声音节之后且紧邻（间距 ≤ 50 ms）→ 合并到左侧（延伸左侧结束时间）
        '-' 在句首、静音后、或跨越句间停顿（间距 > 50 ms）→ 直接删除
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
        lang: str = 'zh',
        english_word_align: bool = False,
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
            
            if lang in ('zh', 'cmn'):
                lines = self._process_zh_words(word_tier, phone_items, text, english_word_align=english_word_align)
            elif lang == 'en':
                lines = self._process_en_words(word_tier, phone_items, text, english_word_align=english_word_align)
            elif lang == 'ja':
                lines = self._process_ja_words(word_tier, phone_items, text)
            elif lang == 'ko':
                lines = self._process_ko_words(word_tier, phone_items, text, english_word_align=english_word_align)
            elif lang == 'yue':
                lines = self._process_yue_words(word_tier, phone_items, text, english_word_align=english_word_align)
            else:
                lines = self._process_en_words(word_tier, phone_items, text, english_word_align=english_word_align)
            
            # ★ 后处理：日语 romaji → hiragana；所有语言合并 '-' 标记
            lines = self._apply_lab_postprocess(lines, lang)
            return "\n".join(lines)
            
        except Exception as e:
            logger.error(f"手工解析失败: {e}", exc_info=True)
            return ""

    def _textgrid_to_lab(self, textgrid_path: str, text: str, lang: str = 'zh',
                          english_word_align: bool = False) -> str:
        return self._textgrid_to_lab_word_tier_primary(textgrid_path, text, lang,
                                                        english_word_align=english_word_align)

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
    # 主流程（修改位置）
    # =====================================================================
    def process(self, audio_file, text: str, language: str = "cmn",
                english_word_align: bool = False) -> Dict:
        """处理单个音频文件"""
        start_time = time.time()
        try:
            self.temp_dir = tempfile.mkdtemp(prefix="mfa_")
            raw_lang = (language or "cmn").lower().strip()

            # ★ 关键新增点：在任何语言分支提取前，优先使用新规则统一清洗输入的原始文本
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
                # ★ 新增：送入 MFA 之前，把文本里的英语单词换成 sudachipy 给出
                # 的片假名读音——japanese_mfa 词典不认识拉丁字母拼写，原样送
                # 进去只会被判定为 OOV（spn），完全拿不到音素级对齐结果。
                text_for_mfa = self._normalize_japanese_text_for_mfa(text.strip())
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

            lab_content = self._textgrid_to_lab(textgrid_path, text, lang=lang,
                                                  english_word_align=english_word_align)

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
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except Exception as e:
                logger.warning(f"清理失败: {e}")