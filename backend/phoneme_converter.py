# -*- coding: utf-8 -*-
"""
phoneme_converter.py (Enhanced Korean IPA→Jamo support)
Converts MFA output phonemes (IPA) to target phoneme sets for GPT-SoVITS.

Supported conversions:
  ja / jpn  →  Romaji      (MFA japanese_mfa IPA output)
  en / eng  →  ARPAbet     (MFA english_us_mfa output)
  zh / cmn  →  Pinyin      (MFA mandarin_china_mfa, pass-through)
  ko / kor  →  Hangul Jamo (MFA korean_mfa, IPA→Jamo conversion) ★ NEW!
  yue       →  Jyutping    (MFA mandarin_china_mfa, pass-through)
"""

from __future__ import annotations
from typing import Optional
import logging

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# Silence / boundary tokens (language-agnostic)
# ══════════════════════════════════════════════════��═══════════

SILENCE_TOKENS: set[str] = {
    "", "sp", "sil", "spn", "SIL", "SP", "SPN",
    "<eps>", "<unk>", "<UNK>",
}


# ══════════════════════════════════════════════════════════════
# Japanese  MFA IPA  →  Romaji (完整映射表)
# ══════════════════════════════════════════════════════════════

JA_IPA_TO_ROMAJI: dict[str, str] = {

    # ── 元音 (Vowels) ────────────────────────────────────────
    "a":  "a",
    "e":  "e",
    "i":  "i",
    "o":  "o",
    "ɯ":  "u",        # U+026F  闭后不圆元音 (日语 /u/)
    "ɯ̥": "u",        # U+026F + U+0325  清后元音
    "i̥":  "i",       # U+0069 + U+0325  清i
    "ɨ̥":  "u",       # U+0268 + U+0325  清中央元音

    # ── 塞音 (Stops) ──────────────────────────────────────────
    "p":  "p",
    "b":  "b",
    "t":  "t",
    "d":  "d",
    "k":  "k",
    "g":  "g",        # ASCII g
    "ɡ":  "g",        # U+0261  IPA版g
    "c":  "k",        # U+0063  清硬腭塞音
    "ʔ":  "",         # U+0294  声门塞音 → 省略

    # ── 摩擦音 (Fricatives) ───────────────────────────────────
    "s":  "s",
    "z":  "z",
    "h":  "h",
    "ɸ":  "f",        # U+0278  双唇摩擦音 (ふ)
    "f":  "f",
    "v":  "v",
    "ɕ":  "sh",       # U+0255  清硬腭摩擦音 (し/しゃ/しゅ/しょ)
    "ʑ":  "j",        # U+0291  浊硬腭摩擦音

    # ── 塞擦音 (Affricates) ───────────────────────────────────
    "ts":  "ts",
    "dz":  "z",
    "tɕ":  "ch",      # 清硬腭塞擦音 (ち/ちゃ/etc.)
    "dʑ":  "j",       # 浊硬腭塞擦音 (じ/じゃ/etc.)
    "ch":  "ch",      # 已是Romaji形式

    # ── 鼻音 (Nasals) ─────────────────────────────────────────
    "n":   "n",
    "m":   "m",
    "ŋ":   "ng",      # U+014B  软腭鼻音
    "ng":  "ng",      # 已是Romaji形式
    "ɲ":   "ny",      # U+0272  硬腭鼻音
    "ɴ":   "N",       # U+0274  鼻化鼻音/mora nasal
    "N":   "N",       # 拨音 (ん)

    # ── 液音 (Liquids) ────────────────────────────────────────
    "r":   "r",
    "ɾ":   "r",       # U+027E  齿龈轻拍音

    # ── 半元音/滑音 (Semivowels) ──────────────────────────────
    "w":  "w",
    "j":  "y",        # U+006A  硬腭近音
    "y":  "y",        # 已是Romaji形式

    # ── 软腭化辅音 (Palatalized Consonants) ──────────────────
    "ky": "ky",
    "gy": "gy",
    "ny": "ny",
    "hy": "hy",
    "my": "my",
    "ry": "ry",
    "py": "py",
    "by": "by",
    "ty": "ty",
    "dy": "dy",

    # ── 已是Romaji形式的二合字 (Digraphs) ──────────────────
    "sh": "sh",
}


# ══════════════════════════════════════════════════════════════
# English  MFA IPA / ARPAbet  →  ARPAbet (lowercase)
# ══════════════════════════════════════════════════════════════

EN_IPA_TO_ARPABET: dict[str, str] = {

    # ── 元音 (Vowels - IPA → ARPAbet) ──────────────────────────
    "iː": "iy",  "i":  "iy",
    "ɪ":  "ih",
    "eɪ": "ey",
    "ɛ":  "eh",  "e":  "eh",
    "æ":  "ae",
    "ɑː": "aa",  "ɑ":  "aa",
    "ɔː": "ao",  "ɔ":  "ao",
    "oʊ": "ow",
    "ʊ":  "uh",
    "uː": "uw",  "u":  "uw",
    "ʌ":  "ah",
    "ɜː": "er",  "ɜ":  "er",  "ɚ": "er",
    "ə":  "ax",
    "aɪ": "ay",
    "aʊ": "aw",
    "ɔɪ": "oy",

    # ── 元音 (ARPAbet pass-through) ────────────────────────────
    "aa": "aa", "ae": "ae", "ah": "ah", "ao": "ao",
    "aw": "aw", "ax": "ax", "ay": "ay",
    "eh": "eh", "er": "er", "ey": "ey",
    "ih": "ih", "iy": "iy",
    "ow": "ow", "oy": "oy",
    "uh": "uh", "uw": "uw",

    # ── 辅音 (Consonants - IPA → ARPAbet) ──────────────────────
    "p":  "p",   "b":  "b",
    "t":  "t",   "d":  "d",
    "k":  "k",
    "g":  "g",   "ɡ":  "g",
    "f":  "f",   "v":  "v",
    "θ":  "th",  "ð":  "dh",
    "s":  "s",   "z":  "z",
    "ʃ":  "sh",  "ʒ":  "zh",
    "h":  "hh",  "ɦ":  "hh",
    "tʃ": "ch",  "dʒ": "jh",
    "m":  "m",   "n":  "n",
    "ŋ":  "ng",
    "l":  "l",
    "ɹ":  "r",   "r":  "r",
    "w":  "w",   "j":  "y",
    "ʔ":  "",

    # ── 辅音 (ARPAbet pass-through) ────────────────────────────
    "ch": "ch", "dh": "dh", "dx": "dx",
    "dr": "dr", "tr": "tr",
    "hh": "hh", "jh": "jh",
    "ng": "ng", "sh": "sh", "th": "th", "zh": "zh",
}


# ══════════════════════════════════════════════════════════════
# Korean IPA → Hangul Jamo ★ NEW & ENHANCED!
# ══════════════════════════════════════════════════════════════
# 
# 韩文字母 (Hangul Jamo) 映射表
# 韩语 MFA 输出的 IPA 音标转换为韩文字母 (초성/중성/종성)
#
# 关键 IPA 符号：
#   ɐ   (U+0250) 中央元音低 → ㅏ /a/
#   ɛ   (U+025B) 开前元音 → ㅓ /eo/
#   ɨ   (U+0268) 中央元音闭 → ㅡ /eu/
#   ə   (U+0259) schwa → ㅗ /o/
#   ɭ   (U+026D) 咽化卷舌音 → ㄹ /l/
#   ɲ   (U+0272) 硬腭鼻音 → ㄴ /n/
#   sʰ  (U+0073 + 02B7) aspirated /s/ → ㅆ /ss/
#   t̚   (U+0074 + 031A) unreleased /t/ → ㄷ /t/
#   ng  U+014B) 软腭鼻音 → ㅇ /ng/
#
# 转换逻辑：
#   1. IPA → 韩文字母 (1-1映射)
#   2. 多字符序列 → 拆分处理 (e.g., "tʰ" → "ㅌ")
#   3. 未知音素 → 日志警告 + 直接通过

KO_IPA_TO_JAMO: dict[str, str] = {
    # ── 元音 (Vowels / 모음) ──────────────────────────────────────
    "a":    "ㅏ",      # IPA /a/ → Korean /a/
    "ɐ":    "ㅏ",      # U+0250 중앙모음저 → /a/ ★ KEY!
    "ɑ":    "ㅏ",      # IPA /ɑ/ → /a/
    "ɑː":   "ㅏ",      # long /ɑ/ → /a/
    "ə":    "ㅗ",      # schwa → /o/
    "ɛ":    "ㅓ",      # IPA /ɛ/ → /eo/ ★ KEY!
    "e":    "ㅔ",      # IPA /e/ → /e/
    "i":    "ㅣ",      # IPA /i/
    "ɪ":    "ㅣ",      # IPA /ɪ/ → /i/
    "iː":   "ㅣ",      # long /i/
    "o":    "ㅗ",      # IPA /o/
    "ɔ":    "ㅗ",      # IPA /ɔ/ → /o/
    "ɔː":   "ㅗ",      # long /ɔ/ → /o/
    "u":    "ㅜ",      # IPA /u/
    "ʊ":    "ㅜ",      # IPA /ʊ/ → /u/
    "uː":   "ㅜ",      # long /u/
    "ʌ":    "ㅏ",      # IPA /ʌ/ → /a/
    "ɜ":    "ㅓ",      # IPA /ɜ/ → /eo/
    "ɜː":   "ㅓ",      # long /ɜ/ → /eo/
    "æ":    "ㅐ",      # IPA /æ/ → /ae/
    "ɯ":    "ㅡ",      # IPA /ɯ/ → /eu/ (closed back unrounded)
    "ɨ":    "ㅡ",      # U+0268 중앙모음고 → /eu/ ★ KEY!

    # ── 辅音 (Consonants / 자음) ──────────────────────────────────
    # 塞音 (Stops / 파열음)
    "p":    "ㅂ",      # IPA /p/
    "pʰ":   "ㅍ",      # aspirated /p'/
    "b":    "ㅂ",      # IPA /b/ → /p/ (Korean no /b/)
    "t":    "ㄷ",      # IPA /t/
    "t̚":    "ㄷ",      # U+0074+031A unreleased /t/ → /t/ ★ KEY!
    "tʰ":   "ㅌ",      # aspirated /t'/
    "d":    "ㄷ",      # IPA /d/ → /t/
    "k":    "ㄱ",      # IPA /k/
    "kʰ":   "ㅋ",      # aspirated /k'/
    "g":    "ㄱ",      # IPA /g/ → /k/
    "ɡ":    "ㄱ",      # IPA /ɡ/ (Unicode g) → /k/

    # 摩擦音 (Fricatives / 마찰음)
    "f":    "ㅍ",      # IPA /f/ → /p'/
    "v":    "ㅂ",      # IPA /v/ → /p/
    "θ":    "ㄷ",      # IPA /θ/ → /t/
    "ð":    "ㄷ",      # IPA /ð/ → /t/
    "s":    "ㅅ",      # IPA /s/
    "sʰ":   "ㅆ",      # U+0073+02B7 aspirated /s/ → /ss/ ★ KEY!
    "ʃ":    "ㅅ",      # IPA /ʃ/ → /s/
    "z":    "ㅊ",      # IPA /z/ → /s/ or /j/
    "ʒ":    "ㅊ",      # IPA /ʒ/ → /s/ or /j/
    "h":    "ㅎ",      # IPA /h/

    # 塞擦音 (Affricates / 파찰음)
    "tʃ":   "ㅊ",      # IPA /tʃ/ → /ch/
    "dʒ":   "ㅈ",      # IPA /dʒ/ → /j/
    "ts":   "ㄷ",      # IPA /ts/ → /t/ + /s/
    "dz":   "ㅈ",      # IPA /dz/ → /j/

    # 鼻音 (Nasals / 비음)
    "m":    "ㅁ",      # IPA /m/
    "n":    "ㄴ",      # IPA /n/
    "ŋ":    "ㅇ",      # U+014B velar nasal → /ng/ ★ KEY!
    "ɲ":    "ㄴ",      # U+0272 palatal nasal → /n/ ★ KEY!
    "ɴ":    "ㅇ",      # U+0274 velarized nasal → /ng/

    # 液音 (Liquids / 유음)
    "l":    "ㄹ",      # IPA /l/
    "r":    "ㄹ",      # IPA /r/ → /l/ (Korean /r/l/ are allophones)
    "ɾ":    "ㄹ",      # U+027E flap → /l/ ★ KEY!
    "ɭ":    "ㄹ",      # U+026D retroflex approximant → /l/ ★ KEY!
    "ɹ":    "ㄹ",      # IPA /ɹ/ (approximant) → /l/

    # 半元音 (Semivowels / 반모음)
    "w":    "ㅇ",      # IPA /w/ → initial /zero/ + /o/
    "j":    "ㅇ",      # IPA /j/ → initial /zero/ + /i/
    "y":    "ㅇ",      # IPA /y/ → /j/

    # 特殊 (Special)
    "ʔ":    "",        # glottal stop → omit
    
    # ── 已是韩文字母的直通 (Pass-through) ─────────────────────
    "ㄱ": "ㄱ",  "ㄴ": "ㄴ",  "ㄷ": "ㄷ",  "ㄹ": "ㄹ",
    "ㅁ": "ㅁ",  "ㅂ": "ㅂ",  "ㅅ": "ㅅ",  "ㅇ": "ㅇ",
    "ㅈ": "ㅈ",  "ㅉ": "ㅉ",  "ㅊ": "ㅊ",  "ㅋ": "ㅋ",
    "ㅌ": "ㅌ",  "ㅍ": "ㅍ",  "ㅎ": "ㅎ",
    "ㅏ": "ㅏ",  "ㅑ": "ㅑ",  "ㅓ": "ㅓ",  "ㅕ": "ㅕ",
    "ㅗ": "ㅗ",  "ㅜ": "ㅜ",  "ㅠ": "ㅠ",  "ㅡ": "ㅡ",
    "ㅢ": "ㅢ",  "ㅝ": "ㅝ",  "ㅞ": "ㅞ",  "ㅙ": "ㅙ",
    "ㅚ": "ㅚ",  "ㅘ": "ㅘ",
}


# ══════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════

def convert_phoneme(phoneme: str, language: str) -> Optional[str]:
    """
    Convert a single MFA phoneme string to the target format.

    Returns:
      str   – converted phoneme (may equal the input for pass-through languages)
      ""    – phoneme that should be omitted (e.g. glottal stop ʔ)
      None  – silence / boundary token; caller should skip this interval
    """
    # Silence / boundary → skip
    if phoneme in SILENCE_TOKENS:
        return None

    lang = language.lower()

    # ── Japanese → Romaji ────────────────────────────────────
    if lang in ("ja", "jpn"):
        result = JA_IPA_TO_ROMAJI.get(phoneme)
        if result is None:
            logger.warning(
                "Japanese phoneme not in mapping: %r (U+%s) – passing through",
                phoneme,
                " U+".join(f"{ord(c):04X}" for c in phoneme),
            )
            return phoneme
        return result if result != "" else None

    # ── English → ARPAbet ────────────────────────────────────
    if lang in ("en", "eng"):
        result = EN_IPA_TO_ARPABET.get(phoneme)
        if result is None:
            logger.warning(
                "English phoneme not in mapping: %r – using lowercase passthrough",
                phoneme,
            )
            return phoneme.lower()
        return result if result != "" else None

    # ── Korean (IPA) → Hangul Jamo ★ NEW! ────────────────────
    if lang in ("ko", "kor"):
        result = KO_IPA_TO_JAMO.get(phoneme)
        if result is None:
            logger.warning(
                "Korean phoneme not in mapping: %r (U+%s) – passing through as-is",
                phoneme,
                " U+".join(f"{ord(c):04X}" for c in phoneme),
            )
            return phoneme  # Pass through unknown phonemes
        return result if result != "" else None

    # ── Chinese (Pinyin), Cantonese (Jyutping) ──
    # MFA dictionaries for these languages already emit the target
    # phoneme symbols; no conversion needed.
    return phoneme


def convert_phoneme_list(
    phonemes: list[str],
    language: str,
    drop_silence: bool = True,
) -> list[str]:
    """
    Convert a list of MFA phonemes to the target format.

    Args:
      phonemes:      Raw phoneme tokens from MFA output.
      language:      ISO 639-1/3 language code (ja, jpn, en, eng, zh, cmn, ko, kor, …).
      drop_silence:  If True, silence/boundary/omitted tokens are excluded.

    Returns:
      List of converted phoneme strings.
    """
    out: list[str] = []
    for ph in phonemes:
        converted = convert_phoneme(ph, language)
        if converted is None:          # silence or omit
            if not drop_silence:
                out.append("sp")       # replace silence with explicit token
            continue
        out.append(converted)
    return out


def debug_unknown_phonemes(
    phonemes: list[str],
    language: str,
) -> list[str]:
    """
    Return a list of phonemes that have no explicit mapping.
    Useful for auditing MFA output from a new model/language.
    """
    lang = language.lower()
    if lang in ("ja", "jpn"):
        table = JA_IPA_TO_ROMAJI
    elif lang in ("en", "eng"):
        table = EN_IPA_TO_ARPABET
    elif lang in ("ko", "kor"):
        table = KO_IPA_TO_JAMO
    else:
        return []
    
    return [
        ph for ph in phonemes
        if ph not in SILENCE_TOKENS and ph not in table
    ]


# ══════════════════════════════════════════════════════════════
# LAB 文件转换函数
# ══════════════════════════════════════════════════════════════

def convert_lab_file(
    lab_content: str,
    language: str = "ja",
) -> str:
    """
    Convert entire LAB file from IPA to target phoneme format.
    
    LAB format: start_time end_time phoneme
    (times in 100-nanosecond units, as per MFA TextGrid)
    
    Args:
      lab_content:  Raw LAB file content (multiline, space-separated)
      language:     Target language ('ja', 'en', 'zh', 'ko', etc.)
    
    Returns:
      Converted LAB content
    """
    lines = lab_content.strip().split('\n')
    result_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        parts = line.split()
        if len(parts) < 3:
            result_lines.append(line)
            continue
        
        start_time = parts[0]
        end_time = parts[1]
        phoneme = parts[2]
        
        # Convert phoneme
        converted = convert_phoneme(phoneme, language)
        
        if converted is None:
            continue
        elif converted == "":
            continue
        else:
            result_lines.append(f"{start_time} {end_time} {converted}")
    
    return "\n".join(result_lines)
