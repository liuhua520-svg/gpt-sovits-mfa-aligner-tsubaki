# -*- coding: utf-8 -*-
"""
phoneme_converter.py  v2.0
Converts MFA output phonemes (IPA) to target phoneme sets for GPT-SoVITS.

Supported conversions:
  ja / jpn  →  Hiragana   (MFA japanese_mfa IPA → romaji → hiragana + merge)
  en / eng  →  ARPAbet    (MFA english_us_mfa, with diacritic normalization)
  zh / cmn  →  Pinyin     (MFA mandarin_china_mfa, pass-through)
  ko / kor  →  Hangul Jamo(MFA korean_mfa, IPA → Jamo)
  yue       →  Jyutping   (MFA mandarin_china_mfa, pass-through)

New in v2.0:
  - EN: diacritic-strip fallback (d̪→d, pʲ→p, tʰ→t, kʰ→k, ɝ→er, ʉː→uw …)
  - JA: JA_CV_TO_HIRAGANA table + build_ja_hiragana_lab() converter
  - ALL: merge_lab_silence() post-processor (absorb / delete '-' segments)
"""

from __future__ import annotations
import unicodedata
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# Silence / boundary tokens (language-agnostic)
# ══════════════════════════════════════════════════════════════

SILENCE_TOKENS: set[str] = {
    "", "sp", "sil", "spn", "SIL", "SP", "SPN",
    "<eps>", "<unk>", "<UNK>",
}


# ══════════════════════════════════════════════════════════════
# Japanese  MFA IPA  →  Romaji  (individual phoneme mapping)
# ══════════════════════════════════════════════════════════════

JA_IPA_TO_ROMAJI: dict[str, str] = {
    # ── 元音 (Vowels) ────────────────────────────────────────
    "a":  "a",
    "e":  "e",
    "i":  "i",
    "o":  "o",
    "ɯ":  "u",   # U+026F  闭后不圆元音 (日语 /u/)
    "ɯ̥":  "u",   # U+026F + U+0325  清后元音
    "i̥":  "i",   # U+0069 + U+0325  清i
    "ɨ̥":  "u",   # U+0268 + U+0325  清中央元音

    # ── 塞音 (Stops) ──────────────────────────────────────────
    "p":  "p",   "b":  "b",
    "t":  "t",   "d":  "d",
    "k":  "k",
    "g":  "g",   "ɡ":  "g",  # ASCII g / IPA g
    "c":  "k",               # 清硬腭塞音
    "ʔ":  "",                # 声门塞音 → 省略

    # ── 摩擦音 (Fricatives) ───────────────────────────────────
    "s":  "s",   "z":  "z",   "h":  "h",
    "ɸ":  "f",   "f":  "f",   "v":  "v",
    "ɕ":  "sh",              # U+0255  し行
    "ʑ":  "j",               # U+0291  じ行

    # ── 塞擦音 (Affricates) ───────────────────────────────────
    "ts":  "ts",
    "dz":  "dz",
    "tɕ":  "ch",             # ち行
    "dʑ":  "j",              # じ行
    "ch":  "ch",
    "sh":  "sh",

    # ── 鼻音 (Nasals) ─────────────────────────────────────────
    "n":  "n",   "m":  "m",
    "ŋ":  "ng",              # U+014B 軟口蓋鼻音
    "ng": "ng",
    "ɲ":  "ny",              # U+0272 硬口蓋鼻音
    "ɴ":  "N",               # U+0274 拨音 mora nasal
    "N":  "N",               # 拨音 (ん)

    # ── 液音 (Liquids) ────────────────────────────────────────
    "r":  "r",   "ɾ":  "r",  # U+027E 齿龈轻拍音

    # ── 半元音/滑音 (Semivowels) ──────────────────────────────
    "w":  "w",
    "j":  "y",   "y":  "y",

    # ── 软腭化辅音 (Palatalized Consonants) ──────────────────
    "ky": "ky", "gy": "gy", "ny": "ny", "hy": "hy",
    "my": "my", "ry": "ry", "py": "py", "by": "by",
    "ty": "ty", "dy": "dy",
}


# ══════════════════════════════════════════════════════════════
# Japanese Romaji C+V  →  Hiragana  (look-up table)
# ══════════════════════════════════════════════════════════════
# Key = consonant-onset (romaji) + vowel (a/i/u/e/o)
# Standalone vowels are also included (key = vowel only).

JA_CV_TO_HIRAGANA: dict[str, str] = {
    # ── 単独元音 (Standalone Vowels) ─────────────────────────
    "a": "あ", "i": "い", "u": "う", "e": "え", "o": "お",

    # ── か行 ─────────────────────────────────────────────────
    "ka": "か", "ki": "き", "ku": "く", "ke": "け", "ko": "こ",
    "ga": "が", "gi": "ぎ", "gu": "ぐ", "ge": "げ", "go": "ご",

    # ── さ行 ─────────────────────────────────────────────────
    "sa": "さ", "si": "し", "su": "す", "se": "せ", "so": "そ",
    "sha": "しゃ", "shi": "し", "shu": "しゅ", "she": "しぇ", "sho": "しょ",
    "sya": "しゃ", "syu": "しゅ", "syo": "しょ",

    # ── ざ行 ─────────────────────────────────────────────────
    "za": "ざ", "zi": "じ", "zu": "ず", "ze": "ぜ", "zo": "ぞ",
    "ja": "じゃ", "ji": "じ", "ju": "じゅ", "je": "じぇ", "jo": "じょ",
    "dza": "ざ", "dzi": "じ", "dzu": "づ", "dze": "ぜ", "dzo": "ぞ",

    # ── た行 ─────────────────────────────────────────────────
    "ta": "た", "ti": "ち", "tu": "つ", "te": "て", "to": "と",
    "cha": "ちゃ", "chi": "ち", "chu": "ちゅ", "che": "ちぇ", "cho": "ちょ",
    "tya": "ちゃ", "tyi": "ち", "tyu": "ちゅ", "tyo": "ちょ",
    "tsa": "つぁ", "tsi": "つぃ", "tsu": "つ", "tse": "つぇ", "tso": "つぉ",

    # ── だ行 ─────────────────────────────────────────────────
    "da": "だ", "di": "ぢ", "du": "づ", "de": "で", "do": "ど",

    # ── な行 ─────────────────────────────────────────────────
    "na": "な", "ni": "に", "nu": "ぬ", "ne": "ね", "no": "の",
    "nya": "にゃ", "nyi": "に",  "nyu": "にゅ", "nyo": "にょ",

    # ── は行 ─────────────────────────────────────────────────
    "ha": "は", "hi": "ひ", "hu": "ふ", "he": "へ", "ho": "ほ",
    "hya": "ひゃ", "hyu": "ひゅ", "hyo": "ひょ",
    "fa": "ふぁ", "fi": "ふぃ", "fu": "ふ", "fe": "ふぇ", "fo": "ふぉ",

    # ── ば行 ─────────────────────────────────────────────────
    "ba": "ば", "bi": "び", "bu": "ぶ", "be": "べ", "bo": "ぼ",
    "bya": "びゃ", "byu": "びゅ", "byo": "びょ",

    # ── ぱ行 ─────────────────────────────────────────────────
    "pa": "ぱ", "pi": "ぴ", "pu": "ぷ", "pe": "ぺ", "po": "ぽ",
    "pya": "ぴゃ", "pyu": "ぴゅ", "pyo": "ぴょ",

    # ── ま行 ─────────────────────────────────────────────────
    "ma": "ま", "mi": "み", "mu": "む", "me": "め", "mo": "も",
    "mya": "みゃ", "myu": "みゅ", "myo": "みょ",

    # ── や行 ─────────────────────────────────────────────────
    "ya": "や", "yu": "ゆ", "yo": "よ",

    # ── ら行 ─────────────────────────────────────────────────
    "ra": "ら", "ri": "り", "ru": "る", "re": "れ", "ro": "ろ",
    "rya": "りゃ", "ryu": "りゅ", "ryo": "りょ",

    # ── わ行 ─────────────────────────────────────────────────
    "wa": "わ", "wi": "ゐ", "we": "ゑ", "wo": "を",

    # ── き行拗音 ─────────────────────────────────────────────
    "kya": "きゃ", "kyu": "きゅ", "kyo": "きょ",
    "gya": "ぎゃ", "gyu": "ぎゅ", "gyo": "ぎょ",
}

# Romaji vowels in Japanese phoneme context
JA_VOWELS: frozenset[str] = frozenset({"a", "i", "u", "e", "o"})

# Phonemes that act as mora nasals (ん) when before a consonant or at phrase end
JA_MORA_NASAL_PHONEMES: frozenset[str] = frozenset({"N", "ɴ"})

# Phonemes that are nasal consonants and need look-ahead to decide:
# before vowel → onset; before consonant → ん
JA_AMBIGUOUS_NASALS: frozenset[str] = frozenset({"n", "m", "ng", "ny"})


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
    "ɜː": "er",  "ɜ":  "er",
    "ɚ":  "er",               # r-colored schwa
    "ɝ":  "er",               # ★ U+025D stressed rhotacized mid-central vowel
    "ə":  "ax",
    "aɪ": "ay",  "aj": "ay", # ★ aj = IPA notation for /aɪ/
    "aʊ": "aw",
    "ɔɪ": "oy",
    "ʉ":  "uw",               # ★ U+0289 close central rounded
    "ʉː": "uw",               # ★ long variant

    # ── 元音 ARPAbet pass-through ────────────────────────────────
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

    # ── 辅音 ARPAbet pass-through ────────────────────────────────
    "ch": "ch", "dh": "dh", "dx": "dx",
    "dr": "dr", "tr": "tr",
    "hh": "hh", "jh": "jh",
    "ng": "ng", "sh": "sh", "th": "th", "zh": "zh",
}


def _strip_en_diacritics(ph: str) -> str:
    """
    Remove Unicode combining diacritics (category Mn/Mc) and modifier
    letters (category Lm) for English phoneme fallback lookup.

    Examples
    --------
    d̪  (d + U+032A COMBINING BRIDGE BELOW) → d
    pʲ (p + U+02B2 MODIFIER LETTER SMALL J) → p
    tʰ (t + U+02B0 MODIFIER LETTER SMALL H) → t
    kʰ → k,  sʲ → s,  fʲ → f,  vʲ → v
    ʉː (U+0289 + U+02D0 LENGTH MARK)        → ʉ  (still needs table entry)
    """
    nfd = unicodedata.normalize("NFD", ph)
    return "".join(
        c for c in nfd
        if unicodedata.category(c) not in ("Mn", "Mc", "Lm")
    )


# ══════════════════════════════════════════════════════════════
# Korean IPA → Hangul Jamo
# ══════════════════════════════════════════════════════════════

KO_IPA_TO_JAMO: dict[str, str] = {
    # ── 元音 (Vowels / 모음) ──────────────────────────────────────
    "a": "ㅏ", "ɐ": "ㅏ", "ɑ": "ㅏ", "ɑː": "ㅏ",
    "ə": "ㅗ", "ɛ": "ㅓ", "e": "ㅔ",
    "i": "ㅣ", "ɪ": "ㅣ", "iː": "ㅣ",
    "o": "ㅗ", "ɔ": "ㅗ", "ɔː": "ㅗ",
    "u": "ㅜ", "ʊ": "ㅜ", "uː": "ㅜ",
    "ʌ": "ㅏ",
    "ɜ": "ㅓ", "ɜː": "ㅓ",
    "æ": "ㅐ",
    "ɯ": "ㅡ", "ɨ": "ㅡ",

    # ── 塞音 (Stops / 파열음) ────────────────────────────────────
    "p": "ㅂ", "pʰ": "ㅍ", "b": "ㅂ",
    "t": "ㄷ", "t̚": "ㄷ", "tʰ": "ㅌ", "d": "ㄷ",
    "k": "ㄱ", "kʰ": "ㅋ", "g": "ㄱ", "ɡ": "ㄱ",

    # ── 摩擦音 (Fricatives / 마찰음) ────────────────────────────
    "f": "ㅍ", "v": "ㅂ", "θ": "ㄷ", "ð": "ㄷ",
    "s": "ㅅ", "sʰ": "ㅆ", "ʃ": "ㅅ",
    "z": "ㅊ", "ʒ": "ㅊ", "h": "ㅎ",

    # ── 塞擦音 (Affricates / 파찰음) ────────────────────────────
    "tʃ": "ㅊ", "dʒ": "ㅈ", "ts": "ㄷ", "dz": "ㅈ",

    # ── 鼻音 (Nasals / 비음) ────────────────────────────────────
    "m": "ㅁ", "n": "ㄴ", "ŋ": "ㅇ", "ɲ": "ㄴ", "ɴ": "ㅇ",

    # ── 液音 / 半元音 (Liquids / Semivowels) ────────────────────
    "l": "ㄹ", "r": "ㄹ", "ɾ": "ㄹ", "ɭ": "ㄹ", "ɹ": "ㄹ",
    "w": "ㅇ", "j": "ㅇ", "y": "ㅇ",

    # ── 特殊 ─────────────────────────────────────────────────────
    "ʔ": "",

    # ── 韩文字母直通 (Pass-through) ──────────────────────────────
    "ㄱ": "ㄱ", "ㄴ": "ㄴ", "ㄷ": "ㄷ", "ㄹ": "ㄹ",
    "ㅁ": "ㅁ", "ㅂ": "ㅂ", "ㅅ": "ㅅ", "ㅇ": "ㅇ",
    "ㅈ": "ㅈ", "ㅉ": "ㅉ", "ㅊ": "ㅊ", "ㅋ": "ㅋ",
    "ㅌ": "ㅌ", "ㅍ": "ㅍ", "ㅎ": "ㅎ",
    "ㅏ": "ㅏ", "ㅑ": "ㅑ", "ㅓ": "ㅓ", "ㅕ": "ㅕ",
    "ㅗ": "ㅗ", "ㅜ": "ㅜ", "ㅠ": "ㅠ", "ㅡ": "ㅡ",
    "ㅢ": "ㅢ", "ㅝ": "ㅝ", "ㅞ": "ㅞ", "ㅙ": "ㅙ",
    "ㅚ": "ㅚ", "ㅘ": "ㅘ",
}


# ══════════════════════════════════════════════════════════════
# Public API — single-phoneme conversion
# ══════════════════════════════════════════════════════════════

def convert_phoneme(phoneme: str, language: str) -> Optional[str]:
    """
    Convert a single MFA phoneme string to the target format.

    Returns
    -------
    str   converted phoneme (may equal input for pass-through languages)
    ""    phoneme should be omitted (e.g. glottal stop ʔ)
    None  silence / boundary token; caller should skip this interval
    """
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

    # ── English → ARPAbet (with diacritic-strip fallback) ────
    if lang in ("en", "eng"):
        result = EN_IPA_TO_ARPABET.get(phoneme)
        if result is not None:
            return result if result != "" else None

        # Fallback: strip combining diacritics / modifier letters
        stripped = _strip_en_diacritics(phoneme)
        if stripped != phoneme:
            result = EN_IPA_TO_ARPABET.get(stripped)
            if result is not None:
                logger.debug(
                    "EN diacritic-strip: %r → %r → %r", phoneme, stripped, result
                )
                return result if result != "" else None

        # Still unknown → lowercase passthrough
        logger.warning("English phoneme not in mapping: %r – lowercase passthrough", phoneme)
        return phoneme.lower()

    # ── Korean (IPA) → Hangul Jamo ────────────────────────────
    if lang in ("ko", "kor"):
        result = KO_IPA_TO_JAMO.get(phoneme)
        if result is None:
            logger.warning(
                "Korean phoneme not in mapping: %r (U+%s) – passing through",
                phoneme,
                " U+".join(f"{ord(c):04X}" for c in phoneme),
            )
            return phoneme
        return result if result != "" else None

    # ── Chinese (Pinyin), Cantonese (Jyutping) – pass-through ─
    return phoneme


def convert_phoneme_list(
    phonemes: list[str],
    language: str,
    drop_silence: bool = True,
) -> list[str]:
    """Convert a list of MFA phonemes to the target format."""
    out: list[str] = []
    for ph in phonemes:
        converted = convert_phoneme(ph, language)
        if converted is None:
            if not drop_silence:
                out.append("sp")
            continue
        out.append(converted)
    return out


def debug_unknown_phonemes(phonemes: list[str], language: str) -> list[str]:
    """Return phonemes with no explicit mapping (useful for auditing new models)."""
    lang = language.lower()
    if lang in ("ja", "jpn"):
        table = JA_IPA_TO_ROMAJI
    elif lang in ("en", "eng"):
        table = EN_IPA_TO_ARPABET
    elif lang in ("ko", "kor"):
        table = KO_IPA_TO_JAMO
    else:
        return []
    return [ph for ph in phonemes if ph not in SILENCE_TOKENS and ph not in table]


# ══════════════════════════════════════════════════════════════
# Japanese Hiragana LAB builder
# ══════════════════════════════════════════════════════════════

def build_ja_hiragana_lab(
    entries: list[tuple[int, int, str]],
) -> list[tuple[int, int, str]]:
    """
    Convert a sequence of individual romaji phoneme segments into hiragana
    LAB segments, inserting '-' markers for consonant onsets.

    Algorithm
    ---------
    • Vowel                       → hiragana character (or C+V if preceded by
                                     a pending consonant onset)
    • Hard mora nasal (N / ɴ)    → ん  (always)
    • Ambiguous nasal (n / m / ng) before consonant or at end → ん
    • Ambiguous nasal before vowel → treated as consonant onset (pending)
    • Other consonant             → '-'  (pending; combines with next vowel)

    Parameters
    ----------
    entries : list of (start_100ns, end_100ns, romaji_phoneme)

    Returns
    -------
    list of (start_100ns, end_100ns, hiragana_or_dash)
    """
    result: list[tuple[int, int, str]] = []
    n = len(entries)
    pending: tuple[int, int, str] | None = None  # (start, end, romaji_consonant)

    def flush_pending_as_dash() -> None:
        nonlocal pending
        if pending is not None:
            result.append((pending[0], pending[1], "-"))
            pending = None

    i = 0
    while i < n:
        start, end, ph = entries[i]

        # ── Vowel ────────────────────────────────────────────
        if ph in JA_VOWELS:
            if pending is not None:
                cv = pending[2] + ph
                hiragana = JA_CV_TO_HIRAGANA.get(cv) or JA_CV_TO_HIRAGANA.get(ph, ph)
                # consonant segment → '-'
                result.append((pending[0], pending[1], "-"))
                pending = None
            else:
                hiragana = JA_CV_TO_HIRAGANA.get(ph, ph)
            result.append((start, end, hiragana))
            i += 1
            continue

        # ── Hard mora nasal (N / ɴ) ───────────────────────────
        if ph in JA_MORA_NASAL_PHONEMES:
            flush_pending_as_dash()
            result.append((start, end, "ん"))
            i += 1
            continue

        # ── Ambiguous nasal (n / m / ng / ny) ────────────────
        if ph in JA_AMBIGUOUS_NASALS:
            # Look ahead: if followed by a vowel, treat as consonant onset
            next_ph = entries[i + 1][2] if i + 1 < n else None
            is_mora = (next_ph is None) or (next_ph not in JA_VOWELS)
            if is_mora:
                flush_pending_as_dash()
                result.append((start, end, "ん"))
            else:
                # Onset: may combine with next vowel (e.g. n+a = な)
                flush_pending_as_dash()
                pending = (start, end, ph)
            i += 1
            continue

        # ── Other consonant ────────────────────────────────────
        flush_pending_as_dash()
        pending = (start, end, ph)
        i += 1

    # Trailing consonant with no following vowel
    flush_pending_as_dash()
    return result


# ══════════════════════════════════════════════════════════════
# Universal LAB silence merger
# ══════════════════════════════════════════════════════════════

def merge_lab_silence(
    entries: list[tuple[int, int, str]],
    max_gap_100ns: int = 500_000,   # 50 ms — gap larger than this = phrase boundary
) -> list[tuple[int, int, str]]:
    """
    Post-process a LAB segment list to handle '-' (consonant onset) markers.

    Rules
    -----
    1. '-' that starts ≤ max_gap_100ns after a voiced (non-sil, non-'-') segment
       → absorbed into that segment (extend its end time).
    2. '-' at the start, following 'sil' / another '-', OR across a phrase gap
       (gap > max_gap_100ns) → deleted entirely.

    'sil' segments are never modified or deleted.

    Cross-phrase gap fix
    --------------------
    Without the gap check, a '-' that begins a new phrase (e.g. Korean 요····[-]무)
    would incorrectly extend the previous phrase's last syllable across the silence.
    The gap threshold (default 50 ms) prevents this: within a syllable the '-' is
    always adjacent (gap ≈ 0), while inter-phrase silences are typically ≥ 200 ms.

    Examples (Chinese)
    ------------------
    [sil, -, hai, -, da, -, jia]
        sil → kept
        - after sil → deleted
        hai → kept
        - after hai → absorbed (hai extends)
        da → kept
        - after da → absorbed (da extends)
        jia → kept
    Result: [sil, hai(extended), da(extended), jia]

    Examples (Korean, two phrases with 200 ms gap)
    ------------------------------------------------
    [-, 안, -, 녕, …, 요, GAP 200ms, -, 무, …]
        leading '-'      → deleted (result empty)
        - after 안       → absorbed (안 extends)
        - after 녕       → absorbed (녕 extends)
        …
        - after GAP 200ms → gap > 50 ms → deleted (not merged into 요)
        무               → kept
    """
    result: list[tuple[int, int, str]] = []
    for seg_start, seg_end, label in entries:
        if label == "-":
            if (result
                    and result[-1][2] not in ("-", "sil")
                    and seg_start - result[-1][1] <= max_gap_100ns):
                # Adjacent enough to previous voiced segment → absorb
                ps, _, pl = result[-1]
                result[-1] = (ps, seg_end, pl)
            # else: delete (no nearby voiced predecessor, or phrase gap)
        else:
            result.append((seg_start, seg_end, label))
    return result


# ══════════════════════════════════════════════════════════════
# LAB file-level conversion (utility, called from tests / CLI)
# ══════════════════════════════════════════════════════════════

def convert_lab_file(lab_content: str, language: str = "ja") -> str:
    """
    Convert an entire LAB file's phoneme column to the target format.
    Applies hiragana grouping for Japanese and silence merging for all languages.

    LAB format: start_time end_time phoneme  (one entry per line)
    Times are in 100-nanosecond units.
    """
    lines = lab_content.strip().split("\n")
    raw: list[tuple[int, int, str]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        raw.append((int(parts[0]), int(parts[1]), parts[2]))

    lang = language.lower()
    converted: list[tuple[int, int, str]] = []

    if lang in ("ja", "jpn"):
        # Phase 1: IPA → romaji per phoneme
        romaji_entries: list[tuple[int, int, str]] = []
        for start, end, ph in raw:
            r = convert_phoneme(ph, "ja")
            if r is not None and r != "":
                romaji_entries.append((start, end, r))
        # Phase 2: romaji → hiragana + '-' for consonant onsets
        converted = build_ja_hiragana_lab(romaji_entries)
    else:
        for start, end, ph in raw:
            c = convert_phoneme(ph, language)
            if c is None or c == "":
                continue
            converted.append((start, end, c))

    # Phase 3: merge '-' segments for all languages
    merged = merge_lab_silence(converted)
    return "\n".join(f"{s} {e} {p}" for s, e, p in merged)


# ══════════════════════════════════════════════════════════════
# Hiragana ↔ Katakana
# ══════════════════════════════════════════════════════════════

_HIRAGANA_START = 0x3041  # ぁ
_HIRAGANA_END   = 0x3096  # ゖ
_HIRA_TO_KATA   = 0x60   # codepoint offset to katakana block


def hiragana_to_katakana(text: str) -> str:
    """Convert hiragana characters to katakana (one-to-one codepoint shift)."""
    return "".join(
        chr(ord(c) + _HIRA_TO_KATA)
        if _HIRAGANA_START <= ord(c) <= _HIRAGANA_END
        else c
        for c in text
    )


# ══════════════════════════════════════════════════════════════
# Japanese merged C+V LAB builder  (合并辅音 / 平假名 / 片假名)
# ══════════════════════════════════════════════════════════════

def build_ja_merged_lab(
    entries: list[tuple[int, int, str]],
    output: str = "romaji",
) -> list[tuple[int, int, str]]:
    """
    Merge consecutive consonant-onset + vowel segments into a single entry
    whose time span covers *both* the consonant and the vowel.

    Unlike build_ja_hiragana_lab(), which emits a '-' note for the consonant
    and a separate note for the vowel, this function produces **one** merged
    entry: (consonant_start, vowel_end, label).

    Parameters
    ----------
    entries : list of (start_100ns, end_100ns, romaji_phoneme)
              Input must already be in romaji (IPA → romaji conversion should
              be done beforehand via convert_phoneme(ph, 'ja')).
    output  :
        'romaji'   → keep as merged romaji string  (e.g. 'sa', 'N', 'pu', 'ru')
        'hiragana' → convert to hiragana            (e.g. 'さ', 'ん', 'ぷ', 'る')
        'katakana' → convert to katakana            (e.g. 'サ', 'ン', 'プ', 'ル')

    Returns
    -------
    list of (start_100ns, end_100ns, label)

    Examples
    --------
    Input (romaji):
        (50000,  1450000, 's')
        (1450000, 2200000, 'a')
        (2200000, 3050000, 'N')
        (3050000, 3800000, 'p')
        (3800000, 4750000, 'u')
        (4750000, 5250000, 'r')
        (5250000, 7000000, 'u')

    output='romaji'   → [(50000,2200000,'sa'), (2200000,3050000,'N'),
                          (3050000,4750000,'pu'), (4750000,7000000,'ru')]
    output='hiragana' → [(50000,2200000,'さ'), (2200000,3050000,'ん'),
                          (3050000,4750000,'ぷ'), (4750000,7000000,'る')]
    output='katakana' → [(50000,2200000,'サ'), (2200000,3050000,'ン'),
                          (3050000,4750000,'プ'), (4750000,7000000,'ル')]
    """

    def _mora_label() -> str:
        if output == "hiragana":
            return "ん"
        if output == "katakana":
            return "ン"
        return "N"

    def _cv_label(cv: str) -> str:
        """Convert a C+V romaji string to the target script."""
        if output in ("hiragana", "katakana"):
            # Try full CV key first, then fall back to vowel-only
            hira = (
                JA_CV_TO_HIRAGANA.get(cv)
                or JA_CV_TO_HIRAGANA.get(cv[-1:], cv)
            )
            return hiragana_to_katakana(hira) if output == "katakana" else hira
        return cv  # romaji passthrough

    result: list[tuple[int, int, str]] = []
    n = len(entries)
    # pending = (start_100ns, end_100ns, romaji_consonant) awaiting a vowel
    pending: tuple[int, int, str] | None = None

    i = 0
    while i < n:
        start, end, ph = entries[i]

        # ── Vowel ────────────────────────────────────────────
        if ph in JA_VOWELS:
            if pending is not None:
                cv = pending[2] + ph
                # Merged entry: consonant_start → vowel_end
                result.append((pending[0], end, _cv_label(cv)))
                pending = None
            else:
                result.append((start, end, _cv_label(ph)))
            i += 1
            continue

        # ── Hard mora nasal (N / ɴ) ── always becomes ん/ン/N ──────────
        if ph in JA_MORA_NASAL_PHONEMES:
            if pending is not None:
                # Flush pending consonant without a vowel → '-'
                result.append((pending[0], pending[1], "-"))
                pending = None
            result.append((start, end, _mora_label()))
            i += 1
            continue

        # ── Ambiguous nasal (n / m / ng / ny) ────────────────────────
        if ph in JA_AMBIGUOUS_NASALS:
            next_ph = entries[i + 1][2] if i + 1 < n else None
            is_mora = (next_ph is None) or (next_ph not in JA_VOWELS)
            if is_mora:
                if pending is not None:
                    result.append((pending[0], pending[1], "-"))
                    pending = None
                result.append((start, end, _mora_label()))
            else:
                # Treat as consonant onset (will combine with next vowel)
                if pending is not None:
                    result.append((pending[0], pending[1], "-"))
                pending = (start, end, ph)
            i += 1
            continue

        # ── Other consonant ────────────────────────────────────────────
        if pending is not None:
            result.append((pending[0], pending[1], "-"))
        pending = (start, end, ph)
        i += 1

    # Trailing consonant with no following vowel → '-'
    if pending is not None:
        result.append((pending[0], pending[1], "-"))

    return result


# ══════════════════════════════════════════════════════════════
# Universal phoneme-mode transformer for LAB segment lists
# ══════════════════════════════════════════════════════════════

#: Labels treated as silence; pass through untouched by the merge algorithm.
_MERGE_SILENCE: frozenset[str] = frozenset(
    {"sil", "pau", "sp", "spn", "br", "silence", "noise", "ap", "blank"}
)


def apply_phoneme_mode(
    segments: list[tuple[int, int, str]],
    mode: str,
) -> list[tuple[int, int, str]]:
    """
    Apply a phoneme-mode transformation to a flat list of LAB segments.

    Parameters
    ----------
    segments : list of (start_100ns, end_100ns, label)
               Labels are expected to be romaji phonemes (or already hiragana/
               mixed) – no IPA-to-romaji conversion is performed here.
    mode     :
        'none'     → return segments unchanged
        'merge'    → merge C+V pairs into romaji syllables (s+a → sa)
        'hiragana' → merge C+V and convert to hiragana     (s+a → さ)
        'katakana' → merge C+V and convert to katakana     (s+a → サ)

    Returns
    -------
    Transformed list of (start_100ns, end_100ns, label), sorted by start time.

    Notes
    -----
    • Silence segments (sil / pau / sp etc.) are passed through unchanged and
      act as phrase boundaries that break the merge context.
    • This function is designed for use in "project-only" mode where the user
      provides a raw MFA LAB file with individual romaji phonemes for Japanese.
    • For non-Japanese LAB files the merge algorithm may produce unexpected
      results; use mode='none' for Chinese / English / Korean LAB files.
    """
    if mode == "none":
        return list(segments)

    phoneme_output = "romaji" if mode == "merge" else mode

    # Split on silence boundaries, process each phoneme run with
    # build_ja_merged_lab(), then reassemble with silence in place.
    result: list[tuple[int, int, str]] = []
    phoneme_buf: list[tuple[int, int, str]] = []

    def _flush() -> None:
        if phoneme_buf:
            merged = build_ja_merged_lab(phoneme_buf, output=phoneme_output)
            result.extend(merged)
            phoneme_buf.clear()

    for seg in segments:
        label = seg[2]
        if label.strip().lower() in _MERGE_SILENCE:
            _flush()
            result.append(seg)
        else:
            phoneme_buf.append(seg)

    _flush()

    result.sort(key=lambda x: x[0])
    return result