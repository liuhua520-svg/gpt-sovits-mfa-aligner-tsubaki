# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Literal
import json
import re
import unicodedata

Language = Literal["auto", "zh", "en", "ja", "ko", "yue"]

SILENCE_TOKENS = {
    "", "-", "sp", "spn", "sil", "silence", "pau", "breath", "noise", "ap", "blank"
}

# 英语 ARPAbet 元音
EN_VOWELS = {
    "aa", "ae", "ah", "ao", "aw", "ax", "ay",
    "eh", "er", "ey", "ih", "iy", "ow", "oy", "uh", "uw",
}

# 英语常见 IPA / 变体 -> ARPAbet
EN_IPA_TO_ARPABET = {
    # vowels
    "iː": "iy",
    "i": "iy",
    "ɪ": "ih",
    "eɪ": "ey",
    "ɛ": "eh",
    "e": "eh",
    "æ": "ae",
    "ɑː": "aa",
    "ɑ": "aa",
    "ɔː": "ao",
    "ɔ": "ao",
    "oʊ": "ow",
    "ʊ": "uh",
    "uː": "uw",
    "u": "uw",
    "ʌ": "ah",
    "ɜː": "er",
    "ɜ": "er",
    "ɝ": "er",
    "ɚ": "er",
    "ə": "ax",
    "aɪ": "ay",
    "aʊ": "aw",
    "ɔɪ": "oy",
    "ɨ": "ih",
    "ʉ": "uw",
    "ʉː": "uw",

    # consonants
    "p": "p",
    "b": "b",
    "t": "t",
    "d": "d",
    "k": "k",
    "g": "g",
    "ɡ": "g",
    "f": "f",
    "v": "v",
    "θ": "th",
    "ð": "dh",
    "s": "s",
    "z": "z",
    "ʃ": "sh",
    "ʒ": "zh",
    "h": "hh",
    "ɦ": "hh",
    "tʃ": "ch",
    "dʒ": "jh",
    "m": "m",
    "n": "n",
    "ŋ": "ng",
    "l": "l",
    "ɹ": "r",
    "r": "r",
    "w": "w",
    "j": "y",
    "ʔ": "",

    # pass-through
    "ch": "ch",
    "dh": "dh",
    "dx": "dx",
    "dr": "dr",
    "tr": "tr",
    "hh": "hh",
    "jh": "jh",
    "ng": "ng",
    "sh": "sh",
    "th": "th",
    "zh": "zh",
}

# 日语表层音节常见元音感 token
GENERIC_VOWEL_CHARS = set("aeiouüvɑæɐɒɜɝɚɪʊʉəɛɔɯɨ")

JA_SYLLABIC_NASAL = {"ん", "N"}

LINE_RE = re.compile(r"^\s*(\d+)\s+(\d+)\s+(.+?)\s*$")
IPA_DIACRITIC_CHARS = set("ːʲʷʰ̪̟̠̚˞̩̯̃ˈˌˠˤˀˡˢ")


@dataclass
class LabRow:
    start: int
    end: int
    phone: str
    decision: str = ""


def strip_marks(token: str) -> str:
    token = unicodedata.normalize("NFC", token.strip())
    token = "".join(ch for ch in token if ch not in IPA_DIACRITIC_CHARS)
    return token


def normalize_en_phone(phone: str) -> str:
    p = strip_marks(phone).lower()
    if p in EN_IPA_TO_ARPABET:
        return EN_IPA_TO_ARPABET[p]

    # 去掉附着符号后再试一次
    p2 = re.sub(r"[^a-zɑæɐɒɜɝɚɪʊʉəɛɔŋθðʃʒ]", "", p)
    return EN_IPA_TO_ARPABET.get(p2, p2 or p)


def normalize_phone(phone: str, lang: str = "auto") -> str:
    p = phone.strip()
    if not p or p in SILENCE_TOKENS:
        return "-"
    if lang == "en":
        return normalize_en_phone(p)
    return p


def looks_like_syllable(token: str) -> bool:
    low = token.lower()
    if any(ch in GENERIC_VOWEL_CHARS for ch in low):
        return True
    return any(
        ("\u3040" <= ch <= "\u30ff") or ("\u4e00" <= ch <= "\u9fff")
        for ch in token
    )


def detect_lang(rows: List[LabRow]) -> str:
    phones = [r.phone.strip().lower() for r in rows if r.phone.strip()]
    if any(any(ch in "ぁあぃいぅうぇえぉおァアィイゥウェエォオ" for ch in p) for p in phones):
        return "ja"
    if any(
        p in EN_IPA_TO_ARPABET
        or any(ch in p for ch in ["ː", "ʲ", "ʷ", "ʰ", "̪", "ɝ", "ɚ", "ə", "ɪ", "ʊ", "θ", "ð", "ʃ", "ʒ"])
        for p in phones
    ):
        return "en"
    return "zh"


def parse_lab_text(text: str) -> List[LabRow]:
    rows: List[LabRow] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        m = LINE_RE.match(line)
        if not m:
            continue
        rows.append(LabRow(int(m.group(1)), int(m.group(2)), m.group(3).strip()))
    return rows


def _next_significant_phone(rows: List[LabRow], idx: int, lang: str) -> Optional[str]:
    for r in rows[idx + 1:]:
        raw = r.phone.strip()
        if not raw or raw in SILENCE_TOKENS:
            continue
        return normalize_phone(raw, lang)
    return None


def should_keep_and_surface(
    i: int,
    rows: List[LabRow],
    lang: str,
    norm_phone: str,
) -> Tuple[bool, str, str]:
    """
    返回：
      keep / decision / surface_phone

    规则：
    1. 纯辅音段 -> "-"
    2. 如果前面已经出现过元音/表层标签，则记为 merged_left
    3. 日语里把 coda 鼻音（m/n/ng/N）在合适上下文里转成 ん
    """
    if norm_phone == "-":
        return False, ("deleted" if i == 0 else "merged_left"), "-"

    # 英语：只保留元音；辅音统统变 "-"
    if lang == "en":
        if norm_phone in EN_VOWELS:
            return True, "keep", norm_phone
        return False, ("deleted" if i == 0 else "merged_left"), "-"

    # 日语：表层音节优先保留；纯辅音段删除
    if lang == "ja":
        low = norm_phone.lower()

        if low in JA_SYLLABIC_NASAL:
            return True, "keep_special", "ん"

        # coda 鼻音：尽量落成 ん
        if low in {"m", "n", "ng", "ɴ"}:
            nxt = _next_significant_phone(rows, i, lang)
            if nxt is None or not looks_like_syllable(nxt):
                return True, "keep_special", "ん"
            return False, ("deleted" if i == 0 else "merged_left"), "-"

        if looks_like_syllable(norm_phone):
            return True, "keep", norm_phone

        return False, ("deleted" if i == 0 else "merged_left"), "-"

    # 中文 / 其他：像表层音节就保留，否则变 "-"
    if looks_like_syllable(norm_phone):
        return True, "keep", norm_phone

    return False, ("deleted" if i == 0 else "merged_left"), "-"


def process_rows(rows: List[LabRow], lang: str = "auto") -> List[LabRow]:
    if lang == "auto":
        lang = detect_lang(rows)

    out: List[LabRow] = []
    for i, row in enumerate(rows):
        norm = normalize_phone(row.phone, lang)
        keep, decision, surface = should_keep_and_surface(i, rows, lang, norm)
        row.phone = surface if keep else "-"
        row.decision = decision
        out.append(row)
    return out


def format_rows(rows: List[LabRow]) -> str:
    return "\n".join(f"{r.start} {r.end} {r.phone}" for r in rows)


def process_lab_text(text: str, lang: str = "auto", write_meta: bool = False):
    rows = parse_lab_text(text)
    processed = process_rows(rows, lang=lang)
    out_text = format_rows(processed)

    if not write_meta:
        return out_text

    meta = [
        {
            "start": r.start,
            "end": r.end,
            "phone": r.phone,
            "decision": r.decision,
        }
        for r in processed
    ]
    return out_text, meta


def process_lab_file(
    input_path: str | Path,
    output_path: str | Path,
    lang: str = "auto",
    meta_path: str | Path | None = None,
) -> None:
    input_path = Path(input_path)
    output_path = Path(output_path)

    text = input_path.read_text(encoding="utf-8", errors="ignore")
    result = process_lab_text(text, lang=lang, write_meta=meta_path is not None)

    if meta_path is not None:
        out_text, meta = result
    else:
        out_text = result
        meta = None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(out_text, encoding="utf-8")

    if meta_path is not None:
        meta_path = Path(meta_path)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )