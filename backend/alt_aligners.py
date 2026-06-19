# -*- coding: utf-8 -*-
"""
alt_aligners.py — 替代音素对齐后端
支持 WhisperX / Qwen3-ASR-1.7B / Qwen3-ForcedAligner-0.6B 作为 MFA 的替代选项

模型文件路径策略（优先级）：
  1. 环境变量 TSUBAKI_MODELS_DIR
  2. <当前文件所在目录>/models/      → 即 backend/models/
     ├── hf_cache/         HuggingFace 统一缓存 (Qwen3-ASR / Qwen3-FA)
     │   └── hub/
     └── rmvpe/            RMVPE 模型 (已有，路径不变)

标点/静音处理：
  Qwen3 不在对齐输出中输出标点（标点不可发声）。
  本模块在生成 LAB 后自动扫描时间轴间隙，将 ≥ 50ms 的空白补全为 SIL 条目。
  用户无需为标点担心，静音标记由时间间隙自动推断。
"""
from __future__ import annotations

import logging
import os
import time
import unicodedata
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# 1. 模型文件路径管理（模块加载时立即执行，确保在任何 HF 导入之前完成）
# ═════════════════════════════════════════════════════════════════════════════

def resolve_models_dir() -> Path:
    """
    解析模型文件根目录。
    优先读取 TSUBAKI_MODELS_DIR 环境变量；否则使用 <backend>/models/。
    """
    env = os.environ.get("TSUBAKI_MODELS_DIR", "").strip()
    if env:
        p = Path(env).resolve()
        logger.info(f"[alt_aligners] 使用环境变量 TSUBAKI_MODELS_DIR: {p}")
    else:
        p = Path(__file__).resolve().parent / "models"
    p.mkdir(parents=True, exist_ok=True)
    return p


# ── 目录常量 ──────────────────────────────────────────────────────────────────
_MODELS_DIR:    Path = resolve_models_dir()
_HF_CACHE:      Path = _MODELS_DIR / "hf_cache"   # HuggingFace Hub 缓存
_HF_HUB:        Path = _HF_CACHE   / "hub"        # transformers 子目录
_WHISPER_CACHE: Path = _MODELS_DIR / "whisper"    # OpenAI Whisper 模型缓存

for _d in (_HF_CACHE, _HF_HUB, _WHISPER_CACHE):
    _d.mkdir(parents=True, exist_ok=True)

# 将 HuggingFace 缓存重定向到 backend/models/hf_cache/
# 使用 setdefault 不覆盖用户已配置的环境变量
os.environ.setdefault("HF_HOME",                       str(_HF_CACHE))
os.environ.setdefault("HF_HUB_CACHE",                  str(_HF_HUB))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE",         str(_HF_HUB))
os.environ.setdefault("TRANSFORMERS_CACHE",            str(_HF_HUB))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")   # 消除 Windows 警告

logger.info(
    f"[alt_aligners] 模型目录: {_MODELS_DIR}\n"
    f"  HF 缓存   → {_HF_HUB}\n"
    f"  Whisper缓存 → {_WHISPER_CACHE}"
)


# ═════════════════════════════════════════════════════════════════════════════
# 1b. PyTorch 2.6+ 兼容性补丁
#     PyTorch 2.6 起 torch.load 默认 weights_only=True，可能导致部分
#     HuggingFace 模型权重（内含自定义对象，如 omegaconf.ListConfig 等）
#     加载失败，抛出 _pickle.UnpicklingError。
#     这些权重来自官方 HF 仓库，可信，因此在模块加载时统一把
#     torch.load 的默认行为改回 weights_only=False。
# ═════════════════════════════════════════════════════════════════════════════
try:
    import torch as _torch

    if not getattr(_torch.load, "_tsubaki_patched", False):
        _original_torch_load = _torch.load

        def _patched_torch_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return _original_torch_load(*args, **kwargs)

        _patched_torch_load._tsubaki_patched = True
        _torch.load = _patched_torch_load
        logger.info(
            "[alt_aligners] 已应用 torch.load 兼容性补丁 "
            "(weights_only 默认改为 False)"
        )
except ImportError:
    pass  # torch 未安装时跳过；Qwen3 后端本身也用不了


# ═════════════════════════════════════════════════════════════════════════════
# 2. 语言代码映射
# ═════════════════════════════════════════════════════════════════════════════

def _to_qwen_lang_name(lang: str) -> Optional[str]:
    """
    内部语言代码 → 官方 qwen-asr 包 Qwen3ASRModel.transcribe() /
    Qwen3ForcedAligner.align() 所要求的完整语言名（如 "Chinese"）。

    依据官方示例 (QwenLM/Qwen3-ASR examples/example_qwen3_asr_transformers.py,
    examples/example_qwen3_forced_aligner.py)：language 参数接受完整英文语言名
    （"Chinese" / "English" / "Japanese" / "Korean" / "Cantonese" ...），
    不是 ISO 短代码；返回 None 表示交给 Qwen3-ASR 自动语言检测（仅 ASR 支持，
    ForcedAligner 必须显式指定语言）。
    """
    return {
        "cmn": "Chinese", "zh": "Chinese", "zh-cn": "Chinese", "mandarin": "Chinese",
        "yue": "Cantonese", "cantonese": "Cantonese", "zh-yue": "Cantonese",
        "eng": "English", "en": "English", "english": "English",
        "jpn": "Japanese", "ja": "Japanese", "japanese": "Japanese",
        "kor": "Korean", "ko": "Korean", "korean": "Korean",
    }.get(lang.lower())


def _normalize_lang(lang: str) -> str:
    """各种语言代码 → 内部短代码 (zh / yue / en / ja / ko)"""
    return {
        "cmn": "zh", "zh-cn": "zh", "mandarin": "zh",
        "yue": "yue", "cantonese": "yue", "zh-yue": "yue",
        "eng": "en", "english": "en",
        "jpn": "ja", "japanese": "ja",
        "kor": "ko", "korean": "ko",
    }.get(lang.lower(), lang.lower())


def _to_whisperx_lang(lang: str) -> str:
    """内部语言代码 → WhisperX / Whisper 语言代码"""
    return {
        "cmn": "zh", "zh": "zh", "zh-cn": "zh",
        "yue": "zh",   # 粤语用 zh 近似；WhisperX 暂无独立粤语对齐模型
        "eng": "en", "en": "en",
        "jpn": "ja", "ja": "ja",
        "kor": "ko", "ko": "ko",
    }.get(lang.lower(), lang.lower())


# ═════════════════════════════════════════════════════════════════════════════
# 2b. WhisperX 对齐前文本预处理
#     WhisperX 强制对齐依赖"单调时间映射假设"：text ≈ audio 的顺序单调映射。
#     结构化文本（编号、列表符号、markdown 标题、CJK 标点）会破坏该假设，
#     导致 word-level alignment 失败甚至崩溃。
#     本函数在对齐前将参考文本口语化，使其对 wav2vec2 模型更友好。
# ═════════════════════════════════════════════════════════════════════════════

import re as _re

def normalize_text_for_whisperx(text: str, lang: str = "zh") -> str:
    """
    将结构化/格式化文本转换为 WhisperX 强制对齐友好的口语化文本。

    处理内容：
      - 移除列表编号（1. 2. 一、二、①② 等）
      - 移除 Markdown 标题符号（# ## ###）
      - 将 CJK 标点替换为空格（，。！？：；「」『』）
      - 将连续空白压缩为单个空格

    参数
    ----
    text : str
        原始参考文本（可含结构符号）
    lang : str
        内部语言短代码（zh / en / ja / ko / yue），目前对所有语言统一处理

    返回
    ----
    str
        口语化清洗后的文本，保留所有可发音字符
    """
    if not text:
        return text

    # ── 1. 移除 Markdown 标题（# / ## / ### 开头）──────────────────────────
    text = _re.sub(r"^#{1,6}\s*", "", text, flags=_re.MULTILINE)

    # ── 2. 移除行首列表编号 ─────────────────────────────────────────────────
    #    匹配：1. / 1) / (1) / 一、/ 二、/ ① ② ③ 等各类编号
    text = _re.sub(r"(?m)^\s*(?:\d+[.、）)]\s*|[①②③④⑤⑥⑦⑧⑨⑩]+\s*|[一二三四五六七八九十]+[、.]\s*)", "", text)

    # ── 3. 移除行内的短编号标记（"一、" 出现在句中也要清除）──────────────────
    text = _re.sub(r"[一二三四五六七八九十]+[、]", "", text)

    # ── 4. 将 CJK 标点替换为空格（不可发音，且会产生无法对齐的 token）─────────
    cjk_puncts = r"[，。！？：；、「」『』【】《》〈〉—…·~～｜│「｣『｣]"
    text = _re.sub(cjk_puncts, " ", text)

    # ── 5. 移除英文引号、括号（对强制对齐同样干扰）──────────────────────────
    text = _re.sub(r'["""\'()\[\]{}]', " ", text)

    # ── 6. 将换行 / 制表符统一为空格 ─────────────────────────────────────────
    text = _re.sub(r"[\r\n\t]+", " ", text)

    # ── 7. 压缩多余空白 ───────────────────────────────────────────────────────
    text = _re.sub(r"\s{2,}", " ", text).strip()

    if text:
        logger.debug(f"[WhisperX normalize] → {text[:80]}")

    return text


# ═════════════════════════════════════════════════════════════════════════════
# 2c. ASR 转录文本里的阿拉伯数字编号伪影清洗
#
#     背景（诊断结论）：normalize_text_for_whisperx() 清洗的是"用户提供
#     的参考文本"，但 WhisperXAligner 的强制对齐阶段实际对齐的是
#     Whisper 自己识别出来的转录文本（asr_out["segments"] 里的 text），
#     参考文本只在对齐完成后用于音素映射，不会经过上面的清洗函数。
#
#     实测发现：当朗读内容本身包含中文序数词枚举（"一，声库。"）时，
#     Whisper 的中文转写有时会把"一，"自动格式化成阿拉伯数字列表样式
#     "1."。"1" 这个字符在 wav2vec2 的中文音素词表里没有对应发音，
#     会在该处造成局部强制对齐错乱——典型表现为紧邻该位置的音素被
#     压缩成几毫秒，或前后音素时长被异常拉伸（与 _check_alignment_quality
#     的整体性退化插值是两种不同成因，但外观上都是"音素时长不正常"）。
#
#     这里只清除"单个数字 + 紧跟句点/顿号、且不是多位数字或小数"这种
#     典型列表标记模式，避免误伤歌词里真实出现的数字（如年份 "2026"
#     或小数 "3.14"）。
# ═════════════════════════════════════════════════════════════════════════════

_ASR_NUMERAL_ARTIFACT_RE = _re.compile(r"(?<![\d.])([1-9])[.、](?!\d)\s*")


def strip_asr_numeral_artifacts(text: str) -> str:
    """
    清除 Whisper ASR 输出文本中形如 "1." "2、" 的孤立阿拉伯数字编号伪影。
    只在 WhisperXAligner 内部对 ASR 自己的转录文本调用，不应用于
    用户提供的参考文本（那部分由 normalize_text_for_whisperx 处理）。
    """
    if not text:
        return text
    cleaned = _ASR_NUMERAL_ARTIFACT_RE.sub("", text)
    if cleaned != text:
        logger.info(f"[WhisperX] 清除 ASR 编号伪影: {text!r} → {cleaned!r}")
    return cleaned


# ═════════════════════════════════════════════════════════════════════════════
# 3. 工具函数
# ═════════════════════════════════════════════════════════════════════════════

class _MI:
    """模拟 textgrid.Interval，供 MFAProcessor 内部逻辑使用（时间单位：秒）"""
    __slots__ = ("minTime", "maxTime", "mark", "text")

    def __init__(self, start_sec: float, end_sec: float, mark: str):
        self.minTime = float(start_sec)
        self.maxTime = float(end_sec)
        self.mark = mark
        self.text = mark


def _is_cjk_punct(text: str) -> bool:
    """
    判断字符串是否全为标点 / 空白 / 符号（用于过滤 ASR 输出中的标点字符）。

    注：标点本身不可发音，不应出现在 LAB 的音素层。
    中文句末停顿（。！？）和停顿符（，、）对应的 LAB 条目由
    _fill_silences_lab() 根据时间轴间隙自动插入 SP / SIL。
    """
    if not text:
        return True
    for ch in text:
        cat = unicodedata.category(ch)
        if not cat.startswith(("P", "Z", "S")):
            return False
    return True


def _fill_silences_lab(
    lab_content: str,
    min_gap_100ns: int = 500_000,       # 50ms
    long_sil_100ns: int = 5_000_000,    # 500ms → 统一输出 SIL
) -> str:
    """
    扫描 LAB 时间轴，在 ≥ 50ms 的间隙自动补全 SIL 条目。

    背景：Qwen3 不输出标点字符的时间戳，但句末/句中停顿
    会在相邻字符之间留下时间间隙，本函数将这些间隙转换为 LAB 静音标记。

    Parameters
    ----------
    min_gap_100ns : 插入静音的最小间隙（默认 50ms）
    long_sil_100ns : 超过此值仍输出 SIL（保留参数以兼容旧调用）
    """
    if not lab_content.strip():
        return lab_content

    parsed: List[Tuple[int, int, str]] = []
    for line in lab_content.strip().splitlines():
        parts = line.strip().split()
        if len(parts) >= 3:
            try:
                parsed.append((int(parts[0]), int(parts[1]), parts[2]))
            except ValueError:
                continue

    if not parsed:
        return lab_content

    result: List[Tuple[int, int, str]] = []

    # ── 音频开头静音 ──────────────────────────────────────────────────────
    if parsed[0][0] > min_gap_100ns:
        result.append((0, parsed[0][0], "SIL"))

    for i, (s, e, ph) in enumerate(parsed):
        result.append((s, e, ph))
        if i + 1 < len(parsed):
            gap_s = e
            gap_e = parsed[i + 1][0]
            gap   = gap_e - gap_s
            if gap > min_gap_100ns:
                result.append((gap_s, gap_e, "SIL"))

    return "\n".join(f"{s} {e} {p}" for s, e, p in result)


def _count_spoken_chars(text: str, int_lang: str) -> int:
    """统计参考文本中的可发音字符数（排除标点/空白），用于与 entries 数量对比"""
    count = 0
    for ch in text:
        cat = unicodedata.category(ch)
        if cat.startswith(("P", "Z", "S")):
            continue
        # 对 CJK 语言只计汉字和假名
        if int_lang in ("zh", "yue"):
            if '\u4e00' <= ch <= '\u9fff' or '\u3400' <= ch <= '\u4dbf':
                count += 1
        elif int_lang == "ja":
            if '\u3040' <= ch <= '\u30ff' or '\u4e00' <= ch <= '\u9fff':
                count += 1
        else:
            if ch.strip():
                count += 1
    return count


# ═════════════════════════════════════════════════════════════════════════════
# 3b. WhisperX 对齐前句子级预切分
#
#     背景（诊断结论）：
#     wav2vec2 CTC 强制对齐在"长且内部无明显停顿"的整段音频上容易整体失败
#     （找不到有效解码路径）。whisperx.align() 内部对失败的片段有一个静默
#     兜底逻辑：退化为"按字符数比例均匀插值"，而不是抛出异常。
#     这种兜底结果的典型特征是：同一段音频里几乎所有音素时长高度雷同
#     （只有个位数的几个不同取值，标准差极小），与真实发音的自然节奏
#     （时长应有较大方差）明显不符。
#
#     注意：这一步切分的是 Whisper 自己识别出来的 segment 文本
#     （asr_out["segments"] 里的 text，是 Whisper 实际"听到"并转写出来的
#     内容），不是用户填写的参考文本。WhisperXAligner.align() 的强制对齐
#     阶段本来就不使用用户参考文本（只在对齐完成后，由 _word_entries_to_lab()
#     用参考文本驱动音素/拼音转换），所以参考文本里的编号、Markdown 标题等
#     结构化标记，并不会直接导致这里的对齐失败——除非这些标记在用户提供
#     的参考文本和实际音频之间不一致，导致后续音素映射错位（这是另一个、
#     独立的问题，见 _word_entries_to_lab 里的字符数校验告警）。
# ═════════════════════════════════════════════════════════════════════════════

_SENTENCE_END_RE = _re.compile(r"[。！？!?；;]")
# 次级停顿符（逗号/顿号）：句末标点切不动、但估算时长仍超限时，
# 优先按这里切，而不是直接按字符数硬切——这样切出来的边界更贴近
# 真实语流停顿，能降低 wav2vec2 在"非真实语音边界"处对齐错乱的概率。
_PAUSE_RE = _re.compile(r"[，,、]")


def _force_char_split(p: str, est_dur: float, max_chunk_sec: float) -> List[str]:
    """
    按字符数把过长的子句强制切成多块。
    仅作兜底：只有在句末标点、逗号/顿号都切不出有效边界时才会用到。
    """
    n_sub = max(1, int(est_dur // max_chunk_sec) + 1)
    sub_len = max(1, -(-len(p) // n_sub))  # 向上取整
    chars = list(p)
    out: List[str] = []
    for i in range(0, len(chars), sub_len):
        sub_piece = "".join(chars[i:i + sub_len])
        if sub_piece:
            out.append(sub_piece)
    return out


def _split_segment_by_punctuation(
    seg_start: float,
    seg_end: float,
    seg_text: str,
    max_chunk_sec: float = 6.0,
) -> List[Tuple[float, float, str]]:
    """
    把单个 Whisper segment 按句末标点切成更小的子段。

    子段的起止时间只是按字符数比例估算出来的"粗略窗口"——真正精确的
    逐音素时间仍然由 wav2vec2 CTC 在每个子段内部重新计算。把过长 /
    多句合并的 segment 提前拆开，能显著降低 CTC 在长音频上整体对齐
    失败的概率；即使个别子段仍然失败，受影响的时间范围也会小得多，
    不会像现在这样让整段几十个字符全部退化成均匀插值。

    Parameters
    ----------
    max_chunk_sec : 没有标点可切、但时长仍超过该值时，按字符数强制平均切分
                    （防止单个子段仍然过长）。
    """
    seg_text = seg_text.strip()
    duration = seg_end - seg_start
    if not seg_text or duration <= 0:
        return [(seg_start, seg_end, seg_text)]

    # 按句末标点切分（标点保留在前一句末尾，便于保留语气/停顿信息）
    parts = _SENTENCE_END_RE.split(seg_text)
    puncts = _SENTENCE_END_RE.findall(seg_text)
    pieces: List[str] = []
    for i, p in enumerate(parts):
        p = p.strip()
        mark = puncts[i] if i < len(puncts) else ""
        piece = (p + mark).strip()
        if piece:
            pieces.append(piece)

    if not pieces:
        return [(seg_start, seg_end, seg_text)]
    if len(pieces) == 1 and duration <= max_chunk_sec:
        # 单句、且不算长，不需要切分
        return [(seg_start, seg_end, seg_text)]

    # 第二遍：任何一个标点切出来的子句，若按字符比例估算出的时长仍然
    # 超过 max_chunk_sec（常见于"没有任何标点的长句"，包括整句只有
    # 一个 piece 的情况，比如 ASR 把多句话识别成一个不含句末标点的
    # 长 segment 时），先尝试按逗号/顿号等次级停顿符切，切不出来
    # 再按字符数强制切，避免子段仍然过长、也避免硬切在非真实语音
    # 边界上下刀。
    total_chars = sum(len(p) for p in pieces) or 1
    final_pieces: List[str] = []
    for p in pieces:
        est_dur = duration * (len(p) / total_chars)
        if est_dur > max_chunk_sec and len(p) > 1:
            sub_parts = _PAUSE_RE.split(p)
            sub_marks = _PAUSE_RE.findall(p)
            comma_pieces: List[str] = []
            for j, sp in enumerate(sub_parts):
                sp = sp.strip()
                m = sub_marks[j] if j < len(sub_marks) else ""
                cp = (sp + m).strip()
                if cp:
                    comma_pieces.append(cp)

            if len(comma_pieces) > 1:
                sub_total = sum(len(cp) for cp in comma_pieces) or 1
                for cp in comma_pieces:
                    cp_est_dur = est_dur * (len(cp) / sub_total)
                    if cp_est_dur > max_chunk_sec and len(cp) > 1:
                        final_pieces.extend(
                            _force_char_split(cp, cp_est_dur, max_chunk_sec)
                        )
                    else:
                        final_pieces.append(cp)
            else:
                final_pieces.extend(_force_char_split(p, est_dur, max_chunk_sec))
        else:
            final_pieces.append(p)

    if len(final_pieces) <= 1:
        return [(seg_start, seg_end, seg_text)]

    # 按各子段字符数比例分配时间边界（仅作为 CTC 对齐窗口的估计起止点）
    weights = [max(len(p), 1) for p in final_pieces]
    total_w = sum(weights)
    out: List[Tuple[float, float, str]] = []
    cursor = seg_start
    for p, w in zip(final_pieces, weights):
        piece_dur = duration * (w / total_w)
        piece_end = cursor + piece_dur
        out.append((cursor, piece_end, p))
        cursor = piece_end
    if out:
        s, _, p = out[-1]
        out[-1] = (s, seg_end, p)  # 消除浮点误差累积，确保末尾对齐到原 segment 结束时间
    return out


def _resegment_for_alignment(
    segments: List[Dict],
    max_chunk_sec: float = 6.0,
) -> List[Dict]:
    """
    在调用 whisperx.align() 之前，把 Whisper 自己输出的 segments 按句末标点
    （和必要时的强制字符数切分）拆成更小的子段，降低 wav2vec2 CTC
    在长音频上整体对齐失败、静默退化为均匀插值的概率。

    返回值仍是 whisperx.align() 所需的 dict 列表格式（至少含 start/end/text）。
    """
    new_segments: List[Dict] = []
    n_split = 0
    for seg in segments:
        s = seg.get("start")
        e = seg.get("end")
        t = (seg.get("text") or "").strip()
        if s is None or e is None or not t:
            new_segments.append(seg)
            continue

        sub = _split_segment_by_punctuation(float(s), float(e), t, max_chunk_sec)
        if len(sub) <= 1:
            new_segments.append(seg)
            continue

        n_split += 1
        for sub_s, sub_e, sub_t in sub:
            new_seg = dict(seg)  # 保留原 segment 的其他字段
            new_seg["start"] = sub_s
            new_seg["end"] = sub_e
            new_seg["text"] = sub_t
            # word/char 级字段是上一阶段（若有）针对原 segment 计算的，
            # 切分后已不适用，交给 align() 重新计算
            new_seg.pop("words", None)
            new_seg.pop("chars", None)
            new_segments.append(new_seg)

    if n_split:
        logger.info(
            f"[WhisperX] 句子级预切分：{n_split} 个原始 segment 被拆分为更小的对齐子段"
        )
    return new_segments


def _check_alignment_quality(
    entries: List[Tuple[float, float, str]],
    context: str = "",
) -> None:
    """
    粗略 QC 检查：若对齐结果里音素时长高度雷同（不同取值占比极低），
    很可能是 wav2vec2 CTC 对该片段（或整段）对齐失败，触发了 whisperx
    内部"按字符数比例均匀插值"的静默兜底，而不是真实的声学对齐结果。

    只记录警告，不阻断流程；建议人工检查涉及的音频片段，或调小
    max_align_chunk_sec 做更细粒度的预切分。
    """
    if len(entries) < 10:
        return
    durs = [round(e - s, 4) for s, e, _ in entries]
    distinct_ratio = len(set(durs)) / len(durs)
    if distinct_ratio < 0.35:
        prefix = f"[{context}] " if context else ""
        logger.warning(
            f"[WhisperX] ⚠ {prefix}对齐结果时长高度雷同"
            f"（{len(set(durs))}/{len(durs)} 个不同取值，占比 {distinct_ratio:.0%}），"
            "疑似 wav2vec2 强制对齐对部分/全部音频片段失败，退化为按字符比例"
            "均匀插值，而非真实声学对齐。建议检查该音频片段是否存在："
            "长时间无停顿的连续朗读、ASR 识别文本与音频内容不匹配、或音频质量问题；"
            "也可尝试调小 WhisperXAligner(max_align_chunk_sec=...) 做更细粒度预切分。"
        )


# ═════════════════════════════════════════════════════════════════════════════
# 3c. "边界压缩"伪影修复
#
#     背景（基于实际对齐结果排查）：即便整体对齐质量正常（没有触发上面
#     的"均匀插值"退化告警），wav2vec2 CTC 强制对齐在"语音→静音"边界
#     处仍会系统性地出现两类局部伪影：
#
#       A. 紧邻一段真实停顿（句末/分句间隙）之前的最后一个音素，常被
#          压缩成几毫秒到二十毫秒——CTC 解码在语音概率骤降的位置武断
#          截断，把本该属于该音素收尾部分的帧错误地划给了"空白"。
#          在 piano roll / 音素编辑器里看，就是这个音素几乎"消失"或
#          被紧紧挤扁在前一个音素旁边。
#
#       B. 夹在两个正常时长音素之间、左右都没有明显停顿的音素也可能
#          被异常压缩（CTC 在相近音素之间概率分布不稳定造成的边界
#          误判），表现为相邻音素之间没有自然过渡，"音标连在一起"。
#
#     这两类问题在同一份 102 条目的对齐结果里出现了 5 处，且全部精确
#     等于地板值（约 12–20ms），与真实发音节奏的自然方差明显不符，
#     是可以低风险、规则化修复的局部伪影，而不是需要重新对齐才能
#     解决的整体性失败。
# ═════════════════════════════════════════════════════════════════════════════

def _smooth_short_phones(
    entries: List[Tuple[float, float, str]],
    total_audio_sec: Optional[float] = None,
    min_phone_sec: float = 0.05,
    gap_threshold_sec: float = 0.25,
    gap_borrow_ratio: float = 0.5,
    max_gap_borrow_sec: float = 0.08,
    neighbor_borrow_sec: float = 0.015,
    neighbor_min_remaining_sec: float = 0.06,
) -> List[Tuple[float, float, str]]:
    """
    规则 A：当前音素时长 < min_phone_sec，且紧随其后是一段真实停顿
    （间隙 ≥ gap_threshold_sec）→ 从这段停顿里"借"一部分时间延长
    当前音素的收尾（不动前一个音素，也给停顿留够 0.05s 以上余量，
    避免把停顿完全吃掉）。若是文件最后一个条目，用 total_audio_sec
    （音频总时长）替代"下一条目的开始时间"来判断右侧是否有空白可借。

    规则 B：当前音素时长 < min_phone_sec，但左右都没有明显停顿
    （说明问题出在 CTC 内部，不是真实静音被误吞）→ 从左右邻居各借
    一点点时间（默认每侧最多 15ms），且只在邻居自身借出后仍不低于
    neighbor_min_remaining_sec 时才借，避免拆东墙补西墙引入新的
    短时长。

    两条规则都只调整边界、不增删条目，单次调整幅度都有上限，
    是"修正局部伪影"而非"重新分配整体节奏"。
    """
    if len(entries) < 2:
        return entries

    fixed = [[s, e, ph] for s, e, ph in entries]
    n = len(fixed)

    for i in range(n):
        s, e, ph = fixed[i]
        dur = e - s
        if dur >= min_phone_sec:
            continue

        # ── 规则 A：右侧是真实停顿 ──────────────────────────────────────
        if i + 1 < n:
            gap = fixed[i + 1][0] - e
        elif total_audio_sec is not None:
            gap = total_audio_sec - e
        else:
            gap = 0.0

        if gap >= gap_threshold_sec:
            borrow = min(gap * gap_borrow_ratio, max_gap_borrow_sec, gap - 0.05)
            if borrow > 0:
                fixed[i][1] = e + borrow
                continue  # 已处理，不再叠加规则 B

        # ── 规则 B：左右都没有明显停顿，各邻居借一点点 ───────────────────
        if i - 1 >= 0 and i + 1 < n:
            prev_s, prev_e, _ = fixed[i - 1]
            next_s, next_e, _ = fixed[i + 1]
            prev_gap = s - prev_e
            next_gap = next_s - e
            if prev_gap < gap_threshold_sec and next_gap < gap_threshold_sec:
                prev_dur = prev_e - prev_s
                next_dur = next_e - next_s
                lend_left = (
                    neighbor_borrow_sec
                    if prev_dur - neighbor_borrow_sec >= neighbor_min_remaining_sec
                    else 0.0
                )
                lend_right = (
                    neighbor_borrow_sec
                    if next_dur - neighbor_borrow_sec >= neighbor_min_remaining_sec
                    else 0.0
                )
                if lend_left > 0:
                    fixed[i - 1][1] -= lend_left
                    fixed[i][0] -= lend_left
                if lend_right > 0:
                    fixed[i + 1][0] += lend_right
                    fixed[i][1] += lend_right

    return [(s, e, ph) for s, e, ph in fixed]


# ═════════════════════════════════════════════════════════════════════════════
# 4. 基类
# ═════════════════════════════════════════════════════════════════════════════

class AltAlignerBase:
    """
    所有替代对齐后端的公共基类。
    共享 MFAProcessor 的音素转换 / 后处理逻辑，通过 _word_entries_to_lab() 复用。
    """

    def __init__(self):
        from mfa_processor import MFAProcessor
        self._mfa = MFAProcessor()

    def align(self, audio_path: str, text: Optional[str], language: str) -> Dict:
        raise NotImplementedError

    # ── 词语时间戳 → LAB（含静音间隙补全）────────────────────────────────
    def _word_entries_to_lab(
        self,
        word_entries: List[Tuple[float, float, str]],
        text: str,
        language: str,
        fill_silences: bool = False,
    ) -> str:
        """
        将词语 / 字符级时间戳 → LAB 格式，复用 MFAProcessor 的音素转换逻辑。
        fill_silences=True 时自动在时间间隙中插入 SIL。

        关于标点：
          Qwen3 不产生标点字符的对齐时间戳（标点不可发音）。
          _text_to_syllables() 在提取音素序列时也会忽略标点，因此参考文本中
          的标点不影响音素分布——只要参考文本的可发音字符数与 entries 数量一致即可。
          句末 / 句中的停顿由 fill_silences 根据时间间隙自动插入 SIL。
        """
        if not word_entries:
            return ""

        lang = _normalize_lang(language)

        # 字符数不匹配时提前警告（便于调试）
        if text and lang in ("zh", "yue", "ja"):
            spoken_n = _count_spoken_chars(text, lang)
            entries_n = len(word_entries)
            if spoken_n != entries_n:
                logger.warning(
                    f"[alt_aligners] 参考文本可发音字符数 {spoken_n} ≠ "
                    f"对齐条目数 {entries_n}。如出现音素偏移，请检查参考文本是否与音频一致。"
                )

        # 日语 / 韩语需要特殊处理
        if lang == "ja":
            lab = self._ja_entries_to_lab(word_entries, text)
            return _fill_silences_lab(lab) if fill_silences else lab
        if lang == "ko":
            lab = self._ko_entries_to_lab(word_entries, text)
            return _fill_silences_lab(lab) if fill_silences else lab

        word_tier = [_MI(s, e, w) for s, e, w in word_entries]
        phone_items: List[Tuple[int, int, str]] = []

        if lang in ("zh", "cmn"):
            lines = self._mfa._process_zh_words(word_tier, phone_items, text)
        elif lang == "yue":
            lines = self._mfa._process_yue_words(word_tier, phone_items, text)
        else:
            lines = self._mfa._process_en_words(word_tier, phone_items, text)

        lines = self._mfa._apply_lab_postprocess(lines, lang)
        lab = "\n".join(lines)
        return _fill_silences_lab(lab) if fill_silences else lab

    def _ja_entries_to_lab(
        self,
        word_entries: List[Tuple[float, float, str]],
        text: str,
    ) -> str:
        try:
            import pykakasi
            kks = pykakasi.kakasi()
        except ImportError:
            lines = []
            for s, e, ch in word_entries:
                if ch.strip() and not _is_cjk_punct(ch):
                    lines.append(f"{int(s*10_000_000)} {int(e*10_000_000)} {ch}")
            return "\n".join(lines)

        lines: List[str] = []
        for s, e, ch in word_entries:
            ch = ch.strip()
            if not ch or _is_cjk_punct(ch):
                continue
            result = kks.convert(ch)
            hira = "".join(r.get("hira", r.get("orig", "")) for r in result).strip()
            lines.append(f"{int(s*10_000_000)} {int(e*10_000_000)} {hira or ch}")

        from mfa_processor import MFAProcessor
        entries_p = MFAProcessor._parse_lab_lines(lines)
        from phoneme_converter import build_ja_hiragana_lab, merge_lab_silence
        entries_p = build_ja_hiragana_lab(entries_p)
        merged = merge_lab_silence(entries_p)
        return "\n".join(f"{s} {e} {p}" for s, e, p in merged)

    def _ko_entries_to_lab(
        self,
        word_entries: List[Tuple[float, float, str]],
        text: str,
    ) -> str:
        lines: List[str] = []
        for s, e, ch in word_entries:
            ch = ch.strip()
            if not ch:
                continue
            s100 = int(s * 10_000_000)
            e100 = int(e * 10_000_000)
            dur  = e100 - s100
            if self._mfa._is_korean_text(ch) and len(ch) == 1:
                has_init = self._mfa._get_korean_initial_consonant(ch) == "has_initial"
                if has_init:
                    dash_dur = max(dur // 3, 60_000)
                    dash_end = min(s100 + dash_dur, e100 - 60_000)
                    lines.append(f"{s100} {dash_end} -")
                    lines.append(f"{dash_end} {e100} {ch}")
                else:
                    lines.append(f"{s100} {e100} {ch}")
            else:
                lines.append(f"{s100} {e100} {ch}")

        from mfa_processor import MFAProcessor
        entries_p = MFAProcessor._parse_lab_lines(lines)
        from phoneme_converter import merge_lab_silence
        merged = merge_lab_silence(entries_p)
        return "\n".join(f"{s} {e} {p}" for s, e, p in merged)

    def _get_audio_duration_100ns(self, audio_path: str) -> int:
        return self._mfa._get_audio_duration(audio_path)

# ═════════════════════════════════════════════════════════════════════════════
# 5. WhisperXAligner
# ═════════════════════════════════════════════════════════════════════════════

class WhisperXAligner(AltAlignerBase):
    """
    WhisperX 对齐后端（自动语音识别 + wav2vec2 强制音素对齐）
    https://github.com/m-bain/whisperx

    优势：
      - 不需要参考文本（自动转录模式）
      - 字符级对齐（中日韩），词语级对齐（英语等）
      - 支持 GPU 加速（CUDA）

    注意：
      - 结构化文本（编号/列表/Markdown 标题）会破坏"单调时间映射假设"，
        导致 wav2vec2 对齐失败。本类在调用对齐前自动调用
        normalize_text_for_whisperx() 进行口语化清洗。
      - 安装：pip install whisperx
    """

    def __init__(
        self,
        whisper_model: str = "large-v2",
        device: str = "auto",
        compute_type: str = "float16",
        batch_size: int = 16,
        hf_token: Optional[str] = None,
        whisper_chunk_size: int = 12,
        max_align_chunk_sec: float = 6.0,
        min_phone_sec: float = 0.05,
    ):
        super().__init__()
        self.whisper_model = whisper_model
        self._device = self._resolve_device(device)
        # CPU 不支持 float16
        self.compute_type = compute_type if self._device != "cpu" else "int8"
        self.batch_size = batch_size
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")

        # ── 长音频/连续朗读对齐稳定性参数 ──────────────────────────────────
        # whisper_chunk_size: 传给 whisperx 的 transcribe(chunk_size=...)，
        #   控制 Whisper 自身（VAD 合并后）单个 segment 的最大时长（秒）。
        #   默认值比 whisperx 的内部默认 30s 更保守，让连续朗读（无明显
        #   停顿）的音频也能被切成较短的 segment。
        # max_align_chunk_sec: _resegment_for_alignment() 在调用
        #   whisperx.align() 之前，对 Whisper 输出的 segment 做句子级
        #   预切分时使用的目标时长上限（秒）。
        # min_phone_sec: _smooth_short_phones() 用来判定"音素是否被
        #   异常压缩"的下限阈值（秒）。实测正常发音的音素时长基本都在
        #   0.06s 以上，<0.05s 几乎一定是边界压缩伪影而非真实发音。
        self.whisper_chunk_size = whisper_chunk_size
        self.max_align_chunk_sec = max_align_chunk_sec
        self.min_phone_sec = min_phone_sec

        self._asr_model = None
        self._align_models: Dict[str, object] = {}   # {lang_code: (model_a, metadata)}

    # ── 类方法 ──────────────────────────────────────────────────────────────
    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return device

    @staticmethod
    def check_available() -> Tuple[bool, str]:
        try:
            import whisperx  # noqa: F401
            return True, "OK"
        except ImportError as e:
            return False, f"未安装: pip install whisperx ({e})"
        except Exception as e:
            return False, str(e)

    # ── 懒加载 ──────────────────────────────────────────────────────────────
    def _load_asr(self):
        if self._asr_model is None:
            import whisperx
            logger.info(f"[WhisperX] 加载 ASR 模型: {self.whisper_model} ({self._device})")
            self._asr_model = whisperx.load_model(
                self.whisper_model,
                self._device,
                compute_type=self.compute_type,
                download_root=str(_WHISPER_CACHE),
            )
            logger.info("[WhisperX] ✓ ASR 模型已加载")

    def _load_align(self, lang_code: str):
        if lang_code not in self._align_models:
            import whisperx
            logger.info(f"[WhisperX] 加载对齐模型: {lang_code}")
            model_a, metadata = whisperx.load_align_model(
                language_code=lang_code, device=self._device
            )
            self._align_models[lang_code] = (model_a, metadata)
            logger.info(f"[WhisperX] ✓ 对齐模型 ({lang_code}) 已加载")
        return self._align_models[lang_code]

    # ── 核心对齐 ─────────────────────────────────────────────────────────────
    def align(self, audio_path: str, text: Optional[str], language: str) -> Dict:
        t0 = time.time()
        try:
            import whisperx

            wx_lang = _to_whisperx_lang(language)
            int_lang = _normalize_lang(language)

            # 1. 加载音频
            audio = whisperx.load_audio(audio_path)

            # 2. ASR
            self._load_asr()
            logger.info("[WhisperX] 开始 ASR 转录...")
            try:
                # chunk_size 控制 VAD 合并后单个 segment 的最大时长（秒）。
                # 比 whisperx 默认的 30s 更保守，避免长时间无停顿的连续
                # 朗读被合并成一个过长的 segment（过长 segment 是后续
                # wav2vec2 CTC 强制对齐整体失败、退化为均匀插值的主因之一）。
                asr_out = self._asr_model.transcribe(
                    audio, batch_size=self.batch_size, language=wx_lang,
                    chunk_size=self.whisper_chunk_size,
                )
            except TypeError:
                # 旧版 whisperx 不支持 chunk_size 参数，回退到默认行为
                asr_out = self._asr_model.transcribe(
                    audio, batch_size=self.batch_size, language=wx_lang
                )
            if not asr_out.get("segments"):
                return self._err("WhisperX ASR 无输出，请检查音频质量", t0)

            # 2.5 清除 ASR 转录里可能出现的阿拉伯数字编号伪影（例如把
            #     朗读时说的"一，"自动格式化成"1."）。必须在这里清洗，
            #     而不是只清洗用户参考文本——因为接下来的预切分 + 强制
            #     对齐用的正是这里的 ASR 文本，留着"1."这种不可发音的
            #     token 会在该处造成局部对齐错乱。
            for _seg in asr_out["segments"]:
                _seg["text"] = strip_asr_numeral_artifacts(_seg.get("text", ""))

            asr_text = " ".join(s.get("text", "") for s in asr_out["segments"]).strip()
            logger.info(f"[WhisperX] ASR 文本: {asr_text[:80]}")

            # 3. 句子级预切分：把过长 / 多句合并的 segment 按标点拆成更小的
            #    对齐子段，降低 wav2vec2 CTC 在长音频上整体对齐失败、
            #    静默退化为"按字符比例均匀插值"的概率。
            #    （直接使用 ASR 转录的 segments/文本，不覆盖为参考文本——
            #    把参考文本整体注入所有 segments 会让 WhisperX 把全部字符
            #    塞进每个 segment 的时间范围内，导致字符间静音消失、
            #    句末音节时间被拉伸到下一句开头。）
            segments_for_align = _resegment_for_alignment(
                asr_out["segments"], max_chunk_sec=self.max_align_chunk_sec
            )

            model_a, metadata = self._load_align(wx_lang)
            logger.info("[WhisperX] 开始强制对齐...")
            aligned = whisperx.align(
                segments_for_align, model_a, metadata, audio, self._device,
                return_char_alignments=True,   # CJK 关键：字符级对齐
            )

            # 4. 提取词语时间戳
            entries = self._extract_entries(aligned, int_lang)
            if not entries:
                return self._err("强制对齐无输出，请检查语言代码和音频质量", t0)

            # 4.5 修复"语音→静音"边界压缩伪影（详见 _smooth_short_phones
            #     上方注释）：句末/分句停顿前最后一个音素被压缩成几毫秒，
            #     或夹在两个正常音素之间被异常压缩，都在这里做小幅边界
            #     修正，而不是留给后续的 SIL 间隙填充（那只处理"间隙"，
            #     不处理"音素本身被压扁"）。
            total_dur_sec = self._get_audio_duration_100ns(audio_path) / 10_000_000
            entries = _smooth_short_phones(
                entries, total_audio_sec=total_dur_sec, min_phone_sec=self.min_phone_sec
            )

            _check_alignment_quality(entries, context=os.path.basename(audio_path))

            # 5. 转换为 LAB（优先使用用户提供的参考文本驱动音素转换）
            final_text = text.strip() if text else asr_text
            lab = self._word_entries_to_lab(entries, final_text, language)

            return {
                "success": True,
                "lab_content": lab,
                "raw_text": final_text,
                "phoneme_text": asr_text,
                "audio_duration": self._get_audio_duration_100ns(audio_path),
                "processing_time": int((time.time() - t0) * 1000),
                "backend": "whisperx",
            }

        except ImportError as e:
            return self._err(
                f"whisperx 未安装: {e}，请执行 pip install whisperx", t0
            )
        except Exception as e:
            logger.error(f"[WhisperX] 对齐失败: {e}", exc_info=True)
            return self._err(str(e), t0)

    def _extract_entries(
        self, aligned: Dict, int_lang: str
    ) -> List[Tuple[float, float, str]]:
        """从 WhisperX 对齐结果提取 (start_sec, end_sec, text) 列表"""
        entries: List[Tuple[float, float, str]] = []

        for seg in aligned.get("segments", []):
            # CJK 优先使用字符级对齐
            chars = seg.get("chars", [])
            words = seg.get("words", [])

            if int_lang in ("zh", "yue", "ja") and chars:
                for ch in chars:
                    s = ch.get("start")
                    e = ch.get("end")
                    t = (ch.get("char") or ch.get("text") or "").strip()
                    if s is not None and e is not None and t and not _is_cjk_punct(t):
                        entries.append((float(s), float(e), t))
            elif words:
                for w in words:
                    s = w.get("start")
                    e = w.get("end")
                    t = (w.get("word") or w.get("text") or "").strip()
                    if s is not None and e is not None and t:
                        entries.append((float(s), float(e), t))

        entries.sort(key=lambda x: x[0])
        return entries

    @staticmethod
    def _err(msg: str, t0: float) -> Dict:
        return {
            "success": False,
            "error": msg,
            "processing_time": int((time.time() - t0) * 1000),
        }


# ═════════════════════════════════════════════════════════════════════════════
# 6. Qwen3ASRAligner
# ═════════════════════════════════════════════════════════════════════════════

class Qwen3ASRAligner(AltAlignerBase):
    """
    Qwen3-ASR 独立服务客户端
    只通过 HTTP 调用 qwen3_server.py，不在当前进程内加载模型。
    """

    DEFAULT_MODEL = "Qwen/Qwen3-ASR-1.7B"
    DEFAULT_ENDPOINT = "http://127.0.0.1:5001/asr"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        device: str = "auto",
        endpoint: str = DEFAULT_ENDPOINT,
    ):
        super().__init__()
        self.model_id = model_id
        self._device = device
        self.endpoint = endpoint.rstrip("/")
        self._session = None

    @staticmethod
    def check_available() -> Tuple[bool, str]:
        try:
            import requests  # noqa: F401
        except ImportError as e:
            return False, f"未安装 requests: pip install requests ({e})"

        try:
            r = requests.get("http://127.0.0.1:5001/", timeout=2)
            return True, "Qwen3-ASR 独立服务已可访问"
        except Exception as e:
            return False, f"Qwen3-ASR 独立服务不可访问: {e}"

    def _load_model(self):
        """
        独立服务模式下，不加载本地模型。
        这里只做轻量级连接初始化。
        """
        if self._session is None:
            self._session = requests.Session()

    def _call_qwen3_service(self, audio_path: str, language: str, context: str = "") -> Dict:
        self._load_model()

        payload = {
            "audio": audio_path,
            "language": language,
            "context": context,
        }

        resp = self._session.post(self.endpoint, json=payload, timeout=1800)
        resp.raise_for_status()
        data = resp.json()

        if not data.get("success", False):
            raise RuntimeError(data.get("error", "Qwen3-ASR 服务返回失败"))

        return data

    @staticmethod
    def _flatten_segments_to_entries(segments, int_lang: str):
        """
        将独立服务返回的 segments 转成:
        [(start_sec, end_sec, text), ...]
        """
        entries = []

        for seg in segments or []:
            text = (seg.get("text") or "").strip()
            time_stamps = seg.get("time_stamps") or seg.get("timestamp") or []

            if not text:
                continue

            # 兼容几种返回形式：
            # 1) [[s, e], [s, e], ...]
            # 2) [{"start": s, "end": e, "text": "x"}, ...]
            # 3) 单个 [s, e]
            if isinstance(time_stamps, list) and time_stamps and isinstance(time_stamps[0], list):
                # 多时间片
                if int_lang in ("zh", "yue", "ja") and len(text) > 1 and len(time_stamps) == len(text):
                    dur_each = sum((e - s) for s, e in time_stamps if s is not None and e is not None) / max(len(text), 1)
                    for i, ch in enumerate(text):
                        if i < len(time_stamps):
                            s, e = time_stamps[i]
                            if s is not None and e is not None and not _is_cjk_punct(ch):
                                entries.append((float(s), float(e), ch))
                else:
                    for item in time_stamps:
                        if isinstance(item, list) and len(item) >= 2:
                            s, e = item[0], item[1]
                            if s is not None and e is not None and not _is_cjk_punct(t):
                                entries.append((float(s), float(e), text))
            elif isinstance(time_stamps, list) and len(time_stamps) >= 2 and isinstance(time_stamps[0], (int, float)):
                s, e = time_stamps[0], time_stamps[1]
                if s is not None and e is not None and not _is_cjk_punct(t):
                    entries.append((float(s), float(e), text))
            elif isinstance(time_stamps, list) and time_stamps and isinstance(time_stamps[0], dict):
                for item in time_stamps:
                    s = item.get("start")
                    e = item.get("end")
                    t = (item.get("text") or "").strip()
                    if s is not None and e is not None and t and not _is_cjk_punct(t):
                        entries.append((float(s), float(e), t))

        return entries

    def align(self, audio_path: str, text: Optional[str], language: str) -> Dict:
        t0 = time.time()
        try:
            int_lang = _normalize_lang(language)
            asr_lang = {
                "zh": "Chinese",
                "yue": "Cantonese",
                "en": "English",
                "ja": "Japanese",
                "ko": "Korean",
            }.get(int_lang, language)

            logger.info(f"[Qwen3-ASR] 调用独立服务: {self.endpoint}")
            result = self._call_qwen3_service(
                audio_path=audio_path,
                language=asr_lang,
                context="",
            )

            transcribed = (result.get("raw_text") or "").strip()
            segments = result.get("segments") or []

            entries = self._flatten_segments_to_entries(segments, int_lang)

            logger.info(f"[Qwen3-ASR] 转录文本: {transcribed[:120]}")

            if not entries and not transcribed:
                return {
                    "success": False,
                    "error": "Qwen3-ASR 无转录结果",
                    "processing_time": int((time.time() - t0) * 1000),
                }

            # 如果独立服务只返回文本，没有时间戳，则退化为均分
            if not entries and transcribed:
                total_s = self._get_audio_duration_100ns(audio_path) / 1e7
                units = list(transcribed) if int_lang in ("zh", "yue", "ja") else transcribed.split()
                units = [u for u in units if u.strip()]
                if units:
                    dur = total_s / max(len(units), 1)
                    entries = [
                        (i * dur, (i + 1) * dur, u)
                        for i, u in enumerate(units)
                        if not _is_cjk_punct(u)
                    ]

            if not entries:
                return {
                    "success": False,
                    "error": "Qwen3-ASR 无时间戳输出",
                    "processing_time": int((time.time() - t0) * 1000),
                }

            final_text = transcribed or (text.strip() if text else "")
            if not transcribed and text:
                logger.warning("[Qwen3-ASR] 未返回转录文本，回退使用外部 text 进行后处理")
            lab = self._word_entries_to_lab(
                entries,
                final_text,
                language,
                fill_silences=False,
            )

            return {
                "success": True,
                "lab_content": lab,
                "raw_text": final_text,
                "phoneme_text": transcribed,
                "audio_duration": self._get_audio_duration_100ns(audio_path),
                "processing_time": int((time.time() - t0) * 1000),
                "backend": "qwen3_asr_http",
            }

        except Exception as e:
            logger.error(f"[Qwen3-ASR] 失败: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "processing_time": int((time.time() - t0) * 1000),
            }

# ═════════════════════════════════════════════════════════════════════════════
# 7. Qwen3ForcedAligner
# ═════════════════════════════════════════════════════════════════════════════

class Qwen3ForcedAligner(AltAlignerBase):
    DEFAULT_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B"

    def __init__(self, *args, device="cpu", **kwargs):
        super().__init__(*args, **kwargs)

        self._device = device
        self._model = None

        # ✅ 补上这一行（关键修复）
        self.model_id = kwargs.get("model_id", self.DEFAULT_MODEL)

    @staticmethod
    def check_available() -> Tuple[bool, str]:
        try:
            import qwen_asr  # noqa: F401
            return True, "qwen-asr 已就绪"
        except ImportError as e:
            return False, f"未安装 qwen-asr: pip install -U qwen-asr ({e})"

    def _load_model(self):
        if self._model is not None:
            return

        import torch
        from qwen_asr import Qwen3ForcedAligner as Qwen3FA

        device = getattr(self, "_device", "cpu")
        dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32

        self._model = Qwen3FA.from_pretrained(
            self.model_id,
            dtype=dtype,
            device_map=device,
        )
        # qwen_asr.Qwen3ForcedAligner 是一个普通包装类（非 nn.Module），
        # 没有 .eval() 方法；真正的底层模型在 self._model.model 上，
        # 且 align() 内部已用 @torch.inference_mode() 包裹，因此无需也不能调用 .eval()

    def align(self, audio_path: str, text: Optional[str], language: str) -> Dict:
        t0 = time.time()
        if not text:
            return {
                "success": False,
                "error": "Qwen3-ForcedAligner 需要参考文本（text 不能为空）",
                "processing_time": 0,
            }

        try:
            self._load_model()

            lang_name = _to_qwen_lang_name(language)
            if not lang_name:
                return {
                    "success": False,
                    "error": f"Qwen3-ForcedAligner 不支持语言: {language}",
                    "processing_time": int((time.time() - t0) * 1000),
                }

            results = self._model.align(
                audio=audio_path,
                text=text,
                language=lang_name,
            )

            # 官方示例里 results[0][0].text / start_time / end_time
            word_entries = []
            for item in (results[0] if results else []):
                tok = (getattr(item, "text", "") or "").strip()
                if not tok or _is_cjk_punct(tok):
                    continue
                word_entries.append(
                    (float(item.start_time), float(item.end_time), tok)
                )

            if not word_entries:
                return {
                    "success": False,
                    "error": "Qwen3-ForcedAligner 无对齐输出",
                    "processing_time": int((time.time() - t0) * 1000),
                }

            lab = self._word_entries_to_lab(
                word_entries, text, language, fill_silences=False
            )

            return {
                "success": True,
                "lab_content": lab,
                "raw_text": text,
                "phoneme_text": text,
                "audio_duration": self._get_audio_duration_100ns(audio_path),
                "processing_time": int((time.time() - t0) * 1000),
                "backend": "qwen3_aligner",
            }

        except Exception as e:
            logger.error(f"[Qwen3-FA] 失败: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "processing_time": int((time.time() - t0) * 1000),
            }

    def _extract_timestamps(
        self,
        outputs,
        text: str,
        int_lang: str,
        total_sec: float,
    ) -> List[Tuple[float, float, str]]:
        import torch

        entries: List[Tuple[float, float, str]] = []

        # ── 方案 A：CTC logits + torchaudio forced_align ─────────────────
        if hasattr(outputs, "logits"):
            try:
                import torchaudio
                logits    = outputs.logits[0].float()
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                vocab     = self._processor.tokenizer.get_vocab()
                tokens    = self._processor.tokenizer(
                    text, return_tensors="pt"
                ).input_ids[0]
                frame_dur = total_sec / logits.shape[0]

                if hasattr(torchaudio.functional, "forced_align"):
                    aligned = torchaudio.functional.forced_align(
                        log_probs.unsqueeze(0).cpu(),
                        tokens.unsqueeze(0),
                        blank=vocab.get("<pad>", 0),
                    )
                    spans = torchaudio.functional.merge_tokens(
                        aligned[0][0], self._processor.tokenizer.pad_token_id
                    )
                    id2tok = {v: k for k, v in vocab.items()}
                    for span in spans:
                        tok = id2tok.get(span.token, "").lstrip("▁").strip()
                        if tok:
                            entries.append((
                                span.start * frame_dur,
                                span.end   * frame_dur,
                                tok,
                            ))
                else:
                    # 无 forced_align：贪婪 CTC 解码
                    pred_ids = torch.argmax(logits, dim=-1).cpu().numpy()
                    blank_id = vocab.get("<pad>", 0)
                    id2tok   = {v: k for k, v in vocab.items()}
                    cur_tok, cur_start = None, 0
                    for i, tid in enumerate(pred_ids):
                        if int(tid) == blank_id:
                            if cur_tok is not None:
                                tok = id2tok.get(cur_tok, "").lstrip("▁").strip()
                                if tok:
                                    entries.append((
                                        cur_start * frame_dur, i * frame_dur, tok
                                    ))
                                cur_tok = None
                        else:
                            if cur_tok != int(tid):
                                if cur_tok is not None:
                                    tok = id2tok.get(cur_tok, "").lstrip("▁").strip()
                                    if tok:
                                        entries.append((
                                            cur_start * frame_dur, i * frame_dur, tok
                                        ))
                                cur_tok, cur_start = int(tid), i
                    if cur_tok is not None:
                        tok = id2tok.get(cur_tok, "").lstrip("▁").strip()
                        if tok:
                            entries.append((
                                cur_start * frame_dur,
                                len(pred_ids) * frame_dur,
                                tok,
                            ))
            except Exception as e:
                logger.warning(f"[Qwen3-FA] CTC 时间戳提取失败: {e}")

        # ── 方案 B：Seq2Seq timestamp token ──────────────────────────────
        if not entries and hasattr(outputs, "sequences"):
            try:
                decoded = self._processor.batch_decode(
                    outputs.sequences, output_offsets=True, skip_special_tokens=True
                )
                for item in decoded:
                    if isinstance(item, dict) and "chunks" in item:
                        for chunk in item["chunks"]:
                            ts  = chunk.get("timestamp") or (None, None)
                            tok = (chunk.get("text") or "").strip()
                            if ts[0] is not None and tok:
                                entries.append((
                                    float(ts[0]),
                                    float(ts[1] or ts[0] + 0.25),
                                    tok,
                                ))
            except Exception as e:
                logger.warning(f"[Qwen3-FA] Seq2Seq 时间戳提取失败: {e}")

        # ── 降级：均匀分配 ────────────────────────────────────────────────
        if not entries:
            logger.warning("[Qwen3-FA] 无法提取时间戳，改用均匀分配（精度较低）")
            units = list(text) if int_lang in ("zh", "yue", "ja") else text.split()
            units = [u for u in units if u.strip() and not _is_cjk_punct(u)]
            if units:
                dur = total_sec / len(units)
                entries = [(i * dur, (i + 1) * dur, u) for i, u in enumerate(units)]

        return entries


# ═════════════════════════════════════════════════════════════════════════════
# 8. 单例缓存与工厂函数
# ═════════════════════════════════════════════════════════════════════════════

_SINGLETON: Dict[str, AltAlignerBase] = {}


def get_aligner(backend: str, device: str = "auto", **kwargs) -> AltAlignerBase:
    """
    工厂函数：按 backend 名称创建或复用对齐器单例。
    backend: "whisperx" | "qwen3_asr" | "qwen3_aligner"
    """
    global _SINGLETON
    if backend not in _SINGLETON:
        if backend == "whisperx":
            _SINGLETON[backend] = WhisperXAligner(device=device, **kwargs)
        elif backend == "qwen3_asr":
            _SINGLETON[backend] = Qwen3ASRAligner(device=device, **kwargs)
        elif backend == "qwen3_aligner":
            _SINGLETON[backend] = Qwen3ForcedAligner(device=device, **kwargs)
        else:
            raise ValueError(f"未知对齐后端: {backend}")
    return _SINGLETON[backend]


def get_alt_aligner_status() -> Dict:
    """检查所有替代对齐后端的可用状态（含模型文件目录信息）"""
    wx_ok, wx_msg = WhisperXAligner.check_available()
    qa_ok, qa_msg = Qwen3ASRAligner.check_available()
    qf_ok, qf_msg = Qwen3ForcedAligner.check_available()

    return {
        "models_dir": str(_MODELS_DIR),        # ← 前端可展示此路径
        "whisperx": {
            "available":     wx_ok,
            "message":       wx_msg,
            "requires_text": False,
            "description":   "WhisperX (Whisper ASR + wav2vec2 强制对齐)",
        },
        "qwen3_asr": {
            "available":     qa_ok,
            "message":       qa_msg,
            "requires_text": False,
            "description":   "Qwen3-ASR-1.7B (自动语音识别)",
            "model_paths": {
                "hf_cache": str(_HF_HUB),
            },
        },
        "qwen3_aligner": {
            "available":     qf_ok,
            "message":       qf_msg,
            "requires_text": True,
            "description":   "Qwen3-ForcedAligner-0.6B (强制对齐)",
            "model_paths": {
                "hf_cache": str(_HF_HUB),
            },
        },
    }