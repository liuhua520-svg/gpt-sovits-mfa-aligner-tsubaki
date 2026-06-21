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
import warnings
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# 屏蔽 pyannote.audio 在 torchcodec DLL 找不到时输出的 UserWarning
# （非致命：pyannote 会自动回退到其他解码后端）
warnings.filterwarnings(
    "ignore",
    message=r".*torchcodec.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*torchaudio\._backend\.list_audio_backends.*",
    category=UserWarning,
)


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
_TORCH_CACHE: Path = _MODELS_DIR / "torch_cache"
_TORCH_CACHE.mkdir(parents=True, exist_ok=True)

for _d in (_HF_CACHE, _HF_HUB, _WHISPER_CACHE):
    _d.mkdir(parents=True, exist_ok=True)

# 将 HuggingFace 缓存重定向到 backend/models/hf_cache/
# 使用 setdefault 不覆盖用户已配置的环境变量
os.environ.setdefault("HF_HOME",                       str(_HF_CACHE))
os.environ.setdefault("HF_HUB_CACHE",                  str(_HF_HUB))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE",         str(_HF_HUB))
os.environ.setdefault("TRANSFORMERS_CACHE",            str(_HF_HUB))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")   # 消除 Windows 警告
os.environ["TORCH_HOME"] = str(_TORCH_CACHE)

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
import re  # Bug修复: normalize_text_for_whisperx 函数体使用裸 re 名，需同时暴露非别名版本

def normalize_text_for_whisperx(text: str, lang: str = "zh") -> str:
    """
    轻量清洗版本（保留句子结构）
    专为 forced alignment / VOCALOID / SynthV 设计
    """

    if not text:
        return text

    # 1. 去 Markdown 标题
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)

    # 2. 去列表符号（保留内容）
    text = re.sub(r"(?m)^\s*(\d+[.、）)]|[①②③④⑤⑥⑦⑧⑨⑩]+|[一二三四五六七八九十]+[、.])\s*", "", text)

    # 3. ⚠️只替换“部分标点”，保留句末符号！
    text = re.sub(r"[「」『』【】《》〈〉]", " ", text)

    # 4. 英文引号/括号
    text = re.sub(r'[\"\'()\[\]{}]', " ", text)

    # 5. 空白统一
    text = re.sub(r"[ \t\r]+", " ", text)

    return text.strip()


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


def _clean_align_text(text: str) -> str:
    """
    清洗送入强制对齐模型（wav2vec2 / whisperx.align）的文本：剥离标点
    符号，但保留空白（词边界）和单词内部撇号（英语缩略形式如
    "what's"）。

    【关键 bug 修复】此前直接复用 _is_cjk_punct() 逐字符过滤这段文本：
    该函数把空白字符（Unicode 类别 Z*，包括普通空格）也判定为"标点"
    一并清除，导致任何依赖空格分词的语言（英语、韩语等）在送进
    whisperx.align() 之前所有空格被吃掉、整句被拼接成一个不可分割
    的"伪单词"（如 "Hello world, What's Up" → "HelloworldWhatsUp"）。
    wav2vec2 按空白切词得到的 words 列表因此永远只有 1 个跨越全句的
    条目，下游既无法按英语单词切分做逐词 G2P，也无法在 ASR 词典/
    g2p_en 中查到这个被拼接出来的生造词，只能整句原样兜底输出。
    中文/日语本身走字符级（chars）通道、不依赖空格分词，这个 bug
    此前被掩盖，只在英语/韩语（走 words 通道）上暴露。

    撇号特殊保留：去掉撇号会让 "what's" 在发音词典里查不到对应词条
    （词典存的是 "what's" 而非 "whats"），被迫退化为整词输出。
    """
    if not text:
        return text
    out_chars: List[str] = []
    for ch in text:
        if ch.isspace():
            out_chars.append(" ")
            continue
        if ch in ("'", "\u2019"):   # ASCII 撇号 / 右单引号（缩略形式）
            out_chars.append(ch)
            continue
        if _is_cjk_punct(ch):
            continue
        out_chars.append(ch)
    return _re.sub(r"\s+", " ", "".join(out_chars)).strip()


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


# ── 句末/句内标点 → 停顿时长映射 ──────────────────────────────────────────────
_HEAVY_END_PUNCT = "。！？…!?"
_LIGHT_END_PUNCT = "、，；,;"

# 内部静音占位符。必须为小写 "sil"，原因：
#   1. MFAProcessor.SIL_PHONES（mfa_processor.py）以小写匹配 "sil"/"sp"/"spn" 等，
#      _process_zh_words / _process_en_words / _process_yue_words 命中后一律
#      原样转换为 LAB 中的字面量 "sil"，且不消耗参考文本的拼音/音节配额。
#   2. phoneme_converter.merge_lab_silence() 把字面量 "sil" 当作"不可修改/
#      不可删除"的边界标记（用于正确处理相邻 "-" 辅音声母标记的吸收/丢弃逻辑），
#      而 "sp" 等其他变体不享有这条特殊保护——日语路径（_ja_entries_to_lab）
#      会用到这条规则，因此这里统一用 "sil" 而不是 "sp"。
_SIL_MARK = "sil"

# ─────────────────────────────────────────────────────────────────────────
# SudachiPy 惰性单例（替代 pykakasi，避免 GPL-3.0 许可传染；
# SudachiPy 为 Apache-2.0 许可）。
# 词典加载较慢，进程内只创建一次。
# ─────────────────────────────────────────────────────────────────────────
_sudachi_tokenizer_obj = None
_sudachi_split_mode_obj = None


def _get_sudachi_tokenizer():
    """返回 (tokenizer_obj, split_mode) 单例，首次调用时加载词典。"""
    global _sudachi_tokenizer_obj, _sudachi_split_mode_obj
    if _sudachi_tokenizer_obj is None:
        from sudachipy import dictionary, tokenizer as sudachi_tokenizer
        _sudachi_tokenizer_obj = dictionary.Dictionary().create()
        # SplitMode.C：最长单位切分（更接近自然分词，多音字读音更准确）
        _sudachi_split_mode_obj = sudachi_tokenizer.Tokenizer.SplitMode.C
    return _sudachi_tokenizer_obj, _sudachi_split_mode_obj


def _kata_to_hira(text: str) -> str:
    """片假名 → 平假名（纯 Unicode 码位偏移，无第三方依赖）。

    片假名 U+30A1-U+30F6 与平假名 U+3041-U+3096 之间存在固定偏移 0x60；
    非片假名字符（如长音符 'ー'、促音、汉字、标点）原样保留。
    """
    out = []
    for ch in text:
        code = ord(ch)
        if 0x30A1 <= code <= 0x30F6:
            out.append(chr(code - 0x60))
        else:
            out.append(ch)
    return "".join(out)


# 拗音 / 外来语小假名：与前一个假名合并为同一个 mora（如 'きゃ' 算 1 个
# mora，而不是 2 个）。促音「っ」和拨音「ん」不在此列，它们各自单独成 mora。
_JA_SMALL_KANA = frozenset("ゃゅょぁぃぅぇぉ")


def _split_ja_mora(text: str) -> List[str]:
    """
    把 Sudachi reading_form() 给出的假名读音字符串拆分为 mora 列表。

    用途：_ja_entries_to_lab() 中，多字符 morpheme（如「気持ち」→ きもち）
    原本会把整段读音揉成一条跨越全词时间区间的 LAB 行，丢弃了 wav2vec2
    本来给出的逐字符时间戳。当 morpheme 的表层字数与读音 mora 数恰好相等
    时（纯假名词、大多数简单汉字词都满足），可以把读音逐字拆回各自原始
    时间戳，恢复逐字精度；字数与 mora 数不一致时（典型如「大変」2 字
    对应 たいへん 4 mora、「僕」1 字对应 ぼく 2 mora），由
    _distribute_mora_across_chars() 把 mora 尽量均匀地分配给各个原始
    字符、再在每个字符自身时间戳内部按 mora 数等分，不再退回整词合并。
    """
    morae: List[str] = []
    for ch in text:
        if ch in _JA_SMALL_KANA and morae:
            morae[-1] += ch
        else:
            morae.append(ch)
    return morae


def _distribute_mora_across_chars(
    piece: List[Tuple[float, float, str]],
    mora_list: List[str],
) -> List[str]:
    """
    把一个 morpheme 的 mora 序列分配给该 morpheme 对应的原始字符时间戳
    （piece），用于字符数与 mora 数不一致的情况（如「僕」1 字对应
    ぼく 2 mora、「大変」2 字对应 たいへん 4 mora）。

    【这是本次修复的核心】此前遇到这种不一致就直接放弃逐字精度，把
    整段读音揉成一条跨越整个 morpheme 时间区间的 LAB 行（见
    _split_ja_mora 旧版文档字符串），这正是用户反馈"音标连在一起，
    不是一个假名一个音标"的根因——「僕」「本」「大変」等词被输出成
    单条 LAB 行，内部塞进了 2~4 个假名。

    分配规则：
      1. mora 总数（通常 ≥ 字符数，因为一个汉字常对应 1~2 个 mora）
         按字符顺序尽量平均分组（divmod 取整，余数分给靠前的字符），
         每个字符至少分到 1 个 mora。
      2. 罕见的反向情况（字符数多于 mora 数）：分不到 mora 的字符并入
         前一个字符的分组，与其共享同一组 mora 的时间戳，避免产生
         空区间。
      3. 每个字符（或合并后的字符组）分到的 mora 子序列，在该字符
         自身已有的 [start, end] 时间戳内部按数量等分——只细分已知
         的字符级时间戳，不触碰字符之间的边界，不会引入新的跨字符
         错位。这是在"没有逐 mora 强制对齐证据"前提下合理的最佳近似，
         且明显优于把整个词压成一条目。
    """
    n_chars = len(piece)
    n_mora = len(mora_list)
    if n_chars == 0 or n_mora == 0:
        return []

    if n_mora >= n_chars:
        base, rem = divmod(n_mora, n_chars)
        counts = [base + (1 if i < rem else 0) for i in range(n_chars)]
    else:
        # 字符数多于 mora 数：前 n_mora 个字符各分 1 个，其余分 0，
        # 稍后并入前一组。
        counts = [1] * n_mora + [0] * (n_chars - n_mora)

    merged_piece: List[Tuple[float, float]] = []
    merged_counts: List[int] = []
    for (ps, pe, _ch), c in zip(piece, counts):
        if c == 0 and merged_piece:
            prev_s, _prev_e = merged_piece[-1]
            merged_piece[-1] = (prev_s, pe)
        else:
            merged_piece.append((ps, pe))
            merged_counts.append(c)
    if merged_counts and merged_counts[0] == 0:
        # 防御性兜底：理论上不会出现（n_mora>=1 时第一个字符必分到
        # 至少 1 个 mora），仅避免万一发生时丢失这组 mora。
        merged_counts[0] = 1

    lines: List[str] = []
    mora_idx = 0
    for (ps, pe), c in zip(merged_piece, merged_counts):
        group = mora_list[mora_idx: mora_idx + c]
        mora_idx += c
        if not group:
            continue
        if len(group) == 1:
            lines.append(f"{int(ps*10_000_000)} {int(pe*10_000_000)} {group[0]}")
            continue
        dur = pe - ps
        step = dur / len(group)
        for j, mora in enumerate(group):
            sub_s = ps + step * j
            sub_e = pe if j == len(group) - 1 else ps + step * (j + 1)
            lines.append(f"{int(sub_s*10_000_000)} {int(sub_e*10_000_000)} {mora}")

    return lines


def _bind_ref_text_by_asr_count(
    cleaned_ref: str,
    raw_segments: List[Dict],
    int_lang: str,
) -> bool:
    """
    按"每段 ASR 自己识别出的可发音字符数"作为配额，把完整参考文本
    （保留原始标点）按顺序切给对应的段，而不是直接放弃参考文本退回
    ASR 识别结果。

    关键点（区别于按时长比例分配的早期方案）：
    分配用的配额是每段 ASR 文本自身的字符数，而不是音频时长。
    这保证替换后 seg_text 的可发音字符数与该段稍后实际送入
    wav2vec2 做强制对齐的字符数完全一致——避免"参考文本字数与该段
    音频实际内容字数不匹配 → 强制对齐被迫拉伸/压缩 → 时间戳错位"
    的问题。按时长比例分配做不到这一点：时长占比只是粗略估算，
    跟该段 ASR 真实识别出的字符数可能完全不是一回事。

    全局总字数（ASR 总识别字数 vs 参考文本总字数）几乎总会有一定差异
    ——音频越长，ASR 漏听/多听的绝对字数也越容易变多，这是正常现象，
    不代表参考文本就跟音频对不上。早期版本一旦差异超过阈值就整体放弃
    替换，导致长文本几乎必然退回 ASR 的（几乎不带标点的）识别文本，
    这正是"长文本反而又粘连"的根因。

    现在改为：按比例把每段配额整体缩放，使配额总和精确等于参考文本
    总字数（用最大余数法取整），把全局差异均摊到每一段上，而不是
    集中甩给某一段或整体放弃——每段拿到的字符数仍然是一个跟它自身
    原始 ASR 字数接近的整数，强制对齐不会被某一段突然暴增/清零的
    字数带偏。

    会原地修改 raw_segments 中每个元素的 "text" 字段。
    返回 True 表示已替换；False 仅在彻底无法估算配额时返回（如总字数为 0）。
    """
    def _is_spoken(ch: str) -> bool:
        if int_lang in ("zh", "yue"):
            return '\u4e00' <= ch <= '\u9fff' or '\u3400' <= ch <= '\u4dbf'
        if int_lang == "ja":
            return '\u3040' <= ch <= '\u30ff' or '\u4e00' <= ch <= '\u9fff'
        return ch.strip() != "" and not _is_cjk_punct(ch)

    # 每段 ASR 自己的可发音字符数 = 该段稍后会送入 wav2vec2 对齐的字符数
    quotas = [
        sum(1 for ch in (seg.get("text", "") or "") if _is_spoken(ch))
        for seg in raw_segments
    ]
    total_quota = sum(quotas)
    ref_spoken_total = sum(1 for ch in cleaned_ref if _is_spoken(ch))

    if total_quota == 0 or ref_spoken_total == 0:
        return False

    diff_ratio = abs(total_quota - ref_spoken_total) / max(ref_spoken_total, 1)
    logger.info(
        f"[alt_aligners] ASR 总识别字数={total_quota}，参考文本总字数="
        f"{ref_spoken_total}，差异 {diff_ratio:.1%}（按比例缩放各段配额，"
        "均摊差异，不整体放弃参考文本）"
    )

    # 按比例缩放配额，使其总和精确等于参考文本总字数（最大余数法取整）
    scale = ref_spoken_total / total_quota
    scaled = [q * scale for q in quotas]
    int_quotas = [int(x) for x in scaled]
    remainder = ref_spoken_total - sum(int_quotas)
    if remainder > 0:
        order = sorted(
            range(len(scaled)), key=lambda i: scaled[i] - int_quotas[i], reverse=True
        )
        for i in order[:remainder]:
            int_quotas[i] += 1

    cum_quota: List[int] = []
    acc = 0
    for q in int_quotas:
        acc += q
        cum_quota.append(acc)

    chunks = ["" for _ in raw_segments]
    spoken_seen = 0
    seg_idx = 0
    for ch in cleaned_ref:
        if _is_spoken(ch):
            spoken_seen += 1
            while seg_idx < len(cum_quota) - 1 and spoken_seen > cum_quota[seg_idx]:
                seg_idx += 1
        chunks[seg_idx] += ch

    for seg, chunk in zip(raw_segments, chunks):
        seg["text"] = chunk.strip()
    return True


def _inject_sentence_pauses(
    seg_entries: List[Tuple[float, float, str]],
    seg_text: str,
    heavy_gap_sec: float = 0.08,
    light_gap_sec: float = 0.04,
    sil_mark: str = _SIL_MARK,
) -> List[Tuple[float, float, str]]:
    """
    在 seg_text 中句末/句内标点出现的位置，插入真正的静音条目
    （mark=sil_mark），而不只是挪动相邻字符的时间戳。

    【修复说明】旧版本只挪动了前一个字符的结束时间，在数值上腾出了
    一段空隙，但没有写入任何 SIL/SP 标记。下游 _word_entries_to_lab()
    会原样把这段"空隙"写进 LAB——但 LAB 里相邻两行时间戳之间的数值
    间隔，并不会被 SVP 工程生成阶段（tsubaki_processor._is_true_silence()
    / 步骤①"如果是 lab 里的显式静音标签 sil/pau/sp，直接跳过不生成音符"）
    识别为停顿，因为那一步只检查"这一行的 label 是不是静音词"，根本
    不会去看前后两行时间戳之间是否存在数值间隙。所以旧版本制造出的
    停顿在 LAB 里确实存在（时间戳对得上），但在 SVP 里完全不可见，
    音符还是会首尾相连——这正是"gan、ge、shu、ling 等句末音标连在
    一起"的根因。

    现在改为：
      1. 若该字符与下一个字符之间天然就有 ≥ target_gap 的时间间隙
         （wav2vec2 偶尔会留下这种间隙），直接把整个天然间隙转换成
         一条 sil 条目。
      2. 若天然间隙不够，从当前字符尾部"偷"出最多 target_gap 的时长
         （且最多偷取该字符自身时长的一半，避免压出负数/异常短音），
         在腾出来的空隙里插入 sil 条目。
      3. 不改变原有字符的相对顺序、不跨字符插队，因此不会引入新的
         拼音/音节错位；插入的 sil 条目会被 MFAProcessor 的
         _process_*_words() 识别（mark in SIL_PHONES）并原样转换为
         LAB 中的字面量 "sil"，不消耗参考文本的拼音配额（syl_index
         不会因为这条 sil 而前进）。

    heavy_gap_sec : 句末强标点（。！？…）对应的停顿时长
    light_gap_sec : 句内轻标点（、，；）对应的停顿时长
    sil_mark      : 插入的静音占位符文本（须为小写，与 SIL_PHONES 命名一致）

    注：对英文等以单词为单位对齐的语言，本函数仍按"参考文本里的单个
    字符"逐一计算停顿位置，与 entries（单词级）并非严格一一对应——
    这是沿用自旧版本的已知局限，不在本次修复范围内（本次修复的对象
    是中／日逐字对齐路径，这里维持原有行为不做改动）。
    """
    if not seg_entries or not seg_text:
        return seg_entries

    gap_after: List[float] = []
    n = len(seg_text)
    i = 0
    while i < n:
        ch = seg_text[i]
        if _is_cjk_punct(ch) or ch.isspace():
            i += 1
            continue
        j = i + 1
        while j < n and seg_text[j].isspace():
            j += 1
        gap = 0.0
        if j < n:
            if seg_text[j] in _HEAVY_END_PUNCT:
                gap = heavy_gap_sec
            elif seg_text[j] in _LIGHT_END_PUNCT:
                gap = light_gap_sec
        gap_after.append(gap)
        i += 1

    m = min(len(seg_entries), len(gap_after))
    result: List[Tuple[float, float, str]] = []
    for k in range(len(seg_entries)):
        s, e, t = seg_entries[k]

        if k < m - 1:
            target_gap = gap_after[k]
            if target_gap > 0:
                next_s = seg_entries[k + 1][0]
                avail = next_s - e
                if avail < target_gap:
                    shrink = min(target_gap - avail, max(0.0, (e - s) * 0.5))
                    if shrink > 0:
                        e = e - shrink
                result.append((s, e, t))
                # 腾出的空隙（天然间隙 + 借用时长）写成一条真正的 sil 条目，
                # 而不是仅仅留下一个数值上的空隙。
                if next_s > e:
                    result.append((e, next_s, sil_mark))
                continue

        result.append((s, e, t))

    return result


def _refine_sil_boundaries_by_energy(
    seg_entries: List[Tuple[float, float, str]],
    cropped_audio,
    sr: int,
    seg_start_sec: float,
    sil_mark: str = _SIL_MARK,
    frame_sec: float = 0.01,
    hop_sec: float = 0.005,
    rel_threshold: float = 0.06,
    abs_floor: float = 0.0008,
    abs_ceiling: float = 0.003,
    max_extra_claim_sec: float = 0.4,
    min_keep_sec: float = 0.06,
) -> List[Tuple[float, float, str]]:
    """
    用该句真实音频的短时能量，把 _inject_sentence_pauses() 插入的固定
    时长（40ms/80ms）sil 条目向左右两侧扩展到真正安静的区域边界。

    【背景】_inject_sentence_pauses() 给出的 40/80ms 只是"标点处至少要
    有多长停顿"的下限，不代表音频里真实停顿就是这么短。wav2vec2 字符级
    强制对齐没有静音词表，经常会把真正的换气/停顿时间错误地"焊"进
    紧邻标点前最后一个字的时长里——例如"令"字符的对齐区间可能一路
    延伸到下一句开始前，把演唱者实际换气的两三百毫秒真静音都算成
    "令"的发音时长，而 _inject_sentence_pauses() 只能在这段被吃掉的
    时长里再"借"出 40/80ms，借不出真正的停顿长度。

    【做法】对该句裁剪出的真实音频（cropped_audio）做短时 RMS 能量
    扫描，对每条 sil 条目：
      - 从 sil 起点向左探测：能量持续低于阈值就持续把 sil 起点往左推
        （即从前一个字符"偷"时长），直到能量回升到阈值以上（说明已
        经进入前一个字符真正的发声区）、或达到 max_extra_claim_sec
        上限、或前一个字符被压到 min_keep_sec 下限为止。
      - 从 sil 终点向右探测同理，找到下一个字符真正的发声起点。
      - 若标点处其实是连唱、从一开始能量就已经偏高（没有真实停顿），
        探测会在第一步就停下，不做任何扩展——保留
        _inject_sentence_pauses() 给出的最小停顿即可，不会把正常的
        延音误判为停顿来源。

    阈值 = clip(rel_threshold × 该句 70 分位能量, abs_floor, abs_ceiling)，
    既能随录音整体响度自适应，又不会因为某一句特别响/特别轻而跑到
    不合理的区间——这三个常量都是用本项目实际样例反复试出来的安全区间，
    不是理论推导值，如果某些素材效果不理想，优先调整 rel_threshold。

    注：本函数只移动已存在的 sil / 相邻字符边界，不会增删条目，因此
    不会打乱 _process_*_words() 的拼音/音节配额对应关系。
    """
    if not seg_entries or cropped_audio is None:
        return seg_entries
    try:
        import numpy as _np
    except ImportError:
        return seg_entries

    n_samples = len(cropped_audio)
    if n_samples == 0:
        return seg_entries

    frame_n = max(1, int(frame_sec * sr))
    hop_n = max(1, int(hop_sec * sr))
    n_frames = max(0, (n_samples - frame_n) // hop_n + 1)
    if n_frames < 2:
        return seg_entries

    audio64 = _np.asarray(cropped_audio, dtype=_np.float64)
    rms = _np.empty(n_frames, dtype=_np.float64)
    for fi in range(n_frames):
        st = fi * hop_n
        chunk = audio64[st: st + frame_n]
        rms[fi] = float(_np.sqrt(_np.mean(chunk * chunk))) if len(chunk) else 0.0

    voiced_level = float(_np.percentile(rms, 70))
    threshold = max(abs_floor, min(rel_threshold * voiced_level, abs_ceiling))

    def _rms_at(local_t: float) -> float:
        idx = int(round(local_t * sr / hop_n))
        idx = max(0, min(n_frames - 1, idx))
        return rms[idx]

    seg_len_sec = n_samples / float(sr)
    result = list(seg_entries)
    n = len(result)
    for k in range(1, n - 1):
        s, e, t = result[k]
        if t != sil_mark:
            continue
        prev_s, prev_e, prev_t = result[k - 1]
        next_s, next_e, next_t = result[k + 1]
        if prev_t == sil_mark or next_t == sil_mark:
            continue   # 理论上不会相邻出现两条 sil，安全起见跳过

        sil_s_local = s - seg_start_sec
        sil_e_local = e - seg_start_sec

        left_limit = max(
            0.0,
            (prev_s - seg_start_sec) + min_keep_sec,
            sil_s_local - max_extra_claim_sec,
        )
        probe = sil_s_local
        while probe - hop_sec >= left_limit and _rms_at(probe - hop_sec) < threshold:
            probe -= hop_sec
        new_sil_s_local = probe

        right_limit = min(
            seg_len_sec,
            (next_e - seg_start_sec) - min_keep_sec,
            sil_e_local + max_extra_claim_sec,
        )
        probe = sil_e_local
        while probe + hop_sec <= right_limit and _rms_at(probe + hop_sec) < threshold:
            probe += hop_sec
        new_sil_e_local = probe

        if new_sil_e_local <= new_sil_s_local:
            continue   # 探测结果异常（区间反转）则保持原状，不做改动

        new_sil_s = new_sil_s_local + seg_start_sec
        new_sil_e = new_sil_e_local + seg_start_sec

        result[k - 1] = (prev_s, new_sil_s, prev_t)
        result[k]     = (new_sil_s, new_sil_e, sil_mark)
        result[k + 1] = (new_sil_e, next_e, next_t)

    return result


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
        # 注：entries_n 要排除我们自己插入的静音条目（_inject_sentence_pauses
        # 写入的 sil 标记），否则每次插入停顿都会触发一次误报式警告。
        if text and lang in ("zh", "yue", "ja", "ko"):
            spoken_n = _count_spoken_chars(text, lang)
            entries_n = sum(
                1 for _, _, w in word_entries
                if (w or "").strip().lower() not in self._mfa.SIL_PHONES
            )
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
        """
        【修复说明】word_entries 现在可能混入 _inject_sentence_pauses() 插入的
        静音条目（mark == _SIL_MARK == "sil"）。这些条目必须在送入
        假名转换之前被摘出来，否则会被当作日文文本误转换；
        更重要的是，build_ja_hiragana_lab() 的元音/辅音状态机会把任何
        非元音、非鼻音的输入当作"待合并的辅音声母"（pending）。

        修复方式：先把静音条目摘出来单独保存，只把真正的发音字符喂给
        SudachiPy；状态机跑完之后，再按时间顺序把静音条目插回，最后统一
        交给 merge_lab_silence()——它对字面量 "sil" 有专门的"不可修改/
        不可删除"保护（见模块顶部 _SIL_MARK 的注释）。

        【假名转换说明】原实现使用 pykakasi 逐字符调用 convert()。
        pykakasi 是 GPL-3.0 许可，会对本项目产生许可传染，故替换为
        Apache-2.0 许可的 SudachiPy（本项目其他模块已依赖该库）。
        SudachiPy 是分词器，逐字符调用会丢失上下文，导致多音字读音
        变差（例如「今」单独转换得到「いま」，但在「今日」中应为
        「きょう」的「きょ」部分）。因此这里改为对整句拼接文本做一次
        tokenize，按各 morpheme 的 surface 字符数把读音重新切回到
        每个原始字符的时间戳上，多音字消歧效果优于逐字符调用。

        【本次修复 - 关键 bug】此前的实现在拿到 Sudachi 的 morpheme 读音
        （已经是最终假名，如 "きもち"/"さんぷる"）之后，又把这些 LAB 行
        送进了 phoneme_converter.build_ja_hiragana_lab()。但那个函数的
        状态机只认识"单个罗马字音素"（MFA japanese_mfa 输出的 a/i/u/e/o
        或 ky/ny 等单字母/双字母音素），任何不是元音、不是鼻音的输入都会
        被当成"待合并的辅音声母"塞进 pending，等下一条目到来时被 flush
        成字面量 "-"。结果是：除最后一条之外的*每一条*假名读音都被强制
        改写成 "-"，循环结束后最后一条也在收尾 flush 中变成 "-"。
        merge_lab_silence() 随后处理这些 "-"：规则是"前面没有可吸收的
        真实音节（非 -/非 sil）就直接删除"——因为这时全部都是 "-"，没有
        任何条目能当吸收对象，于是全部被删除，只剩 _inject_sentence_pauses()
        写入的字面量 "sil"。这正是"WhisperX/Qwen3 对齐日语文本后只剩
        sil、没有任何音标"的根因（已用合成时间戳复现并验证修复有效）。

        修复：Sudachi 给出的读音已经是最终假名形式，不需要也不能再走一遍
        "罗马音素→假名"的状态机，直接跳过 build_ja_hiragana_lab()。

        【顺带增强，后续修复见 _distribute_mora_across_chars】多字符
        morpheme 原本把整段读音揉成一条跨越全词时间区间的 LAB 行（如
        「気持ち」→ 一条 "きもち"），丢弃了 wav2vec2 本来给出的逐字符
        时间戳。当 morpheme 的表层字数与读音 mora 数（经 _split_ja_mora
        拆分）恰好相等时，把读音逐字拆回各自原始时间戳，恢复逐字精度
        （典型如纯假名 ASR 文本、大多数简单汉字词）；数量不一致时
        （典型如「大変」2 字对应 たいへん 4 mora、「僕」1 字对应 ぼく
        2 mora）不再退回整词合并区块——那正是用户反馈"音标连在一起，
        不是一个假名一个音标"的根因。现在由 _distribute_mora_across_chars()
        把 mora 尽量均匀分配给各个原始字符、再在每个字符自身时间戳内部
        按 mora 数等分，逐 mora 输出，仅在分配粒度上做近似，不再合并
        成一条目（详见该函数顶部说明）。
        """
        sil_entries: List[Tuple[int, int, str]] = []
        spoken_entries: List[Tuple[float, float, str]] = []
        for s, e, ch in word_entries:
            ch = (ch or "").strip()
            if ch.lower() in self._mfa.SIL_PHONES:
                sil_entries.append((int(s * 10_000_000), int(e * 10_000_000), _SIL_MARK))
            else:
                spoken_entries.append((s, e, ch))

        # 过滤标点，得到真正参与假名转换的字符序列（保留原时间戳）
        char_entries = [
            (s, e, ch) for s, e, ch in spoken_entries
            if ch and not _is_cjk_punct(ch)
        ]

        try:
            from sudachipy import dictionary as _sudachi_dictionary  # noqa: F401
        except ImportError:
            lines = [
                f"{int(s*10_000_000)} {int(e*10_000_000)} {ch}"
                for s, e, ch in char_entries
            ]
        else:
            joined_text = "".join(ch for _, _, ch in char_entries)
            lines = []
            if joined_text:
                tok, mode = _get_sudachi_tokenizer()
                morphemes = tok.tokenize(joined_text, mode)

                # 把逐字符时间戳和 morpheme 切分对齐：按每个 morpheme 的
                # surface 长度，依次"消费"对应数量的原始字符时间戳。
                idx = 0  # char_entries 游标
                for m in morphemes:
                    surface = m.surface()
                    n = len(surface)
                    if n <= 0:
                        continue
                    piece = char_entries[idx: idx + n]
                    idx += n
                    if not piece:
                        continue
                    reading_kata = m.reading_form() or surface
                    reading_hira = _kata_to_hira(reading_kata)

                    if n == 1:
                        # 单字符词：该字符自身只有一个时间戳，但读音可能不止
                        # 1 个 mora（如「僕」→ ぼく、「本」→ ほん，1 字 2
                        # mora）。统一交给 mora_list 分支判断，不再无条件
                        # 把整段读音直接塞进这一个时间戳——这正是此前「僕」
                        # 「本」等词被输出成单条 LAB 行（而非逐 mora）的根因。
                        mora_list = _split_ja_mora(reading_hira) if reading_hira else []
                        if not mora_list:
                            s, e, _ch = piece[0]
                            lines.append(
                                f"{int(s*10_000_000)} {int(e*10_000_000)} "
                                f"{reading_hira or surface}"
                            )
                        elif len(mora_list) == 1:
                            s, e, _ch = piece[0]
                            lines.append(
                                f"{int(s*10_000_000)} {int(e*10_000_000)} {mora_list[0]}"
                            )
                        else:
                            lines.extend(_distribute_mora_across_chars(piece, mora_list))
                    else:
                        # 多字符词（如「今日」「大変」）：字数与 mora 数恰好
                        # 相等时逐字直接对应（最常见、最精确）；不相等时
                        # （典型如「大変」2 字对应 たいへん 4 mora）由
                        # _distribute_mora_across_chars() 把 mora 尽量均匀
                        # 分配给各字符、再在每个字符自身时间戳内部按 mora
                        # 数等分，不再退回"整词合并成一条"的旧兜底。
                        mora_list = _split_ja_mora(reading_hira) if reading_hira else []
                        if not mora_list:
                            s = piece[0][0]
                            e = piece[-1][1]
                            lines.append(
                                f"{int(s*10_000_000)} {int(e*10_000_000)} "
                                f"{reading_hira or surface}"
                            )
                        elif len(piece) == len(mora_list):
                            for (ps, pe, _pch), mora in zip(piece, mora_list):
                                lines.append(
                                    f"{int(ps*10_000_000)} {int(pe*10_000_000)} {mora}"
                                )
                        else:
                            lines.extend(_distribute_mora_across_chars(piece, mora_list))

                # 极端情况下 tokenize 输出的字符总数与输入不符（理论上
                # 不应发生，但做个兜底，避免静默丢字）。
                if idx < len(char_entries):
                    for s, e, ch in char_entries[idx:]:
                        lines.append(f"{int(s*10_000_000)} {int(e*10_000_000)} {ch}")

        from mfa_processor import MFAProcessor
        entries_p = MFAProcessor._parse_lab_lines(lines)
        from phoneme_converter import merge_lab_silence
        # 注意：此处不再调用 build_ja_hiragana_lab()——sudachi 已经直接给出
        # 最终假名读音，不是需要状态机合并的单个罗马音素。二次转换会把每条
        # 假名读音误判为"待合并辅音声母"全部拆成 '-'，再被 merge_lab_silence()
        # 当作孤立辅音声母删除，导致最终 LAB 只剩 sil（见上方修复说明）。

        # 把摘出去的静音条目按时间顺序插回——状态机全程没见过它们，
        # 不会把它们的区间错误地并入相邻假名音节。
        combined = sorted(entries_p + sil_entries, key=lambda x: x[0])
        merged = merge_lab_silence(combined)
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

            if self._mfa._is_korean_text(ch):
                # 【修复说明】旧版对 len(ch) == 1 才走逐字分解路径，多字情况
                # （整词 / 整句落在一个 entry 里，如 WhisperX wav2vec2-ko
                # 只能输出 eojeol 级时间戳、或对齐异常降级为全句单条目时）
                # 直接落入 else 分支，整个字符串写成一行 LAB，后续 SVP 无法
                # 按音节切分，表现为"整句粘连为单一时间区间"。
                #
                # 修复：无论单字还是多字（甚至整句），统一调用 MFA 的韩语逐字
                # 分解逻辑 _decompose_korean_syllable_with_onset：
                #   - phone_items=None → 走等比例时长兜底路径，按 jamo 数
                #     给每个音节块分配时间，再按有无初声 (- + char / char) 输出。
                #   - 单字情况行为与旧版完全一致（向后兼容）。
                #
                # 同时过滤掉混入 ch 的标点 / 空格（降级回退时 seg_text 可能
                # 含逗号、问号等），保留纯韩文字符交给分解函数。
                ko_only = "".join(c for c in ch if self._mfa._is_korean_text(c))
                if not ko_only:
                    continue
                syllable_entries = self._mfa._decompose_korean_syllable_with_onset(
                    s100, e100, ko_only, phone_items=None
                )
                for se, ee, pe in syllable_entries:
                    lines.append(f"{se} {ee} {pe}")
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
        min_phoneme_dur: float = 0.025,   # PDG 最小音素时长（秒），25ms
    ):
        super().__init__()
        self.whisper_model = whisper_model
        self._device = self._resolve_device(device)
        # CPU 不支持 float16
        self.compute_type = compute_type if self._device != "cpu" else "int8"
        self.batch_size = batch_size
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.min_phoneme_dur = min_phoneme_dur

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

    # ── 核心对齐（句子隔离版）────────────────────────────────────────────────
    def align(self, audio_path: str, text: Optional[str], language: str) -> Dict:
        """
        句子隔离强制对齐（Sentence-Isolated Alignment）。

        改进点（对比旧版）：
          1. 逐句裁剪音频 → 在极短时序空间内单独对齐，消除长文本累计漂移。
          2. 参考文本与 ASR 句数匹配时，将参考文本绑定到对应句子（修正繁简/识别错误）。
          3. 每句独立完成 LAB 转换，避免字符数不一致导致的全局偏移。
          4. 音素时长守护（PDG）消除极短音标（< 25ms）。
          5. 【已修复】不再是"全程 fill_silences=False，输出零 SP/SIL 的纯净
             连续音标序列"——wav2vec2 在句子内部几乎不会留出时间间隙，旧版
             仅靠 fill_silences 做不到任何停顿。现在改为 _inject_sentence_pauses()
             在标点位置主动插入真正的 sil 条目（见该函数顶部说明），fill_silences
             本身仍为 False（句内不需要按"时间间隙"再做一遍全局扫描，标点
             位置已经显式处理）。
          6. 【已修复】上一步插入的 sil 只有固定的 40/80ms，远短于实际录音
             里的换气/停顿（wav2vec2 经常把真实静音错误地算进标点前最后
             一个字的时长里，实测可达 200~400ms+）。现在加一步
             _refine_sil_boundaries_by_energy()，用该句真实裁剪音频的短时
             能量扫描，把 sil 边界扩展到真正安静的区域，让停顿长度跟随
             这一句实际演唱内容，而不是停在一个固定值上（详见该函数顶部
             说明）。
        """
        t0 = time.time()
        try:
            import whisperx

            wx_lang  = _to_whisperx_lang(language)
            int_lang = _normalize_lang(language)
            _SR      = 16_000   # WhisperX load_audio 固定输出 16kHz

            # ── 1. 加载音频 ──────────────────────────────────────────────────
            # whisperx.load_audio() 依赖 ffmpeg 子进程；若环境中 ffmpeg 不可用
            # 则回退到 soundfile（直接读 WAV/FLAC）+ librosa 重采样，避免崩溃。
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=".*torchcodec.*",   # 屏蔽 pyannote torchcodec 警告
                    )
                    audio = whisperx.load_audio(audio_path)   # float32 numpy, 16kHz
            except Exception as _ffmpeg_err:
                logger.warning(
                    f"[WhisperX] whisperx.load_audio 失败（{_ffmpeg_err}），"
                    "尝试用 soundfile + librosa 回退加载…"
                )
                try:
                    import soundfile as _sf
                    import numpy as _np
                    _data, _orig_sr = _sf.read(audio_path, always_2d=False)
                    if _data.ndim > 1:
                        _data = _data.mean(axis=1)   # 混音为单声道
                    _data = _data.astype(_np.float32)
                    if _orig_sr != _SR:
                        import librosa as _librosa
                        _data = _librosa.resample(_data, orig_sr=_orig_sr, target_sr=_SR)
                    audio = _data
                    logger.info(
                        f"[WhisperX] soundfile 回退加载成功: "
                        f"{len(audio)/float(_SR):.2f}s @ {_SR}Hz"
                    )
                except Exception as _sf_err:
                    return self._err(
                        f"音频加载失败 (ffmpeg: {_ffmpeg_err}; soundfile: {_sf_err})。"
                        "请在系统 PATH 中安装 FFmpeg，或确保 soundfile 已安装。",
                        t0,
                    )

            # ── 2. ASR 转录（仅用于获取句子级时序边界）──────────────────────
            self._load_asr()
            logger.info("[WhisperX] 开始 ASR 转录...")
            asr_out      = self._asr_model.transcribe(
                audio, batch_size=self.batch_size, language=wx_lang
            )
            raw_segments = asr_out.get("segments", [])
            if not raw_segments:
                return self._err("WhisperX ASR 无输出，请检查音频质量", t0)

            asr_text_full = " ".join(s.get("text", "") for s in raw_segments).strip()
            logger.info(f"[WhisperX] ASR 文本: {asr_text_full[:120]}")
            logger.info(f"[WhisperX] ASR 共检出 {len(raw_segments)} 句")

            # ── 3. 参考文本预处理：断句并与 ASR 句段绑定 ────────────────────
            #    句数完全一致时，按句直接绑定；
            #    句数不一致（绝大多数情况）时，按"每段 ASR 自己的识别字数"
            #    为配额，把参考文本（保留标点）顺序切给各段——配额用 ASR
            #    字数而不是音频时长，是为了保证替换后送进 wav2vec2 对齐的
            #    字符数不变，不会导致强制对齐被迫拉伸/压缩造成时间戳错位。
            if text:
                cleaned_ref  = normalize_text_for_whisperx(text, lang=int_lang)
                ref_sentences: List[str] = [
                    s.strip()
                    for s in _re.split(r'[。！？；\n…!?]+', cleaned_ref)
                    if s.strip()
                ]
                if len(ref_sentences) == len(raw_segments):
                    logger.info(
                        f"[WhisperX] 参考文本句数 {len(ref_sentences)} == ASR 句段数，"
                        "绑定参考文本 → 每句使用参考文本对齐"
                    )
                    for i, seg in enumerate(raw_segments):
                        seg["text"] = ref_sentences[i]
                else:
                    bound = _bind_ref_text_by_asr_count(cleaned_ref, raw_segments, int_lang)
                    if bound:
                        logger.warning(
                            f"[WhisperX] 参考文本切出 {len(ref_sentences)} 句 ≠ "
                            f"ASR 段数 {len(raw_segments)}，按各段 ASR 识别字数为配额"
                            "分配参考文本（保留标点，字数严格对应，不退回 ASR 文本）"
                        )
                    else:
                        logger.warning(
                            f"[WhisperX] 参考文本切出 {len(ref_sentences)} 句 ≠ "
                            f"ASR 段数 {len(raw_segments)}，保留 ASR 识别文本逐句对齐"
                        )

            # ── 4. 加载对齐模型 ──────────────────────────────────────────────
            model_a, metadata = self._load_align(wx_lang)
            logger.info(f"[WhisperX] 开始逐句隔离强制对齐（共 {len(raw_segments)} 句）...")

            # ── 5. 句子隔离强制对齐核心循环 ──────────────────────────────────
            #    对每句：① 物理裁剪音频 → ② 在局部短时序空间内对齐
            #          → ③ 局部时间戳 + 句子偏移 = 全局绝对时间戳
            #    完全消除跨句累计漂移和 CTC 路径崩溃导致的音标粘连。
            # seg_pair_list: [(entries_for_this_seg, text_for_this_seg), ...]
            seg_pair_list: List[Tuple[List[Tuple[float, float, str]], str]] = []

            for idx, seg in enumerate(raw_segments):
                start_sec = float(seg.get("start", 0.0))
                end_sec   = float(seg.get("end",   0.0))
                seg_text  = seg.get("text", "").strip()

                if not seg_text or end_sec <= start_sec:
                    continue

                # 物理裁剪：提取该句的音频片段
                st_samp = max(0, int(start_sec * _SR))
                en_samp = min(len(audio), int(end_sec   * _SR))
                cropped = audio[st_samp:en_samp]

                if len(cropped) < 160:      # < 10ms，跳过
                    logger.warning(
                        f"[WhisperX] 第 {idx+1} 句裁剪后过短（{len(cropped)} samples），跳过"
                    )
                    continue

                # 对齐模型接受的文本：剥离标点符号（，。！？：等传入会
                # 导致 wav2vec2 词表缺失而跳过整句），但保留空白和单词
                # 内部撇号——前者是英语/韩语等多词语言的词边界，被误删
                # 会导致整句被拼接成一个伪单词，wav2vec2 只能返回 1 个
                # 跨越全句的 word 条目（详见 _clean_align_text() 顶部的
                # bug 说明）。改用 _clean_align_text()，不再用
                # _is_cjk_punct() 逐字符过滤（该函数把空白也判定为
                # "标点"一并清除）。
                seg_text_for_align = _clean_align_text(seg_text)
                if not seg_text_for_align:
                    continue

                # 单句任务：局部时间从 0 开始
                local_seg_list = [{"text": seg_text_for_align, "start": 0.0, "end": end_sec - start_sec}]

                seg_entries: List[Tuple[float, float, str]] = []
                try:
                    local_aligned = whisperx.align(
                        local_seg_list, model_a, metadata, cropped, self._device,
                        return_char_alignments=True,   # CJK 字符级对齐
                    )

                    for a_seg in local_aligned.get("segments", []):
                        # 中/粤/日/韩都按字符级（chars）切分：中日粤本身
                        # 不用空格分词；韩语虽然书写时用空格分隔"词"
                        # （어절），但歌唱场景下需要的是逐音节字符级时间戳
                        # （和中文逐字一致），不是整个词组一条目——
                        # _ko_entries_to_lab() 早就实现了逐字符的韩语
                        # 处理（含初声"-"占位拆分），此前却一直走 else
                        # 分支取 words（词组级），导致该函数从未真正吃到
                        # 单字符输入，对齐结果停留在"整句/整词组一条目"。
                        if int_lang in ("zh", "yue", "ja", "ko"):
                            units    = a_seg.get("chars", [])
                            text_key = "char"
                        else:
                            units    = a_seg.get("words", [])
                            text_key = "word"

                        for unit in units:
                            s = unit.get("start")
                            e = unit.get("end")
                            t = (unit.get(text_key) or unit.get("text") or "").strip()
                            if s is None or e is None or not t or _is_cjk_punct(t):
                                continue
                            # 局部时间 → 全局绝对时间
                            seg_entries.append(
                                (float(s) + start_sec, float(e) + start_sec, t)
                            )

                except Exception as exc:
                    logger.error(
                        f"[WhisperX] 第 {idx+1} 句对齐异常（'{seg_text[:30]}'）: {exc}"
                    )
                    # 降级：整句作为单一条目，保持时间轴不断裂
                    seg_entries = [(start_sec, end_sec, seg_text)]

                # 句内标点停顿注入：在标点对应位置插入真正的 sil 条目
                # （详见 _inject_sentence_pauses 顶部说明），条目数量会
                # 因此增多，但不改变原有发音字符的相对顺序，不会引入
                # 新的拼音/音节错位。
                seg_entries = _inject_sentence_pauses(seg_entries, seg_text)

                # 上一步给的 40/80ms 只是"至少要有多长停顿"的下限，
                # 不代表音频里真实的换气/停顿就这么短——wav2vec2 经常
                # 把真正的静音错误地算进标点前最后一个字的时长里。这里
                # 用该句裁剪出来的真实音频（cropped，16kHz）做短时能量
                # 扫描，把 sil 边界扩展到真正安静的区域（详见函数顶部
                # 说明），让 SVP 里的停顿长度跟到这一句实际演唱的换气
                # 时长，而不是一个跟内容无关的固定值。
                seg_entries = _refine_sil_boundaries_by_energy(
                    seg_entries, cropped, _SR, start_sec
                )

                if seg_entries:
                    seg_pair_list.append((seg_entries, seg_text))

            if not seg_pair_list:
                return self._err("所有句子对齐均失败，请检查音频质量和语言设置", t0)

            # ── 6. 音素时长守护（PDG）──────────────────────────────────────
            #    每句内部独立运行，将 < min_phoneme_dur 的极短音标扩展到安全时长。
            #    句间间隙（说话停顿）不受影响，总时长严格守恒。
            guarded_pair_list: List[Tuple[List[Tuple[float, float, str]], str]] = []
            for seg_entries, seg_text in seg_pair_list:
                guarded = self._apply_duration_guard(seg_entries, self.min_phoneme_dur)
                guarded_pair_list.append((guarded, seg_text))

            # ── 7. 逐句转换为 LAB（标点处含真实 sil 条目）──────────────────────
            #    每句独立调用 _word_entries_to_lab，用当句文本驱动音素转换，
            #    彻底杜绝字符数不一致跨句传播的偏移错误。fill_silences=False
            #    是因为句内停顿已经由上面第 5 步的 _inject_sentence_pauses()
            #    显式写入了 sil 条目，不需要再做一次基于时间间隙的全局扫描。
            lab_blocks: List[str] = []
            for seg_entries, seg_text in guarded_pair_list:
                if not seg_entries:
                    continue
                block = self._word_entries_to_lab(
                    seg_entries, seg_text, language, fill_silences=False
                )
                if block.strip():
                    lab_blocks.append(block)

            lab = "\n".join(lab_blocks)

            return {
                "success": True,
                "lab_content": lab,
                "raw_text":     text.strip() if text else asr_text_full,
                "phoneme_text": asr_text_full,
                "audio_duration": self._get_audio_duration_100ns(audio_path),
                "processing_time": int((time.time() - t0) * 1000),
                "backend": "whisperx",
            }

        except ImportError as e:
            return self._err(f"whisperx 未安装: {e}，请执行 pip install whisperx", t0)
        except Exception as e:
            logger.error(f"[WhisperX] 对齐失败: {e}", exc_info=True)
            return self._err(str(e), t0)

    # ── 音素时长守护算法（PDG）────────────────────────────────────────────────
    @staticmethod
    def _apply_duration_guard(
        entries: List[Tuple[float, float, str]],
        min_dur_sec: float = 0.025,
    ) -> List[Tuple[float, float, str]]:
        """
        音素时长守护（Phoneme Duration Guard, PDG）。

        对时长 < min_dur_sec 的极短音标，采用双向邻近贪心借用算法进行扩展：
          - 向左右邻居各借用一半时差，邻居自身不低于 min_dur_sec；
          - 单侧不足时由另一侧补足；
          - 首/尾条目仅向另一侧借用；
          - 修正浮点精度导致的边界倒置。
        全局总时长严格守恒，句首/句尾绝对时间不改变。
        """
        if not entries:
            return entries

        es = [[s, e, t] for s, e, t in entries]
        n  = len(es)

        for i in range(n):
            dur = es[i][1] - es[i][0]
            if dur >= min_dur_sec:
                continue

            deficit  = min_dur_sec - dur
            is_first = (i == 0)
            is_last  = (i == n - 1)

            if is_first and is_last:
                # 单条目：强制拉伸右边界
                es[i][1] = es[i][0] + min_dur_sec

            elif is_first:
                # 首部：仅向右借
                avail  = max(0.0, (es[i+1][1] - es[i+1][0]) - min_dur_sec)
                borrow = min(deficit, avail)
                es[i][1]   += borrow
                es[i+1][0] += borrow

            elif is_last:
                # 末尾：仅向左借
                avail  = max(0.0, (es[i-1][1] - es[i-1][0]) - min_dur_sec)
                borrow = min(deficit, avail)
                es[i-1][1] -= borrow
                es[i][0]   -= borrow

            else:
                # 中间：双向对称借用，不足时由另一侧补足
                l_avail  = max(0.0, (es[i-1][1] - es[i-1][0]) - min_dur_sec)
                r_avail  = max(0.0, (es[i+1][1] - es[i+1][0]) - min_dur_sec)
                b_left   = min(deficit / 2.0, l_avail)
                b_right  = min(deficit - b_left, r_avail)
                # 右边不足时左边再补
                if b_right < deficit - b_left:
                    extra   = (deficit - b_left - b_right)
                    b_left += min(extra, l_avail - b_left)
                    b_right = min(deficit - b_left, r_avail)

                es[i-1][1] -= b_left
                es[i][0]   -= b_left
                es[i][1]   += b_right
                es[i+1][0] += b_right

        # 修复浮点误差导致的相邻边界倒置
        for i in range(n - 1):
            if es[i][1] > es[i+1][0]:
                mid = (es[i][1] + es[i+1][0]) / 2.0
                es[i][1]   = mid
                es[i+1][0] = mid

        return [(s, e, t) for s, e, t in es]

    # ── 兼容性保留（旧版内部辅助方法，新版 align() 不再调用）────────────────
    def _extract_entries(
        self, aligned: Dict, int_lang: str
    ) -> List[Tuple[float, float, str]]:
        """[已弃用] 从全局 aligned 结果提取展平条目列表。"""
        entries: List[Tuple[float, float, str]] = []
        for seg in aligned.get("segments", []):
            chars = seg.get("chars", [])
            words = seg.get("words", [])
            if int_lang in ("zh", "yue", "ja") and chars:
                for ch in chars:
                    s = ch.get("start"); e = ch.get("end")
                    t = (ch.get("char") or ch.get("text") or "").strip()
                    if s is not None and e is not None and t and not _is_cjk_punct(t):
                        entries.append((float(s), float(e), t))
            elif words:
                for w in words:
                    s = w.get("start"); e = w.get("end")
                    t = (w.get("word") or w.get("text") or "").strip()
                    if s is not None and e is not None and t:
                        entries.append((float(s), float(e), t))
        entries.sort(key=lambda x: x[0])
        return entries

    def _extract_entries_per_segment(
        self, aligned: Dict, int_lang: str
    ) -> List[List[Tuple[float, float, str]]]:
        """[已弃用] 从全局 aligned 结果按 segment 提取条目列表。"""
        result: List[List[Tuple[float, float, str]]] = []
        for seg in aligned.get("segments", []):
            chars = seg.get("chars", []); words = seg.get("words", [])
            seg_e: List[Tuple[float, float, str]] = []
            if int_lang in ("zh", "yue", "ja") and chars:
                for ch in chars:
                    s = ch.get("start"); e = ch.get("end")
                    t = (ch.get("char") or ch.get("text") or "").strip()
                    if s is not None and e is not None and t and not _is_cjk_punct(t):
                        seg_e.append((float(s), float(e), t))
            elif words:
                for w in words:
                    s = w.get("start"); e = w.get("end")
                    t = (w.get("word") or w.get("text") or "").strip()
                    if s is not None and e is not None and t:
                        seg_e.append((float(s), float(e), t))
            seg_e.sort(key=lambda x: x[0])
            if seg_e:
                result.append(seg_e)
        return result

    def _segments_to_lab(
        self,
        seg_entries_list: List[List[Tuple[float, float, str]]],
        full_text: str,
        language: str,
    ) -> str:
        """[已弃用] 旧版逐段转 LAB，保留供外部调用兼容。"""
        if not seg_entries_list:
            return ""
        lang = _normalize_lang(language)
        if lang not in ("zh", "yue", "ja") or not full_text:
            flat = [e for seg in seg_entries_list for e in seg]
            return self._word_entries_to_lab(flat, full_text, language, fill_silences=False)
        spoken_chars = [
            ch for ch in full_text
            if not unicodedata.category(ch).startswith(("P", "Z", "S"))
        ]
        total_entries = sum(len(s) for s in seg_entries_list)
        if len(spoken_chars) != total_entries:
            logger.warning(
                f"[WhisperX] 参考文本可发音字符 {len(spoken_chars)} ≠ 条目数 {total_entries}，"
                "逐段用 ASR 字符独立转换（不再退化为全局展平）"
            )
            blocks = []
            for seg_entries in seg_entries_list:
                seg_text = "".join(t for _, _, t in seg_entries)
                b = self._word_entries_to_lab(seg_entries, seg_text, language, fill_silences=False)
                if b.strip():
                    blocks.append(b)
            return "\n".join(blocks)
        blocks = []
        cursor = 0
        for seg_entries in seg_entries_list:
            n = len(seg_entries)
            seg_text = "".join(spoken_chars[cursor:cursor + n])
            cursor += n
            b = self._word_entries_to_lab(seg_entries, seg_text, language, fill_silences=False)
            if b.strip():
                blocks.append(b)
        return "\n".join(blocks)

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
                # 【修复】原来只有 "zh"/"yue"/"ja"，韩语 "ko" 被遗漏，
                # 导致 Qwen3-ASR 逐字符时间戳无法分配到每个音节块。
                if int_lang in ("zh", "yue", "ja", "ko") and len(text) > 1 and len(time_stamps) == len(text):
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
                # 【修复】原来只有 "zh"/"yue"/"ja" 走逐字符均分，"ko" 走
                # split()（按空格切词组），导致均分的每一格是整个 eojeol
                # 而非单个音节块。
                units = list(transcribed) if int_lang in ("zh", "yue", "ja", "ko") else transcribed.split()
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
            # 【修复说明】Qwen3-ASR 不为标点输出时间戳，句末/句中停顿只能
            # 体现为相邻字符之间天然的时间间隙。之前这里传 fill_silences=False，
            # 导致这些天然间隙跟 WhisperX 旧版本一样只是数值上的空隙，没有
            # 写入 SIL 标记，SVP 工程生成阶段识别不到，音符仍然连在一起。
            # 改为 True 后交给 _fill_silences_lab() 按 ≥50ms 间隙自动补 SIL，
            # 这也是本模块文档注释里一直描述的预期行为。
            lab = self._word_entries_to_lab(
                entries,
                final_text,
                language,
                fill_silences=True,
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

            # 同上：Qwen3-ForcedAligner 同样不输出标点的时间戳，停顿只能
            # 体现为真实词与词之间的时间间隙，需要 fill_silences=True
            # 才能把这些间隙转换成 SVP 能识别的 SIL 标记。
            lab = self._word_entries_to_lab(
                word_entries, text, language, fill_silences=True
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
            # 【修复】同 Qwen3ASRAligner：补上 "ko"，韩语走逐字符均分而非按空格切词组
            units = list(text) if int_lang in ("zh", "yue", "ja", "ko") else text.split()
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