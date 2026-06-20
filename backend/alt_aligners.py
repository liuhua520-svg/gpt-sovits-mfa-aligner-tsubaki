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
import re
import unicodedata
from typing import Dict, List, Optional, Tuple

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
    判断字符串是否全为标点 / 空白 / 符号。
    """
    if not text:
        return True
    for ch in text:
        cat = unicodedata.category(ch)
        if not cat.startswith(("P", "Z", "S")):
            return False
    return True

def _split_spoken_units(text: str, lang: str) -> List[str]:
    """
    将 full_text 切成可发声单位：
    - zh/yue/ja：按字符
    - 其他语言：按词
    """
    if not text:
        return []

    lang = _normalize_lang(lang)

    if lang in ("zh", "yue", "ja"):
        units = []
        for ch in text:
            cat = unicodedata.category(ch)
            if cat.startswith(("P", "Z", "S")):
                continue
            units.append(ch)
        return units

    units = []
    for tok in re.split(r"\s+", text):
        tok = tok.strip()
        if not tok:
            continue
        tok = tok.strip("，。！？；、…・｜─—,.;:!?\"'()[]{}<>《》「」『』“”‘’")
        if tok:
            units.append(tok)
    return units

def _split_ref_sentences(text: str) -> List[str]:
    """
    只按强句号断句，尽量保留原句结构。
    """
    if not text:
        return []
    parts = re.split(r"[。！？；\n…!?]+", text)
    return [p.strip() for p in parts if p and p.strip()]


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


def _bind_ref_text_by_asr_count(
    cleaned_ref: str,
    raw_segments: List[Dict],
    int_lang: str,
) -> bool:
    """
    稳定版参考文本绑定：
    - 只在“句数完全一致”时绑定
    - 不再按 ASR 字数强行切分参考文本
    - 返回 True 表示完成绑定
    """
    if not cleaned_ref or not raw_segments:
        return False

    ref_sentences = _split_ref_sentences(cleaned_ref)

    if len(ref_sentences) != len(raw_segments):
        logger.warning(
            f"[WhisperX] 参考文本句数 {len(ref_sentences)} != ASR 段数 {len(raw_segments)}，"
            "跳过参考文本硬绑定，保留 ASR 识别文本"
        )
        return False

    for i, seg in enumerate(raw_segments):
        sentence = ref_sentences[i].strip()
        if sentence:
            seg["text"] = sentence

    return True


def _inject_sentence_pauses(
    seg_entries: List[Tuple[float, float, str]],
    seg_text: str,
    heavy_gap_sec: float = 0.08,
    light_gap_sec: float = 0.04,
) -> List[Tuple[float, float, str]]:
    """
    在 seg_text 中句末/句内标点出现的位置，从前一个已发音字符的结尾处
    "偷"出一小段时长，人为制造一个停顿——不依赖音频里是否真的有静音，
    也不改变字符总数/顺序（不会引入错位）。

    heavy_gap_sec : 句末强标点（。！？…）对应的停顿时长
    light_gap_sec : 句内轻标点（、，；）对应的停顿时长
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
    result = list(seg_entries)
    for k in range(m - 1):
        target_gap = gap_after[k]
        if target_gap <= 0:
            continue
        s, e, t = result[k]
        next_s = result[k + 1][0]
        avail = next_s - e
        if avail >= target_gap:
            continue
        shrink = min(target_gap - avail, max(0.0, (e - s) * 0.5))
        if shrink > 0:
            result[k] = (s, e - shrink, t)
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
        entries: List[Tuple[float, float, str]],
        full_text: str,
        language: str,
        fill_silences: bool = False,
    ) -> str:
        """
        稳定版 LAB 输出：
        - 严格保持时间单调
        - 不制造新音素、不重排字符
        - 仅在“文本单位数 == 条目数”时，用 full_text 覆盖标签
        - fill_silences=False 时，不插入 SIL/SP
        """
        if not entries:
            return ""

        lang = _normalize_lang(language)

        cleaned: List[Tuple[float, float, str]] = []
        last_end = 0.0
        for item in sorted(entries, key=lambda x: (float(x[0]), float(x[1]))):
            if len(item) < 3:
                continue

            s, e, t = item
            if s is None or e is None:
                continue

            s = float(s)
            e = float(e)
            t = (t or "").strip()

            if not t or _is_cjk_punct(t):
                continue

            if s < last_end:
                s = last_end
            if e < s:
                e = s + 1e-4

            cleaned.append((s, e, t))
            last_end = e

        if not cleaned:
            return ""

        spoken_units = _split_spoken_units(full_text or "", lang)
        use_full_text_units = len(spoken_units) == len(cleaned) and len(spoken_units) > 0

        lines: List[str] = []
        last_end = None

        for idx, (s, e, t) in enumerate(cleaned):
            label = spoken_units[idx] if use_full_text_units else t
            label = (label or "").strip()

            if not label or _is_cjk_punct(label):
                continue

            if fill_silences and last_end is not None:
                gap = s - last_end
                if gap >= 0.05:
                    lines.append(f"{last_end:.6f} {s:.6f} SIL")

            lines.append(f"{s:.6f} {e:.6f} {label}")
            last_end = e

        return "\n".join(lines)

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
        句子隔离强制对齐（稳定修复版）

        核心原则：
          1. 句数一致才绑定参考文本
          2. 句数不一致时，保留 ASR 文本，不做危险硬切分
          3. 对齐输入里标点改空格，不直接删除成一坨
          4. 用真实裁剪时长做 local alignment
          5. 对齐失败时整句回填，保证时间轴不断裂
        """
        t0 = time.time()
        try:
            import whisperx

            wx_lang = self._to_whisperx_lang(language)
            int_lang = self._normalize_lang(language)
            _SR = 16_000

            # 1) 加载音频
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*torchcodec.*")
                    audio = whisperx.load_audio(audio_path)
            except Exception as _ffmpeg_err:
                logger.warning(
                    f"[WhisperX] whisperx.load_audio 失败（{_ffmpeg_err}），尝试用 soundfile + librosa 回退加载…"
                )
                try:
                    import soundfile as _sf
                    import numpy as _np

                    _data, _orig_sr = _sf.read(audio_path, always_2d=False)
                    if _data.ndim > 1:
                        _data = _data.mean(axis=1)
                    _data = _data.astype(_np.float32)
                    if _orig_sr != _SR:
                        import librosa as _librosa
                        _data = _librosa.resample(_data, orig_sr=_orig_sr, target_sr=_SR)

                    audio = _data
                    logger.info(
                        f"[WhisperX] soundfile 回退加载成功: {len(audio)/float(_SR):.2f}s @ {_SR}Hz"
                    )
                except Exception as _sf_err:
                    return self._err(
                        f"音频加载失败 (ffmpeg: {_ffmpeg_err}; soundfile: {_sf_err})。"
                        "请在系统 PATH 中安装 FFmpeg，或确保 soundfile 已安装。",
                        t0,
                    )

            # 2) ASR 转录
            self._load_asr()
            logger.info("[WhisperX] 开始 ASR 转录...")
            asr_out = self._asr_model.transcribe(audio, batch_size=self.batch_size, language=wx_lang)
            raw_segments = asr_out.get("segments", [])
            if not raw_segments:
                return self._err("WhisperX ASR 无输出，请检查音频质量", t0)

            asr_text_full = " ".join(s.get("text", "") for s in raw_segments).strip()
            logger.info(f"[WhisperX] ASR 文本: {asr_text_full[:120]}")
            logger.info(f"[WhisperX] ASR 共检出 {len(raw_segments)} 句")

            # 3) 参考文本预处理：只在句数一致时绑定
            if text:
                cleaned_ref = normalize_text_for_whisperx(text, lang=int_lang)
                ref_sentences: List[str] = _split_ref_sentences(cleaned_ref)

                if len(ref_sentences) == len(raw_segments) and len(ref_sentences) > 0:
                    logger.info(
                        f"[WhisperX] 参考文本句数 {len(ref_sentences)} == ASR 句段数 {len(raw_segments)}，逐句绑定参考文本"
                    )
                    for i, seg in enumerate(raw_segments):
                        seg["text"] = ref_sentences[i]
                else:
                    logger.warning(
                        f"[WhisperX] 参考文本句数 {len(ref_sentences)} != ASR 句段数 {len(raw_segments)}，"
                        "保留 ASR 识别文本逐句对齐（更稳定）"
                    )

            # 4) 加载对齐模型
            model_a, metadata = self._load_align(wx_lang)
            logger.info(f"[WhisperX] 开始逐句隔离强制对齐（共 {len(raw_segments)} 句）...")

            # 5) 逐句对齐
            seg_pair_list: List[Tuple[List[Tuple[float, float, str]], str]] = []

            for idx, seg in enumerate(raw_segments):
                start_sec = float(seg.get("start", 0.0))
                end_sec = float(seg.get("end", 0.0))
                seg_text = (seg.get("text", "") or "").strip()

                if not seg_text or end_sec <= start_sec:
                    continue

                st_samp = max(0, int(start_sec * _SR))
                en_samp = min(len(audio), int(end_sec * _SR))
                cropped = audio[st_samp:en_samp]

                if len(cropped) < 160:
                    logger.warning(
                        f"[WhisperX] 第 {idx + 1} 句裁剪后过短（{len(cropped)} samples），跳过"
                    )
                    continue

                real_dur = len(cropped) / float(_SR)

                # 关键修复：标点改空格，不要删除
                seg_text_for_align = _segment_text_for_align(seg_text)
                if not seg_text_for_align:
                    seg_pair_list.append(([(start_sec, end_sec, seg_text)], seg_text))
                    continue

                local_seg_list = [{
                    "text": seg_text_for_align,
                    "start": 0.0,
                    "end": real_dur,
                }]

                seg_entries: List[Tuple[float, float, str]] = []
                try:
                    local_aligned = whisperx.align(
                        local_seg_list,
                        model_a,
                        metadata,
                        cropped,
                        self._device,
                        return_char_alignments=True,
                    )

                    for a_seg in local_aligned.get("segments", []):
                        if int_lang in ("zh", "yue", "ja"):
                            units = a_seg.get("chars", []) or []
                            text_key = "char"
                        else:
                            units = a_seg.get("words", []) or []
                            text_key = "word"

                        if not units:
                            units = a_seg.get("words", []) or a_seg.get("chars", []) or []
                            text_key = "word"

                        for unit in units:
                            s = unit.get("start")
                            e = unit.get("end")
                            t = (unit.get(text_key) or unit.get("text") or "").strip()

                            if s is None or e is None or not t or _is_cjk_punct(t):
                                continue

                            seg_entries.append((float(s) + start_sec, float(e) + start_sec, t))

                except Exception as exc:
                    logger.error(
                        f"[WhisperX] 第 {idx + 1} 句对齐异常（'{seg_text[:30]}'）: {exc}"
                    )

                if not seg_entries:
                    seg_entries = [(start_sec, end_sec, seg_text)]

                seg_entries = self._apply_duration_guard(
                    seg_entries,
                    getattr(self, "min_phoneme_dur", 0.025),
                )

                if seg_entries:
                    seg_pair_list.append((seg_entries, seg_text))

            if not seg_pair_list:
                return self._err("所有句子对齐均失败，请检查音频质量和语言设置", t0)

            # 6) 转 LAB
            lab_blocks: List[str] = []
            for seg_entries, seg_text in seg_pair_list:
                if not seg_entries:
                    continue

                block = self._word_entries_to_lab(
                    seg_entries,
                    seg_text,
                    language,
                    fill_silences=False,
                )
                if block and block.strip():
                    lab_blocks.append(block.strip())

            lab = "\n".join(lab_blocks).strip()

            return {
                "success": True,
                "lab_content": lab,
                "raw_text": text.strip() if text else asr_text_full,
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
