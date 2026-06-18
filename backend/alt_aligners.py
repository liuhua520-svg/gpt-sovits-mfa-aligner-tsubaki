# -*- coding: utf-8 -*-
"""
alt_aligners.py — 替代音素对齐后端
支持 Qwen3-ASR-1.7B / Qwen3-ForcedAligner-0.6B 作为 MFA 的替代选项

模型文件路径策略（优先级）：
  1. 环境变量 TSUBAKI_MODELS_DIR
  2. <当前文件所在目录>/models/      → 即 backend/models/
     ├── hf_cache/         HuggingFace 统一缓存 (Qwen3-ASR / Qwen3-FA)
     │   └── hub/
     └── rmvpe/            RMVPE 模型 (已有，路径不变)

标点/静音处理：
  Qwen3 不在对齐输出中输出标点（标点不可发声）。
  本模块在生成 LAB 后自动扫描时间轴间隙，将 ≥ 50ms 的空白补全为 SP / SIL 条目。
  用户无需为标点担心，静音标记由时间间隙自动推断。
"""
from __future__ import annotations

import logging
import os
import time
import unicodedata
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
_MODELS_DIR: Path = resolve_models_dir()
_HF_CACHE:    Path = _MODELS_DIR / "hf_cache"     # HuggingFace Hub 缓存
_HF_HUB:      Path = _HF_CACHE   / "hub"          # transformers 子目录

for _d in (_HF_CACHE, _HF_HUB):
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
    f"  HF 缓存 → {_HF_HUB}"
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
    long_sil_100ns: int = 5_000_000,    # 500ms → 用 SIL 而非 SP
) -> str:
    """
    扫描 LAB 时间轴，在 ≥ 50ms 的间隙自动补全 SP / SIL 条目。

    背景：Qwen3 不输出标点字符的时间戳，但句末/句中停顿
    会在相邻字符之间留下时间间隙，本函数将这些间隙转换为 LAB 静音标记。

    Parameters
    ----------
    min_gap_100ns : 插入 SP 的最小间隙（默认 50ms）
    long_sil_100ns : 超过此值用 SIL 代替 SP（默认 500ms）
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
        result.append((0, parsed[0][0], "SP"))

    for i, (s, e, ph) in enumerate(parsed):
        result.append((s, e, ph))
        if i + 1 < len(parsed):
            gap_s = e
            gap_e = parsed[i + 1][0]
            gap   = gap_e - gap_s
            if gap > min_gap_100ns:
                label = "SIL" if gap >= long_sil_100ns else "SP"
                result.append((gap_s, gap_e, label))

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

    # ── 词语时间戳 → LAB（含 SP 间隙补全）──────────────────────────────────
    def _word_entries_to_lab(
        self,
        word_entries: List[Tuple[float, float, str]],
        text: str,
        language: str,
        fill_silences: bool = True,
    ) -> str:
        """
        将词语 / 字符级时间戳 → LAB 格式，复用 MFAProcessor 的音素转换逻辑。
        fill_silences=True 时自动在时间间隙中插入 SP / SIL。

        关于标点：
          Qwen3 不产生标点字符的对齐时间戳（标点不可发音）。
          _text_to_syllables() 在提取音素序列时也会忽略标点，因此参考文本中
          的标点不影响音素分布——只要参考文本的可发音字符数与 entries 数量一致即可。
          句末 / 句中的停顿由 fill_silences 根据时间间隙自动插入。
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
# 5. Qwen3ASRAligner
# ═════════════════════════════════════════════════════════════════════════════

class Qwen3ASRAligner(AltAlignerBase):
    """
    Qwen3-ASR-1.7B 对齐后端（自动语音识别 + 词语级时间戳）
    https://huggingface.co/Qwen/Qwen3-ASR-1.7B

    模型文件存放位置：
      backend/models/hf_cache/hub/models--Qwen--Qwen3-ASR-1.7B/  (~3.5GB)
    """

    DEFAULT_MODEL = "Qwen/Qwen3-ASR-1.7B"

    def __init__(self, model_id: str = DEFAULT_MODEL, device: str = "auto"):
        super().__init__()
        self.model_id = model_id
        self._device = self._resolve_device(device)
        self._pipe = None

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            try:
                import torch
                return "cuda:0" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return device

    @staticmethod
    def check_available() -> Tuple[bool, str]:
        # Qwen3-ASR 使用 funasr 推理，不依赖标准 transformers AutoModel
        try:
            import funasr  # noqa: F401
            return True, "funasr 已就绪（Qwen3-ASR 官方推理框架）"
        except ImportError as e:
            funasr_err = str(e)
        except Exception as e:
            # funasr 已安装，但内部 import 出错（常见原因：huggingface-hub
            # 版本与 funasr 要求不符，例如环境内其他包把 hub 降级到 <1.0，
            # funasr 引用的新版 hub API 不存在）
            funasr_err = f"{type(e).__name__}: {e}"
        # 降级：检查 transformers（仅能加载缓存 config，无法真正推理）
        try:
            import transformers  # noqa: F401
            return (
                False,
                f"funasr 导入失败（{funasr_err}）。\n"
                "若已执行过 pip install funasr modelscope 但仍报此错，"
                "大概率是 huggingface-hub 版本冲突（环境内某个包要求"
                "huggingface-hub < 1.0，与 funasr 所需的新版冲突）。"
                "建议为 Qwen3 后端单独建立虚拟环境，"
                "或运行 `python -c \"import funasr\"` 查看完整 traceback 确认。",
            )
        except ImportError as e:
            return False, f"未安装: pip install funasr modelscope ({e})"

    def _load_model(self):
        if self._pipe is not None:
            return

        logger.info(
            f"[Qwen3-ASR] 加载模型: {self.model_id}\n"
            f"  缓存目录 → {_HF_HUB}"
        )

        # ── 优先路径：funasr（官方推荐，支持 qwen3_asr 架构）────────────
        try:
            from funasr import AutoModel as FunASRAutoModel  # type: ignore

            device = "cuda" if "cuda" in self._device else "cpu"
            self._pipe = FunASRAutoModel(
                model=self.model_id,
                device=device,
                hub="hf",
                model_path=str(_HF_HUB),   # 本地缓存目录
            )
            self._pipe_backend = "funasr"
            logger.info("[Qwen3-ASR] ✓ 模型已加载（funasr 后端）")
            return
        except ImportError as e:
            logger.warning(
                f"[Qwen3-ASR] funasr 导入失败: {type(e).__name__}: {e}\n"
                "  若已安装过 funasr，这通常是 huggingface-hub 版本冲突"
                "导致的，而非真的缺包。\n"
                "  尝试降级 transformers pipeline 路径..."
            )
        except Exception as e:
            logger.warning(f"[Qwen3-ASR] funasr 加载失败: {e}，尝试降级")
            import traceback
            logger.debug(traceback.format_exc())

        # ── 降级路径：transformers pipeline（仅当模型已有本地权重时有效）──
        # 注意：Qwen3-ASR-1.7B 的 config.model_type = 'qwen3_asr'，
        # 标准 transformers <= 5.x 不含此类，from_pretrained 会抛 KeyError/ValueError。
        # 如果到达此处，说明 funasr 未安装且降级也无法成功，需提示用户安装 funasr。
        try:
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
            import torch

            dtype = torch.float16 if "cuda" in self._device else torch.float32
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                cache_dir=str(_HF_HUB),
                trust_remote_code=True,     # ← 允许加载模型自带的 modeling_qwen3_asr.py
            )
            model.to(self._device)
            processor = AutoProcessor.from_pretrained(
                self.model_id,
                cache_dir=str(_HF_HUB),
                trust_remote_code=True,
            )
            self._pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                torch_dtype=dtype,
                device=self._device,
            )
            self._pipe_backend = "transformers"
            logger.info("[Qwen3-ASR] ✓ 模型已加载（transformers + trust_remote_code）")
        except Exception as e:
            raise RuntimeError(
                f"[Qwen3-ASR] 无法加载模型：{e}\n\n"
                "Qwen3-ASR 使用自定义 'qwen3_asr' 架构，标准 transformers 不支持此类型。\n"
                "请安装官方推理框架后重试：\n"
                "  pip install funasr modelscope\n"
                "或从源码安装最新 transformers：\n"
                "  pip install git+https://github.com/huggingface/transformers.git"
            ) from e

    def align(self, audio_path: str, text: Optional[str], language: str) -> Dict:
        t0 = time.time()
        try:
            int_lang = _normalize_lang(language)
            # funasr 的 language 参数接受 ISO 短代码（zh/en/ja/ko）
            asr_lang = {"zh": "zh", "yue": "zh", "en": "en",
                        "ja": "ja", "ko": "ko"}.get(int_lang, int_lang)

            self._load_model()
            logger.info("[Qwen3-ASR] 开始转录...")

            backend = getattr(self, "_pipe_backend", "transformers")

            if backend == "funasr":
                # funasr 输出：list of dict，包含 "text" 和 "timestamp"
                # generate_kwargs：language 映射到 funasr 的 language 参数
                raw = self._pipe.generate(
                    audio_path,
                    language=asr_lang,
                    return_raw_text=True,
                )
                # raw 形如: [{"text": "你好世界", "timestamp": [[0.0, 0.2], [0.2, 0.4], ...]}]
                if not raw:
                    return {"success": False, "error": "Qwen3-ASR (funasr) 无转录结果",
                            "processing_time": int((time.time() - t0) * 1000)}
                transcribed = " ".join(
                    item.get("text", "") for item in raw
                ).strip()
                # 将 funasr 时间戳格式统一转换为 chunks 格式
                chunks = []
                for item in raw:
                    item_text = item.get("text", "")
                    item_ts   = item.get("timestamp", [])  # [[s_ms, e_ms], ...]
                    if item_ts and item_text:
                        # funasr 时间戳单位：毫秒
                        for char_text, (s_ms, e_ms) in zip(list(item_text), item_ts):
                            chunks.append({
                                "text": char_text,
                                "timestamp": (s_ms / 1000.0, e_ms / 1000.0),
                            })
                    elif item_text:
                        chunks.append({"text": item_text, "timestamp": (None, None)})
            else:
                # transformers pipeline 输出
                result = self._pipe(
                    audio_path,
                    generate_kwargs={"language": asr_lang, "task": "transcribe"},
                    return_timestamps="word",
                    chunk_length_s=30,
                    stride_length_s=5,
                )
                chunks      = result.get("chunks", [])
                transcribed = result.get("text", "").strip()

            logger.info(f"[Qwen3-ASR] 转录文本: {transcribed[:120]}")

            if not chunks and not transcribed:
                return {"success": False, "error": "Qwen3-ASR 无转录结果",
                        "processing_time": int((time.time() - t0) * 1000)}

            # 无 chunk 降级：按字符均匀分配时间
            if not chunks and transcribed:
                total_s = self._get_audio_duration_100ns(audio_path) / 1e7
                units = (list(transcribed) if int_lang in ("zh", "yue", "ja")
                         else transcribed.split())
                dur = total_s / max(len(units), 1)
                chunks = [{"text": u, "timestamp": (i * dur, (i + 1) * dur)}
                          for i, u in enumerate(units) if u.strip()]

            entries: List[Tuple[float, float, str]] = []
            for chunk in chunks:
                ts       = chunk.get("timestamp") or (None, None)
                ch_text  = (chunk.get("text") or "").strip()
                if ts[0] is None or ts[1] is None or not ch_text:
                    continue
                # CJK：chunk 可能含多字，逐字均匀拆分
                if int_lang in ("zh", "yue") and len(ch_text) > 1:
                    dur_each = (ts[1] - ts[0]) / len(ch_text)
                    for i, ch in enumerate(ch_text):
                        if not _is_cjk_punct(ch):
                            entries.append((
                                ts[0] + i * dur_each,
                                ts[0] + (i + 1) * dur_each,
                                ch,
                            ))
                else:
                    if not _is_cjk_punct(ch_text):
                        entries.append((float(ts[0]), float(ts[1]), ch_text))

            if not entries:
                return {"success": False, "error": "Qwen3-ASR 无时间戳输出",
                        "processing_time": int((time.time() - t0) * 1000)}

            final_text = text.strip() if text else transcribed
            lab = self._word_entries_to_lab(entries, final_text, language,
                                            fill_silences=True)

            return {
                "success":        True,
                "lab_content":    lab,
                "raw_text":       final_text,
                "phoneme_text":   transcribed,
                "audio_duration": self._get_audio_duration_100ns(audio_path),
                "processing_time": int((time.time() - t0) * 1000),
                "backend":        "qwen3_asr",
            }

        except ImportError as e:
            return {"success": False,
                    "error": f"transformers 未安装: {e}",
                    "processing_time": int((time.time() - t0) * 1000)}
        except Exception as e:
            logger.error(f"[Qwen3-ASR] 失败: {e}", exc_info=True)
            return {"success": False, "error": str(e),
                    "processing_time": int((time.time() - t0) * 1000)}


# ═════════════════════════════════════════════════════════════════════════════
# 6. Qwen3ForcedAligner
# ═════════════════════════════════════════════════════════════════════════════

class Qwen3ForcedAligner(AltAlignerBase):
    """
    Qwen3-ForcedAligner-0.6B 强制对齐后端（需要参考文本）
    https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B

    模型文件存放位置：
      backend/models/hf_cache/hub/models--Qwen--Qwen3-ForcedAligner-0.6B/  (~1.2GB)

    注意：此模型较新，以下实现基于标准 CTC 强制对齐接口 + Seq2Seq 备用路径。
    若模型 API 有更新，请参照官方文档调整 _load_model() 和 _extract_timestamps()。
    """

    DEFAULT_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B"

    def __init__(self, model_id: str = DEFAULT_MODEL, device: str = "auto"):
        super().__init__()
        self.model_id = model_id
        self._device = self._resolve_device(device)
        self._model    = None
        self._processor = None

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
            import transformers  # noqa: F401
            return True, "transformers 已就绪（需下载 Qwen3-ForcedAligner-0.6B）"
        except ImportError as e:
            return (
                False,
                f"funasr 导入失败（{funasr_err}）。\n"
                "推荐方案：\n"
                "   pip uninstall -y transformers funasr\n"
                "   pip install funasr modelscope\n"
                "   pip install --upgrade git+https://github.com/huggingface/transformers.git"
            )

    def _load_model(self):
        if self._model is not None:
            return
        from transformers import AutoProcessor
        import torch

        logger.info(
            f"[Qwen3-FA] 加载模型: {self.model_id}\n"
            f"  缓存目录 → {_HF_HUB}"
        )
        dtype = torch.float16 if "cuda" in self._device else torch.float32

        # Qwen3-ForcedAligner-0.6B 同样使用 'qwen3_asr' model_type，
        # 标准 transformers 不含此架构，必须加 trust_remote_code=True
        # 让 transformers 加载模型仓库自带的 modeling_*.py。
        common_kwargs = dict(
            torch_dtype=dtype,
            cache_dir=str(_HF_HUB),
            trust_remote_code=True,     # ← 关键：允许加载自定义架构代码
        )
        try:
            from transformers import AutoModelForCTC
            self._model = AutoModelForCTC.from_pretrained(
                self.model_id, **common_kwargs
            ).to(self._device)
            self._model_type = "ctc"
        except (ValueError, OSError, KeyError):
            # 若模型不是 CTC 架构，降级到 Seq2Seq
            try:
                from transformers import AutoModelForSpeechSeq2Seq
                self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    self.model_id, **common_kwargs
                ).to(self._device)
                self._model_type = "seq2seq"
            except Exception as e:
                raise RuntimeError(
                    f"[Qwen3-FA] 无法加载模型：{e}\n\n"
                    "Qwen3-ForcedAligner-0.6B 使用 'qwen3_asr' 自定义架构。\n"
                    "请确保模型仓库已完整下载（含 modeling_qwen3_asr.py），\n"
                    "或安装官方推理框架：pip install funasr modelscope\n"
                    "或从源码安装最新 transformers：\n"
                    "  pip install git+https://github.com/huggingface/transformers.git"
                ) from e

        self._processor = AutoProcessor.from_pretrained(
            self.model_id,
            cache_dir=str(_HF_HUB),
            trust_remote_code=True,
        )
        self._model.eval()
        logger.info(f"[Qwen3-FA] ✓ 模型已加载（类型: {self._model_type}）")

    def align(self, audio_path: str, text: Optional[str], language: str) -> Dict:
        t0 = time.time()
        if not text:
            return {"success": False,
                    "error": "Qwen3-ForcedAligner 需要参考文本（text 不能为空）",
                    "processing_time": 0}
        try:
            import torch, numpy as np, soundfile as sf

            self._load_model()
            int_lang = _normalize_lang(language)

            audio_arr, sr = sf.read(audio_path)
            if audio_arr.ndim > 1:
                audio_arr = audio_arr.mean(axis=1)
            if sr != 16000:
                try:
                    import librosa
                    audio_arr = librosa.resample(
                        audio_arr.astype(np.float32), orig_sr=sr, target_sr=16000
                    )
                    sr = 16000
                except ImportError:
                    pass

            total_sec = len(audio_arr) / sr
            logger.info("[Qwen3-FA] 开始强制对齐...")

            inputs = self._processor(
                audio_arr, sampling_rate=sr, text=text, return_tensors="pt"
            )
            inputs = {k: v.to(self._device) if hasattr(v, "to") else v
                      for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)

            entries = self._extract_timestamps(outputs, text, int_lang, total_sec)
            if not entries:
                return {"success": False, "error": "Qwen3-ForcedAligner 无对齐输出",
                        "processing_time": int((time.time() - t0) * 1000)}

            lab = self._word_entries_to_lab(entries, text, language, fill_silences=True)
            return {
                "success":        True,
                "lab_content":    lab,
                "raw_text":       text,
                "phoneme_text":   text,
                "audio_duration": self._get_audio_duration_100ns(audio_path),
                "processing_time": int((time.time() - t0) * 1000),
                "backend":        "qwen3_aligner",
            }
        except ImportError as e:
            return {"success": False,
                    "error": f"依赖未安装: {e}，请执行 pip install transformers torch torchaudio",
                    "processing_time": int((time.time() - t0) * 1000)}
        except Exception as e:
            logger.error(f"[Qwen3-FA] 失败: {e}", exc_info=True)
            return {"success": False, "error": str(e),
                    "processing_time": int((time.time() - t0) * 1000)}

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
# 7. 单例缓存与工厂函数
# ═════════════════════════════════════════════════════════════════════════════

_SINGLETON: Dict[str, AltAlignerBase] = {}


def get_aligner(backend: str, device: str = "auto", **kwargs) -> AltAlignerBase:
    """
    工厂函数：按 backend 名称创建或复用对齐器单例。
    backend: "qwen3_asr" | "qwen3_aligner"
    """
    global _SINGLETON
    if backend not in _SINGLETON:
        if backend == "qwen3_asr":
            _SINGLETON[backend] = Qwen3ASRAligner(device=device, **kwargs)
        elif backend == "qwen3_aligner":
            _SINGLETON[backend] = Qwen3ForcedAligner(device=device, **kwargs)
        else:
            raise ValueError(f"未知对齐后端: {backend}")
    return _SINGLETON[backend]


def get_alt_aligner_status() -> Dict:
    """检查所有替代对齐后端的可用状态（含模型文件目录信息）"""
    qa_ok, qa_msg = Qwen3ASRAligner.check_available()
    qf_ok, qf_msg = Qwen3ForcedAligner.check_available()

    return {
        "models_dir": str(_MODELS_DIR),        # ← 前端可展示此路径
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