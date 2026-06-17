# -*- coding: utf-8 -*-
"""
alt_aligners.py — 替代音素对齐后端
支持 WhisperX / Qwen3-ASR-1.7B / Qwen3-ForcedAligner-0.6B 作为 MFA 的替代选项

架构：
  每个后端产出与 MFAProcessor.process() 完全兼容的字典
  {"success", "lab_content", "raw_text", "phoneme_text", "audio_duration", "processing_time", "backend"}
  内部借用 MFAProcessor 中的音素转换逻辑（拼音/ARPABET/罗马字等），
  通过 MockInterval 将词语级时间戳"伪装"成 Word Tier 对象注入。
"""
from __future__ import annotations

import logging
import os
import time
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 语言代码映射
# ─────────────────────────────────────────────────────────────────────────────

def _to_whisperx_lang(lang: str) -> str:
    """内部语言代码 → WhisperX / Whisper 语言代码"""
    return {
        "cmn": "zh", "zh": "zh", "zh-cn": "zh",
        "yue": "zh",   # 粤语用 zh 近似；WhisperX 暂无独立粤语对齐模型
        "eng": "en", "en": "en",
        "jpn": "ja", "ja": "ja",
        "kor": "ko", "ko": "ko",
    }.get(lang.lower(), lang.lower())


def _normalize_lang(lang: str) -> str:
    """各种语言代码 → 内部短代码 (zh / yue / en / ja / ko)"""
    return {
        "cmn": "zh", "zh-cn": "zh", "mandarin": "zh",
        "yue": "yue", "cantonese": "yue", "zh-yue": "yue",
        "eng": "en", "english": "en",
        "jpn": "ja", "japanese": "ja",
        "kor": "ko", "korean": "ko",
    }.get(lang.lower(), lang.lower())


# ─────────────────────────────────────────────────────────────────────────────
# MockInterval — 让 MFAProcessor 的 Word-Tier 处理函数可以复用
# ─────────────────────────────────────────────────────────────────────────────

class _MI:
    """模拟 textgrid.Interval，供 MFAProcessor 内部逻辑使用（时间单位：秒）"""
    __slots__ = ("minTime", "maxTime", "mark", "text")

    def __init__(self, start_sec: float, end_sec: float, mark: str):
        self.minTime = float(start_sec)
        self.maxTime = float(end_sec)
        self.mark = mark
        self.text = mark


# ─────────────────────────────────────────────────────────────────────────────
# 基类
# ─────────────────────────────────────────────────────────────────────────────

class AltAlignerBase:
    """
    所有替代对齐后端的公共基类。
    共享 MFAProcessor 的音素转换 / 后处理逻辑，通过 _word_entries_to_lab() 复用。
    """

    def __init__(self):
        # 延迟导入，避免循环引用
        from mfa_processor import MFAProcessor
        self._mfa = MFAProcessor()

    # ── 子类必须实现 ──────────────────────────────────────────────────────────
    def align(self, audio_path: str, text: Optional[str], language: str) -> Dict:
        raise NotImplementedError

    # ── 共享工具：词语时间戳 → LAB ─────────────────────────────────────────────
    def _word_entries_to_lab(
        self,
        word_entries: List[Tuple[float, float, str]],   # (start_sec, end_sec, text_unit)
        text: str,
        language: str,
    ) -> str:
        """
        将词语 / 字符级别时间戳转换为 LAB 格式，复用 MFAProcessor 的音素转换逻辑。

        对于中文: 每个 word_entry 应为单个汉字 → 自动转为拼音 + con 标记
        对于英文: 每个 word_entry 为英文单词 → 直接写入词语标签（无 phone_items 时）
        对于日语: 见 _ja_entries_to_lab()（单独处理）
        对于韩语: 见 _ko_entries_to_lab()（单独处理）
        """
        if not word_entries:
            return ""

        lang = _normalize_lang(language)

        # 日语 / 韩语需要特殊处理（因为 MFAProcessor 的对应函数强依赖 phone_items）
        if lang == "ja":
            return self._ja_entries_to_lab(word_entries, text)
        if lang == "ko":
            return self._ko_entries_to_lab(word_entries, text)

        # 构造 Mock Word Tier
        word_tier = [_MI(s, e, w) for s, e, w in word_entries]
        phone_items: List[Tuple[int, int, str]] = []   # 替代后端无音素级数据

        if lang in ("zh", "cmn"):
            lines = self._mfa._process_zh_words(word_tier, phone_items, text)
        elif lang == "yue":
            lines = self._mfa._process_yue_words(word_tier, phone_items, text)
        else:
            # 英语及其他语言：回退到英语处理（词语级 ARPABET，无 phone_items 时直接输出词语）
            lines = self._mfa._process_en_words(word_tier, phone_items, text)

        lines = self._mfa._apply_lab_postprocess(lines, lang)
        return "\n".join(lines)

    def _ja_entries_to_lab(
        self,
        word_entries: List[Tuple[float, float, str]],
        text: str,
    ) -> str:
        """
        日语：直接把字符时间戳转为假名 LAB。
        不依赖 MFA Phone Tier IPA，而是用 pykakasi 做字符→罗马字→假名转换。
        """
        try:
            import pykakasi
            kks = pykakasi.kakasi()
        except ImportError:
            # 无 pykakasi：直接输出原始字符作为标注
            lines = []
            for s, e, ch in word_entries:
                s100 = int(s * 10_000_000)
                e100 = int(e * 10_000_000)
                if ch.strip() and not _is_cjk_punct(ch):
                    lines.append(f"{s100} {e100} {ch}")
            return "\n".join(lines)

        lines: List[str] = []
        for s, e, ch in word_entries:
            ch = ch.strip()
            if not ch or _is_cjk_punct(ch):
                continue
            s100 = int(s * 10_000_000)
            e100 = int(e * 10_000_000)
            result = kks.convert(ch)
            hira = "".join(r.get("hira", r.get("orig", "")) for r in result).strip()
            label = hira if hira else ch
            lines.append(f"{s100} {e100} {label}")

        # 日语后处理（假名化 + merge '-'）
        from mfa_processor import MFAProcessor
        entries = MFAProcessor._parse_lab_lines(lines)
        from phoneme_converter import build_ja_hiragana_lab, merge_lab_silence
        entries = build_ja_hiragana_lab(entries)
        merged = merge_lab_silence(entries)
        return "\n".join(f"{s} {e} {p}" for s, e, p in merged)

    def _ko_entries_to_lab(
        self,
        word_entries: List[Tuple[float, float, str]],
        text: str,
    ) -> str:
        """
        韩语：直接把字符时间戳写入 LAB（韩文字符已经是合适的标注单元）。
        复用 MFAProcessor 的 `-` con 标记逻辑。
        """
        lines: List[str] = []
        for s, e, ch in word_entries:
            ch = ch.strip()
            if not ch:
                continue
            s100 = int(s * 10_000_000)
            e100 = int(e * 10_000_000)
            dur = e100 - s100
            # 每个韩文字都可能有初声（添加 '-' 前置标记）
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
        entries = MFAProcessor._parse_lab_lines(lines)
        from phoneme_converter import merge_lab_silence
        merged = merge_lab_silence(entries)
        return "\n".join(f"{s} {e} {p}" for s, e, p in merged)

    def _get_audio_duration_100ns(self, audio_path: str) -> int:
        return self._mfa._get_audio_duration(audio_path)


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def _is_cjk_punct(text: str) -> bool:
    """判断字符串是否全为标点 / 空格"""
    if not text:
        return True
    for ch in text:
        cat = unicodedata.category(ch)
        if not cat.startswith(("P", "Z", "S")):
            return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# WhisperXAligner
# ─────────────────────────────────────────────────────────────────────────────

class WhisperXAligner(AltAlignerBase):
    """
    WhisperX 对齐后端（自动语音识别 + 强制音素对齐）
    https://github.com/m-bain/whisperx

    优势：
      - 不需要参考文本（自动转录）
      - 字符级对齐（中日韩），词语级对齐（英语等）
      - 支持 GPU 加速（CUDA）

    安装：pip install whisperx
    """

    def __init__(
        self,
        whisper_model: str = "large-v2",
        device: str = "auto",
        compute_type: str = "float16",
        batch_size: int = 16,
        hf_token: Optional[str] = None,
    ):
        super().__init__()
        self.whisper_model = whisper_model
        self._device = self._resolve_device(device)
        # CPU 不支持 float16
        self.compute_type = compute_type if self._device != "cpu" else "int8"
        self.batch_size = batch_size
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")

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
                self.whisper_model, self._device, compute_type=self.compute_type
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
            asr_out = self._asr_model.transcribe(
                audio, batch_size=self.batch_size, language=wx_lang
            )
            if not asr_out.get("segments"):
                return self._err("WhisperX ASR 无输出，请检查音频质量", t0)

            asr_text = " ".join(s.get("text", "") for s in asr_out["segments"]).strip()
            logger.info(f"[WhisperX] ASR 文本: {asr_text[:80]}")

            # 3. 强制对齐
            model_a, metadata = self._load_align(wx_lang)
            logger.info("[WhisperX] 开始强制对齐...")
            aligned = whisperx.align(
                asr_out["segments"], model_a, metadata, audio, self._device,
                return_char_alignments=True,   # CJK 关键：字符级对齐
            )

            # 4. 提取词语时间戳
            entries = self._extract_entries(aligned, int_lang)
            if not entries:
                return self._err("强制对齐无输出，请检查语言代码和音频质量", t0)

            # 5. 转换为 LAB
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
            return self._err(f"whisperx 未安装: {e}，请执行 pip install whisperx", t0)
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


# ─────────────────────────────────────────────────────────────────────────────
# Qwen3ASRAligner
# ─────────────────────────────────────────────────────────────────────────────

class Qwen3ASRAligner(AltAlignerBase):
    """
    Qwen3-ASR-1.7B 对齐后端（自动语音识别 + 词语级时间戳）
    https://huggingface.co/Qwen/Qwen3-ASR-1.7B

    优势：
      - 不需要参考文本（自动转录）
      - 对中文多口音 / 方言容忍度更高
      - 支持 GPU 加速

    安装：pip install transformers accelerate torch
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
        try:
            import transformers  # noqa: F401
            return True, "transformers 已就绪"
        except ImportError as e:
            return False, f"未安装: pip install transformers ({e})"

    def _load_model(self):
        if self._pipe is not None:
            return
        try:
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
            import torch

            logger.info(f"[Qwen3-ASR] 加载模型: {self.model_id}")
            dtype = torch.float16 if "cuda" in self._device else torch.float32

            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
            )
            model.to(self._device)
            processor = AutoProcessor.from_pretrained(self.model_id)

            self._pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                torch_dtype=dtype,
                device=self._device,
            )
            logger.info("[Qwen3-ASR] ✓ 模型已加载")
        except Exception as e:
            raise RuntimeError(f"Qwen3-ASR 模型加载失败: {e}") from e

    def align(self, audio_path: str, text: Optional[str], language: str) -> Dict:
        t0 = time.time()
        try:
            int_lang = _normalize_lang(language)
            wx_lang = _to_whisperx_lang(language)

            self._load_model()
            logger.info("[Qwen3-ASR] 开始转录...")

            result = self._pipe(
                audio_path,
                generate_kwargs={"language": wx_lang, "task": "transcribe"},
                return_timestamps="word",
                chunk_length_s=30,
                stride_length_s=5,
            )

            chunks = result.get("chunks", [])
            transcribed = result.get("text", "").strip()

            if not chunks and not transcribed:
                return {"success": False, "error": "Qwen3-ASR 无转录结果",
                        "processing_time": int((time.time() - t0) * 1000)}

            # 无 chunk 时降级：按字符 / 词均匀分配时间
            if not chunks and transcribed:
                total_s = self._get_audio_duration_100ns(audio_path) / 1e7
                units = list(transcribed) if int_lang in ("zh", "yue", "ja") else transcribed.split()
                dur = total_s / max(len(units), 1)
                chunks = [{"text": u, "timestamp": (i * dur, (i + 1) * dur)}
                          for i, u in enumerate(units) if u.strip()]

            entries: List[Tuple[float, float, str]] = []
            for chunk in chunks:
                ts = chunk.get("timestamp") or (None, None)
                ch_text = (chunk.get("text") or "").strip()
                if ts[0] is not None and ts[1] is not None and ch_text:
                    # 若是 CJK 语言且 chunk 含多字符，拆开再均匀分配
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
            lab = self._word_entries_to_lab(entries, final_text, language)

            return {
                "success": True,
                "lab_content": lab,
                "raw_text": final_text,
                "phoneme_text": transcribed,
                "audio_duration": self._get_audio_duration_100ns(audio_path),
                "processing_time": int((time.time() - t0) * 1000),
                "backend": "qwen3_asr",
            }
        except ImportError as e:
            return {"success": False,
                    "error": f"transformers 未安装: {e}，请执行 pip install transformers",
                    "processing_time": int((time.time() - t0) * 1000)}
        except Exception as e:
            logger.error(f"[Qwen3-ASR] 失败: {e}", exc_info=True)
            return {"success": False, "error": str(e),
                    "processing_time": int((time.time() - t0) * 1000)}


# ─────────────────────────────────────────────────────────────────────────────
# Qwen3ForcedAligner
# ─────────────────────────────────────────────────────────────────────────────

class Qwen3ForcedAligner(AltAlignerBase):
    """
    Qwen3-ForcedAligner-0.6B 强制对齐后端（需要参考文本）
    https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B

    优势：
      - 专为歌声/语音精确对齐设计
      - 轻量级（0.6B），GPU 占用低
      - 提供参考文本时精度高于纯 ASR

    注意：Qwen3-ForcedAligner 为较新发布的模型，以下实现基于标准 CTC
    强制对齐接口。如模型 API 有更新，请参照官方文档调整 _load_model()
    和 _run_ctc_align() 部分。

    安装：pip install transformers torch torchaudio accelerate
    """

    DEFAULT_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B"

    def __init__(self, model_id: str = DEFAULT_MODEL, device: str = "auto"):
        super().__init__()
        self.model_id = model_id
        self._device = self._resolve_device(device)
        self._model = None
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
            return True, "transformers 已就绪（需下载 Qwen3-ForcedAligner-0.6B 模型）"
        except ImportError as e:
            return False, f"未安装: pip install transformers ({e})"

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from transformers import AutoProcessor
            import torch

            logger.info(f"[Qwen3-FA] 加载模型: {self.model_id}")
            dtype = torch.float16 if "cuda" in self._device else torch.float32

            # Qwen3-ForcedAligner 预期为 CTC 架构
            # 若 API 变更（如改为 Seq2Seq），请修改此处
            try:
                from transformers import AutoModelForCTC
                self._model = AutoModelForCTC.from_pretrained(
                    self.model_id, torch_dtype=dtype
                ).to(self._device)
            except (ValueError, OSError):
                # 备用：Seq2Seq 方式
                from transformers import AutoModelForSpeechSeq2Seq
                self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    self.model_id, torch_dtype=dtype
                ).to(self._device)

            self._processor = AutoProcessor.from_pretrained(self.model_id)
            self._model.eval()
            logger.info("[Qwen3-FA] ✓ 模型已加载")
        except Exception as e:
            raise RuntimeError(f"Qwen3-ForcedAligner 模型加载失败: {e}") from e

    def align(self, audio_path: str, text: Optional[str], language: str) -> Dict:
        t0 = time.time()
        if not text:
            return {"success": False,
                    "error": "Qwen3-ForcedAligner 需要参考文本（text 不能为空）",
                    "processing_time": 0}
        try:
            import torch
            import numpy as np
            import soundfile as sf

            self._load_model()
            int_lang = _normalize_lang(language)

            # 读取并预处理音频（单声道 16kHz）
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
                    pass   # 保持原采样率，精度可能略降

            total_sec = len(audio_arr) / sr
            logger.info("[Qwen3-FA] 开始强制对齐...")

            # 准备模型输入
            inputs = self._processor(
                audio_arr,
                sampling_rate=sr,
                text=text,
                return_tensors="pt",
            )
            inputs = {k: v.to(self._device) if hasattr(v, "to") else v
                      for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)

            # 从模型输出提取时间戳
            entries = self._extract_timestamps(outputs, text, int_lang, total_sec)

            if not entries:
                return {"success": False, "error": "Qwen3-ForcedAligner 无对齐输出",
                        "processing_time": int((time.time() - t0) * 1000)}

            lab = self._word_entries_to_lab(entries, text, language)
            return {
                "success": True,
                "lab_content": lab,
                "raw_text": text,
                "phoneme_text": text,
                "audio_duration": self._get_audio_duration_100ns(audio_path),
                "processing_time": int((time.time() - t0) * 1000),
                "backend": "qwen3_aligner",
            }

        except ImportError as e:
            return {"success": False,
                    "error": f"依赖未安装: {e}，请执行 pip install transformers torch torchaudio",
                    "processing_time": int((time.time() - t0) * 1000)}
        except Exception as e:
            logger.error(f"[Qwen3-FA] 对齐失败: {e}", exc_info=True)
            return {"success": False, "error": str(e),
                    "processing_time": int((time.time() - t0) * 1000)}

    def _extract_timestamps(
        self,
        outputs,
        text: str,
        int_lang: str,
        total_sec: float,
    ) -> List[Tuple[float, float, str]]:
        """
        从模型输出中提取对齐时间戳。
        支持 CTC logits（标准 torchaudio forced_align）和 Seq2Seq chunks 两种输出格式。
        """
        import torch

        entries: List[Tuple[float, float, str]] = []

        # ── 方案 A：CTC logits + torchaudio.functional.forced_align ────────
        if hasattr(outputs, "logits"):
            try:
                import torchaudio

                logits = outputs.logits[0].float()      # [T, vocab]
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

                vocab = self._processor.tokenizer.get_vocab()
                tokens = self._processor.tokenizer(text, return_tensors="pt").input_ids[0]
                frame_dur = total_sec / logits.shape[0]

                # torchaudio >= 2.1 支持 forced_align
                if hasattr(torchaudio.functional, "forced_align"):
                    aligned = torchaudio.functional.forced_align(
                        log_probs.unsqueeze(0).cpu(),
                        tokens.unsqueeze(0),
                        blank=vocab.get("<pad>", 0),
                    )
                    # aligned: (token_indices, scores)
                    token_spans = torchaudio.functional.merge_tokens(
                        aligned[0][0], self._processor.tokenizer.pad_token_id
                    )
                    id2tok = {v: k for k, v in vocab.items()}
                    for span in token_spans:
                        tok_text = id2tok.get(span.token, "").lstrip("▁").strip()
                        if tok_text:
                            entries.append((
                                span.start * frame_dur,
                                span.end * frame_dur,
                                tok_text,
                            ))
                else:
                    # 无 forced_align API：改用 Viterbi 贪婪 CTC 解码时间戳
                    frame_dur = total_sec / logits.shape[0]
                    pred_ids = torch.argmax(logits, dim=-1).cpu().numpy()
                    blank_id = vocab.get("<pad>", 0)
                    id2tok = {v: k for k, v in vocab.items()}
                    cur_tok = None
                    cur_start = 0
                    for i, tid in enumerate(pred_ids):
                        if int(tid) == blank_id:
                            if cur_tok is not None:
                                tok_text = id2tok.get(cur_tok, "").lstrip("▁").strip()
                                if tok_text:
                                    entries.append((cur_start * frame_dur, i * frame_dur, tok_text))
                                cur_tok = None
                        else:
                            if cur_tok != int(tid):
                                if cur_tok is not None:
                                    tok_text = id2tok.get(cur_tok, "").lstrip("▁").strip()
                                    if tok_text:
                                        entries.append((cur_start * frame_dur, i * frame_dur, tok_text))
                                cur_tok = int(tid)
                                cur_start = i
                    if cur_tok is not None:
                        tok_text = id2tok.get(cur_tok, "").lstrip("▁").strip()
                        if tok_text:
                            entries.append((cur_start * frame_dur, len(pred_ids) * frame_dur, tok_text))
            except Exception as e:
                logger.warning(f"[Qwen3-FA] CTC 时间戳提取失败，降级均匀分配: {e}")

        # ── 方案 B：Seq2Seq chunks（带 timestamp token 输出）───────────────
        elif hasattr(outputs, "sequences"):
            try:
                decoded = self._processor.batch_decode(
                    outputs.sequences, output_offsets=True, skip_special_tokens=True
                )
                for item in decoded:
                    if isinstance(item, dict) and "chunks" in item:
                        for chunk in item["chunks"]:
                            ts = chunk.get("timestamp") or (None, None)
                            ch_text = (chunk.get("text") or "").strip()
                            if ts[0] is not None and ch_text:
                                entries.append((
                                    float(ts[0]),
                                    float(ts[1] or ts[0] + 0.25),
                                    ch_text,
                                ))
            except Exception as e:
                logger.warning(f"[Qwen3-FA] Seq2Seq 时间戳提取失败，降级均匀分配: {e}")

        # ── 降级：均匀时间分配 ────────────────────────────────────────────────
        if not entries:
            logger.warning("[Qwen3-FA] 无法提取时间戳，改用均匀分配（精度较低）")
            units = list(text) if int_lang in ("zh", "yue", "ja") else text.split()
            units = [u for u in units if u.strip() and not _is_cjk_punct(u)]
            if units:
                dur = total_sec / len(units)
                entries = [(i * dur, (i + 1) * dur, u) for i, u in enumerate(units)]

        return entries


# ─────────────────────────────────────────────────────────────────────────────
# 单例缓存与工厂函数
# ─────────────────────────────────────────────────────────────────────────────

_SINGLETON: Dict[str, AltAlignerBase] = {}


def get_aligner(backend: str, device: str = "auto", **kwargs) -> AltAlignerBase:
    """
    工厂函数：按 backend 名称创建或复用对齐器实例。
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
    """检查所有替代对齐后端的可用状态"""
    wx_ok, wx_msg = WhisperXAligner.check_available()
    qa_ok, qa_msg = Qwen3ASRAligner.check_available()
    qf_ok, qf_msg = Qwen3ForcedAligner.check_available()
    return {
        "whisperx":      {"available": wx_ok, "message": wx_msg,
                          "requires_text": False, "description": "WhisperX (Whisper ASR + 强制对齐)"},
        "qwen3_asr":     {"available": qa_ok, "message": qa_msg,
                          "requires_text": False, "description": "Qwen3-ASR-1.7B (自动语音识别)"},
        "qwen3_aligner": {"available": qf_ok, "message": qf_msg,
                          "requires_text": True,  "description": "Qwen3-ForcedAligner-0.6B (强制对齐)"},
    }
