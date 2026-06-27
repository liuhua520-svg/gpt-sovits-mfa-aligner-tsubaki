# -*- coding: utf-8 -*-
"""
完整处理流程管道（v2.1 — 多后端对齐支持）
整合 MFA / Qwen3-ASR / Qwen3-ForcedAligner + 音高处理 + 工程文件生成

新增参数: aligner_backend
  "mfa"           — Montreal Forced Aligner（默认）
  "qwen3_asr"     — Qwen3-ASR-1.7B (自动语音识别，文本可选)
  "qwen3_aligner" — Qwen3-ForcedAligner-0.6B (强制对齐，需要参考文本)
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, Optional

from mfa_processor import MFAProcessor
from tsubaki_processor import TsubakiProcessor, AudioProcessingConfig

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# 数字 → 各语种读法文字 转换
#   把参考文本里的阿拉伯数字 0-9 按当前选择的语种转换成对应的读法文字。
#   逐字转换（一三一四 → 一三一四），不做"完整数值"解析（不会把 "1234"
#   读成"一千二百三十四"），与歌词/对齐场景里数字常被逐字唱出的习惯一致
#   （电话号码、年份、编号等通常也是逐字读出）。
#
#   挂在 _run_alignment() 顶部、四个后端分支之前调用一次，MFA /
#   WhisperX / Qwen3-ASR / Qwen3-FA 四个后端因此都能复用同一份转换结果，
#   不需要在各自的对齐逻辑里各自实现一遍。
# ═════════════════════════════════════════════════════════════════════════════

_DIGIT_WORDS: Dict[str, list] = {
    "zh":  ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九"],
    "yue": ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九"],
    "en":  ["zero", "one", "two", "three", "four",
            "five", "six", "seven", "eight", "nine"],
    "ja":  ["ぜろ", "いち", "に", "さん", "よん",
            "ご", "ろく", "なな", "はち", "きゅう"],
    "ko":  ["영", "일", "이", "삼", "사", "오", "육", "칠", "팔", "구"],
}

# 中/粤文字本身书写不依赖空格，逐字直接拼接（如 "1234" → "一二三四"）。
# 英/日/韩每个数字的读法之间用空格隔开：
#   - 英语词级对齐依赖空格做分词，连写会变成一个词典/G2P 都查不到的
#     生造词（"onetwothree"），必须保留分隔。
#   - 日语/韩语虽然书写习惯上不强制空格，但连续多个数字逐字读出时
#     （如电话号码、编号）保留分隔更贴近真实唱法，也避免相邻数字的
#     读法互相粘连产生歧义（如 いち+に 连写成"いちに"）。
_DIGIT_WORD_SEPARATOR: Dict[str, str] = {
    "zh": "", "yue": "", "en": " ", "ja": " ", "ko": " ",
}

_DIGIT_RUN_RE = re.compile(r"\d+")


def _convert_digits_to_words(text: str, language: str) -> str:
    """
    把 text 里所有阿拉伯数字 0-9 按 language 对应的语种转换成读法文字。

    转换规则（与产品需求给出的样例完全一致）：
      中文/粤语 1234567890 → 一二三四五六七八九零（无分隔符）
      英语      1234567890 → one two three four five six seven eight nine zero
      日语      1234567890 → いち に さん よん ご ろく なな はち きゅう ぜろ
      韩语      1234567890 → 일 이 삼 사 오 육 칠 팔 구 영

    text 中非数字部分原样保留；language 经 _normalize_lang() 规整后若不在
    上表中（暂不支持该语种的数字转换），原样返回 text，不做任何改动。
    text 中本就不含数字时直接返回原文本，不做无意义的字符串重建。
    """
    if not text or not any(ch.isdigit() for ch in text):
        return text

    # 延迟导入：避免给 MFA-only 的主路径增加 alt_aligners.py 的导入开销，
    # 与 _run_alignment() 里对替代后端分支的现有延迟导入风格保持一致。
    from alt_aligners import _normalize_lang

    int_lang = _normalize_lang(language)
    words = _DIGIT_WORDS.get(int_lang)
    if not words:
        logger.debug(f"数字转换：语种 '{language}'（规整为 '{int_lang}'）暂不支持，文本原样保留")
        return text

    sep = _DIGIT_WORD_SEPARATOR.get(int_lang, " ")

    def _replace_run(match: "re.Match") -> str:
        digit_run = match.group(0)
        return sep.join(words[int(d)] for d in digit_run)

    converted = _DIGIT_RUN_RE.sub(_replace_run, text)
    if converted != text:
        logger.info(f"数字转换 [{int_lang}]：'{text}' → '{converted}'")
    return converted


def _run_alignment(
    audio_file,             # Flask FileStorage 或 FileStorageWrapper
    text: str,
    language: str,
    backend: str = "mfa",
    f0_device: str = "auto",
    whisperx_model: str = "large-v3",
    english_word_align: bool = False,
) -> Dict:
    """
    统一调度对齐后端，返回与 MFAProcessor.process() 格式兼容的字典。
    audio_file.save(path) 和 audio_file.filename 必须可用。
    """
    # 数字 → 读法文字转换，放在四个后端分支之前统一处理一次：
    # MFA / WhisperX / Qwen3-ASR / Qwen3-FA 都从这里拿到转换后的文本，
    # 不需要各自重复实现。text 为空（如 Qwen3-ASR 纯识别模式不提供
    # 参考文本）时 _convert_digits_to_words 内部直接原样返回，不受影响。
    text = _convert_digits_to_words(text, language)

    if backend == "mfa":
        processor = MFAProcessor()
        return processor.process(audio_file, text, language,
                                 english_word_align=english_word_align)

    # ── 替代后端 ──────────────────────────────────────────────────────────
    import tempfile, shutil, os
    from alt_aligners import get_aligner

    # 把文件保存到临时目录，获取路径供 alt aligner 使用
    tmp_dir = tempfile.mkdtemp(prefix="alt_aligner_")
    try:
        filename = getattr(audio_file, "filename", "audio.wav")
        tmp_wav = os.path.join(tmp_dir, filename)
        audio_file.save(tmp_wav)

        extra = {}
        if backend == "whisperx":
            extra["whisper_model"] = whisperx_model
        aligner = get_aligner(backend, device=f0_device, **extra)
        return aligner.align(tmp_wav, text or None, language,
                             english_word_align=english_word_align)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


class AudioProcessingPipeline:
    """
    完整音频处理流程：
    1. 音频上传
    2. 对齐标注 (生成 LAB 文件)  ← 现在支持多后端
    3. 音高提取 (PyWORLD / CREPE / RMVPE)
    4. 工程文件生成 (SVP / USTX)
    """

    def __init__(self, work_dir: str):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        self.mfa_processor = MFAProcessor()
        self.tsubaki_processor = TsubakiProcessor(str(self.work_dir))

        logger.info(f"✓ 处理流程初始化，工作目录: {self.work_dir}")

    def process_full(
        self,
        audio_file,
        text: str,
        language: str = "cmn",
        output_format: str = "sv",
        project_title: str = "Project",
        bpm: float = 120.0,
        base_pitch: int = 60,
        f0_method: str = "dio",
        f0_smooth: bool = True,
        f0_smooth_window: int = 5,
        use_double_precision: bool = False,
        f0_floor: float = 71.0,
        f0_ceil: float = 800.0,
        refine_pitch: bool = False,
        export_pitch_line: bool = True,
        f0_device: str = "auto",
        crepe_model: str = "full",
        aligner_backend: str = "mfa",           # ← 新增
        whisperx_model: str = "large-v3",       # ← WhisperX 模型大小
        english_word_align: bool = False,        # ← 英语单词级对齐
        vsqx_singer: str = "MIKU_V4_Chinese",           # ← VSQX 声库名（由 app.py 按语种注入）
        vsqx_singer_id: str = "BNGE7CP7EMTRSNC3",       # ← VSQX 声库 ID
        vsqx_singer_bs: int = 4,                         # ← VSQX 声库 Bank Select（VOCALOID4 内部编号）
    ) -> Dict:
        config = AudioProcessingConfig(
            bpm=bpm,
            base_pitch=base_pitch,
            f0_method=f0_method,
            f0_smooth=f0_smooth,
            f0_smooth_window=f0_smooth_window,
            refine_pitch=refine_pitch,
            export_pitch_line=export_pitch_line,
            use_double_precision=use_double_precision,
            f0_floor=f0_floor,
            f0_ceil=f0_ceil,
            f0_device=f0_device,
            crepe_model=crepe_model,
        )

        import time
        start_time = time.time()

        try:
            logger.info("=" * 60)
            logger.info(f"开始完整处理流程 [aligner={aligner_backend}]")
            logger.info("=" * 60)

            audio_filename = getattr(audio_file, "filename", "audio")
            stem = Path(audio_filename).stem
            wav_path = str(self.work_dir / f"{stem}.wav")

            logger.info(f"正在保存音频: {wav_path}")
            Path(wav_path).parent.mkdir(parents=True, exist_ok=True)
            audio_file.seek(0)
            audio_file.save(wav_path)
            audio_file.seek(0)

            # ── 步骤 1：对齐标注 ──────────────────────────────────────────
            logger.info(f"[ 步骤 1/3 ] 对齐标注 (backend={aligner_backend})...")
            align_result = _run_alignment(audio_file, text, language, aligner_backend,
                                           f0_device, whisperx_model,
                                           english_word_align=english_word_align)
            if not align_result.get("success"):
                error = align_result.get("error", "对齐处理失败")
                logger.error(f"✗ 对齐失败: {error}")
                return {
                    "success": False, "error": error,
                    "stage": "alignment",
                    "processing_time": int((time.time() - start_time) * 1000),
                }

            lab_content = align_result.get("lab_content", "")
            lab_path = str(self.work_dir / f"{stem}.lab")
            with open(lab_path, "w", encoding="utf-8") as f:
                f.write(lab_content)
            logger.info(f"✓ LAB 标注完成: {lab_path}")

            # ── 步骤 2：F0 提取 ──────────────────────────────────────────
            logger.info("[ 步骤 2/3 ] 音高提取...")
            try:
                audio_data = self.tsubaki_processor.process_audio_f0(wav_path, config)
                if not audio_data or not audio_data.get("success"):
                    logger.warning(
                        f"⚠ F0 提取失败({(audio_data or {}).get('error', 'unknown')})，继续（不含音高曲线）"
                    )
                    audio_data = None
            except Exception as e:
                logger.warning(f"⚠ 音高提取异常: {e}，继续生成工程文件")
                audio_data = None
            logger.info("✓ 音高提取完成")

            # ── 步骤 3：生成工程文件 ─────────────────────────────────────
            logger.info(f"[ 步骤 3/3 ] 生成 {output_format.upper()} 工程文件...")
            project_result = self.tsubaki_processor.process_full_pipeline(
                wav_path=wav_path,
                lab_path=lab_path,
                output_format=output_format,
                project_title=project_title,
                config=config,
                audio_f0_data=audio_data,
                vsqx_singer=vsqx_singer,
                vsqx_singer_id=vsqx_singer_id,
                vsqx_singer_bs=vsqx_singer_bs,
            )

            if not project_result.get("success"):
                error = project_result.get("error", "工程文件生成失败")
                logger.error(f"✗ 工程文件生成失败: {error}")
                return {
                    "success": False, "error": error,
                    "stage": "project_generation",
                    "lab_path": lab_path,
                    "processing_time": int((time.time() - start_time) * 1000),
                }

            project_path = project_result.get("output_path")
            processing_time = int((time.time() - start_time) * 1000)

            logger.info("=" * 60)
            logger.info("✓ 完整处理流程完成")
            logger.info(f" LAB 文件: {lab_path}")
            logger.info(f" 工程文件: {project_path}")
            logger.info(f" 耗时: {processing_time}ms")
            logger.info("=" * 60)

            return {
                "success": True,
                "lab_path": lab_path,
                "lab_content": lab_content,
                "project_path": project_path,
                "project_format": project_result.get("format", output_format),
                "requested_format": output_format,
                "segments": project_result.get("segments", 0),
                "processing_time": processing_time,
                "config": config.to_dict(),
                "aligner_backend": aligner_backend,
                "message": f'完整处理完成: {project_result.get("segments", 0)} 个标注段',
            }
        except Exception as e:
            logger.error(f"✗ 处理流程异常: {e}", exc_info=True)
            return {
                "success": False, "error": str(e), "stage": "unknown",
                "processing_time": int((time.time() - start_time) * 1000),
            }

    def process_mfa_only(
        self,
        audio_file,
        text: str,
        language: str = "cmn",
        aligner_backend: str = "mfa",           # ← 新增
        f0_device: str = "auto",
        whisperx_model: str = "large-v3",       # ← WhisperX 模型大小
        english_word_align: bool = False,        # ← 英语单词级对齐
    ) -> Dict:
        """仅执行对齐标注（不生成工程文件）"""
        import time
        start_time = time.time()

        try:
            logger.info(f"[ 标注模式 ] 执行自动标注 (backend={aligner_backend})")

            result = _run_alignment(audio_file, text, language, aligner_backend,
                                    f0_device, whisperx_model,
                                    english_word_align=english_word_align)

            if result.get("success"):
                lab_content = result.get("lab_content", "")
                audio_filename = getattr(audio_file, "filename", "audio")
                stem = Path(audio_filename).stem
                lab_path = str(self.work_dir / f"{stem}.lab")

                Path(lab_path).parent.mkdir(parents=True, exist_ok=True)
                with open(lab_path, "w", encoding="utf-8") as f:
                    f.write(lab_content)

                logger.info(f"✓ LAB 标注已保存: {lab_path}")
                result["lab_path"] = lab_path
                result["processing_time"] = int((time.time() - start_time) * 1000)
                result["aligner_backend"] = aligner_backend
                return result
            else:
                return {
                    **result,
                    "processing_time": int((time.time() - start_time) * 1000),
                    "aligner_backend": aligner_backend,
                }

        except Exception as e:
            logger.error(f"✗ 标注处理异常: {e}", exc_info=True)
            return {
                "success": False, "error": str(e),
                "processing_time": int((time.time() - start_time) * 1000),
            }

    def process_project_only(
        self,
        wav_path: str,
        lab_path: Optional[str] = None,
        output_format: str = "sv",
        project_title: str = "Project",
        bpm: float = 120.0,
        base_pitch: int = 60,
        f0_method: str = "dio",
        f0_smooth: bool = True,
        f0_smooth_window: int = 5,
        use_double_precision: bool = False,
        f0_floor: float = 71.0,
        f0_ceil: float = 800.0,
        refine_pitch: bool = False,
        export_pitch_line: bool = True,
        f0_device: str = "auto",
        crepe_model: str = "full",
        phoneme_mode: str = "none",
        midi_path: str = None,
        lyrics_text: str = "",
        vsqx_singer: str = "MIKU_V4X_Original_EVEC",    # ← 仅生成工程默认日语声库
        vsqx_singer_id: str = "BCNFCY43LB2LZCD4",       # ← 对应声库 ID
        vsqx_singer_bs: int = 0,                          # ← 仅生成工程默认 bs（日语声库=0）
    ) -> Dict:
        """仅执行工程文件生成（已有 WAV 以及 LAB/MIDI 之一）"""
        import time
        start_time = time.time()

        try:
            logger.info("[ 工程文件模式 ] 生成项目文件")
            logger.info(f" 音频: {wav_path}")
            logger.info(f" 标注: {lab_path or '(无 LAB)'}")
            logger.info(f" MIDI: {midi_path or '(无 MIDI)'}")
            logger.info(f" 格式: {output_format}")

            if not Path(wav_path).exists():
                return {
                    "success": False, "error": f"WAV 文件不存在: {wav_path}",
                    "processing_time": 0,
                }

            lab_exists  = bool(lab_path  and Path(lab_path).exists())
            midi_exists = bool(midi_path and Path(midi_path).exists())

            if not lab_exists and not midi_exists:
                return {
                    "success": False,
                    "error": "需要 LAB 文件或 MIDI 文件（至少提供其中一个）",
                    "processing_time": 0,
                }

            config = AudioProcessingConfig(
                bpm=bpm, base_pitch=base_pitch,
                f0_floor=f0_floor, f0_ceil=f0_ceil,
                f0_method=f0_method, f0_smooth=f0_smooth,
                f0_smooth_window=f0_smooth_window,
                use_double_precision=use_double_precision,
                refine_pitch=refine_pitch,
                export_pitch_line=export_pitch_line,
                f0_device=f0_device, crepe_model=crepe_model,
            )

            audio_data = None
            try:
                audio_data = self.tsubaki_processor.process_audio_f0(wav_path, config)
                if not audio_data or not audio_data.get("success"):
                    logger.warning(
                        f"⚠ F0 提取失败({(audio_data or {}).get('error', 'unknown')})，继续生成工程文件"
                    )
                    audio_data = None
            except Exception as e:
                logger.warning(f"⚠ 音高提取异常: {e}，继续生成工程文件")

            result = self.tsubaki_processor.process_full_pipeline(
                wav_path=wav_path,
                lab_path=lab_path if lab_exists else None,
                output_format=output_format,
                project_title=project_title,
                config=config,
                audio_f0_data=audio_data,
                phoneme_mode=phoneme_mode,
                midi_path=midi_path or None,
                lyrics_text=lyrics_text,
                vsqx_singer=vsqx_singer,
                vsqx_singer_id=vsqx_singer_id,
                vsqx_singer_bs=vsqx_singer_bs,
            )

            result["processing_time"] = int((time.time() - start_time) * 1000)
            return result

        except Exception as e:
            logger.error(f"✗ 工程文件生成异常: {e}", exc_info=True)
            return {
                "success": False, "error": str(e),
                "processing_time": int((time.time() - start_time) * 1000),
            }

    def process_f0_only(
        self,
        wav_path: str,
        method: str = "dio",
        f0_floor: float = 71.0,
        f0_ceil: float = 800.0,
        f0_smooth: bool = True,
        f0_smooth_window: int = 5,
        use_double_precision: bool = False,
        f0_device: str = "auto",
        crepe_model: str = "full",
    ) -> Dict:
        """仅执行 F0 提取"""
        import time
        start_time = time.time()
        try:
            logger.info(f"[ F0 模式 ] 提取音高 (method={method})")
            config = AudioProcessingConfig(
                f0_method=method, f0_floor=f0_floor, f0_ceil=f0_ceil,
                f0_smooth=f0_smooth, f0_smooth_window=f0_smooth_window,
                use_double_precision=use_double_precision,
                f0_device=f0_device, crepe_model=crepe_model,
            )
            audio_data = self.tsubaki_processor.process_audio_f0(wav_path, config)
            if audio_data and audio_data.get("success"):
                logger.info("✓ F0 提取完成")
                return {
                    "success": True, "method": method,
                    "frames": len(audio_data.get("f0", [])),
                    "sample_rate": audio_data.get("sr", 0),
                    "processing_time": int((time.time() - start_time) * 1000),
                    "message": f'F0 提取完成: {len(audio_data.get("f0", []))} 帧',
                }
            else:
                return {
                    "success": False,
                    "error": (audio_data or {}).get("error", "F0 提取失败"),
                    "processing_time": int((time.time() - start_time) * 1000),
                }
        except Exception as e:
            logger.error(f"✗ F0 提取异常: {e}", exc_info=True)
            return {
                "success": False, "error": str(e),
                "processing_time": int((time.time() - start_time) * 1000),
            }

    def get_supported_formats(self) -> Dict:
        return {
            "formats": list(self.tsubaki_processor.SUPPORTED_FORMATS.keys()),
            "details": self.tsubaki_processor.SUPPORTED_FORMATS,
        }

    def get_status(self) -> Dict:
        from mfa_utils import MFAChecker
        from f0_extractors import get_f0_backend_status
        from alt_aligners import get_alt_aligner_status

        mfa_status = MFAChecker.get_status()
        pyworld_available = self.tsubaki_processor.process_audio_f0.__globals__.get("pw") is not None

        return {
            "initialized": True,
            "work_dir": str(self.work_dir),
            "mfa": mfa_status,
            "audio_processing": {
                "pyworld_available": pyworld_available,
                "supported_formats": list(self.tsubaki_processor.SUPPORTED_FORMATS.keys()),
                "f0_backends": get_f0_backend_status(),
            },
            "alt_aligners": get_alt_aligner_status(),   # ← 新增
        }
