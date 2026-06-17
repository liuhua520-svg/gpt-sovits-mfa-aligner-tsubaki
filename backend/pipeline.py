# -*- coding: utf-8 -*-
"""
完整处理流程管道（v2.1 — 多后端对齐支持）
整合 MFA / WhisperX / Qwen3-ASR / Qwen3-ForcedAligner + 音高处理 + 工程文件生成

新增参数: aligner_backend
  "mfa"           — Montreal Forced Aligner（默认）
  "whisperx"      — WhisperX (Whisper ASR + wav2vec2 强制对齐)
  "qwen3_asr"     — Qwen3-ASR-1.7B (自动语音识别，文本可选)
  "qwen3_aligner" — Qwen3-ForcedAligner-0.6B (强制对齐，需要参考文本)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

from mfa_processor import MFAProcessor
from tsubaki_processor import TsubakiProcessor, AudioProcessingConfig

logger = logging.getLogger(__name__)


def _run_alignment(
    audio_file,             # Flask FileStorage 或 FileStorageWrapper
    text: str,
    language: str,
    backend: str = "mfa",
    f0_device: str = "auto",
) -> Dict:
    """
    统一调度对齐后端，返回与 MFAProcessor.process() 格式兼容的字典。
    audio_file.save(path) 和 audio_file.filename 必须可用。
    """
    if backend == "mfa":
        processor = MFAProcessor()
        return processor.process(audio_file, text, language)

    # ── 替代后端 ──────────────────────────────────────────────────────────
    import tempfile, shutil, os
    from alt_aligners import get_aligner

    # 把文件保存到临时目录，获取路径供 alt aligner 使用
    tmp_dir = tempfile.mkdtemp(prefix="alt_aligner_")
    try:
        filename = getattr(audio_file, "filename", "audio.wav")
        tmp_wav = os.path.join(tmp_dir, filename)
        audio_file.save(tmp_wav)

        aligner = get_aligner(backend, device=f0_device)
        return aligner.align(tmp_wav, text or None, language)
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
            align_result = _run_alignment(audio_file, text, language, aligner_backend, f0_device)
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
    ) -> Dict:
        """仅执行对齐标注（不生成工程文件）"""
        import time
        start_time = time.time()

        try:
            logger.info(f"[ 标注模式 ] 执行自动标注 (backend={aligner_backend})")

            result = _run_alignment(audio_file, text, language, aligner_backend, f0_device)

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
