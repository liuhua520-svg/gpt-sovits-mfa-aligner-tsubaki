# -*- coding: utf-8 -*-
"""
完整处理流程管道
整合 MFA 自动标注 + 音高处理 + 工程文件生成
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

from mfa_processor import MFAProcessor
from tsubaki_processor import TsubakiProcessor, AudioProcessingConfig

logger = logging.getLogger(__name__)


class AudioProcessingPipeline:
    """
    完整音频处理流程：
    1. 音频上传
    2. MFA 自动标注 (生成 LAB 文件)
    3. 音高提取 (使用 PyWORLD)
    4. 工程文件生成 (SV / USTX)
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
        ) -> Dict:
            config = AudioProcessingConfig(
                bpm=bpm,
                base_pitch=base_pitch,
                f0_method=f0_method,
                f0_smooth=f0_smooth,
                f0_smooth_window=f0_smooth_window,
                refine_pitch=refine_pitch,
                use_double_precision=use_double_precision,
                f0_floor=f0_floor,
                f0_ceil=f0_ceil,
            )

            import time
            start_time = time.time()

            try:
                logger.info("=" * 60)
                logger.info("开始完整处理流程")
                logger.info("=" * 60)

                # === 【前置核心修改：统一且强制先保存当前的最新音频】 ===
                audio_filename = getattr(audio_file, "filename", "audio")
                stem = Path(audio_filename).stem
                wav_path = str(self.work_dir / f"{stem}.wav")
                
                logger.info(f"正在同步保存本次上传的音频文件: {wav_path}")
                Path(wav_path).parent.mkdir(parents=True, exist_ok=True)
                audio_file.seek(0)
                audio_file.save(wav_path)
                audio_file.seek(0)  # 保持好习惯，保存完立刻复位
                # ====================================================

                # 步骤 1: MFA 自动标注
                logger.info("[ 步骤 1/3 ] MFA 自动标注...")
                mfa_result = self.mfa_processor.process(audio_file, text, language)

                if not mfa_result.get("success"):
                    error = mfa_result.get("error", "MFA 处理失败")
                    logger.error(f"✗ MFA 失败: {error}")
                    return {
                        "success": False,
                        "error": error,
                        "stage": "mfa_alignment",
                        "processing_time": int((time.time() - start_time) * 1000),
                    }

                lab_content = mfa_result.get("lab_content", "")
                lab_path = str(self.work_dir / f"{stem}.lab")

                with open(lab_path, "w", encoding="utf-8") as f:
                    f.write(lab_content)

                logger.info(f"✓ LAB 标注完成: {lab_path}")

                # 步骤 2: 音高提取 （直接使用上面确定的 wav_path）
                logger.info("[ 步骤 2/3 ] 音高提取...")
                try:
                    audio_data = self.tsubaki_processor.process_audio_f0(wav_path, config)
                    if not audio_data:
                        logger.warning("⚠ F0 提取失败或 PyWORLD 未安装，继续处理")
                        audio_data = None
                except Exception as e:
                    logger.warning(f"⚠ 音高提取异常: {e}，继续处理")
                    audio_data = None

                logger.info("✓ 音高提取完成")

                # 步骤 3: 工程文件生成
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
                        "success": False,
                        "error": error,
                        "stage": "project_generation",
                        "lab_path": lab_path,
                        "processing_time": int((time.time() - start_time) * 1000),
                    }

                project_path = project_result.get("output_path")
                processing_time = int((time.time() - start_time) * 1000)

                logger.info("=" * 60)
                logger.info("✓ 完整处理流程完成")
                logger.info(f"  LAB 文件: {lab_path}")
                logger.info(f"  工程文件: {project_path}")
                logger.info(f"  耗时: {processing_time}ms")
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
                    "message": f'完整处理完成: {project_result.get("segments", 0)} 个标注段',
                }

            except Exception as e:
                logger.error(f"✗ 处理流程异常: {e}", exc_info=True)
                return {
                    "success": False,
                    "error": str(e),
                    "stage": "unknown",
                    "processing_time": int((time.time() - start_time) * 1000),
                }

    def process_mfa_only(
        self,
        audio_file,
        text: str,
        language: str = "cmn"
    ) -> Dict:
        """仅执行 MFA 标注（不生成工程文件）"""
        import time
        start_time = time.time()

        try:
            logger.info("[ MFA 模式 ] 仅执行自动标注")

            result = self.mfa_processor.process(audio_file, text, language)

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
                return result
            else:
                return {
                    **result,
                    "processing_time": int((time.time() - start_time) * 1000)
                }

        except Exception as e:
            logger.error(f"✗ MFA 处理异常: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "processing_time": int((time.time() - start_time) * 1000)
            }

    def process_project_only(
        self,
        wav_path: str,
        lab_path: str,
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
    ) -> Dict:
        """仅执行工程文件生成（已有 WAV 和 LAB）"""
        import time
        start_time = time.time()

        try:
            logger.info("[ 工程文件模式 ] 生成项目文件")
            logger.info(f"  音频: {wav_path}")
            logger.info(f"  标注: {lab_path}")
            logger.info(f"  格式: {output_format}")

            config = AudioProcessingConfig(
                bpm=bpm,
                base_pitch=base_pitch,
                f0_floor=f0_floor,
                f0_ceil=f0_ceil,
                f0_method=f0_method,
                f0_smooth=f0_smooth,
                f0_smooth_window=f0_smooth_window,
                use_double_precision=use_double_precision,
                refine_pitch=refine_pitch,
            )

            result = self.tsubaki_processor.process_full_pipeline(
                wav_path=wav_path,
                lab_path=lab_path,
                output_format=output_format,
                project_title=project_title,
                config=config
            )

            result["processing_time"] = int((time.time() - start_time) * 1000)
            return result

        except Exception as e:
            logger.error(f"✗ 工程文件生成异常: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "processing_time": int((time.time() - start_time) * 1000)
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
    ) -> Dict:
        """仅执行 F0 提取"""
        import time
        start_time = time.time()

        try:
            logger.info("[ F0 模式 ] 提取音高")

            config = AudioProcessingConfig(
                f0_method=method,
                f0_floor=f0_floor,
                f0_ceil=f0_ceil,
                f0_smooth=f0_smooth,
                f0_smooth_window=f0_smooth_window,
                use_double_precision=use_double_precision,
            )

            audio_data = self.tsubaki_processor.process_audio_f0(wav_path, config)

            if audio_data:
                logger.info("✓ F0 提取完成")
                return {
                    "success": True,
                    "method": method,
                    "frames": len(audio_data.get("f0", [])),
                    "sample_rate": audio_data.get("sr", 0),
                    "processing_time": int((time.time() - start_time) * 1000),
                    "message": f'F0 提取完成: {len(audio_data.get("f0", []))} 帧'
                }
            else:
                return {
                    "success": False,
                    "error": "F0 提取失败",
                    "processing_time": int((time.time() - start_time) * 1000)
                }

        except Exception as e:
            logger.error(f"✗ F0 提取异常: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "processing_time": int((time.time() - start_time) * 1000)
            }

    def get_supported_formats(self) -> Dict:
        return {
            "formats": list(self.tsubaki_processor.SUPPORTED_FORMATS.keys()),
            "details": self.tsubaki_processor.SUPPORTED_FORMATS
        }

    def get_status(self) -> Dict:
        from mfa_utils import MFAChecker

        mfa_status = MFAChecker.get_status()
        pyworld_available = self.tsubaki_processor.process_audio_f0.__globals__.get("pw") is not None

        return {
            "initialized": True,
            "work_dir": str(self.work_dir),
            "mfa": mfa_status,
            "audio_processing": {
                "pyworld_available": pyworld_available,
                "supported_formats": list(self.tsubaki_processor.SUPPORTED_FORMATS.keys())
            }
        }
