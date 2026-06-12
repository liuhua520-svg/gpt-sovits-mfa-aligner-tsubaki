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
        export_pitch_line: bool = True,
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
        )

        import time
        start_time = time.time()

        try:
            logger.info("=" * 60)
            logger.info("开始完整处理流程")
            logger.info("=" * 60)

            audio_filename = getattr(audio_file, "filename", "audio")
            stem = Path(audio_filename).stem
            wav_path = str(self.work_dir / f"{stem}.wav")

            logger.info(f"正在同步保存本次上传的音频文件: {wav_path}")
            Path(wav_path).parent.mkdir(parents=True, exist_ok=True)
            audio_file.seek(0)
            audio_file.save(wav_path)
            audio_file.seek(0)

            logger.info("[ 步骤 1/3 ] MFA 自动标注...")
            mfa_result = self.mfa_processor.process(audio_file, text, language,
                                                    save_dir=str(self.work_dir))
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

            logger.info(f"[ 步骤 3/3 ] 生成 {output_format.upper()} 工程文件...")
            # ★ 优先传入 TextGrid，让 tsubaki_processor 直接用 TextGrid 确定音符边界
            saved_tg_path = mfa_result.get("textgrid_path", "")
            project_result = self.tsubaki_processor.process_full_pipeline(
                wav_path=wav_path,
                lab_path=lab_path,
                textgrid_path=saved_tg_path if saved_tg_path else None,
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
            logger.info(f" LAB 文件: {lab_path}")
            logger.info(f" 工程文件: {project_path}")
            logger.info(f" 耗时: {processing_time}ms")
            logger.info("=" * 60)

            return {
                "success": True,
                "lab_path": lab_path,
                "lab_content": lab_content,
                "textgrid_path": mfa_result.get("textgrid_path", ""),
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

            result = self.mfa_processor.process(audio_file, text, language,
                                                save_dir=str(self.work_dir))

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
        export_pitch_line: bool = True,
    ) -> Dict:
        """仅执行工程文件生成（已有 WAV 和 LAB）"""
        import time
        start_time = time.time()

        try:
            logger.info("[ 工程文件模式 ] 生成项目文件")
            logger.info(f" 音频: {wav_path}")
            logger.info(f" 标注: {lab_path}")
            logger.info(f" 格式: {output_format}")

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
                export_pitch_line=export_pitch_line,
            )

            audio_data = None
            try:
                audio_data = self.tsubaki_processor.process_audio_f0(wav_path, config)
                if not audio_data:
                    logger.warning("⚠ F0 提取失败或 PyWORLD 未安装，继续生成工程文件")
                    audio_data = None
            except Exception as e:
                logger.warning(f"⚠ 音高提取异常: {e}，继续生成工程文件")
                audio_data = None

            result = self.tsubaki_processor.process_full_pipeline(
                wav_path=wav_path,
                lab_path=lab_path,
                output_format=output_format,
                project_title=project_title,
                config=config,
                audio_f0_data=audio_data,
            )

            result["processing_time"] = int((time.time() - start_time) * 1000)
            return result

        except Exception as e:
            logger.error(f"✗ 工程文件生成异常: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
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

    def textgrid_to_lab(
        self,
        textgrid_path: str,
        text: str,
        language: str = "cmn",
    ) -> Dict:
        """
        将 TextGrid 文件转换为 LAB 格式。
        可在「仅生成工程」模式中，用 TextGrid 代替手动准备的 LAB 文件。
        """
        import time
        start_time = time.time()
        try:
            if not Path(textgrid_path).exists():
                return {"success": False, "error": f"TextGrid 文件不存在: {textgrid_path}"}

            # 复用 mfa_processor 的转换逻辑
            lang_map = {
                "cmn": "zh", "zh": "zh", "zh-cn": "zh",
                "yue": "yue", "cantonese": "yue",
                "eng": "en", "en": "en", "english": "en",
                "jpn": "ja", "ja": "ja", "japanese": "ja",
                "kor": "ko", "ko": "ko", "korean": "ko",
            }
            lang = lang_map.get(language.lower(), language.lower())

            lab_content = self.mfa_processor._textgrid_to_lab(textgrid_path, text, lang=lang)

            if not lab_content:
                return {"success": False, "error": "TextGrid 转换结果为空"}

            stem = Path(textgrid_path).stem
            lab_path = self.work_dir / f"{stem}.lab"
            lab_path.write_text(lab_content, encoding="utf-8")
            logger.info(f"✓ TextGrid → LAB 完成: {lab_path}")

            return {
                "success": True,
                "lab_content": lab_content,
                "lab_path": str(lab_path),
                "processing_time": int((time.time() - start_time) * 1000),
            }
        except Exception as e:
            logger.error(f"TextGrid → LAB 失败: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "processing_time": int((time.time() - start_time) * 1000),
            }

    def process_from_textgrid(
        self,
        textgrid_path: str,
        wav_path: str,
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
    ) -> Dict:
        """
        从 TextGrid 文件直接生成工程文件（TextGrid → LAB → SVP/USTX）。
        保证音符时长与 TextGrid phone tier 一致，彻底避免比例分配造成的错位。
        """
        import time
        start_time = time.time()
        try:
            # 步骤 1：TextGrid → LAB
            tg_result = self.textgrid_to_lab(textgrid_path, text, language)
            if not tg_result.get("success"):
                return {**tg_result, "processing_time": int((time.time() - start_time) * 1000)}

            lab_path = tg_result["lab_path"]

            # 步骤 2：生成工程文件
            project_result = self.process_project_only(
                wav_path=wav_path,
                lab_path=lab_path,
                output_format=output_format,
                project_title=project_title,
                bpm=bpm,
                base_pitch=base_pitch,
                f0_method=f0_method,
                f0_smooth=f0_smooth,
                f0_smooth_window=f0_smooth_window,
                use_double_precision=use_double_precision,
                f0_floor=f0_floor,
                f0_ceil=f0_ceil,
                refine_pitch=refine_pitch,
                export_pitch_line=export_pitch_line,
            )
            project_result["lab_content"] = tg_result.get("lab_content", "")
            project_result["lab_path"] = lab_path
            project_result["textgrid_path"] = textgrid_path
            project_result["processing_time"] = int((time.time() - start_time) * 1000)
            return project_result
        except Exception as e:
            logger.error(f"TextGrid 工程生成失败: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "processing_time": int((time.time() - start_time) * 1000),
            }

    def get_status(self) -> Dict:
        """返回系统状态：MFA 安装状态、模型下载状态、PyWORLD 可用性。"""
        from mfa_utils import MFAChecker

        mfa_status = MFAChecker.get_status()

        # 检测 PyWORLD 是否可用（直接尝试 import）
        try:
            import pyworld  # noqa: F401
            pyworld_available = True
        except ImportError:
            pyworld_available = False

        return {
            "initialized": True,
            "work_dir": str(self.work_dir),
            "mfa": mfa_status,
            "audio_processing": {
                "pyworld_available": pyworld_available,
                "supported_formats": list(self.tsubaki_processor.SUPPORTED_FORMATS.keys()),
            },
        }
