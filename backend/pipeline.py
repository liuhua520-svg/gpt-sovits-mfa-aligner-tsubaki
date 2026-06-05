# -*- coding: utf-8 -*-
"""
完整处理流程管道
整合 MFA 自动标注 + 音高处理 + 工程文件生成
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Optional

from mfa_processor import MFAProcessor
from tsubaki_processor import TsubakiProcessor

logger = logging.getLogger(__name__)


class AudioProcessingPipeline:
    """
    完整音频处理流程：
    1. 音频上传
    2. MFA 自动标注 (生成 LAB 文件)
    3. 音高提取 (使用 PyWORLD)
    4. 工程文件生成 (SV / UTAU)
    """

    def __init__(self, work_dir: str):
        """
        初始化处理流程
        
        Args:
            work_dir: 工作目录
        """
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        self.mfa_processor = MFAProcessor()
        self.tsubaki_processor = TsubakiProcessor(str(self.work_dir / "projects"))
        
        logger.info(f"✓ 处理流程初始化，工作目录: {self.work_dir}")

    def process_full(
        self,
        audio_file,
        text: str,
        language: str = "cmn",
        output_format: str = "sv",
        project_title: str = "Project"
    ) -> Dict:
        """
        完整处理流程
        
        Args:
            audio_file: 上传的音频文件对象
            text: 转录文本
            language: 语言代码
            output_format: 输出工程格式 ('sv' 或 'utau')
            project_title: 工程文件标题
            
        Returns:
            处理结果字典，包含:
            - success: 是否成功
            - error: 错误信息（失败时）
            - lab_path: LAB 标注文件路径
            - project_path: 生成的工程文件路径
            - project_format: 工程文件格式
            - segments: 标注段数
            - processing_time: 处理耗时（毫秒）
        """
        import time
        start_time = time.time()
        
        try:
            logger.info("=" * 60)
            logger.info("开始完整处理流程")
            logger.info("=" * 60)
            
            # 步骤 1: MFA 自动标注
            logger.info("[ 步骤 1/3 ] MFA 自动标注...")
            mfa_result = self.mfa_processor.process(audio_file, text, language)
            
            if not mfa_result.get("success"):
                error = mfa_result.get("error", "MFA 处理失败")
                logger.error(f"✗ MFA 失败: {error}")
                return {
                    'success': False,
                    'error': error,
                    'stage': 'mfa_alignment',
                    'processing_time': int((time.time() - start_time) * 1000)
                }
            
            # 保存 LAB 文件
            lab_content = mfa_result.get("lab_content", "")
            audio_filename = getattr(audio_file, 'filename', 'audio')
            stem = Path(audio_filename).stem
            lab_path = str(self.work_dir / f"{stem}.lab")
            
            Path(lab_path).parent.mkdir(parents=True, exist_ok=True)
            with open(lab_path, 'w', encoding='utf-8') as f:
                f.write(lab_content)
            
            logger.info(f"✓ LAB 标注完成: {lab_path}")
            
            # 获取音频文件路径（从 MFA 结果中）
            # MFA 处理后的音频应该存在于工作目录
            wav_path = None
            for wav_file in self.work_dir.glob("**/*.wav"):
                if stem in wav_file.stem:
                    wav_path = str(wav_file)
                    break
            
            if not wav_path:
                logger.warning("未找到 WAV 文件，尝试保存上传的文件")
                wav_path = str(self.work_dir / f"{stem}.wav")
                audio_file.seek(0)
                audio_file.save(wav_path)
                audio_file.seek(0)
            
            # 步骤 2: 音高提取
            logger.info("[ 步骤 2/3 ] 音高提取...")
            try:
                audio_data = self.tsubaki_processor.process_audio_f0(wav_path, method='dio')
                if not audio_data:
                    logger.warning("⚠ F0 提取失败或 PyWORLD 未安装，继续处理")
            except Exception as e:
                logger.warning(f"⚠ 音高提取异常: {e}，继续处理")
            
            logger.info("✓ 音高提取完成")
            
            # 步骤 3: 工程文件生成
            logger.info(f"[ 步骤 3/3 ] 生成 {output_format.upper()} 工程文件...")
            
            project_result = self.tsubaki_processor.process_full_pipeline(
                wav_path=wav_path,
                lab_path=lab_path,
                output_format=output_format,
                project_title=project_title
            )
            
            if not project_result.get("success"):
                error = project_result.get("error", "工程文件生成失败")
                logger.error(f"✗ 工程文件生成失败: {error}")
                return {
                    'success': False,
                    'error': error,
                    'stage': 'project_generation',
                    'lab_path': lab_path,
                    'processing_time': int((time.time() - start_time) * 1000)
                }
            
            project_path = project_result.get("output_path")
            logger.info(f"✓ 工程文件生成完成: {project_path}")
            
            processing_time = int((time.time() - start_time) * 1000)
            
            logger.info("=" * 60)
            logger.info("✓ 完整处理流程完成")
            logger.info(f"  LAB 文件: {lab_path}")
            logger.info(f"  工程文件: {project_path}")
            logger.info(f"  耗时: {processing_time}ms")
            logger.info("=" * 60)
            
            return {
                'success': True,
                'lab_path': lab_path,
                'lab_content': lab_content,
                'project_path': project_path,
                'project_format': output_format,
                'segments': project_result.get('segments', 0),
                'processing_time': processing_time,
                'message': f'完整处理完成: {project_result.get("segments", 0)} 个标注段'
            }
            
        except Exception as e:
            logger.error(f"✗ 处理流程异常: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'stage': 'unknown',
                'processing_time': int((time.time() - start_time) * 1000)
            }

    def process_mfa_only(
        self,
        audio_file,
        text: str,
        language: str = "cmn"
    ) -> Dict:
        """
        仅执行 MFA 标注（不生成工程文件）
        
        Args:
            audio_file: 上传的音频文件对象
            text: 转录文本
            language: 语言代码
            
        Returns:
            MFA 处理结果
        """
        import time
        start_time = time.time()
        
        try:
            logger.info("[ MFA 模式 ] 仅执行自动标注")
            
            # 执行 MFA 处理
            result = self.mfa_processor.process(audio_file, text, language)
            
            if result.get("success"):
                # 保存 LAB 文件
                lab_content = result.get("lab_content", "")
                audio_filename = getattr(audio_file, 'filename', 'audio')
                stem = Path(audio_filename).stem
                lab_path = str(self.work_dir / f"{stem}.lab")
                
                Path(lab_path).parent.mkdir(parents=True, exist_ok=True)
                with open(lab_path, 'w', encoding='utf-8') as f:
                    f.write(lab_content)
                
                logger.info(f"✓ LAB 标注已保存: {lab_path}")
                
                result['lab_path'] = lab_path
                result['processing_time'] = int((time.time() - start_time) * 1000)
                return result
            else:
                return {
                    **result,
                    'processing_time': int((time.time() - start_time) * 1000)
                }
                
        except Exception as e:
            logger.error(f"✗ MFA 处理异常: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'processing_time': int((time.time() - start_time) * 1000)
            }

    def process_project_only(
        self,
        wav_path: str,
        lab_path: str,
        output_format: str = "sv",
        project_title: str = "Project"
    ) -> Dict:
        """
        仅执行工程文件生成（已有 WAV 和 LAB）
        
        Args:
            wav_path: 音频文件路径
            lab_path: LAB 标注文件路径
            output_format: 输出格式
            project_title: 工程标题
            
        Returns:
            工程文件生成结果
        """
        import time
        start_time = time.time()
        
        try:
            logger.info("[ 工程文件模式 ] 生成项目文件")
            logger.info(f"  音频: {wav_path}")
            logger.info(f"  标注: {lab_path}")
            logger.info(f"  格式: {output_format}")
            
            result = self.tsubaki_processor.process_full_pipeline(
                wav_path=wav_path,
                lab_path=lab_path,
                output_format=output_format,
                project_title=project_title
            )
            
            result['processing_time'] = int((time.time() - start_time) * 1000)
            return result
            
        except Exception as e:
            logger.error(f"✗ 工程文件生成异常: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'processing_time': int((time.time() - start_time) * 1000)
            }

    def process_f0_only(
        self,
        wav_path: str,
        method: str = 'dio'
    ) -> Dict:
        """
        仅执行 F0 提取
        
        Args:
            wav_path: 音频文件路径
            method: 提取方法 ('dio' 或 'harvest')
            
        Returns:
            F0 提取结果
        """
        import time
        start_time = time.time()
        
        try:
            logger.info("[ F0 模式 ] 提取音高")
            
            audio_data = self.tsubaki_processor.process_audio_f0(wav_path, method=method)
            
            if audio_data:
                logger.info(f"✓ F0 提取完成")
                return {
                    'success': True,
                    'method': method,
                    'frames': len(audio_data.get('f0', [])),
                    'sample_rate': audio_data.get('sr', 0),
                    'processing_time': int((time.time() - start_time) * 1000),
                    'message': f'F0 提取完成: {len(audio_data.get("f0", []))} 帧'
                }
            else:
                return {
                    'success': False,
                    'error': 'F0 提取失败',
                    'processing_time': int((time.time() - start_time) * 1000)
                }
                
        except Exception as e:
            logger.error(f"✗ F0 提取异常: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'processing_time': int((time.time() - start_time) * 1000)
            }

    def get_supported_formats(self) -> Dict:
        """获取支持的输出格式"""
        return {
            'formats': list(self.tsubaki_processor.SUPPORTED_FORMATS.keys()),
            'details': self.tsubaki_processor.SUPPORTED_FORMATS
        }

    def get_status(self) -> Dict:
        """获取处理流程的状态"""
        from mfa_utils import MFAChecker
        
        mfa_status = MFAChecker.get_status()
        pyworld_available = self.tsubaki_processor.process_audio_f0.__module__ != 'None'
        
        return {
            'initialized': True,
            'work_dir': str(self.work_dir),
            'mfa': mfa_status,
            'audio_processing': {
                'pyworld_available': pyworld_available,
                'supported_formats': list(self.tsubaki_processor.SUPPORTED_FORMATS.keys())
            }
        }