# -*- coding: utf-8 -*-
"""
音频标注和音高处理模块 - 工程文件生成引擎
用于处理 LAB 标注文件和音频数据，生成专业音乐制作工程文件
"""
from __future__ import annotations

import os
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import soundfile as sf

try:
    import pyworld as pw
except ImportError:
    pw = None

logger = logging.getLogger(__name__)


@dataclass
class AudioFrame:
    """音频帧数据"""
    start_time: int  # 100ns 单位
    end_time: int
    frequency: float  # F0 频率
    confidence: float = 1.0


@dataclass
class LabelSegment:
    """标注段"""
    start_time: int  # 100ns 单位
    end_time: int
    label: str


@dataclass
class AudioProcessingConfig:
    """音频处理配置"""
    # 基本参数
    bpm: float = 120.0  # 每分钟节拍数
    base_pitch: int = 60  # 基准音高（MIDI note）
    
    # F0 提取参数
    f0_floor: float = 71.0      # 最低 F0（Hz）
    f0_ceil: float = 800.0      # 最高 F0（Hz）
    f0_method: str = 'dio'      # 'dio' 或 'harvest'
    
    # F0 细化参数
    f0_smooth: bool = True      # 是否平滑 F0
    f0_smooth_window: int = 5   # 平滑窗口大小
    
    # 精度设置
    use_double_precision: bool = False  # 使用双精度浮点数
    
    # 大音频处理
    chunk_size: int = 30 * 60 * 48000  # 30秒音频块大小（48kHz采样率）
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'bpm': self.bpm,
            'base_pitch': self.base_pitch,
            'f0_floor': self.f0_floor,
            'f0_ceil': self.f0_ceil,
            'f0_method': self.f0_method,
            'f0_smooth': self.f0_smooth,
            'f0_smooth_window': self.f0_smooth_window,
            'use_double_precision': self.use_double_precision,
        }


class TsubakiProcessor:
    """
    标注和音高处理器
    
    功能：
    - 音频 F0 提取（使用 PyWORLD）
    - LAB 标注文件解析
    - 工程文件生成（Synthesizer V / OpenUtau 格式）
    - 高级音频处理选项
    """

    # 支持的输出格式
    SUPPORTED_FORMATS = {
        'sv': 'Synthesizer V Studio',
        'utau': 'OpenUtau/UTAU',
    }

    # WORLD 声码器默认参数
    WORLD_PARAMS = {
        'f0_floor': 71.0,      # 最低 F0（Hz）
        'f0_ceil': 800.0,      # 最高 F0（Hz）
        'channels_in_octave': 2,
    }

    def __init__(self, work_dir: Optional[str] = None):
        """
        初始化处理器
        
        Args:
            work_dir: 工作目录，默认使用临时目录
        """
        self.work_dir = Path(work_dir or tempfile.gettempdir())
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        if pw is None:
            logger.warning("PyWORLD 未安装，F0 提取功能将不可用")

    def process_audio_f0(
        self, 
        wav_path: str,
        config: Optional[AudioProcessingConfig] = None
    ) -> Dict[str, np.ndarray]:
        """
        提取音频的基频（F0）
        
        Args:
            wav_path: 音频文件路径
            config: 处理配置
            
        Returns:
            包含 f0、t（时间）、sp（频谱包络）、ap（非周期性）的字典
        """
        if pw is None:
            logger.error("PyWORLD 未安装，无法提取 F0")
            return {}

        if config is None:
            config = AudioProcessingConfig()

        try:
            # 读取音频
            audio, sr = sf.read(wav_path)
            
            # 转换为单声道
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            
            # 精度设置
            if config.use_double_precision:
                audio = audio.astype(np.float64)
            else:
                audio = audio.astype(np.float32)
            
            logger.info(f"✓ 加载音频: {wav_path} ({sr}Hz, {len(audio)} samples, 精度: {'双' if config.use_double_precision else '单'})")
            
            # 处理大音频（分块）
            if len(audio) > config.chunk_size:
                logger.info(f"⚠ 检测到大音频，使用分块处理")
                return self._process_large_audio(audio, sr, config)
            
            # 提取 F0
            if config.f0_method == 'harvest':
                _f0, t = pw.harvest(
                    audio, sr,
                    f0_floor=config.f0_floor,
                    f0_ceil=config.f0_ceil
                )
            else:  # dio (default)
                _f0, t = pw.dio(
                    audio, sr,
                    f0_floor=config.f0_floor,
                    f0_ceil=config.f0_ceil
                )
            
            # 细化 F0
            f0 = pw.stonemask(audio, _f0, t, sr)
            logger.info(f"✓ F0 提取完成 ({config.f0_method}): {len(f0)} 帧")
            
            # F0 平滑处理
            if config.f0_smooth:
                f0 = self._smooth_f0(f0, config.f0_smooth_window)
                logger.info(f"✓ F0 平滑完成（窗口大小: {config.f0_smooth_window}）")
            
            # 提取频谱参数
            sp = pw.cheaptrick(audio, f0, t, sr)
            ap = pw.d4c(audio, f0, t, sr)
            logger.info(f"✓ 频谱参数提取完成")
            
            return {
                'f0': f0,
                't': t,
                'sp': sp,
                'ap': ap,
                'sr': sr,
                'audio': audio,
                'config': config,
            }
            
        except Exception as e:
            logger.error(f"✗ F0 提取失败: {e}", exc_info=True)
            return {}

    def _process_large_audio(
        self,
        audio: np.ndarray,
        sr: int,
        config: AudioProcessingConfig
    ) -> Dict[str, np.ndarray]:
        """
        处理大音频文件（分块处理）
        
        Args:
            audio: 音频数据
            sr: 采样率
            config: 处理配置
            
        Returns:
            处理结果
        """
        chunk_samples = config.chunk_size
        hop_samples = chunk_samples // 2  # 50% 重叠
        
        f0_list = []
        t_list = []
        sp_list = []
        ap_list = []
        
        time_offset = 0
        
        for i in range(0, len(audio), hop_samples):
            if i + chunk_samples > len(audio):
                chunk = audio[i:]
            else:
                chunk = audio[i:i + chunk_samples]
            
            logger.info(f"  处理音频块 {i // hop_samples + 1}: [{i} - {i + len(chunk)}]")
            
            # 提取该块的 F0
            if config.f0_method == 'harvest':
                _f0, t = pw.harvest(
                    chunk, sr,
                    f0_floor=config.f0_floor,
                    f0_ceil=config.f0_ceil
                )
            else:
                _f0, t = pw.dio(
                    chunk, sr,
                    f0_floor=config.f0_floor,
                    f0_ceil=config.f0_ceil
                )
            
            f0 = pw.stonemask(chunk, _f0, t, sr)
            sp = pw.cheaptrick(chunk, f0, t, sr)
            ap = pw.d4c(chunk, f0, t, sr)
            
            # 时间偏移调整
            t = t + (i / sr)
            
            f0_list.append(f0)
            t_list.append(t)
            sp_list.append(sp)
            ap_list.append(ap)
        
        # 合并结果
        f0 = np.concatenate(f0_list)
        t = np.concatenate(t_list)
        sp = np.concatenate(sp_list, axis=0)
        ap = np.concatenate(ap_list, axis=0)
        
        logger.info(f"✓ 大音频处理完成: {len(f0)} 帧")
        
        return {
            'f0': f0,
            't': t,
            'sp': sp,
            'ap': ap,
            'sr': sr,
            'audio': audio,
            'config': config,
        }

    @staticmethod
    def _smooth_f0(
        f0: np.ndarray,
        window_size: int = 5
    ) -> np.ndarray:
        """
        平滑 F0 曲线
        
        Args:
            f0: F0 数据
            window_size: 平滑窗口大小
            
        Returns:
            平滑后的 F0
        """
        from scipy.signal import medfilt
        
        # 只平滑有效的 F0 值（非零）
        voiced = f0 > 0
        
        if np.sum(voiced) == 0:
            return f0
        
        # 中值滤波平滑
        f0_smooth = f0.copy()
        f0_smooth[voiced] = medfilt(f0[voiced], kernel_size=window_size if window_size % 2 == 1 else window_size + 1)
        
        return f0_smooth

    def parse_lab_file(self, lab_path: str) -> List[LabelSegment]:
        """
        解析 LAB 标注文件
        
        LAB 格式: {start_time} {end_time} {label}
        时间单位：100ns
        
        Args:
            lab_path: LAB 文件路径
            
        Returns:
            标注段列表
        """
        segments = []
        try:
            with open(lab_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    parts = line.split()
                    if len(parts) < 3:
                        logger.warning(f"跳过无效行: {line}")
                        continue
                    
                    try:
                        start = int(parts[0])
                        end = int(parts[1])
                        label = ' '.join(parts[2:])
                        
                        segments.append(LabelSegment(start, end, label))
                    except ValueError as e:
                        logger.warning(f"解析行失败: {line} - {e}")
                        continue
            
            logger.info(f"✓ 解析 LAB 文件: {len(segments)} 个标注段")
            return segments
            
        except Exception as e:
            logger.error(f"✗ 读取 LAB 文件失败: {e}", exc_info=True)
            return []

    def generate_sv_project(
        self,
        wav_path: str,
        lab_path: str,
        output_path: str,
        project_title: str = "Project",
        config: Optional[AudioProcessingConfig] = None
    ) -> Dict:
        """
        生成 Synthesizer V Studio 工程文件
        
        Args:
            wav_path: 音频文件路径
            lab_path: LAB 标注文件路径
            output_path: 输出文件路径 (.ustx)
            project_title: 工程标题
            config: 处理配置
            
        Returns:
            处理结果字典
        """
        if config is None:
            config = AudioProcessingConfig()

        try:
            # 提取音频 F0
            audio_data = self.process_audio_f0(wav_path, config)
            if not audio_data:
                return {'success': False, 'error': 'F0 提取失败'}
            
            # 解析 LAB 文件
            segments = self.parse_lab_file(lab_path)
            if not segments:
                return {'success': False, 'error': 'LAB 文件解析失败'}
            
            f0 = audio_data['f0']
            t = audio_data['t']
            sr = audio_data['sr']
            
            # 生成基本 USTx 格式
            project_xml = self._build_sv_project_xml(
                project_title, segments, f0, t, sr, wav_path, config
            )
            
            # 保存文件
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(project_xml)
            
            logger.info(f"✓ 生成 SV 工程文件: {output_path}")
            return {
                'success': True,
                'format': 'sv',
                'output_path': output_path,
                'segments': len(segments),
                'config': config.to_dict(),
                'message': f'生成 Synthesizer V 工程文件: {len(segments)} 个标注段'
            }
            
        except Exception as e:
            logger.error(f"✗ SV 工程生成失败: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}

    def generate_utau_project(
        self,
        wav_path: str,
        lab_path: str,
        output_path: str,
        project_title: str = "Project",
        config: Optional[AudioProcessingConfig] = None
    ) -> Dict:
        """
        生成 OpenUtau/UTAU 工程文件
        
        Args:
            wav_path: 音频文件路径
            lab_path: LAB 标注文件路径
            output_path: 输出文件路径 (.ustx)
            project_title: 工程标题
            config: 处理配置
            
        Returns:
            处理结果字典
        """
        if config is None:
            config = AudioProcessingConfig()

        try:
            # 提取音频 F0
            audio_data = self.process_audio_f0(wav_path, config)
            if not audio_data:
                return {'success': False, 'error': 'F0 提取失败'}
            
            # 解析 LAB 文件
            segments = self.parse_lab_file(lab_path)
            if not segments:
                return {'success': False, 'error': 'LAB 文件解析失败'}
            
            f0 = audio_data['f0']
            t = audio_data['t']
            sr = audio_data['sr']
            
            # 生成 UTAU/USTx 格式
            project_xml = self._build_utau_project_xml(
                project_title, segments, f0, t, sr, wav_path, config
            )
            
            # 保存文件
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(project_xml)
            
            logger.info(f"✓ 生成 UTAU 工程文件: {output_path}")
            return {
                'success': True,
                'format': 'utau',
                'output_path': output_path,
                'segments': len(segments),
                'config': config.to_dict(),
                'message': f'生成 OpenUtau/UTAU 工程文件: {len(segments)} 个标注段'
            }
            
        except Exception as e:
            logger.error(f"✗ UTAU 工程生成失败: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}

    def _build_sv_project_xml(
        self,
        title: str,
        segments: List[LabelSegment],
        f0: np.ndarray,
        t: np.ndarray,
        sr: int,
        wav_path: str,
        config: AudioProcessingConfig
    ) -> str:
        """构建 Synthesizer V USTx XML"""
        xml_lines = [
            '<?xml version="1.0" encoding="utf-8"?>',
            '<project version="1.0">',
            f'  <title>{self._escape_xml(title)}</title>',
            '  <audio>',
            f'    <path>{self._escape_xml(str(Path(wav_path).absolute()))}</path>',
            '  </audio>',
            '  <voice>',
            '    <database name="default" />',
            '  </voice>',
            '  <metadata>',
            f'    <bpm>{config.bpm}</bpm>',
            f'    <basePitch>{config.base_pitch}</basePitch>',
            '  </metadata>',
            '  <script>',
        ]
        
        # 添加音符数据
        for i, segment in enumerate(segments):
            # 将 100ns 转换为秒
            start_sec = segment.start_time / 10000000.0
            end_sec = segment.end_time / 10000000.0
            duration_sec = end_sec - start_sec
            
            # 提取该时间段的平均 F0
            time_mask = (t >= start_sec) & (t < end_sec)
            segment_f0 = f0[time_mask]
            avg_f0 = np.mean(segment_f0[segment_f0 > 0]) if np.any(segment_f0 > 0) else 0
            
            xml_lines.append('    <note>')
            xml_lines.append(f'      <pos>{i}</pos>')
            xml_lines.append(f'      <lyric>{self._escape_xml(segment.label)}</lyric>')
            xml_lines.append(f'      <duration>{int(duration_sec * 1920)}</duration>')  # 四分音符 = 1920
            if avg_f0 > 0:
                xml_lines.append(f'      <pitch>{int(np.log2(avg_f0 / 440) * 1200)}</pitch>')
            xml_lines.append('    </note>')
        
        xml_lines.extend([
            '  </script>',
            '</project>',
        ])
        
        return '\n'.join(xml_lines)

    def _build_utau_project_xml(
        self,
        title: str,
        segments: List[LabelSegment],
        f0: np.ndarray,
        t: np.ndarray,
        sr: int,
        wav_path: str,
        config: AudioProcessingConfig
    ) -> str:
        """构建 UTAU/USTx XML"""
        xml_lines = [
            '<?xml version="1.0" encoding="utf-8"?>',
            '<project version="1.0">',
            f'  <title>{self._escape_xml(title)}</title>',
            '  <audio>',
            f'    <path>{self._escape_xml(str(Path(wav_path).absolute()))}</path>',
            '  </audio>',
            '  <metadata>',
            f'    <bpm>{config.bpm}</bpm>',
            f'    <basePitch>{config.base_pitch}</basePitch>',
            '  </metadata>',
            '  <notes>',
        ]
        
        # 添加音符数据
        for i, segment in enumerate(segments):
            # 将 100ns 转换为秒
            start_sec = segment.start_time / 10000000.0
            end_sec = segment.end_time / 10000000.0
            duration_sec = end_sec - start_sec
            
            # 提取该时间段的平均 F0
            time_mask = (t >= start_sec) & (t < end_sec)
            segment_f0 = f0[time_mask]
            avg_f0 = np.mean(segment_f0[segment_f0 > 0]) if np.any(segment_f0 > 0) else 0
            
            xml_lines.append('    <note>')
            xml_lines.append(f'      <index>{i}</index>')
            xml_lines.append(f'      <lyric>{self._escape_xml(segment.label)}</lyric>')
            xml_lines.append(f'      <start>{int(start_sec * 1000)}</start>')
            xml_lines.append(f'      <duration>{int(duration_sec * 1000)}</duration>')
            if avg_f0 > 0:
                # MIDI 音高计算
                midi_pitch = int(12 * np.log2(avg_f0 / 440) + 69)
                xml_lines.append(f'      <pitch>{midi_pitch}</pitch>')
            xml_lines.append('    </note>')
        
        xml_lines.extend([
            '  </notes>',
            '</project>',
        ])
        
        return '\n'.join(xml_lines)

    @staticmethod
    def _escape_xml(text: str) -> str:
        """转义 XML 特殊字符"""
        return (text
                .replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('"', '&quot;')
                .replace("'", '&apos;'))

    def process_full_pipeline(
        self,
        wav_path: str,
        lab_path: str,
        output_format: str = 'sv',
        project_title: str = "Project",
        config: Optional[AudioProcessingConfig] = None
    ) -> Dict:
        """
        完整处理流程：音频 → F0 → 标注 → 工程文件
        
        Args:
            wav_path: 音频文件路径
            lab_path: LAB 标注文件路径
            output_format: 输出格式 ('sv' 或 'utau')
            project_title: 工程标题
            config: 处理配置
            
        Returns:
            处理结果字典
        """
        if output_format not in self.SUPPORTED_FORMATS:
            return {
                'success': False,
                'error': f"不支持的格式: {output_format}，支持: {', '.join(self.SUPPORTED_FORMATS.keys())}"
            }
        
        if config is None:
            config = AudioProcessingConfig()
        
        # 确定输出文件名
        stem = Path(lab_path).stem
        output_ext = '.svp' if output_format == 'sv' else '.ustx'
        output_path = str(self.work_dir / f"{stem}_{output_format}{output_ext}")
        
        # 调用相应的生成函数
        if output_format == 'sv':
            return self.generate_sv_project(wav_path, lab_path, output_path, project_title, config)
        elif output_format == 'utau':
            return self.generate_utau_project(wav_path, lab_path, output_path, project_title, config)
        
        return {'success': False, 'error': '未知格式'}