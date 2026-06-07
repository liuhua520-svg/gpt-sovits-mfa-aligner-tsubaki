# -*- coding: utf-8 -*-
"""
音频标注和音高处理模块 - 工程文件生成引擎
用于处理 LAB 标注文件和音频数据，生成 Synthesizer V / OpenUtau 工程文件
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    bpm: float = 120.0
    base_pitch: int = 60  # MIDI note，60 = C4

    # F0 提取参数
    f0_floor: float = 71.0
    f0_ceil: float = 800.0
    f0_method: str = "dio"  # 'dio' 或 'harvest'

    # F0 细化参数
    f0_smooth: bool = True
    f0_smooth_window: int = 5

    # 是否细化音高（控制 LAB 音符音高，是否用 F0 平均音高决定 tone）
    refine_pitch: bool = False

    use_double_precision: bool = False

    def to_dict(self) -> Dict:
        return {
            "bpm": self.bpm,
            "base_pitch": self.base_pitch,
            "f0_floor": self.f0_floor,
            "f0_ceil": self.f0_ceil,
            "f0_method": self.f0_method,
            "f0_smooth": self.f0_smooth,
            "f0_smooth_window": self.f0_smooth_window,
            "refine_pitch": self.refine_pitch,
            "use_double_precision": self.use_double_precision,
        }


class TsubakiProcessor:
    """
    工程文件生成器。

    说明：
    - Synthesizer V 的正确结构需要把音符放进 tracks[0].mainGroup.notes，
      并让 tracks[0].mainRef.groupID 指向 mainGroup.uuid。
    - OpenUtau 则输出标准 USTX。
    """

    SUPPORTED_FORMATS = {
        "sv": "Synthesizer V Project (.svp)",
        "ustx": "OpenUtau Project (.ustx)",
        "utau": "OpenUtau Project (.ustx) [alias]",
    }

    OUTPUT_ALIASES = {
        "svp": "sv",
        "synthv": "sv",
        "synthesizer_v": "sv",
        "synthesizerv": "sv",
        "openutau": "ustx",
        "utau": "ustx",
    }

    def __init__(self, work_dir: str):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # 基础工具
    # ----------------------------
    @staticmethod
    def _normalize_output_format(output_format: str) -> str:
        fmt = (output_format or "").strip().lower()
        fmt = TsubakiProcessor.OUTPUT_ALIASES.get(fmt, fmt)
        return fmt

    @staticmethod
    def _midi_to_freq(midi_note: int) -> float:
        return 440.0 * (2 ** ((midi_note - 69) / 12))

    @staticmethod
    def _freq_to_midi(freq: float) -> float:
        return 69 + 12 * np.log2(float(freq) / 440.0)

    @staticmethod
    def _lab_time_to_seconds(lab_time: int) -> float:
        return float(lab_time) / 10000000.0

    @staticmethod
    def _is_silence_label(label: str) -> bool:
        return label.strip().lower() in {"pau", "sil", "sp", "br", "", "-", "spn"}

    @staticmethod
    def _safe_int(value) -> int:
        try:
            return int(round(float(value)))
        except Exception:
            return 0

    def _read_audio_duration_sec(self, wav_path: str) -> float:
        try:
            info = sf.info(wav_path)
            if info.samplerate <= 0:
                return 0.0
            return float(info.frames) / float(info.samplerate)
        except Exception:
            return 0.0

    def _load_lab_segments(self, lab_path: str) -> List[LabelSegment]:
        segments: List[LabelSegment] = []
        with open(lab_path, "r", encoding="utf-8", errors="replace") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split()
                if len(parts) < 3:
                    continue

                try:
                    start_time = int(float(parts[0]))
                    end_time = int(float(parts[1]))
                except Exception:
                    continue

                label = " ".join(parts[2:]).strip()
                segments.append(LabelSegment(start_time=start_time, end_time=end_time, label=label))

        return segments

    def _default_note(self, lyric: str, tone: int) -> Dict:
        return {
            "musicalType": "singing",
            "onset": 0,
            "duration": 0,
            "lyrics": lyric,
            "phonemes": "",
            "accent": "",
            "pitch": int(tone),
            "detune": 0,
            "instantMode": False,
            "attributes": {"evenSyllableDuration": True},
            "systemAttributes": {"evenSyllableDuration": True},
            "pitchTakes": {"activeTakeId": 0, "takes": [{"id": 0, "expr": 0, "liked": False}]},
            "timbreTakes": {"activeTakeId": 0, "takes": [{"id": 0, "expr": 0, "liked": False}]},
        }

    # ----------------------------
    # F0 提取
    # ----------------------------
    def process_audio_f0(self, audio_path: str, config: AudioProcessingConfig) -> Dict:
        """
        使用 PyWORLD 提取音频的 F0 基频曲线
        """
        try:
            if pw is None:
                logger.error("PyWORLD 未安装，无法提取 F0 基频。")
                return {"success": False, "error": "PyWORLD not installed"}

            x, sr = sf.read(audio_path)
            if len(x.shape) > 1:
                x = np.mean(x, axis=1)
            x = x.astype(np.float64)

            if config.f0_method.lower() == "harvest":
                f0, t = pw.harvest(x, sr, f0_floor=config.f0_floor, f0_ceil=config.f0_ceil)
            else:
                f0, t = pw.dio(x, sr, f0_floor=config.f0_floor, f0_ceil=config.f0_ceil)
                f0 = pw.stonemask(x, f0, t, sr)

            if config.f0_smooth and config.f0_smooth_window > 1 and len(f0) > 0:
                window = int(config.f0_smooth_window)
                if window % 2 == 0:
                    window += 1
                window = max(3, window)

                voiced_mask = f0 > 0
                padded = np.pad(f0, window // 2, mode="edge")
                smoothed = np.convolve(padded, np.ones(window) / window, mode="valid")
                f0 = np.where(voiced_mask, smoothed, 0.0)

            return {
                "success": True,
                "f0": f0,
                "t": t,
                "sr": sr,
            }
        except Exception as e:
            logger.error(f"F0 基频提取过程中发生异常: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    # ----------------------------
    # SVP 生成
    # ----------------------------
    def _build_svp_project_text(
        self,
        title: str,
        segments: List[LabelSegment],
        f0: Optional[np.ndarray],
        t: Optional[np.ndarray],
        sr: Optional[int],
        wav_path: str,
        config: AudioProcessingConfig,
        audio_duration_sec: Optional[float] = None,
    ) -> str:
        """
        生成可被 Synthesizer V 正确识别的 SVP JSON 文本

        关键点：
        1. 音符必须放进 tracks[0].mainGroup.notes
        2. tracks[0].mainRef.groupID 必须和 mainGroup.uuid 对应
        3. pitch 曲线写入 tracks[0].mainGroup.parameters.pitchDelta.points
        """
        if audio_duration_sec is None:
            audio_duration_sec = 0.0

        bpm = float(config.bpm)
        blicks_per_sec = (bpm / 60.0) * 705600000  # 1 quarter = 705600000 blicks

        notes: List[Dict] = []
        pitch_data: List[float] = []

        for seg in segments:
            if self._is_silence_label(seg.label):
                continue

            start_sec = self._lab_time_to_seconds(seg.start_time)
            end_sec = self._lab_time_to_seconds(seg.end_time)
            if end_sec <= start_sec:
                continue

            onset = int(round(start_sec * blicks_per_sec))
            end_onset = int(round(end_sec * blicks_per_sec))
            duration = max(1, end_onset - onset)

            tone = int(config.base_pitch)
            if config.refine_pitch and t is not None and f0 is not None and len(f0) > 0:
                mask = (t >= start_sec) & (t < end_sec)
                seg_f0 = f0[mask]
                voiced = seg_f0[seg_f0 > 0]
                if len(voiced) > 0:
                    avg_f0 = float(np.mean(voiced))
                    tone = int(round(self._freq_to_midi(avg_f0)))
                    tone = max(12, min(127, tone))

            note = self._default_note(seg.label, tone)
            note["onset"] = onset
            note["duration"] = duration
            notes.append(note)

        # pitchDelta points：按 SV 的 blick 坐标写入
        if t is not None and f0 is not None and len(f0) > 0:
            for ti, f0i in zip(t, f0):
                if f0i <= 0:
                    continue

                pos = int(round(float(ti) * blicks_per_sec))

                nominal_pitch = int(config.base_pitch)
                for note in notes:
                    if note["onset"] <= pos <= (note["onset"] + note["duration"]):
                        nominal_pitch = int(note["pitch"])
                        break

                pitch_midi = self._freq_to_midi(float(f0i))
                deviation_cents = int(round((pitch_midi - nominal_pitch) * 100))

                pitch_data.append(pos)
                pitch_data.append(deviation_cents)

        import uuid

        group_uuid = str(uuid.uuid4()).lower()

        main_group = {
            "uuid": group_uuid,
            "name": title,
            "comment": "",
            "trackNo": 0,
            "notes": notes,
            "parameters": {
                "pitchDelta": {
                    "definition": "pitchDelta",
                    "points": pitch_data,
                }
            },
        }

        project = {
            "version": 7,
            "time": {
                "bpm": bpm,
                "meter": [{"index": 0, "numerator": 4, "denominator": 4}],
                "tempo": [{"position": 0, "bpm": bpm}],
            },
            "library": [],
            "tracks": [
                {
                    "name": title,
                    "mainGroup": main_group,
                    "mainRef": {
                        "groupID": group_uuid,
                        "voice": {
                            "voter": "",
                            "language": "",
                            "name": "",
                            "dF0Vbr": 0,
                        },
                        "audio": {
                            "relativeFilename": str(Path(wav_path).name),
                            "duration": float(audio_duration_sec),
                        },
                    },
                    "refs": [],
                }
            ],
            "renderConfig": {},
        }

        return json.dumps(project, ensure_ascii=False, indent=2)

    # ----------------------------
    # USTX 生成
    # ----------------------------
    def _build_utau_project_text(
        self,
        title: str,
        segments: List[LabelSegment],
        f0: Optional[np.ndarray],
        t: Optional[np.ndarray],
        sr: Optional[int],
        wav_path: str,
        config: AudioProcessingConfig,
        audio_duration_sec: Optional[float] = None,
    ) -> str:
        """
        生成 USTX 工程文件内容
        """
        if audio_duration_sec is None:
            audio_duration_sec = 0.0

        from ruamel.yaml import YAML
        from ruamel.yaml.comments import CommentedMap

        resolution = 480
        ticks_per_sec = (float(config.bpm) / 60.0) * resolution

        def make_default_pitch():
            return {
                "data": [
                    {"x": -1, "y": 0, "shape": "io"},
                    {"x": 1, "y": 0, "shape": "io"},
                ],
                "snap_first": True,
            }

        def make_default_vibrato():
            return {
                "length": 0,
                "period": 175,
                "depth": 25,
                "in": 10,
                "out": 10,
                "shift": 0,
                "drift": 0,
            }

        notes = []
        for seg in segments:
            if self._is_silence_label(seg.label):
                continue

            start_sec = self._lab_time_to_seconds(seg.start_time)
            end_sec = self._lab_time_to_seconds(seg.end_time)
            if end_sec <= start_sec:
                continue

            pos = int(round(start_sec * ticks_per_sec))
            end_pos = int(round(end_sec * ticks_per_sec))
            duration = max(1, end_pos - pos)

            tone = int(config.base_pitch)
            if config.refine_pitch and t is not None and f0 is not None and len(f0) > 0:
                mask = (t >= start_sec) & (t < end_sec)
                seg_f0 = f0[mask]
                voiced = seg_f0[seg_f0 > 0]
                if len(voiced) > 0:
                    avg_f0 = float(np.mean(voiced))
                    tone = int(round(self._freq_to_midi(avg_f0)))
                    tone = max(12, min(127, tone))

            notes.append({
                "position": pos,
                "duration": duration,
                "tone": tone,
                "lyric": seg.label,
                "pitch": make_default_pitch(),
                "vibrato": make_default_vibrato(),
                "phoneme_expressions": [],
                "phoneme_overrides": [],
            })

        xs = []
        ys = []
        if t is not None and f0 is not None:
            for ti, f0i in zip(t, f0):
                if f0i <= 0:
                    continue
                tick = int(round(float(ti) * ticks_per_sec))

                nominal_pitch = int(config.base_pitch)
                for note in notes:
                    if note["position"] <= tick <= (note["position"] + note["duration"]):
                        nominal_pitch = int(note["tone"])
                        break

                pitch_midi = self._freq_to_midi(float(f0i))
                deviation_cents = int(round((pitch_midi - nominal_pitch) * 100))

                xs.append(tick)
                ys.append(deviation_cents)

        part = CommentedMap({
            "name": title,
            "comment": "",
            "track_no": 0,
            "position": 0,
            "notes": notes,
            "curves": [
                {
                    "xs": xs,
                    "ys": ys,
                    "abbr": "pitd",
                }
            ],
        })
        part.yaml_set_tag("!UVoicePart")

        wave_part = CommentedMap({
            "name": "Reference",
            "track_no": 1,
            "position": 0,
            "relative_path": str(Path(wav_path).name),
            "file_duration_ms": float(audio_duration_sec * 1000.0),
        })
        wave_part.yaml_set_tag("!UWavePart")

        project = {
            "name": title,
            "comment": "",
            "output_dir": "Vocal",
            "cache_dir": "UCache",
            "ustx_version": "0.6",
            "resolution": resolution,
            "bpm": int(config.bpm),
            "beat_per_bar": 4,
            "beat_unit": 4,
            "expressions": self._get_default_expressions(),
            "time_signatures": [{"bar_position": 0, "beat_per_bar": 4, "beat_unit": 4}],
            "tempos": [{"position": 0, "bpm": int(config.bpm)}],
            "tracks": [
                {
                    "name": "Singing Track",
                    "comment": "",
                    "singer": "",
                    "phonemizer": "OpenUtau.Core.DefaultPhonemizer",
                    "renderer_settings": {"renderer": "", "resampler": "", "wavtool": ""},
                    "mute": False,
                    "solo": False,
                    "volume": 0,
                    "pan": 0,
                },
                {
                    "name": "Audio Track",
                    "comment": "",
                    "singer": "",
                    "phonemizer": "OpenUtau.Core.DefaultPhonemizer",
                    "renderer_settings": {"renderer": "", "resampler": "", "wavtool": ""},
                    "mute": False,
                    "solo": False,
                    "volume": 0,
                    "pan": 0,
                },
            ],
            "parts": [part, wave_part],
        }

        yaml = YAML()
        yaml.default_flow_style = False
        yaml.allow_unicode = True
        yaml.sort_base_mapping_type_on_output = False

        stream = StringIO()
        yaml.dump(project, stream)
        return stream.getvalue()

    def _get_default_expressions(self) -> Dict:
        return {
            "dyn": {"name": "dynamics (curve)", "abbr": "dyn", "type": "Curve", "min": -240, "max": 120, "default_value": 0, "is_flag": False, "flag": ""},
            "pitd": {"name": "pitch deviation (curve)", "abbr": "pitd", "type": "Curve", "min": -1200, "max": 1200, "default_value": 0, "is_flag": False, "flag": ""},
            "clr": {"name": "voice color", "abbr": "clr", "type": "Options", "min": 0, "max": -1, "default_value": 0, "is_flag": False, "options": []},
            "eng": {"name": "resampler engine", "abbr": "eng", "type": "Options", "min": 0, "max": 1, "default_value": 0, "is_flag": False, "options": ["", "worldline"]},
            "vel": {"name": "velocity", "abbr": "vel", "type": "Numerical", "min": 0, "max": 200, "default_value": 100, "is_flag": False, "flag": ""},
            "vol": {"name": "volume", "abbr": "vol", "type": "Numerical", "min": 0, "max": 200, "default_value": 100, "is_flag": False, "flag": ""},
            "atk": {"name": "attack", "abbr": "atk", "type": "Numerical", "min": 0, "max": 200, "default_value": 100, "is_flag": False, "flag": ""},
            "dec": {"name": "decay", "abbr": "dec", "type": "Numerical", "min": 0, "max": 100, "default_value": 0, "is_flag": False, "flag": ""},
            "gen": {"name": "gender", "abbr": "gen", "type": "Numerical", "min": -100, "max": 100, "default_value": 0, "is_flag": True, "flag": "g"},
            "genc": {"name": "gender (curve)", "abbr": "genc", "type": "Curve", "min": -100, "max": 100, "default_value": 0, "is_flag": False, "flag": ""},
            "bre": {"name": "breath", "abbr": "bre", "type": "Numerical", "min": 0, "max": 100, "default_value": 0, "is_flag": True, "flag": "B"},
            "brec": {"name": "breathiness (curve)", "abbr": "brec", "type": "Curve", "min": -100, "max": 100, "default_value": 0, "is_flag": False, "flag": ""},
            "lpf": {"name": "lowpass", "abbr": "lpf", "type": "Numerical", "min": 0, "max": 100, "default_value": 0, "is_flag": True, "flag": "H"},
            "mod": {"name": "modulation", "abbr": "mod", "type": "Numerical", "min": 0, "max": 100, "default_value": 0, "is_flag": False, "flag": ""},
            "alt": {"name": "alternate", "abbr": "alt", "type": "Numerical", "min": 0, "max": 16, "default_value": 0, "is_flag": False, "flag": ""},
            "shft": {"name": "tone shift", "abbr": "shft", "type": "Numerical", "min": -36, "max": 36, "default_value": 0, "is_flag": False, "flag": ""},
            "shfc": {"name": "tone shift (curve)", "abbr": "shfc", "type": "Curve", "min": -1200, "max": 1200, "default_value": 0, "is_flag": False, "flag": ""},
        }

    # ----------------------------
    # 完整流程入口
    # ----------------------------
    def process_full_pipeline(
        self,
        wav_path: str,
        lab_path: str,
        output_format: str = "sv",
        project_title: str = "Project",
        config: Optional[AudioProcessingConfig] = None,
        audio_f0_data: Optional[Dict] = None,
    ) -> Dict:
        """
        完整工程文件生成入口：
        读取 WAV + LAB，生成 SVP / USTX 文件
        """
        try:
            config = config or AudioProcessingConfig()

            wav_path = str(wav_path)
            lab_path = str(lab_path)

            if not Path(wav_path).exists():
                return {"success": False, "error": f"WAV 文件不存在: {wav_path}"}
            if not Path(lab_path).exists():
                return {"success": False, "error": f"LAB 文件不存在: {lab_path}"}

            segments = self._load_lab_segments(lab_path)
            audio_duration_sec = self._read_audio_duration_sec(wav_path)

            # F0 数据可由上游传入；未传入时保持为空
            f0 = None
            t = None
            sr = None
            if audio_f0_data and audio_f0_data.get("success"):
                f0 = audio_f0_data.get("f0")
                t = audio_f0_data.get("t")
                sr = audio_f0_data.get("sr")

            fmt = self._normalize_output_format(output_format)

            if fmt == "sv":
                project_text = self._build_svp_project_text(
                    title=project_title,
                    segments=segments,
                    f0=f0,
                    t=t,
                    sr=sr,
                    wav_path=wav_path,
                    config=config,
                    audio_duration_sec=audio_duration_sec,
                )
                out_path = self.work_dir / f"{Path(wav_path).stem}.svp"

            elif fmt == "ustx":
                project_text = self._build_utau_project_text(
                    title=project_title,
                    segments=segments,
                    f0=f0,
                    t=t,
                    sr=sr,
                    wav_path=wav_path,
                    config=config,
                    audio_duration_sec=audio_duration_sec,
                )
                out_path = self.work_dir / f"{Path(wav_path).stem}.ustx"

            else:
                return {
                    "success": False,
                    "error": f"不支持的格式: {output_format}。支持: sv / ustx / utau",
                }

            out_path.write_text(project_text, encoding="utf-8")

            return {
                "success": True,
                "output_path": str(out_path),
                "segments": len(segments),
                "format": fmt,
                "title": project_title,
            }

        except Exception as e:
            logger.error(f"工程文件生成失败: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
