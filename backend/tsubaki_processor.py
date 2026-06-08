# -*- coding: utf-8 -*-
"""
音频标注和音高处理模块 - 工程文件生成引擎
用于处理 LAB 标注文件和音频数据，生成 Synthesizer V / OpenUtau 工程文件
"""
from __future__ import annotations

import json
import logging
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import medfilt, savgol_filter
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
    f0_smooth_window: int = 11   # 11 frames × 5ms = 55ms — matches human pitch perception

    # 是否细化音高（控制 LAB 音符音高，是否用 F0 中位音高决定 tone）
    refine_pitch: bool = False

    # 是否将 F0 曲线写入工程文件（False = 纯净音符模式，不写 pitchDelta/pitd 曲线）
    export_pitch_line: bool = True

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
            "export_pitch_line": self.export_pitch_line,
            "use_double_precision": self.use_double_precision,
        }


# ---------------------------------------------------------------------------
# 100 nanosecond / blick 换算常量
# Synthesizer V 的 blick 单位即 100 纳秒（绝对时间，与 BPM 无关）
# 即: blick 值 = LAB 时间戳（100ns 单位）
# ---------------------------------------------------------------------------
_BLICKS_PER_SECOND = 10_000_000      # 1 秒 = 10,000,000 blicks (100ns)
_TICKS_PER_SECOND_DEFAULT = 480.0    # USTX default resolution per quarter note


class TsubakiProcessor:
    """
    工程文件生成器。

    SVP 正确结构（参照 Synthesizer V 实际格式 / koharu-label 参考文件）：
    ─────────────────────────────────────────────────────────────────────
    • library[0]
        uuid   = group_uuid
        notes  = 全部音符（包含从 LAB 读入的 '-' consonant 音符 +
                  填充 'sil/sp' 间隙的 '-' 音符）
        parameters.pitchDelta.mode  = "cubic"
        parameters.pitchDelta.points = [pos, float_cents, ...]   ← 浮点

    • tracks[0]
        mainGroup  = {uuid=main_uuid, notes=[], parameters=<全空 cubic 曲线>}
        mainRef    = {groupID=main_uuid, ...}        ← 指向空 mainGroup
        groups[0]  = {groupID=group_uuid, ...}       ← 指向 library[0]

    USTX 修复要点：
    ─────────────────────────────────────────────────────────────────────
    • ruamel.yaml ≥ 0.18 已弃用 yaml_set_tag()，改用：
        part.yaml_set_ctag(Tag(suffix="!UVoicePart"))
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

    # 真正的静音标签：跳过（不生成音符），但用 '-' 填充其占用的时间段
    _TRUE_SILENCE = {"pau", "sil", "sp", "spn", "br", "silence", "noise", "ap", "blank"}

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
        return float(lab_time) / 10_000_000.0

    @staticmethod
    def _is_true_silence(label: str) -> bool:
        """判断是否为真正的静音段（会被 '-' 填充，不生成可见音符）。
        注意：'-' 本身不是静音，它是 consonant phoneme onset 标记，应保留为音符。
        """
        return label.strip().lower() in TsubakiProcessor._TRUE_SILENCE

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

    def _default_svp_note(self, lyric: str, tone: int, onset: int = 0, duration: int = 0) -> Dict:
        """生成 SVP 音符（使用 SV 标准属性格式，去除 3.x 独有字段）。"""
        return {
            "onset": onset,
            "duration": max(1, duration),
            "lyrics": lyric,
            "phonemes": "",
            "pitch": int(tone),
            "detune": 0,
            "attributes": {
                "tF0Offset": 0,
                "tF0Left":   0,
                "tF0Right":  0,
                "dF0Left":   0,
                "dF0Right":  0,
                "dF0Vbr":    0,
            },
        }

    def _empty_svp_parameters(self) -> Dict:
        """生成全空 SVP 参数曲线（用于 mainGroup）。"""
        empty = {"mode": "cubic", "points": []}
        return {
            "pitchDelta":  dict(empty),
            "vibratoEnv":  dict(empty),
            "loudness":    dict(empty),
            "tension":     dict(empty),
            "breathiness": dict(empty),
            "voicing":     dict(empty),
            "gender":      dict(empty),
            "toneShift":   dict(empty),
        }

    def _group_ref(self, group_uuid: str) -> Dict:
        """生成 group 引用结构（mainRef / groups 元素通用）。"""
        return {
            "groupID":        group_uuid,
            "blickOffset":    0,
            "pitchOffset":    0,
            "isInstrumental": False,
            "database": {
                "name":        "",
                "language":    "",
                "phoneset":    "",
                "backendType": "",
            },
            "dictionary": "",
            "voice":      {},
        }

    # ----------------------------
    # F0 后处理工具
    # ----------------------------
    @staticmethod
    def _median_filter_1d(arr: np.ndarray, kernel_size: int) -> np.ndarray:
        """纯 numpy 1D 中值滤波（无需 scipy），使用步进技巧。"""
        n = len(arr)
        if n == 0:
            return arr
        k = max(1, int(kernel_size))
        half = k // 2
        padded = np.pad(arr, half, mode="edge")
        strides = (padded.strides[0], padded.strides[0])
        windows = np.lib.stride_tricks.as_strided(
            padded, shape=(n, k), strides=strides
        )
        return np.median(windows, axis=1)

    def _contiguous_runs(self, mask: np.ndarray) -> List[Tuple[int, int]]:
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return []
        splits = np.where(np.diff(idx) > 1)[0] + 1
        groups = np.split(idx, splits)
        return [(int(g[0]), int(g[-1]) + 1) for g in groups if len(g) > 0]

    def _soft_reject_spikes(
        self,
        f0: np.ndarray,
        suspicious: np.ndarray,
        max_jump_semitones: float = 3.0,
        local_win: int = 5,
    ) -> np.ndarray:
        out = np.array(f0, dtype=np.float64, copy=True)
        voiced = out > 0
        if voiced.sum() < 3:
            return out

        midi = np.full(out.shape, np.nan, dtype=np.float64)
        midi[voiced] = 69.0 + 12.0 * np.log2(np.maximum(out[voiced], 1e-12) / 440.0)

        runs = self._contiguous_runs(voiced)
        for s, e in runs:
            seg = midi[s:e]
            valid = np.isfinite(seg)
            if valid.sum() < 3:
                continue

            local_med = float(np.nanmedian(seg[valid]))
            for i in range(s, e):
                if not suspicious[i] or not np.isfinite(midi[i]):
                    continue
                if abs(midi[i] - local_med) >= max_jump_semitones:
                    out[i] = 0.0

        return out

    def _post_process_f0(self, f0: np.ndarray, config: AudioProcessingConfig) -> np.ndarray:
        """
        分段平滑版：
        1. 清理非法值
        2. 去掉孤立尖峰
        3. 只补短空隙
        4. 只对连续有声段做平滑
        """
        f0 = np.asarray(f0, dtype=np.float64).copy()
        if f0.size == 0:
            return f0

        f0[~np.isfinite(f0)] = 0.0
        f0[(f0 < config.f0_floor * 0.6) | (f0 > config.f0_ceil * 1.15)] = 0.0

        # 先补很短的断裂，避免被切碎
        voiced = f0 > 0
        runs = self._contiguous_runs(voiced)
        for (s1, e1), (s2, e2) in zip(runs, runs[1:]):
            gap = s2 - e1
            if 1 <= gap <= 3 and f0[e1 - 1] > 0 and f0[s2] > 0:
                f0[e1:s2] = np.linspace(f0[e1 - 1], f0[s2], gap + 2)[1:-1]

        voiced = f0 > 0
        runs = self._contiguous_runs(voiced)
        out = np.zeros_like(f0)

        for s, e in runs:
            seg = np.array(f0[s:e], dtype=np.float64, copy=True)
            n = len(seg)

            if n < 3:
                out[s:e] = np.clip(seg, config.f0_floor, config.f0_ceil)
                continue

            # 先轻微中值滤波，压掉单帧毛刺
            if n >= 5:
                seg = medfilt(seg, kernel_size=3)

            if config.f0_smooth and n >= 5:
                win = int(config.f0_smooth_window)
                win = max(5, min(win, n if n % 2 == 1 else n - 1))
                if win % 2 == 0:
                    win -= 1

                poly = 3 if win >= 7 else 2
                poly = min(poly, win - 1)

                try:
                    seg = savgol_filter(seg, win, poly)
                except Exception as e:
                    logger.warning(f"Savitzky-Golay 平滑失败，退回中值结果: {e}")

            out[s:e] = np.clip(seg, config.f0_floor, config.f0_ceil)

        return out

    # ----------------------------
    # F0 提取
    # ----------------------------
    def process_audio_f0(self, audio_path: str, config: AudioProcessingConfig) -> Dict:
        """
        稳定版 F0 提取：
        - 保留 dio/harvest + stonemask
        - d4c 只做软提示，不做硬门槛
        - 后处理改成分段平滑
        """
        try:
            if pw is None:
                logger.error("PyWORLD 未安装，无法提取 F0 基频。")
                return {"success": False, "error": "PyWORLD not installed"}

            logger.info(f"正在读取音频文件: {audio_path}")
            x, sr = sf.read(audio_path)

            if len(x.shape) > 1:
                x = np.mean(x, axis=1)

            x = np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            x = np.ascontiguousarray(x, dtype=np.float64)

            total_samples = len(x)
            duration_sec = total_samples / sr
            logger.info(f"✓ 音频加载完成，总长度: {duration_sec:.2f} 秒 ({duration_sec/60:.2f} 分钟)")

            frame_period = 5.0
            if duration_sec > 600:
                frame_period = 20.0
            if duration_sec > 3600:
                frame_period = 40.0

            logger.info(f"当前设定的提取帧移 (frame_period) 为: {frame_period} ms")

            def _extract_block(x_block: np.ndarray):
                if config.f0_method.lower() == "harvest":
                    f0_b, t_b = pw.harvest(
                        x_block, sr,
                        f0_floor=config.f0_floor,
                        f0_ceil=config.f0_ceil,
                        frame_period=frame_period
                    )
                else:
                    f0_b, t_b = pw.dio(
                        x_block, sr,
                        f0_floor=config.f0_floor,
                        f0_ceil=config.f0_ceil,
                        frame_period=frame_period
                    )
                    f0_b = pw.stonemask(x_block, f0_b, t_b, sr)

                f0_b = np.asarray(f0_b, dtype=np.float64)
                f0_b[~np.isfinite(f0_b)] = 0.0
                f0_b[(f0_b < config.f0_floor * 0.6) | (f0_b > config.f0_ceil * 1.15)] = 0.0
                return f0_b, t_b

            chunk_length_sec = 300
            chunk_size = int(chunk_length_sec * sr)

            if total_samples <= chunk_size:
                f0, t = _extract_block(x)

                # d4c 只做软拒绝
                try:
                    ap = pw.d4c(x, f0, t, sr)
                    ap0 = np.asarray(ap[:, 0], dtype=np.float64)
                    suspicious = (ap0 > 0.82) & (f0 > 0)
                    f0 = self._soft_reject_spikes(f0, suspicious, max_jump_semitones=3.0, local_win=5)
                except Exception as e:
                    logger.warning(f"d4c 检查失败，继续使用纯 F0 后处理: {e}")

            else:
                logger.info("检测到超长音频，启动分块流式处理...")
                f0_list = []
                t_list = []

                for start_idx in range(0, total_samples, chunk_size):
                    end_idx = min(start_idx + chunk_size, total_samples)
                    x_chunk = x[start_idx:end_idx]

                    if len(x_chunk) < 256:
                        continue

                    current_chunk = start_idx // chunk_size + 1
                    total_chunks = (total_samples + chunk_size - 1) // chunk_size
                    logger.info(f" -> 正在提取音高曲线: 第 {current_chunk}/{total_chunks} 分块...")

                    f0_c, t_c = _extract_block(x_chunk)

                    try:
                        ap_c = pw.d4c(x_chunk, f0_c, t_c, sr)
                        ap0_c = np.asarray(ap_c[:, 0], dtype=np.float64)
                        suspicious_c = (ap0_c > 0.82) & (f0_c > 0)
                        f0_c = self._soft_reject_spikes(f0_c, suspicious_c, max_jump_semitones=3.0, local_win=5)
                    except Exception as e:
                        logger.warning(f"分块 d4c 检查失败，继续处理: {e}")

                    t_c = t_c + (start_idx / sr)
                    f0_list.append(f0_c)
                    t_list.append(t_c)

                if not f0_list:
                    return {"success": False, "error": "未提取到任何有效 F0 分块"}

                f0 = np.concatenate(f0_list)
                t = np.concatenate(t_list)
                logger.info("✓ 所有分块音高提取完毕，已成功无缝连接全局时间轴。")

            f0 = self._post_process_f0(f0, config)
            return {"success": True, "f0": f0, "t": t, "sr": sr}

        except Exception as e:
            logger.error(f"✗ F0 基频提取发生严重异常: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    # ----------------------------
    # SVP 生成（完整修复版）
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
        生成可被 Synthesizer V 正确识别的 SVP JSON 文本。

        修复要点
        ────────
        1. blicks = LAB 时间戳（100ns 单位）不可使用，即 blicks_per_second = 10,000,000。
           需要使用 (bpm/60)*705600000 换算，那个公式会让时间缩小 141 倍。

        2. 音符放入 library[0].notes，而非 mainGroup.notes：
           • '-' (consonant onset) 保留为实际音符（不跳过）
           • 'sil/pau/sp/spn' 真正静音段 → 用 '-' 填充其时间范围
           • 相邻有声音符之间如有剩余间隙 → 再补 '-' 填充

        3. mainGroup 保持空 notes；tracks[0].groups[0] 引用 library[0]。

        4. pitchDelta 格式：mode="cubic"，y 值使用 float cents（不取整）。

        5. time 结构去除 version 7 格式的顶层 bpm 字段，改为 version 119 格式。
        """
        if audio_duration_sec is None:
            audio_duration_sec = 0.0

        import uuid

        base_tone = int(config.base_pitch)

        # ── 步骤 1：从 LAB segments 构建时间轴音符列表 ───────────────────────
        # 规则：
        #   '-'                → 保留（consonant onset 标记）
        #   sil / pau / sp     → 转换为 '-' dash 音符（填充间隙，使时间轴连续）
        #   其他（hai, da ...） → 保留

        offset_ratio = config.bpm * 705600000 / 60 
        
        all_notes: List[Dict] = []
        voiced_refs: List[Dict] = [] 

        for seg in segments:
            if seg.end_time <= seg.start_time:
                continue

            # 后续的重构公式保持不变
            onset = int(seg.start_time * offset_ratio / 10000000)
            end_offset = int(seg.end_time * offset_ratio / 10000000)
            dur = max(1, end_offset - onset)

            # 【修改点】如果是 lab 里的显式静音标签（sil/pau/sp），直接跳过不生成音符，使其在 SVP 中保持物理空白
            if self._is_true_silence(seg.label):
                continue

            # 下面不再需要 else，原本 else 内部的代码直接靠左对齐正常运行即可
            tone = base_tone
            if config.refine_pitch and t is not None and f0 is not None and len(f0) > 0:
                start_sec = self._lab_time_to_seconds(seg.start_time)
                end_sec   = self._lab_time_to_seconds(seg.end_time)
                mask      = (t >= start_sec) & (t < end_sec)
                seg_f0    = f0[mask]
                voiced    = seg_f0[seg_f0 > 0]
                if len(voiced) > 0:
                    # 在 MIDI（半音）空间取中位数，避免 Hz 域平均偏向高频
                    midi_vals = 69.0 + 12.0 * np.log2(voiced / 440.0)
                    tone      = int(round(float(np.median(midi_vals))))
                    tone      = max(12, min(127, tone))

            # 原本就有的 "-" 标签由于不属于 _is_true_silence，会顺利走到这里，被正确保留为 "-" 音符
            note = self._default_svp_note(seg.label, tone, onset, dur)
            all_notes.append(note)
            if seg.label != "-":
                voiced_refs.append(note)

        # ── 步骤 2：检查并填补音符之间的空隙 ────────────────────────────────
        # （正常情况下 LAB 已连续，此步作为保险）
        all_notes.sort(key=lambda n: n["onset"])

        # ── 步骤 3：pitchDelta points（float cents，cubic mode）──────────────
        pitch_data: List[float] = []

        if config.export_pitch_line and t is not None and f0 is not None and len(f0) > 0:
            for ti, f0i in zip(t, f0):
                if f0i <= 0: # 或者是对齐 JS 的 f0i < 55
                    continue
                # 3. 音频时间（秒）转换公式重构
                pos = int(float(ti) * offset_ratio)

                nominal_tone = base_tone
                for vn in voiced_refs:
                    if vn["onset"] <= pos <= (vn["onset"] + vn["duration"]):
                        nominal_tone = int(vn["pitch"])
                        break

                pitch_midi       = self._freq_to_midi(float(f0i))
                deviation_cents  = float((pitch_midi - nominal_tone) * 100)

                pitch_data.append(pos)
                pitch_data.append(deviation_cents)

        # ── 步骤 4：组装 JSON ─────────────────────────────────────────────────
        main_uuid  = str(uuid.uuid4()).lower()
        group_uuid = str(uuid.uuid4()).lower()

        library_group = {
            "name": title,
            "uuid": group_uuid,
            "parameters": {
                "pitchDelta": {
                    "mode":   "cubic",
                    "points": pitch_data,
                },
                "vibratoEnv":  {"mode": "cubic", "points": []},
                "loudness":    {"mode": "cubic", "points": []},
                "tension":     {"mode": "cubic", "points": []},
                "breathiness": {"mode": "cubic", "points": []},
                "voicing":     {"mode": "cubic", "points": []},
                "gender":      {"mode": "cubic", "points": []},
                "toneShift":   {"mode": "cubic", "points": []},
            },
            "notes": all_notes,
        }

        main_group = {
            "name":       "main",
            "uuid":       main_uuid,
            "parameters": self._empty_svp_parameters(),
            "notes":      [],
        }

        project = {
            "version": 119,
            "time": {
                "meter": [{"index": 0, "numerator": 4, "denominator": 4}],
                "tempo": [{"position": 0, "bpm": float(config.bpm)}],
            },
            "library": [library_group],
            "tracks": [
                {
                    "name":          title,
                    "dispColor":     "ff7db235",
                    "dispOrder":     0,
                    "renderEnabled": False,
                    "mixer": {
                        "gainDecibel": 0,
                        "pan":         0,
                        "mute":        False,
                        "solo":        False,
                        "display":     True,
                    },
                    "mainGroup": main_group,
                    "mainRef":   self._group_ref(main_uuid),
                    "groups":    [self._group_ref(group_uuid)],
                }
            ],
            "renderConfig": {
                "destination":     "./",
                "filename":        title,
                "numChannels":     1,
                "aspirationFormat":"noAspiration",
                "bitDepth":        16,
                "sampleRate":      48000,
                "exportMixDown":   True,
            },
        }

        return json.dumps(project, ensure_ascii=False, indent=2)

    # ----------------------------
    # USTX 生成（修复版）
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
        生成纯 MIDI 音轨的 USTX 文件内容（不包含原音频伴奏轨）。
        """
        from ruamel.yaml import YAML
        from io import StringIO

        resolution    = 480
        ticks_per_sec = (float(config.bpm) / 60.0) * resolution

        # ── 工具函数 ─────────────────────────────────────────────────────────
        def make_default_pitch():
            return {
                "data": [
                    {"x": -1, "y": 0, "shape": "io"},
                    {"x":  1, "y": 0, "shape": "io"},
                ],
                "snap_first": True,
            }

        def make_default_vibrato():
            return {
                "length": 0, "period": 175, "depth": 25,
                "in": 10, "out": 10, "shift": 0, "drift": 0,
            }

        # ── 1. 音符收集 ──────────────────────────────────────────────────────
        notes = []
        for seg in segments:
            if seg.end_time <= seg.start_time:
                continue
            label = seg.label.strip()
            if not label or self._is_true_silence(label):
                continue

            start_sec = self._lab_time_to_seconds(seg.start_time)
            end_sec   = self._lab_time_to_seconds(seg.end_time)
            if end_sec <= start_sec:
                continue

            pos      = int(round(start_sec * ticks_per_sec))
            end_pos  = int(round(end_sec   * ticks_per_sec))
            dur_tick = max(1, end_pos - pos)

            tone = int(config.base_pitch)
            if config.refine_pitch and f0 is not None and t is not None:
                mask   = (t >= start_sec) & (t < end_sec)
                seg_f0 = f0[mask]
                voiced = seg_f0[seg_f0 > 0]
                if len(voiced) > 0:
                    # 在 MIDI（半音）空间取中位数，避免 Hz 域平均偏向高频
                    midi_vals = 69.0 + 12.0 * np.log2(voiced / 440.0)
                    tone      = int(round(float(np.median(midi_vals))))
                    tone      = max(0, min(127, tone))

            notes.append({
                "position": pos,
                "duration": dur_tick,
                "tone":     tone,
                "lyric":    label,
                "pitch":    make_default_pitch(),
                "vibrato":  make_default_vibrato(),
                "phoneme_expressions": [],
                "phoneme_overrides":   [],
            })

        voice_duration = max(
            (n["position"] + n["duration"] for n in notes),
            default=0,
        )

        # ── 2. Pitch 偏移曲线 ────────────────────────────────────────────────
        xs, ys = [], []
        if config.export_pitch_line and t is not None and f0 is not None:
            for ti, f0i in zip(t, f0):
                if f0i <= 0:
                    continue
                tick          = int(round(ti * ticks_per_sec))
                nominal_pitch = int(config.base_pitch)
                for note in notes:
                    if note["position"] <= tick <= (note["position"] + note["duration"]):
                        nominal_pitch = note["tone"]
                        break
                deviation = int(round((self._freq_to_midi(float(f0i)) - nominal_pitch) * 100))
                xs.append(tick)
                ys.append(deviation)

        # ── 3. 轨道配置（仅保留单轨，对应 track_no: 0） ──────────────────────────
        def _make_track():
            return {
                "phonemizer":        "OpenUtau.Core.DefaultPhonemizer",
                "renderer_settings": {},
                "mute":   False,
                "solo":   False,
                "volume": 0,
            }

        tracks = [_make_track()]

        # ── 4. Voice Part 配置 ──────────────────────────────────────────────
        voice_part = {
            "name":     title,
            "comment":  "",
            "duration": voice_duration,
            "track_no": 0,
            "position": 0,
            "notes":    notes,
            "curves":   [{"xs": xs, "ys": ys, "abbr": "pitd"}],
        }

        # ── 5. 顶层项目结构（wave_parts 固定为空） ───────────────────────────────
        project_obj = {
            "name":        title,
            "comment":     "",
            "output_dir":  "Vocal",
            "cache_dir":   "UCache",
            "ustx_version": "0.6",
            "resolution":  resolution,
            "bpm":         float(config.bpm),
            "beat_per_bar": 4,
            "beat_unit":    4,
            "time_signatures": [{"bar_position": 0, "beat_per_bar": 4, "beat_unit": 4}],
            "tempos":          [{"position": 0, "bpm": float(config.bpm)}],
            "expressions":      self._get_default_expressions(),
            "tracks":          tracks,       
            "voice_parts":     [voice_part], 
            "wave_parts":      [],           # 彻底移除音频轨
        }

        yaml = YAML()
        yaml.default_flow_style               = False
        yaml.allow_unicode                  = True
        yaml.sort_base_mapping_type_on_output = False

        stream = StringIO()
        yaml.dump(project_obj, stream)
        return stream.getvalue()

    def _get_default_expressions(self) -> Dict:
            """补齐了样本文件中定义的所有弯曲控制和核心包络表达式"""
            return {
                "dyn":  {"name": "dynamics (curve)",        "abbr": "dyn",  "type": "Curve",     "min": -240,  "max": 120,  "default_value": 0,   "is_flag": False, "flag": ""},
                "pitd": {"name": "pitch deviation (curve)", "abbr": "pitd", "type": "Curve",     "min": -1200, "max": 1200, "default_value": 0,   "is_flag": False, "flag": ""},
                "clr":  {"name": "voice color",             "abbr": "clr",  "type": "Options",   "min": 0,     "max": -1,   "default_value": 0,   "is_flag": False, "options": []},
                "eng":  {"name": "resampler engine",        "abbr": "eng",  "type": "Options",   "min": 0,     "max": 1,    "default_value": 0,   "is_flag": False, "options": ["", "worldline"]},
                "vel":  {"name": "velocity",                "abbr": "vel",  "type": "Numerical", "min": 0,     "max": 200,  "default_value": 100, "is_flag": False, "flag": ""},
                "vol":  {"name": "volume",                  "abbr": "vol",  "type": "Numerical", "min": 0,     "max": 200,  "default_value": 100, "is_flag": False, "flag": ""},
                "atk":  {"name": "attack",                  "abbr": "atk",  "type": "Numerical", "min": 0,     "max": 200,  "default_value": 100, "is_flag": False, "flag": ""},
                "dec":  {"name": "decay",                   "abbr": "dec",  "type": "Numerical", "min": 0,     "max": 100,  "default_value": 0,   "is_flag": False, "flag": ""},
                "gen":  {"name": "gender",                  "abbr": "gen",  "type": "Numerical", "min": -100,  "max": 100,  "default_value": 0,   "is_flag": True,  "flag": "g"},
                "genc": {"name": "gender (curve)",          "abbr": "genc", "type": "Curve",     "min": -100,  "max": 100,  "default_value": 0,   "is_flag": False, "flag": ""},
                "bre":  {"name": "breath",                  "abbr": "bre",  "type": "Numerical", "min": 0,     "max": 100,  "default_value": 0,   "is_flag": True,  "flag": "B"},
                "brec": {"name": "breathiness (curve)",     "abbr": "brec", "type": "Curve",     "min": -100,  "max": 100,  "default_value": 0,   "is_flag": False, "flag": ""},
                "lpf":  {"name": "lowpass",                 "abbr": "lpf",  "type": "Numerical", "min": 0,     "max": 100,  "default_value": 0,   "is_flag": True,  "flag": "H"},
                "mod":  {"name": "modulation",              "abbr": "mod",  "type": "Numerical", "min": 0,     "max": 100,  "default_value": 0,   "is_flag": False, "flag": ""},
                "alt":  {"name": "alternate",               "abbr": "alt",  "type": "Numerical", "min": 0,     "max": 16,   "default_value": 0,   "is_flag": False, "flag": ""},
                "shft": {"name": "tone shift",              "abbr": "shft", "type": "Numerical", "min": -36,   "max": 36,   "default_value": 0,   "is_flag": False, "flag": ""},
                "shfc": {"name": "tone shift (curve)",      "abbr": "shfc", "type": "Curve",     "min": -1200, "max": 1200, "default_value": 0,   "is_flag": False, "flag": ""},
                "tenc": {"name": "tension (curve)",         "abbr": "tenc", "type": "Curve",     "min": -100,  "max": 100,  "default_value": 0,   "is_flag": False, "flag": ""},
                "voic": {"name": "voicing (curve)",         "abbr": "voic", "type": "Curve",     "min": 0,     "max": 100,  "default_value": 100, "is_flag": False, "flag": ""},
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
        """完整工程文件生成入口：读取 WAV + LAB，生成 SVP / USTX 文件。"""
        try:
            config   = config or AudioProcessingConfig()
            wav_path = str(wav_path)
            lab_path = str(lab_path)

            if not Path(wav_path).exists():
                return {"success": False, "error": f"WAV 文件不存在: {wav_path}"}
            if not Path(lab_path).exists():
                return {"success": False, "error": f"LAB 文件不存在: {lab_path}"}

            segments          = self._load_lab_segments(lab_path)
            audio_duration_sec = self._read_audio_duration_sec(wav_path)

            f0, t, sr = None, None, None
            if audio_f0_data and audio_f0_data.get("success"):
                f0 = audio_f0_data.get("f0")
                t  = audio_f0_data.get("t")
                sr = audio_f0_data.get("sr")

            fmt = self._normalize_output_format(output_format)

            if fmt == "sv":
                project_text = self._build_svp_project_text(
                    title=project_title, segments=segments,
                    f0=f0, t=t, sr=sr, wav_path=wav_path,
                    config=config, audio_duration_sec=audio_duration_sec,
                )
                out_path = self.work_dir / f"{Path(wav_path).stem}.svp"

            elif fmt == "ustx":
                project_text = self._build_utau_project_text(
                    title=project_title, segments=segments,
                    f0=f0, t=t, sr=sr, wav_path=wav_path,
                    config=config, audio_duration_sec=audio_duration_sec,
                )
                out_path = self.work_dir / f"{Path(wav_path).stem}.ustx"

            else:
                return {
                    "success": False,
                    "error": f"不支持的格式: {output_format}。支持: sv / ustx / utau",
                }

            out_path.write_text(project_text, encoding="utf-8")

            return {
                "success":     True,
                "output_path": str(out_path),
                "segments":    len(segments),
                "format":      fmt,
                "title":       project_title,
            }

        except Exception as e:
            logger.error(f"工程文件生成失败: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
