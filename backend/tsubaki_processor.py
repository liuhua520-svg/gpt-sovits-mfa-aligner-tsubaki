# -*- coding: utf-8 -*-
"""
音频标注和音高处理模块 - 工程文件生成引擎
用于处理 LAB 标注文件和音频数据，生成 Synthesizer V / OpenUtau 工程文件
"""
from __future__ import annotations

import json
import logging
import numpy as np
import multiprocessing as mp
import traceback
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
    f0_method: str = "dio"  # 'dio' / 'harvest' / 'crepe' / 'rmvpe'

    # CREPE / RMVPE 运行设备："auto"（自动选择 cuda，否则 cpu）/ "cpu" / "cuda"
    f0_device: str = "auto"

    # CREPE 模型规格："full"（精度高）或 "tiny"（速度快）
    crepe_model: str = "full"

    # F0 细化参数
    f0_smooth: bool = True
    f0_smooth_window: int = 11   # 11 frames × 5ms = 55ms

    # 是否细化音高（控制 LAB 音符音高，是否用 F0 中位音高决定 tone）
    refine_pitch: bool = False

    # 是否将 F0 曲线写入工程文件
    export_pitch_line: bool = True

    use_double_precision: bool = False

    # 新增：默认关闭 d4c 硬筛，避免把有效 F0 删空
    enable_ap_check: bool = False

    def to_dict(self) -> Dict:
        return {
            "bpm": self.bpm,
            "base_pitch": self.base_pitch,
            "f0_floor": self.f0_floor,
            "f0_ceil": self.f0_ceil,
            "f0_method": self.f0_method,
            "f0_device": self.f0_device,
            "crepe_model": self.crepe_model,
            "f0_smooth": self.f0_smooth,
            "f0_smooth_window": self.f0_smooth_window,
            "refine_pitch": self.refine_pitch,
            "export_pitch_line": self.export_pitch_line,
            "use_double_precision": self.use_double_precision,
            "enable_ap_check": self.enable_ap_check,
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

    SVP 正确结构（参照 Synthesizer V 实际格式）：
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

    def process_full(self, *args, **kwargs):
        return self.process_full_pipeline(*args, **kwargs)

    # ----------------------------
    # F0 后处理工具
    # ----------------------------
    @staticmethod
    def _median_filter_1d(arr: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """纯 numpy 实现的一维中值滤波"""
        if len(arr) < kernel_size:
            return arr
        pad_size = kernel_size // 2
        padded = np.pad(arr, (pad_size, pad_size), mode='edge')
        windows = np.lib.stride_tricks.sliding_window_view(padded, kernel_size)
        return np.median(windows, axis=1)

    @staticmethod
    def _contiguous_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
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

            half = max(1, local_win // 2)
            for i in range(s, e):
                if not suspicious[i] or not np.isfinite(midi[i]):
                    continue

                left = max(s, i - half)
                right = min(e, i + half + 1)
                win = midi[left:right]
                win = win[np.isfinite(win)]
                if win.size < 3:
                    continue

                local_med = float(np.nanmedian(win))
                if abs(midi[i] - local_med) >= max_jump_semitones:
                    out[i] = 0.0

        return out


    def _post_process_f0(self, f0: np.ndarray, config: AudioProcessingConfig) -> np.ndarray:
            """六步 F0 后处理流水线 (对数域、桥接、平滑)"""
            f0_clean = f0.copy()
            
            # Step 1: 消除高频假阳性 (擦音尖峰)
            ceiling_threshold = config.f0_ceil * 0.92
            f0_clean[f0_clean > ceiling_threshold] = 0.0
            
            # Step 2: 移除极短的孤立有声段 (单帧爆破音噪声)
            voiced_mask = (f0_clean > 0).astype(float)
            voiced_mask = self._median_filter_1d(voiced_mask, 3)
            f0_clean[voiced_mask == 0] = 0.0

            if not config.f0_smooth:
                return f0_clean

            # Step 3: 转换到 Log 域 (半音域) 避免 Hz 域计算导致高频异常拉高均值
            f0_log = np.zeros_like(f0_clean)
            voiced_idx = f0_clean > 0
            f0_log[voiced_idx] = np.log2(f0_clean[voiced_idx])

            # Step 4: 去毛刺 (中值滤波)
            if config.f0_smooth_window >= 3:
                f0_log[voiced_idx] = self._median_filter_1d(f0_log[voiced_idx], 3)

            # Step 5: 对短促的无声间隙进行线性插值 (桥接)
            max_gap = config.f0_smooth_window * 2
            is_zero = f0_log == 0
            
            # 寻找连续 0 区间的起始和结束索引
            diffs = np.diff(np.concatenate(([0], is_zero.view(np.int8), [0])))
            starts = np.where(diffs == 1)[0]
            ends = np.where(diffs == -1)[0]
            
            for s, e in zip(starts, ends):
                length = e - s
                # 仅桥接首尾都接触到有声段的短暂无声区
                if length <= max_gap and s > 0 and e < len(f0_log):
                    start_val = f0_log[s - 1]
                    end_val = f0_log[e]
                    f0_log[s:e] = np.linspace(start_val, end_val, length + 2)[1:-1]

            # Step 6: 移动平均平滑 (均值滤波) - 仅在有声区域及其桥接带内平滑
            window = config.f0_smooth_window
            if window > 0:
                kernel = np.ones(window) / window
                smoothed_log = np.zeros_like(f0_log)
                active_mask = f0_log > 0
                
                if np.any(active_mask):
                    padded = np.pad(f0_log, (window//2, window//2), mode='edge')
                    smoothed_log_full = np.convolve(padded, kernel, mode="valid")
                    # 只有现在有效（即非0）的位置才覆盖回去，防止拉拽0点
                    smoothed_log[active_mask] = smoothed_log_full[active_mask]
                
                f0_log = smoothed_log

            # 还原到 Hz 域
            f0_final = np.zeros_like(f0_clean)
            final_voiced = f0_log > 0
            f0_final[final_voiced] = np.exp2(f0_log[final_voiced])
            
            return f0_final

    # ----------------------------
    # F0 提取
    # ----------------------------
    # 替换 process_audio_f0
    # ----------------------------
    # F0 后处理工具
    # ----------------------------
    @staticmethod
    def _median_filter_1d(arr: np.ndarray, kernel_size: int) -> np.ndarray:
        """纯 numpy 1D 中值滤波（无需 scipy）。"""
        arr = np.asarray(arr, dtype=np.float64)
        if arr.size == 0:
            return arr.copy()

        k = int(kernel_size)
        if k <= 1:
            return arr.copy()
        if k % 2 == 0:
            k += 1

        pad = k // 2
        padded = np.pad(arr, pad, mode="edge")

        try:
            windows = np.lib.stride_tricks.sliding_window_view(padded, k)
        except AttributeError:
            strides = (padded.strides[0], padded.strides[0])
            windows = np.lib.stride_tricks.as_strided(
                padded,
                shape=(arr.size, k),
                strides=strides,
            )

        return np.median(windows, axis=-1)

    @staticmethod
    def _moving_average_1d(arr: np.ndarray, kernel_size: int) -> np.ndarray:
        """纯 numpy 1D 均值平滑。"""
        arr = np.asarray(arr, dtype=np.float64)
        if arr.size == 0:
            return arr.copy()

        k = int(kernel_size)
        if k <= 1:
            return arr.copy()
        if k % 2 == 0:
            k += 1

        pad = k // 2
        padded = np.pad(arr, (pad, pad), mode="edge")
        kernel = np.ones(k, dtype=np.float64) / float(k)
        return np.convolve(padded, kernel, mode="valid")

    @staticmethod
    def _contiguous_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
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
        """只处理疑似擦音假峰，不扩大影响范围。"""
        out = np.array(f0, dtype=np.float64, copy=True)
        voiced = out > 0
        if voiced.sum() < 3:
            return out

        midi = np.full(out.shape, np.nan, dtype=np.float64)
        midi[voiced] = 69.0 + 12.0 * np.log2(np.maximum(out[voiced], 1e-12) / 440.0)

        runs = self._contiguous_runs(voiced)
        for s, e in runs:
            half = max(1, local_win // 2)
            for i in range(s, e):
                if not suspicious[i] or not np.isfinite(midi[i]):
                    continue

                left = max(s, i - half)
                right = min(e, i + half + 1)
                win = midi[left:right]
                win = win[np.isfinite(win)]
                if win.size < 3:
                    continue

                local_med = float(np.nanmedian(win))
                if abs(midi[i] - local_med) >= max_jump_semitones:
                    out[i] = 0.0

        return out

    def _post_process_f0(self, f0: np.ndarray, config: AudioProcessingConfig) -> np.ndarray:
        """
        后处理：
        - f0_smooth=False: 直接原样返回，不做平滑
        - f0_smooth=True : 先去尖峰，再做 log2 域平滑
        """
        f0 = np.asarray(f0, dtype=np.float64).copy()
        if f0.size == 0:
            return f0

        # 关闭平滑时：尽量保留原样
        if not config.f0_smooth:
            return f0

        # 只在开启平滑时做清理与平滑
        f0[~np.isfinite(f0)] = 0.0
        f0[(f0 < config.f0_floor * 0.6) | (f0 > config.f0_ceil * 1.15)] = 0.0

        voiced = f0 > 0
        runs = self._contiguous_runs(voiced)

        # 只补很短的断裂
        for (s1, e1), (s2, e2) in zip(runs, runs[1:]):
            gap = s2 - e1
            if 1 <= gap <= 3 and f0[e1 - 1] > 0 and f0[s2] > 0:
                left = np.log2(max(float(f0[e1 - 1]), 1e-12))
                right = np.log2(max(float(f0[s2]), 1e-12))
                bridge = np.linspace(left, right, gap + 2, dtype=np.float64)[1:-1]
                f0[e1:s2] = np.power(2.0, bridge)

        # 软去尖峰
        suspicious = (f0 >= config.f0_ceil * 0.92) & (f0 > 0)
        f0 = self._soft_reject_spikes(f0, suspicious, max_jump_semitones=3.0, local_win=5)

        voiced = f0 > 0
        runs = self._contiguous_runs(voiced)
        out = np.zeros_like(f0)

        for s, e in runs:
            seg = np.array(f0[s:e], dtype=np.float64, copy=True)
            n = len(seg)

            if n < 3:
                out[s:e] = np.clip(seg, config.f0_floor, config.f0_ceil)
                continue

            # 先中值滤波去毛刺
            if n >= 5:
                seg = self._median_filter_1d(seg, 3)

            # 再在 log2 域做平滑
            win = int(config.f0_smooth_window)
            win = max(5, win)
            if win % 2 == 0:
                win += 1
            if win > n:
                win = n if n % 2 == 1 else n - 1

            if win >= 5 and n >= win:
                log_seg = np.log2(np.maximum(seg, 1e-12))
                log_seg = self._moving_average_1d(log_seg, win)
                seg = np.power(2.0, log_seg)

            out[s:e] = np.clip(seg, config.f0_floor, config.f0_ceil)

        return out

    # ----------------------------
    # F0 提取
    # ----------------------------
    def process_audio_f0(self, audio_path: str, config: AudioProcessingConfig) -> Dict:
        """提取基频 F0。

        支持四种方法：
            - dio / harvest : PyWORLD（本地 DSP 算法，速度快，无额外依赖）
            - crepe         : torchcrepe 神经网络（抗噪，需要 torch + torchcrepe）
            - rmvpe         : RMVPE 深度模型（对人声极为鲁棒，需要 torch + 模型权重）

        无论使用哪种方法，统一返回 {"success": True, "f0": ndarray, "t": ndarray, "sr": int}，
        其中 f0 为 Hz（未发声 = 0），t 为对应时间戳（秒）。该结果可直接传入
        _build_svp_project_text / _build_utau_project_text 写入 pitchDelta / pitd 曲线。
        """
        import numpy as np
        import soundfile as sf

        method = (config.f0_method or "dio").strip().lower()

        try:
            x, sr = sf.read(audio_path)
            if x.ndim > 1:
                x = np.mean(x, axis=1)
            x = x.astype(np.float64)

            if method == "harvest":
                import pyworld as pw
                f0, t = pw.harvest(
                    x, sr,
                    f0_floor=config.f0_floor, f0_ceil=config.f0_ceil,
                    frame_period=5.0
                )

            elif method == "crepe":
                from f0_extractors import extract_f0_crepe
                f0, t = extract_f0_crepe(
                    x, sr,
                    f0_floor=config.f0_floor,
                    f0_ceil=config.f0_ceil,
                    model_size=config.crepe_model,
                    device=config.f0_device,
                )

            elif method == "rmvpe":
                from f0_extractors import extract_f0_rmvpe
                f0, t = extract_f0_rmvpe(
                    x, sr,
                    f0_floor=config.f0_floor,
                    f0_ceil=config.f0_ceil,
                    device=config.f0_device,
                )

            else:
                # 默认 / 'dio'
                import pyworld as pw
                _f0, t = pw.dio(
                    x, sr,
                    f0_floor=config.f0_floor, f0_ceil=config.f0_ceil,
                    frame_period=5.0
                )
                f0 = pw.stonemask(x, _f0, t, sr)

            # 统一类型
            f0 = np.asarray(f0, dtype=np.float64)
            t  = np.asarray(t,  dtype=np.float64)

            # 线性插值无声段（仅当存在足够的有声帧时）
            voiced_idx = np.nonzero(f0 > 0)[0]
            if len(voiced_idx) > 1:
                unvoiced_idx = np.where(f0 == 0)[0]
                if len(unvoiced_idx) > 0:
                    f0[unvoiced_idx] = np.interp(
                        unvoiced_idx, voiced_idx, f0[voiced_idx]
                    )

            # 可选平滑（移动平均）
            if config.f0_smooth and len(f0) > 2:
                win = int(config.f0_smooth_window)
                if win % 2 == 0:
                    win += 1
                if win > 1:
                    padded = np.pad(f0, win // 2, mode="edge")
                    f0 = np.convolve(padded, np.ones(win) / win, mode="valid")

            return {"success": True, "f0": f0, "t": t, "sr": sr, "method": method}

        except ImportError as e:
            logger.error(f"F0 提取失败（依赖缺失，method={method}）: {e}")
            return {
                "success": False,
                "error": f"F0 方法 '{method}' 所需依赖未安装: {e}",
            }
        except FileNotFoundError as e:
            logger.error(f"F0 提取失败（缺少模型权重，method={method}）: {e}")
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.error("F0 提取失败:" + str(e), exc_info=True)
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
        midi_notes: Optional[List] = None,   # MIDI 导入：(start_sec, end_sec, pitch) 列表
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
            if midi_notes is not None:
                # ── MIDI 模式：直接从 MIDI 文件读取该段的音符音高 ──
                from midi_processor import map_segment_to_midi_pitch
                _start_sec = self._lab_time_to_seconds(seg.start_time)
                _end_sec   = self._lab_time_to_seconds(seg.end_time)
                tone = map_segment_to_midi_pitch(_start_sec, _end_sec, midi_notes,
                                                 base_pitch=base_tone)
            elif config.refine_pitch and t is not None and f0 is not None and len(f0) > 0:
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
        voiced_sample_count = 0

        logger.info(
            f"Pitch Export Debug: export_pitch_line={config.export_pitch_line}, "
            f"f0_len={0 if f0 is None else len(f0)}, "
            f"t_len={0 if t is None else len(t)}"
        )

        if config.export_pitch_line and t is not None and f0 is not None and len(f0) > 0:
            for ti, f0i in zip(t, f0):
                if not np.isfinite(f0i) or f0i <= 0:
                    continue

                voiced_sample_count += 1

                # 音频时间（秒）→ SV blick 轴
                pos = int(float(ti) * offset_ratio)

                nominal_tone = base_tone
                for vn in voiced_refs:
                    if vn["onset"] <= pos <= (vn["onset"] + vn["duration"]):
                        nominal_tone = int(vn["pitch"])
                        break

                pitch_midi = self._freq_to_midi(float(f0i))
                deviation_cents = float((pitch_midi - nominal_tone) * 100)

                pitch_data.extend([pos, deviation_cents])

            logger.info(
                f"Pitch Export Debug: valid_f0={voiced_sample_count}, "
                f"generated_points={len(pitch_data)//2}"
            )

            if not pitch_data:
                logger.warning("pitchDelta 为空：导出前的 F0 基本被清空了，写入零线兜底。")
                for note in all_notes:
                    pitch_data.extend([note["onset"], 0.0])
                    pitch_data.extend([note["onset"] + note["duration"], 0.0])

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
        midi_notes: Optional[List] = None,   # MIDI 导入：(start_sec, end_sec, pitch) 列表
    ) -> str:
        """
        生成 USTX 工程文件内容。

        修复说明
        ────────
        Bug 1 — 字段错误：F0 数据原先写入每个 note 的 pitch.data（该字段
                是 OpenUtau 用于音符间过渡曲线的内部字段，不是音高偏移曲线），
                导致 PITD 曲线在 OpenUtau 中始终为空。
                正确做法：写入 voice_part["curves"] 列表，abbr = "pitd"。

        Bug 2 — 单位错误：原代码用 (midi_f0 - tone) × 1000，但 OpenUtau
                pitd 的单位是 cents（100 = 1 个半音），应为 × 100。
                原来的 × 1000 让所有偏差放大 10 倍，音高曲线完全错误。

        音符音高说明
        ────────────
        · refine_pitch=False：所有音符固定在 base_pitch，pitd 曲线存储
          相对于 base_pitch 的完整 F0 偏移（偏差较大，但保留真实音高走势）。
        · refine_pitch=True ：每个音符的 tone 设置为该段 F0 的中位数半音，
          pitd 曲线存储相对于该 tone 的细粒度偏移（与 SVP 的行为一致）。
        """
        from ruamel.yaml import YAML
        from io import StringIO

        resolution    = 480
        ticks_per_sec = (float(config.bpm) / 60.0) * resolution

        # ── 工具：默认 pitch 过渡曲线（两端锚点，不携带 F0 数据） ───────────
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
            if midi_notes is not None:
                # ── MIDI 模式：直接从 MIDI 文件读取该段的音符音高 ──
                from midi_processor import map_segment_to_midi_pitch
                tone = map_segment_to_midi_pitch(start_sec, end_sec, midi_notes,
                                                 base_pitch=int(config.base_pitch))
            elif config.refine_pitch and f0 is not None and t is not None:
                # refine_pitch：用该段 F0 中位数半音作为音符音高
                mask   = (t >= start_sec) & (t < end_sec)
                seg_f0 = f0[mask]
                voiced = seg_f0[seg_f0 > 0]
                if len(voiced) > 0:
                    midi_vals = 69.0 + 12.0 * np.log2(voiced / 440.0)
                    tone      = int(round(float(np.median(midi_vals))))
                    tone      = max(0, min(127, tone))

            notes.append({
                "position": pos,
                "duration": dur_tick,
                "tone":     tone,
                "lyric":    label,
                # pitch.data = 默认两端锚点，不存 F0（F0 走 voice_part curves）
                "pitch":    make_default_pitch(),
                "vibrato":  make_default_vibrato(),
                "phoneme_expressions": [],
                "phoneme_overrides":   [],
            })

        voice_duration = max(
            (n["position"] + n["duration"] for n in notes),
            default=0,
        )

        # ── 2. 构建全局 pitd 曲线 ────────────────────────────────────────────
        # OpenUtau pitd 单位：cents（100 = 1 个半音），范围 -1200 ~ +1200。
        # 数据放在 voice_part["curves"] 而非 note["pitch"]["data"]。
        xs: List[int] = []
        ys: List[int] = []

        if config.export_pitch_line and f0 is not None and t is not None and len(f0) > 0:
            # 预建 note tone 查询表：(start_tick, end_tick, tone) 按 start 排序
            note_ranges = np.array(
                [(n["position"], n["position"] + n["duration"], n["tone"])
                 for n in notes],
                dtype=np.int64,
            ) if notes else np.empty((0, 3), dtype=np.int64)

            note_starts = note_ranges[:, 0] if len(note_ranges) else np.array([], dtype=np.int64)
            note_ends   = note_ranges[:, 1] if len(note_ranges) else np.array([], dtype=np.int64)
            note_tones  = note_ranges[:, 2] if len(note_ranges) else np.array([], dtype=np.int64)

            for ti, f0i in zip(t, f0):
                if f0i <= 0.0:
                    continue
                tick    = int(round(float(ti) * ticks_per_sec))
                midi_f0 = 69.0 + 12.0 * np.log2(float(f0i) / 440.0)

                # 找该 tick 所属音符的 tone（二分查找，O(log n)）
                nominal = int(config.base_pitch)
                if len(note_starts):
                    idx = int(np.searchsorted(note_starts, tick, side="right")) - 1
                    if 0 <= idx < len(note_tones) and tick <= note_ends[idx]:
                        nominal = int(note_tones[idx])

                # ★ 单位修复：× 100（cents），不是 × 1000
                deviation = int(round((midi_f0 - nominal) * 100))
                xs.append(tick)
                ys.append(deviation)

        # ★ 字段修复：pitd 曲线放在 voice_part["curves"]，而非 note["pitch"]["data"]
        curves = [{"xs": xs, "ys": ys, "abbr": "pitd"}] if xs else []

        # ── 3. Track + VoicePart ─────────────────────────────────────────────
        track = {
            "phonemizer":        "OpenUtau.Core.DefaultPhonemizer",
            "renderer_settings": {},
            "mute":   False,
            "solo":   False,
            "volume": 0,
        }

        voice_part = {
            "name":     title,
            "comment":  "",
            "duration": voice_duration,
            "track_no": 0,
            "position": 0,
            "notes":    notes,
            "curves":   curves,   # ← pitd 写这里
        }

        # ── 4. 顶层项目结构 ──────────────────────────────────────────────────
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
            "tracks":           [track],
            "voice_parts":      [voice_part],
            "wave_parts":       [],
        }

        yaml = YAML()
        yaml.default_flow_style               = False
        yaml.allow_unicode                    = True
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
        phoneme_mode: str = "none",
        midi_path: Optional[str] = None,   # MIDI 文件路径（可选）
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

            # ── 音素转换模式 (project-only 专用) ──────────────────────────────
            # 在将 LAB 段落写入工程文件之前，可选地把独立的辅音+元音对合并为
            # 音节级标签：
            #   'none'     → 不转换，保持原始 LAB 标签（默认）
            #   'merge'    → 合并辅音，如 s + a → sa
            #   'hiragana' → 合并后转平假名，如 s + a → さ，N → ん
            #   'katakana' → 合并后转片假名，如 s + a → サ，N → ン
            if phoneme_mode and phoneme_mode != "none":
                try:
                    from phoneme_converter import apply_phoneme_mode
                    seg_tuples = [
                        (s.start_time, s.end_time, s.label) for s in segments
                    ]
                    converted = apply_phoneme_mode(seg_tuples, phoneme_mode)
                    segments = [
                        LabelSegment(start_time=t[0], end_time=t[1], label=t[2])
                        for t in converted
                    ]
                    logger.info(
                        f"音素转换完成 (mode={phoneme_mode}): "
                        f"{len(segments)} 个音节段落"
                    )
                except Exception as _pm_err:
                    logger.warning(
                        f"音素转换失败 (mode={phoneme_mode}): {_pm_err}，"
                        f"回退到原始音素"
                    )

            f0, t, sr = None, None, None
            if audio_f0_data and audio_f0_data.get("success"):
                f0 = audio_f0_data.get("f0")
                t  = audio_f0_data.get("t")
                sr = audio_f0_data.get("sr")

            # ── MIDI 导入：解析音符音高 + 覆盖 BPM ──────────────────────────
            midi_notes = None
            if midi_path and Path(midi_path).exists():
                try:
                    from dataclasses import replace as _dc_replace
                    from midi_processor import parse_midi_notes
                    midi_bpm, midi_notes = parse_midi_notes(midi_path)
                    if midi_bpm and midi_bpm > 0:
                        config = _dc_replace(config, bpm=float(midi_bpm))
                        logger.info(f"✓ MIDI BPM 已覆盖 config.bpm → {midi_bpm:.1f}")
                    logger.info(f"✓ MIDI 音符导入: {len(midi_notes)} 个音符")
                except Exception as _midi_err:
                    logger.warning(f"⚠ MIDI 解析失败，回退到默认模式: {_midi_err}")
                    midi_notes = None

            fmt = self._normalize_output_format(output_format)

            if fmt == "sv":
                project_text = self._build_svp_project_text(
                    title=project_title, segments=segments,
                    f0=f0, t=t, sr=sr, wav_path=wav_path,
                    config=config, audio_duration_sec=audio_duration_sec,
                    midi_notes=midi_notes,
                )
                out_path = self.work_dir / f"{Path(wav_path).stem}.svp"

            elif fmt == "ustx":
                project_text = self._build_utau_project_text(
                    title=project_title, segments=segments,
                    f0=f0, t=t, sr=sr, wav_path=wav_path,
                    config=config, audio_duration_sec=audio_duration_sec,
                    midi_notes=midi_notes,
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

def _f0_worker(audio_path: str, config_dict: dict, q: mp.Queue) -> None:
    """
    子进程里做 F0 提取。
    将所有危险的 C++ PyWORLD 运算（包括分块和内存连续化）隔离在独立进程中。
    即使底层发生了 Access Violation (0xC0000005)，主进程也不会闪退！
    """
    try:
        import numpy as np
        import soundfile as sf
        try:
            import pyworld as pw
        except ImportError:
            q.put({"success": False, "error": "PyWORLD 未安装"})
            return

        x, sr = sf.read(audio_path)
        if len(x.shape) > 1:
            x = np.mean(x, axis=1)

        # 终极防崩溃：彻底清洗内存，转化为 C语言连续的 float64 数组
        x = np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.ascontiguousarray(x, dtype=np.float64)

        total_samples = len(x)
        duration_sec = total_samples / sr

        frame_period = 5.0
        if duration_sec > 600:
            frame_period = 10.0
        if duration_sec > 3600:
            frame_period = 20.0

        chunk_length_sec = 300
        chunk_size = int(chunk_length_sec * sr)

        f0_method = config_dict.get("f0_method", "dio").lower()
        f0_floor = config_dict.get("f0_floor", 71.0)
        f0_ceil = config_dict.get("f0_ceil", 800.0)

        def _extract_block(x_block: np.ndarray):
            if f0_method == "harvest":
                f0_b, t_b = pw.harvest(x_block, sr, f0_floor=f0_floor, f0_ceil=f0_ceil, frame_period=frame_period)
            else:
                f0_b, t_b = pw.dio(x_block, sr, f0_floor=f0_floor, f0_ceil=f0_ceil, frame_period=frame_period)
                f0_b = pw.stonemask(x_block, f0_b, t_b, sr)
            
            f0_b = np.asarray(f0_b, dtype=np.float64)
            f0_b[~np.isfinite(f0_b)] = 0.0
            # 使用原生的阈值切除假阳性高频
            f0_b[(f0_b < f0_floor * 0.6) | (f0_b > f0_ceil * 0.95)] = 0.0
            return f0_b, t_b

        if total_samples <= chunk_size:
            f0, t = _extract_block(x)
        else:
            f0_list = []
            t_list = []
            for start_idx in range(0, total_samples, chunk_size):
                end_idx = min(start_idx + chunk_size, total_samples)
                x_chunk = x[start_idx:end_idx]
                if len(x_chunk) < 256: 
                    continue
                f0_c, t_c = _extract_block(x_chunk)
                t_c = t_c + (start_idx / sr)
                f0_list.append(f0_c)
                t_list.append(t_c)

            f0 = np.concatenate(f0_list)
            t = np.concatenate(t_list)

        # 仅将干净的数据送出队列，避免在主进程做危险计算
        q.put({"success": True, "f0": f0, "t": t, "sr": sr})

    except Exception as e:
        import traceback
        q.put({"success": False, "error": f"{e}\n{traceback.format_exc()}"})