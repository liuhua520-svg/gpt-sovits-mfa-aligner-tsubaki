# -*- coding: utf-8 -*-
"""
音频标注和音高处理模块 - 工程文件生成引擎
用于处理 LAB 标注文件和音频数据，生成 Synthesizer V / OpenUtau 工程文件
"""
from __future__ import annotations

import json
import uuid
import logging
import re
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




def _split_lyrics_to_words(text: str) -> List[str]:
    """
    将歌词文本拆分为按音符对应的字/词列表。

    规则：
    - CJK 汉字、平假名、片假名、韩文 → 每个字符单独一个元素
    - 单独的 "-" 保留为占位符
    - 拉丁字母等 → 以空白符为分隔符拆成单词
    - 空格、其他标点、换行跳过
    """
    import unicodedata

    result: List[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        cp = ord(ch)

        if (
            (0x4E00 <= cp <= 0x9FFF)
            or (0x3400 <= cp <= 0x4DBF)
            or (0x20000 <= cp <= 0x2A6DF)
            or (0x3040 <= cp <= 0x309F)
            or (0x30A0 <= cp <= 0x30FF)
            or (0xAC00 <= cp <= 0xD7A3)
            or (0x1100 <= cp <= 0x11FF)
        ):
            result.append(ch)
            i += 1
        elif ch in (' ', '\t', '\n', '\r'):
            i += 1
        elif ch == '-':
            result.append('-')
            i += 1
        elif unicodedata.category(ch).startswith('P'):
            i += 1
        else:
            j = i
            while j < len(text) and text[j] not in (' ', '\t', '\n', '\r'):
                j += 1
            word = text[i:j].strip()
            if word:
                result.append(word)
            i = j

    return result


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
        "sv":   "Synthesizer V Project (.svp)",
        "ustx": "OpenUtau Project (.ustx)",
        "utau": "OpenUtau Project (.ustx) [alias]",
        "midi": "MIDI 标准文件 (.mid)",
        "vsqx": "VOCALOID 4 Project (.vsqx)",
    }

    OUTPUT_ALIASES = {
        "svp":           "sv",
        "synthv":        "sv",
        "synthesizer_v": "sv",
        "synthesizerv":  "sv",
        "openutau":      "ustx",
        "utau":          "ustx",
        "mid":           "midi",
        "vsq4":          "vsqx",
        "vocaloid":      "vsqx",
        "vocaloid4":     "vsqx",
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

    # 匹配纯 ASCII 字母 + 撇号组成的英语单词（'t, I'm 等缩写）
    _ASCII_WORD_RE = re.compile(r"^[a-zA-Z][a-zA-Z']*$")

    @staticmethod
    def _is_ascii_word_label(label: str) -> bool:
        """
        判断 label 是否为可能的英语单词（非静音、非辅音占位符、纯 ASCII 字母）。

        用于 word_phoneme_map 开关：只对英语单词调用 word_to_arpabet()，
        跳过 '-' 占位符、CJK 字符段、pinyin/jyutping/hangul 音素段。

        注意：单纯的 ASCII 字母检查无法完全区分 "en 模式下的音素 hh/sh"
        和 "英语词 hello"——但 word_phoneme_map 只在 english_word_align=True
        时才启用，后者保证了 LAB 中的英语词以完整词（而非音素序列）形式存在，
        因此在实际场景中误判概率极低。
        """
        if not label or label == "-":
            return False
        if label.strip().lower() in TsubakiProcessor._TRUE_SILENCE:
            return False
        return bool(TsubakiProcessor._ASCII_WORD_RE.match(label))

    @staticmethod
    def _label_is_english_word(
        label: str,
        language: str,
        native_english_words: Optional[set] = None,
    ) -> bool:
        """
        判断 label 在给定语言上下文中是否应作为英语单词进行音素映射。

        逻辑
        ────
        · 语言为英语 (en/eng)：信任 _is_ascii_word_label，走完整 G2P 流程
          （MFA 词典 + g2p_en OOV 兜底），与旧行为完全一致。
        · 非英语语言（zh/ja/ko/yue/…）：
          a. 若 native_english_words 不为 None（由调用方从原始汉字/韩文文本
             预提取），则以该集合为唯一判断依据：
               - 原始文本里没有拉丁字母 → 集合为空 → 所有 label 均不命中
               - 混入了 "love/hello" 等真实英文词 → 只有这些词命中
             优势：彻底消除"拼音碰巧是英语词"的误判（rang/wang/dong/shi 等）。
          b. native_english_words 为 None（兼容旧调用路径，如 project-only 模式
             未传原始文本）：回退到词典查询（is_in_english_dict），保持向后兼容。
        · 词典文件缺失时，非英语语言一律返回 False（保守策略）。
        """
        lang = (language or "").lower().strip().rstrip("-")
        if lang in ("en", "eng", "english"):
            return True          # English: full G2P pipeline

        # ── 非英语语言 ────────────────────────────────────────────────────
        if native_english_words is not None:
            # 优先路径：用从原始文本预提取的集合判断（最准确）
            clean = (label or "").strip().lower()
            return bool(clean) and clean in native_english_words

        # 兼容回退：词典查询（旧行为）
        try:
            from phoneme_converter import is_in_english_dict
            return is_in_english_dict(label)
        except Exception:
            return False

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

    @staticmethod
    def _segments_from_midi_notes(
        midi_notes: List[Tuple[float, float, int]],
        label: str = "-",
        lyric_words: Optional[List[str]] = None,
    ) -> List[LabelSegment]:
        """
        将 MIDI 音符列表转换为 LabelSegment 列表。

        每个 MIDI 音符对应一个 LabelSegment。
        音高信息由调用处通过 midi_notes + map_segment_to_midi_pitch 获取，
        不需要编码在 label 内。

        Parameters
        ----------
        midi_notes : list of (start_sec, end_sec, pitch)
        label : str
            未提供 lyric_words 或超出长度时的兜底歌词，默认 '-'
        lyric_words : list of str, optional
            按音符顺序对应的歌词字/词列表。提供时逐一写入对应音符；
            列表比音符少时，超出部分使用 label 填充。
        """
        segments: List[LabelSegment] = []
        for i, (start_sec, end_sec, _pitch) in enumerate(midi_notes):
            start_100ns = int(round(float(start_sec) * 10_000_000))
            end_100ns   = int(round(float(end_sec)   * 10_000_000))
            if end_100ns > start_100ns:
                note_label = (
                    lyric_words[i]
                    if lyric_words and i < len(lyric_words)
                    else label
                )
                segments.append(LabelSegment(
                    start_time=start_100ns,
                    end_time=end_100ns,
                    label=note_label,
                ))
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
    def _build_svp_track_group(
        self,
        title: str,
        segments: List[LabelSegment],
        f0: Optional[np.ndarray],
        t: Optional[np.ndarray],
        config: AudioProcessingConfig,
        midi_notes: Optional[List] = None,
        word_phoneme_map: bool = False,
        language: str = "",
        native_english_words: Optional[set] = None,
        dict_source: str = "default",
        disp_order: int = 0,
        disp_color: str = "ff7db235",
    ) -> Tuple[Dict, Dict]:
        """
        构建单条 SVP 音轨所需的 (library_group, track) 结构对。

        这是从 _build_svp_project_text 中抽取出的"单轨"计算逻辑，供
        单轨（_build_svp_project_text）与多轨（_build_svp_multi_track_project_text，
        对话文本框批量处理用）两个上层函数共用，避免核心音符/音高/词典查询
        逻辑维护两份。抽取过程未改动任何计算细节，单轨调用方输出与抽取前
        完全一致。
        """
        base_tone = int(config.base_pitch)

        # ── 步骤 1：从 LAB segments 构建时间轴音符列表 ───────────────────────
        offset_ratio = config.bpm * 705600000 / 60

        all_notes: List[Dict] = []
        voiced_refs: List[Dict] = []

        for seg in segments:
            if seg.end_time <= seg.start_time:
                continue

            onset = int(seg.start_time * offset_ratio / 10000000)
            end_offset = int(seg.end_time * offset_ratio / 10000000)
            dur = max(1, end_offset - onset)

            if self._is_true_silence(seg.label):
                continue

            tone = base_tone
            if midi_notes is not None:
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
                    midi_vals = 69.0 + 12.0 * np.log2(voiced / 440.0)
                    tone      = int(round(float(np.median(midi_vals))))
                    tone      = max(12, min(127, tone))

            note = self._default_svp_note(seg.label, tone, onset, dur)

            if (
                word_phoneme_map
                and self._is_ascii_word_label(seg.label)
                and self._label_is_english_word(seg.label, language, native_english_words)
            ):
                try:
                    resolved_phones: Optional[List[str]] = None

                    if dict_source == "synthesizerv":
                        from dictionary_manager import lookup_word as _dict_lookup
                        user_phones_str = _dict_lookup(seg.label, dict_source)
                        if user_phones_str:
                            resolved_phones = user_phones_str.split()
                            logger.debug(
                                "[SVP word_phoneme_map] 用户词典命中 %r → %s",
                                seg.label, user_phones_str,
                            )

                    if resolved_phones is None:
                        from phoneme_converter import word_to_arpabet
                        resolved_phones = word_to_arpabet(seg.label)

                    if resolved_phones:
                        note["phonemes"] = " ".join(resolved_phones)
                        logger.debug(
                            "[SVP word_phoneme_map] %r → %s",
                            seg.label, note["phonemes"],
                        )
                except Exception as _wp_err:
                    logger.warning(
                        "[SVP word_phoneme_map] 转换失败 %r: %s", seg.label, _wp_err
                    )

            all_notes.append(note)
            if seg.label != "-":
                voiced_refs.append(note)

        all_notes.sort(key=lambda n: n["onset"])

        # ── 步骤 2：pitchDelta points（float cents，cubic mode）──────────────
        pitch_data: List[float] = []
        voiced_sample_count = 0

        if config.export_pitch_line and t is not None and f0 is not None and len(f0) > 0:
            for ti, f0i in zip(t, f0):
                if not np.isfinite(f0i) or f0i <= 0:
                    continue

                voiced_sample_count += 1
                pos = int(float(ti) * offset_ratio)

                nominal_tone = base_tone
                for vn in voiced_refs:
                    if vn["onset"] <= pos <= (vn["onset"] + vn["duration"]):
                        nominal_tone = int(vn["pitch"])
                        break

                pitch_midi = self._freq_to_midi(float(f0i))
                deviation_cents = float((pitch_midi - nominal_tone) * 100)

                pitch_data.extend([pos, deviation_cents])

            if not pitch_data:
                logger.warning("pitchDelta 为空：导出前的 F0 基本被清空了，写入零线兜底。")
                for note in all_notes:
                    pitch_data.extend([note["onset"], 0.0])
                    pitch_data.extend([note["onset"] + note["duration"], 0.0])

        # ── 步骤 3：组装 library_group / track ───────────────────────────────
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

        track = {
            "name":          title,
            "dispColor":     disp_color,
            "dispOrder":     disp_order,
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

        return library_group, track

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
        word_phoneme_map: bool = False,       # 英语单词 → ARPABET 音素写入 phonemes 字段
        language: str = "",                   # 语种（用于 word_phoneme_map 防误判）
        native_english_words: Optional[set] = None,  # 从原始文本预提取的英语单词集合（防拼音误判）
        dict_source: str = "default",         # 单词→音素词典来源："default"/"synthesizerv"/"vocaloid"
    ) -> str:
        """
        生成可被 Synthesizer V 正确识别的 SVP JSON 文本（单轨）。

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

        实际的单轨音符/音高计算已抽取到 _build_svp_track_group()，本函数只负责
        调用一次并组装成完整工程；多轨版本见 _build_svp_multi_track_project_text()。
        """
        if audio_duration_sec is None:
            audio_duration_sec = 0.0

        library_group, track = self._build_svp_track_group(
            title=title, segments=segments, f0=f0, t=t, config=config,
            midi_notes=midi_notes, word_phoneme_map=word_phoneme_map,
            language=language, native_english_words=native_english_words,
            dict_source=dict_source, disp_order=0,
        )

        project = {
            "version": 119,
            "time": {
                "meter": [{"index": 0, "numerator": 4, "denominator": 4}],
                "tempo": [{"position": 0, "bpm": float(config.bpm)}],
            },
            "library": [library_group],
            "tracks": [track],
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

    def _build_svp_multi_track_project_text(
        self,
        title: str,
        track_inputs: List[Dict],
        config: AudioProcessingConfig,
    ) -> str:
        """
        生成多轨 SVP 工程（对话文本框批量处理用）：每个 track_inputs 元素
        对应一个对话框，成为工程文件里的一条独立音轨，而不是分别导出
        多个工程文件。

        Parameters
        ----------
        track_inputs : list of dict，每个 dict 支持的键：
            name (str)                          — 音轨名称
            segments (List[LabelSegment])        — 必填
            f0 / t (Optional[np.ndarray])
            midi_notes (Optional[List])
            word_phoneme_map (bool)
            language (str)
            native_english_words (Optional[set])
            dict_source (str)

        说明：SVP 格式全局只有一条 tempo 曲线，所有音轨共用调用方传入的
        同一个 config.bpm；每条音轨的音符 onset 都从 0 开始（各轨道相当于
        彼此独立的时间轴起点一致的片段集合），不做音轨间的时间偏移。
        """
        library_list: List[Dict] = []
        track_list: List[Dict] = []

        palette = ["ff7db235", "ff4a90d9", "ffd9744a", "ff9b59b6", "ffe6b800", "ff3fbf8f"]

        for idx, item in enumerate(track_inputs):
            library_group, track = self._build_svp_track_group(
                title=item.get("name") or f"{title} #{idx + 1}",
                segments=item["segments"],
                f0=item.get("f0"),
                t=item.get("t"),
                config=config,
                midi_notes=item.get("midi_notes"),
                word_phoneme_map=item.get("word_phoneme_map", False),
                language=item.get("language", ""),
                native_english_words=item.get("native_english_words"),
                dict_source=item.get("dict_source", "default"),
                disp_order=idx,
                disp_color=palette[idx % len(palette)],
            )
            library_list.append(library_group)
            track_list.append(track)

        project = {
            "version": 119,
            "time": {
                "meter": [{"index": 0, "numerator": 4, "denominator": 4}],
                "tempo": [{"position": 0, "bpm": float(config.bpm)}],
            },
            "library": library_list,
            "tracks": track_list,
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
    def _build_ustx_track(
        self,
        title: str,
        segments: List[LabelSegment],
        f0: Optional[np.ndarray],
        t: Optional[np.ndarray],
        config: AudioProcessingConfig,
        midi_notes: Optional[List] = None,
        track_no: int = 0,
    ) -> Tuple[Dict, Dict]:
        """
        构建单条 USTX 音轨所需的 (track, voice_part) 结构对。

        从 _build_utau_project_text 中抽取出的"单轨"计算逻辑，供单轨
        （_build_utau_project_text）与多轨（_build_utau_multi_track_project_text，
        对话文本框批量处理用）共用。抽取过程未改动任何计算细节，单轨调用方
        输出与抽取前完全一致（track_no 固定为 0）。
        """
        resolution    = 480
        ticks_per_sec = (float(config.bpm) / 60.0) * resolution

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
                from midi_processor import map_segment_to_midi_pitch
                tone = map_segment_to_midi_pitch(start_sec, end_sec, midi_notes,
                                                 base_pitch=int(config.base_pitch))
            elif config.refine_pitch and f0 is not None and t is not None:
                mask   = (t >= start_sec) & (t < end_sec)
                seg_f0 = f0[mask]
                voiced = seg_f0[seg_f0 > 0]
                if len(voiced) > 0:
                    midi_vals = 69.0 + 12.0 * np.log2(voiced / 440.0)
                    tone      = int(round(float(np.median(midi_vals))))
                    tone      = max(0, min(127, tone))

            utau_lyric = "+" if label in ("ー", "-") else label

            notes.append({
                "position": pos,
                "duration": dur_tick,
                "tone":     tone,
                "lyric":    utau_lyric,
                "pitch":    make_default_pitch(),
                "vibrato":  make_default_vibrato(),
                "phoneme_expressions": [],
                "phoneme_overrides":   [],
            })

        voice_duration = max(
            (n["position"] + n["duration"] for n in notes),
            default=0,
        )

        # ── 2. pitd 曲线 ─────────────────────────────────────────────────────
        xs: List[int] = []
        ys: List[int] = []

        if config.export_pitch_line and f0 is not None and t is not None and len(f0) > 0:
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

                nominal = int(config.base_pitch)
                if len(note_starts):
                    idx = int(np.searchsorted(note_starts, tick, side="right")) - 1
                    if 0 <= idx < len(note_tones) and tick <= note_ends[idx]:
                        nominal = int(note_tones[idx])

                deviation = int(round((midi_f0 - nominal) * 100))
                xs.append(tick)
                ys.append(deviation)

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
            "track_no": track_no,
            "position": 0,
            "notes":    notes,
            "curves":   curves,
        }

        return track, voice_part

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
        生成 USTX 工程文件内容（单轨）。

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

        实际的单轨音符/音高计算已抽取到 _build_ustx_track()，本函数只负责
        调用一次（track_no=0）并组装成完整工程；多轨版本见
        _build_utau_multi_track_project_text()。
        """
        from ruamel.yaml import YAML
        from io import StringIO

        resolution = 480

        track, voice_part = self._build_ustx_track(
            title=title, segments=segments, f0=f0, t=t, config=config,
            midi_notes=midi_notes, track_no=0,
        )

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

    def _build_utau_multi_track_project_text(
        self,
        title: str,
        track_inputs: List[Dict],
        config: AudioProcessingConfig,
    ) -> str:
        """
        生成多轨 USTX 工程（对话文本框批量处理用）：每个 track_inputs 元素
        对应一个对话框，成为工程文件里的一条独立音轨 + voice_part，而不是
        分别导出多个工程文件。

        Parameters
        ----------
        track_inputs : list of dict，每个 dict 支持的键：
            name (str), segments (List[LabelSegment]),
            f0 / t (Optional[np.ndarray]), midi_notes (Optional[List])

        说明：与 SVP/VSQX 一样，USTX 全局只有一条 tempo，所有音轨共用
        调用方传入的同一个 config.bpm；每条音轨的音符 position 都从 0
        开始，不做音轨间的时间偏移。
        """
        from ruamel.yaml import YAML
        from io import StringIO

        resolution = 480

        tracks: List[Dict] = []
        voice_parts: List[Dict] = []

        for idx, item in enumerate(track_inputs):
            track, voice_part = self._build_ustx_track(
                title=item.get("name") or f"{title} #{idx + 1}",
                segments=item["segments"],
                f0=item.get("f0"),
                t=item.get("t"),
                config=config,
                midi_notes=item.get("midi_notes"),
                track_no=idx,
            )
            tracks.append(track)
            voice_parts.append(voice_part)

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
            "tracks":           tracks,
            "voice_parts":      voice_parts,
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
        lab_path: Optional[str] = None,   # 可选：提供 LAB 或 MIDI 之一即可
        output_format: str = "sv",
        project_title: str = "Project",
        config: Optional[AudioProcessingConfig] = None,
        audio_f0_data: Optional[Dict] = None,
        phoneme_mode: str = "none",
        midi_path: Optional[str] = None,   # MIDI 文件路径（可选）
        lyrics_text: str = "",             # 纯 MIDI 模式下的用户歌词原文
        vsqx_singer: str = "MIKU_V4_Chinese",       # VSQX 歌手名（由前端按语种/模式传入）
        vsqx_singer_id: str = "BNGE7CP7EMTRSNC3",  # VSQX 歌手 ID
        vsqx_singer_bs: int = 4,                    # VSQX 歌手 Bank Select（VOCALOID4 内部编号，系统相关）
        word_phoneme_map: bool = False,             # 英语单词 → 音素写入 SVP phonemes / VSQX <p lock="1">
        language: str = "",                         # 语种，传给 SVP/VSQX 构建器防误判
        original_text: str = "",                    # 原始歌词文本（pypinyin 转换前的汉字/韩文原文）
                                                    # 用于预提取真实英语单词集合，防止拼音被误判
        dict_source: str = "default",               # 单词→音素词典来源："default"/"synthesizerv"/"vocaloid"
    ) -> Dict:
        """完整工程文件生成入口。

        输入约束
        ────────
        • wav_path  —— 始终必须（F0 提取来源）
        • lab_path  —— 可选；提供时从 LAB 读取音素段落
        • midi_path —— 可选；提供时解析 BPM + 音符音高；
                        若同时缺少 lab_path，则从 MIDI 音符自动生成段落

        至少需要 lab_path 或 midi_path 其中一个。

        输出格式
        ────────
        sv / svp        → Synthesizer V .svp
        ustx / utau     → OpenUtau .ustx
        midi / mid      → MIDI 标准文件 .mid
        """
        try:
            config   = config or AudioProcessingConfig()
            wav_path = str(wav_path)

            # ── 预提取原始文本中的真实英语单词集合（word_phoneme_map 防误判）──
            # 必须在 pypinyin 等转换之前，从原始汉字/韩文文本提取。
            # 纯中文/韩文时集合为空，所有 label 均不会被当作英语词处理；
            # 混入真实英语时（如 "I love you 很多"），只有这些词命中。
            native_english_words: Optional[set] = None
            _lang_norm = (language or "").lower().strip().rstrip("-")
            if word_phoneme_map and _lang_norm not in ("en", "eng", "english") and original_text:
                try:
                    from phoneme_converter import extract_native_english_words
                    native_english_words = extract_native_english_words(original_text)
                    logger.info(
                        "[word_phoneme_map] 原始文本预提取英语单词 %d 个: %s",
                        len(native_english_words),
                        sorted(native_english_words)[:10] if native_english_words else "（无）",
                    )
                except Exception as _nee_err:
                    logger.warning("[word_phoneme_map] 预提取英语单词失败: %s，回退词典查询", _nee_err)
                    # native_english_words 保持 None → 兼容旧行为

            # ── 文件检查 ───────────────────────────────────────────────────────
            if not Path(wav_path).exists():
                return {"success": False, "error": f"WAV 文件不存在: {wav_path}"}

            lab_exists  = bool(lab_path  and Path(str(lab_path)).exists())
            midi_exists = bool(midi_path and Path(str(midi_path)).exists())

            if not lab_exists and not midi_exists:
                return {
                    "success": False,
                    "error": "需要 LAB 文件或 MIDI 文件（至少提供其中一个）",
                }

            audio_duration_sec = self._read_audio_duration_sec(wav_path)

            # ── ① MIDI 优先解析（BPM + 音符，供后续用） ───────────────────────
            midi_notes = None
            midi_lyric_words = []

            if midi_exists:
                try:
                    from dataclasses import replace as _dc_replace
                    from midi_processor import parse_midi_notes_with_lyrics

                    midi_bpm, midi_notes, midi_lyrics = parse_midi_notes_with_lyrics(midi_path)

                    if midi_bpm and midi_bpm > 0:
                        config = _dc_replace(config, bpm=float(midi_bpm))
                        logger.info(f"✓ MIDI BPM 已覆盖 config.bpm → {midi_bpm:.1f}")

                    logger.info(f"✓ MIDI 音符导入: {len(midi_notes)} 个")

                    midi_lyric_words = self._midi_lyrics_to_words(midi_lyrics)
                    logger.info(f"✓ MIDI 歌词导入: {len(midi_lyric_words)} 个词")

                except Exception as _midi_err:
                    logger.warning(f"⚠ MIDI 解析失败，回退到默认模式: {_midi_err}")
                    midi_notes = None
                    midi_lyric_words = []

            # ── ② 段落来源：优先 LAB，否则从 MIDI 音符生成 ──────────────────
            if lab_exists:
                segments = self._load_lab_segments(str(lab_path))

                # 如果 MIDI 里有歌词，就覆盖 LAB 的 label
                if midi_lyric_words:
                    segments = self._apply_midi_lyrics_to_segments(segments, midi_lyric_words)

                if phoneme_mode and phoneme_mode != "none":
                    try:
                        from phoneme_converter import apply_phoneme_mode
                        seg_tuples = [(s.start_time, s.end_time, s.label) for s in segments]
                        converted = apply_phoneme_mode(seg_tuples, phoneme_mode)
                        segments = [
                            LabelSegment(start_time=t[0], end_time=t[1], label=t[2])
                            for t in converted
                        ]
                        logger.info(f"音素转换完成 (mode={phoneme_mode}): {len(segments)} 个音节段落")
                    except Exception as _pm_err:
                        logger.warning(f"音素转换失败 (mode={phoneme_mode}): {_pm_err}，回退到原始音素")
            else:
                if not midi_notes:
                    return {"success": False, "error": "MIDI 文件解析失败，无法生成段落"}

                # 纯 MIDI 模式：优先用 MIDI 自带歌词；没有的话再用外部 lyrics_text
                lyric_words = midi_lyric_words
                if not lyric_words and lyrics_text and lyrics_text.strip():
                    lyric_words = _split_lyrics_to_words(lyrics_text)

                segments = self._segments_from_midi_notes(
                    midi_notes,
                    lyric_words=lyric_words
                )

                logger.info(
                    f"✓ 从 MIDI 生成段落: {len(segments)} 个"
                    f"（歌词字数: {len(lyric_words) if lyric_words else 0}）"
                )

            # ── ③ F0 数据 ─────────────────────────────────────────────────────
            f0, t, sr = None, None, None
            if audio_f0_data and audio_f0_data.get("success"):
                f0 = audio_f0_data.get("f0")
                t  = audio_f0_data.get("t")
                sr = audio_f0_data.get("sr")

            # ── ④ 生成输出文件 ───────────────────────────────────────────────
            fmt = self._normalize_output_format(output_format)

            if fmt == "sv":
                project_text = self._build_svp_project_text(
                    title=project_title, segments=segments,
                    f0=f0, t=t, sr=sr, wav_path=wav_path,
                    config=config, audio_duration_sec=audio_duration_sec,
                    midi_notes=midi_notes,
                    word_phoneme_map=word_phoneme_map,
                    language=language,
                    native_english_words=native_english_words,
                    dict_source=dict_source,
                )
                out_path = self.work_dir / f"{Path(wav_path).stem}.svp"
                out_path.write_text(project_text, encoding="utf-8")

            elif fmt == "ustx":
                project_text = self._build_utau_project_text(
                    title=project_title, segments=segments,
                    f0=f0, t=t, sr=sr, wav_path=wav_path,
                    config=config, audio_duration_sec=audio_duration_sec,
                    midi_notes=midi_notes,
                )
                out_path = self.work_dir / f"{Path(wav_path).stem}.ustx"
                out_path.write_text(project_text, encoding="utf-8")

            elif fmt == "midi":
                out_path = self.work_dir / f"{Path(wav_path).stem}.mid"
                self._build_midi_output(
                    output_path=str(out_path),
                    segments=segments,
                    config=config,
                    f0=f0, t=t,
                    midi_notes=midi_notes,
                )

            elif fmt == "vsqx":
                project_text = self._build_vsqx_project_text(
                    title=project_title, segments=segments,
                    f0=f0, t=t, sr=sr, wav_path=wav_path,
                    config=config, audio_duration_sec=audio_duration_sec,
                    midi_notes=midi_notes,
                    vsqx_singer=vsqx_singer,
                    vsqx_singer_id=vsqx_singer_id,
                    vsqx_singer_bs=vsqx_singer_bs,
                    word_phoneme_map=word_phoneme_map,
                    language=language,
                    native_english_words=native_english_words,
                    dict_source=dict_source,
                )
                out_path = self.work_dir / f"{Path(wav_path).stem}.vsqx"
                out_path.write_text(project_text, encoding="utf-8")

            else:
                return {
                    "success": False,
                    "error": (
                        f"不支持的格式: {output_format}。"
                        "支持: sv / ustx / vsqx"
                    ),
                }

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

    def _midi_lyrics_to_words(self, midi_lyrics) -> List[str]:
        """
        把 MIDI lyrics meta event 转成顺序词表。
        这里复用你现有的 _split_lyrics_to_words 逻辑。
        """
        words: List[str] = []
        if not midi_lyrics:
            return words

        for ev in midi_lyrics:
            txt = (ev.text or "").strip()
            if not txt:
                continue
            words.extend(_split_lyrics_to_words(txt))
        return words


    def _apply_midi_lyrics_to_segments(self, segments, lyric_words: List[str]):
        """
        用 MIDI 歌词覆盖段落 label。
        静音段只在歌词当前位置也是占位符 '-' 时才消费一个 token，
        这样可以避免把后面的词前移。
        """
        if not lyric_words:
            return segments

        silence_tokens = {"sil", "pau", "sp", "spn", "rest"}
        out = []
        idx = 0

        for seg in segments:
            label = (seg.label or "").strip()
            lower = label.lower()

            if lower in silence_tokens:
                if idx < len(lyric_words) and lyric_words[idx] == "-":
                    out.append(
                        LabelSegment(
                            start_time=seg.start_time,
                            end_time=seg.end_time,
                            label="-",
                        )
                    )
                    idx += 1
                else:
                    out.append(seg)
                continue

            if idx < len(lyric_words):
                new_label = lyric_words[idx]
                idx += 1
            else:
                new_label = label

            out.append(
                LabelSegment(
                    start_time=seg.start_time,
                    end_time=seg.end_time,
                    label=new_label,
                )
            )

        return out

    # ----------------------------
    # VSQX 生成
    # ----------------------------
    def _build_vsqx_track_xml(
        self,
        title: str,
        segments: List[LabelSegment],
        f0: Optional[np.ndarray],
        t: Optional[np.ndarray],
        config: AudioProcessingConfig,
        midi_notes: Optional[List] = None,
        vsqx_singer_bs: int = 4,
        word_phoneme_map: bool = False,
        language: str = "",
        native_english_words: Optional[set] = None,
        dict_source: str = "default",
        track_no: int = 0,
    ) -> str:
        """
        构建单条 VSQX 音轨（<vsTrack>...</vsTrack> 完整片段）。

        从 _build_vsqx_project_text 中抽取出的"单轨"计算逻辑，供单轨
        （_build_vsqx_project_text）与多轨（_build_vsqx_multi_track_project_text，
        对话文本框批量处理用）共用。抽取过程未改动任何计算细节，单轨调用方
        输出与抽取前完全一致（track_no 固定为 0）。
        """
        RESOLUTION  = 480
        PBS         = 12
        PART_OFFSET = 1920
        VELOCITY    = 64
        bpm = float(config.bpm)

        def sec_to_ticks(sec: float) -> int:
            return int(round(float(sec) * (bpm / 60.0) * RESOLUTION))

        base_tone = int(config.base_pitch)

        note_entries: List[Tuple[int, int, int, str, str]] = []
        voiced_refs:  List[Tuple[int, int, int]]           = []

        _word_to_arpabet   = None
        _arpabet_to_vocaloid4 = None
        if word_phoneme_map:
            try:
                from phoneme_converter import word_to_arpabet as _wta, arpabet_to_vocaloid4 as _atv4
                _word_to_arpabet      = _wta
                _arpabet_to_vocaloid4 = _atv4
            except Exception as _imp_err:
                logger.warning("[VSQX word_phoneme_map] 导入失败: %s，功能已禁用", _imp_err)

        for seg in segments:
            if seg.end_time <= seg.start_time:
                continue
            if self._is_true_silence(seg.label):
                continue

            start_sec = self._lab_time_to_seconds(seg.start_time)
            end_sec   = self._lab_time_to_seconds(seg.end_time)
            t_on  = sec_to_ticks(start_sec)
            t_off = max(t_on + 1, sec_to_ticks(end_sec))
            dur   = t_off - t_on

            tone = base_tone
            if midi_notes is not None:
                from midi_processor import map_segment_to_midi_pitch
                tone = map_segment_to_midi_pitch(
                    start_sec, end_sec, midi_notes, base_pitch=base_tone)
            elif config.refine_pitch and f0 is not None and t is not None and len(f0) > 0:
                mask  = (t >= start_sec) & (t < end_sec)
                seg_f0 = f0[mask]
                voiced = seg_f0[seg_f0 > 0]
                if len(voiced) > 0:
                    midi_vals = 69.0 + 12.0 * np.log2(voiced / 440.0)
                    tone      = int(round(float(np.median(midi_vals))))
                    tone      = max(12, min(127, tone))

            label = seg.label

            p_tag = '<p></p>'
            if (
                word_phoneme_map
                and self._is_ascii_word_label(label)
                and self._label_is_english_word(label, language, native_english_words)
            ):
                try:
                    v4_phones: Optional[str] = None

                    if dict_source in ("vocaloid", "synthesizerv"):
                        from dictionary_manager import lookup_word as _dict_lookup
                        user_phones_str = _dict_lookup(label, dict_source)
                        if user_phones_str:
                            if dict_source == "vocaloid":
                                v4_phones = user_phones_str
                            elif _arpabet_to_vocaloid4 is not None:
                                v4_phones = _arpabet_to_vocaloid4(user_phones_str.split())
                            if v4_phones:
                                logger.debug(
                                    "[VSQX word_phoneme_map] 用户词典命中(%s) %r → V4 %s",
                                    dict_source, label, v4_phones,
                                )

                    if v4_phones is None and _word_to_arpabet is not None:
                        arpabet_phones = _word_to_arpabet(label)
                        if arpabet_phones and _arpabet_to_vocaloid4 is not None:
                            v4_phones = _arpabet_to_vocaloid4(arpabet_phones)
                            logger.debug(
                                "[VSQX word_phoneme_map] %r → ARPABET %s → V4 %s",
                                label, " ".join(arpabet_phones), v4_phones,
                            )

                    if v4_phones:
                        p_tag = f'<p lock="1"><![CDATA[{v4_phones}]]></p>'
                except Exception as _wp_err:
                    logger.warning(
                        "[VSQX word_phoneme_map] 转换失败 %r: %s", label, _wp_err
                    )

            note_entries.append((t_on, dur, tone, label, p_tag))
            voiced_refs.append((t_on, t_off, tone))

        # ── 音高曲线（P 控制器）───────────────────────────────────────────
        pit_lines: List[str] = []
        if config.export_pitch_line and f0 is not None and t is not None and len(f0) > 0:
            prev_val: Optional[int] = None
            vr_arr = np.array(voiced_refs, dtype=np.int64) if voiced_refs else None

            for ti, f0i in zip(t, f0):
                if not np.isfinite(f0i) or f0i <= 0:
                    continue
                tick    = sec_to_ticks(float(ti))
                midi_f0 = 69.0 + 12.0 * np.log2(float(f0i) / 440.0)

                nominal = base_tone
                if vr_arr is not None and len(vr_arr):
                    idx = int(np.searchsorted(vr_arr[:, 0], tick, side="right")) - 1
                    if 0 <= idx < len(vr_arr) and tick <= vr_arr[idx, 1]:
                        nominal = int(vr_arr[idx, 2])

                p_val = int(round((midi_f0 - nominal) / PBS * 8190))
                p_val = max(-8190, min(8190, p_val))

                if p_val != prev_val:
                    pit_lines.append(
                        f'\t\t\t<cc><t>{tick}</t><v id="P">{p_val}</v></cc>'
                    )
                    prev_val = p_val

        if note_entries:
            play_time = max(t_on + dur for t_on, dur, *_ in note_entries) + RESOLUTION
        else:
            play_time = RESOLUTION * 4

        pit_block = f'\t\t\t<cc><t>0</t><v id="S">{PBS}</v></cc>\n'
        if pit_lines:
            pit_block += "\n".join(pit_lines) + "\n"

        note_block = ""
        for t_on, dur, tone, lyric, p_tag in note_entries:
            note_block += (
                f'\t\t\t<note>\n'
                f'\t\t\t\t<t>{t_on}</t>\n'
                f'\t\t\t\t<dur>{dur}</dur>\n'
                f'\t\t\t\t<n>{tone}</n>\n'
                f'\t\t\t\t<v>{VELOCITY}</v>\n'
                f'\t\t\t\t<y><![CDATA[{lyric}]]></y>\n'
                f'\t\t\t\t{p_tag}\n'
                f'\t\t\t\t<nStyle>\n'
                f'\t\t\t\t\t<v id="accent">50</v>\n'
                f'\t\t\t\t</nStyle>\n'
                f'\t\t\t</note>\n'
            )

        track_xml = (
            '\t<vsTrack>\n'
            f'\t\t<tNo>{track_no}</tNo>\n'
            f'\t\t<name><![CDATA[{title}]]></name>\n'
            '\t\t<comment><![CDATA[Track]]></comment>\n'
            '\t\t<vsPart>\n'
            f'\t\t\t<t>{PART_OFFSET}</t>\n'
            f'\t\t\t<playTime>{play_time}</playTime>\n'
            '\t\t\t<name><![CDATA[NewPart]]></name>\n'
            '\t\t\t<comment><![CDATA[New Musical Part]]></comment>\n'
            '\t\t\t<sPlug>\n'
            '\t\t\t\t<id><![CDATA[ACA9C502-A04B-42b5-B2EB-5CEA36D16FCE]]></id>\n'
            '\t\t\t\t<name><![CDATA[VOCALOID2 Compatible Style]]></name>\n'
            '\t\t\t\t<version><![CDATA[3.0.0.1]]></version>\n'
            '\t\t\t</sPlug>\n'
            '\t\t\t<pStyle>\n'
            '\t\t\t\t<v id="accent">50</v>\n'
            '\t\t\t\t<v id="bendDep">8</v>\n'
            '\t\t\t\t<v id="bendLen">0</v>\n'
            '\t\t\t\t<v id="decay">50</v>\n'
            '\t\t\t\t<v id="fallPort">0</v>\n'
            '\t\t\t\t<v id="opening">127</v>\n'
            '\t\t\t\t<v id="risePort">0</v>\n'
            '\t\t\t</pStyle>\n'
            '\t\t\t<singer>\n'
            '\t\t\t\t<t>0</t>\n'
            f'\t\t\t\t<bs>{vsqx_singer_bs}</bs>\n'
            '\t\t\t\t<pc>0</pc>\n'
            '\t\t\t</singer>\n'
            + pit_block
            + note_block
            + '\t\t</vsPart>\n'
            '\t</vsTrack>\n'
        )
        return track_xml

    @staticmethod
    def _vsqx_unit_xml(track_no: int) -> str:
        """构建单条 <vsUnit>（mixer 中每条音轨对应的音量/声像单元）。"""
        return (
            '\t\t<vsUnit>\n'
            f'\t\t\t<tNo>{track_no}</tNo>\n'
            '\t\t\t<iGin>0</iGin>\n'
            '\t\t\t<sLvl>-898</sLvl>\n'
            '\t\t\t<sEnable>0</sEnable>\n'
            '\t\t\t<m>0</m>\n'
            '\t\t\t<s>0</s>\n'
            '\t\t\t<pan>64</pan>\n'
            '\t\t\t<vol>0</vol>\n'
            '\t\t</vsUnit>\n'
        )

    def _build_vsqx_project_text(
        self,
        title: str,
        segments: List[LabelSegment],
        f0: Optional[np.ndarray],
        t: Optional[np.ndarray],
        sr: Optional[int],
        wav_path: str,
        config: AudioProcessingConfig,
        audio_duration_sec: Optional[float] = None,
        midi_notes: Optional[List] = None,
        vsqx_singer: str = "MIKU_V4_Chinese",
        vsqx_singer_id: str = "BNGE7CP7EMTRSNC3",
        vsqx_singer_bs: int = 4,           # VOCALOID4 内部声库 Bank Select 编号（系统相关）
        word_phoneme_map: bool = False,    # 英语单词 → VOCALOID4 音素写入 <p lock="1">
        language: str = "",                # 语种（用于 word_phoneme_map 防误判）
        native_english_words: Optional[set] = None,  # 从原始文本预提取的英语单词集合（防拼音误判）
        dict_source: str = "default",      # 单词→音素词典来源："default"/"synthesizerv"/"vocaloid"
    ) -> str:
        """
        生成 VOCALOID4 VSQX 工程文件（XML 格式，单轨）。

        格式要点
        ────────
        • resolution = 480 ticks/拍
        • tempo 写法：<tempo><t>0</t><v>{BPM×100}</v></tempo>
          例：120 BPM → v=12000；实际文件验证一致。
        • vsPart 从 tick=1920 开始（preMeasure=1，4/4，480×4=1920）
        • note 和 cc（音高曲线）的 <t> 均为相对于 vsPart 起点的偏移
        • 音高曲线（P 控制器）范围 ±8190，对应 ±PBS 个半音
          PBS 默认值 = 12（来自实际 vsqx 文件）
        • 歌词 <y>：使用 segment label；<p> 音素字段留空，
          由 VOCALOID 自带 G2P 处理（用户可在 VOCALOID Editor 中手动编辑）。

        实际的单轨音符/音高计算已抽取到 _build_vsqx_track_xml()，本函数只负责
        调用一次（track_no=0）并组装成完整工程；多轨版本见
        _build_vsqx_multi_track_project_text()。
        """
        bpm = float(config.bpm)
        tempo_v = int(round(bpm * 100))

        track_xml = self._build_vsqx_track_xml(
            title=title, segments=segments, f0=f0, t=t, config=config,
            midi_notes=midi_notes, vsqx_singer_bs=vsqx_singer_bs,
            word_phoneme_map=word_phoneme_map, language=language,
            native_english_words=native_english_words, dict_source=dict_source,
            track_no=0,
        )
        vs_unit_xml = self._vsqx_unit_xml(0)

        xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
            '<vsq4 xmlns="http://www.yamaha.co.jp/vocaloid/schema/vsq4/"\n'
            '      xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n'
            '      xsi:schemaLocation="http://www.yamaha.co.jp/vocaloid/schema/vsq4/'
            ' vsq4.xsd">\n'
            '\t<vender><![CDATA[Yamaha corporation]]></vender>\n'
            '\t<version><![CDATA[4.0.0.3]]></version>\n'
            '\t<vVoiceTable>\n'
            '\t\t<vVoice>\n'
            f'\t\t\t<bs>{vsqx_singer_bs}</bs>\n'
            '\t\t\t<pc>0</pc>\n'
            f'\t\t\t<id><![CDATA[{vsqx_singer_id}]]></id>\n'
            f'\t\t\t<name><![CDATA[{vsqx_singer}]]></name>\n'
            '\t\t\t<vPrm>\n'
            '\t\t\t\t<bre>0</bre>\n'
            '\t\t\t\t<bri>0</bri>\n'
            '\t\t\t\t<cle>0</cle>\n'
            '\t\t\t\t<gen>0</gen>\n'
            '\t\t\t\t<ope>0</ope>\n'
            '\t\t\t</vPrm>\n'
            '\t\t</vVoice>\n'
            '\t</vVoiceTable>\n'
            '\t<mixer>\n'
            '\t\t<masterUnit>\n'
            '\t\t\t<oDev>0</oDev>\n'
            '\t\t\t<rLvl>0</rLvl>\n'
            '\t\t\t<vol>0</vol>\n'
            '\t\t</masterUnit>\n'
            + vs_unit_xml +
            '\t\t<monoUnit>\n'
            '\t\t\t<iGin>0</iGin>\n'
            '\t\t\t<sLvl>-898</sLvl>\n'
            '\t\t\t<sEnable>0</sEnable>\n'
            '\t\t\t<m>0</m>\n'
            '\t\t\t<s>0</s>\n'
            '\t\t\t<pan>64</pan>\n'
            '\t\t\t<vol>0</vol>\n'
            '\t\t</monoUnit>\n'
            '\t\t<stUnit>\n'
            '\t\t\t<iGin>0</iGin>\n'
            '\t\t\t<m>0</m>\n'
            '\t\t\t<s>0</s>\n'
            '\t\t\t<vol>-129</vol>\n'
            '\t\t</stUnit>\n'
            '\t</mixer>\n'
            '\t<masterTrack>\n'
            f'\t\t<seqName><![CDATA[{title}]]></seqName>\n'
            '\t\t<comment><![CDATA[New VSQ File]]></comment>\n'
            '\t\t<resolution>480</resolution>\n'
            '\t\t<preMeasure>1</preMeasure>\n'
            '\t\t<timeSig><m>0</m><nu>4</nu><de>4</de></timeSig>\n'
            f'\t\t<tempo><t>0</t><v>{tempo_v}</v></tempo>\n'
            '\t</masterTrack>\n'
            + track_xml +
            '\t<monoTrack>\n'
            '\t</monoTrack>\n'
            '\t<stTrack>\n'
            '\t</stTrack>\n'
            '\t<aux>\n'
            '\t\t<id><![CDATA[AUX_VST_HOST_CHUNK_INFO]]></id>\n'
            '\t\t<content>'
            '<![CDATA[VlNDSwAAAAADAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=]]>'
            '</content>\n'
            '\t</aux>\n'
            '</vsq4>\n'
        )
        return xml

    def _build_vsqx_multi_track_project_text(
        self,
        title: str,
        track_inputs: List[Dict],
        config: AudioProcessingConfig,
        vsqx_singer: str = "MIKU_V4_Chinese",
        vsqx_singer_id: str = "BNGE7CP7EMTRSNC3",
        vsqx_singer_bs: int = 4,
    ) -> str:
        """
        生成多轨 VSQX 工程（对话文本框批量处理用）：每个 track_inputs 元素
        对应一个对话框，成为工程文件里的一条独立 <vsTrack>，而不是分别导出
        多个工程文件。所有音轨共用同一个 VOCALOID 声库（与单文件处理页面
        "相同的高级功能"面板保持一致，批量页面不支持逐行单独选择音源）。

        Parameters
        ----------
        track_inputs : list of dict，每个 dict 支持的键：
            name, segments, f0, t, midi_notes,
            word_phoneme_map, language, native_english_words, dict_source
        """
        bpm = float(config.bpm)
        tempo_v = int(round(bpm * 100))

        track_xmls: List[str] = []
        unit_xmls: List[str] = []

        for idx, item in enumerate(track_inputs):
            track_xmls.append(self._build_vsqx_track_xml(
                title=item.get("name") or f"{title} #{idx + 1}",
                segments=item["segments"],
                f0=item.get("f0"),
                t=item.get("t"),
                config=config,
                midi_notes=item.get("midi_notes"),
                vsqx_singer_bs=vsqx_singer_bs,
                word_phoneme_map=item.get("word_phoneme_map", False),
                language=item.get("language", ""),
                native_english_words=item.get("native_english_words"),
                dict_source=item.get("dict_source", "default"),
                track_no=idx,
            ))
            unit_xmls.append(self._vsqx_unit_xml(idx))

        xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
            '<vsq4 xmlns="http://www.yamaha.co.jp/vocaloid/schema/vsq4/"\n'
            '      xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n'
            '      xsi:schemaLocation="http://www.yamaha.co.jp/vocaloid/schema/vsq4/'
            ' vsq4.xsd">\n'
            '\t<vender><![CDATA[Yamaha corporation]]></vender>\n'
            '\t<version><![CDATA[4.0.0.3]]></version>\n'
            '\t<vVoiceTable>\n'
            '\t\t<vVoice>\n'
            f'\t\t\t<bs>{vsqx_singer_bs}</bs>\n'
            '\t\t\t<pc>0</pc>\n'
            f'\t\t\t<id><![CDATA[{vsqx_singer_id}]]></id>\n'
            f'\t\t\t<name><![CDATA[{vsqx_singer}]]></name>\n'
            '\t\t\t<vPrm>\n'
            '\t\t\t\t<bre>0</bre>\n'
            '\t\t\t\t<bri>0</bri>\n'
            '\t\t\t\t<cle>0</cle>\n'
            '\t\t\t\t<gen>0</gen>\n'
            '\t\t\t\t<ope>0</ope>\n'
            '\t\t\t</vPrm>\n'
            '\t\t</vVoice>\n'
            '\t</vVoiceTable>\n'
            '\t<mixer>\n'
            '\t\t<masterUnit>\n'
            '\t\t\t<oDev>0</oDev>\n'
            '\t\t\t<rLvl>0</rLvl>\n'
            '\t\t\t<vol>0</vol>\n'
            '\t\t</masterUnit>\n'
            + "".join(unit_xmls) +
            '\t\t<monoUnit>\n'
            '\t\t\t<iGin>0</iGin>\n'
            '\t\t\t<sLvl>-898</sLvl>\n'
            '\t\t\t<sEnable>0</sEnable>\n'
            '\t\t\t<m>0</m>\n'
            '\t\t\t<s>0</s>\n'
            '\t\t\t<pan>64</pan>\n'
            '\t\t\t<vol>0</vol>\n'
            '\t\t</monoUnit>\n'
            '\t\t<stUnit>\n'
            '\t\t\t<iGin>0</iGin>\n'
            '\t\t\t<m>0</m>\n'
            '\t\t\t<s>0</s>\n'
            '\t\t\t<vol>-129</vol>\n'
            '\t\t</stUnit>\n'
            '\t</mixer>\n'
            '\t<masterTrack>\n'
            f'\t\t<seqName><![CDATA[{title}]]></seqName>\n'
            '\t\t<comment><![CDATA[New VSQ File]]></comment>\n'
            '\t\t<resolution>480</resolution>\n'
            '\t\t<preMeasure>1</preMeasure>\n'
            '\t\t<timeSig><m>0</m><nu>4</nu><de>4</de></timeSig>\n'
            f'\t\t<tempo><t>0</t><v>{tempo_v}</v></tempo>\n'
            '\t</masterTrack>\n'
            + "".join(track_xmls) +
            '\t<monoTrack>\n'
            '\t</monoTrack>\n'
            '\t<stTrack>\n'
            '\t</stTrack>\n'
            '\t<aux>\n'
            '\t\t<id><![CDATA[AUX_VST_HOST_CHUNK_INFO]]></id>\n'
            '\t\t<content>'
            '<![CDATA[VlNDSwAAAAADAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=]]>'
            '</content>\n'
            '\t</aux>\n'
            '</vsq4>\n'
        )
        return xml

    def _build_midi_output(
        self,
        output_path: str,
        segments: List[LabelSegment],
        config: AudioProcessingConfig,
        f0: Optional[np.ndarray] = None,
        t: Optional[np.ndarray] = None,
        midi_notes: Optional[List] = None,
    ) -> None:
        """
        将段落列表写成 MIDI 文件。

        音高优先级（与 SVP/USTX 生成保持一致）：
          1. MIDI 导入音符（midi_notes 覆盖）
          2. F0 中位音高（refine_pitch=True 时）
          3. config.base_pitch 默认值
        """
        from midi_processor import build_midi_from_segments, map_segment_to_midi_pitch

        note_data: List[Tuple[float, float, int, str]] = []
        base_tone = int(config.base_pitch)

        for seg in segments:
            if self._is_true_silence(seg.label):
                continue
            if seg.end_time <= seg.start_time:
                continue

            start_sec = self._lab_time_to_seconds(seg.start_time)
            end_sec   = self._lab_time_to_seconds(seg.end_time)

            tone = base_tone
            if midi_notes is not None:
                tone = map_segment_to_midi_pitch(
                    start_sec, end_sec, midi_notes,
                    base_pitch=base_tone,
                )
            elif config.refine_pitch and f0 is not None and t is not None and len(f0) > 0:
                mask = (t >= start_sec) & (t < end_sec)
                voiced = f0[mask]
                voiced = voiced[voiced > 0]
                if len(voiced) > 0:
                    midi_vals = 69.0 + 12.0 * np.log2(voiced / 440.0)
                    tone = int(round(float(np.median(midi_vals))))
                    tone = max(12, min(127, tone))

            note_data.append((start_sec, end_sec, tone, seg.label))

        build_midi_from_segments(
            segments=note_data,
            bpm=config.bpm,
            output_path=output_path,
        )


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