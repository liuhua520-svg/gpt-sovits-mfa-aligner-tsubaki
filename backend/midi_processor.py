# -*- coding: utf-8 -*-
"""
MIDI 文件解析模块
用于从 MIDI 文件中提取 BPM、音符时序/音高，以及可选歌词。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

logger = logging.getLogger(__name__)

_NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


@dataclass
class MidiLyricEvent:
    time_sec: float
    text: str


def midi_note_name(pitch: int) -> str:
    """MIDI 音符号 → 音名（如 60 → C4）"""
    octave = (pitch // 12) - 1
    return f"{_NOTE_NAMES[pitch % 12]}{octave}"


def parse_midi_notes_with_lyrics(
    midi_path: str,
) -> Tuple[float, List[Tuple[float, float, int]], List[MidiLyricEvent]]:
    """
    解析 MIDI 文件，提取全局 BPM、所有音符，以及 lyrics meta event。

    Parameters
    ----------
    midi_path : str
        .mid / .midi 文件路径

    Returns
    -------
    bpm : float
        文件中第一个 set_tempo 事件的 BPM（默认 120.0）
    notes : list of (start_sec, end_sec, pitch)
        按开始时间升序排列的音符列表，pitch 为 MIDI 音高 (0–127)
    lyrics : list of MidiLyricEvent
        按时间升序排列的歌词事件
    """
    try:
        import mido
    except ImportError:
        raise ImportError("请安装 mido 库以使用 MIDI 导入功能: pip install mido")

    mid = mido.MidiFile(midi_path)
    ticks_per_beat = mid.ticks_per_beat

    bpm: float = 120.0
    notes_result: List[Tuple[float, float, int]] = []
    lyrics_result: List[MidiLyricEvent] = []

    for track in mid.tracks:
        tempo = 500_000      # 默认 120 BPM（微秒/拍）
        abs_time_sec = 0.0
        active_notes: dict = {}  # pitch -> start_sec（等待 note_off 的音符）

        for msg in track:
            # delta time (ticks) → 秒
            delta_sec = mido.tick2second(msg.time, ticks_per_beat, tempo)
            abs_time_sec += delta_sec

            if msg.type == 'set_tempo':
                tempo = msg.tempo
                # 仅取第一个 tempo 事件作为全局 BPM
                if bpm == 120.0:
                    bpm = round(60_000_000 / max(tempo, 1), 3)
            elif msg.type == 'lyrics':
                text = (getattr(msg, "text", "") or "").strip()
                if text:
                    lyrics_result.append(MidiLyricEvent(
                        time_sec=abs_time_sec,
                        text=text
                    ))

            elif msg.type == 'note_on' and msg.velocity > 0:
                active_notes[msg.note] = abs_time_sec

            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                if msg.note in active_notes:
                    start_sec = active_notes.pop(msg.note)
                    end_sec = abs_time_sec
                    if end_sec > start_sec:          # 过滤零长度音符
                        notes_result.append((start_sec, end_sec, msg.note))

    notes_result.sort(key=lambda n: n[0])
    lyrics_result.sort(key=lambda e: e.time_sec)

    if notes_result:
        pitches = [n[2] for n in notes_result]
        logger.info(
            f"MIDI 解析完成: BPM={bpm:.1f}, "
            f"共 {len(notes_result)} 个音符, "
            f"音高范围 {midi_note_name(min(pitches))}–{midi_note_name(max(pitches))}"
        )
    else:
        logger.warning(f"MIDI 解析完成但未找到任何音符: {midi_path}")

    return bpm, notes_result, lyrics_result


def parse_midi_notes(midi_path: str) -> Tuple[float, List[Tuple[float, float, int]]]:
    """
    兼容旧接口：仍然只返回 bpm + notes。
    """
    bpm, notes, _lyrics = parse_midi_notes_with_lyrics(midi_path)
    return bpm, notes


def build_midi_from_segments(
    segments: List[Tuple[float, float, int, str]],
    bpm: float = 120.0,
    output_path: str = "output.mid",
    ticks_per_beat: int = 480,
    default_velocity: int = 80,
) -> str:
    """
    从段落列表生成 MIDI 文件。

    Parameters
    ----------
    segments : list of (start_sec, end_sec, pitch_midi, label)
        段落列表。label 会写入 MIDI 的 lyrics meta event。
    bpm : float
        每分钟拍数，写入 set_tempo 事件。
    output_path : str
        输出 .mid 文件路径。
    ticks_per_beat : int
        MIDI 分辨率，默认 480 ticks/beat。
    default_velocity : int
        音符力度（0-127），默认 80。

    Returns
    -------
    str : 实际写入的文件路径（同 output_path）。
    """
    try:
        import mido
    except ImportError:
        raise ImportError("请安装 mido 库以使用 MIDI 导出功能: pip install mido")

    def sec_to_ticks(sec: float) -> int:
        return int(round(sec * (bpm / 60.0) * ticks_per_beat))

    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    # --- meta events ---
    tempo = int(round(60_000_000 / max(bpm, 1e-6)))
    track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))
    track.append(mido.MetaMessage(
        "time_signature", numerator=4, denominator=4,
        clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0
    ))

    # --- note events: interleave note_on and note_off by absolute tick ---
    events: List[Tuple[int, int, str, int, str]] = []  # (abs_tick, sort_key, type, pitch, text)
    for start_sec, end_sec, pitch, label in segments:
        pitch = max(0, min(127, int(pitch)))
        t_on  = sec_to_ticks(float(start_sec))
        t_off = max(t_on + 1, sec_to_ticks(float(end_sec)))
        clean_label = (label or "").strip()
        is_silence = clean_label.lower() in {"-", "sil", "pau", "sp", "spn", "rest", ""}

        # lyrics 事件放在音符开始位置
        if not is_silence:
            events.append((t_on,  1, "lyrics",   pitch, clean_label))

        events.append((t_on,  2, "note_on",  pitch, ""))   # note_on
        events.append((t_off, 0, "note_off", pitch, ""))   # note_off

    # note_off -> lyrics -> note_on，确保同 tick 事件顺序稳定
    events.sort(key=lambda e: (e[0], e[1]))

    prev_tick = 0
    for abs_tick, _sk, etype, pitch, text in events:
        delta = max(0, abs_tick - prev_tick)
        prev_tick = abs_tick
        if etype == "lyrics":
            track.append(mido.MetaMessage("lyrics", text=text, time=delta))
            continue
        if etype == "note_on":
            track.append(mido.Message("note_on", note=pitch,
                                      velocity=default_velocity, time=delta))
        else:
            track.append(mido.Message("note_off", note=pitch,
                                      velocity=0, time=delta))
                                      
                                      
def map_segment_to_midi_pitch(
    lab_start_sec: float,
    lab_end_sec: float,
    midi_notes: List[Tuple[float, float, int]],
    base_pitch: int = 60,
    min_overlap_sec: float = 0.01,
) -> int:
    """
    将一个 LAB/SVP 段落映射到时间重叠最多的 MIDI 音符音高。

    Parameters
    ----------
    lab_start_sec, lab_end_sec : float
        当前段的起止时间（秒）
    midi_notes : list of (start_sec, end_sec, pitch)
        已按 start_sec 排序的 MIDI 音符列表
    base_pitch : int
        没有找到匹配时返回的默认音高
    min_overlap_sec : float
        参与匹配的最小重叠时长阈值

    Returns
    -------
    int
        MIDI 音高 (0–127)
    """
    best_pitch = int(base_pitch)
    best_overlap = 0.0

    for start_sec, end_sec, pitch in midi_notes:
        if end_sec < lab_start_sec:
            continue
        if start_sec > lab_end_sec:
            break

        overlap = max(0.0, min(lab_end_sec, end_sec) - max(lab_start_sec, start_sec))
        if overlap >= min_overlap_sec and overlap > best_overlap:
            best_overlap = overlap
            best_pitch = int(pitch)

    return best_pitch