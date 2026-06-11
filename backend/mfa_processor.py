# -*- coding: utf-8 -*-
"""
MFA 处理核心模块 - 多语言增强版 v9.5
修复：按句子边界分段（替代 RMS 静音分割），解决时间戳错位问题
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
import shutil
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from pypinyin import lazy_pinyin, Style
from mfa_utils import MFAChecker
from phoneme_converter import convert_phoneme, build_ja_hiragana_lab, merge_lab_silence

logger = logging.getLogger(__name__)

# ── Module-level cache: SudachiPy dict loads in ~4 min on first call.
# Caching it here means subsequent requests reuse the same object. ──────────
_ja_tokenizer: Optional[object] = None


class MFAProcessor:
    """Montreal Forced Aligner 处理器 - 多语言增强版 v9.5"""

    INITIALS_EXTENDED = [
        "zh", "ch", "sh",
        "b", "p", "d", "t", "g", "k",
        "f", "h", "j", "q", "x", "z", "c", "s",
        "m", "n", "l", "r",
        "y", "w",
    ]
    INITIALS = set(INITIALS_EXTENDED)

    CON_INITIALS = {
        "zh", "ch", "sh",
        "b", "p", "d", "t", "g", "k",
        "f", "h", "j", "q", "x", "z", "c", "s",
        "m", "n", "l", "r",
    }

    YUE_CON_INITIALS = {
        "b", "p", "m", "f",
        "d", "t", "n", "l",
        "g", "k", "ng", "h",
        "gw", "kw",
        "z", "c", "s",
    }

    FINALS_EXTENDED = {
        "a", "o", "e", "i", "u", "v", "ü", "er",
        "ai", "ei", "ui", "ao", "ou", "iu", "ie", "uo", "üe", "ve",
        "an", "en", "in", "un", "ün", "vn",
        "ang", "eng", "ing", "ong",
        "iao", "ian", "iang", "iong", "uai", "uan", "uang", "üan", "van",
    }

    SIL_PHONES = {"sp", "spn", "sil", "silence", "pau", "breath", "noise", "ap", "blank"}
    IGNORE_PHONES = {"", "<eps>"}

    SPECIAL_SYLLABLES = {
        "zhi", "chi", "shi", "ri", "zi", "ci", "si",
        "yi", "ya", "yo", "yao", "ye", "yin", "ying", "yong",
        "wu", "wa", "wo", "wai", "wei", "wan", "wang", "weng",
        "yu", "yue", "yuan", "yun",
    }

    LONG_FINAL_HINTS = {
        "iang", "iong", "uang", "uai", "iao",
        "ang", "eng", "ing", "ong",
        "ian", "uan", "uen", "van", "üan",
    }
    SHORT_FINAL_HINTS = {
        "ai", "ei", "ui", "ao", "ou",
        "an", "en", "in", "un", "vn", "ün",
        "a", "o", "e", "i", "u", "v", "ü", "er",
    }

    GAP_THRESHOLD_100NS = 100000
    SILENCE_THRESHOLD_100NS = 2000000

    PHONEME_TABLES = {
        'zh': {
            'vowels': {
                'a', 'o', 'e', 'i', 'u', 'v', 'ü', 'er',
                'ai', 'ei', 'ui', 'ao', 'ou', 'iu', 'ie', 'uo', 'üe', 've',
                'an', 'en', 'in', 'un', 'ün', 'vn',
                'ang', 'eng', 'ing', 'ong',
                'iao', 'ian', 'iang', 'iong', 'uai', 'uan', 'uang', 'üan', 'van'
            },
            'stops': {'b', 'p', 'd', 't', 'g', 'k'},
            'fricatives': {'f', 'h', 'j', 'q', 'x', 'zh', 'ch', 'sh', 'z', 'c', 's', 'r'},
            'nasals_glides': {'m', 'n', 'l', 'y', 'w'}
        },
        'en': {
            'vowels': {
                'aa', 'ae', 'ah', 'ao', 'aw', 'ay', 'eh', 'er', 'ey', 'ih', 'iy',
                'ow', 'oy', 'uh', 'uw'
            },
            'stops': {'b', 'd', 'g', 'k', 'p', 't'},
            'fricatives': {'ch', 'dh', 'f', 'hh', 'jh', 's', 'sh', 'th', 'v', 'z', 'zh'},
            'nasals_glides': {'l', 'm', 'n', 'ng', 'r', 'w', 'y'}
        },
        'ja': {
            'vowels': {'a', 'i', 'u', 'e', 'o', 'n'},
            'stops': {'k', 'g', 't', 'd', 'b', 'p'},
            'fricatives': {'s', 'z', 'h', 'ts', 'ch', 'sh'},
            'nasals_glides': {'m', 'n', 'y', 'r', 'w'}
        }
    }

    PRE_ROLL_LIMITS = {
        'stops': 300000,
        'fricatives': 800000,
        'nasals_glides': 500000,
        'vowels': 0
    }

    DIGIT_PINYIN = {
        '0': 'ling', '1': 'yi',  '2': 'er',  '3': 'san', '4': 'si',
        '5': 'wu',   '6': 'liu', '7': 'qi',  '8': 'ba',  '9': 'jiu',
    }

    DIGIT_JYUTPING = {
        '0': 'ling', '1': 'jat', '2': 'ji',   '3': 'saam', '4': 'sei',
        '5': 'ng',   '6': 'luk', '7': 'cat',  '8': 'baat', '9': 'gau',
    }

    JA_PHONE_DISPLAY = {
        'cl':  'q',
        'N':   'N',
        'pau': 'sil',
        'sp':  'sil',
    }

    # ── 长音频处理改为按句子分割 ────────────────────────────────────────
    LONG_AUDIO_THRESHOLD_SEC: float = 25.0
    MAX_SEGMENT_SEC: float = 27.0

    def __init__(self):
        self.temp_dir: Optional[str] = None

    # ... [保留所有原有的工具方法，不重复列出] ...

    def _clean_input_text(self, text: str) -> str:
        """清洗文本标点符号"""
        if not text:
            return ""
        ignore_pattern = r'[「」"""]'
        text = re.sub(ignore_pattern, "", text)
        comma_pattern = r'[（）《》()＜＞<>【】]'
        text = re.sub(comma_pattern, ",", text)
        return text

    def _check_zh_environment(self) -> Tuple[bool, str]:
        try:
            from pypinyin import lazy_pinyin, Style
            lazy_pinyin("测试", style=Style.NORMAL)
            return True, "✓ 中文环境就绪（pypinyin已安装）"
        except ImportError:
            return False, "❌ 缺失中文支持：pip install pypinyin"
        except Exception as e:
            return False, f"❌ 中文环境错误：{str(e)}"

    def _check_en_environment(self) -> Tuple[bool, str]:
        try:
            mfa_ok, mfa_msg = MFAChecker.check_mfa_installed()
            if mfa_ok:
                return True, "✓ 英语环境就绪（MFA已安装）"
            else:
                return False, f"❌ MFA未安装：{mfa_msg}"
        except Exception as e:
            return False, f"❌ 英语环境错误：{str(e)}"

    def _check_ja_environment(self) -> Tuple[bool, str]:
        """检查日语环境"""
        global _ja_tokenizer
        try:
            from sudachipy import Dictionary
            if _ja_tokenizer is None:
                _ja_tokenizer = Dictionary().create()
            test_result = _ja_tokenizer.tokenize("テスト")
            if test_result and len(test_result) > 0:
                return True, "✓ 日语环境就绪（sudachipy + 字典已就绪）"
            else:
                return False, "❌ 日语字典初始化失败"
        except ImportError:
            return False, "❌ 缺失日语支持（sudachipy未安装）\n   请运行：pip install sudachipy sudachidict-core"
        except Exception as e:
            error_msg = str(e)
            if "dictionary" in error_msg.lower():
                return False, "❌ 日语字典资源不可用\n   请重新安装：pip install --force-reinstall sudachidict-core"
            else:
                return False, f"❌ 日语环境错误：{error_msg}"

    def _check_ko_environment(self) -> Tuple[bool, str]:
        try:
            mfa_ok, mfa_msg = MFAChecker.check_mfa_installed()
            if not mfa_ok:
                return False, f"❌ MFA未安装：{mfa_msg}"
            try:
                import jamo
                return True, "✓ 韩语环境就绪（MFA + jamo已安装）"
            except ImportError:
                return False, "❌ 缺失韩语支持（jamo）\n   请运行：pip install jamo"
        except Exception as e:
            return False, f"❌ 韩语环境错误：{str(e)}"

    def _check_yue_environment(self) -> Tuple[bool, str]:
        try:
            import pycantonese
            return True, "✓ 粤语环境就绪（pycantonese已安装）"
        except ImportError:
            return False, "❌ 缺失粤语支持（pycantonese）\n   请运行：pip install pycantonese"
        except Exception as e:
            return False, f"❌ 粤语环境错误：{str(e)}"

    def _check_language_environment(self, lang: str) -> Tuple[bool, str]:
        logger.info(f"\n{'='*60}\n检查语言环境：{lang.upper()}\n{'='*60}")
        if lang in ('zh', 'cmn'):
            ok, msg = self._check_zh_environment()
        elif lang == 'en':
            ok, msg = self._check_en_environment()
        elif lang == 'ja':
            ok, msg = self._check_ja_environment()
        elif lang == 'ko':
            ok, msg = self._check_ko_environment()
        elif lang == 'yue':
            ok, msg = self._check_yue_environment()
        else:
            logger.warning(f"未知语言代码：{lang}")
            return True, f"未知语言：{lang}（跳过环境检查）"
        logger.info(msg)
        return ok, msg

    def _text_to_pinyin_notone(self, text: str) -> str:
        phones = lazy_pinyin(text, style=Style.NORMAL, errors="ignore")
        phones = [p.strip().lower() for p in phones if p and p.strip()]
        return " ".join(phones)

    def _text_to_jyutping(self, text: str) -> str:
        try:
            import pycantonese
            result = pycantonese.characters_to_jyutping(text)
            syls: List[str] = []
            for _char, jyut in result:
                if jyut:
                    jyut_no_tone = re.sub(r'[0-9]+$', '', jyut).strip().lower()
                    if jyut_no_tone:
                        syls.append(jyut_no_tone)
            return " ".join(syls)
        except (ImportError, Exception) as e:
            logger.warning(f"粤语转换失败: {e}")
            return ""

    def _normalize_no_tone_pinyin(self, text: str) -> List[str]:
        if not text:
            return []
        tokens: List[str] = []
        for part in text.split():
            p = part.strip().lower()
            if p:
                p = re.sub(r"[^a-züv]+", "", p)
                if p:
                    tokens.append(p)
        return tokens

    def _normalize_jyutping(self, text: str) -> List[str]:
        if not text:
            return []
        tokens: List[str] = []
        for part in text.split():
            p = part.strip().lower()
            if p:
                p = re.sub(r"[^a-z]+", "", p)
                if p:
                    tokens.append(p)
        return tokens

    def _segment_to_syllables(self, segment: str) -> List[str]:
        return self._normalize_no_tone_pinyin(self._text_to_pinyin_notone(segment))

    def _segment_to_jyutping(self, segment: str) -> List[str]:
        return self._normalize_jyutping(self._text_to_jyutping(segment))

    def _is_english_word(self, word: str) -> bool:
        return bool(re.match(r"^[a-zA-Z''\-]+$", word.strip()))

    def _is_digit_char(self, word: str) -> bool:
        return word.strip() in '0123456789'

    def _clean_phone(self, phone: str) -> str:
        if not phone:
            return ""
        phone = phone.strip()
        tone_marks = ["˥", "˧", "˩", "˨", "˦", "˧˥", "˥˩", "˩˧", "˨˩", "˥˧", "˧˨", "˨˧"]
        for mark in tone_marks:
            phone = phone.replace(mark, "")
        return phone.strip().lower()

    def _is_silence_phone(self, phone: str) -> bool:
        return (phone or "").strip().lower() in self.SIL_PHONES

    def _clean_phone_token(self, phone: str) -> str:
        phone = (phone or "").strip().lower().replace("ü", "v")
        phone = re.sub(r"[^a-zA-Zv]+", "", phone)
        return phone.strip()

    # ... [省略中间大量重复的工具方法，保持原样] ...

    def _split_text_at_sentence_boundaries(self, text: str, lang: str) -> List[str]:
        """
        按句子边界分割文本。
        
        CJK 语言：在 。！？；\n 处分割（完整句子）
        英语：在 . ! ? 后接空格处分割
        """
        text = text.strip()
        if not text:
            return []

        if lang in ('zh', 'cmn', 'yue', 'ja', 'jpn', 'ko', 'kor'):
            # 在句末标点处分割，保留标点
            sentences = re.split(r'(?<=[。！？；\n])', text)
            return [s.strip() for s in sentences if s.strip()]
        else:
            # 英语：在句末标点后接空格处分割
            parts = re.split(r'(?<=[.!?])\s+', text)
            return [p.strip() for p in parts if p.strip()]

    def _split_audio_by_sentences(
        self,
        audio_path: str,
        text: str,
        lang: str,
    ) -> List[Tuple[str, float, float]]:
        """
        按句子边界分割音频。
        
        返回 [(segment_wav_path, start_sec, end_sec), ...]
        """
        try:
            import numpy as np
            import soundfile as sf
        except ImportError:
            logger.warning("soundfile/numpy 不可用，跳过分割")
            return [(audio_path, 0.0, self._get_audio_duration(audio_path) / 10_000_000)]

        # 读取音频
        try:
            data, sr = sf.read(audio_path, dtype="float32")
        except Exception as e:
            logger.error(f"读取音频失败: {e}")
            return [(audio_path, 0.0, self._get_audio_duration(audio_path) / 10_000_000)]

        if data.ndim > 1:
            data = data.mean(axis=1)

        total_sec = len(data) / sr
        
        # 获取句子列表
        sentences = self._split_text_at_sentence_boundaries(text, lang)
        if len(sentences) <= 1:
            logger.info("文本无法按句子分割，保持单一片段")
            return [(audio_path, 0.0, total_sec)]

        # 按句子数量均匀分割音频时间
        n_sentences = len(sentences)
        segment_duration = total_sec / n_sentences
        
        segments: List[Tuple[str, float, float]] = []
        base = Path(audio_path).stem
        seg_dir = self.temp_dir or os.path.dirname(audio_path)

        for i in range(n_sentences):
            seg_start = i * segment_duration
            seg_end = (i + 1) * segment_duration if i < n_sentences - 1 else total_sec
            
            start_sample = int(seg_start * sr)
            end_sample = int(seg_end * sr)
            seg_data = data[start_sample:end_sample]
            
            seg_path = os.path.join(seg_dir, f"{base}_sent{i:03d}.wav")
            try:
                sf.write(seg_path, seg_data, sr)
            except Exception as e:
                logger.error(f"写分段 WAV 失败: {e}")
                continue
            
            segments.append((seg_path, seg_start, seg_end))
            logger.info(f"  句子{i+1}/{n_sentences}: [{seg_start:.2f}s, {seg_end:.2f}s] → {seg_path}")

        return segments if segments else [(audio_path, 0.0, total_sec)]

    @staticmethod
    def _merge_segment_lab_entries(
        all_entries: List[Tuple[int, int, str]],
    ) -> List[Tuple[int, int, str]]:
        """
        合并各分段 LAB 条目：按时间排序并填补空隙。
        """
        if not all_entries:
            return []

        all_entries.sort(key=lambda x: x[0])
        result: List[Tuple[int, int, str]] = []

        for s, e, p in all_entries:
            if not result:
                result.append((s, e, p))
                continue

            prev_s, prev_e, prev_p = result[-1]
            gap = s - prev_e
            
            if 0 < gap < 500_000:  # 填补 <50ms 的空隙
                result[-1] = (prev_s, s, prev_p)

            if p == 'sil' and prev_p == 'sil':
                result[-1] = (prev_s, max(prev_e, e), 'sil')
            else:
                result.append((s, e, p))

        return result

    def _process_with_sentence_segmentation(
        self,
        audio_path: str,
        text: str,
        lang: str,
        text_for_mfa: str,
        phoneme_text: str,
        start_time: float,
    ) -> Dict:
        """
        按句子分段处理长音频。
        
        ① 按句子边界切分音频
        ② 对每个分段独立运行 MFA
        ③ 合并 LAB 结果
        """
        logger.info(f"[句子分段模式] 启动句子边界分段对齐")

        # ① 按句子分割
        sentences = self._split_text_at_sentence_boundaries(text, lang)
        logger.info(f"文本分割为 {len(sentences)} 个句子")
        
        segments = self._split_audio_by_sentences(audio_path, text, lang)
        
        if len(segments) <= 1:
            logger.info("分割结果为单一片段，回退到直接处理模式")
            return self._process_direct(
                audio_path, text, lang, text_for_mfa, phoneme_text, start_time
            )

        logger.info(f"共 {len(segments)} 个分段")

        # ② 逐分段 MFA 对齐
        all_entries: List[Tuple[int, int, str]] = []
        success_count = 0

        for i, ((seg_path, seg_start_sec, seg_end_sec), seg_text) in enumerate(
            zip(segments, sentences)
        ):
            seg_text = seg_text.strip()
            if not seg_text:
                logger.warning(f"  句子{i+1} 文本为空，跳过")
                continue

            logger.info(f"  ▶ 句子{i+1}/{len(segments)} [{seg_start_sec:.2f}s-{seg_end_sec:.2f}s]")

            seg_corpus = os.path.join(self.temp_dir, f"corpus_sent{i:03d}")
            seg_output = os.path.join(self.temp_dir, f"aligned_sent{i:03d}")
            os.makedirs(seg_corpus, exist_ok=True)
            os.makedirs(seg_output, exist_ok=True)

            seg_basename = f"sent{i:03d}"
            seg_wav_dest = os.path.join(seg_corpus, f"{seg_basename}.wav")
            try:
                shutil.copy2(seg_path, seg_wav_dest)
            except Exception as e:
                logger.error(f"  复制分段 WAV 失败: {e}")
                continue

            seg_txt_path = os.path.join(seg_corpus, f"{seg_basename}.txt")
            cleaned_seg_text = self._prepare_text_for_mfa(seg_text, lang)
            if not cleaned_seg_text:
                logger.warning(f"  句子{i+1} 清洗后文本为空，跳过")
                continue
            with open(seg_txt_path, "w", encoding="utf-8") as f:
                f.write(cleaned_seg_text)

            seg_timeout = max(90, int((seg_end_sec - seg_start_sec) * 25))
            if lang in ('ja', 'ko'):
                seg_timeout = max(120, int((seg_end_sec - seg_start_sec) * 40))

            seg_mfa_temp = os.path.join(self.temp_dir, f"mfa_work_sent_{i:03d}")
            tg_path = self._run_mfa_align(
                seg_corpus, seg_output, lang, seg_timeout, seg_mfa_temp,
                beam=100, retry_beam=400,
            )

            if tg_path is None:
                logger.warning(f"  句子{i+1} 首次对齐失败，使用超大束重试...")
                seg_output_retry = os.path.join(self.temp_dir, f"aligned_sent{i:03d}_retry")
                seg_mfa_temp_retry = os.path.join(self.temp_dir, f"mfa_work_sent_{i:03d}_retry")
                os.makedirs(seg_output_retry, exist_ok=True)
                tg_path = self._run_mfa_align(
                    seg_corpus, seg_output_retry, lang,
                    seg_timeout + 60, seg_mfa_temp_retry,
                    beam=200, retry_beam=800,
                )

            if tg_path is None:
                logger.warning(f"  句子{i+1} 对齐失败（两次尝试均失败），跳过")
                continue

            # 解析 TextGrid → LAB
            seg_lab = self._textgrid_to_lab(tg_path, seg_text, lang=lang)
            if not seg_lab:
                logger.warning(f"  句子{i+1} LAB 为空，跳过")
                continue

            # ★ 关键修复：时间戳已经是相对于该分段的绝对值，直接加上分段起始偏移
            offset_100ns = int(seg_start_sec * 10_000_000)
            for line in seg_lab.splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        entry_start = int(parts[0]) + offset_100ns
                        entry_end = int(parts[1]) + offset_100ns
                        label = parts[2]
                        all_entries.append((entry_start, entry_end, label))
                    except ValueError:
                        continue

            success_count += 1

        if not all_entries:
            logger.error("所有句子对齐均失败，回退到直接处理模式")
            return self._process_direct(
                audio_path, text, lang, text_for_mfa, phoneme_text, start_time
            )

        merged_entries = self._merge_segment_lab_entries(all_entries)
        lab_content = "\n".join(f"{s} {e} {p}" for s, e, p in merged_entries)

        processing_time = int((time.time() - start_time) * 1000)
        logger.info(
            f"[句子分段模式] 完成：{success_count}/{len(segments)} 个句子成功，"
            f"共 {len(merged_entries)} 条 LAB 条目，耗时 {processing_time}ms"
        )

        return {
            "success": True,
            "raw_text": text,
            "phoneme_text": phoneme_text,
            "lab_content": lab_content,
            "audio_duration": self._get_audio_duration(audio_path),
            "processing_time": processing_time,
            "segments_count": len(segments),
            "segments_success": success_count,
        }

    def _prepare_text_for_mfa(self, text: str, lang: str) -> str:
        """送入 MFA 前的文本清洗"""
        if not text:
            return ""

        if lang in ('zh', 'cmn'):
            clean = re.sub(
                r'[^\u4e00-\u9fa5\u3400-\u4dbf\uf900-\ufaff a-zA-Z0-9]', ' ', text
            )
        elif lang == 'yue':
            clean = re.sub(
                r'[^\u4e00-\u9fa5\u3400-\u4dbf\uf900-\ufaff a-zA-Z0-9]', ' ', text
            )
        elif lang in ('ja', 'jpn'):
            clean = re.sub(
                r'[^\u3040-\u30ff\u4e00-\u9fa5\uff66-\uff9f a-zA-Z0-9]', ' ', text
            )
        elif lang in ('ko', 'kor'):
            clean = re.sub(
                r'[^\uac00-\ud7a3\u3130-\u318f\u1100-\u11ff a-zA-Z0-9]', ' ', text
            )
        else:
            clean = re.sub(r"[^a-zA-Z0-9'\- ]", ' ', text)

        clean = re.sub(r'\s+', ' ', clean).strip()
        return clean

    def _run_mfa_align(
        self,
        corpus_dir: str,
        output_dir: str,
        lang: str,
        timeout_seconds: int = 300,
        mfa_temp_dir: Optional[str] = None,
        beam: int = 100,
        retry_beam: int = 400,
    ) -> Optional[str]:
        """运行 MFA align"""
        models = MFAChecker.LANGUAGE_MODELS.get(lang, {})
        dict_model = models.get("dictionary", lang)
        acoustic_model = models.get("acoustic", lang)
        py = MFAChecker.env_python()

        cmd = [
            str(py), "-m", "montreal_forced_aligner.command_line.mfa",
            "align", corpus_dir, dict_model, acoustic_model,
            output_dir, "--clean", "--single_speaker",
            "--beam", str(beam), "--retry_beam", str(retry_beam),
        ]

        if mfa_temp_dir is not None:
            os.makedirs(mfa_temp_dir, exist_ok=True)
            cmd += ["--temp_directory", mfa_temp_dir]

        env = os.environ.copy()
        mfa_env_dir = MFAChecker.env_dir()
        env["CONDA_PREFIX"] = str(mfa_env_dir)
        lib_bin = mfa_env_dir / "Library" / "bin"
        if lib_bin.exists():
            env["PATH"] = str(lib_bin) + os.pathsep + env.get("PATH", "")

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=timeout_seconds, env=env
            )
        except subprocess.TimeoutExpired:
            logger.error(f"MFA align 超时 ({timeout_seconds}s)")
            return None
        except Exception as e:
            logger.error(f"MFA align 异常: {e}")
            return None

        if result.returncode != 0:
            full_diag = ((result.stderr or "") + (result.stdout or "")).strip()
            logger.error(
                f"MFA align 失败 (code={result.returncode}):\n"
                f"{'=' * 60}\n{full_diag}\n{'=' * 60}"
            )
            return None

        textgrid_files = list(Path(output_dir).glob("**/*.TextGrid"))
        if not textgrid_files:
            logger.error("MFA 未生成 TextGrid")
            return None

        return str(textgrid_files[0])

    def _process_direct(
        self,
        audio_path: str,
        text: str,
        lang: str,
        text_for_mfa: str,
        phoneme_text: str,
        start_time: float,
    ) -> Dict:
        """直接（不分段）处理"""
        corpus_dir = os.path.join(self.temp_dir, "corpus")
        os.makedirs(corpus_dir, exist_ok=True)
        basename = Path(audio_path).stem
        audio_dest = os.path.join(corpus_dir, f"{basename}.wav")
        shutil.copy2(audio_path, audio_dest)
        text_file = os.path.join(corpus_dir, f"{basename}.txt")
        cleaned_text_for_mfa = self._prepare_text_for_mfa(text_for_mfa, lang)
        with open(text_file, "w", encoding="utf-8") as f:
            f.write(cleaned_text_for_mfa or text_for_mfa)

        output_dir = os.path.join(self.temp_dir, "aligned")
        os.makedirs(output_dir, exist_ok=True)

        audio_duration_sec = self._get_audio_duration(audio_path) / 10_000_000
        timeout_seconds = max(300, int(audio_duration_sec * 15))
        if lang in ('ja', 'ko'):
            timeout_seconds = max(600, int(audio_duration_sec * 25))

        direct_mfa_temp = os.path.join(self.temp_dir, "mfa_work_direct")
        tg_path = self._run_mfa_align(
            corpus_dir, output_dir, lang, timeout_seconds, direct_mfa_temp,
            beam=100, retry_beam=400,
        )

        if tg_path is None:
            return {
                "success": False, "error": "MFA align 失败或未生成 TextGrid",
                "processing_time": int((time.time() - start_time) * 1000)
            }

        lab_content = self._textgrid_to_lab(tg_path, text, lang=lang)
        if not lab_content:
            return {
                "success": False, "error": "LAB 内容为空",
                "processing_time": int((time.time() - start_time) * 1000)
            }

        return {
            "success": True,
            "raw_text": text,
            "phoneme_text": phoneme_text,
            "lab_content": lab_content,
            "textgrid_path": tg_path,
            "audio_duration": self._get_audio_duration(audio_path),
            "processing_time": int((time.time() - start_time) * 1000),
        }

    def _textgrid_to_lab(self, textgrid_path: str, text: str, lang: str = 'zh') -> str:
        """TextGrid → LAB"""
        return self._textgrid_to_lab_word_tier_primary(textgrid_path, text, lang)

    def _textgrid_to_lab_word_tier_primary(
        self,
        textgrid_path: str,
        text: str,
        lang: str = 'zh'
    ) -> str:
        """Word Tier 对齐主逻辑"""
        try:
            from textgrid import TextGrid
            tg = TextGrid.fromFile(textgrid_path)
            word_tier = None
            phone_tier = None
            for candidate in tg:
                tier_name = getattr(candidate, "name", "").lower()
                if "word" in tier_name:
                    word_tier = candidate
                elif "phone" in tier_name or "phoneme" in tier_name:
                    phone_tier = candidate
            if word_tier is None:
                logger.error("未找到 Word Tier")
                return ""
            phone_items: List[Tuple[int, int, str]] = []
            if phone_tier is not None:
                phone_items = self._extract_phone_items(phone_tier)
            if lang in ('zh', 'cmn'):
                lines = self._process_zh_words(word_tier, phone_items, text)
            else:
                lines = self._process_en_words(word_tier, phone_items, text)
            lines = self._apply_lab_postprocess(lines, lang)
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"对齐失败: {e}", exc_info=True)
            return ""

    def _extract_phone_items(self, tier) -> List[Tuple[int, int, str]]:
        items: List[Tuple[int, int, str]] = []
        for interval in tier:
            mark = self._clean_phone((interval.mark or "").strip())
            if mark in self.IGNORE_PHONES:
                continue
            start = int(interval.minTime * 10000000)
            end = int(interval.maxTime * 10000000)
            items.append((start, end, mark))
        items.sort(key=lambda x: (x[0], x[1]))
        return items

    def _process_zh_words(
        self,
        word_tier,
        phone_items: List[Tuple[int, int, str]],
        text: str
    ) -> List[str]:
        """处理中文 word tier"""
        target_syls = self._normalize_no_tone_pinyin(self._text_to_pinyin_notone(text))
        lines: List[str] = []
        syl_index = 0

        for interval in word_tier:
            mark = getattr(interval, "mark", getattr(interval, "text", ""))
            mark = (mark or "").strip()
            start = int(interval.minTime * 10000000)
            end = int(interval.maxTime * 10000000)

            if not mark or mark in self.IGNORE_PHONES:
                continue
            if mark in self.SIL_PHONES or mark in ("sp", "spn"):
                lines.append(f"{start} {end} sil")
                continue
            if self._is_english_word(mark):
                lines.append(f"{start} {end} {mark.lower()}")
                continue
            if self._is_digit_char(mark):
                syl = self.DIGIT_PINYIN.get(mark.strip(), mark)
                lines.append(f"{start} {end} {syl}")
                syl_index = min(syl_index + 1, len(target_syls))
                continue

            clean_mark = re.sub(r"[^\w\u4e00-\u9fa5]+", "", mark)
            if not clean_mark:
                continue
            
            mark_syls = self._segment_to_syllables(clean_mark)
            syl_count = len(mark_syls)
            if syl_count == 0:
                continue
            
            current_syls = target_syls[syl_index: syl_index + syl_count]
            syl_index += syl_count
            
            if not current_syls:
                current_syls = mark_syls

            # 简化：直接按时间比例分配
            if syl_count == 1:
                lines.append(f"{start} {end} {current_syls[0]}")
            else:
                syl_dur = (end - start) // syl_count
                for j, syl in enumerate(current_syls):
                    s = start + j * syl_dur
                    e = start + (j + 1) * syl_dur if j < syl_count - 1 else end
                    lines.append(f"{s} {e} {syl}")

        return lines

    def _process_en_words(
        self,
        word_tier,
        phone_items: List[Tuple[int, int, str]],
        text: str
    ) -> List[str]:
        """处理英文 word tier"""
        lines: List[str] = []
        for interval in word_tier:
            mark = getattr(interval, "mark", getattr(interval, "text", ""))
            mark = (mark or "").strip()
            start = int(interval.minTime * 10000000)
            end = int(interval.maxTime * 10000000)
            if not mark or mark in self.IGNORE_PHONES:
                continue
            if mark in self.SIL_PHONES or mark in ("sp", "spn"):
                lines.append(f"{start} {end} sil")
                continue
            lines.append(f"{start} {end} {mark.lower()}")
        return lines

    def _apply_lab_postprocess(
        self,
        lines: List[str],
        lang: str,
    ) -> List[str]:
        """LAB 后处理"""
        entries = self._parse_lab_lines(lines)
        merged = merge_lab_silence(entries)
        return [f"{s} {e} {p}" for s, e, p in merged]

    @staticmethod
    def _parse_lab_lines(lines: List[str]) -> List[Tuple[int, int, str]]:
        entries: List[Tuple[int, int, str]] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                entries.append((int(parts[0]), int(parts[1]), parts[2]))
            except (ValueError, IndexError):
                continue
        return entries

    def _get_audio_duration(self, audio_path: str) -> int:
        """获取音频时长（100ns 单位）"""
        try:
            import soundfile as sf
            data, sr = sf.read(audio_path)
            duration_seconds = len(data) / sr
            return int(duration_seconds * 10000000)
        except Exception:
            try:
                import wave
                with wave.open(audio_path, "rb") as wav_file:
                    n_frames = wav_file.getnframes()
                    framerate = wav_file.getframerate()
                    duration_seconds = n_frames / framerate
                    return int(duration_seconds * 10000000)
            except Exception as e:
                logger.error(f"获取音频时长失败: {e}")
                return 0

    def process(self, audio_file, text: str, language: str = "cmn") -> Dict:
        """主流程"""
        start_time = time.time()
        try:
            self.temp_dir = tempfile.mkdtemp(prefix="mfa_")
            raw_lang = (language or "cmn").lower().strip()

            text = self._clean_input_text(text)

            if raw_lang in ("cmn", "zh", "zh-cn", "mandarin"):
                lang = "zh"
                text_for_mfa = text.strip()
                phoneme_text = self._text_to_pinyin_notone(text)
            elif raw_lang in ("yue", "zh-yue", "cantonese"):
                lang = "yue"
                text_for_mfa = text.strip()
                phoneme_text = self._text_to_jyutping(text) or text.strip()
            elif raw_lang in ("en", "english", "eng"):
                lang = "en"
                text_for_mfa = text.strip()
                phoneme_text = text.strip()
            else:
                lang = raw_lang
                text_for_mfa = text.strip()
                phoneme_text = text.strip()

            env_ok, env_msg = self._check_language_environment(lang)
            if not env_ok:
                logger.error(env_msg)
                return {
                    "success": False, "error": env_msg,
                    "processing_time": int((time.time() - start_time) * 1000)
                }

            audio_path = os.path.join(self.temp_dir, audio_file.filename)
            audio_file.save(audio_path)

            audio_duration_100ns = self._get_audio_duration(audio_path)
            audio_duration_sec = audio_duration_100ns / 10_000_000

            logger.info(
                f"音频时长: {audio_duration_sec:.1f}s | 语言: {lang} | "
                f"长音频阈值: {self.LONG_AUDIO_THRESHOLD_SEC}s"
            )

            # ★ 改为句子分段处理
            if audio_duration_sec > self.LONG_AUDIO_THRESHOLD_SEC:
                return self._process_with_sentence_segmentation(
                    audio_path, text, lang, text_for_mfa, phoneme_text, start_time
                )
            else:
                return self._process_direct(
                    audio_path, text, lang, text_for_mfa, phoneme_text, start_time
                )

        except subprocess.TimeoutExpired:
            return {
                "success": False, "error": "MFA 超时",
                "processing_time": int((time.time() - start_time) * 1000)
            }
        except FileNotFoundError:
            return {
                "success": False, "error": "MFA 命令不存在",
                "processing_time": int((time.time() - start_time) * 1000)
            }
        except Exception as e:
            logger.error(f"处理错误: {e}", exc_info=True)
            return {
                "success": False, "error": str(e),
                "processing_time": int((time.time() - start_time) * 1000)
            }
        finally:
            self._cleanup()

    def _cleanup(self):
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except Exception as e:
                logger.warning(f"清理失败: {e}")
