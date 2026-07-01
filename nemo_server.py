# nemo_server.py
#
# NeMo Forced Aligner (NFA) 独立服务
# https://github.com/NVIDIA-NeMo/Speech/tree/main/tools/nemo_forced_aligner
#
# 与 qwen3_server.py 同样的理由：nemo_toolkit 对 packaging / fsspec /
# omegaconf / hydra-core / lightning 等核心依赖有严格的版本限制，装进主
# Flask 进程所在的 .mfa_env 会跟其它包（比如 pipdeptree 要求的
# packaging>=26）发生版本冲突，把这些包"降级"。所以照搬 Qwen3-ASR 的
# 做法：NeMo 单独装一个 conda/venv 环境，跑成一个本地 HTTP 微服务，
# 主进程（alt_aligners.py 里的 NeMoForcedAligner）只通过 HTTP 调用它，
# 不在 .mfa_env 里 import nemo。
#
# 用法：
#   conda create -n nemo_env python=3.10 -y
#   conda activate nemo_env
#   pip install "nemo_toolkit[asr]>=2.7.0,<2.8.0"
#   python nemo_server.py
#
# 默认监听 127.0.0.1:5002（5001 已被 qwen3_server.py 占用）。
from __future__ import annotations

from flask import Flask, request, jsonify
from pathlib import Path
import os
import logging
import threading
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 项目根目录：nemo_server.py 所在目录（与 qwen3_server.py 约定一致）
PROJECT_DIR = Path(__file__).resolve().parent

# HuggingFace Hub 模型缓存（如 nvidia/stt_zh_citrinet_1024_gamma_0_25、
# nvidia/parakeet-tdt_ctc-0.6b-ja）——独立复用一份缓存目录，不与
# qwen3_server.py 的 HF 缓存共享，避免两个进程同时写同一个 hub 缓存
# 目录产生竞态。
CACHE_DIR = PROJECT_DIR / "backend" / "models" / "nemo_hf_cache"
HUB_CACHE_DIR = CACHE_DIR / "hub"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
HUB_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# NGC pretrained_name（如 stt_en_fastconformer_hybrid_large_pc）的缓存目录
NEMO_CACHE_DIR = PROJECT_DIR / "backend" / "models" / "nemo_cache"
NEMO_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 必须在 import nemo 之前设置
os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["HF_HUB_CACHE"] = str(HUB_CACHE_DIR)
os.environ["NEMO_CACHE_DIR"] = str(NEMO_CACHE_DIR)

# HF_HUB_OFFLINE / HF_ENDPOINT（镜像站）由设置页面统一管理，见 app_settings.py，
# 与 qwen3_server.py / alt_aligners.py 共用同一份配置文件，保持三个进程行为一致。
# 必须在下面 import nemo 相关模块之前完成设置。
try:
    from app_settings import apply_env_from_settings as _apply_hf_env_settings
    _apply_hf_env_settings()
except Exception as _settings_err:
    logger.warning(f"⚠️  读取模型下载设置失败（{_settings_err}），回退到默认联网模式")
    os.environ.setdefault("HF_HUB_OFFLINE", "0")

logger.info(f"HF_HOME = {os.environ['HF_HOME']}")
logger.info(f"HF_HUB_CACHE = {os.environ['HF_HUB_CACHE']}")
logger.info(f"NEMO_CACHE_DIR = {os.environ['NEMO_CACHE_DIR']}")

# ── 各语言默认模型 ───────────────────────────────────────────────────────
# 只收录满足 NFA 限制的官方 checkpoint：纯 CTC 模型，或 CTC 模式下的
# Hybrid CTC-Transducer 模型。纯 Transducer/TDT 模型不能直接用于强制对齐
# （NVIDIA 官方文档明确说明）。
LANGUAGE_MODELS: Dict[str, str] = {
    "en": "stt_en_fastconformer_hybrid_large_pc",       # NGC, Hybrid CTC+RNNT
    "zh": "nvidia/stt_zh_citrinet_1024_gamma_0_25",     # HF Hub, 纯 CTC（字符级）
    "ja": "nvidia/parakeet-tdt_ctc-0.6b-ja",             # HF Hub, Hybrid TDT+CTC
}
# 没有官方 CTC/Hybrid-CTC checkpoint 的语言：不提供默认模型，可通过请求体
# 的 "model" 字段，或环境变量 NEMO_FA_MODEL_{LANG} 自行指定。
_NO_DEFAULT_MODEL_LANGS = frozenset({"ko", "yue"})

_models: Dict[str, Any] = {}          # model_name -> 已加载的 NeMo 模型实例
_model_lock = threading.Lock()
_model_device: str = "auto"           # 记录当前所有已加载模型使用的 device_override


def _pick_device(device_override: str = "auto") -> str:
    """
    与 qwen3_server.py 的 _pick_device_and_dtype 同样的探测逻辑，但 NeMo
    模型走 fp32/AMP 自动管理，这里只需要决定 "cpu" 还是 "cuda"。
    """
    import torch

    if device_override == "cpu":
        return "cpu"
    if not torch.cuda.is_available():
        if device_override == "cuda":
            logger.warning("⚠️  请求 CUDA 但未检测到可用 GPU，回退到 CPU")
        return "cpu"
    try:
        torch.zeros(1, device="cuda")
    except Exception as e:
        logger.warning(f"⚠️  CUDA 初始化失败（{e}），回退到 CPU")
        return "cpu"
    return "cuda"


def _resolve_model_name(language: str, model_override: str = "") -> str:
    """按优先级解析本次请求使用的模型名：请求体 > 环境变量 > 内置默认表。"""
    int_lang = (language or "en").strip().lower()
    # 兼容 ISO 639-2/3 三字码（cmn/eng/jpn/...）传入时的归一化，与
    # alt_aligners.py 中 _normalize_lang() 的映射保持一致
    _alias = {"cmn": "zh", "eng": "en", "jpn": "ja", "kor": "ko"}
    int_lang = _alias.get(int_lang, int_lang)

    if model_override:
        return model_override

    env_key = f"NEMO_FA_MODEL_{int_lang.upper()}"
    env_val = os.environ.get(env_key, "").strip()
    if env_val:
        logger.info(f"使用环境变量 {env_key}={env_val}")
        return env_val

    model_name = LANGUAGE_MODELS.get(int_lang)
    if not model_name:
        if int_lang in _NO_DEFAULT_MODEL_LANGS:
            raise ValueError(
                f"NeMo Forced Aligner 暂无语言 '{int_lang}' 的官方 CTC/Hybrid-CTC "
                f"模型。可在请求体传 'model' 字段指定自有模型，或设置环境变量 "
                f"NEMO_FA_MODEL_{int_lang.upper()}。"
            )
        logger.warning(f"语言 '{int_lang}' 不在内置表中，回退英语模型")
        return LANGUAGE_MODELS["en"]
    return model_name


def load_model(model_name: str, device_override: str = "auto"):
    """惰性加载并缓存 NeMo ASR 模型，按 (model_name, device) 缓存。"""
    global _model_device
    cache_key = f"{model_name}@{device_override}"

    with _model_lock:
        if cache_key in _models:
            return _models[cache_key]

        logger.info(f"正在加载 NeMo 模型: {model_name} (device={device_override}) ...")
        device = _pick_device(device_override)

        import nemo.collections.asr as nemo_asr

        # 临时解除 HF_HUB_OFFLINE 限制以允许下载 HF Hub 托管模型
        # （nvidia/stt_zh_xxx、nvidia/parakeet-tdt_ctc-xxx-ja 等）
        _hf_offline = os.environ.pop("HF_HUB_OFFLINE", None)
        try:
            model = nemo_asr.models.ASRModel.from_pretrained(
                model_name=model_name,
                map_location=device,
            )
        finally:
            if _hf_offline is not None:
                os.environ["HF_HUB_OFFLINE"] = _hf_offline

        # Hybrid CTC-Transducer 模型默认解码器是 RNNT/TDT，NFA 要求强制
        # 切到 CTC 模式才能拿到逐帧 log-probs。纯 CTC 模型没有这个方法。
        if hasattr(model, "change_decoding_strategy"):
            try:
                model.change_decoding_strategy(decoder_type="ctc")
                logger.info("Hybrid 模型已切换至 CTC 解码模式")
            except Exception as switch_err:
                logger.debug(f"change_decoding_strategy 跳过: {switch_err}")

        model = model.to(device)
        model.eval()

        try:
            from nemo.utils import logging as nemo_logging
            nemo_logging.setLevel(logging.WARNING)
        except Exception:
            pass

        _models[cache_key] = model
        _model_device = device_override
        logger.info(f"✅ NeMo 模型加载成功: {model_name} → {device}")
        return model


# ── CTC 辅助函数（与 alt_aligners.py 中曾经的进程内实现逻辑一致，现在搬到
#    这个独立进程里执行）────────────────────────────────────────────────
def _get_blank_id(model) -> int:
    decoder = getattr(model, "decoder", None)
    vocab = getattr(decoder, "vocabulary", None)
    if vocab is not None:
        return len(vocab)
    try:
        return int(model.decoder.num_classes_with_blank) - 1
    except Exception:
        return 0


def _tokenize(model, text: str) -> Tuple[List[int], List[str]]:
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "text_to_ids"):
        ids = tokenizer.text_to_ids(text)
        if ids:
            texts: List[str] = []
            if hasattr(tokenizer, "ids_to_tokens"):
                try:
                    texts = [str(t) for t in tokenizer.ids_to_tokens(ids)]
                except Exception:
                    texts = []
            if len(texts) != len(ids) and hasattr(tokenizer, "id_to_piece"):
                try:
                    texts = [str(tokenizer.id_to_piece(i)) for i in ids]
                except Exception:
                    texts = []
            if len(texts) != len(ids):
                texts = list(text)[: len(ids)] + [""] * max(0, len(ids) - len(text))
            return list(ids), texts

    decoder = getattr(model, "decoder", None)
    vocab = getattr(decoder, "vocabulary", None)
    if vocab:
        vocab_idx = {ch: i for i, ch in enumerate(vocab)}
        ids, texts = [], []
        for ch in text:
            idx = vocab_idx.get(ch)
            if idx is None and ch == " ":
                idx = vocab_idx.get("<space>")
            if idx is not None:
                ids.append(idx)
                texts.append(ch)
        return ids, texts

    return [], []


def _get_log_probs(model, audio_path: str, device: str) -> Tuple["Any", int, float]:
    import torch
    import soundfile as sf

    audio_np, sr = sf.read(audio_path, always_2d=False)
    if getattr(audio_np, "ndim", 1) > 1:
        audio_np = audio_np.mean(axis=1)
    audio_np = audio_np.astype("float32")

    if sr != 16000:
        import torchaudio
        t = torch.from_numpy(audio_np).unsqueeze(0)
        audio_np = torchaudio.functional.resample(t, orig_freq=sr, new_freq=16000).squeeze(0).numpy()
        sr = 16000

    audio_sec = len(audio_np) / sr
    audio_tensor = torch.from_numpy(audio_np).unsqueeze(0).to(device)
    audio_len = torch.tensor([audio_tensor.shape[1]], dtype=torch.long, device=device)

    with torch.no_grad():
        try:
            log_probs, enc_len, _ = model(
                input_signal=audio_tensor, input_signal_length=audio_len,
            )
        except TypeError:
            log_probs, enc_len, _ = model(audio_tensor, audio_len)

    T = int(enc_len[0].item())
    lp = log_probs[0, :T, :].detach().to("cpu").float()
    return lp, T, audio_sec


def _merge_bpe_to_words(
    token_entries: List[Tuple[float, float, str]],
) -> List[Tuple[float, float, str]]:
    words: List[Tuple[float, float, str]] = []
    cur_start: Optional[float] = None
    cur_end: Optional[float] = None
    cur_text = ""

    for s, e, tok in token_entries:
        is_word_start = tok.startswith("▁") or tok.startswith(" ") or not cur_text
        clean_tok = tok.lstrip("▁ ")
        if not clean_tok:
            continue
        if is_word_start:
            if cur_text:
                words.append((cur_start, cur_end, cur_text))
            cur_start, cur_end, cur_text = s, e, clean_tok
        else:
            cur_text += clean_tok
            cur_end = e

    if cur_text:
        words.append((cur_start, cur_end, cur_text))
    return words


@app.get("/")
def health():
    return jsonify(
        {
            "success": True,
            "message": "NeMo Forced Aligner service is running",
            "models_loaded": list(_models.keys()),
            "language_models": LANGUAGE_MODELS,
        }
    )


@app.post("/align")
def align():
    """
    请求体:
      {
        "audio": "本机绝对路径",
        "text": "参考文本（必填，NFA 是强制对齐，不做纯 ASR）",
        "language": "en" / "zh" / "ja" / ...,
        "model": "可选，覆盖默认模型名（NGC 名 或 'nvidia/xxx' HF 名）",
        "device": "auto" | "cpu" | "cuda"
      }

    返回:
      {
        "success": true,
        "token_entries": [[start_sec, end_sec, token_text], ...],
        "model": "实际使用的模型名",
        "audio_duration_sec": ...
      }
    客户端（alt_aligners.py 的 NeMoForcedAligner）拿到 token_entries 后，
    自己做英语 BPE 词合并 + _word_entries_to_lab() 生成最终 LAB，
    与 Qwen3ASRAligner 处理 segments 的方式一致。
    """
    data = request.get_json(force=True) or {}

    audio_path = str(data.get("audio") or "")
    text = (data.get("text") or "").strip()
    language = data.get("language") or "en"
    model_override = (data.get("model") or "").strip()
    device_override = data.get("device", "auto")
    if device_override not in ("auto", "cpu", "cuda"):
        device_override = "auto"

    if not audio_path or not Path(audio_path).exists():
        return jsonify({"success": False, "error": "音频文件不存在或未提供 audio 参数"}), 400
    if not text:
        return jsonify({"success": False, "error": "NeMo Forced Aligner 需要提供参考文本"}), 400

    try:
        model_name = _resolve_model_name(language, model_override)
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400

    try:
        model = load_model(model_name, device_override)
    except Exception as e:
        logger.error(f"模型加载失败: {e}", exc_info=True)
        return jsonify({"success": False, "error": f"模型加载失败: {e}"}), 500

    try:
        import torch
        import torchaudio

        device = _pick_device(device_override)
        log_probs, T, audio_sec = _get_log_probs(model, audio_path, device)
        if T == 0:
            return jsonify({"success": False, "error": "模型未返回任何编码帧，请检查音频文件"}), 500

        token_ids, token_texts = _tokenize(model, text)
        if not token_ids:
            return jsonify({
                "success": False,
                "error": f"文本 tokenization 结果为空，请确认参考文本与模型语言匹配（当前模型: {model_name}）",
            }), 400

        blank_id = _get_blank_id(model)
        frame_sec = audio_sec / T

        logger.info(
            f"[NeMo-FA] 对齐: T={T} frames, {len(token_ids)} tokens, "
            f"blank_id={blank_id}, frame={frame_sec * 1000:.1f}ms, model={model_name}"
        )

        emission = log_probs.unsqueeze(0)
        targets = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)

        spans = None
        try:
            aligned, scores = torchaudio.functional.forced_align(emission, targets, blank=blank_id)
            spans = torchaudio.functional.merge_tokens(aligned[0], scores[0], blank=blank_id)
        except Exception as fa_err:
            logger.warning(f"torchaudio.forced_align 失败: {fa_err}，回退到均匀时间分配")

        if spans is not None:
            # 【修正】上一版补丁假设 spans 里会混入 token==blank_id 的独立
            # blank span，并据此做"向左合并"。但按 torchaudio 官方实现
            # （torchaudio/functional/_alignment.py::merge_tokens()）：
            #     spans = [TokenSpan(token=token, start=start, end=end, ...)
            #              for start, end in zip(changes_wo_blank[:-1], changes_wo_blank[1:])
            #              if (token := tokens[start]) != blank]
            # blank token 在这一步就已经被无条件剔除，merge_tokens() 返回的
            # spans 列表里**根本不存在** token==blank_id 的条目。也就是说
            # `if is_blank_span:` 分支永远不会被命中，等于没有修复任何东西
            # ——这正是用户反馈"仍然没有合并"的真实原因。
            #
            # blank 真正的"藏身之处"不是某个独立的 span，而是相邻两个真实
            # span 在帧轴上的不连续：spans[i+1].start 帧号会大于
            # spans[i].end 帧号，中间那段缺口正是被 merge_tokens() 直接
            # 丢弃的 blank 帧。原先的写法对此视而不见——entry 的
            # t_start/t_end 严格等于 span.start/span.end 换算的秒数，于是
            # 这段被丢弃的帧区间，就变成了 token_entries 序列里一段
            # "无主"的真实时间空隙。这段空隙在数值上没有消失（两侧时间戳
            # 仍然对得上原始音频），但下游 _fill_silences_lab() 一旦扫到
            # 任何 ≥ 50ms 的空隙就会在那里插入一条 SIL——而 NeMo citrinet
            # 等模型的单帧时长（本例中约 79ms）本身就已经超过这个 50ms
            # 阈值，导致几乎每一个相邻字符之间的正常帧量化间隙，都被
            # 错误地当成"真实停顿"转成了 SIL，SVP 里因此出现"每个字之间
            # 都被强行隔开"的现象。
            #
            # 正确修复：按 span 在帧轴上的真实位置检测相邻 span 间的帧缺口，
            # 一旦发现就把这段时长整体并入前一个真实 token 的结尾（向左
            # 合并，而不是放任它变成游离空隙）。音频开头第一个 token 之前
            # 的空隙不在此处处理，交给 _fill_silences_lab() 的"首条目"
            # 判断逻辑负责（开头静音本来就该是 SIL）。
            token_entries: List[Tuple[float, float, str]] = []
            prev_span_end_frame: Optional[float] = None
            for i, span in enumerate(spans):
                if i >= len(token_texts):
                    break
                tok_txt = token_texts[i]

                # 向左合并：当前 span 起始帧若晚于上一个 span 的结束帧，
                # 中间这段就是被 merge_tokens() 抹掉的 blank 帧，整体
                # 计入"上一个已生成的 token_entries 条目"的结尾。
                if prev_span_end_frame is not None and span.start > prev_span_end_frame:
                    if token_entries:
                        merge_end = min(span.start * frame_sec, audio_sec)
                        prev_s, prev_e, prev_tok = token_entries[-1]
                        token_entries[-1] = (prev_s, max(prev_e, merge_end), prev_tok)

                t_start = span.start * frame_sec
                t_end = min(span.end * frame_sec, audio_sec)
                if t_end > t_start and tok_txt:
                    token_entries.append((t_start, t_end, tok_txt))

                prev_span_end_frame = span.end
        else:
            spoken = [t for t in token_texts if (t or "").strip()]
            if spoken:
                dur = audio_sec / len(spoken)
                token_entries = [
                    (i * dur, min((i + 1) * dur, audio_sec), t)
                    for i, t in enumerate(spoken)
                ]
            else:
                token_entries = []

        if not token_entries:
            return jsonify({"success": False, "error": "强制对齐未产生任何时间戳条目"}), 500

        int_lang = {"cmn": "zh", "eng": "en", "jpn": "ja", "kor": "ko"}.get(
            (language or "en").strip().lower(), (language or "en").strip().lower()
        )
        if int_lang == "en":
            has_bpe_marker = any(
                (t.startswith("▁") or t.startswith(" ")) for _, _, t in token_entries
            )
            if has_bpe_marker:
                token_entries = _merge_bpe_to_words(token_entries)

        return jsonify(
            {
                "success": True,
                "token_entries": [[s, e, t] for s, e, t in token_entries],
                "model": model_name,
                "audio_duration_sec": audio_sec,
            }
        )

    except Exception as e:
        logger.error(f"对齐失败: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    # 生产环境建议改成 waitress / gevent / gunicorn，与 qwen3_server.py 一致
    app.run(host="127.0.0.1", port=5002, debug=False)
