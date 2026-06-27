# qwen3_server.py
from __future__ import annotations

from flask import Flask, request, jsonify
from pathlib import Path
import os
import logging
import threading
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 项目根目录：qwen3_server.py 所在目录
PROJECT_DIR = Path(__file__).resolve().parent

# 缓存固定到当前应用内
CACHE_DIR = PROJECT_DIR / "backend" / "models" / "hf_cache"
HUB_CACHE_DIR = CACHE_DIR / "hub"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
HUB_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 必须在导入 qwen_asr / transformers 相关包之前设置
os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["HF_HUB_CACHE"] = str(HUB_CACHE_DIR)

# 👇 新增这一行
# os.environ["HF_HUB_OFFLINE"] = "1"

logger.info(f"HF_HOME = {os.environ['HF_HOME']}")
logger.info(f"HF_HUB_CACHE = {os.environ['HF_HUB_CACHE']}")

MODEL_ID = "Qwen/Qwen3-ASR-1.7B"
FORCED_ALIGNER_ID = "Qwen/Qwen3-ForcedAligner-0.6B"

_model = None
_model_lock = threading.Lock()


def _pick_device_and_dtype(device_override: str = "auto"):
    """
    根据设备参数和实际 GPU 架构选择运行设备与数据类型。

    dtype 选择策略（避免在不支持的 GPU 上使用错误精度）：
      - bfloat16：需要 Ampere (CC ≥ 8.0，RTX 30xx / A100+)
      - float16 ：Pascal (CC 6.x) / Volta (CC 7.0) / Turing (CC 7.5) 均支持
      - float32 ：CPU 或无法确定 GPU 能力时的保底选项

    P106-100 (Pascal, CC 6.1) → float16（不是 bfloat16！）
    """
    import torch

    # 强制 CPU
    if device_override == "cpu":
        return "cpu", torch.float32

    if not torch.cuda.is_available():
        if device_override == "cuda":
            logger.warning("⚠️  请求 CUDA 但未检测到可用 GPU，回退到 CPU")
        return "cpu", torch.float32

    # CUDA smoke-test：防止 CPU-only PyTorch 版本误报 is_available()
    try:
        torch.zeros(1, device="cuda")
    except Exception as e:
        logger.warning(f"⚠️  CUDA 初始化失败（{e}），回退到 CPU")
        return "cpu", torch.float32

    device = "cuda:0"
    try:
        props = torch.cuda.get_device_properties(0)
        cc_major = props.major
        logger.info(f"GPU: {props.name}  compute capability: {cc_major}.{props.minor}")

        if cc_major >= 8:
            # Ampere / Ada Lovelace / Hopper → bfloat16（训练稳定性更好）
            dtype = torch.bfloat16
        elif cc_major >= 6:
            # Pascal / Volta / Turing → float16（bfloat16 硬件不支持）
            dtype = torch.float16
        else:
            # 超旧卡保底
            dtype = torch.float32

        logger.info(f"自动选择 dtype: {dtype}")
        return device, dtype
    except Exception as e:
        logger.warning(f"⚠️  GPU 能力查询失败（{e}），使用 float32 保底")
        return device, torch.float32


def _normalize_time_stamps(value: Any) -> List[List[Optional[float]]]:
    """
    尽量把不同返回形态统一成 [[start, end], ...]
    """
    if value is None:
        return []

    # 形如 [(s, e), (s, e)]
    if isinstance(value, list) and value and isinstance(value[0], (list, tuple)):
        out: List[List[Optional[float]]] = []
        for item in value:
            if len(item) >= 2:
                out.append([item[0], item[1]])
        return out

    # 形如 {"start": s, "end": e}
    if isinstance(value, dict):
        s = value.get("start")
        e = value.get("end")
        if s is not None or e is not None:
            return [[s, e]]
        return []

    return []


def _normalize_segments(result: Any) -> List[Dict[str, Any]]:
    """
    将 qwen_asr 的返回结果统一成客户端容易消费的格式：
    [
      {
        "language": "...",
        "text": "...",
        "time_stamps": [[s, e], ...]
      }
    ]
    """
    segments: List[Dict[str, Any]] = []

    if result is None:
        return segments

    # 1) 如果是 list
    if isinstance(result, list):
        for item in result:
            if isinstance(item, dict):
                text = (item.get("text") or "").strip()
                lang = item.get("language")
                ts = item.get("time_stamps")
                if ts is None:
                    ts = item.get("timestamp")
                segments.append(
                    {
                        "language": lang,
                        "text": text,
                        "time_stamps": _normalize_time_stamps(ts),
                    }
                )
            else:
                # 兼容对象形式
                text = (getattr(item, "text", "") or "").strip()
                lang = getattr(item, "language", None)
                ts = getattr(item, "time_stamps", None)
                if ts is None:
                    ts = getattr(item, "timestamp", None)
                segments.append(
                    {
                        "language": lang,
                        "text": text,
                        "time_stamps": _normalize_time_stamps(ts),
                    }
                )
        return segments

    # 2) 如果是 dict
    if isinstance(result, dict):
        # 常见情况：直接给一个整体结果
        text = (result.get("text") or result.get("raw_text") or "").strip()
        lang = result.get("language")
        ts = result.get("time_stamps")
        if ts is None:
            ts = result.get("timestamp")

        # 可能本身就带 chunks / segments
        if "segments" in result and isinstance(result["segments"], list):
            for seg in result["segments"]:
                if isinstance(seg, dict):
                    segments.append(
                        {
                            "language": seg.get("language", lang),
                            "text": (seg.get("text") or "").strip(),
                            "time_stamps": _normalize_time_stamps(
                                seg.get("time_stamps", seg.get("timestamp"))
                            ),
                        }
                    )
            if segments:
                return segments

        if "chunks" in result and isinstance(result["chunks"], list):
            for ch in result["chunks"]:
                if isinstance(ch, dict):
                    segments.append(
                        {
                            "language": ch.get("language", lang),
                            "text": (ch.get("text") or "").strip(),
                            "time_stamps": _normalize_time_stamps(
                                ch.get("time_stamps", ch.get("timestamp"))
                            ),
                        }
                    )
            if segments:
                return segments

        segments.append(
            {
                "language": lang,
                "text": text,
                "time_stamps": _normalize_time_stamps(ts),
            }
        )
        return segments

    # 3) 其他对象
    text = (getattr(result, "text", "") or "").strip()
    lang = getattr(result, "language", None)
    ts = getattr(result, "time_stamps", None)
    if ts is None:
        ts = getattr(result, "timestamp", None)

    segments.append(
        {
            "language": lang,
            "text": text,
            "time_stamps": _normalize_time_stamps(ts),
        }
    )
    return segments


_model_device: str = "auto"   # 记录当前模型加载时所用的 device_override


def load_model(device_override: str = "auto"):
    global _model, _model_device

    with _model_lock:
        # 如果已有模型且设备未变化，直接复用
        if _model is not None and _model_device == device_override:
            return _model

        # 设备变化或首次加载
        if _model is not None:
            logger.info(f"设备从 '{_model_device}' 切换到 '{device_override}'，重新加载模型...")
            _model = None

        logger.info("正在初始化 Qwen3-ASR 服务...")
        device_map, dtype = _pick_device_and_dtype(device_override)
        logger.info(f"使用设备: {device_map}, dtype: {dtype}")

        try:
            import torch
            from qwen_asr import Qwen3ASRModel

            kwargs = {
                "dtype": dtype,
                "device_map": device_map,
                "max_inference_batch_size": 8 if device_map.startswith("cuda") else 1,
                "max_new_tokens": 256,
            }

            # 统一启用 forced aligner，这样客户端更容易拿到时间戳
            kwargs["forced_aligner"] = FORCED_ALIGNER_ID
            kwargs["forced_aligner_kwargs"] = {
                "dtype": dtype,
                "device_map": device_map,
            }

            _model = Qwen3ASRModel.from_pretrained(MODEL_ID, **kwargs)
            _model_device = device_override
            logger.info("✅ Qwen3-ASR 模型加载成功！服务已就绪。")
            return _model

        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}", exc_info=True)
            _model = None
            return None


@app.get("/")
def health():
    return jsonify(
        {
            "success": True,
            "message": "Qwen3-ASR service is running",
            "model_loaded": _model is not None,
            "model_id": MODEL_ID,
        }
    )


@app.post("/asr")
def asr():
    data = request.get_json(force=True) or {}

    # 客户端可传 "device": "auto"|"cpu"|"cuda" 控制运行设备
    device_override = data.get("device", "auto")
    if device_override not in ("auto", "cpu", "cuda"):
        device_override = "auto"

    model = load_model(device_override)
    if model is None:
        return jsonify({"success": False, "error": "模型未加载"}), 500

    try:
        audio_path = data.get("audio")
        language = data.get("language")
        context = data.get("context", "")

        if not audio_path:
            return jsonify({"success": False, "error": "缺少 audio 参数"}), 400

        audio_path = str(audio_path)
        if not Path(audio_path).exists():
            return jsonify({"success": False, "error": "音频文件不存在"}), 400

        # qwen_asr 官方接口：transcribe
        # return_time_stamps=True 便于客户端构建 LAB
        result = model.transcribe(
            audio=audio_path,
            language=language,
            context=context,
            return_time_stamps=True,
        )

        segments = _normalize_segments(result)
        raw_text = "".join([seg.get("text", "") for seg in segments]).strip()

        return jsonify(
            {
                "success": True,
                "segments": segments,
                "raw_text": raw_text,
                "model_id": MODEL_ID,
            }
        )

    except Exception as e:
        logger.error(f"推理失败: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    # 生产环境建议改成 waitress / gevent / gunicorn
    app.run(host="127.0.0.1", port=5001, debug=False)