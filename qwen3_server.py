# qwen3_server.py
from flask import Flask, request, jsonify
from pathlib import Path
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 项目根目录：当前脚本所在目录
PROJECT_DIR = Path(__file__).resolve().parent

# 缓存固定到当前应用内
CACHE_DIR = PROJECT_DIR / "backend" / "models" / "hf_cache"
HUB_CACHE_DIR = CACHE_DIR / "hub"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
HUB_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 必须在导入 qwen_asr 之前设置
os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["HF_HUB_CACHE"] = str(HUB_CACHE_DIR)

logger.info(f"HF_HOME = {os.environ['HF_HOME']}")
logger.info(f"HF_HUB_CACHE = {os.environ['HF_HUB_CACHE']}")

logger.info("正在初始化 Qwen3-ASR 服务...")

model = None

try:
    import torch
    from qwen_asr import Qwen3ASRModel

    model = Qwen3ASRModel.from_pretrained(
        "Qwen/Qwen3-ASR-1.7B",
        dtype=torch.bfloat16,
        device_map="cuda:0",  # 没有 GPU 就按官方接口改成你可用的后端方案
        max_inference_batch_size=32,
        max_new_tokens=256,
        forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
        forced_aligner_kwargs=dict(
            dtype=torch.bfloat16,
            device_map="cuda:0",
        ),
    )

    logger.info("✅ Qwen3-ASR 模型加载成功！服务已就绪。")
except Exception as e:
    logger.error(f"❌ 模型加载失败: {e}", exc_info=True)

@app.post("/asr")
def asr():
    if model is None:
        return jsonify({"success": False, "error": "模型未加载"}), 500

    try:
        data = request.get_json(force=True)
        audio_path = data.get("audio")
        language = data.get("language")  # 例如: "Chinese" / "English" / None
        context = data.get("context", "")

        if not audio_path or not Path(audio_path).exists():
            return jsonify({"success": False, "error": "音频文件不存在"}), 400

        # 官方 API 参数名是 return_time_stamps
        res = model.transcribe(
            audio=audio_path,
            context=context,
            language=language,
            return_time_stamps=True,
        )

        # res 是一个列表；每个元素含 language / text / time_stamps
        items = []
        for item in res:
            items.append({
                "language": item.language,
                "text": item.text,
                "time_stamps": item.time_stamps,
            })

        return jsonify({
            "success": True,
            "segments": items,
            "raw_text": "".join([x["text"] for x in items]),
        })

    except Exception as e:
        logger.error(f"推理失败: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False)