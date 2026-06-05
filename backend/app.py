# -*- coding: utf-8 -*-
import os
import re
import sys
import uuid
import logging
import webbrowser
from threading import Thread
from time import sleep
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory, abort
from flask_cors import CORS

from mfa_utils import MFAChecker
from mfa_processor import MFAProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIST = (BASE_DIR.parent / "frontend" / "dist").resolve()
WORK_DIR = (BASE_DIR / "work").resolve()

WINDOWS_SAFE_PATH_LIMIT = 248
WORK_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, static_folder=str(FRONTEND_DIST), static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024
CORS(app, supports_credentials=True)

mfa_processor = MFAProcessor()


def abs_norm(path: str) -> str:
    return os.path.abspath(os.path.normpath(path))


def path_len(path: str) -> int:
    return len(abs_norm(path))


def sanitize_stem(name: str) -> str:
    stem = os.path.splitext(os.path.basename(name))[0]
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", stem)
    stem = re.sub(r"\s+", "_", stem).strip("._ ")
    return stem or uuid.uuid4().hex[:12]


def fit_stem_to_limit(base_dir: str, stem: str, limit: int = WINDOWS_SAFE_PATH_LIMIT) -> str:
    base_abs = abs_norm(base_dir)
    fixed_overhead = len(base_abs) + 1 + 4
    max_stem_len = limit - fixed_overhead
    if max_stem_len < 8:
        raise ValueError("工作目录太深，无法在 248 字符限制内保存文件。")

    if len(stem) > max_stem_len:
        stem = stem[:max_stem_len].rstrip("._ ")
    return stem or uuid.uuid4().hex[:12]


def build_job_paths(original_filename: str):
    stem = sanitize_stem(original_filename)
    stem = fit_stem_to_limit(str(WORK_DIR), stem)

    wav_path = WORK_DIR / f"{stem}.wav"
    lab_path = WORK_DIR / f"{stem}.lab"

    if path_len(str(wav_path)) > WINDOWS_SAFE_PATH_LIMIT or path_len(str(lab_path)) > WINDOWS_SAFE_PATH_LIMIT:
        raise ValueError("生成后的文件路径仍然超过 248 字符，请把项目目录放得更浅一些。")

    return stem, wav_path, lab_path


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    if path.startswith("api/"):
        abort(404)

    full_path = (FRONTEND_DIST / path).resolve()
    if path and full_path.is_file():
        return send_from_directory(str(FRONTEND_DIST), path)

    index_path = FRONTEND_DIST / "index.html"
    if index_path.is_file():
        return send_from_directory(str(FRONTEND_DIST), "index.html")

    return jsonify({
        "error": "前端文件未找到",
        "message": "请在 frontend/ 目录下执行 npm install && npm run build",
        "expected_dir": str(FRONTEND_DIST)
    }), 404


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "version": "1.0.0",
        "app": "GPT-SOVITS MFA Aligner"
    }), 200


@app.route("/api/debug/runtime", methods=["GET"])
def debug_runtime():
    return jsonify({
        "python_executable": sys.executable,
        "python_version": sys.version,
        "conda_prefix": os.environ.get("CONDA_PREFIX", ""),
        "env_dir": str(MFAChecker.env_dir()),
    }), 200


@app.route("/api/mfa/status", methods=["GET"])
def mfa_status():
    return jsonify(MFAChecker.get_status()), 200


@app.route("/api/upload", methods=["POST"])
def upload_wav_and_text():
    try:
        if "audio_file" not in request.files:
            return jsonify({"error": "缺少 audio_file"}), 400
        if "text" not in request.form:
            return jsonify({"error": "缺少 text"}), 400

        audio_file = request.files["audio_file"]
        text = request.form.get("text", "").strip()

        if not audio_file or not text:
            return jsonify({"error": "输入无效"}), 400

        stem, wav_path, lab_path = build_job_paths(audio_file.filename or "audio.wav")

        audio_file.save(str(wav_path))
        lab_path.write_text(text + "\n", encoding="utf-8")

        return jsonify({
            "success": True,
            "stem": stem,
            "wav_path": str(wav_path),
            "lab_path": str(lab_path),
            "lab": text,
            "lab_content": text,
            "message": "已保存同名 wav / lab"
        }), 200

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error("上传保存失败: %s", e, exc_info=True)
        return jsonify({"error": f"保存失败: {str(e)}"}), 500


@app.route("/api/mfa/process", methods=["POST"])
def process_mfa():
    try:
        if "audio_file" not in request.files:
            return jsonify({"error": "缺少 audio_file"}), 400
        if "text" not in request.form:
            return jsonify({"error": "缺少 text"}), 400

        audio_file = request.files["audio_file"]
        text = request.form.get("text", "").strip()
        language = request.form.get("language", "cmn")

        if not audio_file or not text:
            return jsonify({"error": "输入无效"}), 400

        result = mfa_processor.process(audio_file, text, language)

        if result.get("success"):
            return jsonify(result), 200

        return jsonify({
            "success": False,
            "error": result.get("error", "处理失败"),
            "processing_time_ms": result.get("processing_time", 0)
        }), 500

    except Exception as e:
        logger.error("处理错误: %s", e, exc_info=True)
        return jsonify({"error": f"处理出错: {str(e)}"}), 500


@app.route("/api/mfa/download-model/<language>", methods=["POST"])
def download_mfa_model(language: str):
    try:
        valid_languages = ["cmn", "zh", "eng", "jpn", "kor", "yue"]
        if language not in valid_languages:
            return jsonify({"error": f"不支持的语言: {language}"}), 400

        success, message = MFAChecker.download_model(language)

        if success:
            return jsonify({"success": True, "message": message}), 200
        return jsonify({"success": False, "error": message}), 400

    except Exception as e:
        logger.error("下载模型错误: %s", e, exc_info=True)
        return jsonify({"error": f"下载失败: {str(e)}"}), 500


def open_browser(host: str, port: int):
    sleep(2)
    webbrowser.open(f"http://{host}:{port}")


def main(host: str = "127.0.0.1", port: int = 5000):
    print(f"\n{'=' * 50}")
    print("🚀 启动 GPT-SOVITS MFA Aligner")
    print(f"📍 访问地址: http://{host}:{port}")
    print(f"📂 工作目录: {WORK_DIR}")
    print(f"📂 前端目录: {FRONTEND_DIST}")
    print(f"⏹️  按 Ctrl+C 停止服务")
    print(f"{'=' * 50}\n")

    Thread(target=open_browser, args=(host, port), daemon=True).start()
    app.run(host=host, port=port, debug=True, use_reloader=False)


if __name__ == "__main__":
    main()