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
from pipeline import AudioProcessingPipeline

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
pipeline = AudioProcessingPipeline(str(WORK_DIR))
from threading import Thread, Lock
from datetime import datetime

JOB_LOCK = Lock()
JOBS = {}

def set_job(job_id: str, **kwargs):
    with JOB_LOCK:
        JOBS[job_id] = {
            **JOBS.get(job_id, {}),
            **kwargs,
        }


def get_job(job_id: str):
    with JOB_LOCK:
        return JOBS.get(job_id)

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
    
    # 【优化】为文件名注入 6 位随机标识符，彻底避免连续点击时发生文件覆盖/锁死冲突
    unique_suffix = uuid.uuid4().hex[:6]
    stem = f"{stem}_{unique_suffix}"
    
    stem = fit_stem_to_limit(str(WORK_DIR), stem)

    wav_path = WORK_DIR / f"{stem}.wav"
    lab_path = WORK_DIR / f"{stem}.lab"

    if path_len(str(wav_path)) > WINDOWS_SAFE_PATH_LIMIT or path_len(str(lab_path)) > WINDOWS_SAFE_PATH_LIMIT:
        raise ValueError("生成后的文件路径仍然超过 248 字符，请把项目目录放得更浅一些。")

    return stem, wav_path, lab_path

@app.after_request
def disable_keepalive(response):
    """
    强制告诉浏览器关闭当前连接，不复用 TCP Socket。
    彻底解决 Werkzeug 开发服务器在连续上传大文件时，因 Keep-Alive 复用导致的 Connection Reset (Failed to fetch) 问题。
    """
    response.headers["Connection"] = "close"
    return response

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
        "version": "2.0.0",
        "app": "Audio Processing Aligner with MFA + PyWORLD"
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


@app.route("/api/pipeline/status", methods=["GET"])
def pipeline_status():
    try:
        status = pipeline.get_status()
        return jsonify({
            "success": True,
            "status": status
        }), 200
    except Exception as e:
        logger.error(f"查询状态失败: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# 新增
@app.route("/api/pipeline/job/<job_id>", methods=["GET"])
def pipeline_job_status(job_id):
    job = get_job(job_id)

    if not job:
        return jsonify({
            "success": False,
            "error": "任务不存在"
        }), 404

    return jsonify({
        "success": True,
        "job": job
    }), 200


@app.route("/api/pipeline/formats", methods=["GET"])
def pipeline_formats():
    """获取支持的输出格式"""
    try:
        formats = pipeline.get_supported_formats()
        return jsonify({
            "success": True,
            "formats": formats
        }), 200
    except Exception as e:
        logger.error(f"查询格式失败: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/upload", methods=["POST"])
def upload_wav_and_text():
    """上传音频和文本（仅保存，不处理）"""
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
    """MFA 自动标注处理"""
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

def run_pipeline_job(
    job_id: str,
    wav_path: str,
    text: str,
    language: str,
    output_format: str,
    project_title: str,
    bpm: float,
    base_pitch: int,
    f0_method: str,
    f0_smooth: bool,
    f0_smooth_window: int,
    use_double_precision: bool,
    f0_floor: float,
    f0_ceil: float,
    refine_pitch: bool,
    export_pitch_line: bool,
):
    try:
        set_job(
            job_id,
            status="running",
            started_at=datetime.now().isoformat(),
        )

        # === 【新增：伪造 FileStorage 包装器】 ===
        class FileStorageWrapper:
            def __init__(self, local_path):
                self.path = os.path.abspath(local_path)
                self.filename = os.path.basename(local_path)

            def save(self, dst):
                # 如果目标路径和当前文件路径不同，则执行复制
                import shutil
                if os.path.abspath(dst) != self.path:
                    shutil.copy(self.path, dst)

            def seek(self, *args, **kwargs):
                # 兼容 pipeline.py 里的 audio_file.seek(0)
                pass

        # 将路径字符串包装为兼容的对象
        compat_audio_file = FileStorageWrapper(wav_path)
        # ========================================

        result = pipeline.process_full(
            audio_file=compat_audio_file,  # 这里改为传入包装后的对象
            text=text,
            language=language,
            output_format=output_format,
            project_title=project_title,
            bpm=bpm,
            base_pitch=base_pitch,
            f0_method=f0_method,
            f0_smooth=f0_smooth,
            f0_smooth_window=f0_smooth_window,
            use_double_precision=use_double_precision,
            f0_floor=f0_floor,
            f0_ceil=f0_ceil,
            refine_pitch=refine_pitch,
            export_pitch_line=export_pitch_line,
        )

        if result.get("success"):
            set_job(
                job_id,
                status="done",
                finished_at=datetime.now().isoformat(),
                result=result,
            )
        else:
            set_job(
                job_id,
                status="failed",
                finished_at=datetime.now().isoformat(),
                error=result.get("error"),
                result=result,
            )

    except Exception as e:
        logger.exception("后台任务异常")

        set_job(
            job_id,
            status="failed",
            finished_at=datetime.now().isoformat(),
            error=str(e),
        )


@app.route("/api/pipeline/full", methods=["POST"])
def pipeline_full_process():
    """
    MFA + F0 + 工程文件生成
    异步后台任务版
    """
    try:
        if "audio_file" not in request.files:
            return jsonify({"error": "缺少 audio_file"}), 400

        if "text" not in request.form:
            return jsonify({"error": "缺少 text"}), 400

        audio_file = request.files["audio_file"]
        text = request.form.get("text", "").strip()

        language = request.form.get("language", "cmn")
        output_format = request.form.get("format", "sv")
        project_title = request.form.get("title", "Project")

        bpm = float(request.form.get("bpm", 120))
        base_pitch = int(request.form.get("base_pitch", 60))

        f0_method = request.form.get("f0_method", "dio")
        f0_smooth = request.form.get("f0_smooth", "true").lower() == "true"

        f0_smooth_window = int(
            request.form.get("f0_smooth_window", 5)
        )

        use_double_precision = (
            request.form.get("precision", "single").lower()
            == "double"
        )

        f0_floor = float(request.form.get("f0_floor", 71.0))
        f0_ceil = float(request.form.get("f0_ceil", 800.0))

        # 【修复】前端发送的是 auto_note_pitch，而非 refine_pitch
        refine_pitch = (
            request.form.get("auto_note_pitch", "false").lower()
            == "true"
        )

        # 【修复】前端发送的 export_pitch_line 决定是否将 F0 曲线写入工程文件
        export_pitch_line = (
            request.form.get("export_pitch_line", "true").lower()
            == "true"
        )

        stem, wav_path, lab_path = build_job_paths(
            audio_file.filename or "audio.wav"
        )

        audio_file.save(str(wav_path))

        lab_path.write_text(
            text + "\n",
            encoding="utf-8"
        )

        job_id = uuid.uuid4().hex

        set_job(
            job_id,
            status="queued",
            created_at=datetime.now().isoformat(),
        )

        Thread(
            target=run_pipeline_job,
            daemon=True,
            args=(
                job_id,
                str(wav_path),
                text,
                language,
                output_format,
                project_title,
                bpm,
                base_pitch,
                f0_method,
                f0_smooth,
                f0_smooth_window,
                use_double_precision,
                f0_floor,
                f0_ceil,
                refine_pitch,
                export_pitch_line,
            ),
        ).start()

        return jsonify(
            {
                "success": True,
                "job_id": job_id,
                "status": "queued",
            }
        ), 202

    except Exception as e:
        logger.exception("完整流程启动失败")
        return jsonify({"error": str(e)}), 500
    

# =====================================================================
# 替换原有的 pipeline_mfa_only 函数，新增 run_mfa_only_job 异步任务
# =====================================================================

def run_mfa_only_job(job_id: str, wav_path: str, text: str, language: str):
    try:
        set_job(
            job_id,
            status="running",
            started_at=datetime.now().isoformat(),
        )

        # 伪造 FileStorage 包装器，兼容 pipeline 的写入逻辑
        class FileStorageWrapper:
            def __init__(self, local_path):
                self.path = os.path.abspath(local_path)
                self.filename = os.path.basename(local_path)

            def save(self, dst):
                import shutil
                if os.path.abspath(dst) != self.path:
                    shutil.copy(self.path, dst)

            def seek(self, *args, **kwargs):
                pass

        compat_audio_file = FileStorageWrapper(wav_path)

        # 执行耗时的 MFA 标注
        result = pipeline.process_mfa_only(compat_audio_file, text, language)

        if result.get("success"):
            set_job(
                job_id,
                status="done",
                finished_at=datetime.now().isoformat(),
                result=result,
            )
        else:
            set_job(
                job_id,
                status="failed",
                finished_at=datetime.now().isoformat(),
                error=result.get("error"),
                result=result,
            )

    except Exception as e:
        logger.exception("后台MFA任务异常")
        set_job(
            job_id,
            status="failed",
            finished_at=datetime.now().isoformat(),
            error=str(e),
        )


@app.route("/api/pipeline/mfa-only", methods=["POST"])
def pipeline_mfa_only():
    """仅执行 MFA 标注 (异步后台任务轮询版)"""
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

        # 1. 马上保存文件，生成路径
        stem, wav_path, lab_path = build_job_paths(audio_file.filename or "audio.wav")
        audio_file.save(str(wav_path))

        # 2. 创建任务 ID
        job_id = uuid.uuid4().hex
        set_job(
            job_id,
            status="queued",
            created_at=datetime.now().isoformat(),
        )

        logger.info(f"MFA 标注模式启动，投递后台任务: {job_id}")

        # 3. 启动后台线程执行耗时任务
        Thread(
            target=run_mfa_only_job,
            daemon=True,
            args=(job_id, str(wav_path), text, language),
        ).start()

        # 4. 立即返回 job_id 供前端轮询
        return jsonify({
            "success": True,
            "job_id": job_id,
            "status": "queued",
        }), 202

    except Exception as e:
        logger.error("MFA 模式异常: %s", e, exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/pipeline/textgrid-to-lab", methods=["POST"])
def pipeline_textgrid_to_lab():
    """将 TextGrid 文件转换为 LAB 格式（可直接上传 .TextGrid 文件）。"""
    try:
        text = request.form.get("text", "").strip()
        language = request.form.get("language", "cmn")

        tg_file = request.files.get("textgrid_file")
        tg_path = request.form.get("textgrid_path")

        if tg_file is not None:
            basename = Path(tg_file.filename or "alignment.TextGrid").stem
            dest_path = WORK_DIR / f"{basename}_{uuid.uuid4().hex[:6]}.TextGrid"
            tg_file.save(str(dest_path))
            tg_path = str(dest_path)

        if not tg_path:
            return jsonify({"error": "请提供 textgrid_file 或 textgrid_path"}), 400
        if not os.path.exists(tg_path):
            return jsonify({"error": f"TextGrid 文件不存在: {tg_path}"}), 400
        if not text:
            return jsonify({"error": "请提供文本（text）以完成 TextGrid → LAB 转换"}), 400

        result = pipeline.textgrid_to_lab(tg_path, text, language)
        if result.get("success"):
            return jsonify(result), 200
        return jsonify(result), 500
    except Exception as e:
        logger.error("TextGrid→LAB 失败: %s", e, exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/pipeline/from-textgrid", methods=["POST"])
def pipeline_from_textgrid():
    """
    TextGrid → LAB → 工程文件（一步完成）。
    需要上传 TextGrid 文件 + WAV 文件（或提供 wav_path）。
    """
    try:
        text = request.form.get("text", "").strip()
        language = request.form.get("language", "cmn")
        output_format = request.form.get("format", "sv")
        project_title = request.form.get("title", "Project")
        bpm = float(request.form.get("bpm", 120))
        base_pitch = int(request.form.get("base_pitch", 60))
        f0_method = request.form.get("f0_method", "dio")
        f0_smooth = request.form.get("f0_smooth", "true").lower() == "true"
        f0_smooth_window = int(request.form.get("f0_smooth_window", 5))
        use_double_precision = request.form.get("precision", "single").lower() == "double"
        f0_floor = float(request.form.get("f0_floor", 71.0))
        f0_ceil = float(request.form.get("f0_ceil", 800.0))
        refine_pitch = request.form.get("auto_note_pitch", "false").lower() == "true"
        export_pitch_line = request.form.get("export_pitch_line", "true").lower() == "true"

        if not text:
            return jsonify({"error": "请提供文本（text）以完成 TextGrid → LAB 转换"}), 400

        # TextGrid 文件
        tg_file = request.files.get("textgrid_file")
        tg_path = request.form.get("textgrid_path")
        if tg_file is not None:
            basename = Path(tg_file.filename or "alignment.TextGrid").stem
            dest_path = WORK_DIR / f"{basename}_{uuid.uuid4().hex[:6]}.TextGrid"
            tg_file.save(str(dest_path))
            tg_path = str(dest_path)
        if not tg_path or not os.path.exists(tg_path):
            return jsonify({"error": "请提供 textgrid_file 或有效的 textgrid_path"}), 400

        # WAV 文件
        wav_file = request.files.get("wav_file")
        wav_path = request.form.get("wav_path")
        if wav_file is not None:
            stem, wav_path_obj, _ = build_job_paths(wav_file.filename or "audio.wav")
            wav_file.save(str(wav_path_obj))
            wav_path = str(wav_path_obj)
        if not wav_path or not os.path.exists(wav_path):
            return jsonify({"error": "请提供 wav_file 或有效的 wav_path"}), 400

        result = pipeline.process_from_textgrid(
            textgrid_path=tg_path,
            wav_path=wav_path,
            text=text,
            language=language,
            output_format=output_format,
            project_title=project_title,
            bpm=bpm,
            base_pitch=base_pitch,
            f0_method=f0_method,
            f0_smooth=f0_smooth,
            f0_smooth_window=f0_smooth_window,
            use_double_precision=use_double_precision,
            f0_floor=f0_floor,
            f0_ceil=f0_ceil,
            refine_pitch=refine_pitch,
            export_pitch_line=export_pitch_line,
        )

        if result.get("success"):
            result.setdefault("project_path", result.get("output_path"))
            result.setdefault("project_format", output_format)
            result.setdefault("requested_format", output_format)
            return jsonify(result), 200
        return jsonify(result), 500
    except Exception as e:
        logger.error("TextGrid 工程生成失败: %s", e, exc_info=True)
        return jsonify({"error": str(e)}), 500



@app.route("/api/pipeline/project-only", methods=["POST"])
def pipeline_project_only():
    """
    仅执行工程文件生成：
    - 只需要 WAV + LAB
    - 不需要 text
    - 继续复用 pipeline.process_project_only(...)
    """
    try:
        # 读取工程参数
        output_format = request.form.get("format", "sv")
        project_title = request.form.get("title", "Project")
        bpm = float(request.form.get("bpm", 120))
        base_pitch = int(request.form.get("base_pitch", 60))
        f0_method = request.form.get("f0_method", "dio")
        f0_smooth = request.form.get("f0_smooth", "true").lower() == "true"
        f0_smooth_window = int(request.form.get("f0_smooth_window", 5))
        use_double_precision = request.form.get("precision", "single").lower() == "double"
        f0_floor = float(request.form.get("f0_floor", 71.0))
        f0_ceil = float(request.form.get("f0_ceil", 800.0))

        # 是否根据 F0 细化音高
        refine_pitch = request.form.get("auto_note_pitch", "false").lower() == "true"

        # 是否导出音高线到工程文件
        export_pitch_line = request.form.get("export_pitch_line", "false").lower() == "true"

        # 兼容两种输入：
        # 1) 前端上传文件：wav_file + lab_file
        # 2) 已有路径：wav_path + lab_path
        wav_path = request.form.get("wav_path")
        lab_path = request.form.get("lab_path")

        wav_file = request.files.get("wav_file")
        lab_file = request.files.get("lab_file")

        if wav_file is not None and lab_file is not None:
            # 用 WAV 文件名作为同名基底，确保 wav/lab 成对保存
            stem, wav_path_obj, lab_path_obj = build_job_paths(wav_file.filename or "audio.wav")
            wav_file.save(str(wav_path_obj))
            lab_file.save(str(lab_path_obj))
            wav_path = str(wav_path_obj)
            lab_path = str(lab_path_obj)

        if not wav_path or not lab_path:
            return jsonify({"error": "请提供 wav_path/lab_path 或 wav_file/lab_file"}), 400

        supported_formats = pipeline.get_supported_formats().get("formats", [])
        if output_format not in supported_formats:
            return jsonify({
                "error": f"不支持的格式: {output_format}",
                "supported": supported_formats
            }), 400

        if not os.path.exists(wav_path):
            return jsonify({"error": f"WAV 文件不存在: {wav_path}"}), 400

        if not os.path.exists(lab_path):
            return jsonify({"error": f"LAB 文件不存在: {lab_path}"}), 400

        logger.info(
            "工程文件模式启动: format=%s wav=%s lab=%s",
            output_format, wav_path, lab_path
        )

        result = pipeline.process_project_only(
            wav_path=wav_path,
            lab_path=lab_path,
            output_format=output_format,
            project_title=project_title,
            bpm=bpm,
            base_pitch=base_pitch,
            f0_method=f0_method,
            f0_smooth=f0_smooth,
            f0_smooth_window=f0_smooth_window,
            use_double_precision=use_double_precision,
            f0_floor=f0_floor,
            f0_ceil=f0_ceil,
            refine_pitch=refine_pitch,
            export_pitch_line=export_pitch_line,
        )

        if result.get("success"):
            # 统一补齐前端常见字段，方便直接显示
            result.setdefault("project_path", result.get("project_path") or result.get("output_path"))
            result.setdefault("project_format", result.get("project_format", output_format))
            result.setdefault("requested_format", output_format)
            result.setdefault("project_title", project_title)
            return jsonify(result), 200

        return jsonify({
            "success": False,
            "error": result.get("error", "工程生成失败"),
            "processing_time": result.get("processing_time", 0),
        }), 500

    except Exception as e:
        logger.error("工程文件生成异常: %s", e, exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/pipeline/f0-only", methods=["POST"])
def pipeline_f0_only():
    """仅执行 F0 提取"""
    try:
        wav_path = request.form.get("wav_path")
        method = request.form.get("method", "dio")

        if not wav_path:
            return jsonify({"error": "缺少 wav_path"}), 400

        if method not in ["dio", "harvest"]:
            return jsonify({
                "error": f"不支持的方法: {method}",
                "supported": ["dio", "harvest"]
            }), 400

        # 验证文件存在
        if not os.path.exists(wav_path):
            return jsonify({"error": f"WAV 文件不存在: {wav_path}"}), 400

        logger.info(f"F0 提取模式启动: {method} 方法")
        result = pipeline.process_f0_only(wav_path, method=method)

        if result.get("success"):
            return jsonify(result), 200

        return jsonify({
            "success": False,
            "error": "F0 提取失败",
            "processing_time": result.get("processing_time", 0)
        }), 500

    except Exception as e:
        logger.error("F0 提取模式异常: %s", e, exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/mfa/download-model/<language>", methods=["POST"])
def download_mfa_model(language: str):
    """下载 MFA 模型"""
    try:
        valid_languages = ["cmn", "zh", "eng", "en", "jpn", "ja", "kor", "ko", "yue"]
        if language not in valid_languages:
            return jsonify({"error": f"不支持的语言: {language}"}), 400

        success, message = MFAChecker.download_model(language)

        if success:
            return jsonify({"success": True, "message": message}), 200
        return jsonify({"success": False, "error": message}), 400

    except Exception as e:
        logger.error("下载模型错误: %s", e, exc_info=True)
        return jsonify({"error": f"下载失败: {str(e)}"}), 500


@app.route("/api/work-dir/files", methods=["GET"])
def list_work_dir_files():
    """列出工作目录中的文件"""
    try:
        files = []
        
        # 列出 WAV 文件
        for wav_file in WORK_DIR.glob("*.wav"):
            files.append({
                "name": wav_file.name,
                "path": str(wav_file),
                "type": "audio",
                "size": wav_file.stat().st_size,
                "modified": wav_file.stat().st_mtime
            })
        
        # 列出 LAB 文件
        for lab_file in WORK_DIR.glob("*.lab"):
            files.append({
                "name": lab_file.name,
                "path": str(lab_file),
                "type": "label",
                "size": lab_file.stat().st_size,
                "modified": lab_file.stat().st_mtime
            })
        
        # 列出 TextGrid 文件
        for tg_file in WORK_DIR.glob("*.TextGrid"):
            files.append({
                "name": tg_file.name,
                "path": str(tg_file),
                "type": "textgrid",
                "size": tg_file.stat().st_size,
                "modified": tg_file.stat().st_mtime
            })

        # 列出工程文件
        for project_file in WORK_DIR.glob("**/*.ustx"):
            files.append({
                "name": project_file.name,
                "path": str(project_file),
                "type": "project",
                "size": project_file.stat().st_size,
                "modified": project_file.stat().st_mtime
            })
        
        # 按修改时间排序
        files.sort(key=lambda x: x['modified'], reverse=True)
        
        return jsonify({
            "success": True,
            "work_dir": str(WORK_DIR),
            "file_count": len(files),
            "files": files
        }), 200
        
    except Exception as e:
        logger.error("列出文件失败: %s", e, exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/work-dir/download/<path:filename>", methods=["GET"])
def download_work_file(filename: str):
    """下载工作目录中的文件"""
    try:
        # 获取文件的相对路径和名称
        filename = filename.replace('\\', '/')  # 规范化路径分隔符
        
        # 提取最后的文件名
        basename = filename.split('/')[-1]
        
        # 构造完整路径
        if 'projects' in filename:
            # 来自 projects 子目录
            file_path = WORK_DIR / 'projects' / basename
        else:
            # 来自工作目录
            file_path = WORK_DIR / basename
        
        logger.info(f"下载请求: {filename} -> {file_path}")
        
        # 安全检查
        try:
            file_path = file_path.resolve()
            work_dir_resolved = (WORK_DIR).resolve()
            
            if not str(file_path).startswith(str(work_dir_resolved)):
                return jsonify({"error": "无权访问该文件"}), 403
        except ValueError:
            return jsonify({"error": "路径错误"}), 400
        
        if not file_path.exists():
            logger.error(f"文件不存在: {file_path}")
            return jsonify({"error": f"文件不存在: {file_path}"}), 404
        
        if not file_path.is_file():
            return jsonify({"error": "不是文件"}), 400
        
        logger.info(f"下载文件成功: {file_path}")
        return send_from_directory(str(file_path.parent), file_path.name, as_attachment=True)
        
    except Exception as e:
        logger.error(f"下载文件失败: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/work-dir/clear", methods=["POST"])
def clear_work_dir():
    """清空工作目录"""
    try:
        import shutil
        
        # 只删除特定类型的文件
        patterns = ["*.wav", "*.lab", "**/*.ustx", "*.txt", "*.TextGrid", "**/*.svp"]
        
        deleted_count = 0
        for pattern in patterns:
            for file_path in WORK_DIR.glob(pattern):
                if file_path.is_file():
                    file_path.unlink()
                    deleted_count += 1
        
        logger.info(f"清空工作目录: 删除 {deleted_count} 个文件")
        
        return jsonify({
            "success": True,
            "message": f"已删除 {deleted_count} 个文件"
        }), 200
        
    except Exception as e:
        logger.error("清空工作目录失败: %s", e, exc_info=True)
        return jsonify({"error": str(e)}), 500


def open_browser(host: str, port: int):
    sleep(2)
    webbrowser.open(f"http://{host}:{port}")


def main(host: str = "127.0.0.1", port: int = 5000):
    print(f"\n{'=' * 60}")
    print("🚀 启动 SVS Lab Aligner with MFA + PyWORLD")
    print(f"📍 访问地址: http://{host}:{port}")
    print(f"📂 工作目录: {WORK_DIR}")
    print(f"📂 前端目录: {FRONTEND_DIST}")
    print(f"⏹️  按 Ctrl+C 停止服务")
    print(f"{'=' * 60}\n")

    Thread(target=open_browser, args=(host, port), daemon=True).start()
    app.run(host=host, port=port, debug=True, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()