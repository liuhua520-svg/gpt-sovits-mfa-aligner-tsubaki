# locales/en.py — English UI strings
STRINGS = {
    # ── App shell ──────────────────────────────────────────────────
    "app_title": "SVS Lab Aligner",
    "app_subtitle": "MFA Alignment · Pitch Extraction · Project File Generation",
    "system_ready": "✓ System Ready",
    "system_not_ready": "⚠ Configuration Required",
    "footer_text": "SVS Lab Aligner  •  PyWORLD + MFA + PySide6",
    "language": "Language",

    # ── Card header ────────────────────────────────────────────────
    "single_file_processing": "📁 Single File Processing",
    "github_link": "🔗 GitHub",
    "check_status": "🔄 Check Status",

    # ── Audio drop area ────────────────────────────────────────────
    "audio_file": "Audio File",
    "drop_audio_hint": "Drag & drop or click to select audio",
    "audio_formats_tip": "WAV / MP3 / FLAC / M4A / AAC  •  Max 512 MB",
    "browse": "Browse…",
    "clear": "Clear",

    # ── Aligner backend ────────────────────────────────────────────
    "aligner_backend": "Aligner Backend",
    "backend_mfa_desc": "🎯 MFA: Kaldi-based phoneme-level forced alignment. High accuracy; requires MFA + language models. Reference text required.",
    "backend_whisperx_desc": "🤖 WhisperX: Whisper ASR + wav2vec2 forced alignment. No pre-installed models needed. Text optional.",
    "backend_qwen3_asr_desc": "🌐 Qwen3-ASR-1.7B: More robust for Chinese accented speech. Text optional.",
    "backend_qwen3_aligner_desc": "📌 Qwen3-ForcedAligner-0.6B: Lightweight neural forced alignment designed for singing. Text required.",

    # ── Aligner device ─────────────────────────────────────────────
    "aligner_device": "Aligner Device",
    "device_auto": "Auto (prefer GPU)",
    "device_cpu": "CPU",
    "device_cuda": "CUDA (GPU)",
    "whisperx_gpu_hint": "GPU precision auto-selected: Turing/RTX 20xx+ → float16, Pascal/old → int8",
    "qwen3_gpu_hint": "Qwen3 GPU: Ampere → bfloat16; Pascal/Volta/Turing → float16; CPU → float32",
    "cpu_mode_hint": "⚠ CPU mode is slow — use only when no GPU or VRAM is insufficient",
    "auto_device_hint": "Auto: uses GPU if available; precision auto-selected by architecture",

    # ── LAB/MIDI drop area ─────────────────────────────────────────
    "lab_midi_file": "LAB / MIDI File",
    "drop_lab_hint": "Drag & drop or click to select one file (LAB or MIDI)",
    "lab_midi_tip": "One file only: .lab phoneme annotation  or  .mid / .midi note source",

    # ── Text input ─────────────────────────────────────────────────
    "input_text": "Input Text",
    "text_placeholder_optional": "(Optional) Leave blank for auto-transcription, or paste reference text",
    "text_placeholder_required": "Paste reference text content here",
    "text_optional_hint": "✓ ASR mode supported — text can be left empty",
    "char_count": "Characters: {count}",

    # ── Language select ────────────────────────────────────────────
    "language_select": "Language",
    "lang_cmn": "Mandarin 🇨🇳",
    "lang_eng": "English 🇬🇧",
    "lang_jpn": "Japanese 🇯🇵",
    "lang_kor": "Korean 🇰🇷",
    "lang_yue": "Cantonese 🇭🇰",

    # ── Processing mode ────────────────────────────────────────────
    "processing_mode": "Processing Mode",
    "mode_mfa_only": "Label Only (Fast)",
    "mode_full": "Full Pipeline (Label + F0 + Project)",
    "mode_project_only": "Project Only (WAV + LAB / MIDI)",
    "mode_mfa_only_desc": "Auto alignment only — generates a LAB file",
    "mode_full_desc": "Full pipeline: alignment → F0 extraction → project file",
    "mode_project_only_desc": "Merge existing WAV + LAB/MIDI into a project file; skips alignment",

    # ── Output format ──────────────────────────────────────────────
    "output_format": "Output Format",
    "format_sv": "Synthesizer V Studio (.svp)",
    "format_utau": "OpenUtau / UTAU (.ustx)",

    # ── Phoneme mode ───────────────────────────────────────────────
    "phoneme_conversion": "Phoneme Conversion",
    "phoneme_none": "No Conversion",
    "phoneme_merge": "Merge Consonants",
    "phoneme_hiragana": "Hiragana",
    "phoneme_katakana": "Katakana",
    "phoneme_none_desc": "Keep original phoneme labels from LAB (all languages)",
    "phoneme_merge_desc": "Merge consonant + vowel into romaji syllable (s+a→sa, N→N)",
    "phoneme_hiragana_desc": "Merge + convert to Hiragana (s+a→さ, N→ん)",
    "phoneme_katakana_desc": "Merge + convert to Katakana (s+a→サ, N→ン)",
    "phoneme_warning": "⚠ Merge/kana conversion is for Japanese LAB files with individual phonemes (raw MFA output)",

    # ── Track name ─────────────────────────────────────────────────
    "track_name": "Track Name",
    "track_name_placeholder": "Enter track name",

    # ── Advanced settings ──────────────────────────────────────────
    "advanced_settings": "⚙️  Advanced Settings",
    "advanced_hide": "⚙️  Hide Advanced Settings",
    "bpm": "BPM",
    "base_pitch": "Base Pitch (MIDI Note)",
    "pitch_control": "📈  Pitch Fine Control",
    "auto_note_pitch": "Auto Note Pitch",
    "auto_note_pitch_on": "Auto-align to detected pitch",
    "auto_note_pitch_off": "Fixed at base pitch",
    "export_pitch_line": "Export Continuous Pitch",
    "export_pitch_line_on": "Write F0 curve parameter",
    "export_pitch_line_off": "Generate clean notes only",
    "f0_method_section": "F0 Extraction Algorithm & Range",
    "f0_method": "F0 Method",
    "f0_dio": "DIO  (Fast)",
    "f0_harvest": "Harvest  (Accurate)",
    "f0_crepe": "CREPE  (Neural, noise-robust)",
    "f0_rmvpe": "RMVPE  (Deep model, most robust)",
    "crepe_model_size": "CREPE Model Size",
    "crepe_full": "Full  (higher accuracy)",
    "crepe_tiny": "Tiny  (faster)",
    "f0_device": "Run Device",
    "precision": "Floating-Point Precision",
    "precision_single": "Single  (Float32)",
    "precision_double": "Double  (Float64)",
    "f0_smooth": "F0 Smoothing",
    "f0_smooth_window": "Smooth Window Size",
    "f0_floor_hz": "Min Frequency (Hz)",
    "f0_ceil_hz": "Max Frequency (Hz)",
    "midi_lock_tip": "🔒 Read from MIDI",
    "smooth_tip": "Recommended: 3-7  (larger = smoother)",
    "f0_range_tip": "Female: 150–300 Hz  •  Male: 80–150 Hz",

    # ── MIDI info ──────────────────────────────────────────────────
    "midi_loaded_banner": "🔒 MIDI Imported — options below are controlled by MIDI data",
    "midi_lock_desc": "🔒 Auto pitch · BPM · Base pitch — will be read directly from MIDI file",
    "bpm_from_midi": "🔒 {bpm} BPM  (from MIDI)",

    # ── Action buttons ─────────────────────────────────────────────
    "start_processing": "🚀  Start Processing",
    "processing_btn": "Processing…  {percent}%",
    "reset": "🔄  Reset",
    "system_not_ready_hint": "(System not ready or language model not downloaded)",

    # ── Results ────────────────────────────────────────────────────
    "result_title": "✅  Processing Results",
    "processing_time": "Processing Time: {time}",
    "lab_file_label": "LAB File: {name}",
    "project_file_label": "Project File: {name}",
    "segments_label": "Segments: {count}",
    "tab_lab_content": "LAB Annotation",
    "tab_file_info": "File Info",
    "tab_details": "Processing Details",
    "copy_lab": "📋  Copy LAB",
    "download_lab": "📥  Download LAB",
    "download_project": "📥  Download Project",
    "process_next": "🔄  Process Next",

    # ── Processing details table ───────────────────────────────────
    "col_stage": "Stage",
    "col_status": "Status",
    "col_details": "Details",
    "stage_alignment": "1. Alignment",
    "stage_f0": "2. F0 Extraction",
    "stage_project": "3. Project Generation",
    "status_waiting": "Waiting",
    "status_running": "Running",
    "status_done": "Done",
    "status_skipped": "Skipped",
    "status_failed": "Failed",

    # ── File info display ──────────────────────────────────────────
    "lab_path_label": "LAB Annotation File:",
    "project_path_label": "Project File:",
    "output_format_label": "Output Format:",
    "config_label": "Processing Config:",
    "cfg_bpm": "BPM: {v}",
    "cfg_base_pitch": "Base Pitch: {note}  (MIDI {midi})",
    "cfg_auto_pitch": "Auto Note Pitch: {state}",
    "cfg_export_pitch": "Export Continuous Pitch: {state}",
    "cfg_f0_method": "F0 Method: {method}",
    "cfg_device": "Run Device: {device}",
    "cfg_precision": "Precision: {prec}",
    "state_on": "Enabled",
    "state_off": "Disabled",
    "sv_studio": "Synthesizer V Studio",
    "openutau": "OpenUtau / UTAU",

    # ── System status panel ────────────────────────────────────────
    "system_status": "🔧  System Status",
    "mfa_status": "MFA Status:",
    "mfa_installed": "✓ Installed",
    "mfa_not_installed": "✗ Not Installed",
    "mfa_version": "Version:",
    "language_models": "Language Models:",
    "processing_modules": "Processing Modules:",
    "alt_backends_section": "Alternative Alignment Backends:",
    "model_cache_dir": "📁 Model Cache:",
    "download_btn": "Download",
    "downloading_btn": "Downloading…",

    "model_cmn": "CMN  (Mandarin)",
    "model_eng": "ENG  (English)",
    "model_jpn": "JPN  (Japanese)",
    "model_kor": "KOR  (Korean)",
    "model_yue": "YUE  (Cantonese)",

    # ── Warnings ───────────────────────────────────────────────────
    "warn_mfa_not_installed": "❌  MFA Not Installed",
    "warn_mfa_install_cmd": "pip install montreal-forced-aligner",
    "warn_not_ready_title": "⚠️  Component Not Ready",
    "warn_not_ready_msg": "Please download the required language models or check system status.",
    "warn_install_hint": "Then download the required language models.",

    # ── Toast / status messages ────────────────────────────────────
    "msg_only_one_file": "Only one file can be uploaded",
    "msg_invalid_lab_type": "Only .lab / .mid / .midi files are supported",
    "msg_no_lab_content": "No LAB content to download",
    "msg_copied": "Copied to clipboard",
    "msg_copy_failed": "Copy failed",
    "msg_model_ok": "Model '{lang}' downloaded successfully",
    "msg_model_fail": "Model download failed: {error}",
    "msg_backend_unreachable": "Cannot connect to backend",
    "msg_processing_ok": "✅  Processing successful!",
    "msg_project_ok": "✅  Project file generated!",
    "msg_status_refreshed": "System status refreshed",
    "msg_select_audio": "Please select an audio file",
    "msg_select_lab_midi": "Please select a LAB or MIDI file",
    "msg_enter_text": "Please enter text (MFA / Qwen3-FA requires reference text)",
    "msg_backend_not_ready": "Alignment backend not ready — check system status",
    "msg_audio_too_large": "Audio file > 512 MB — consider splitting it first",
    "msg_lab_downloaded": "LAB file downloaded: {name}",

    # ── Install hints ──────────────────────────────────────────────
    "install_whisperx": "pip install whisperx",
    "install_qwen3": "pip install transformers torch torchaudio accelerate",
    "need_install": "Needs Installation",
    "status_available": "Available",

    # ── Backend URL ────────────────────────────────────────────────
    "backend_url_label": "Backend URL:",
    "backend_url_placeholder": "http://127.0.0.1:5000",
    "connect": "Connect",
}
