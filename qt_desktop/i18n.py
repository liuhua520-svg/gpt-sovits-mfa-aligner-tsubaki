# i18n.py — Internationalization engine
from __future__ import annotations
from PySide6.QtCore import QObject, Signal


class I18n(QObject):
    """
    Lightweight i18n engine.

    Usage
    -----
    i18n = I18n("zh_cn")        # create
    text = i18n.t("app_title")  # translate key
    i18n.set_language("en")     # switch → emits language_changed
    """

    language_changed = Signal()   # connect widgets to this

    # Display names shown in the language selector combo-box
    DISPLAY_NAMES: dict[str, str] = {
        "zh_cn": "简体中文",
        "zh_tw": "繁體中文",
        "ja":    "日本語",
        "ko":    "한국어",
        "en":    "English",
    }
    # Canonical order for the combo-box
    LANGUAGE_CODES = ["zh_cn", "zh_tw", "ja", "ko", "en"]

    def __init__(self, default_lang: str = "zh_cn", parent=None):
        super().__init__(parent)
        self._lang = default_lang if default_lang in self.DISPLAY_NAMES else "zh_cn"
        self._tables: dict[str, dict] = {}
        self._load_all()

    # ── internal ──────────────────────────────────────────────────
    def _load_all(self):
        from locales import en, zh_cn, zh_tw, ja, ko
        self._tables = {
            "en":    en,
            "zh_cn": zh_cn,
            "zh_tw": zh_tw,
            "ja":    ja,
            "ko":    ko,
        }

    # ── public API ────────────────────────────────────────────────
    def set_language(self, code: str):
        if code in self._tables and code != self._lang:
            self._lang = code
            self.language_changed.emit()

    def t(self, key: str, **kwargs) -> str:
        """Translate *key* in the current language, with optional format args."""
        table = self._tables.get(self._lang, {})
        text = table.get(key)
        if text is None:                          # fall back to English
            text = self._tables["en"].get(key, key)
        if kwargs:
            try:
                text = text.format(**kwargs)
            except (KeyError, IndexError):
                pass
        return text

    @property
    def current(self) -> str:
        return self._lang

    @property
    def display_name(self) -> str:
        return self.DISPLAY_NAMES.get(self._lang, self._lang)
