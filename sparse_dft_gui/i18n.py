"""Translation catalogs (JSON) and locale detection.

Qt Linguist ``.qm`` files need ``lrelease``, which the PyQt6 wheels do not
ship, so the GUI uses JSON catalogs ``sparse_dft_gui/i18n/<lang>.json`` with a
:func:`tr` helper.  Qt's own ``qtbase_<lang>.qm`` (standard dialogs) is
loaded from ``QLibraryInfo`` when present.
"""
from __future__ import annotations

import json
from pathlib import Path

LANGUAGES = ('en', 'zh_CN', 'zh_TW', 'ja')
DEFAULT_LANGUAGE = 'en'
_TRADITIONAL_TERRITORIES = {'TW', 'HK', 'MO'}


def catalog_dir() -> Path:
    try:
        from sparse_dft.resources import translations_dir
        path = translations_dir()
        if path.is_dir():
            return path
    except Exception:  # noqa: BLE001
        pass
    return Path(__file__).resolve().parent / 'i18n'


def load_catalog(language: str) -> dict:
    path = catalog_dir() / f'{language}.json'
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}
    return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}


def _from_name(name: str) -> str:
    """Map a BCP-47/POSIX locale name to a supported language (pure string logic)."""
    text = (name or '').replace('-', '_').split('.')[0].split('@')[0]
    parts = [p for p in text.split('_') if p]
    if not parts:
        return DEFAULT_LANGUAGE
    lang = parts[0].lower()
    if lang == 'zh':
        rest = [p for p in parts[1:]]
        scripts = {p.title() for p in rest if len(p) == 4}
        territories = {p.upper() for p in rest if len(p) in (2, 3)}
        if 'Hant' in scripts or territories & _TRADITIONAL_TERRITORIES:
            return 'zh_TW'
        return 'zh_CN'
    if lang == 'ja':
        return 'ja'
    return DEFAULT_LANGUAGE


def detect_language(locale=None) -> str:
    """Supported language for ``locale`` (a ``QLocale``, a locale name, or ``None`` for the system).

    Chinese with Traditional script or territory TW/HK/MO -> ``zh_TW``; other
    Chinese -> ``zh_CN``; Japanese -> ``ja``; anything else, or any failure -> ``en``.
    """
    try:
        if isinstance(locale, str):
            return _from_name(locale)
        from PyQt6.QtCore import QLocale
        if locale is None:
            locale = QLocale.system()
        language = locale.language()
        if language == QLocale.Language.Chinese:
            script = locale.script()
            territory = locale.territory()
            traditional = {QLocale.Country.Taiwan, QLocale.Country.HongKong, QLocale.Country.Macau}
            if script == QLocale.Script.TraditionalHanScript or territory in traditional:
                return 'zh_TW'
            return 'zh_CN'
        if language == QLocale.Language.Japanese:
            return 'ja'
        return DEFAULT_LANGUAGE
    except Exception:  # noqa: BLE001 - detection must never break start-up
        return DEFAULT_LANGUAGE


class Translator:
    """Holds the active catalog; English is the fallback for missing keys."""

    def __init__(self, language: str = DEFAULT_LANGUAGE):
        self._english = load_catalog('en')
        self.language = DEFAULT_LANGUAGE
        self._catalog = self._english
        self.set_language(language)

    def set_language(self, language: str) -> str:
        language = language if language in LANGUAGES else DEFAULT_LANGUAGE
        self.language = language
        self._catalog = self._english if language == 'en' else (load_catalog(language) or self._english)
        return language

    def has(self, key: str) -> bool:
        return key in self._english

    def tr(self, key: str, default: str | None = None, **fmt) -> str:
        text = self._catalog.get(key) or self._english.get(key)
        if text is None:
            text = default if default is not None else key
        if fmt:
            try:
                text = text.format(**fmt)
            except (KeyError, IndexError, ValueError):
                pass
        return text

    def english(self, key: str) -> str | None:
        return self._english.get(key)

    # ---- argparse integration
    def option_label(self, spec) -> str:
        key = f'opt.{spec.dest}'
        if key in self._english:
            return self.tr(key)
        return spec.dest.replace('_', ' ').capitalize()

    def option_help(self, spec) -> str:
        """Translated help; untranslated or changed CLI help falls back to the argparse English text."""
        key = None
        for candidate in (f'help.{spec.command}.{spec.dest}', f'help.{spec.dest}'):
            if candidate in self._english:
                key = candidate
                break
        if key is None:
            return spec.help or ''
        if spec.help and self._english[key] != spec.help:
            return spec.help  # the CLI text changed since translation; English is authoritative
        return self.tr(key)

    def command_title(self, command: str) -> str:
        key = f'cmd.{command}'
        return self.tr(key) if key in self._english else command


_ACTIVE = Translator(DEFAULT_LANGUAGE)


def translator() -> Translator:
    return _ACTIVE


def set_language(language: str) -> str:
    return _ACTIVE.set_language(language)


def tr(key: str, default: str | None = None, **fmt) -> str:
    return _ACTIVE.tr(key, default, **fmt)


def qt_translations_path() -> str | None:
    try:
        from PyQt6.QtCore import QLibraryInfo
        return QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath)
    except Exception:  # noqa: BLE001
        return None
