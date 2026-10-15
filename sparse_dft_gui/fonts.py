"""Font selection per language and platform.

Fonts in ``data/fonts/`` (``.ttf``/``.otf``) are registered first and lead
every chain; nothing is downloaded.  Otherwise a platform chain is used so
CJK text never falls back to tofu boxes.
"""
from __future__ import annotations

import sys
from pathlib import Path

CHAINS = {
    'windows': {
        'zh_CN': ['Microsoft YaHei UI', 'Microsoft YaHei', 'Segoe UI'],
        'zh_TW': ['Microsoft JhengHei UI', 'Microsoft JhengHei', 'Segoe UI'],
        'ja': ['Yu Gothic UI', 'Meiryo UI', 'Meiryo', 'Segoe UI'],
        'en': ['Segoe UI', 'Microsoft YaHei UI'],
    },
    'macos': {
        'zh_CN': ['PingFang SC', 'Hiragino Sans GB', 'Heiti SC'],
        'zh_TW': ['PingFang TC', 'PingFang HK', 'Heiti TC'],
        'ja': ['Hiragino Sans', 'Hiragino Kaku Gothic ProN', 'PingFang SC'],
        'en': ['Helvetica Neue', 'PingFang SC'],
    },
    'linux': {
        'zh_CN': ['Noto Sans CJK SC', 'Noto Sans SC', 'Source Han Sans SC', 'WenQuanYi Micro Hei', 'Noto Sans', 'DejaVu Sans'],
        'zh_TW': ['Noto Sans CJK TC', 'Noto Sans TC', 'Source Han Sans TC', 'Noto Sans', 'DejaVu Sans'],
        'ja': ['Noto Sans CJK JP', 'Noto Sans JP', 'Source Han Sans JP', 'Noto Sans', 'DejaVu Sans'],
        'en': ['Noto Sans', 'DejaVu Sans', 'Noto Sans CJK SC'],
    },
}
# CJK fallbacks appended to every chain, so file names in any script render.
FALLBACKS = {
    'windows': ['Microsoft YaHei UI', 'Microsoft JhengHei UI', 'Yu Gothic UI', 'Segoe UI Symbol'],
    'macos': ['PingFang SC', 'PingFang TC', 'Hiragino Sans', 'Apple Symbols'],
    'linux': ['Noto Sans CJK SC', 'Noto Sans CJK TC', 'Noto Sans CJK JP', 'DejaVu Sans'],
}
MONO = {
    'windows': ['Cascadia Mono', 'Consolas', 'Courier New'],
    'macos': ['SF Mono', 'Menlo', 'Monaco'],
    'linux': ['DejaVu Sans Mono', 'Noto Sans Mono', 'Liberation Mono'],
}
_REGISTERED: list[str] = []


def platform_key(platform: str | None = None) -> str:
    platform = platform or sys.platform
    if platform.startswith('win'):
        return 'windows'
    if platform == 'darwin':
        return 'macos'
    return 'linux'


def chain(language: str, platform: str | None = None, *, bundled=(), available=None) -> list[str]:
    """Ordered font families for ``language``; ``available`` filters to installed families."""
    key = platform_key(platform)
    names = [*bundled, *CHAINS[key].get(language, CHAINS[key]['en']), *FALLBACKS[key]]
    seen, out = set(), []
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        if available is None or name in available:
            out.append(name)
    return out


def register_bundled_fonts(directory: Path | None = None) -> list[str]:
    """Register ``.ttf``/``.otf`` files with Qt; returns their family names (once per process)."""
    if _REGISTERED:
        return list(_REGISTERED)
    from PyQt6.QtGui import QFontDatabase
    if directory is None:
        from sparse_dft.resources import fonts_dir
        directory = fonts_dir()
    directory = Path(directory)
    if not directory.is_dir():
        return []
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in ('.ttf', '.otf', '.ttc'):
            continue
        font_id = QFontDatabase.addApplicationFont(str(path))
        if font_id >= 0:
            for family in QFontDatabase.applicationFontFamilies(font_id):
                if family not in _REGISTERED:
                    _REGISTERED.append(family)
    return list(_REGISTERED)


def ui_font(language: str, pixel_size: int = 13):
    from PyQt6.QtGui import QFont, QFontDatabase
    families = set(QFontDatabase.families())
    names = chain(language, bundled=_REGISTERED, available=families)
    font = QFont()
    if platform_key() == 'macos' and language == 'en':
        # The system UI font (San Francisco) is not listed by name; keep it first.
        system = QFontDatabase.systemFont(QFontDatabase.SystemFont.GeneralFont)
        names = [system.family(), *names]
    if names:
        font.setFamilies(names)
    font.setPixelSize(pixel_size)
    font.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
    return font


def mono_font(language: str, pixel_size: int = 12):
    from PyQt6.QtGui import QFont, QFontDatabase
    families = set(QFontDatabase.families())
    key = platform_key()
    names = [n for n in MONO[key] if n in families]
    names += chain(language, available=families)
    font = QFont()
    font.setFamilies(names or ['monospace'])
    font.setStyleHint(QFont.StyleHint.Monospace)
    font.setFixedPitch(True)
    font.setPixelSize(pixel_size)
    return font


def mpl_families(language: str) -> list[str]:
    """Matplotlib font list (installed families only) for chart text in ``language``."""
    try:
        from matplotlib import font_manager
        installed = {f.name for f in font_manager.fontManager.ttflist}
    except Exception:  # noqa: BLE001
        installed = None
    names = chain(language, bundled=_REGISTERED, available=installed)
    return [*names, 'DejaVu Sans']
