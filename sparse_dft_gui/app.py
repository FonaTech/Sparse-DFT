"""Application entry point (``python -m sparse_dft_gui`` / ``sparse-dft-gui``).

``sparse-dft-gui --worker JOB.json [--status STATUS.json]`` runs one job without
starting Qt; the frozen application uses this to launch its own workers.
"""
from __future__ import annotations

import os
import sys

_QT_TRANSLATORS: list = []


def _worker_argv(argv):
    if argv and argv[0] == '--worker':
        return argv[1:]
    if argv and argv[0] == 'worker':
        return argv[1:]
    return None


def install_qt_translator(app, language: str) -> bool:
    """Load Qt's own ``qtbase_<lang>.qm`` for standard dialogs when it ships with PyQt6."""
    from PyQt6.QtCore import QTranslator
    from .i18n import qt_translations_path
    for translator in _QT_TRANSLATORS:
        app.removeTranslator(translator)
    _QT_TRANSLATORS.clear()
    if language == 'en':
        return False
    path = qt_translations_path()
    if not path:
        return False
    translator = QTranslator(app)
    if translator.load(f'qtbase_{language}', path):
        app.installTranslator(translator)
        _QT_TRANSLATORS.append(translator)
        return True
    return False


def create_app(argv=None):
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QGuiApplication
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        QGuiApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
        app = QApplication(list(argv if argv is not None else sys.argv[:1]))
    app.setApplicationName('Sparse-DFT Studio')
    app.setOrganizationName('Sparse-DFT')
    app.setOrganizationDomain('sparse_dft.local')
    app.setStyle('Fusion')
    from . import fonts
    fonts.register_bundled_fonts()
    icon_path = asset_path('sparse_dft.svg')
    if icon_path.is_file():
        from PyQt6.QtGui import QIcon
        app.setWindowIcon(QIcon(str(icon_path)))
    return app


def asset_path(name: str):
    """Bundled asset (``sparse_dft_gui/assets``) in a checkout, a wheel or a frozen app."""
    from pathlib import Path
    from sparse_dft.resources import is_frozen, resource_root
    base = resource_root() / 'sparse_dft_gui' / 'assets' if is_frozen() else Path(__file__).resolve().parent / 'assets'
    return base / name


def create_window(app, prefs=None, **kwargs):
    from . import fonts
    from .window import MainWindow
    window = MainWindow(prefs, app=app, **kwargs)
    app.setFont(fonts.ui_font(window.language, window.font_px))
    install_qt_translator(app, window.language)
    window.set_theme(window.theme_name)
    return window


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    worker = _worker_argv(argv)
    if worker is not None:
        from sparse_dft import runner
        return runner.main(worker)
    if getattr(sys, 'frozen', False):
        # Frozen builds: make sure subprocesses and matplotlib caches use writable locations.
        os.environ.setdefault('PYTHONUTF8', '1')
        os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
    try:
        app = create_app([sys.argv[0], *argv])
    except ImportError as error:
        print(f'Sparse-DFT GUI needs PyQt6 (pip install "sparse_dft[gui]"): {error}', file=sys.stderr)
        return 2
    window = create_window(app)
    window.show()
    for item in argv:
        if not item.startswith('-') and os.path.exists(item):
            window.show_page('structure').load(item)
            break
    return app.exec()


if __name__ == '__main__':
    raise SystemExit(main())
