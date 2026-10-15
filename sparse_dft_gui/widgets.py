"""Reusable widgets: cards, banners, path pickers, collapsible sections, async helper."""
from __future__ import annotations

import traceback

from PyQt6.QtCore import QEvent, QObject, QRegularExpression, QRunnable, QThreadPool, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QRegularExpressionValidator
from PyQt6.QtWidgets import (QFileDialog, QFrame, QHBoxLayout, QLabel, QLineEdit, QSizePolicy,
                             QToolButton, QVBoxLayout, QWidget)

from . import icons
from .i18n import tr

FLOAT_RE = QRegularExpression(r'^\s*[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?\s*$|^\s*$')
INT_RE = QRegularExpression(r'^\s*[+-]?\d*\s*$')


def repolish(widget):
    """Re-apply QSS after a dynamic property changed."""
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()


def set_role(label: QLabel, role: str) -> QLabel:
    label.setProperty('role', role)
    return label


class Card(QFrame):
    def __init__(self, parent=None, *, margins=(16, 14, 16, 14), spacing=10):
        super().__init__(parent)
        self.setProperty('card', True)
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(*margins)
        self.body.setSpacing(spacing)


class NoWheelFilter(QObject):
    """Ignore wheel events on unfocused inputs so scrolling a form never edits it."""

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Wheel and not obj.hasFocus():
            event.ignore()
            return True
        return False


_NO_WHEEL = None


def no_wheel(widget):
    global _NO_WHEEL
    if _NO_WHEEL is None:
        _NO_WHEEL = NoWheelFilter()
    widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    widget.installEventFilter(_NO_WHEEL)
    return widget


def float_edit(parent=None) -> QLineEdit:
    edit = QLineEdit(parent)
    edit.setValidator(QRegularExpressionValidator(FLOAT_RE, edit))
    return edit


def int_edit(parent=None) -> QLineEdit:
    edit = QLineEdit(parent)
    edit.setValidator(QRegularExpressionValidator(INT_RE, edit))
    return edit


class PathEdit(QWidget):
    """Line edit plus browse button; ``mode`` is ``open``, ``dir`` or ``save``."""

    textChanged = pyqtSignal(str)

    def __init__(self, mode='open', file_filter='All files (*)', parent=None):
        super().__init__(parent)
        self.mode, self.file_filter = mode, file_filter
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.edit = QLineEdit(self)
        self.edit.textChanged.connect(self.textChanged)
        self.button = QToolButton(self)
        self.button.setProperty('kind', 'ghost')
        self.button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.button.clicked.connect(self.browse)
        layout.addWidget(self.edit, 1)
        layout.addWidget(self.button)
        self.set_icon_colour('#5E5A6E')
        self.retranslate()

    def set_icon_colour(self, colour):
        self.button.setIcon(icons.icon('folder' if self.mode == 'dir' else 'file', colour, 18))

    def retranslate(self):
        self.button.setToolTip(tr('btn.browse'))

    def text(self) -> str:
        return self.edit.text()

    def setText(self, text):
        self.edit.setText('' if text is None else str(text))

    def setPlaceholderText(self, text):
        self.edit.setPlaceholderText(text)

    def browse(self):
        start = self.edit.text().strip()
        if self.mode == 'dir':
            path = QFileDialog.getExistingDirectory(self, tr('btn.browse'), start)
        elif self.mode == 'save':
            path, _ = QFileDialog.getSaveFileName(self, tr('btn.browse'), start, self.file_filter)
        else:
            path, _ = QFileDialog.getOpenFileName(self, tr('btn.browse'), start, self.file_filter)
        if path:
            self.edit.setText(path)


class CollapsibleSection(QFrame):
    """Card with a clickable header that shows or hides its body."""

    toggled = pyqtSignal(bool)
    DOTS = {'inputs': 'pistachio', 'basis': 'lavender', 'convergence': 'blue', 'spin': 'rose',
            'relax': 'lemon', 'parallel': 'blue', 'group': 'peach', 'options': 'lavender'}

    def __init__(self, title='', section='options', expanded=True, parent=None):
        super().__init__(parent)
        self.setProperty('card', True)
        self.section = section
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.header = QToolButton(self)
        self.header.setObjectName('SectionHeader')
        self.header.setCheckable(True)
        self.header.setChecked(expanded)
        self.header.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.header.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.header.setProperty('section', section)
        self.header.toggled.connect(self._on_toggle)
        outer.addWidget(self.header)
        self.body = QWidget(self)
        self.body.setObjectName('SectionBody')
        outer.addWidget(self.body)
        self.body.setVisible(expanded)
        self.setTitle(title)

    def setTitle(self, title):
        self._title = title
        self._update_arrow()

    def _update_arrow(self):
        arrow = '▾' if self.header.isChecked() else '▸'
        self.header.setText(f'{arrow}  {self._title}')

    def _on_toggle(self, expanded):
        self.body.setVisible(expanded)
        self._update_arrow()
        self.toggled.emit(expanded)

    def expanded(self) -> bool:
        return self.header.isChecked()

    def setExpanded(self, expanded: bool):
        self.header.setChecked(expanded)


class Banner(QFrame):
    """Non-blocking message strip (info/ok/warning/error) that hides itself after a while."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('Banner')
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 8, 8)
        layout.setSpacing(10)
        self.icon = QLabel(self)
        self.text = QLabel(self)
        self.text.setWordWrap(True)
        self.text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.close_button = QToolButton(self)
        self.close_button.setProperty('kind', 'ghost')
        self.close_button.clicked.connect(self.hide)
        layout.addWidget(self.icon)
        layout.addWidget(self.text, 1)
        layout.addWidget(self.close_button)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.hide)
        self.palette_colours = {}
        self.hide()

    def show_message(self, text, level='info', timeout_ms=None):
        level = level if level in ('info', 'ok', 'warning', 'error') else 'info'
        self.setProperty('level', level)
        self.text.setProperty('level', level)
        self.text.setText(text)
        colour = self.palette_colours.get(level, '#2B2A33')
        self.icon.setPixmap(icons.icon(level, colour, 18).pixmap(18, 18))
        self.close_button.setIcon(icons.icon('close', colour, 14))
        self.close_button.setToolTip(tr('btn.dismiss'))
        repolish(self); repolish(self.text)
        self.show()
        if timeout_ms is None:
            timeout_ms = 0 if level == 'error' else 6000
        if timeout_ms:
            self.timer.start(timeout_ms)
        else:
            self.timer.stop()


class Chip(QLabel):
    def __init__(self, state='idle', parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.set_state(state)

    def set_state(self, state, text=None):
        self.setProperty('chip', state)
        self.setText(text if text is not None else tr(f'state.{state}'))
        repolish(self)


class _Signals(QObject):
    done = pyqtSignal(object)
    failed = pyqtSignal(str)


class _Task(QRunnable):
    def __init__(self, fn, args, kwargs, signals):
        super().__init__()
        self.fn, self.args, self.kwargs, self.signals = fn, args, kwargs, signals
        self.setAutoDelete(True)

    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
        except BaseException as error:  # noqa: BLE001 - reported to the UI, never raised
            detail = ''.join(traceback.format_exception_only(type(error), error)).strip()
            self._emit('failed', detail)
            return
        self._emit('done', result)

    def _emit(self, name, value):
        try:
            getattr(self.signals, name).emit(value)
        except RuntimeError:            # the signal holder was deleted (app closed while the task ran)
            pass


_KEEP: set = set()


def run_async(fn, *args, on_done=None, on_error=None, **kwargs):
    """Run ``fn`` on the global thread pool; callbacks are delivered on the GUI thread."""
    signals = _Signals()
    _KEEP.add(signals)

    def finish(callback, value):
        _KEEP.discard(signals)
        if callback is not None:
            callback(value)
    signals.done.connect(lambda value: finish(on_done, value))
    signals.failed.connect(lambda text: finish(on_error, text))
    QThreadPool.globalInstance().start(_Task(fn, args, kwargs, signals))
    return signals
