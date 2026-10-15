"""Log console: level filter, search, copy, UTF-8 export, auto-scroll, monospace font."""
from __future__ import annotations

import re
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QTextCharFormat, QTextCursor, QColor
from PyQt6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel,
                             QLineEdit, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget)

from . import icons
from .i18n import tr
from .widgets import no_wheel

LEVELS = ('all', 'info', 'warning', 'error')
_ERROR_RE = re.compile(r'(error|traceback|exception|failed|"code":\s*"[A-Z_]+ERROR")', re.IGNORECASE)
_WARN_RE = re.compile(r'(warn|cancel)', re.IGNORECASE)
MAX_LINES = 50000


def classify(stream: str, text: str) -> str:
    """Heuristic level of a line: worker stderr may carry ordinary progress, so inspect the text."""
    if _ERROR_RE.search(text) and 'returncode=0' not in text:
        return 'error'
    if _WARN_RE.search(text):
        return 'warning'
    return 'info'


class LogConsole(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.entries: list[tuple[str, str, str, str]] = []  # (job_id, level, stream, text)
        self.job_filter = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        bar = QHBoxLayout()
        bar.setSpacing(6)
        self.title = QLabel(self)
        self.title.setProperty('role', 'section')
        self.level = no_wheel(QComboBox(self))
        for level in LEVELS:
            self.level.addItem(level, level)
        self.level.currentIndexChanged.connect(self.refilter)
        self.search = QLineEdit(self)
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self.refilter)
        self.autoscroll = QCheckBox(self)
        self.autoscroll.setChecked(True)
        self.copy_button = QPushButton(self)
        self.copy_button.setProperty('kind', 'ghost')
        self.copy_button.clicked.connect(self.copy_all)
        self.export_button = QPushButton(self)
        self.export_button.setProperty('kind', 'ghost')
        self.export_button.clicked.connect(self.export_dialog)
        self.clear_button = QPushButton(self)
        self.clear_button.setProperty('kind', 'ghost')
        self.clear_button.clicked.connect(self.clear)
        for button in (self.copy_button, self.export_button, self.clear_button):
            button.setFixedWidth(34)
        bar.addWidget(self.title)
        bar.addStretch(1)
        bar.addWidget(self.autoscroll)
        bar.addWidget(self.copy_button)
        bar.addWidget(self.export_button)
        bar.addWidget(self.clear_button)
        layout.addLayout(bar)
        filters = QHBoxLayout()
        filters.setSpacing(6)
        filters.addWidget(self.level)
        filters.addWidget(self.search, 1)
        layout.addLayout(filters)
        self.view = QPlainTextEdit(self)
        self.view.setObjectName('Console')
        self.view.setReadOnly(True)
        self.view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.view.setMaximumBlockCount(MAX_LINES)
        self.view.setUndoRedoEnabled(False)
        layout.addWidget(self.view, 1)
        self.colours = {'error': '#9C2B4B', 'warning': '#8A5418', 'info': None, 'muted': '#5E5A6E'}
        self.retranslate()

    # ---- data
    def append(self, job_id: str, stream: str, text: str, level: str | None = None):
        level = level or classify(stream, text)
        self.entries.append((job_id, level, stream, text))
        if len(self.entries) > MAX_LINES:
            del self.entries[: len(self.entries) - MAX_LINES]
        if self._visible(job_id, level, text):
            self._write(level, text)

    def _visible(self, job_id, level, text) -> bool:
        wanted = self.level.currentData() or 'all'
        if wanted == 'warning' and level not in ('warning', 'error'):
            return False
        if wanted == 'error' and level != 'error':
            return False
        if wanted == 'info' and level != 'info':
            return False
        if self.job_filter and job_id != self.job_filter:
            return False
        needle = self.search.text().strip().lower()
        return not needle or needle in text.lower()

    def _write(self, level, text):
        bar = self.view.verticalScrollBar()
        cursor = QTextCursor(self.view.document())
        cursor.movePosition(QTextCursor.MoveOperation.End)
        fmt = QTextCharFormat()
        colour = self.colours.get(level)
        if colour:
            fmt.setForeground(QColor(colour))
        if self.view.document().isEmpty():
            cursor.insertText(text, fmt)
        else:
            cursor.insertBlock()
            cursor.insertText(text, fmt)
        if self.autoscroll.isChecked():
            bar.setValue(bar.maximum())

    def refilter(self):
        self.view.clear()
        for job_id, level, _stream, text in self.entries:
            if self._visible(job_id, level, text):
                self._write(level, text)

    def set_job_filter(self, job_id):
        self.job_filter = job_id
        self.refilter()

    def clear(self):
        self.entries.clear()
        self.view.clear()

    def visible_text(self) -> str:
        return self.view.toPlainText()

    def copy_all(self):
        cursor = self.view.textCursor()
        QApplication.clipboard().setText(cursor.selectedText().replace(' ', '\n') if cursor.hasSelection()
                                         else self.visible_text())

    def export(self, path) -> Path:
        path = Path(path)
        path.write_text(self.visible_text() + '\n', encoding='utf-8', newline='\n')
        return path

    def export_dialog(self):
        path, _ = QFileDialog.getSaveFileName(self, tr('log.export_title'), 'sparse-dft-log.txt', 'Text (*.txt *.log)')
        if path:
            self.export(path)

    # ---- appearance
    def set_theme(self, p: dict):
        self.colours = {'error': p['err_text'], 'warning': p['warn_text'], 'info': None, 'muted': p['muted']}
        colour = p['muted']
        self.copy_button.setIcon(icons.icon('copy', colour, 16))
        self.export_button.setIcon(icons.icon('export', colour, 16))
        self.clear_button.setIcon(icons.icon('trash', colour, 16))
        self.refilter()

    def set_mono_font(self, font):
        self.view.setFont(font)

    def retranslate(self):
        self.title.setText(tr('log.title'))
        for i, level in enumerate(LEVELS):
            self.level.setItemText(i, tr(f'log.{level}'))
        self.search.setPlaceholderText(tr('log.search'))
        self.autoscroll.setText(tr('log.autoscroll'))
        # Icon-only buttons keep the toolbar narrow in every language; text goes to tooltips.
        self.copy_button.setToolTip(tr('btn.copy'))
        self.export_button.setToolTip(tr('log.export_title'))
        self.clear_button.setToolTip(tr('btn.clear'))
