"""Page scaffolding: header, scrollable body and multi-command form pages."""
from __future__ import annotations

from PyQt6.QtWidgets import QLabel, QScrollArea, QTabWidget, QVBoxLayout, QWidget

from ..forms import CommandForm
from ..i18n import translator, tr
from ..widgets import Card


class Page(QWidget):
    """Title, subtitle and a scrollable column of cards."""

    key = 'page'

    def __init__(self, window, parent=None, *, scroll=True):
        super().__init__(parent)
        self.main = window
        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 20, 20, 12)
        outer.setSpacing(4)
        self.title = QLabel(self)
        self.title.setProperty('role', 'title')
        self.subtitle = QLabel(self)
        self.subtitle.setProperty('role', 'subtitle')
        self.subtitle.setWordWrap(True)
        outer.addWidget(self.title)
        outer.addWidget(self.subtitle)
        outer.addSpacing(10)
        self.body_widget = QWidget()
        self.body_widget.setObjectName('PageBody')
        self.body = QVBoxLayout(self.body_widget)
        self.body.setContentsMargins(0, 0, 6, 12)
        self.body.setSpacing(12)
        if scroll:
            area = QScrollArea(self)
            area.setWidgetResizable(True)
            area.setWidget(self.body_widget)
            outer.addWidget(area, 1)
        else:
            outer.addWidget(self.body_widget, 1)

    def retranslate(self):
        self.title.setText(tr(f'nav.{self.key}'))
        self.subtitle.setText(tr(f'page.{self.key}.sub'))

    def set_theme(self, p: dict):
        pass

    def card(self) -> Card:
        card = Card(self.body_widget)
        self.body.addWidget(card)
        return card


class CommandPage(Page):
    """One CommandForm per sparse_dft subcommand; several commands are shown as tabs."""

    commands: tuple = ()

    def __init__(self, window, parent=None, commands=None):
        super().__init__(window, parent)
        self.forms: dict[str, CommandForm] = {}
        self.tabs = None
        names = [c for c in (commands or self.commands) if c in window.available_commands]
        self.extra_top = QVBoxLayout()
        self.extra_top.setSpacing(12)
        self.body.addLayout(self.extra_top)
        if len(names) > 1:
            self.tabs = QTabWidget(self.body_widget)
            self.tabs.setDocumentMode(False)
            self.body.addWidget(self.tabs)
        for name in names:
            form = window.make_form(name)
            if self.tabs is not None:
                holder = QWidget()
                layout = QVBoxLayout(holder)
                layout.setContentsMargins(12, 14, 12, 12)
                layout.addWidget(form)
                self.tabs.addTab(holder, name)
            else:
                self.body.addWidget(form)
            self.forms[name] = form
        self.body.addStretch(1)

    def retranslate(self):
        super().retranslate()
        for index, name in enumerate(self.forms):
            self.forms[name].retranslate()
            if self.tabs is not None:
                self.tabs.setTabText(index, translator().command_title(name))
