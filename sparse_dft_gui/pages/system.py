"""Diagnostics (backend-info, doctor, validate) and the task queue."""
from __future__ import annotations

import datetime as _dt
from pathlib import Path

from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (QAbstractItemView, QGridLayout, QHBoxLayout, QHeaderView, QLabel, QPushButton,
                             QTableWidget, QTableWidgetItem)

from sparse_dft.resources import find_executable, platform_tag, resource_root

from ..i18n import tr
from ..jobs import format_duration
from ..widgets import Chip
from .base import CommandPage, Page


class DiagnosticsPage(CommandPage):
    key = 'diagnostics'
    commands = ('backend-info', 'doctor', 'validate', 'progress')

    def __init__(self, window, parent=None):
        super().__init__(window, parent)
        card = self.card()
        self.body.removeWidget(card)
        self.extra_top.addWidget(card)
        self.exe_title = QLabel(card)
        self.exe_title.setProperty('role', 'section')
        card.body.addWidget(self.exe_title)
        self.exe_grid = QGridLayout()
        self.exe_grid.setHorizontalSpacing(12)
        self.exe_grid.setVerticalSpacing(6)
        self.exe_grid.setColumnStretch(2, 1)
        card.body.addLayout(self.exe_grid)
        self.platform = QLabel(card)
        self.platform.setProperty('role', 'muted')
        card.body.addWidget(self.platform)
        self.refresh_button = QPushButton(card)
        self.refresh_button.setProperty('kind', 'ghost')
        self.refresh_button.clicked.connect(self.refresh)
        card.body.addWidget(self.refresh_button, 0, Qt.AlignmentFlag.AlignLeft)
        self.rows = {}
        for i, name in enumerate(('sdft_scf', 'sdft_relax_step')):
            label = QLabel(name, card)
            label.setFont(window.mono_font)
            chip = Chip('idle', card)
            path = QLabel(card)
            path.setProperty('role', 'muted')
            path.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            path.setWordWrap(True)
            self.exe_grid.addWidget(label, i, 0)
            self.exe_grid.addWidget(chip, i, 1)
            self.exe_grid.addWidget(path, i, 2)
            self.rows[name] = (chip, path)
        self.refresh()

    def refresh(self):
        env = self.main.prefs.job_env() if hasattr(self.main, 'prefs') else {}
        import os
        saved = os.environ.get('SDFT_SCF_EXE')
        try:
            if env.get('SDFT_SCF_EXE'):
                os.environ['SDFT_SCF_EXE'] = env['SDFT_SCF_EXE']
            for name, (chip, path) in self.rows.items():
                try:
                    found = find_executable(name)
                    chip.set_state('finished', tr('diag.found'))
                    path.setText(str(found))
                except Exception as error:  # noqa: BLE001
                    chip.set_state('failed', tr('diag.missing'))
                    path.setText(str(getattr(error, 'message', error))[:400])
        finally:
            if saved is None:
                os.environ.pop('SDFT_SCF_EXE', None)
            else:
                os.environ['SDFT_SCF_EXE'] = saved
        self.platform.setText(f'platform: {platform_tag()} · resources: {resource_root()}')

    def retranslate(self):
        super().retranslate()
        self.exe_title.setText(tr('diag.executables'))
        self.refresh_button.setText(tr('btn.refresh'))
        self.refresh()


COLUMNS = ('queue.label', 'queue.command', 'queue.state', 'queue.started', 'queue.duration',
           'queue.returncode', 'queue.output')


class QueuePage(Page):
    key = 'queue'

    def __init__(self, window, parent=None):
        super().__init__(window, parent, scroll=False)
        self.rows: dict[str, int] = {}
        bar = QHBoxLayout()
        bar.setSpacing(8)
        self.summary = QLabel(self.body_widget)
        self.summary.setProperty('role', 'muted')
        self.cancel_button = QPushButton(self.body_widget)
        self.cancel_button.setProperty('kind', 'danger')
        self.rerun_button = QPushButton(self.body_widget)
        self.open_button = QPushButton(self.body_widget)
        self.log_button = QPushButton(self.body_widget)
        self.remove_button = QPushButton(self.body_widget)
        for button in (self.rerun_button, self.open_button, self.log_button, self.remove_button):
            button.setProperty('kind', 'ghost')
        self.cancel_button.clicked.connect(lambda: self._act('cancel'))
        self.rerun_button.clicked.connect(lambda: self._act('rerun'))
        self.open_button.clicked.connect(lambda: self._act('open'))
        self.log_button.clicked.connect(lambda: self._act('log'))
        self.remove_button.clicked.connect(self.remove_finished)
        # Job actions on the first row, housekeeping on the second: fits 1100 px in every language.
        # Run control and inspection on separate rows: fits a 1100 px window in every language.
        for button in (self.cancel_button, self.rerun_button):
            bar.addWidget(button)
        bar.addStretch(1)
        bar.addWidget(self.remove_button)
        self.body.addLayout(bar)
        second = QHBoxLayout()
        second.setSpacing(8)
        second.addWidget(self.open_button)
        second.addWidget(self.log_button)
        second.addSpacing(8)
        self.summary.setWordWrap(True)
        second.addWidget(self.summary, 1)
        self.body.addLayout(second)
        self.table = QTableWidget(0, len(COLUMNS), self.body_widget)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        self.table.verticalHeader().setDefaultSectionSize(38)
        header = self.table.horizontalHeader()
        for col in range(len(COLUMNS)):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.Interactive)
        self.table.itemSelectionChanged.connect(self._update_buttons)
        self.table.cellDoubleClicked.connect(lambda *_: self._act('log'))
        self.body.addWidget(self.table, 1)
        self.empty = QLabel(self.body_widget)
        self.empty.setProperty('role', 'muted')
        self.empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.body.addWidget(self.empty)
        self._update_buttons()

    # ---- rows
    def _job_at(self, row):
        item = self.table.item(row, 0)
        if item is None:
            return None
        job_id = item.data(Qt.ItemDataRole.UserRole)
        return next((j for j in self.main.jobs.jobs if j.id == job_id), None)

    def selected_job(self):
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        return self._job_at(rows[0].row()) if rows else None

    def add_job(self, job):
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.rows[job.id] = row
        for col in range(len(COLUMNS)):
            if col == 2:
                self.table.setCellWidget(row, col, Chip(job.state, self.table))
            else:
                self.table.setItem(row, col, QTableWidgetItem(''))
        self.table.item(row, 0).setData(Qt.ItemDataRole.UserRole, job.id)
        self.update_job(job)
        self.table.selectRow(row)

    def update_job(self, job):
        row = self.rows.get(job.id)
        if row is None:
            return
        started = _dt.datetime.fromtimestamp(job.started).strftime('%H:%M:%S') if job.started else '—'
        values = {0: job.label, 1: job.display, 3: started, 4: format_duration(job.duration),
                  5: '' if job.returncode is None else str(job.returncode), 6: job.output_dir or ''}
        for col, text in values.items():
            item = self.table.item(row, col)
            if item is not None and item.text() != text:
                item.setText(text)
                item.setToolTip(text)
        chip = self.table.cellWidget(row, 2)
        if chip is not None:
            chip.set_state(job.state)
        self._update_summary()
        self._update_buttons()

    def remove_finished(self):
        self.main.jobs.remove_finished()
        self.table.setRowCount(0)
        self.rows.clear()
        for job in self.main.jobs.jobs:
            self.add_job(job)
        self._update_summary()

    def _update_summary(self):
        jobs = self.main.jobs.jobs
        counts = {s: sum(1 for j in jobs if j.state == s) for s in ('running', 'finished', 'failed')}
        self.summary.setText(tr('queue.summary', **counts))
        self.empty.setVisible(not jobs)
        self.table.setVisible(bool(jobs))

    def _update_buttons(self):
        job = self.selected_job()
        self.cancel_button.setEnabled(bool(job and not job.finished()))
        self.rerun_button.setEnabled(bool(job and job.finished()))
        self.open_button.setEnabled(bool(job))
        self.log_button.setEnabled(bool(job))

    def _act(self, action):
        job = self.selected_job()
        if job is None:
            return
        if action == 'cancel':
            self.main.jobs.cancel(job)
        elif action == 'rerun':
            self.main.jobs.rerun(job)
        elif action == 'log':
            self.main.focus_job(job)
        elif action == 'open':
            folder = job.output_dir if job.output_dir and Path(job.output_dir).exists() else str(job.workdir)
            if not Path(folder).exists():
                self.main.notify(tr('msg.no_output'), 'warning')
                return
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(folder).resolve())))

    def retranslate(self):
        super().retranslate()
        self.table.setHorizontalHeaderLabels([tr(c) for c in COLUMNS])
        self.cancel_button.setText(tr('btn.cancel'))
        self.rerun_button.setText(tr('btn.rerun'))
        self.open_button.setText(tr('btn.open_folder'))
        self.log_button.setText(tr('btn.show_log'))
        self.remove_button.setText(tr('btn.remove'))
        for button in (self.cancel_button, self.rerun_button, self.open_button, self.log_button, self.remove_button):
            button.setMinimumWidth(button.sizeHint().width())
        self.empty.setText(tr('queue.empty'))
        for job in self.main.jobs.jobs:
            self.update_job(job)
        self._update_summary()

    def set_theme(self, p):
        from .. import icons
        self.cancel_button.setIcon(icons.icon('stop', '#3A1020', 16))
        self.rerun_button.setIcon(icons.icon('rerun', p['muted'], 16))
        self.open_button.setIcon(icons.icon('folder', p['muted'], 16))
        self.log_button.setIcon(icons.icon('file', p['muted'], 16))
        self.remove_button.setIcon(icons.icon('trash', p['muted'], 16))
