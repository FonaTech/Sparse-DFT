"""Structure classification card and "Apply recommended" preview (GUI side of :mod:`sparse_dft.presets`)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QHBoxLayout, QHeaderView, QLabel,
                             QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout)

from .i18n import tr
from .widgets import Card, no_wheel

TASKS = ('scf', 'bands', 'dos', 'relax')
ACCURACY = ('low', 'standard', 'high', 'production')
# Options the recommendation owns: a value missing from the suggestion clears
# the field (unless the user edited it), so a previous structure's spin seed
# or +U never leaks into the next one.
PRESET_DESTS = ('accuracy', 'scf_grid', 'mesh_mode', 'smearing_ev', 'spin_channels', 'initial_moments',
                'sparsity', 'cell', 'hubbard', 'dos_grid')
# Tied to the loaded structure: always rewritten.
ALWAYS = ('structure',)


def analyze(path, paw_root=None, accuracy='standard') -> dict:
    """Profile, per-task suggestions and band path for a structure file (runs in a worker thread)."""
    from sparse_dft import presets
    from sparse_dft.structure import read_structure
    structure = read_structure(path)
    errors = []
    profile = None
    if paw_root:
        try:
            profile = presets.analyze_structure(structure, paw_root)
        except Exception as exc:  # noqa: BLE001 - a wrong PAW root only drops the PAW-based items
            errors.append(f'PAW root: {exc}')
    if profile is None:
        profile = presets.analyze_structure(structure)
    suggestions = {task: presets.suggest(profile, task, structure, accuracy=accuracy) for task in TASKS}
    path_data = None
    if profile.geometric_dimensionality > 0:
        try:
            path_data = presets.band_path(structure)
        except Exception as exc:  # noqa: BLE001
            errors.append(f'band path: {exc}')
    return {'path': str(path), 'accuracy': accuracy, 'profile': profile.as_dict(), 'suggestions': suggestions,
            'band_path': path_data, 'hubbard': presets.suggested_hubbard(profile), 'errors': errors}


def band_path_target(output_root, structure_path) -> Path:
    """``<output root>/band_paths/<stem>-<hash>.json``; the hash keeps same-named files apart."""
    resolved = str(Path(structure_path).expanduser().resolve())
    digest = hashlib.sha256(resolved.encode('utf-8')).hexdigest()[:8]
    return Path(output_root).expanduser() / 'band_paths' / f'{Path(structure_path).stem}-{digest}.json'


def write_band_path(result: dict, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result['band_path'], indent=1) + '\n', encoding='utf-8')
    return target


def form_values(result: dict, task: str, *, hubbard: bool = False, band_path_file=None) -> dict:
    """Typed ``{dest: value}`` for the ``task`` form; ``None`` resets an owned option to its default."""
    values = {dest: None for dest in PRESET_DESTS}
    values.update(result['suggestions'][task]['options'])
    if hubbard and result.get('hubbard'):
        values['hubbard'] = list(result['hubbard']['values'])
    values['structure'] = result['path']
    if task == 'bands' and band_path_file is not None and result.get('band_path'):
        values['path'] = str(band_path_file)
    return values


def display(value) -> str:
    if isinstance(value, bool):
        return '✓' if value else '✗'
    if isinstance(value, (list, tuple)):
        text = ' '.join(str(v) for v in value if str(v) != '')
        return text or tr('label.unset')
    return tr('label.unset') if value in ('', None) else str(value)


def _table(parent, columns: int) -> QTableWidget:
    table = QTableWidget(0, columns, parent)
    table.verticalHeader().setVisible(False)
    table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
    table.setWordWrap(True)
    header = table.horizontalHeader()
    for col in range(columns - 1):
        header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
    header.setSectionResizeMode(columns - 1, QHeaderView.ResizeMode.Stretch)
    return table


def _fill(table: QTableWidget, rows: list[list[str]], tooltips: list[str] | None = None):
    table.setRowCount(len(rows))
    for r, cells in enumerate(rows):
        for c, text in enumerate(cells):
            item = QTableWidgetItem(text)
            if tooltips:
                item.setToolTip(tooltips[r])
            table.setItem(r, c, item)
    table.resizeRowsToContents()


class ClassificationCard(Card):
    """Category, reasons and warnings of the loaded structure, and the recommended options per task."""

    apply_requested = pyqtSignal()
    accuracy_changed = pyqtSignal(str)
    auto_apply_changed = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.result = None
        self.state = 'none'  # none | analyzing | ready | failed
        self.error = ''
        self._base_notes: list[str] = []
        head = QHBoxLayout()
        self.title = QLabel(self)
        self.title.setProperty('role', 'section')
        self.category = QLabel(self)
        self.category.setProperty('role', 'subtitle')
        head.addWidget(self.title)
        head.addWidget(self.category, 1)
        self.body.addLayout(head)
        self.details = QLabel(self)
        self.details.setWordWrap(True)
        self.body.addWidget(self.details)
        self.notes = QLabel(self)
        self.notes.setProperty('role', 'muted')
        self.notes.setWordWrap(True)
        self.body.addWidget(self.notes)
        controls = QHBoxLayout()
        self.task_label = QLabel(self)
        self.task = no_wheel(QComboBox(self))
        for task in TASKS:
            self.task.addItem(task, task)
        self.task.currentIndexChanged.connect(self._render_table)
        self.accuracy_label = QLabel(self)
        self.accuracy = no_wheel(QComboBox(self))
        for level in ACCURACY:
            self.accuracy.addItem(level, level)
        self.accuracy.setCurrentIndex(ACCURACY.index('standard'))
        self.accuracy.currentIndexChanged.connect(lambda: self.accuracy_changed.emit(self.accuracy.currentData()))
        controls.addWidget(self.task_label)
        controls.addWidget(self.task)
        controls.addSpacing(12)
        controls.addWidget(self.accuracy_label)
        controls.addWidget(self.accuracy)
        controls.addStretch(1)
        self.body.addLayout(controls)
        self.table = _table(self, 4)
        self.table.setMinimumHeight(160)
        self.body.addWidget(self.table)
        bottom = QHBoxLayout()
        self.hubbard = QCheckBox(self)
        self.hubbard.setEnabled(False)
        self.auto_apply = QCheckBox(self)
        self.auto_apply.toggled.connect(self.auto_apply_changed)
        self.apply_button = QPushButton(self)
        self.apply_button.setProperty('kind', 'accent')
        self.apply_button.setEnabled(False)
        self.apply_button.clicked.connect(self.apply_requested)
        bottom.addWidget(self.hubbard)
        bottom.addStretch(1)
        bottom.addWidget(self.auto_apply)
        bottom.addWidget(self.apply_button)
        self.body.addLayout(bottom)

    # ---- state
    def selected_task(self) -> str:
        return self.task.currentData()

    def set_accuracy(self, level: str):
        self.accuracy.blockSignals(True)
        self.accuracy.setCurrentIndex(ACCURACY.index(level) if level in ACCURACY else ACCURACY.index('standard'))
        self.accuracy.blockSignals(False)

    def set_analyzing(self):
        self.state, self.error = 'analyzing', ''
        self.apply_button.setEnabled(False)
        self._render()

    def set_failed(self, text: str):
        self.state, self.error, self.result = 'failed', text, None
        self.apply_button.setEnabled(False)
        self._render()

    def set_result(self, result: dict):
        self.state, self.error, self.result = 'ready', '', result
        hubbard = result.get('hubbard')
        self.hubbard.setChecked(False)  # +U is suggested, never switched on by default
        self.hubbard.setEnabled(bool(hubbard))
        self.hubbard.setToolTip(hubbard['note'] if hubbard else '')
        self.apply_button.setEnabled(True)
        self._render()

    # ---- text
    def _render(self):
        self.category.setText('')
        self.details.setText('')
        self.notes.setText('')
        if self.state != 'ready':
            text = {'none': tr('preset.none'), 'analyzing': tr('preset.analyzing'),
                    'failed': tr('preset.failed', message=self.error)}[self.state]
            self.details.setText(text)
            self.table.setRowCount(0)
            self.hubbard.setText(tr('preset.hubbard_none'))
            return
        r, p = self.result, self.result['profile']
        self.category.setText(f'{tr("preset.category." + p["category"])} · {p["formula"]}')
        lines = [tr('preset.dimensionality', geometric=p['geometric_dimensionality'],
                    bonded=p['bonded_dimensionality'], natoms=p['natoms'])]
        sym = p.get('symmetry') or {}
        if 'spacegroup_international' in sym:
            lines.append(tr('preset.symmetry', symbol=sym['spacegroup_international'],
                            number=sym['spacegroup_number'], operations=sym.get('operations', '?')))
        if p.get('electrons') is not None:
            lines.append(tr('preset.electrons', electrons=f'{p["electrons"]:g}') +
                         (f' ({tr("preset.odd")})' if p.get('odd_electrons') else ''))
        if p.get('tags'):
            lines.append(f'{tr("preset.tags")}: {", ".join(p["tags"])}')
        self.details.setText('\n'.join(lines))
        notes = []
        if p.get('reasons'):
            notes.append(f'{tr("preset.reasons")}: ' + '; '.join(p['reasons']))
        warnings = list(p.get('warnings') or []) + list(r.get('errors') or [])
        if warnings:
            notes.append(f'{tr("preset.warnings")}: ' + '; '.join(warnings))
        self._base_notes = notes
        hubbard = r.get('hubbard')
        self.hubbard.setText(tr('preset.hubbard', values=' '.join(hubbard['values'])) if hubbard
                             else tr('preset.hubbard_none'))
        self._render_table()

    def _render_table(self, *_):
        if self.state != 'ready' or self.result is None:
            return
        rows, tips = [], []
        for item in self.result['suggestions'][self.selected_task()]['items']:
            automatic = item['value'] is None or (item['key'] == 'bands' and item['value'] == 0)
            value = tr('preset.automatic') if automatic else display(item['value'])
            if item['key'] == 'cutoff_ev' and not automatic:
                value = f'{item["value"]:g} eV ({tr("preset.automatic")})'
            option = item['option'] or f'({item["key"]})'
            rows.append([option, value, tr(f'preset.basis.{item["basis"]}'), item['note']])
            tips.append(item['note'])
        _fill(self.table, rows, tips)
        notes = list(self._base_notes)
        task_notes = self.result['suggestions'][self.selected_task()].get('notes') or []
        if task_notes:
            notes.append(f'{tr("preset.notes")}: ' + '; '.join(task_notes))
        self.notes.setText('\n'.join(notes))

    def retranslate(self):
        self.title.setText(tr('preset.title'))
        self.task_label.setText(tr('preset.task'))
        self.accuracy_label.setText(tr('preset.accuracy'))
        self.auto_apply.setText(tr('preset.auto_apply'))
        self.apply_button.setText(tr('preset.apply'))
        self.table.setHorizontalHeaderLabels([tr('preset.option'), tr('preset.value'), tr('preset.basis'),
                                              tr('preset.note')])
        self._render()


ACTIONS = ('set', 'kept', 'same')


class PreviewDialog(QDialog):
    """Field-by-field preview of the recommended values; ``rows_for(force)`` builds the rows."""

    def __init__(self, rows_for, parent=None):
        super().__init__(parent)
        self.rows_for = rows_for
        self.setWindowTitle(tr('preset.preview_title'))
        self.resize(820, 520)
        layout = QVBoxLayout(self)
        self.summary = QLabel(self)
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.table = _table(self, 5)
        layout.addWidget(self.table, 1)
        options = QHBoxLayout()
        self.force = QCheckBox(tr('preset.force'), self)
        self.force.toggled.connect(self.refresh)
        self.show_same = QCheckBox(tr('preset.show_same'), self)
        self.show_same.toggled.connect(self.refresh)
        options.addWidget(self.force)
        options.addWidget(self.show_same)
        options.addStretch(1)
        layout.addLayout(options)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText(tr('preset.apply'))
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText(tr('btn.cancel'))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.table.setHorizontalHeaderLabels([tr('preset.command'), tr('preset.option'), tr('preset.current'),
                                              tr('preset.new'), tr('preset.action')])
        self.refresh()

    def refresh(self, *_):
        rows = self.rows_for(self.force.isChecked())
        counts = {a: sum(1 for r in rows if r['action'] == a) for a in ACTIONS}
        self.summary.setText(tr('preset.preview_summary', set=counts['set'], kept=counts['kept'], same=counts['same']))
        shown = [r for r in rows if self.show_same.isChecked() or r['action'] != 'same']
        _fill(self.table, [[r['command'], r['dest'], display(r['current']), display(r['new']),
                            tr(f'preset.action.{r["action"]}')] for r in shown])
