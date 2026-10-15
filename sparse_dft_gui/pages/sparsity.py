"""Sparsity page: explains components S1-S7 and applies presets to the SCF/Bands forms."""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QButtonGroup, QComboBox, QHBoxLayout, QHeaderView, QLabel, QPushButton, QRadioButton,
                             QTableWidget, QTableWidgetItem, QVBoxLayout)

from ..i18n import tr
from ..widgets import no_wheel, run_async
from .base import Page

LEVELS = ('conservative', 'balanced', 'aggressive')

COMPONENTS = (
    ('S1', 'A', '--density-cutoff-factor', 'density_cutoff_factor'),
    ('S2', 'A', '--xc-padding', 'xc_padding'),
    ('S3', 'E', '--eigen-search-factor', 'eigen_search_factor'),
    ('S4', 'E', '--block-ortho', 'block_ortho'),
    ('S5', 'E', '--paw-compensation', 'paw_compensation'),
    ('S6', 'E', '--projector-space, --projector-radius-factor', 'projector_space'),
    ('S7', 'E', '--search-precision', 'search_precision'),
)
COMPONENT_DESTS = ('density_cutoff_factor', 'xc_padding', 'eigen_search_factor', 'block_ortho',
                   'paw_compensation', 'projector_space', 'projector_radius_factor', 'search_precision')
# Preset -> {dest: value}; unset components are cleared.  Values for the search
# preset follow the calibrated defaults in sparse_dft/sparsity_rules.json (S3.default,
# S4.all, S5.blocks).
PRESETS = {
    'off': {'sparsity': 'off'},
    'search': {'sparsity': 'custom', 'eigen_search_factor': '0.0001', 'block_ortho': '1',
               'paw_compensation': 'blocks'},
    'auto': {'sparsity': 'auto'},
}


def implementation_status(dest: str) -> str:
    try:
        from sparse_dft.sparsity import IMPLEMENTED, PROJECTOR_SPACE, SEARCH_PRECISION
    except Exception:  # noqa: BLE001
        return 'implemented'
    allowed = IMPLEMENTED.get(dest)
    if allowed is None:
        return 'implemented'
    full = {'projector_space': set(PROJECTOR_SPACE), 'search_precision': set(SEARCH_PRECISION)}.get(dest, allowed)
    if dest == 'search_precision' and allowed == {'fp64'}:
        return 'not_implemented'
    return 'implemented' if set(allowed) >= set(full) else 'partial'


def measured_runs() -> list[dict]:
    """v15 sweep: --sparsity auto vs off wall time per system (sparse_dft/sparsity_measured.json)."""
    import json
    from pathlib import Path
    try:
        import sparse_dft
        data = json.loads((Path(sparse_dft.__file__).with_name('sparsity_measured.json')).read_text(encoding='utf-8'))
    except Exception:  # noqa: BLE001
        return []
    return list(data.get('runs', []))


def rule_rows() -> list[tuple[str, str, str, str]]:
    """(component, rule id, when -> value, evidence) for every calibrated rule."""
    try:
        from sparse_dft.sparsity import load_rules
        rules = load_rules()
    except Exception:  # noqa: BLE001
        return []
    s1 = rules['density_cutoff']
    rows = [('S1', s1['id'], f"tail <= {s1['tail_budget']:g} ({s1['metric']}); fallback f{s1['fallback_factor']:g}",
             s1.get('evidence', ''))]
    for cond in s1.get('stricter_when', []):
        rows.append(('S1', cond['id'], f"{cond['feature']}: factor >= {cond.get('minimum_factor')}",
                     cond.get('evidence', '')))
    for sid, key in (('S2', 'xc_padding'), ('S3', 'eigen_search_factor'), ('S4', 'block_ortho'),
                     ('S5', 'paw_compensation'), ('S6', 'projector_space'), ('S6', 'projector_radius_factor'),
                     ('S7', 'search_precision')):
        for rule in rules.get(key, []):
            when = ', '.join(f'{k}={v}' for k, v in rule.get('when', {}).items()) or 'always'
            rows.append((sid, rule['id'], f"{when} -> {rule['value']}", rule.get('evidence', '')))
    return rows


AUTO_DESTS = ('sparsity_level',)


def preset_values(name: str, command: str, level: str | None = None) -> dict:
    values = {dest: '' for dest in COMPONENT_DESTS + AUTO_DESTS}
    values.update(PRESETS[name])
    if name == 'auto' and level:
        values['sparsity_level'] = level
    if command in ('bands', 'dos') and values.get('sparsity') == 'auto':
        # NSCF refuses auto; reuse the profile recorded by the SCF (the checkpoint keeps it).
        values['sparsity'] = 'off'
        values['sparsity_level'] = ''
    return values


def _table(parent, columns):
    table = QTableWidget(0, columns, parent)
    table.verticalHeader().setVisible(False)
    table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
    table.setWordWrap(True)
    header = table.horizontalHeader()
    for c in range(columns - 1):
        header.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
    header.setSectionResizeMode(columns - 1, QHeaderView.ResizeMode.Stretch)
    return table


def _fill(table, rows, mono=None, mono_columns=()):
    table.setRowCount(len(rows))
    for r, row in enumerate(rows):
        for c, text in enumerate(row):
            item = QTableWidgetItem(str(text))
            if mono is not None and c in mono_columns:
                item.setFont(mono)
            table.setItem(r, c, item)
    table.resizeRowsToContents()
    height = table.horizontalHeader().height() + sum(table.rowHeight(r) for r in range(table.rowCount())) + 6
    table.setFixedHeight(min(height, 420))


class SparsityPage(Page):
    key = 'sparsity'

    def __init__(self, window, parent=None):
        super().__init__(window, parent)
        intro = self.card()
        self.intro = QLabel(intro)
        self.intro.setWordWrap(True)
        intro.body.addWidget(self.intro)
        table_card = self.card()
        self.table = QTableWidget(len(COMPONENTS), 5, table_card)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.table.setWordWrap(True)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        table_card.body.addWidget(self.table)
        preset_card = self.card()
        self.preset_title = QLabel(preset_card)
        self.preset_title.setProperty('role', 'section')
        preset_card.body.addWidget(self.preset_title)
        self.group = QButtonGroup(self)
        self.radios = {}
        column = QVBoxLayout()
        for name in PRESETS:
            radio = QRadioButton(preset_card)
            self.group.addButton(radio)
            self.radios[name] = radio
            column.addWidget(radio)
        self.radios['off'].setChecked(True)
        preset_card.body.addLayout(column)
        row = QHBoxLayout()
        self.apply_button = QPushButton(preset_card)
        self.apply_button.setProperty('kind', 'accent')
        self.apply_button.clicked.connect(self.apply_preset)
        row.addWidget(self.apply_button)
        row.addStretch(1)
        preset_card.body.addLayout(row)
        self.rules = QLabel(preset_card)
        self.rules.setProperty('role', 'muted')
        self.rules.setWordWrap(True)
        self.rules.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        preset_card.body.addWidget(self.rules)
        # auto controls: level, a probe-backed preview for the loaded structure, rules and measurements
        auto_card = self.card()
        self.auto_title = QLabel(auto_card)
        self.auto_title.setProperty('role', 'section')
        auto_card.body.addWidget(self.auto_title)
        line = QHBoxLayout()
        self.level_label = QLabel(auto_card)
        self.level = no_wheel(QComboBox(auto_card))
        for level in LEVELS:
            self.level.addItem(level, level)
        self.level.setCurrentIndex(LEVELS.index('balanced'))
        self.apply_auto_button = QPushButton(auto_card)
        self.apply_auto_button.setProperty('kind', 'accent')
        self.apply_auto_button.clicked.connect(lambda: self.apply_preset('auto', level=self.level.currentData()))
        self.preview_button = QPushButton(auto_card)
        self.preview_button.clicked.connect(self.preview)
        for w in (self.level_label, self.level, self.apply_auto_button, self.preview_button):
            line.addWidget(w)
        line.addStretch(1)
        auto_card.body.addLayout(line)
        self.level_note = QLabel(auto_card)
        self.level_note.setProperty('role', 'muted')
        self.level_note.setWordWrap(True)
        auto_card.body.addWidget(self.level_note)
        self.preview_status = QLabel(auto_card)
        self.preview_status.setWordWrap(True)
        auto_card.body.addWidget(self.preview_status)
        self.preview_table = _table(auto_card, 6)
        self.preview_table.hide()
        auto_card.body.addWidget(self.preview_table)
        self.measured_title = QLabel(auto_card)
        self.measured_title.setProperty('role', 'section')
        auto_card.body.addWidget(self.measured_title)
        self.measured_table = _table(auto_card, 6)
        auto_card.body.addWidget(self.measured_table)
        self.rules_title = QLabel(auto_card)
        self.rules_title.setProperty('role', 'section')
        auto_card.body.addWidget(self.rules_title)
        self.rule_table = _table(auto_card, 4)
        auto_card.body.addWidget(self.rule_table)
        self.preview_result = None
        self._preview_key = None
        self.body.addStretch(1)

    def selected(self) -> str:
        for name, radio in self.radios.items():
            if radio.isChecked():
                return name
        return 'off'

    def apply_preset(self, name=None, *, level=None):
        name = name or self.selected()
        if name == 'auto' and level is None:
            level = self.level.currentData()
        for command in ('scf', 'bands', 'dos'):
            form = self.main.forms.get(command)
            if form is not None:
                form.set_values({k: v for k, v in preset_values(name, command, level).items() if k in form.fields})
        self.main.notify(tr('sparsity.applied'), 'ok')

    # ---- probe preview
    def _preview_inputs(self):
        form = self.main.forms.get('scf')
        if form is None:
            return None
        values = {d: form.fields[d].value() for d in ('structure', 'paw_root', 'smearing_ev', 'spin_channels', 'cutoff')
                  if d in form.fields}
        structure = (values.get('structure') or '').strip()
        paw = (values.get('paw_root') or '').strip() or self.main.prefs.get('paw_root')
        if not structure or not paw:
            return None
        grid = form.fields['scf_grid'].value() if 'scf_grid' in form.fields else None
        return {'structure': structure, 'paw_root': paw,
                'scf_grid': [int(x) for x in grid] if grid and all(str(x).strip() for x in grid) else None,
                'smearing_ev': float(values.get('smearing_ev') or 0.0),
                'spin_channels': int(values.get('spin_channels') or 1),
                'cutoff_ev': float(values['cutoff']) if str(values.get('cutoff') or '').strip() else None}

    def preview(self):
        inputs = self._preview_inputs()
        if inputs is None:
            self.preview_status.setText(tr('sparsity.preview_needs'))
            return
        import tempfile
        from sparse_dft.scf import preview_sparsity
        out = tempfile.mkdtemp(prefix='sparse_dft_sparsity_preview_')
        key = (inputs['structure'], str(inputs['scf_grid']), inputs['smearing_ev'])
        self._preview_key = key
        self.preview_status.setText(tr('sparsity.preview_running'))
        self.preview_button.setEnabled(False)
        run_async(preview_sparsity, inputs['structure'], inputs['paw_root'], out, scf_grid=inputs['scf_grid'],
                  smearing_ev=inputs['smearing_ev'], spin_channels=inputs['spin_channels'], cutoff_ev=inputs['cutoff_ev'],
                  on_done=lambda result, k=key: self._previewed(k, result),
                  on_error=lambda text, k=key: self._preview_failed(k, text))

    def _previewed(self, key, result):
        self.preview_button.setEnabled(True)
        if key != self._preview_key:
            return
        self.preview_result = result
        self.preview_status.setText(tr('sparsity.preview_done', path=key[0], cutoff=f"{result['cutoff_ev']:g}"))
        self._fill_preview()

    def _preview_failed(self, key, text):
        self.preview_button.setEnabled(True)
        if key == self._preview_key:
            self.preview_status.setText(tr('sparsity.preview_failed', message=str(text).splitlines()[-1][:300]))

    def _fill_preview(self):
        result = self.preview_result
        if not result:
            self.preview_table.hide()
            return
        rows = []
        for level in LEVELS:
            v = result['levels'][level]
            c = v['components']
            rows.append((level, f"{c.get('density_cutoff_factor', 16):g}", c.get('xc_padding', ''),
                         f"{v['tail_budget']:.3g}", '—' if v['tail_at_choice'] is None else f"{v['tail_at_choice']:.3g}",
                         f"{v['density_modes_ratio_vs_16x_estimate']:.3f}" + (' · ' + tr('sparsity.fallback')
                                                                               if v['fallback'] else '')))
        self.preview_table.setHorizontalHeaderLabels([tr('sparsity.level'), 'S1 factor', 'S2 padding',
                                                      tr('sparsity.tail_budget'), tr('sparsity.tail_at_choice'),
                                                      tr('sparsity.modes_ratio')])
        _fill(self.preview_table, rows)
        self.preview_table.show()

    def retranslate(self):
        super().retranslate()
        self.intro.setText(tr('sparsity.intro'))
        self.table.setHorizontalHeaderLabels([tr('sparsity.component'), tr('sparsity.class'), tr('sparsity.option'),
                                              tr('sparsity.effect'), tr('sparsity.status')])
        for row, (sid, cls, option, dest) in enumerate(COMPONENTS):
            status = implementation_status(dest)
            cells = [sid, tr('sparsity.class_a' if cls == 'A' else 'sparsity.class_e'), option,
                     tr(f'sparsity.{sid.lower()}'), tr(f'sparsity.{status}')]
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if col == 2:
                    item.setFont(self.main.mono_font)
                self.table.setItem(row, col, item)
        self.table.resizeRowsToContents()
        height = self.table.horizontalHeader().height() + sum(self.table.rowHeight(r) for r in range(self.table.rowCount())) + 6
        self.table.setFixedHeight(height)
        self.preset_title.setText(tr('sparsity.presets'))
        self.radios['off'].setText(tr('sparsity.preset_off'))
        self.radios['search'].setText(tr('sparsity.preset_search'))
        self.radios['auto'].setText(tr('sparsity.preset_auto'))
        self.apply_button.setText(tr('sparsity.apply_to'))
        self.rules.setText(self._rules_text())
        self.auto_title.setText(tr('sparsity.auto_title'))
        self.level_label.setText(tr('sparsity.level'))
        for i, level in enumerate(LEVELS):
            self.level.setItemText(i, tr(f'sparsity.level_{level}'))
        self.apply_auto_button.setText(tr('sparsity.apply_auto'))
        self.preview_button.setText(tr('sparsity.preview'))
        self.level_note.setText(tr('sparsity.level_note'))
        self.measured_title.setText(tr('sparsity.measured_title'))
        self.measured_table.setHorizontalHeaderLabels([tr('sparsity.system'), 'S1 factor', 'S2 padding',
                                                       tr('sparsity.wall_auto'), tr('sparsity.wall_off'),
                                                       tr('sparsity.speedup')])
        _fill(self.measured_table, [(r['system'], f"{r['density_cutoff_factor']:g}", r['xc_padding'],
                                     f"{r['wall_seconds']:.1f} s", f"{r['reference_wall_seconds']:.1f} s",
                                     f"{r['speedup']:.2f}×" + ('' if r['table1_pass'] else ' ✗'))
                                    for r in measured_runs()])
        self.rules_title.setText(tr('sparsity.rules_title'))
        self.rule_table.setHorizontalHeaderLabels([tr('sparsity.component'), tr('sparsity.rule'),
                                                   tr('sparsity.rule_value'), tr('sparsity.evidence')])
        _fill(self.rule_table, rule_rows(), self.main.mono_font, (1, 2))
        if self.preview_result:
            self._fill_preview()

    def _rules_text(self) -> str:
        try:
            from sparse_dft.sparsity import load_rules
            rules = load_rules()
        except Exception as error:  # noqa: BLE001
            return f'{tr("sparsity.rules")}: {error}'
        return f'{tr("sparsity.rules")}: {rules.get("schema_version", "?")} · {rules.get("status", "?")}'
