"""Structure page: load a file, inspect it, show lattice, atoms and a projected view."""
from __future__ import annotations

import math
from collections import Counter

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QPainter, QPen
from PyQt6.QtWidgets import (QComboBox, QGridLayout, QHBoxLayout, QHeaderView, QLabel, QPushButton,
                             QSizePolicy, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from ..argspec import STRUCTURE_FILTER
from ..i18n import tr
from ..recommend import ALWAYS, TASKS, ClassificationCard, PreviewDialog, analyze, band_path_target, form_values, \
    write_band_path
from ..widgets import PathEdit, no_wheel, run_async
from .base import Page

AXES = ('iso', 'a', 'b', 'c')


def _inspect(path):
    from sparse_dft.structure import read_structure
    s = read_structure(path)
    normalized = s.normalized()
    return {'structure': normalized, 'dict': normalized.as_dict(), 'natoms': normalized.natoms,
            'volume': normalized.volume, 'cartesian': normalized.cartesian_positions(),
            'warnings': list(normalized.warnings)}


def formula(species) -> str:
    counts = Counter(species)
    return ''.join(f'{el}{n if n > 1 else ""}' for el, n in counts.items())


class StructureView(QWidget):
    """Orthographic projection of the cell (edges) and atoms (filled circles)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.lattice = None
        self.atoms = []  # (species, cartesian)
        self.axis = 'iso'
        self.colours = {}
        self.theme_colours = {'surface': '#FFFFFF', 'border_strong': '#CFC8DA', 'text': '#2B2A33', 'muted': '#5E5A6E'}
        self.swatches = ['#BEE3B0', '#DCCBFA', '#FFD8BE', '#C4E2F7', '#F8CAD6', '#FFF0B3']
        self.outline = ['#3E8E5A', '#7559C0', '#C8673F', '#3D7EBF', '#C2507A', '#9A7B12']
        self.setMinimumSize(260, 240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_structure(self, lattice, atoms):
        self.lattice, self.atoms = lattice, atoms
        species = list(dict.fromkeys(s for s, _ in atoms))
        self.colours = {s: i % len(self.swatches) for i, s in enumerate(species)}
        self.update()

    def set_axis(self, axis):
        self.axis = axis
        self.update()

    def _project(self, v):
        x, y, z = v
        if self.axis == 'a':
            return (y, z, x)
        if self.axis == 'b':
            return (x, z, y)
        if self.axis == 'c':
            return (x, y, z)
        # isometric-like view: rotate about z by 35 deg, tilt about x by 25 deg
        a, t = math.radians(35), math.radians(25)
        x1, y1 = x * math.cos(a) - y * math.sin(a), x * math.sin(a) + y * math.cos(a)
        y2, z2 = y1 * math.cos(t) - z * math.sin(t), y1 * math.sin(t) + z * math.cos(t)
        return (x1, z2, y2)

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        p = self.theme_colours
        painter.fillRect(self.rect(), QColor(p['surface']))
        if not self.atoms:
            painter.setPen(QColor(p['muted']))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, tr('structure.none'))
            return
        corners = []
        if self.lattice:
            a, b, c = self.lattice
            for i in (0, 1):
                for j in (0, 1):
                    for k in (0, 1):
                        corners.append(tuple(i * a[n] + j * b[n] + k * c[n] for n in range(3)))
        points = [self._project(v) for v in corners] + [self._project(pos) for _, pos in self.atoms]
        xs, ys = [q[0] for q in points], [q[1] for q in points]
        span = max(max(xs) - min(xs), max(ys) - min(ys), 1e-6)
        margin = 28
        scale = (min(self.width(), self.height()) - 2 * margin) / span
        cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
        def to_screen(q):
            return QPointF(self.width() / 2 + (q[0] - cx) * scale, self.height() / 2 - (q[1] - cy) * scale)
        if corners:
            pen = QPen(QColor(p['border_strong']), 1.4)
            painter.setPen(pen)
            for i, u in enumerate(corners):
                for j, v in enumerate(corners):
                    if j > i and sum(1 for n in range(3) if abs(u[n] - v[n]) > 1e-9) >= 1:
                        # edges join corners that differ by exactly one lattice vector
                        bits_u, bits_v = format(i, '03b'), format(j, '03b')
                        if sum(x != y for x, y in zip(bits_u, bits_v)) == 1:
                            painter.drawLine(to_screen(self._project(u)), to_screen(self._project(v)))
        radius = max(7.0, min(22.0, scale * 0.45))
        order = sorted(range(len(self.atoms)), key=lambda i: self._project(self.atoms[i][1])[2])
        for i in order:
            species, pos = self.atoms[i]
            idx = self.colours.get(species, 0)
            centre = to_screen(self._project(pos))
            painter.setPen(QPen(QColor(self.outline[idx]), 1.6))
            painter.setBrush(QBrush(QColor(self.swatches[idx])))
            painter.drawEllipse(centre, radius, radius)
        # legend
        painter.setPen(QColor(p['text']))
        y = 18
        for species, idx in self.colours.items():
            painter.setPen(QPen(QColor(self.outline[idx]), 1.2))
            painter.setBrush(QBrush(QColor(self.swatches[idx])))
            painter.drawEllipse(QPointF(16, y), 6, 6)
            painter.setPen(QColor(p['text']))
            painter.drawText(QRectF(28, y - 9, 80, 18), Qt.AlignmentFlag.AlignVCenter, species)
            y += 20
        painter.end()


class StructurePage(Page):
    key = 'structure'

    def __init__(self, window, parent=None):
        super().__init__(window, parent)
        self.current = None
        top = self.card()
        row = QHBoxLayout()
        row.setSpacing(10)
        self.file_label = QLabel(top)
        self.path = PathEdit('open', STRUCTURE_FILTER, top)
        self.path.edit.returnPressed.connect(self.inspect)
        self.inspect_button = QPushButton(top)
        self.inspect_button.setProperty('kind', 'accent')
        self.inspect_button.clicked.connect(self.inspect)
        self.use_button = QPushButton(top)
        self.use_button.setProperty('kind', 'ghost')
        self.use_button.clicked.connect(self.use_in_forms)
        self.use_button.setEnabled(False)
        row.addWidget(self.file_label)
        row.addWidget(self.path, 1)
        row.addWidget(self.inspect_button)
        row.addWidget(self.use_button)
        top.body.addLayout(row)
        self.summary = QLabel(top)
        self.summary.setProperty('role', 'muted')
        self.summary.setWordWrap(True)
        top.body.addWidget(self.summary)

        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(12)
        lattice_card = self._card_widget()
        self.lattice_title = QLabel(lattice_card)
        self.lattice_title.setProperty('role', 'section')
        self.lattice_table = QTableWidget(3, 4, lattice_card)
        self.lattice_table.verticalHeader().setVisible(False)
        self.lattice_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.lattice_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.lattice_table.setFixedHeight(130)
        lattice_card.body.addWidget(self.lattice_title)
        lattice_card.body.addWidget(self.lattice_table)
        atoms_card = self._card_widget()
        self.atoms_title = QLabel(atoms_card)
        self.atoms_title.setProperty('role', 'section')
        self.atoms_table = QTableWidget(0, 5, atoms_card)
        self.atoms_table.verticalHeader().setVisible(False)
        self.atoms_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.atoms_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.atoms_table.setAlternatingRowColors(True)
        self.atoms_table.setMinimumHeight(180)
        atoms_card.body.addWidget(self.atoms_title)
        atoms_card.body.addWidget(self.atoms_table, 1)
        view_card = self._card_widget()
        head = QHBoxLayout()
        self.view_title = QLabel(view_card)
        self.view_title.setProperty('role', 'section')
        self.axis_label = QLabel(view_card)
        self.axis_label.setProperty('role', 'muted')
        self.axis = no_wheel(QComboBox(view_card))
        for axis in AXES:
            self.axis.addItem(axis, axis)
        self.axis.currentIndexChanged.connect(lambda: self.view.set_axis(self.axis.currentData()))
        head.addWidget(self.view_title)
        head.addStretch(1)
        head.addWidget(self.axis_label)
        head.addWidget(self.axis)
        self.view = StructureView(view_card)
        view_card.body.addLayout(head)
        view_card.body.addWidget(self.view, 1)
        grid.addWidget(lattice_card, 0, 0)
        grid.addWidget(atoms_card, 1, 0)
        grid.addWidget(view_card, 0, 1, 2, 1)
        grid.setColumnStretch(0, 3)
        grid.setColumnStretch(1, 2)
        self.body.addLayout(grid, 1)

        self.preset_card = ClassificationCard(self.body_widget)
        self.preset_card.set_accuracy(self.main.prefs.get('preset_accuracy'))
        self.preset_card.auto_apply.setChecked(bool(self.main.prefs.get('auto_apply_presets')))
        self.preset_card.apply_requested.connect(self.apply_recommended)
        self.preset_card.accuracy_changed.connect(self._accuracy_changed)
        self.preset_card.auto_apply_changed.connect(lambda on: self.main.prefs.set('auto_apply_presets', int(on)))
        self.body.addWidget(self.preset_card)
        self.recommendation = None
        self._classified_path = None

    def _card_widget(self):
        from ..widgets import Card
        return Card(self.body_widget)

    def load(self, path):
        self.path.setText(str(path))
        self.inspect()

    def inspect(self):
        path = self.path.text().strip()
        if not path:
            self.main.notify(tr('msg.error', message=tr('structure.none')), 'warning')
            return
        self.inspect_button.setEnabled(False)
        run_async(_inspect, path, on_done=self._show, on_error=self._failed)

    def _failed(self, text):
        self.inspect_button.setEnabled(True)
        self.main.notify(tr('msg.error', message=text), 'error')
        self.main.log_line('gui', 'stderr', f'[structure] {text}')

    def _show(self, info):
        self.inspect_button.setEnabled(True)
        self.use_button.setEnabled(True)
        self.current = info
        s = info['structure']
        volume = f'{info["volume"]:.4f}' if info['volume'] else '—'
        text = tr('structure.summary', natoms=info['natoms'], formula=formula(s.species), volume=volume)
        if info['warnings']:
            text += '\n' + tr('structure.warnings') + ': ' + '; '.join(map(str, info['warnings']))
        self.summary.setText(text)
        lattice = s.lattice or ((0, 0, 0),) * 3
        for i, vector in enumerate(lattice):
            norm = math.sqrt(sum(x * x for x in vector))
            for j, value in enumerate(list(vector) + [norm]):
                item = QTableWidgetItem(f'{value:.6f}')
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.lattice_table.setItem(i, j, item)
        self.atoms_table.setRowCount(len(s.species))
        for i, (species, frac, cart) in enumerate(zip(s.species, s.positions, info['cartesian'])):
            values = [str(i + 1), species] + [f'{x:.6f}' for x in frac]
            for j, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip('cart (Å): ' + ', '.join(f'{x:.5f}' for x in cart))
                if j >= 2:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.atoms_table.setItem(i, j, item)
        self.view.set_structure(s.lattice, list(zip(s.species, info['cartesian'])))
        self.main.log_line('gui', 'stdout', f'[structure] {self.path.text()} -> {info["natoms"]} atoms')
        self.classify()

    # ---- classification and recommended options (sparse_dft.presets)
    def paw_root(self) -> str:
        form = self.main.forms.get('scf')
        typed = form.fields['paw_root'].value().strip() if form is not None and 'paw_root' in form.fields else ''
        return typed or self.main.prefs.get('paw_root')

    def classify(self):
        path = self.path.text().strip()
        if not path:
            return
        self._classified_path = path
        self.recommendation = None
        self.preset_card.set_analyzing()
        accuracy = self.preset_card.accuracy.currentData()
        run_async(analyze, path, self.paw_root() or None, accuracy,
                  on_done=lambda result, p=path: self._classified(p, result),
                  on_error=lambda text, p=path: self._classify_failed(p, text))

    def _classified(self, path, result):
        if path != self._classified_path:  # a newer file was loaded meanwhile
            return
        self.recommendation = result
        self.preset_card.set_result(result)
        for error in result['errors']:
            self.main.log_line('gui', 'stderr', f'[preset] {error}')
        self.main.log_line('gui', 'stdout', f'[preset] {path} -> {result["profile"]["category"]}')
        if self.main.prefs.get('auto_apply_presets'):
            self.apply_recommended(confirm=False)

    def _classify_failed(self, path, text):
        if path != self._classified_path:
            return
        self.preset_card.set_failed(text)
        self.main.log_line('gui', 'stderr', f'[preset] {text}')

    def _accuracy_changed(self, level):
        self.main.prefs.set('preset_accuracy', level)
        if self.current is not None:
            self.classify()

    def _band_path_file(self):
        return band_path_target(self.main.prefs.output_root(), self.recommendation['path'])

    def recommended_rows(self, force=False) -> list[dict]:
        """Preview rows over the SCF, Bands and Relax forms (see CommandForm.recommended_changes)."""
        if self.recommendation is None:
            return []
        rows = []
        hubbard = self.preset_card.hubbard.isChecked()
        for task in TASKS:
            form = self.main.forms.get(task)
            if form is None:
                continue
            values = form_values(self.recommendation, task, hubbard=hubbard, band_path_file=self._band_path_file())
            rows += form.recommended_changes(values, force=force, always=ALWAYS)
        return rows

    def apply_recommended(self, *_, confirm=True, force=False) -> list[dict] | None:
        """Fill the SCF/Bands/Relax forms with the recommendation; returns the rows, None if cancelled."""
        if self.recommendation is None:
            return None
        if confirm:
            dialog = PreviewDialog(self.recommended_rows, self)
            if dialog.exec() != PreviewDialog.DialogCode.Accepted:
                return None
            force = dialog.force.isChecked()
        hubbard = self.preset_card.hubbard.isChecked()
        target = self._band_path_file()
        rows = []
        for task in TASKS:
            form = self.main.forms.get(task)
            if form is None:
                continue
            values = form_values(self.recommendation, task, hubbard=hubbard, band_path_file=target)
            if 'path' in values and not (form.fields['path'].user_edited and not force):
                write_band_path(self.recommendation, target)
            rows += form.apply_recommended(values, force, always=ALWAYS)
        changed = sum(1 for r in rows if r['action'] == 'set')
        kept = sum(1 for r in rows if r['action'] == 'kept')
        self.main.notify(tr('preset.applied', set=changed, kept=kept), 'ok')
        self.main.log_line('gui', 'stdout', '[preset] ' + ', '.join(
            f'{r["command"]}.{r["dest"]}={r["new"]}' for r in rows if r['action'] == 'set'))
        return rows

    def use_in_forms(self):
        self.main.broadcast_values({'structure': self.path.text().strip()},
                                     commands=('scf', 'bands', 'dos', 'relax', 'prepare', 'recommend-kpoints'))
        self.main.notify(tr('structure.use_in_forms') + ' ✓', 'ok')

    def retranslate(self):
        super().retranslate()
        self.file_label.setText(tr('structure.file'))
        self.path.retranslate()
        self.inspect_button.setText(tr('structure.inspect'))
        self.use_button.setText(tr('structure.use_in_forms'))
        self.lattice_title.setText(tr('structure.lattice'))
        self.atoms_title.setText(tr('structure.atoms'))
        self.view_title.setText(tr('structure.view'))
        self.axis_label.setText(tr('structure.axis'))
        self.lattice_table.setHorizontalHeaderLabels(['x', 'y', 'z', '|v|'])
        self.lattice_table.setVerticalHeaderLabels(['a', 'b', 'c'])
        self.atoms_table.setHorizontalHeaderLabels(['#', tr('structure.species'), 'a', 'b', 'c'])
        if self.current is None:
            self.summary.setText(tr('structure.none'))
        self.preset_card.retranslate()
        self.view.update()

    def set_theme(self, p):
        self.view.theme_colours = p
        self.path.set_icon_colour(p['muted'])
        self.view.update()
