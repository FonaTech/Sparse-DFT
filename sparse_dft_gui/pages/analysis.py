"""Analysis & plots: list figure kinds for a result, preview on an embedded canvas, export."""
from __future__ import annotations

import io
from pathlib import Path

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout,
                             QGridLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
                             QPushButton, QSpinBox, QVBoxLayout, QWidget)

from .. import plots
from ..i18n import tr
from ..preview import FigurePreview
from ..widgets import PathEdit, no_wheel, run_async
from .base import Page


def _render_png(fig, dpi=60) -> bytes:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    with plots.PLOT_LOCK:
        FigureCanvasAgg(fig)
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=dpi)
    return buffer.getvalue()


def _thumbnails(source, kinds, options):
    out = []
    for kind in kinds:
        try:
            fig = plots.make_figure(kind, source, **options)
            out.append((kind, _render_png(fig), None))
        except Exception as error:  # noqa: BLE001
            out.append((kind, None, f'{type(error).__name__}: {error}'))
    return out


def _figures(kind, source, options):
    """{'combined': fig, <panel>: fig ...}: multi-panel kinds come split and combined in one load."""
    if plots.is_multi_panel(kind):
        return plots.make_figures(kind, source, panels='both', **options)
    return {'combined': plots.make_figure(kind, source, **options)}


class ExportDialog(QDialog):
    PANEL_CHOICES = ('combined', 'split', 'both', 'current')

    def __init__(self, parent, stem, defaults, *, multi_panel=False):
        super().__init__(parent)
        self.setWindowTitle(tr('plot.export_title'))
        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(10)
        self.stem = PathEdit('save', 'All files (*)', self)
        self.stem.setText(stem)
        form.addRow(tr('plot.stem'), self.stem)
        formats = QWidget(self)
        row = QHBoxLayout(formats)
        row.setContentsMargins(0, 0, 0, 0)
        wanted = set(defaults.get('formats', ('pdf', 'png')))
        self.checks = {}
        for fmt in plots.FORMATS:
            box = QCheckBox(fmt.upper(), formats)
            box.setChecked(fmt in wanted)
            row.addWidget(box)
            self.checks[fmt] = box
        row.addStretch(1)
        form.addRow(tr('plot.formats'), formats)
        self.dpi = no_wheel(QSpinBox(self))
        self.dpi.setRange(72, 2400)
        self.dpi.setSingleStep(50)
        self.dpi.setValue(int(defaults.get('dpi', 600)))
        form.addRow(tr('plot.dpi'), self.dpi)
        self.panels = no_wheel(QComboBox(self))
        for choice in self.PANEL_CHOICES:
            self.panels.addItem(tr(f'plot.export_{choice}'), choice)
        self.panels.setEnabled(multi_panel)
        if multi_panel:
            form.addRow(tr('plot.export_panels'), self.panels)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.resize(520, 200)

    def result_values(self):
        stem = self.stem.text().strip()
        for fmt in plots.FORMATS:
            if stem.lower().endswith('.' + fmt):
                stem = stem[: -len(fmt) - 1]
        return stem, [f for f, box in self.checks.items() if box.isChecked()], self.dpi.value()

    def panel_choice(self) -> str:
        return self.panels.currentData() if self.panels.isEnabled() else 'combined'


class AnalysisPage(Page):
    key = 'analysis'

    def __init__(self, window, parent=None):
        super().__init__(window, parent)
        self.kinds: list[str] = []
        self.figures: dict = {}
        self.view = 'combined'
        self.current_kind = None
        self._generation = 0
        top = self.card()
        row = QHBoxLayout()
        self.source_label = QLabel(top)
        self.source = PathEdit('dir', 'All files (*)', top)
        self.source.edit.returnPressed.connect(self.refresh)
        self.recent = no_wheel(QComboBox(top))
        self.recent.setMinimumWidth(180)
        self.recent.activated.connect(self._recent_chosen)
        self.refresh_button = QPushButton(top)
        self.refresh_button.setProperty('kind', 'accent')
        self.refresh_button.clicked.connect(self.refresh)
        row.addWidget(self.source_label)
        row.addWidget(self.source, 1)
        row.addWidget(self.recent)
        row.addWidget(self.refresh_button)
        top.body.addLayout(row)
        ref_row = QHBoxLayout()
        self.reference_label = QLabel(top)
        self.reference = PathEdit('dir', 'All files (*)', top)
        self.reference.edit.editingFinished.connect(self._options_changed)
        ref_row.addWidget(self.reference_label)
        ref_row.addWidget(self.reference, 1)
        top.body.addLayout(ref_row)
        controls = QHBoxLayout()
        controls.setSpacing(8)
        self.width_label, self.lang_label, self.palette_label = QLabel(top), QLabel(top), QLabel(top)
        for label in (self.width_label, self.lang_label, self.palette_label):
            label.setProperty('role', 'muted')
        self.width_combo = no_wheel(QComboBox(top))
        for w in plots.WIDTHS:
            self.width_combo.addItem(w, w)
        self.lang_combo = no_wheel(QComboBox(top))
        for lang in plots.LANGS:
            self.lang_combo.addItem(tr(f'lang.{lang}'), lang)
        self.palette_combo = no_wheel(QComboBox(top))
        for name in plots.palettes():
            self.palette_combo.addItem(name, name)
        for combo in (self.width_combo, self.lang_combo, self.palette_combo):
            combo.currentIndexChanged.connect(self._options_changed)
        self.export_button = QPushButton(top)
        self.export_button.setProperty('kind', 'primary')
        self.export_button.clicked.connect(self.export_dialog)
        self.export_button.setEnabled(False)
        for widget in (self.width_label, self.width_combo, self.lang_label, self.lang_combo, self.palette_label, self.palette_combo):
            controls.addWidget(widget)
        controls.addStretch(1)
        controls.addWidget(self.export_button)
        top.body.addLayout(controls)
        self.status = QLabel(top)
        self.status.setProperty('role', 'muted')
        self.status.setWordWrap(True)
        top.body.addWidget(self.status)

        content = QGridLayout()
        content.setHorizontalSpacing(12)
        list_card = self._card()
        self.list_title = QLabel(list_card)
        self.list_title.setProperty('role', 'section')
        self.thumbs = QListWidget(list_card)
        self.thumbs.setViewMode(QListWidget.ViewMode.IconMode)
        self.thumbs.setFlow(QListWidget.Flow.TopToBottom)
        self.thumbs.setWrapping(False)
        self.thumbs.setIconSize(QSize(176, 124))
        self.thumbs.setSpacing(6)
        self.thumbs.setMovement(QListWidget.Movement.Static)
        self.thumbs.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.thumbs.setWordWrap(True)
        self.thumbs.setMinimumWidth(214)
        self.thumbs.setMaximumWidth(240)
        self.thumbs.currentItemChanged.connect(self._kind_selected)
        list_card.body.addWidget(self.list_title)
        list_card.body.addWidget(self.thumbs, 1)
        preview_card = self._card()
        self.preview_title = QLabel(preview_card)
        self.preview_title.setProperty('role', 'section')
        self.kind_description = QLabel(preview_card)
        self.kind_description.setProperty('role', 'muted')
        self.kind_description.setWordWrap(True)
        nav = QHBoxLayout()
        nav.setSpacing(6)
        self.view_label = QLabel(preview_card)
        self.view_label.setProperty('role', 'muted')
        self.view_combo = no_wheel(QComboBox(preview_card))
        for mode in ('combined', 'split'):
            self.view_combo.addItem(mode, mode)
        self.view_combo.currentIndexChanged.connect(self._view_changed)
        self.prev_button = QPushButton('‹', preview_card)
        self.next_button = QPushButton('›', preview_card)
        for b, step in ((self.prev_button, -1), (self.next_button, 1)):
            b.setProperty('kind', 'ghost')
            b.setFixedWidth(34)
            b.clicked.connect(lambda _=False, s=step: self.step_panel(s))
        self.panel_label = QLabel(preview_card)
        self.panel_label.setProperty('role', 'muted')
        for w in (self.view_label, self.view_combo, self.prev_button, self.panel_label, self.next_button):
            nav.addWidget(w)
        nav.addStretch(1)
        self.canvas_holder = QVBoxLayout()
        self.placeholder = QLabel(preview_card)
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setProperty('role', 'muted')
        self.placeholder.setMinimumHeight(360)
        self.placeholder.setWordWrap(True)
        self.canvas = FigurePreview(preview_card)
        self.canvas.setMinimumHeight(360)
        self.canvas.hide()
        self.canvas_holder.addWidget(self.placeholder)
        self.canvas_holder.addWidget(self.canvas, 1)
        preview_card.body.addWidget(self.preview_title)
        preview_card.body.addWidget(self.kind_description)
        preview_card.body.addLayout(nav)
        preview_card.body.addLayout(self.canvas_holder, 1)
        self._update_nav()
        content.addWidget(list_card, 0, 0)
        content.addWidget(preview_card, 0, 1)
        content.setColumnStretch(1, 1)
        self.body.addLayout(content, 1)

    def _card(self):
        from ..widgets import Card
        return Card(self.body_widget)

    # ---- options
    def options(self) -> dict:
        out = {'width': self.width_combo.currentData() or 'single', 'lang': self.lang_combo.currentData() or 'en',
               'palette': self.palette_combo.currentData() or 'academic'}
        reference = self.reference.text().strip()
        if reference:
            out['reference'] = reference          # comparison panels / kinds (same as `sparse-dft plot --reference`)
        return out

    def apply_plot_defaults(self, prefs):
        for combo, key in ((self.width_combo, 'plot_width'), (self.lang_combo, 'plot_lang'), (self.palette_combo, 'plot_palette')):
            index = combo.findData(prefs.get(key))
            if index >= 0:
                combo.blockSignals(True); combo.setCurrentIndex(index); combo.blockSignals(False)

    def add_recent(self, path):
        path = str(path)
        if self.recent.findData(path) < 0:
            self.recent.insertItem(0, Path(path).name or path, path)
            self.recent.setItemData(0, path, Qt.ItemDataRole.ToolTipRole)
        self.recent.setCurrentIndex(self.recent.findData(path))

    def _recent_chosen(self, index):
        path = self.recent.itemData(index)
        if path:
            self.source.setText(path)
            self.refresh()

    # ---- listing
    def show_result(self, path):
        self.source.setText(str(path))
        self.add_recent(path)
        return self.refresh()

    def refresh(self):
        path = self.source.text().strip()
        self.thumbs.clear()
        self.kinds = []
        self._clear_canvas(tr('plot.choose'))
        if not path:
            self.status.setText(tr('plot.choose'))
            return False
        if not plots.available():
            self.status.setText(tr('plot.unavailable'))
            self._clear_canvas(tr('plot.unavailable'))
            return False
        try:
            kinds = plots.kinds_for(path)
        except Exception as error:  # noqa: BLE001
            self.status.setText(tr('msg.error', message=str(error)))
            self.main.log_line('gui', 'stderr', f'[plots] available_kinds({path}): {error}')
            return False
        self.kinds = kinds
        if not kinds:
            self.status.setText(tr('plot.no_kinds'))
            self._clear_canvas(tr('plot.no_kinds'))
            return False
        self.status.setText(f'{len(kinds)} · {path}')
        for kind in kinds:
            info = plots.kind_info(kind)
            item = QListWidgetItem(info.get('title') or kind)
            item.setData(Qt.ItemDataRole.UserRole, kind)
            item.setToolTip(info.get('description') or kind)
            item.setSizeHint(QSize(196, 160))
            self.thumbs.addItem(item)
        self._generation += 1
        generation = self._generation
        run_async(_thumbnails, path, kinds, {**self.options()},
                  on_done=lambda result: self._set_thumbnails(result, generation),
                  on_error=lambda text: self.main.log_line('gui', 'stderr', f'[plots] {text}'))
        self.thumbs.setCurrentRow(0)
        return True

    def _set_thumbnails(self, result, generation):
        if generation != self._generation:
            return
        failures = []
        for kind, png, error in result:
            for row in range(self.thumbs.count()):
                item = self.thumbs.item(row)
                if item.data(Qt.ItemDataRole.UserRole) != kind:
                    continue
                if png:
                    pixmap = QPixmap()
                    pixmap.loadFromData(png, 'PNG')
                    item.setIcon(QIcon(pixmap))
                else:
                    item.setToolTip(error or kind)
                    failures.append(f'{kind}: {error}')
        for text in failures:
            self.main.log_line('gui', 'stderr', f'[plots] {text}')
        self.main.page_event('thumbnails', len(result) - len(failures))

    # ---- preview
    def _kind_selected(self, current, _previous=None):
        if current is None:
            return
        kind = current.data(Qt.ItemDataRole.UserRole)
        self.current_kind = kind
        info = plots.kind_info(kind)
        self.preview_title.setText(info.get('title') or kind)
        self.kind_description.setText(info.get('description') or '')
        self.render_current()

    def _options_changed(self):
        if self.kinds:
            self.render_current()

    def render_current(self):
        kind, path = self.current_kind, self.source.text().strip()
        if not kind or not path:
            return
        self._generation += 1
        generation = self._generation
        self._clear_canvas(tr('plot.rendering'))
        self.export_button.setEnabled(False)
        run_async(_figures, kind, path, self.options(),
                  on_done=lambda figs: self._set_figures(figs, generation),
                  on_error=lambda text: self._render_failed(text, generation))

    def _render_failed(self, text, generation):
        if generation != self._generation:
            return
        self._clear_canvas(tr('msg.error', message=text))
        self.main.log_line('gui', 'stderr', f'[plots] {self.current_kind}: {text}')

    @property
    def figure(self):
        """The figure on screen (combined, or the selected panel in split view)."""
        return self.canvas.figure if self.figures else None

    def panel_names(self) -> list[str]:
        return [n for n in self.figures if n != 'combined']

    def _clear_canvas(self, text=''):
        self.figures = {}
        self.canvas.clear()
        self.canvas.hide()
        self.placeholder.setText(text)
        self.placeholder.show()
        self._update_nav()

    def _set_figures(self, figs, generation):
        if generation != self._generation:
            return
        self.figures = dict(figs)
        self.placeholder.hide()
        self.canvas.show()
        self._show_view()
        self.export_button.setEnabled(True)
        self.main.page_event('preview', self.current_kind)

    def _show_view(self):
        if not self.figures:
            return
        if self.view == 'split' and self.panel_names():
            self.canvas.set_figures({n: self.figures[n] for n in self.panel_names()})
        else:
            self.canvas.set_figures({'combined': self.figures['combined']})
        self._update_nav()

    def set_view(self, view: str):
        """'combined' or 'split' (one panel at a time, previous/next)."""
        index = self.view_combo.findData(view)
        if index >= 0 and index != self.view_combo.currentIndex():
            self.view_combo.setCurrentIndex(index)
        else:
            self._view_changed()

    def _view_changed(self, _index=None):
        self.view = self.view_combo.currentData() or 'combined'
        self._show_view()

    def step_panel(self, delta: int):
        self.canvas.step(delta)
        self._update_nav()

    def _update_nav(self):
        multi = bool(self.panel_names())
        for w in (self.view_label, self.view_combo):
            w.setVisible(multi)
        split = multi and self.view == 'split'
        for w in (self.prev_button, self.next_button, self.panel_label):
            w.setVisible(split)
        if split and self.canvas.figures:
            name = self.canvas.names[self.canvas.index]
            self.panel_label.setText(f'{self.canvas.index + 1} / {len(self.canvas.figures)} · '
                                     f'{tr("plot.panel_" + name, name)}')

    # ---- export
    def export_dialog(self):
        if self.figure is None:
            return
        base = plots.preview_source(self.source.text().strip())
        stem = str(base / 'figures' / f'{self.current_kind}')
        prefs = self.main.prefs
        dialog = ExportDialog(self, stem, {'formats': prefs.get('plot_formats').split(','), 'dpi': prefs.get('plot_dpi')},
                              multi_panel=bool(self.panel_names()))
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        stem, formats, dpi = dialog.result_values()
        if not formats or not stem:
            return
        self.export(stem, formats, dpi, panels=dialog.panel_choice())

    def export(self, stem, formats, dpi=600, *, panels='current'):
        """Write the figure(s): 'current' (on screen), 'combined', 'split' (STEM_<panel>) or 'both'."""
        provenance = {'source': self.source.text().strip(), 'kind': self.current_kind, **self.options(),
                      'generator': 'sparse_dft_gui'}
        if not self.figures:
            return []
        if panels == 'current':
            name = self.canvas.names[self.canvas.index] if self.canvas.names else 'combined'
            chosen = {name: self.figure}      # a panel on screen goes to STEM_<panel>, like a split export
        elif panels == 'combined':
            chosen = {'combined': self.figures['combined']}
        elif panels == 'split':
            chosen = {n: self.figures[n] for n in self.panel_names()} or {'combined': self.figures['combined']}
        else:
            chosen = dict(self.figures)
        try:
            paths = plots.export_figures(chosen, stem, formats, dpi=dpi, provenance=provenance)
        except Exception as error:  # noqa: BLE001
            self.main.notify(tr('msg.error', message=str(error)), 'error')
            return []
        self.main.notify(tr('plot.exported', n=len(paths), path=str(Path(stem).parent)), 'ok')
        for path in paths:
            self.main.log_line('gui', 'stdout', f'[plots] wrote {path}')
        return paths

    def retranslate(self):
        super().retranslate()
        self.source_label.setText(tr('plot.source'))
        self.source.retranslate()
        self.reference_label.setText(tr('plot.reference'))
        self.reference.setPlaceholderText(tr('plot.reference_hint'))
        self.reference.retranslate()
        self.refresh_button.setText(tr('btn.refresh'))
        self.width_label.setText(tr('plot.width'))
        self.width_combo.setItemText(0, tr('plot.width_single'))
        self.width_combo.setItemText(1, tr('plot.width_double'))
        self.lang_label.setText(tr('plot.lang'))
        for i, lang in enumerate(plots.LANGS):
            self.lang_combo.setItemText(i, tr(f'lang.{lang}'))
        self.palette_label.setText(tr('plot.palette'))
        self.export_button.setText(tr('btn.export'))
        self.view_label.setText(tr('plot.view'))
        for i, mode in enumerate(('combined', 'split')):
            self.view_combo.setItemText(i, tr(f'plot.view_{mode}'))
        self._update_nav()
        self.list_title.setText(tr('plot.kind'))
        if not self.preview_title.text() or self.current_kind is None:
            self.preview_title.setText(tr('plot.preview'))
        if self.recent.count() == 0:
            self.recent.setPlaceholderText(tr('plot.recent'))
        if not self.kinds:
            text = tr('plot.unavailable') if not plots.available() else tr('plot.choose')
            self.status.setText(text)
            self.placeholder.setText(text)

    def set_theme(self, p):
        self.source.set_icon_colour(p['muted'])
