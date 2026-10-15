"""Settings page: language, theme, paths, execution and plot defaults (stored in QSettings)."""
from __future__ import annotations

from PyQt6.QtWidgets import (QComboBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit,
                             QPushButton, QSpinBox)

from sparse_dft.resources import user_data_dir

from .. import plots
from ..i18n import LANGUAGES, tr
from ..widgets import PathEdit, no_wheel
from .base import Page


class SettingsPage(Page):
    key = 'settings'

    def __init__(self, window, parent=None):
        super().__init__(window, parent)
        prefs = window.prefs
        # appearance
        card = self.card()
        self.appearance_title = QLabel(card)
        self.appearance_title.setProperty('role', 'section')
        card.body.addWidget(self.appearance_title)
        form = self._form(card)
        self.language = no_wheel(QComboBox(card))
        self.language.addItem('', 'auto')
        for lang in LANGUAGES:
            self.language.addItem(tr(f'lang.{lang}'), lang)
        self.theme = no_wheel(QComboBox(card))
        self.theme.addItem('', 'light')
        self.theme.addItem('', 'dark')
        self.font_px = no_wheel(QSpinBox(card))
        self.font_px.setRange(11, 20)
        self.language_label, self.theme_label, self.font_label = QLabel(card), QLabel(card), QLabel(card)
        form.addRow(self.language_label, self.language)
        form.addRow(self.theme_label, self.theme)
        form.addRow(self.font_label, self.font_px)
        # paths
        card = self.card()
        self.paths_title = QLabel(card)
        self.paths_title.setProperty('role', 'section')
        card.body.addWidget(self.paths_title)
        form = self._form(card)
        self.scf_exe = PathEdit('open', 'All files (*)', card)
        self.paw_root = PathEdit('dir', 'All files (*)', card)
        self.output_root = PathEdit('dir', 'All files (*)', card)
        self.scf_label, self.paw_label, self.out_label = QLabel(card), QLabel(card), QLabel(card)
        form.addRow(self.scf_label, self.scf_exe)
        form.addRow(self.paw_label, self.paw_root)
        form.addRow(self.out_label, self.output_root)
        self.data_info = QLabel(card)
        self.data_info.setProperty('role', 'muted')
        self.data_info.setWordWrap(True)
        card.body.addWidget(self.data_info)
        # execution
        card = self.card()
        self.exec_title = QLabel(card)
        self.exec_title.setProperty('role', 'section')
        card.body.addWidget(self.exec_title)
        form = self._form(card)
        self.omp = no_wheel(QSpinBox(card))
        self.omp.setRange(0, 1024)
        self.mpi_launcher = QLineEdit(card)
        self.mpi_launcher.setPlaceholderText('mpiexec -n {ranks}')
        self.mpi_ranks = no_wheel(QSpinBox(card))
        self.mpi_ranks.setRange(0, 4096)
        self.env = QPlainTextEdit(card)
        self.env.setFixedHeight(84)
        self.env.setFont(window.mono_font)
        self.omp_label, self.launcher_label, self.ranks_label, self.env_label = (QLabel(card) for _ in range(4))
        form.addRow(self.omp_label, self.omp)
        form.addRow(self.launcher_label, self.mpi_launcher)
        form.addRow(self.ranks_label, self.mpi_ranks)
        form.addRow(self.env_label, self.env)
        self.env_hint = QLabel(card)
        self.env_hint.setProperty('role', 'muted')
        card.body.addWidget(self.env_hint)
        # plots
        card = self.card()
        self.plots_title = QLabel(card)
        self.plots_title.setProperty('role', 'section')
        card.body.addWidget(self.plots_title)
        form = self._form(card)
        self.plot_width = no_wheel(QComboBox(card))
        for w in plots.WIDTHS:
            self.plot_width.addItem(w, w)
        self.plot_lang = no_wheel(QComboBox(card))
        for lang in plots.LANGS:
            self.plot_lang.addItem(tr(f'lang.{lang}'), lang)
        self.plot_palette = no_wheel(QComboBox(card))
        for name in plots.palettes():
            self.plot_palette.addItem(name, name)
        self.plot_formats = QLineEdit(card)
        self.plot_dpi = no_wheel(QSpinBox(card))
        self.plot_dpi.setRange(72, 2400)
        self.pw_label, self.pl_label, self.pp_label, self.pf_label, self.pd_label = (QLabel(card) for _ in range(5))
        form.addRow(self.pw_label, self.plot_width)
        form.addRow(self.pl_label, self.plot_lang)
        form.addRow(self.pp_label, self.plot_palette)
        form.addRow(self.pf_label, self.plot_formats)
        form.addRow(self.pd_label, self.plot_dpi)
        row = QHBoxLayout()
        row.addStretch(1)
        self.apply_button = QPushButton(self.body_widget)
        self.apply_button.setProperty('kind', 'primary')
        self.apply_button.clicked.connect(self.apply)
        row.addWidget(self.apply_button)
        self.body.addLayout(row)
        self.body.addStretch(1)
        self.load(prefs)
        self.language.currentIndexChanged.connect(lambda: self.main.set_language(self.language.currentData(), persist=True))
        self.theme.currentIndexChanged.connect(lambda: self.main.set_theme(self.theme.currentData(), persist=True))
        self.font_px.valueChanged.connect(lambda v: self.main.set_font_px(v, persist=True))

    @staticmethod
    def _form(card):
        form = QFormLayout()
        form.setHorizontalSpacing(16)
        form.setVerticalSpacing(10)
        form.setLabelAlignment(form.labelAlignment())
        card.body.addLayout(form)
        return form

    def _select(self, combo, value):
        index = combo.findData(value)
        combo.blockSignals(True)
        combo.setCurrentIndex(index if index >= 0 else 0)
        combo.blockSignals(False)

    def load(self, prefs):
        self._select(self.language, prefs.get('language'))
        self._select(self.theme, prefs.get('theme'))
        self.font_px.blockSignals(True); self.font_px.setValue(prefs.get('font_px')); self.font_px.blockSignals(False)
        self.scf_exe.setText(prefs.get('scf_exe'))
        self.paw_root.setText(prefs.get('paw_root'))
        self.output_root.setText(prefs.get('output_root'))
        self.omp.setValue(prefs.get('omp_threads'))
        self.mpi_launcher.setText(prefs.get('mpi_launcher'))
        self.mpi_ranks.setValue(prefs.get('mpi_ranks'))
        self.env.setPlainText(prefs.get('extra_env'))
        self._select(self.plot_width, prefs.get('plot_width'))
        self._select(self.plot_lang, prefs.get('plot_lang'))
        self._select(self.plot_palette, prefs.get('plot_palette'))
        self.plot_formats.setText(prefs.get('plot_formats'))
        self.plot_dpi.setValue(prefs.get('plot_dpi'))

    def apply(self):
        prefs = self.main.prefs
        formats = ','.join(f.strip().lower() for f in self.plot_formats.text().split(',')
                           if f.strip().lower() in plots.FORMATS) or 'pdf,png'
        for key, value in (('scf_exe', self.scf_exe.text().strip()), ('paw_root', self.paw_root.text().strip()),
                           ('output_root', self.output_root.text().strip()), ('omp_threads', self.omp.value()),
                           ('mpi_launcher', self.mpi_launcher.text().strip()), ('mpi_ranks', self.mpi_ranks.value()),
                           ('extra_env', self.env.toPlainText()), ('plot_width', self.plot_width.currentData()),
                           ('plot_lang', self.plot_lang.currentData()), ('plot_palette', self.plot_palette.currentData()),
                           ('plot_formats', formats), ('plot_dpi', self.plot_dpi.value())):
            prefs.set(key, value)
        self.plot_formats.setText(formats)
        self.main.settings_applied()
        self.main.notify(tr('settings.saved'), 'ok')

    def retranslate(self):
        super().retranslate()
        self.appearance_title.setText(tr('settings.appearance'))
        self.language_label.setText(tr('settings.language'))
        self.language.setItemText(0, tr('settings.language_auto'))
        self.theme_label.setText(tr('settings.theme'))
        self.theme.setItemText(0, tr('settings.theme_light'))
        self.theme.setItemText(1, tr('settings.theme_dark'))
        self.font_label.setText(tr('settings.font_size'))
        self.paths_title.setText(tr('settings.paths'))
        self.scf_label.setText(tr('settings.scf_exe'))
        self.scf_exe.setPlaceholderText(tr('settings.scf_exe_hint'))
        self.paw_label.setText(tr('settings.paw_root'))
        self.out_label.setText(tr('settings.output_root'))
        self.data_info.setText(tr('settings.paths_info', path=str(user_data_dir())))
        for edit in (self.scf_exe, self.paw_root, self.output_root):
            edit.retranslate()
        self.exec_title.setText(tr('settings.execution'))
        self.omp_label.setText(tr('settings.omp'))
        self.omp.setSpecialValueText(tr('settings.omp_auto'))
        self.launcher_label.setText(tr('settings.mpi_launcher'))
        self.ranks_label.setText(tr('settings.mpi_ranks'))
        self.mpi_ranks.setSpecialValueText(tr('settings.omp_auto'))
        self.mpi_ranks.setToolTip(tr('settings.mpi_hint'))
        self.env_label.setText(tr('settings.env'))
        self.env_hint.setText(tr('settings.env_hint'))
        self.plots_title.setText(tr('settings.plots'))
        self.pw_label.setText(tr('plot.width'))
        self.plot_width.setItemText(0, tr('plot.width_single'))
        self.plot_width.setItemText(1, tr('plot.width_double'))
        self.pl_label.setText(tr('plot.lang'))
        self.pp_label.setText(tr('plot.palette'))
        self.pf_label.setText(tr('plot.formats'))
        self.pd_label.setText(tr('plot.dpi'))
        self.apply_button.setText(tr('btn.apply'))

    def set_theme(self, p):
        for edit in (self.scf_exe, self.paw_root, self.output_root):
            edit.set_icon_colour(p['muted'])
