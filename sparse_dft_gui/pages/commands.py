"""Pages that are mostly generated command forms."""
from __future__ import annotations

import shutil
from pathlib import Path

from PyQt6.QtWidgets import QFileDialog, QHBoxLayout, QLabel, QListWidget, QMenu, QMessageBox, QPushButton

from sparse_dft.resources import paw_search_paths, user_data_dir

from ..i18n import tr
from ..widgets import run_async
from .base import CommandPage

PAW_FILES = ('manifest.json', 'index.json', 'potentials.jsonl.xz')


def is_paw_root(path) -> bool:
    path = Path(path)
    return all((path / name).is_file() for name in PAW_FILES)


def import_paw_root(source, library=None) -> Path:
    """Copy a PAW archive folder into the user library; refuses to overwrite."""
    source = Path(source)
    if not is_paw_root(source):
        raise ValueError(tr('paw.not_archive'))
    library = Path(library) if library else user_data_dir() / 'paw'
    target = library / source.name
    if target.exists():
        raise FileExistsError(tr('paw.exists', name=source.name))
    staging = library / f'.{source.name}.importing'
    if staging.exists():
        shutil.rmtree(staging)
    library.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, staging)
    staging.rename(target)
    return target


def convert_potcar_set(source, output) -> dict:
    """``sparse_dft.paw_convert.convert`` into a new or empty folder (never overwrites)."""
    from sparse_dft.paw_convert import convert
    return convert(source, output, overwrite=False)


class PawPage(CommandPage):
    key = 'paw'
    commands = ('paw-inventory', 'paw-resolve', 'paw-convert')

    def __init__(self, window, parent=None):
        super().__init__(window, parent)
        card = self.card()
        self.body.removeWidget(card)
        self.extra_top.addWidget(card)
        self.lib_title = QLabel(card)
        self.lib_title.setProperty('role', 'section')
        self.lib_hint = QLabel(card)
        self.lib_hint.setProperty('role', 'muted')
        self.lib_hint.setWordWrap(True)
        self.list = QListWidget(card)
        self.list.setMaximumHeight(118)
        row = QHBoxLayout()
        self.import_button = QPushButton(card)
        self.import_button.setProperty('kind', 'accent')
        self.import_button.clicked.connect(self.import_dialog)
        self.convert_button = QPushButton(card)
        self.convert_button.setProperty('kind', 'ghost')
        self.convert_menu = QMenu(self.convert_button)
        self.convert_folder_action = self.convert_menu.addAction('')
        self.convert_folder_action.triggered.connect(lambda: self.convert_dialog(archive=False))
        self.convert_archive_action = self.convert_menu.addAction('')
        self.convert_archive_action.triggered.connect(lambda: self.convert_dialog(archive=True))
        self.convert_button.setMenu(self.convert_menu)
        self.default_button = QPushButton(card)
        self.default_button.setProperty('kind', 'ghost')
        self.default_button.clicked.connect(self.use_selected)
        self.refresh_button = QPushButton(card)
        self.refresh_button.setProperty('kind', 'ghost')
        self.refresh_button.clicked.connect(self.refresh)
        row.addWidget(self.import_button)
        row.addWidget(self.convert_button)
        row.addWidget(self.default_button)
        row.addStretch(1)
        row.addWidget(self.refresh_button)
        card.body.addWidget(self.lib_title)
        card.body.addWidget(self.lib_hint)
        card.body.addWidget(self.list)
        card.body.addLayout(row)
        self.convert_hint = QLabel(card)
        self.convert_hint.setProperty('role', 'muted')
        self.convert_hint.setWordWrap(True)
        card.body.addWidget(self.convert_hint)
        self.refresh()

    def refresh(self):
        self.list.clear()
        for root in paw_search_paths():
            candidates = [root] if is_paw_root(root) else (
                sorted(p for p in root.iterdir() if p.is_dir() and is_paw_root(p)) if root.is_dir() else [])
            for path in candidates:
                self.list.addItem(str(path))

    def import_dialog(self):
        path = QFileDialog.getExistingDirectory(self, tr('btn.import'))
        if not path:
            return
        self.import_button.setEnabled(False)
        run_async(import_paw_root, path, on_done=self._imported, on_error=self._failed)

    def _imported(self, target):
        self.import_button.setEnabled(True)
        self.refresh()
        self.main.notify(tr('paw.imported', name=Path(target).name), 'ok')
        self.main.log_line('gui', 'stdout', f'[paw] imported {target}')

    def _failed(self, text):
        self.import_button.setEnabled(True)
        self.main.notify(tr('msg.error', message=text), 'error')

    def convert_dialog(self, *, archive=False, source=None, output=None):
        """Pick a POTCAR set (folder or tar archive) and a new/empty output folder, then convert."""
        if source is None:
            if archive:
                source, _ = QFileDialog.getOpenFileName(self, tr('btn.convert_potcar'), '',
                                                        'POTCAR archives (*.tar *.tar.gz *.tgz *.tar.xz)')
            else:
                source = QFileDialog.getExistingDirectory(self, tr('btn.convert_potcar'))
        if not source:
            return
        if output is None:
            output = QFileDialog.getExistingDirectory(self, tr('paw.convert_output'))
        if not output:
            return
        self.convert_button.setEnabled(False)
        self.main.notify(tr('paw.converting', name=Path(source).name), 'info')
        run_async(convert_potcar_set, source, output, on_done=self._converted, on_error=self._convert_failed)

    def _converted(self, result):
        self.convert_button.setEnabled(True)
        out = Path(result['output'])
        self.main.notify(tr('paw.converted', records=result.get('records', 0), elements=result.get('elements', 0),
                            name=out.name), 'ok')
        self.main.log_line('gui', 'stdout', f'[paw] converted into {out}')
        if not is_paw_root(out):
            return
        answer = QMessageBox.question(self, tr('btn.import'), tr('paw.import_converted', name=out.name))
        if answer == QMessageBox.StandardButton.Yes:
            self.import_button.setEnabled(False)
            run_async(import_paw_root, out, on_done=self._imported, on_error=self._failed)

    def _convert_failed(self, text):
        self.convert_button.setEnabled(True)
        self.main.notify(tr('msg.error', message=text), 'error')

    def use_selected(self):
        item = self.list.currentItem()
        if item is None:
            return
        self.main.prefs.set('paw_root', item.text())
        self.main.apply_form_defaults()
        self.main.notify(tr('settings.saved'), 'ok')

    def retranslate(self):
        super().retranslate()
        self.lib_title.setText(tr('paw.library'))
        self.lib_hint.setText(tr('paw.library_hint', path=str(user_data_dir() / 'paw')))
        self.import_button.setText(tr('btn.import'))
        self.convert_button.setText(tr('btn.convert_potcar'))
        self.convert_folder_action.setText(tr('paw.convert_folder'))
        self.convert_archive_action.setText(tr('paw.convert_archive'))
        self.convert_hint.setText(tr('paw.convert_hint'))
        self.default_button.setText(tr('paw.use_default'))
        self.refresh_button.setText(tr('btn.refresh'))


class KpointsPage(CommandPage):
    key = 'kpoints'
    commands = ('recommend-kpoints', 'make-band-path')


class ScfPage(CommandPage):
    key = 'scf'
    commands = ('scf', 'prepare')


class BandsPage(CommandPage):
    key = 'bands'
    commands = ('bands', 'dos', 'bands-compare', 'bands-analysis')


class RelaxPage(CommandPage):
    key = 'relax'
    commands = ('relax',)


class MaterialsProjectPage(CommandPage):
    key = 'mp'
    commands = ('mp-fetch', 'mp-fetch-bands')

    def __init__(self, window, parent=None):
        super().__init__(window, parent)
        self.note = QLabel(self.body_widget)
        self.note.setProperty('role', 'muted')
        self.note.setWordWrap(True)
        self.extra_top.addWidget(self.note)

    def retranslate(self):
        super().retranslate()
        self.note.setText(tr('mp.key_note'))
