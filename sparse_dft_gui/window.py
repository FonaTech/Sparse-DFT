"""Main window: navigation, pages, run monitor (live chart + log) and the job queue."""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (QComboBox, QFrame, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMainWindow,
                             QPushButton, QSplitter, QStackedWidget, QStatusBar, QVBoxLayout, QWidget)

from . import argspec, fonts, icons, plots, theme
from .chart import ConvergenceChart, IterationFollower
from .console import LogConsole
from .forms import CommandForm
from .i18n import LANGUAGES, detect_language, set_language, translator, tr
from .jobs import JobManager, format_duration
from .prefs import Preferences
from .widgets import Banner, Card, Chip, no_wheel

NAV = (
    ('setup', (('structure', 'structure'), ('paw', 'paw'), ('kpoints', 'kpoints'))),
    ('compute', (('scf', 'scf'), ('bands', 'bands'), ('relax', 'relax'), ('sparsity', 'sparsity'))),
    ('results', (('mp', 'mp'), ('analysis', 'analysis'), ('queue', 'queue'))),
    ('system', (('diagnostics', 'diagnostics'), ('settings', 'settings'))),
)
# Commands placed on a dedicated page; any other CLI subcommand gets its own tab on "More".
PLACED = {'inspect', 'paw-inventory', 'paw-resolve', 'paw-convert', 'recommend-kpoints', 'make-band-path', 'scf', 'prepare',
          'bands', 'dos', 'bands-compare', 'bands-analysis', 'relax', 'mp-fetch', 'mp-fetch-bands', 'backend-info',
          'doctor', 'validate', 'progress'}
PLOT_COMMANDS = ('scf', 'bands', 'dos', 'relax', 'bands-analysis', 'bands-compare')


class MainWindow(QMainWindow):
    language_changed = pyqtSignal(str)
    theme_changed = pyqtSignal(str)
    event_log = pyqtSignal(str, object)  # page events, used by tests

    def __init__(self, prefs: Preferences | None = None, *, jobs_root=None, app=None):
        super().__init__()
        self.app = app
        self.prefs = prefs or Preferences()
        self.theme_name = self.prefs.get('theme') if self.prefs.get('theme') in theme.THEMES else 'light'
        self.font_px = self.prefs.get('font_px')
        self.language = self._initial_language()
        set_language(self.language)
        self.mono_font = fonts.mono_font(self.language, max(self.font_px - 1, 10))
        self.parser = _safe_parser()
        self.available_commands = argspec.all_commands(self.parser) if self.parser else []
        self.forms: dict[str, CommandForm] = {}
        self.pages: dict[str, QWidget] = {}
        self.jobs = JobManager(self, root=jobs_root)
        self.jobs.env_provider = self.prefs.job_env
        self.active_job = None
        self.follower = None
        self.events: list = []
        self.setObjectName('Root')
        self.setMinimumSize(1100, 700)
        self.resize(1480, 920)
        self._build()
        self.jobs.job_added.connect(self._job_added)
        self.jobs.job_changed.connect(self._job_changed)
        self.jobs.job_output.connect(self._job_output)
        self.jobs.job_finished.connect(self._job_finished)
        self.poll_timer = QTimer(self)
        self.poll_timer.setInterval(500)
        self.poll_timer.timeout.connect(self._poll)
        self.poll_timer.start()
        self.apply_form_defaults()
        self.set_theme(self.theme_name)
        self.retranslate()
        self.nav.setCurrentRow(self._nav_row('scf'))

    # ---- setup
    def _initial_language(self):
        stored = self.prefs.get('language')
        return stored if stored in LANGUAGES else detect_language()

    def make_form(self, command) -> CommandForm:
        form = CommandForm(command, self.parser, self)
        form.defaults_provider = lambda c=command: self.prefs.form_defaults(c)
        form.submitted.connect(self.submit_form)
        form.message.connect(self.notify)
        self.forms[command] = form
        return form

    def _build(self):
        from .pages.analysis import AnalysisPage
        from .pages.base import CommandPage
        from .pages.commands import BandsPage, KpointsPage, MaterialsProjectPage, PawPage, RelaxPage, ScfPage
        from .pages.settings import SettingsPage
        from .pages.sparsity import SparsityPage
        from .pages.structure import StructurePage
        from .pages.system import DiagnosticsPage, QueuePage
        root = QWidget(self)
        root.setObjectName('Root')
        outer = QHBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        # navigation
        nav_panel = QWidget(root)
        nav_panel.setObjectName('NavPanel')
        nav_panel.setFixedWidth(224)
        nav_layout = QVBoxLayout(nav_panel)
        nav_layout.setContentsMargins(14, 18, 12, 14)
        nav_layout.setSpacing(4)
        self.brand = QLabel(nav_panel)
        self.brand.setObjectName('Brand')
        self.brand_sub = QLabel(nav_panel)
        self.brand_sub.setObjectName('BrandSub')
        nav_layout.addWidget(self.brand)
        nav_layout.addWidget(self.brand_sub)
        nav_layout.addSpacing(12)
        self.nav = QListWidget(nav_panel)
        self.nav.setObjectName('Nav')
        self.nav.setIconSize(QSize(18, 18))
        self.nav.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.nav.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        nav_layout.addWidget(self.nav, 1)
        self.theme_button = QPushButton(nav_panel)
        self.theme_button.setProperty('kind', 'ghost')
        self.theme_button.clicked.connect(lambda: self.set_theme('dark' if self.theme_name == 'light' else 'light', persist=True))
        self.lang_combo = no_wheel(QComboBox(nav_panel))
        for lang in LANGUAGES:
            self.lang_combo.addItem(tr(f'lang.{lang}'), lang)
        self.lang_combo.currentIndexChanged.connect(lambda: self.set_language(self.lang_combo.currentData(), persist=True))
        nav_layout.addWidget(self.lang_combo)
        nav_layout.addWidget(self.theme_button)
        outer.addWidget(nav_panel)
        # pages
        self.stack = QStackedWidget(root)
        self.nav_keys = []
        page_classes = {'structure': StructurePage, 'paw': PawPage, 'kpoints': KpointsPage, 'scf': ScfPage,
                        'bands': BandsPage, 'relax': RelaxPage, 'sparsity': SparsityPage, 'mp': MaterialsProjectPage,
                        'analysis': AnalysisPage, 'queue': QueuePage, 'diagnostics': DiagnosticsPage,
                        'settings': SettingsPage}
        for group, entries in NAV:
            header = QListWidgetItem('')
            header.setFlags(Qt.ItemFlag.NoItemFlags)
            header.setData(Qt.ItemDataRole.UserRole, f'group:{group}')
            self.nav.addItem(header)
            self.nav_keys.append(f'group:{group}')
            for key, icon_name in entries:
                page = page_classes[key](self)
                self.pages[key] = page
                self.stack.addWidget(page)
                item = QListWidgetItem('')
                item.setData(Qt.ItemDataRole.UserRole, key)
                item.setData(Qt.ItemDataRole.UserRole + 1, icon_name)
                self.nav.addItem(item)
                self.nav_keys.append(key)
        extra = [c for c in self.available_commands if c not in PLACED and argspec.command_specs(c, self.parser)]
        if extra:
            page = CommandPage(self, commands=extra)
            page.key = 'more'
            self.pages['more'] = page
            self.stack.addWidget(page)
            item = QListWidgetItem('')
            item.setData(Qt.ItemDataRole.UserRole, 'more')
            item.setData(Qt.ItemDataRole.UserRole + 1, 'diagnostics')
            self.nav.addItem(item)
            self.nav_keys.append('more')
        self.nav.currentRowChanged.connect(self._nav_changed)
        # right side: run monitor (live chart, job selector) above the log console
        self.splitter = QSplitter(Qt.Orientation.Horizontal, root)
        self.splitter.setChildrenCollapsible(False)
        center = QWidget(self.splitter)
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)
        self.banner = Banner(center)
        banner_wrap = QWidget(center)
        banner_layout = QVBoxLayout(banner_wrap)
        banner_layout.setContentsMargins(20, 10, 16, 0)
        banner_layout.addWidget(self.banner)
        center_layout.addWidget(banner_wrap)
        center_layout.addWidget(self.stack, 1)
        side = QSplitter(Qt.Orientation.Vertical, self.splitter)
        side.setChildrenCollapsible(False)
        monitor = Card(side, margins=(14, 12, 14, 12), spacing=8)
        head = QHBoxLayout()
        self.monitor_title = QLabel(monitor)
        self.monitor_title.setProperty('role', 'section')
        self.monitor_chip = Chip('idle', monitor)
        head.addWidget(self.monitor_title)
        head.addStretch(1)
        head.addWidget(self.monitor_chip)
        monitor.body.addLayout(head)
        row = QHBoxLayout()
        self.follow_label = QLabel(monitor)
        self.follow_label.setProperty('role', 'muted')
        self.job_combo = no_wheel(QComboBox(monitor))
        self.job_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.job_combo.setMinimumContentsLength(14)
        self.job_combo.currentIndexChanged.connect(self._job_combo_changed)
        self.monitor_cancel = QPushButton(monitor)
        self.monitor_cancel.setProperty('kind', 'danger')
        self.monitor_cancel.setFixedWidth(34)
        self.monitor_cancel.clicked.connect(lambda: self.active_job and self.jobs.cancel(self.active_job))
        row.addWidget(self.follow_label)
        row.addWidget(self.job_combo, 1)
        row.addWidget(self.monitor_cancel)
        monitor.body.addLayout(row)
        self.monitor_detail = QLabel(monitor)
        self.monitor_detail.setProperty('role', 'muted')
        self.monitor_detail.setWordWrap(True)
        monitor.body.addWidget(self.monitor_detail)
        self.chart = ConvergenceChart(monitor)
        monitor.body.addWidget(self.chart, 1)
        self.follower = IterationFollower(self.chart)
        console_card = Card(side, margins=(14, 12, 14, 12), spacing=6)
        self.console = LogConsole(console_card)
        self.console.set_mono_font(self.mono_font)
        console_card.body.addWidget(self.console)
        side.addWidget(monitor)
        side.addWidget(console_card)
        side.setSizes([430, 380])
        side_wrap = QWidget(self.splitter)
        side_layout = QVBoxLayout(side_wrap)
        side_layout.setContentsMargins(0, 14, 14, 12)
        side_layout.addWidget(side)
        self.splitter.addWidget(center)
        self.splitter.addWidget(side_wrap)
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 2)
        side_wrap.setMinimumWidth(340)
        center.setMinimumWidth(520)
        self._splitter_ratio = 0.6  # centre share until the user drags the handle
        self.splitter.splitterMoved.connect(self._remember_split)
        outer.addWidget(self.splitter, 1)
        self.setCentralWidget(root)
        self.status = QStatusBar(self)
        self.status_label = QLabel(self.status)
        self.status.addWidget(self.status_label, 1)
        self.setStatusBar(self.status)
        self._refresh_job_combo()

    def _remember_split(self, *_):
        sizes = self.splitter.sizes()
        if sum(sizes):
            self._splitter_ratio = sizes[0] / sum(sizes)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        total = sum(self.splitter.sizes()) or self.splitter.width()
        if total > 0:
            centre = max(520, min(total - 340, int(total * self._splitter_ratio)))
            self.splitter.blockSignals(True)
            self.splitter.setSizes([centre, total - centre])
            self.splitter.blockSignals(False)

    # ---- navigation
    def _nav_row(self, key):
        return self.nav_keys.index(key) if key in self.nav_keys else 1

    def _nav_changed(self, row):
        if row < 0 or row >= len(self.nav_keys):
            return
        key = self.nav_keys[row]
        if key.startswith('group:'):
            self.nav.setCurrentRow(min(row + 1, len(self.nav_keys) - 1))
            return
        self.stack.setCurrentWidget(self.pages[key])

    def show_page(self, key):
        self.nav.setCurrentRow(self._nav_row(key))
        return self.pages.get(key)

    def current_page_key(self):
        row = self.nav.currentRow()
        return self.nav_keys[row] if 0 <= row < len(self.nav_keys) else None

    # ---- jobs
    def submit_form(self, payload: dict):
        try:
            job = self.jobs.submit(payload['argv'], label=payload.get('label'), secret_env=payload.get('secret_env'),
                                   command=payload.get('command'))
        except Exception as error:  # noqa: BLE001 - never crash on a bad submission
            self.notify(tr('msg.error', message=str(error)), 'error')
            return None
        return job

    def _job_added(self, job):
        self.pages['queue'].add_job(job)
        self._refresh_job_combo()
        self.follow_job(job)
        self.notify(tr('msg.job_started', label=job.label), 'info')
        self.console.append(job.id, 'stderr', f'$ {job.display}', 'info')

    def _job_changed(self, job):
        self.pages['queue'].update_job(job)
        if job is self.active_job:
            self._update_monitor()
        self._update_status()

    def _job_output(self, job, stream, text):
        self.console.append(job.id, stream, text)

    def _job_finished(self, job):
        self.pages['queue'].update_job(job)
        if job is self.active_job:
            self.follower.poll(final=True)
            self._update_monitor()
        key = {'finished': 'msg.job_finished', 'failed': 'msg.job_failed', 'cancelled': 'msg.job_cancelled'}[job.state]
        level = {'finished': 'ok', 'failed': 'error', 'cancelled': 'warning'}[job.state]
        self.notify(tr(key, label=job.label, code=job.returncode), level)
        self._refresh_job_combo()
        self._update_status()
        self.event_log.emit('job_finished', job)
        self.events.append(('job_finished', job.id, job.state))
        if job.state == 'finished' and job.command in PLOT_COMMANDS and job.output_dir:
            QTimer.singleShot(50, lambda j=job: self.auto_preview(j))

    def auto_preview(self, job):
        """Generate figure previews for a finished result if the plotting API is present."""
        if not plots.available() or not job.output_dir or not Path(job.output_dir).exists():
            return False
        page = self.pages['analysis']
        page.add_recent(job.output_dir)
        ok = page.show_result(job.output_dir)
        if ok:
            self.notify(tr('msg.plots_ready', n=len(page.kinds), label=job.label), 'ok')
        return ok

    def follow_job(self, job):
        self.active_job = job
        self.follower.follow(job.output_dir if job and job.output_dir else None)
        index = self.job_combo.findData(job.id if job else None)
        if index >= 0 and index != self.job_combo.currentIndex():
            self.job_combo.blockSignals(True)
            self.job_combo.setCurrentIndex(index)
            self.job_combo.blockSignals(False)
        self.console.set_job_filter(None)
        self._update_monitor()

    def focus_job(self, job):
        self.follow_job(job)
        self.console.set_job_filter(job.id)

    def _refresh_job_combo(self):
        current = self.active_job.id if self.active_job else None
        self.job_combo.blockSignals(True)
        self.job_combo.clear()
        if not self.jobs.jobs:
            self.job_combo.addItem(tr('monitor.none'), None)
        for job in reversed(self.jobs.jobs):
            self.job_combo.addItem(f'{job.label} · {tr("state." + job.state)}', job.id)
        index = self.job_combo.findData(current)
        self.job_combo.setCurrentIndex(index if index >= 0 else 0)
        self.job_combo.blockSignals(False)

    def _job_combo_changed(self, index):
        job_id = self.job_combo.itemData(index)
        job = next((j for j in self.jobs.jobs if j.id == job_id), None)
        if job is not None and job is not self.active_job:
            self.follow_job(job)

    def _update_monitor(self):
        job = self.active_job
        if job is None:
            self.monitor_chip.set_state('idle')
            self.monitor_detail.setText(tr('monitor.none'))
            self.monitor_cancel.setEnabled(False)
            return
        self.monitor_chip.set_state(job.state)
        detail = f'{job.display}\n{format_duration(job.duration)}'
        if job.output_dir:
            detail += f' · {job.output_dir}'
        self.monitor_detail.setText(detail)
        self.monitor_detail.setToolTip(detail)
        self.monitor_cancel.setEnabled(not job.finished())

    def _poll(self):
        if self.active_job is not None and not self.active_job.finished():
            self.follower.poll()

    def _update_status(self):
        running = len(self.jobs.running())
        self.status_label.setText(tr('msg.running_jobs', n=running) if running else tr('msg.ready'))

    # ---- messages
    def notify(self, text, level='info'):
        self.banner.show_message(text, level)
        if level in ('error', 'warning'):
            self.console.append('gui', 'stderr', f'[gui] {text}', level)

    def log_line(self, job_id, stream, text):
        self.console.append(job_id, stream, text)

    def page_event(self, name, value):
        self.events.append((name, value))
        self.event_log.emit(name, value)

    # ---- forms
    def broadcast_values(self, values, commands=None):
        for name, form in self.forms.items():
            if commands is None or name in commands:
                form.set_values({k: v for k, v in values.items() if k in form.fields})

    def apply_form_defaults(self):
        for form in self.forms.values():
            form.apply_defaults()

    def settings_applied(self):
        self.apply_form_defaults()
        self.pages['analysis'].apply_plot_defaults(self.prefs)
        self.pages['diagnostics'].refresh()

    # ---- appearance
    def set_theme(self, name, *, persist=False):
        name = name if name in theme.THEMES else 'light'
        self.theme_name = name
        p = theme.palette(name)
        app = self.app
        sheet = theme.stylesheet(name, font_px=self.font_px) + _extra_qss(p)
        target = app if app is not None else self
        if app is not None:
            theme.apply_palette(app, name)
        if target.styleSheet() != sheet:  # re-polishing every widget is the expensive part
            target.setStyleSheet(sheet)
        self.banner.palette_colours = {'error': p['err_text'], 'warning': p['warn_text'],
                                       'info': p['info_text'], 'ok': p['ok_text']}
        for form in self.forms.values():
            form.set_icon_colour(p['muted'])
        for page in self.pages.values():
            page.set_theme(p)
        self.console.set_theme(p)
        self.chart.set_theme(name)
        self.monitor_cancel.setIcon(icons.icon('stop', '#3A1020', 14))
        self._update_nav_icons(p)
        self.theme_button.setIcon(icons.icon('moon' if name == 'light' else 'sun', p['muted'], 16))
        self.theme_button.setText(tr('settings.theme_dark' if name == 'light' else 'settings.theme_light'))
        if persist:
            self.prefs.set('theme', name)
            settings = self.pages.get('settings')
            if settings is not None:
                settings._select(settings.theme, name)
        self.theme_changed.emit(name)

    def _update_nav_icons(self, p):
        for row in range(self.nav.count()):
            item = self.nav.item(row)
            icon_name = item.data(Qt.ItemDataRole.UserRole + 1)
            if icon_name:
                item.setIcon(icons.icon(icon_name, p['text'], 18))

    def set_font_px(self, px, *, persist=False):
        self.font_px = int(px)
        self._apply_fonts()
        self.set_theme(self.theme_name)
        if persist:
            self.prefs.set('font_px', self.font_px)

    def _apply_fonts(self):
        font = fonts.ui_font(self.language, self.font_px)
        if self.app is not None:
            self.app.setFont(font)
        self.setFont(font)
        self.mono_font = fonts.mono_font(self.language, max(self.font_px - 1, 10))
        self.console.set_mono_font(self.mono_font)

    def set_language(self, language, *, persist=False):
        if language == 'auto':
            if persist:
                self.prefs.set('language', 'auto')
            language = detect_language()
        language = language if language in LANGUAGES else 'en'
        self.language = set_language(language)
        if persist and self.prefs.get('language') != 'auto':
            self.prefs.set('language', language)
        if self.app is not None:
            from .app import install_qt_translator
            install_qt_translator(self.app, language)
        self._apply_fonts()
        self.set_theme(self.theme_name)
        self.retranslate()
        self.notify(tr('msg.language_changed', name=tr(f'lang.{language}')), 'info')
        self.language_changed.emit(language)
        return language

    def retranslate(self):
        t = translator()
        self.setWindowTitle(f'{tr("app.title")} — {tr("app.subtitle")}')
        self.brand.setText(tr('app.title'))
        self.brand_sub.setText(tr('app.subtitle'))
        for row, key in enumerate(self.nav_keys):
            item = self.nav.item(row)
            if key.startswith('group:'):
                item.setText(tr(f'nav.group.{key.split(":", 1)[1]}').upper() if t.language == 'en'
                             else tr(f'nav.group.{key.split(":", 1)[1]}'))
            else:
                item.setText(tr(f'nav.{key}'))
        for page in self.pages.values():
            page.retranslate()
        self.lang_combo.blockSignals(True)
        for i, lang in enumerate(LANGUAGES):
            self.lang_combo.setItemText(i, tr(f'lang.{lang}'))
        self.lang_combo.setCurrentIndex(LANGUAGES.index(self.language))
        self.lang_combo.blockSignals(False)
        self.theme_button.setText(tr('settings.theme_dark' if self.theme_name == 'light' else 'settings.theme_light'))
        self.monitor_title.setText(tr('monitor.title'))
        self.follow_label.setText(tr('monitor.follow'))
        self.monitor_cancel.setToolTip(tr('btn.cancel'))  # icon-only: the side panel is narrow
        self.console.retranslate()
        self.chart.retranslate()
        self._refresh_job_combo()
        self._update_monitor()
        self._update_status()

    # ---- shutdown
    def closeEvent(self, event):
        # Workers are independent process groups: stop them so no solver outlives the GUI.
        self.jobs.cancel_all(wait_ms=3000)
        super().closeEvent(event)


def _safe_parser():
    try:
        from sparse_dft.cli import build_parser
        return build_parser()
    except Exception:  # noqa: BLE001 - the GUI still opens with a broken CLI import
        return None


def _extra_qss(p) -> str:
    return f"""
QToolButton#SectionHeader {{ background: transparent; border: none; border-bottom: 1px solid {p['border']};
    border-radius: 0; padding: 10px 16px; font-weight: 700; text-align: left; color: {p['text']}; }}
QToolButton#SectionHeader:hover {{ background: {p['surface_alt']}; }}
QToolButton#SectionHeader:!checked {{ border-bottom: 1px solid transparent; }}
QWidget#SectionBody {{ background: transparent; }}
QLabel[role="flag"] {{ color: {p['muted']}; font-family: "Menlo", "Consolas", "DejaVu Sans Mono", monospace;
    font-size: {11}px; }}
QListWidget#Nav::item:disabled {{ color: {p['muted']}; background: transparent; padding: 14px 8px 4px 8px;
    font-size: 11px; font-weight: 700; letter-spacing: 1px; }}
QTabWidget::pane {{ background: {p['surface']}; }}
QTabWidget > QWidget > QWidget {{ background: {p['surface']}; }}
"""
