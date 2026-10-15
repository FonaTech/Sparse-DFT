"""Sparse-DFT GUI (PyQt6, offscreen): forms from argparse, i18n, themes, job queue, plot preview."""
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

try:
    from PyQt6.QtCore import QLocale, QSettings, Qt
    from PyQt6.QtWidgets import QApplication
    HAVE_QT = True
except ImportError:  # pragma: no cover - environments without the gui extra
    HAVE_QT = False

REPO = Path(__file__).resolve().parents[1]
SI = REPO / 'examples' / 'Si.POSCAR'
SAMPLE_RESULTS = [Path('/tmp/sparse_dft_sweep/small') / name / 'reference' for name in ('Si2', 'Ni', 'MgO', 'Al_slab')]
ROOT = Path(__file__).resolve().parents[1]
GRANDCHILD = ("import subprocess, sys\n"
              "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'])\n"
              "print('grandchild', p.pid, '子进程', flush=True)\n"
              "p.wait()\n")
APP = None


def setUpModule():
    global APP
    if HAVE_QT:
        from sparse_dft_gui.app import create_app
        APP = create_app(['test'])


def pump(predicate=None, timeout=10.0):
    """Process Qt events until ``predicate()`` is true (or for ``timeout`` seconds without one)."""
    deadline = time.monotonic() + (timeout if predicate is not None else min(timeout, 0.2))
    while time.monotonic() < deadline:
        APP.processEvents()
        if predicate is not None and predicate():
            return True
        time.sleep(0.01)
    APP.processEvents()
    return predicate() if predicate is not None else True


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    if os.name != 'nt':
        state = subprocess.run(['ps', '-o', 'stat=', '-p', str(pid)], capture_output=True, text=True,
                               encoding='utf-8', errors='replace').stdout.strip()
        return bool(state) and not state.startswith('Z')
    return True


def sample_value(spec):
    """A valid non-default widget value for ``spec`` (used for argv round trips)."""
    if spec.widget == 'check':
        return not bool(spec.default)
    if spec.widget == 'count':
        return 2
    if spec.choices:
        options = [c for c in spec.choices if c != spec.default]
        choice = options[-1] if options else spec.choices[0]
        return [str(choice)] if spec.widget == 'fixed' else (
            str(choice) if spec.widget != 'multi' else str(choice))
    if spec.widget == 'fixed':
        return {'float': ['0.125', '0.25', '0.5'], 'int': ['2', '3', '4']}.get(spec.value_type, ['a', 'b', 'c'])[:spec.count]
    if spec.widget == 'multi':
        return {'float': '1.5 -0.5', 'int': '2 3'}.get(spec.value_type, 'Si O')
    if spec.widget == 'append':
        return 'Fe:2:4 Ni:2:3'
    if spec.widget == 'float':
        return '0.0375' if spec.default != 0.0375 else '0.05'
    if spec.widget == 'int':
        return '7' if spec.default != 7 else '8'
    return f'目录 with spaces/{spec.dest}'


@unittest.skipUnless(HAVE_QT, 'PyQt6 is not installed')
class ArgspecTests(unittest.TestCase):
    def test_every_command_builds_and_round_trips(self):
        from sparse_dft.cli import build_parser
        from sparse_dft_gui import argspec
        from sparse_dft_gui.forms import CommandForm
        parser = build_parser()
        commands = argspec.all_commands(parser)
        self.assertIn('scf', commands)
        self.assertNotIn('worker', commands)
        sub = argspec.subparsers(parser)
        for command in commands:
            with self.subTest(command=command):
                form = CommandForm(command, parser)
                specs = form.specs
                self.assertEqual(set(form.fields), {s.dest for s in specs})
                if not specs:
                    self.assertEqual(form.argv(), [command])
                    continue
                values = {s.dest: sample_value(s) for s in specs}
                form.set_values(values)
                argv = form.argv()
                self.assertEqual(argv[0], command)
                # Delegated commands (plot) own their parser; resolve it the
                # same way the GUI does instead of the dispatcher placeholder.
                real = argspec.parser_for(command, parser) or sub.choices[command]
                namespace = real.parse_args(argv[1:])  # real argparse accepts it
                for spec in specs:
                    self.assertTrue(hasattr(namespace, spec.dest))
                parsed_command, parsed = argspec.parse_argv(argv, parser)
                self.assertEqual(parsed_command, command)
                for spec in specs:
                    expected = form.fields[spec.dest].value()
                    self.assertEqual(parsed[spec.dest], expected, spec.dest)
                fresh = CommandForm(command, parser)
                fresh.load_argv(argv)
                self.assertEqual(fresh.argv(), argv)

    def test_widget_values_from_typed_presets(self):
        from sparse_dft.cli import build_parser
        from sparse_dft_gui import argspec
        specs = {s.dest: s for s in argspec.command_specs('scf', build_parser())}
        relax = {s.dest: s for s in argspec.command_specs('relax', build_parser())}
        wv, cv = argspec.widget_value, argspec.canonical_value
        self.assertEqual(wv(specs['scf_grid'], [8, 8, 1]), ['8', '8', '1'])  # fixed -> list of str
        self.assertEqual(wv(specs['initial_moments'], [2.2, 0.0]), '2.2 0.0')  # multi -> space-joined
        self.assertEqual(wv(specs['hubbard'], ['Fe:2:5.3', 'Ni:2:6.2']), 'Fe:2:5.3 Ni:2:6.2')
        self.assertIs(wv(relax['cell'], False), False)  # check -> bool
        self.assertEqual(wv(specs['spin_channels'], 2), '2')
        self.assertEqual(wv(specs['smearing_ev'], None), argspec.default_value(specs['smearing_ev']))
        # empty == default; typed tokens compare by value, not text
        self.assertEqual(cv(specs['smearing_ev'], ''), cv(specs['smearing_ev'], '0.0'))
        self.assertEqual(cv(specs['scf_grid'], ['8', '8', '8']), cv(specs['scf_grid'], [' 8', '8 ', '8']))
        self.assertNotEqual(cv(specs['scf_grid'], ['', '', '']), cv(specs['scf_grid'], ['8', '8', '8']))
        values = {d: wv(specs[d], v) for d, v in {'scf_grid': [4, 4, 4], 'initial_moments': [2.2],
                                                   'spin_channels': 2, 'structure': 's', 'paw_root': 'p'}.items()}
        ns = build_parser().parse_args(argspec.build_argv('scf', list(specs.values()), values))
        self.assertEqual((ns.scf_grid, ns.initial_moments, ns.spin_channels), ([4, 4, 4], [2.2], 2))

    def test_defaults_are_omitted_and_required_fields_validated(self):
        from sparse_dft_gui import argspec
        from sparse_dft_gui.forms import CommandForm
        form = CommandForm('scf')
        with self.assertRaises(argspec.FormError) as caught:
            form.argv()
        self.assertEqual(set(caught.exception.errors), {'structure', 'paw_root'})
        form.set_values({'structure': str(SI), 'paw_root': '/data/PAW 根'})
        self.assertEqual(form.argv(), ['scf', str(SI), '--paw-root', '/data/PAW 根'])
        form.set_values({'accuracy': 'low', 'max_iterations': '60', 'density_tolerance': '1e-5',
                         'kpoint': ['0', '0', '0.5'], 'hubbard': 'Fe:2:4'})
        argv = form.argv()
        self.assertEqual(argv[argv.index('--accuracy') + 1], 'low')
        self.assertEqual(argv[argv.index('--kpoint') + 1:argv.index('--kpoint') + 4], ['0', '0', '0.5'])
        self.assertIn('--hubbard', argv)
        form.set_values({'density_tolerance': 'abc'})
        with self.assertRaises(argspec.FormError) as caught:
            form.argv()
        self.assertIn('density_tolerance', caught.exception.errors)

    def test_mpi_options_and_future_options_are_picked_up(self):
        # The lead's MPI controls and any option added later must appear without GUI changes.
        from sparse_dft.cli import build_parser
        from sparse_dft_gui import argspec
        from sparse_dft_gui.forms import CommandForm
        parser = build_parser()
        for command in ('scf', 'bands'):
            form = CommandForm(command, parser)
            for dest in ('mpi_ranks', 'kgroups', 'bandgroups', 'threads_per_rank'):
                self.assertIn(dest, form.fields, f'{command}: {dest}')
        sub = argspec.subparsers(parser)
        group = sub.choices['scf'].add_argument_group('future controls (test)')
        group.add_argument('--future-knob', type=float, default=None, help='a later option')
        group.add_argument('--future-mode', choices=['a', 'b'], default='a')
        form = CommandForm('scf', parser)
        self.assertIn('future_knob', form.fields)
        self.assertTrue(any(name.startswith('group:future controls') for name in form.sections))
        form.set_values({'structure': 'a b.POSCAR', 'paw_root': 'p', 'mpi_ranks': '4', 'kgroups': '2',
                         'future_knob': '2.5e-3', 'future_mode': 'b'})
        argv = form.argv()
        self.assertEqual(argv[argv.index('--mpi-ranks') + 1], '4')
        self.assertEqual(argv[argv.index('--kgroups') + 1], '2')
        self.assertEqual(argv[argv.index('--future-knob') + 1], '2.5e-3')
        self.assertEqual(argv[argv.index('--future-mode') + 1], 'b')
        self.assertNotIn('--bandgroups', argv)  # still at its default

    def test_sparsity_group_and_help_tooltips(self):
        from sparse_dft_gui.forms import CommandForm
        form = CommandForm('scf')
        groups = [name for name in form.sections if name.startswith('group:')]
        self.assertTrue(any(g.startswith('group:sparsification') for g in groups), groups)
        self.assertTrue(any(g.startswith('group:process parallelism') for g in groups), groups)
        sparsity_section = next(form.sections[g] for g in groups if g.startswith('group:sparsification'))
        self.assertFalse(sparsity_section.expanded())  # advanced groups start collapsed
        for dest in ('sparsity', 'density_cutoff_factor', 'xc_padding', 'eigen_search_factor', 'block_ortho',
                     'paw_compensation', 'projector_space', 'projector_radius_factor', 'search_precision'):
            self.assertIn(dest, form.fields)
        self.assertIn('S1', form.fields['density_cutoff_factor'].editor.toolTip())
        self.assertEqual(form.fields['structure'].spec.path_mode, 'open')
        self.assertEqual(form.fields['paw_root'].spec.path_mode, 'dir')
        self.assertEqual(form.fields['out'].spec.path_mode, 'dir')
        self.assertEqual(CommandForm('bands').fields['scf_checkpoint'].spec.path_mode, 'open')
        self.assertEqual(CommandForm('make-band-path').fields['out'].spec.path_mode, 'save')

    def test_command_preview_quotes_paths_and_masks_secrets(self):
        from sparse_dft_gui.forms import CommandForm
        form = CommandForm('scf')
        form.set_values({'structure': '/tmp/测试 目录/Si.POSCAR', 'paw_root': '/PAW Raw Data'})
        self.assertEqual(form.preview.text(), "sparse-dft scf '/tmp/测试 目录/Si.POSCAR' --paw-root '/PAW Raw Data'")
        mp = CommandForm('mp-fetch')
        mp.set_values({'api_key': 'SECRET-KEY-123'})
        self.assertNotIn('SECRET-KEY-123', mp.preview.text())
        self.assertIn('--api-key', mp.preview.text())
        self.assertIn('***', mp.preview.text())
        payloads = []
        mp.submitted.connect(payloads.append)
        mp.submit()
        self.assertNotIn('SECRET-KEY-123', json.dumps(payloads[0]['argv']))
        self.assertEqual(payloads[0]['secret_env'], {'MP_API_KEY': 'SECRET-KEY-123'})
        from sparse_dft_gui.argspec import display_command
        self.assertEqual(display_command(['scf', 'C:\\数据 目录\\Si.POSCAR'], windows=True),
                         'sparse-dft scf "C:\\数据 目录\\Si.POSCAR"')


@unittest.skipUnless(HAVE_QT, 'PyQt6 is not installed')
class I18nTests(unittest.TestCase):
    def test_locale_detection(self):
        from sparse_dft_gui.i18n import detect_language
        cases = {'zh_CN': 'zh_CN', 'zh_Hant_TW': 'zh_TW', 'zh_TW': 'zh_TW', 'zh_HK': 'zh_TW', 'zh_MO': 'zh_TW',
                 'zh_SG': 'zh_CN', 'ja_JP': 'ja', 'fr_FR': 'en', 'en_US': 'en', 'de_DE': 'en'}
        for name, expected in cases.items():
            with self.subTest(locale=name):
                self.assertEqual(detect_language(QLocale(name)), expected)
                self.assertEqual(detect_language(name), expected)
        self.assertEqual(detect_language(QLocale('xx_broken')), 'en')
        self.assertEqual(detect_language('!!broken!!'), 'en')
        self.assertEqual(detect_language(object()), 'en')  # any exception -> en
        self.assertIn(detect_language(), ('en', 'zh_CN', 'zh_TW', 'ja'))

    def test_catalogs_consistent(self):
        from sparse_dft_gui.i18n import LANGUAGES, catalog_dir
        catalogs = {lang: json.loads((catalog_dir() / f'{lang}.json').read_text(encoding='utf-8')) for lang in LANGUAGES}
        keys = set(catalogs['en'])
        self.assertGreater(len(keys), 200)
        for lang, catalog in catalogs.items():
            with self.subTest(lang=lang):
                self.assertEqual(set(catalog), keys, f'{lang}: {sorted(set(catalog) ^ keys)[:10]}')
                empty = [k for k, v in catalog.items() if not isinstance(v, str) or not v.strip()]
                self.assertEqual(empty, [])
                self.assertEqual(catalog['_meta.language'], lang)
        # placeholders must match between languages
        import string
        def fields(text):
            return sorted(f for _, f, _, _ in string.Formatter().parse(text) if f)
        for key in keys:
            for lang in LANGUAGES:
                self.assertEqual(fields(catalogs[lang][key]), fields(catalogs['en'][key]), f'{lang}:{key}')

    def test_help_translations_cover_common_groups_and_track_cli(self):
        # scf/bands/relax help (sparsity and parallel groups included) is translated.  The CLI
        # text is authoritative: a catalog entry whose English no longer matches argparse is
        # not shown (English fallback), so stale translations never mislead.  Coverage of
        # current help texts must stay high; mismatches are listed for the next catalog sync.
        from sparse_dft.cli import build_parser
        from sparse_dft_gui import argspec
        from sparse_dft_gui.i18n import Translator
        parser = build_parser()
        english, zh = Translator('en'), Translator('zh_CN')
        total, synced, stale = 0, 0, []
        for command in ('scf', 'bands', 'relax'):
            for spec in argspec.command_specs(command, parser):
                total += 1
                key = next((k for k in (f'help.{command}.{spec.dest}', f'help.{spec.dest}') if english.has(k)), None)
                if key is not None and (not spec.help or english.english(key) == spec.help):
                    synced += 1
                    self.assertNotEqual(zh.option_help(spec), spec.help or '__none__', f'{command}.{spec.dest}')
                else:
                    stale.append(f'{command}.{spec.dest}')
                    self.assertEqual(zh.option_help(spec), spec.help or '')  # English fallback
        self.assertGreaterEqual(synced / total, 0.9, f'untranslated or changed CLI help: {stale}')
        spec = next(s for s in argspec.command_specs('scf', parser) if s.dest == 'smearing_ev')
        self.assertIn('展宽', zh.option_help(spec))
        spec.help = 'changed upstream'
        self.assertEqual(zh.option_help(spec), 'changed upstream')


@unittest.skipUnless(HAVE_QT, 'PyQt6 is not installed')
class WindowTests(unittest.TestCase):
    def setUp(self):
        from sparse_dft_gui.app import create_window
        from sparse_dft_gui.prefs import Preferences
        self._tmp = tempfile.TemporaryDirectory(prefix='sparse-dft gui 测试 ')
        self.tmp = Path(self._tmp.name)
        self.settings = QSettings(str(self.tmp / 'prefs.ini'), QSettings.Format.IniFormat)
        self.prefs = Preferences(self.settings)
        self.prefs.set('language', 'en')
        self.prefs.set('output_root', str(self.tmp / 'runs 输出'))
        self.window = create_window(APP, self.prefs, jobs_root=self.tmp / 'jobs 任务')
        self.window.resize(1400, 900)
        self.window.show()
        pump()

    def tearDown(self):
        self.window.jobs.cancel_all(wait_ms=3000)
        self.window.close()
        self.window.deleteLater()
        pump()
        self._tmp.cleanup()

    def test_main_window_constructs_with_all_pages(self):
        for key in ('structure', 'paw', 'kpoints', 'scf', 'bands', 'relax', 'sparsity', 'mp', 'analysis',
                    'diagnostics', 'queue', 'settings'):
            with self.subTest(page=key):
                page = self.window.show_page(key)
                pump()
                self.assertIs(self.window.stack.currentWidget(), page)
        from sparse_dft_gui import argspec
        covered = set(self.window.forms) | {'inspect'}
        self.assertEqual(set(argspec.all_commands(self.window.parser)) - covered - {'plot'}, set())
        self.assertEqual(self.window.forms['scf'].fields['out'].value(), str(self.tmp / 'runs 输出' / 'scf'))

    def test_runtime_language_switch_retranslates(self):
        w = self.window
        seen = {}
        for lang in ('zh_CN', 'zh_TW', 'ja', 'en'):
            w.set_language(lang)
            pump()
            seen[lang] = (w.pages['scf'].title.text(), w.forms['scf'].run_button.text(),
                          w.forms['scf'].fields['max_iterations'].label.text(), w.windowTitle())
        self.assertEqual(seen['zh_CN'][0], '自洽计算')
        self.assertEqual(seen['zh_TW'][1], '執行')
        self.assertEqual(seen['ja'][2], '最大反復回数')
        self.assertEqual(seen['en'][1], 'Run')
        self.assertEqual(len({v[3] for v in seen.values()}), 4)
        w.set_language('zh_TW', persist=True)
        self.assertEqual(self.prefs.get('language'), 'zh_TW')

    def test_theme_switch(self):
        w = self.window
        w.set_theme('dark', persist=True)
        pump()
        from sparse_dft_gui import theme
        self.assertIn(theme.DARK['bg'], APP.styleSheet())
        self.assertEqual(APP.palette().window().color().name().upper(), theme.DARK['bg'].upper())
        self.assertEqual(self.prefs.get('theme'), 'dark')
        w.set_theme('light')
        self.assertEqual(APP.palette().window().color().name().upper(), theme.LIGHT['bg'].upper())

    def test_theme_contrast_wcag_aa(self):
        from sparse_dft_gui import theme
        for p in (theme.LIGHT, theme.DARK):
            for fg, bg in (('text', 'bg'), ('text', 'surface'), ('muted', 'surface'), ('muted', 'bg'),
                           ('muted', 'nav'), ('text', 'accent'), ('err_text', 'err_bg'), ('info_text', 'info_bg'),
                           ('ok_text', 'ok_bg'), ('warn_text', 'warn_bg'), ('console_text', 'console_bg'),
                           ('accent_strong', 'surface')):
                with self.subTest(theme=p['name'], pair=(fg, bg)):
                    self.assertGreaterEqual(theme.contrast_ratio(p[fg], p[bg]), 4.5)
            for colour in p['lines']:
                self.assertGreaterEqual(theme.contrast_ratio(colour, p['plot_bg']), 3.0)
        # dark text on every pastel chip/button
        for pastel in ('pistachio', 'lavender', 'peach', 'blue', 'rose', 'lemon'):
            self.assertGreaterEqual(theme.contrast_ratio('#24203A', theme.LIGHT[pastel]), 4.5)

    def test_mock_job_runs_and_is_cancelled_through_queue(self):
        w = self.window
        job = w.jobs.submit([sys.executable, '-c', GRANDCHILD], kind='exec', label='模拟 job')
        self.assertTrue(pump(lambda: any(t.startswith('grandchild') for _, t in job.lines), 30), job.lines)
        grandchild = int(next(t for _, t in job.lines if t.startswith('grandchild')).split()[1])
        self.assertEqual(job.state, 'running')
        queue = w.show_page('queue')
        pump()
        self.assertEqual(queue.table.rowCount(), 1)
        queue.table.selectRow(0)
        self.assertTrue(queue.cancel_button.isEnabled())
        started = time.monotonic()
        queue.cancel_button.click()
        self.assertTrue(pump(job.finished, 15))
        self.assertLess(time.monotonic() - started, 10)
        self.assertEqual(job.state, 'cancelled')
        self.assertTrue(pump(lambda: not _alive(grandchild), 8), 'grandchild survived the cancel')
        self.assertIn('子进程', '\n'.join(t for _, t in job.lines))
        self.assertIn('子进程', w.console.visible_text())
        self.assertTrue(queue.rerun_button.isEnabled())
        self.assertIn(('job_finished', job.id, 'cancelled'), w.events)

    def test_form_submission_runs_worker_and_follows_iterations(self):
        w = self.window
        out = self.tmp / 'iter 输出'
        out.mkdir()
        log = out / 'scf.json.iterations.jsonl'
        code = ("import json, time, sys\n"
                f"p = {str(log)!r}\n"
                "with open(p, 'w', encoding='utf-8') as f:\n"
                "    for i in range(1, 6):\n"
                "        f.write(json.dumps({'iteration': i, 'density_residual': 10.0**-i, "
                "'output_magnetic_moment_muB': 0.5}) + '\\n'); f.flush(); time.sleep(0.25)\n"
                "print('完成')\n")
        job = w.jobs.submit([sys.executable, '-c', code], kind='exec', label='iterations')
        job.output_dir = str(out)
        w.follow_job(job)
        self.assertTrue(pump(job.finished, 30))
        self.assertEqual(job.state, 'finished', job.lines)
        self.assertEqual(len(w.chart.records), 5)
        self.assertEqual(w.chart.records[-1]['iteration'], 5)

    def test_gbk_console_environment_keeps_chinese_output(self):
        # Settings may carry a GBK console environment (as on a Chinese Windows); the worker
        # still speaks UTF-8 to QProcess and every line reaches the console intact.
        self.prefs.set('extra_env', 'PYTHONIOENCODING=gbk\nLANG=zh_CN.GBK\nLC_ALL=zh_CN.GBK')
        poscar = self.tmp / '硅 Si 晶体.POSCAR'
        poscar.write_text(SI.read_text(encoding='utf-8'), encoding='utf-8')
        job = self.window.jobs.submit(['inspect', str(poscar)], label='检查 結構 構造')
        self.assertTrue(pump(job.finished, 60))
        self.assertEqual(job.state, 'finished', job.lines)
        text = '\n'.join(t for _, t in job.lines)
        self.assertIn('硅 Si 晶体.POSCAR', text)
        self.assertIn('label=检查 結構 構造', text)
        self.assertNotIn('\ufffd', text)
        self.assertIn('检查 結構 構造', self.window.console.visible_text())

    def test_bad_input_shows_banner_not_crash(self):
        w = self.window
        form = w.forms['scf']
        self.assertIsNone(form.submit())
        self.assertTrue(w.banner.isVisible())
        self.assertEqual(w.banner.property('level'), 'error')
        self.assertEqual(w.jobs.jobs, [])
        structure = w.show_page('structure')
        structure.path.setText(str(self.tmp / 'missing.POSCAR'))
        structure.inspect()
        self.assertTrue(pump(lambda: structure.inspect_button.isEnabled(), 10))
        self.assertEqual(w.banner.property('level'), 'error')

    def test_structure_page_inspects_file(self):
        structure = self.window.show_page('structure')
        structure.load(SI)
        self.assertTrue(pump(lambda: structure.current is not None, 10))
        self.assertEqual(structure.atoms_table.rowCount(), 2)
        structure.use_in_forms()
        self.assertEqual(self.window.forms['scf'].fields['structure'].value(), str(SI))

    def _classify(self, path):
        page = self.window.show_page('structure')
        page.load(path)
        self.assertTrue(pump(lambda: page.preset_card.state in ('ready', 'failed') and
                             page.recommendation is not None and page.recommendation['path'] == str(path), 30),
                        page.preset_card.error)
        return page

    def test_structure_classification_applies_recommended_options(self):
        from sparse_dft.cli import build_parser
        page = self._classify(SI)
        self.assertEqual(page.recommendation['profile']['category'], 'semiconductor')
        self.assertIn('Fd-3m', page.preset_card.details.text())
        self.assertGreater(page.preset_card.table.rowCount(), 5)
        self.assertFalse(page.preset_card.hubbard.isEnabled())  # no +U for Si
        scf, bands = self.window.forms['scf'], self.window.forms['bands']
        # A field the user typed into is kept; everything else follows the recommendation.
        scf.fields['smearing_ev'].editor.setText('0.2')
        self.assertTrue(scf.fields['smearing_ev'].user_edited)
        preview = {(r['command'], r['dest']): r['action'] for r in page.recommended_rows()}
        self.assertEqual(preview[('scf', 'smearing_ev')], 'kept')
        self.assertEqual(preview[('scf', 'scf_grid')], 'set')
        self.assertEqual(preview[('scf', 'mesh_mode')], 'same')  # gamma is already the default
        rows = page.apply_recommended(confirm=False)
        self.assertTrue(any(r['action'] == 'kept' for r in rows))
        self.assertEqual(scf.fields['smearing_ev'].value(), '0.2')
        self.assertEqual(scf.fields['scf_grid'].value(), ['8', '8', '8'])
        self.assertEqual(scf.fields['structure'].value(), str(SI))
        self.assertFalse(scf.fields['scf_grid'].user_edited)  # recommended, not typed
        path_file = Path(bands.fields['path'].value())
        self.assertTrue(path_file.is_file())
        self.assertEqual(path_file.parent, self.tmp / 'runs 输出' / 'band_paths')
        self.assertGreater(len(json.loads(path_file.read_text(encoding='utf-8'))['kpoints']), 20)
        for form in (scf, bands, self.window.forms['relax']):
            form.set_values({'paw_root': str(self.tmp)})
            ns = build_parser().parse_args(form.argv())
            self.assertEqual(ns.scf_grid, [8, 8, 8])
            self.assertEqual(ns.spin_channels, 1)
        self.assertAlmostEqual(build_parser().parse_args(scf.argv()).smearing_ev, 0.2)
        # Forced: the edited field follows the recommendation too.
        page.apply_recommended(confirm=False, force=True)
        self.assertEqual(float(scf.fields['smearing_ev'].value()), 0.0)

    def test_recommendation_clears_values_of_the_previous_structure(self):
        sys.path.insert(0, str(REPO / 'tests'))
        from preset_fixtures import CASES
        from sparse_dft.structure import write_poscar
        fe = self.tmp / 'Fe.POSCAR'
        write_poscar(CASES['Fe'][0](), fe)
        page = self._classify(fe)
        self.assertEqual(page.recommendation['profile']['category'], 'ferromagnetic_metal')
        page.apply_recommended(confirm=False)
        scf = self.window.forms['scf']
        self.assertEqual(scf.fields['spin_channels'].value(), '2')
        self.assertNotEqual(scf.fields['initial_moments'].value(), '')
        page = self._classify(SI)
        page.apply_recommended(confirm=False)
        self.assertEqual(scf.fields['spin_channels'].value(), '1')
        self.assertEqual(scf.fields['initial_moments'].value(), '')  # Fe seed removed
        scf.set_values({'paw_root': 'p'})
        self.assertNotIn('--initial-moments', scf.argv())

    def test_preview_dialog_and_auto_apply(self):
        from sparse_dft_gui.recommend import PreviewDialog
        page = self._classify(SI)
        dialog = PreviewDialog(page.recommended_rows, page)
        self.assertIn('fields will change', dialog.summary.text())
        shown = dialog.table.rowCount()
        dialog.show_same.setChecked(True)
        self.assertGreater(dialog.table.rowCount(), shown)
        dialog.reject()
        self.assertEqual(self.window.forms['scf'].fields['scf_grid'].value(), ['', '', ''])  # nothing applied
        page.preset_card.auto_apply.setChecked(True)
        self.assertEqual(self.prefs.get('auto_apply_presets'), 1)
        self.window.forms['scf'].reset()
        page = self._classify(REPO / 'examples' / 'Si.cif')
        self.assertTrue(pump(lambda: self.window.forms['scf'].fields['scf_grid'].value() == ['8', '8', '8'], 10))

    def test_load_argv_marks_fields_user_edited(self):
        scf = self.window.forms['scf']
        scf.load_argv(['scf', str(SI), '--paw-root', 'p', '--smearing-ev', '0.3'])
        self.assertTrue(scf.fields['smearing_ev'].user_edited)
        scf.reset()
        self.assertFalse(scf.fields['smearing_ev'].user_edited)

    def test_sparsity_presets_fill_forms(self):
        page = self.window.show_page('sparsity')
        page.apply_preset('search')
        scf = self.window.forms['scf']
        scf.set_values({'structure': 's', 'paw_root': 'p'})
        argv = scf.argv()
        self.assertEqual(argv[argv.index('--sparsity') + 1], 'custom')
        self.assertEqual(argv[argv.index('--paw-compensation') + 1], 'blocks')
        page.apply_preset('auto')
        bands = self.window.forms['bands']
        self.assertEqual(scf.fields['sparsity'].value(), 'auto')
        self.assertEqual(bands.fields['sparsity'].value(), 'off')  # NSCF refuses auto
        page.apply_preset('off')
        self.assertNotIn('--sparsity', scf.argv())

    def test_sparsity_level_rules_measured_and_probe_preview(self):
        page = self.window.show_page('sparsity')
        scf, bands = self.window.forms['scf'], self.window.forms['bands']
        scf.set_values({'structure': 's', 'paw_root': 'p'})
        page.apply_preset('auto', level='conservative')
        argv = scf.argv()
        self.assertEqual(argv[argv.index('--sparsity-level') + 1], 'conservative')
        self.assertEqual(bands.fields['sparsity_level'].value(), '')  # NSCF: off, no level
        page.apply_preset('off')
        self.assertNotIn('--sparsity-level', scf.argv())
        self.assertGreaterEqual(page.rule_table.rowCount(), 8)        # S1 + stricter + S2-S7 rules
        self.assertEqual(page.rule_table.item(0, 1).text(), 'S1.weighted_l2.v1')
        self.assertEqual(page.measured_table.rowCount(), 8)          # v15 sweep systems
        page.preview()                                               # structure 's' does not exist
        deadline = time.time() + 30
        while page.preview_button.isEnabled() is False and time.time() < deadline:
            QApplication.processEvents(); time.sleep(0.02)
        self.assertIn('Preview failed', page.preview_status.text())
        paw = ROOT.parent / 'PAW Raw Data'
        if not paw.exists():
            return
        scf.set_values({'structure': str(ROOT / 'examples' / 'Si.POSCAR'), 'paw_root': str(paw)})
        page.preview()
        deadline = time.time() + 120
        while page.preview_result is None and time.time() < deadline:
            QApplication.processEvents(); time.sleep(0.05)
        self.assertIsNotNone(page.preview_result, page.preview_status.text())
        self.assertEqual(page.preview_table.rowCount(), 3)
        self.assertEqual(page.preview_table.item(1, 0).text(), 'balanced')

    def test_paw_convert_action_converts_and_imports(self):
        from unittest import mock
        from PyQt6.QtWidgets import QMessageBox
        try:
            from test_paw_convert import make_set
        except ImportError:                                  # run as a package
            from tests.test_paw_convert import make_set
        page = self.window.show_page('paw')
        self.assertIn('paw-convert', page.forms)
        from sparse_dft_gui.i18n import translator
        self.assertEqual(translator().command_title('paw-convert'), 'Convert POTCAR set')
        src = make_set(self.tmp / 'POTCAR 源')
        out = self.tmp / 'converted 归档'
        library = self.tmp / 'user data'
        with mock.patch.dict(os.environ, {'SDFT_USER_DATA_DIR': str(library)}), \
                mock.patch('sparse_dft_gui.pages.commands.QMessageBox.question',
                           return_value=QMessageBox.StandardButton.Yes):
            page.convert_dialog(source=str(src), output=str(out))
            self.assertTrue(pump(lambda: (library / 'paw' / out.name).is_dir(), 30))
        self.assertTrue(page.convert_button.isEnabled())
        self.assertTrue((out / 'manifest.json').is_file())
        self.assertTrue((library / 'paw' / out.name / 'index.json').is_file())
        # a second conversion into the same (now non-empty) folder is refused, never overwritten
        before = sorted(p.name for p in out.iterdir())
        page.convert_dialog(source=str(src), output=str(out))
        self.assertTrue(pump(lambda: page.convert_button.isEnabled(), 30))
        self.assertEqual(sorted(p.name for p in out.iterdir()), before)

    def test_console_filter_search_and_utf8_export(self):
        console = self.window.console
        console.clear()
        console.append('j1', 'stdout', 'iteration 1 收敛中')
        console.append('j1', 'stderr', 'Traceback: RuntimeError 错误')
        console.append('j2', 'stdout', 'warning: 警告 text')
        self.assertEqual(len(console.visible_text().splitlines()), 3)
        console.level.setCurrentIndex(console.level.findData('error'))
        self.assertEqual(console.visible_text(), 'Traceback: RuntimeError 错误')
        console.level.setCurrentIndex(0)
        console.search.setText('警告')
        self.assertEqual(console.visible_text(), 'warning: 警告 text')
        console.search.clear()
        console.set_job_filter('j1')
        self.assertEqual(len(console.visible_text().splitlines()), 2)
        path = console.export(self.tmp / '日志 log.txt')
        self.assertIn('收敛中', path.read_text(encoding='utf-8'))
        console.set_job_filter(None)


def _plotting_available():
    try:
        from sparse_dft_gui import plots
        return HAVE_QT and plots.available()
    except Exception:  # noqa: BLE001
        return False


@unittest.skipUnless(HAVE_QT, 'PyQt6 is not installed')
class PlotPreviewTests(WindowTests):
    def test_analysis_page_without_plotting_is_graceful(self):
        from sparse_dft_gui import plots
        page = self.window.show_page('analysis')
        page.source.setText(str(self.tmp))
        ok = page.refresh()
        if not plots.available():
            self.assertFalse(ok)
            self.assertIn('sparse_dft.plotting', page.status.text())

    @unittest.skipUnless(_plotting_available(), 'sparse_dft.plotting is not importable')
    def test_preview_real_results(self):
        sources = [p for p in SAMPLE_RESULTS if p.is_dir()]
        if not sources:
            self.skipTest('no sample result directories')
        page = self.window.show_page('analysis')
        for source in sources[:2]:
            with self.subTest(source=str(source)):
                self.assertTrue(page.show_result(source), page.status.text())
                self.assertTrue(pump(lambda: page.figure is not None, 60), page.placeholder.text())
                self.assertIsNotNone(page.canvas)
                exported = page.export(str(self.tmp / 'fig 图' / source.parent.name), ['png', 'svg'], dpi=150)
                self.assertTrue(exported and all(Path(p).exists() for p in exported), exported)

    def test_preview_keeps_figure_aspect_at_every_size(self):
        from sparse_dft.plotting.style import figure_style, new_figure
        from sparse_dft_gui.preview import FigurePreview, fit_rect
        with figure_style('double'):
            fig, axes = new_figure('double', height_mm=60.0)
        axes[0, 0].plot([0, 1], [0, 1])
        aspect = fig.get_figwidth() / fig.get_figheight()
        widget = FigurePreview()
        widget.set_figures([fig])
        for w, h in ((800, 600), (1280, 800), (1920, 1080), (1200, 1600), (2560, 1440)):
            for dpr in (1.0, 2.0):
                with self.subTest(size=(w, h), dpr=dpr):
                    x, y, rw, rh = fit_rect(w, h, aspect)
                    self.assertLessEqual(rw, w + 1e-9); self.assertLessEqual(rh, h + 1e-9)
                    self.assertAlmostEqual(rw / rh, aspect, delta=0.01 * aspect)
                    widget.resize(w // 4, h // 4)
                    image = widget.render_now()
                    self.assertAlmostEqual(image.width() / image.height(), aspect, delta=0.01 * aspect)
                    target = widget.target_rect()
                    self.assertAlmostEqual(target.width() / target.height(), aspect, delta=0.01 * aspect)
        widget.deleteLater()

    @unittest.skipUnless(_plotting_available(), 'sparse_dft.plotting is not importable')
    def test_atlas_split_and_combined_preview_and_export(self):
        source = REPO / 'tests' / 'fixtures' / 'plotting' / 'synthetic_bands'
        if not source.is_dir():
            self.skipTest('no synthetic band fixture')
        page = self.window.show_page('analysis')
        self.assertTrue(page.show_result(source), page.status.text())
        row = next(i for i in range(page.thumbs.count())
                   if page.thumbs.item(i).data(Qt.ItemDataRole.UserRole) == 'electronic_atlas')
        page.thumbs.setCurrentRow(row)
        self.assertTrue(pump(lambda: page.figures and page.figure is not None, 60), page.placeholder.text())
        panels = page.panel_names()
        self.assertIn('bands', panels)
        fig = page.figure
        self.assertAlmostEqual(page.canvas.target_rect().width() / page.canvas.target_rect().height(),
                               fig.get_figwidth() / fig.get_figheight(), delta=0.01 * fig.get_figwidth()
                               / fig.get_figheight())
        page.set_view('split')
        self.assertTrue(page.next_button.isVisible())
        first = page.figure
        page.step_panel(1)
        if len(panels) > 1:
            self.assertIsNot(page.figure, first)
        out = self.tmp / 'atlas 图'
        written = page.export(str(out / 'atlas'), ['png'], dpi=100, panels='both')
        names = sorted(Path(p).name for p in written if str(p).endswith('.png'))
        self.assertEqual(names, sorted(['atlas.png'] + [f'atlas_{n}.png' for n in panels]))
        single = page.export(str(out / 'one'), ['svg'], dpi=100, panels='current')
        self.assertTrue(any(Path(p).name.startswith('one_') and p.suffix == '.svg' for p in map(Path, single)))
        page.set_view('combined')
        self.assertFalse(page.next_button.isVisible())

    # The inherited window tests are not repeated here.
    for _name in [n for n in dir(WindowTests) if n.startswith('test_')]:
        locals()[_name] = None
    del _name


if __name__ == '__main__':
    unittest.main()
