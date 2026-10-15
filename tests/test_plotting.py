"""Structural and numeric tests of sparse_dft.plotting (figures, export, CLI)."""
import contextlib
import io
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

import matplotlib
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

from sparse_dft.analysis.common import sha256_file
from sparse_dft.errors import InputError
from sparse_dft.plotting import (KINDS, PALETTES, WIDTHS_MM, available_kinds, export, figure_style, make_figure,
                           new_figure, panel_labels)
from sparse_dft.plotting.qa import layout_report
from sparse_dft.plotting.style import CJK_FONTS, font_stack, pick_font

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / 'tests' / 'fixtures'
PF = FIX / 'plotting'
NI = PF / 'ni_spin_scf'
SWEEP = Path('/tmp/sparse_dft_sweep')
SPEC_KINDS = {'scf_convergence', 'magnetization', 'energy_terms', 'forces', 'stress', 'timing', 'bands', 'bands_dos',
              'dos', 'pdos', 'fatbands', 'bands_compare', 'relax', 'sparsity_pareto', 'scaling'}


def rc_snapshot():
    rc = matplotlib.rcParams
    return {k: rc._get(k) for k in rc.keys() if k != 'backend'}


def texts_of(fig):
    out = [t.get_text() for t in fig.texts]
    for ax in fig.axes:
        out += [t.get_text() for t in ax.texts] + [ax.get_xlabel(), ax.get_ylabel()]
        out += [t.get_text() for t in ax.get_xticklabels() + ax.get_yticklabels()]
        if ax.get_legend():
            out += [t.get_text() for t in ax.get_legend().get_texts()]
    return out


def run_cli(argv):
    from sparse_dft.plotting.cli import main
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


class StyleContractTests(unittest.TestCase):
    def test_widths_palettes_and_kinds(self):
        self.assertEqual(WIDTHS_MM, {'single': 89.0, 'onehalf': 120.0, 'double': 183.0})
        self.assertIn('okabe_ito', PALETTES)
        self.assertIn('macaron', PALETTES)
        self.assertEqual(PALETTES['okabe_ito'][:3], ['#0072B2', '#D55E00', '#009E73'])
        self.assertTrue(SPEC_KINDS.issubset(KINDS))
        for kind, entry in KINDS.items():
            self.assertTrue({'title', 'inputs', 'description'} <= set(entry), kind)

    def test_figure_style_is_scoped(self):
        before = rc_snapshot()
        with figure_style('double', lang='ja', palette='macaron', base_fontsize=6.5) as spec:
            self.assertEqual(matplotlib.rcParams['pdf.fonttype'], 42)
            self.assertEqual(matplotlib.rcParams['ps.fonttype'], 42)
            self.assertEqual(matplotlib.rcParams['svg.fonttype'], 'none')
            self.assertEqual(matplotlib.rcParams['font.size'], 6.5)
            self.assertFalse(matplotlib.rcParams['axes.spines.top'])
            self.assertEqual(matplotlib.rcParams['xtick.direction'], 'out')
            self.assertEqual(spec.width_mm, 183.0)
        self.assertEqual(rc_snapshot(), before)

    def test_new_figure_is_plain_figure_with_exact_width(self):
        for width, mm in WIDTHS_MM.items():
            fig, axes = new_figure(width, height_mm=50, nrows=1, ncols=2)
            self.assertIsInstance(fig, Figure)
            self.assertEqual(axes.shape, (1, 2))
            self.assertAlmostEqual(fig.get_figwidth() * 25.4, mm, places=6)
            self.assertAlmostEqual(fig.get_figheight() * 25.4, 50.0, places=6)
        fig, axes = new_figure('single', squeeze=True)
        self.assertEqual(axes.shape, (1,))

    def test_panel_labels(self):
        fig, axes = new_figure('double', height_mm=50, nrows=1, ncols=3)
        panel_labels(axes)
        labels = [t for ax in fig.axes for t in ax.texts]
        self.assertEqual([t.get_text() for t in labels], ['a', 'b', 'c'])
        self.assertTrue(all(t.get_fontweight() == 'bold' for t in labels))
        panel_labels(axes[0, :1], start='d')
        self.assertEqual(fig.axes[0].texts[-1].get_text(), 'd')

    def test_font_stacks(self):
        self.assertEqual(font_stack('en')[0], pick_font(('Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans')))
        for lang in ('zh_CN', 'zh_TW', 'ja'):
            stack = font_stack(lang)
            cjk = pick_font(CJK_FONTS[lang])
            if cjk:
                self.assertEqual(stack[0], cjk)
            self.assertIn('DejaVu Sans', stack)
        with self.assertRaises(ValueError):
            with figure_style(lang='fr'):
                pass

    def test_plotting_never_uses_pyplot_or_global_backend(self):
        for path in (ROOT / 'sparse_dft' / 'plotting').glob('*.py'):
            source = path.read_text(encoding='utf-8')
            self.assertNotRegex(source, r'import matplotlib\.pyplot|from matplotlib import pyplot', path.name)
            self.assertNotRegex(source, r'matplotlib\.use\(', path.name)
            self.assertNotRegex(source, r'rcParams\.update|rcParams\[[^\]]+\]\s*=', path.name)
        report = (ROOT / 'sparse_dft' / 'analysis' / 'report.py').read_text(encoding='utf-8')
        self.assertNotIn('matplotlib.use(', report)
        self.assertNotRegex(report, r'import matplotlib\.pyplot|from matplotlib import pyplot')
        self.assertNotRegex(report, r'rcParams\.update')


class ExportTests(unittest.TestCase):
    def test_all_formats_and_provenance(self):
        before = rc_snapshot()
        pyplot_figs = self._pyplot_fignums()
        fig = make_figure('bands', PF / 'synthetic_bands')
        with tempfile.TemporaryDirectory() as td:
            files = export(fig, Path(td) / 'nested' / 'dir' / 'bands', formats='pdf,svg,png,tif,eps', dpi=150)
            names = [f.name for f in files]
            self.assertEqual(names, ['bands.pdf', 'bands.svg', 'bands.png', 'bands.tiff', 'bands.eps',
                                     'bands.provenance.json'])
            self.assertTrue(all(f.stat().st_size > 500 for f in files))
            prov = json.loads(files[-1].read_text(encoding='utf-8'))
            self.assertEqual(prov['kind'], 'bands')
            self.assertEqual(prov['software']['matplotlib'], matplotlib.__version__)
            self.assertIn('created_utc', prov)
            src = PF / 'synthetic_bands' / 'bands.json'
            self.assertIn(sha256_file(src), [i['sha256'] for i in prov['inputs']])
            self.assertEqual(prov['style']['pdf.fonttype'], 42)
            self.assertEqual(prov['style']['width_mm'], 89.0)
            pdf = files[0].read_bytes().decode('latin1')
            self.assertNotIn('/Type3', pdf)
            self.assertIn('/CIDFontType2', pdf)
            svg = files[1].read_text(encoding='utf-8')
            self.assertIn('<text', svg)                     # svg.fonttype='none': editable text
            self.assertIn('/FontType 42', files[4].read_bytes().decode('latin1'))
        self.assertEqual(rc_snapshot(), before)
        self.assertEqual(self._pyplot_fignums(), pyplot_figs)

    def test_export_plain_figure_without_canvas(self):
        fig = Figure(figsize=(2, 1))
        fig.add_subplot().plot([0, 1], [0, 1])
        with tempfile.TemporaryDirectory() as td:
            files = export(fig, Path(td) / 'plain.png', formats=['png'], provenance={'kind': 'custom', 'inputs': []})
            self.assertEqual([f.name for f in files], ['plain.png', 'plain.provenance.json'])
            self.assertEqual(json.loads(files[1].read_text(encoding='utf-8'))['kind'], 'custom')
        with self.assertRaises(ValueError):
            export(fig, 'x', formats=['bmp'])

    @staticmethod
    def _pyplot_fignums():
        plt = sys.modules.get('matplotlib.pyplot')
        return plt.get_fignums() if plt else None


class FigureKindTests(unittest.TestCase):
    def setUp(self):
        self.rc_before = rc_snapshot()

    def tearDown(self):
        self.assertEqual(rc_snapshot(), self.rc_before, 'plotting changed global rcParams')

    def assertClean(self, fig):
        issues = [i for i in layout_report(fig)['issues']]
        self.assertEqual(issues, [], issues)

    def test_scf_convergence(self):
        fig = make_figure('scf_convergence', NI)
        self.assertEqual(len(fig.axes), 2)                    # eigensolver in its own panel, no twin axis
        top, bottom = fig.axes
        self.assertEqual(top.get_yscale(), 'log')
        self.assertEqual(top.get_ylabel(), 'Residual')
        self.assertEqual(bottom.get_xlabel(), 'SCF iteration')
        self.assertEqual(bottom.get_ylabel(), 'Eigen residual (Ha)')
        tol_lines = [l for l in top.lines if len(set(l.get_ydata())) == 1]
        self.assertEqual(sorted({l.get_ydata()[0] for l in tol_lines}), [1e-08])
        self.assertIsNotNone(top.get_legend())
        self.assertEqual(fig.sparse_dft_meta['values']['iterations'], 23)
        self.assertClean(fig)

    def test_magnetization_requires_spin(self):
        fig = make_figure('magnetization', NI)
        self.assertIn('muB', json.dumps(fig.sparse_dft_meta['values']))
        self.assertAlmostEqual(fig.sparse_dft_meta['values']['final_output_moment_muB'], 0.5829974356904134, places=12)
        self.assertClean(fig)
        if (SWEEP / 'small' / 'Si2' / 'reference').is_dir():
            with self.assertRaises(InputError):
                make_figure('magnetization', SWEEP / 'small' / 'Si2' / 'reference')

    def test_energy_terms_waterfall_and_diagnostic(self):
        fig = make_figure('energy_terms', NI)
        meta = fig.sparse_dft_meta
        self.assertEqual(meta['parameters']['mode'], 'waterfall')
        self.assertTrue(meta['values']['additive'])
        ax = fig.axes[0]
        labels = [t.get_text() for t in ax.get_yticklabels()]
        self.assertEqual(labels[-1], 'Total')
        self.assertEqual(len(labels), 13)                     # 12 ledger terms + total
        self.assertTrue(any('diagnostic' in t for t in texts_of(fig)))
        self.assertClean(fig)

    def test_energy_terms_non_additive_falls_back_to_bars(self):
        with tempfile.TemporaryDirectory() as td:
            data = json.loads((NI / 'scf_result.json').read_text(encoding='utf-8'))
            data['xc_energy_ev'] += 1.0
            (Path(td) / 'scf_result.json').write_text(json.dumps(data), encoding='utf-8')
            fig = make_figure('energy_terms', td)
            self.assertEqual(fig.sparse_dft_meta['parameters']['mode'], 'bars')
            self.assertTrue(any('do not sum' in t for t in texts_of(fig)))
            with self.assertRaises(InputError):
                make_figure('energy_terms', td, mode='waterfall')

    def test_forces_and_stress(self):
        fig = make_figure('forces', NI)
        self.assertEqual(len(fig.axes), 3)                    # bars, heatmap, colorbar
        self.assertIn('AA', fig.axes[0].get_ylabel())
        self.assertClean(fig)
        fig = make_figure('stress', NI)
        image = fig.axes[0].images[0]
        lo, hi = image.get_clim()
        self.assertAlmostEqual(lo, -hi, places=12)            # diverging, centred at zero
        raw = json.loads((NI / 'scf_result.json').read_text(encoding='utf-8'))['hf_total_stress_ev_per_angstrom3']
        self.assertAlmostEqual(image.get_array()[0][0], raw[0][0] * 160.21766208, places=10)
        self.assertIn('GPa', fig.axes[1].get_ylabel())
        self.assertClean(fig)

    def test_timing_is_labelled_non_additive(self):
        fig = make_figure('timing', NI, top=10)
        ax = fig.axes[0]
        self.assertEqual(len(ax.patches), 10)
        self.assertTrue(any('not additive' in t for t in texts_of(fig)))
        self.assertTrue(any('calls' in t for t in texts_of(fig)))
        self.assertClean(fig)

    def test_bands_broken_path_gamma_and_gap(self):
        fig = make_figure('bands', PF / 'synthetic_bands')
        ax = fig.axes[0]
        self.assertEqual(ax.get_ylabel(), r'$E - E_\mathrm{F}$ (eV)')
        self.assertEqual([t.get_text() for t in ax.get_xticklabels()],
                         [r'$\Gamma$', r'$\mathrm{X}|\mathrm{K}$', r'$\Gamma$'])
        lc = [c for c in ax.collections if isinstance(c, LineCollection)][0]
        # 4 bands x 2 segments; no segment crosses the X|K jump.
        self.assertEqual(len(lc.get_segments()), 8)
        jump_x = fig.sparse_dft_meta['values']['ticks'][1][0]
        for seg in lc.get_segments():
            xs = seg[:, 0]
            self.assertTrue(xs.max() <= jump_x + 1e-12 or xs.min() >= jump_x - 1e-12)
        values = fig.sparse_dft_meta['values']
        self.assertAlmostEqual(values['gap_ev'], 1.1, places=12)
        self.assertFalse(values['gap_direct'])
        self.assertTrue(any('indirect' in t for t in texts_of(fig)))
        self.assertClean(fig)

    def test_bands_spin_line_styles(self):
        fig = make_figure('bands', FIX / 'sparse_dft' / 'si_gamma_bands.json')
        self.assertEqual(len([c for c in fig.axes[0].collections if isinstance(c, LineCollection)]), 1)
        with tempfile.TemporaryDirectory() as td:
            payload = json.loads((PF / 'synthetic_bands' / 'bands.json').read_text(encoding='utf-8'))
            payload['spin_channels'] = {'up': payload['eigenvalues'],
                                        'down': [[e + 0.3 for e in row] for row in payload['eigenvalues']]}
            payload.pop('occupations')
            (Path(td) / 'bands.json').write_text(json.dumps(payload), encoding='utf-8')
            fig = make_figure('bands', td)
            colls = [c for c in fig.axes[0].collections if isinstance(c, LineCollection)]
            self.assertEqual(len(colls), 2)
            self.assertNotEqual(colls[0].get_linestyle(), colls[1].get_linestyle())
            self.assertEqual([t.get_text() for t in fig.axes[0].get_legend().get_texts()], ['spin up', 'spin down'])

    def test_missing_fermi_raises_with_cli_hint(self):
        with tempfile.TemporaryDirectory() as td:
            payload = json.loads((PF / 'synthetic_bands' / 'bands.json').read_text(encoding='utf-8'))
            payload.pop('fermi_energy_ev')
            (Path(td) / 'bands.json').write_text(json.dumps(payload), encoding='utf-8')
            with self.assertRaisesRegex(InputError, r'--fermi'):
                make_figure('bands', td)
            fig = make_figure('bands', td, fermi=0.2)
            self.assertEqual(fig.sparse_dft_meta['values']['fermi_origin'], 'option')
            fig = make_figure('bands', td, align='vbm')
            self.assertEqual(fig.axes[0].get_ylabel(), r'$E - E_\mathrm{VBM}$ (eV)')

    def test_spin_dos_both_channels_mirrored(self):
        fig = make_figure('dos', NI, sigma=0.1)
        ax = fig.axes[0]
        curves = [l for l in ax.lines if len(l.get_xdata()) > 100]
        self.assertEqual(len(curves), 2)
        self.assertGreater(max(curves[0].get_ydata()), 0)
        self.assertLess(min(curves[1].get_ydata()), 0)          # spin down mirrored
        self.assertEqual([t.get_text() for t in ax.get_legend().get_texts()], ['spin up', 'spin down'])
        self.assertAlmostEqual(fig.sparse_dft_meta['values']['fermi_energy_ev'], 9.36348724614469, places=12)
        self.assertClean(fig)

    def test_pdos_split_per_element_and_l(self):
        fig = make_figure('pdos', PF / 'synthetic_spin_dos')
        axes = fig.axes
        self.assertEqual(len(axes), 2)                          # Mg, O
        for ax in axes:
            curves = [l for l in ax.lines if len(l.get_xdata()) > 50]
            self.assertEqual(len(curves), 6)                    # s, p, d x up/down
            self.assertTrue(any(min(l.get_ydata()) < 0 for l in curves))
        legend = [t.get_text() for t in axes[0].get_legend().get_texts()]
        self.assertEqual(legend, ['s', 'p', 'd'])
        self.assertEqual({t.get_text() for ax in axes for t in ax.texts if t.get_text() in ('Mg', 'O')}, {'Mg', 'O'})
        self.assertClean(fig)

    def test_fatbands_one_collection_per_group(self):
        fig = make_figure('fatbands', PF / 'synthetic_bands')
        ax = fig.axes[0]
        colls = [c for c in ax.collections if isinstance(c, LineCollection)]
        self.assertEqual(len(colls), 4)                         # grey base + s, p, d
        widths = colls[1].get_linewidths()
        self.assertGreater(len(set(round(float(w), 6) for w in widths)), 2)
        self.assertClean(fig)

    def test_bands_dos_shares_energy_axis(self):
        fig = make_figure('bands_dos', [FIX / 'sparse_dft' / 'si_gamma_bands.json', FIX / 'vasp' / 'si2_gamma_100' / 'DOSCAR'],
                          fermi=1.565377064862092)
        self.assertEqual(len(fig.axes), 2)
        self.assertEqual(fig.axes[0].get_ylim(), fig.axes[1].get_ylim())
        self.assertClean(fig)

    def test_bands_compare_panels(self):
        a = PF / 'synthetic_bands'
        with tempfile.TemporaryDirectory() as td:
            payload = json.loads((a / 'bands.json').read_text(encoding='utf-8'))
            payload['eigenvalues'] = [[e + (0.004 if i == 2 else 0.0) for i, e in enumerate(row)] for row in payload['eigenvalues']]
            (Path(td) / 'bands.json').write_text(json.dumps(payload), encoding='utf-8')
            fig = make_figure('bands_compare', [td], reference=a, align='none')
            self.assertEqual(len(fig.axes), 2)
            v = fig.sparse_dft_meta['values']
            self.assertAlmostEqual(v['max_abs_error_ev'], 0.004, places=12)
            self.assertEqual(fig.axes[1].get_xlabel(), r'$\Delta E$ (meV)')
            self.assertClean(fig)

    def test_relax(self):
        fig = make_figure('relax', PF / 'synthetic_relax')
        self.assertEqual(len(fig.axes), 3)                      # energy, max force, step length
        force_ax = fig.axes[1]
        self.assertEqual(force_ax.get_yscale(), 'log')
        self.assertIn(0.02, [l.get_ydata()[0] for l in force_ax.lines if len(set(l.get_ydata())) == 1])
        self.assertEqual(fig.sparse_dft_meta['values']['tolerance_origin'], 'file')
        self.assertClean(fig)

    def test_scaling_math_and_ideal_lines(self):
        fig = make_figure('scaling', PF / 'synthetic_scaling.json')
        builds = fig.sparse_dft_meta['values']['builds']
        v2 = {p['threads']: p for p in builds['v2']}
        # Amdahl rule used by the fixture: T(n) = 100 (0.05 + 0.95/n), user = 100 s.
        self.assertAlmostEqual(v2[8]['speedup'], 1 / (0.05 + 0.95 / 8), places=5)
        self.assertAlmostEqual(v2[8]['efficiency'], 1 / (0.05 + 0.95 / 8) / 8, places=5)
        self.assertAlmostEqual(v2[8]['occupancy'], 100.0 / (100 * (0.05 + 0.95 / 8) * 8), places=5)
        self.assertEqual(len(fig.axes), 4)
        speed_ax = fig.axes[1]
        self.assertTrue(any(l.get_label() == 'ideal' for l in speed_ax.lines))
        self.assertClean(fig)

    def test_scaling_without_baseline_is_explicit(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'scaling.json'
            p.write_text(json.dumps({'schema_version': 'sparse-dft-scaling-1', 'system': 'x', 'runs': [
                {'label': 'a', 'threads': 2, 'ranks': 1, 'wall_seconds': 10.0, 'user_seconds': 19.0, 'sys_seconds': 0.5,
                 'build': 'b'}, {'label': 'b', 'threads': 4, 'ranks': 1, 'wall_seconds': 6.0, 'build': 'b'}]}), encoding='utf-8')
            fig = make_figure('scaling', p)
            pts = fig.sparse_dft_meta['values']['builds']['b']
            self.assertIsNone(pts[0]['speedup'])
            self.assertAlmostEqual(pts[0]['occupancy'], 19.5 / 20.0, places=12)
            self.assertIsNone(pts[1]['occupancy'])
            self.assertTrue(any('speed-up undefined' in t for t in texts_of(fig)))

    def test_languages_and_palettes(self):
        for lang, word in (('zh_CN', '能量'), ('zh_TW', '能量'), ('ja', 'エネルギー')):
            fig = make_figure('bands', PF / 'synthetic_bands', lang=lang, palette='macaron')
            self.assertIn(word, fig.axes[0].get_ylabel())
            if pick_font(CJK_FONTS[lang]):
                issues = [i for i in layout_report(fig)['issues'] if 'glyph' in i]
                self.assertEqual(issues, [], f'{lang}: {issues}')
        fig = make_figure('dos', NI, palette='macaron')
        self.assertEqual(fig.axes[0].lines[0].get_color().lower(), PALETTES['macaron'][0].lower())

    def test_available_kinds(self):
        self.assertEqual(available_kinds(NI), ['scf_convergence', 'magnetization', 'energy_terms', 'forces', 'stress',
                                               'timing', 'dos'])
        # a projected path also supports the atlas (its other panels are omitted with a reason)
        self.assertEqual(available_kinds(PF / 'synthetic_bands'), ['bands', 'fatbands', 'electronic_atlas'])
        self.assertEqual(available_kinds(PF / 'synthetic_spin_dos'), ['dos', 'pdos'])
        self.assertEqual(available_kinds(PF / 'synthetic_relax' / 'relax_result.json'), ['relax'])
        self.assertEqual(available_kinds(PF / 'synthetic_scaling.json'), ['scaling'])
        self.assertEqual(available_kinds(FIX / 'vasp' / 'si2_gamma_100'), ['dos'])     # 1 k-point: not a path
        self.assertIn('bands_dos', available_kinds([PF / 'synthetic_bands', PF / 'synthetic_spin_dos']))
        with self.assertRaises(InputError):
            make_figure('bands', NI)


class CliTests(unittest.TestCase):
    def test_list(self):
        code, out, err = run_cli(['list', str(NI)])
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(out)['kinds'][:2], ['scf_convergence', 'magnetization'])

    def test_specific_kind_and_formats(self):
        with tempfile.TemporaryDirectory() as td:
            code, out, err = run_cli(['bands', str(PF / 'synthetic_bands'), '--out', f'{td}/fig', '--formats', 'pdf,svg',
                                      '--width', 'onehalf', '--lang', 'zh_CN', '--emin', '-3', '--emax', '3'])
            self.assertEqual(code, 0, err)
            result = json.loads(out)
            self.assertEqual([Path(f).name for f in result['files']], ['fig.pdf', 'fig.svg', 'fig.provenance.json'])
            prov = json.loads(Path(td, 'fig.provenance.json').read_text(encoding='utf-8'))
            self.assertEqual(prov['style']['width_mm'], 120.0)
            self.assertEqual(prov['style']['lang'], 'zh_CN')
            self.assertEqual(prov['parameters']['emin'], -3.0)

    def test_auto(self):
        with tempfile.TemporaryDirectory() as td:
            code, out, err = run_cli(['auto', str(NI), str(PF / 'synthetic_bands'), '--out', td, '--formats', 'png',
                                      '--dpi', '100'])
            self.assertEqual(code, 0, err)
            result = json.loads(out)
            kinds = [f['kind'] for f in result['figures']]
            self.assertEqual(len(kinds), len(set(kinds)))
            for kind in ('scf_convergence', 'magnetization', 'bands', 'fatbands', 'dos', 'bands_dos'):
                self.assertIn(kind, kinds)
                self.assertTrue(Path(td, f'{kind}.png').is_file())
                self.assertTrue(Path(td, f'{kind}.provenance.json').is_file())

    def test_errors_are_json_exit_2(self):
        code, out, err = run_cli(['bands', str(NI)])
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(err)['code'], 'INPUT_ERROR')
        code, out, err = run_cli(['nonsense', str(NI)])
        self.assertEqual(code, 2)
        code, out, err = run_cli(['bands', str(PF / 'nope')])
        self.assertEqual(code, 2)
        self.assertIn('not found', json.loads(err)['message'])

    def test_sparse_dft_cli_dispatch(self):
        from sparse_dft.cli import main
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(['plot', 'list', str(PF / 'synthetic_relax')])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out.getvalue())['kinds'], ['relax'])


class LegacyAnalysisTests(unittest.TestCase):
    def test_analyze_bands_report_keys_unchanged(self):
        from sparse_dft.analysis import _normalize_projections, analyze_bands, render_scientific_plots  # noqa: F401
        with tempfile.TemporaryDirectory() as td:
            report = analyze_bands(FIX / 'sparse_dft' / 'si_gamma_bands.json', structure=ROOT / 'examples' / 'Si.POSCAR',
                                   dos_path=FIX / 'vasp' / 'si2_gamma_100' / 'DOSCAR', output_dir=td)
            self.assertEqual(set(report), {'schema_version', 'formula', 'elements', 'bands', 'dos', 'projections',
                                           'provenance', 'warnings', 'plots'})
            self.assertEqual(report['provenance']['parser'], 'sparse_dft.analysis/1')
            self.assertEqual(report['plots']['renderer'], 'matplotlib')
            for name in ('sparse_dft_bands.png', 'sparse_dft_bands.pdf', 'sparse_dft_bands.svg', 'sparse_dft_bands_local.pdf',
                         'sparse_dft_bands.provenance.json'):
                self.assertTrue(Path(td, name).is_file(), name)

    def test_analyze_bands_without_fermi_explains_option(self):
        with tempfile.TemporaryDirectory() as td:
            payload = json.loads((PF / 'synthetic_bands' / 'bands.json').read_text(encoding='utf-8'))
            payload.pop('fermi_energy_ev')
            (Path(td) / 'bands.json').write_text(json.dumps(payload), encoding='utf-8')
            from sparse_dft.analysis import analyze_bands
            with self.assertRaisesRegex(InputError, '--fermi'):
                analyze_bands(td, output_dir=Path(td) / 'out')
            report = analyze_bands(td, output_dir=Path(td) / 'out2', fermi_energy=0.2)
            self.assertTrue(report['plots']['artifacts'])
            report = analyze_bands(td, render=False)            # numeric report needs no Fermi energy
            self.assertIsNone(report['bands']['fermi_energy_ev'])


@unittest.skipUnless((SWEEP / 'report_small.json').is_file(), 'sparsity sweep data not present')
class RealSweepDataTests(unittest.TestCase):
    def test_pareto_from_report(self):
        fig = make_figure('sparsity_pareto', SWEEP / 'report_small.json')
        ax = fig.axes[0]
        self.assertEqual(ax.get_xscale(), 'log')
        self.assertIn(1.0, [l.get_ydata()[0] for l in ax.lines if len(set(l.get_ydata())) == 1])
        pts = fig.sparse_dft_meta['values']['points']
        self.assertTrue(any(not p['pass'] for p in pts))
        self.assertTrue(all((p['gate_use'] <= 1.0) == p['pass'] for p in pts))
        self.assertEqual(layout_report(fig)['issues'], [])

    def test_all_scf_kinds_on_real_results(self):
        for system in ('Si2', 'MgO', 'Al_slab', 'Ni'):
            path = SWEEP / 'small' / system / 'reference'
            if not path.is_dir():
                continue
            for kind in available_kinds(path):
                fig = make_figure(kind, path)
                self.assertEqual(layout_report(fig)['issues'], [], f'{system} {kind}')
