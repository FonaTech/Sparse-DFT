"""Band toolkit figures (sparse_dft.plotting.band_toolkit) on synthetic lm data with exact expectations."""
import json
import math
import tempfile
import unittest
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib.collections import LineCollection, PathCollection  # noqa: E402

from sparse_dft.analysis.electronic import band_edges, load_band_data  # noqa: E402
from sparse_dft.errors import InputError  # noqa: E402
from sparse_dft.plotting import available_kinds, make_figure  # noqa: E402
from sparse_dft.plotting.band_toolkit import (FAT_MODES, band_suite, element_colors, orbital_style,  # noqa: E402
                                        suite_plan)
from sparse_dft.plotting.qa import layout_report  # noqa: E402
from sparse_dft.plotting.style import make_style  # noqa: E402

LM = ['s', 'py', 'pz', 'px', 'dxy', 'dyz', 'dz2', 'dxz', 'dx2-y2']
# fcc reciprocal basis (2 pi included) for a = 5.65 A
_B = [[2 * math.pi / 5.65 * x for x in row] for row in ([-1, 1, 1], [1, -1, 1], [1, 1, -1])]


def _path():
    gamma, x, l = (0, 0, 0), (0.5, 0, 0.5), (0.5, 0.5, 0.5)
    pts = [tuple(g + (b - g) * t / 9 for g, b in zip(gamma, x)) for t in range(10)]
    pts += [tuple(l[i] + (gamma[i] - l[i]) * t / 9 for i in range(3)) for t in range(10)]   # X|L break
    labels = ['GAMMA'] + [None] * 8 + ['X', 'L'] + [None] * 8 + ['GAMMA']
    return pts, labels


def _weights(k, b):
    """[atom][orbital]: band 0 Ga s, 1-2 As p, 3-4 Ga/As mixed p, 5 Ga d (sum < 1)."""
    w = [[0.0] * 9 for _ in range(2)]
    f = 0.5 + 0.4 * math.cos(0.3 * k)
    if b == 0:
        w[0][0] = 0.9
    elif b in (1, 2):
        w[1][1], w[1][2], w[1][3] = 0.3 * f, 0.3, 0.3 * (1 - f)
    elif b in (3, 4):
        w[0][3], w[1][3] = 0.5 * f, 0.5 * (1 - f)
    else:
        w[0][4], w[0][6] = 0.4, 0.2
    return w


def write_bands(directory: Path, *, spin=False):
    pts, labels = _path()
    ev = [[-8 + 0.2 * k % 1, -2 + 0.05 * k, -1.5, 1.0 + 0.04 * k, 2.0, 5.0] for k in range(len(pts))]
    occ = [[2.0, 2.0, 2.0, 0.0, 0.0, 0.0] for _ in pts]
    lm = [[_weights(k, b) for b in range(6)] for k in range(len(pts))]
    payload = {'kpoints': [list(p) for p in pts], 'labels': labels, 'fermi_energy_ev': 0.0,
               'metadata': {'reciprocal_basis': _B, 'reciprocal_basis_includes_2pi': True},
               'projection_atom_symbols': ['Ga', 'As'], 'projection_orbitals': ['s', 'p', 'd'],
               'projector_weights': [1.0] * (len(pts) * 6 * 2 * 3), 'projection_lm_orbitals': LM,
               'projection_lm_weight_definition': 'synthetic sphere weights',
               'projector_lm_weights': [v for k in lm for b in k for a in b for v in a]}
    if spin:
        down = [[e + 0.25 for e in row] for row in ev]
        payload.update(eigenvalues=ev, spin_channels={'up': ev, 'down': down},
                       occupations={'up': [[1, 1, 1, 0, 0, 0]] * len(pts), 'down': [[1, 1, 1, 0, 0, 0]] * len(pts)})
        payload.pop('projector_lm_weights'); payload.pop('projector_weights')
        payload['projections'] = {'projection_orbitals': ['s', 'p', 'd'], 'projection_atom_symbols': ['Ga', 'As'],
                                  'projection_weights': {c: [[[[1, 1, 1]] * 2] * 6] * len(pts) for c in ('up', 'down')},
                                  'projection_lm_orbitals': LM, 'projection_lm_weights': {'up': lm, 'down': lm},
                                  'projection_lm_weight_definition': 'synthetic sphere weights',
                                  'weight_definition': 'legacy'}
    else:
        payload.update(eigenvalues=ev, occupations={'total': occ})
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'bands.json').write_text(json.dumps(payload), encoding='utf-8')
    return directory


def write_dos(directory: Path, *, spin=False):
    energies = [-10 + 0.01 * i for i in range(1601)]

    def g(e0, w):
        return [w * math.exp(-0.5 * ((e - e0) / 0.1) ** 2) / (0.1 * math.sqrt(2 * math.pi)) for e in energies]
    curves_ga = [{'orbital': o, 'spin': s, 'values': g(-8 + i * 0.5, 0.2)}
                 for i, o in enumerate(LM) for s in (('up', 'down') if spin else (None,))]
    curves_as = [{'orbital': o, 'spin': s, 'values': g(-2 + i * 0.3, 0.3)}
                 for i, o in enumerate(LM) for s in (('up', 'down') if spin else (None,))]
    total = [sum(v) for v in zip(*(c['values'] for c in curves_ga + curves_as))]
    payload = {'schema_version': 'sparse-dft-dos-1', 'energies_ev': energies, 'fermi_energy_ev': 0.0,
               'total': ({'total_up': total, 'total_down': total} if spin else {'total': total}),
               'spin_dos': {}, 'pdos': [{'atom': 1, 'element': 'Ga', 'curves': curves_ga},
                                        {'atom': 2, 'element': 'As', 'curves': curves_as}],
               'metadata': {'format': 'Gaussian DOS from a Sparse-DFT NSCF uniform mesh', 'sigma_ev': 0.1,
                            'sampling': 'uniform_mesh', 'nkpoints': 64, 'complete_below_ev': 5.5,
                            'projection_weight_definition': 'synthetic'}}
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'dos.json').write_text(json.dumps(payload), encoding='utf-8')
    return directory


class ColourIdentityTests(unittest.TestCase):
    def test_orbitals_and_elements_have_fixed_colours(self):
        spec = make_style()
        self.assertEqual(orbital_style('p', spec)[0], spec.color(1))
        self.assertEqual(orbital_style('d', spec)[0], spec.color(2))
        shades = {orbital_style(o, spec) for o in ('py', 'pz', 'px')}
        self.assertEqual(len(shades), 3)                        # each m distinct
        self.assertEqual(orbital_style('py', spec)[0], spec.color(1))
        # element colours depend on the element set only, ordered by Z
        a = element_colors(['As', 'Ga'], spec)
        self.assertEqual(a, element_colors(['Ga', 'As', 'Ga'], spec))
        self.assertNotEqual(a['Ga'], a['As'])


class ToolkitFigureTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.bands = write_bands(self.root / 'bands')
        self.spin = write_bands(self.root / 'spin', spin=True)
        self.dos = write_dos(self.root / 'dos')

    def tearDown(self):
        self._tmp.cleanup()

    def assertClean(self, fig):
        self.assertEqual(layout_report(fig)['issues'], [])

    def test_kinds_are_detected(self):
        kinds = available_kinds([self.bands, self.dos])
        for kind in ('bands', 'fatbands', 'pdos', 'bands_pdos'):
            self.assertIn(kind, kinds)
        self.assertNotIn('bands_spin', kinds)
        self.assertIn('bands_spin', available_kinds(self.spin))

    def test_fatbands_every_mode_and_grouping(self):
        for mode in FAT_MODES:
            for group in ('l', 'lm', 'element', 'element+l'):
                if mode == 'rgb' and group != 'element':
                    continue
                with self.subTest(mode=mode, group=group):
                    fig = make_figure('fatbands', self.bands, mode=mode, group=group)
                    values = fig.sparse_dft_meta['values']
                    self.assertEqual(values['resolution'], 'lm')
                    self.assertLessEqual(values['max_state_weight_sum_in_window']['total'], 1.0 + 1e-12)
                    self.assertClean(fig)
        fig = make_figure('fatbands', self.bands, group='element+l')
        self.assertEqual(fig.sparse_dft_meta['values']['groups']['total'], ['Ga s', 'Ga p', 'Ga d', 'As s', 'As p', 'As d'])
        fig = make_figure('fatbands', self.bands, mode='scatter', group='element')
        self.assertEqual(len([c for c in fig.axes[0].collections if isinstance(c, PathCollection)]), 2)

    def test_fatbands_atom_and_orbital_selection(self):
        fig = make_figure('fatbands', self.bands, group='lm', atoms='As', orbitals='p')
        self.assertEqual(fig.sparse_dft_meta['values']['groups']['total'], ['py', 'pz', 'px'])
        # the width of As px on band 1 (index 0 in collection order is the grey base)
        colls = [c for c in fig.axes[0].collections if isinstance(c, LineCollection)]
        self.assertEqual(len(colls), 4)
        with self.assertRaises(InputError):
            make_figure('fatbands', self.bands, mode='rgb', group='lm')       # 9 groups

    def test_bands_decorations_and_bz_inset(self):
        try:
            import scipy  # noqa: F401
        except ImportError:
            self.skipTest('scipy not installed')
        fig = make_figure('bands', self.bands, gap_arrow=True, bz_inset=True)
        ax = fig.axes[0]
        self.assertEqual(len(ax.child_axes), 1)                 # the Brillouin-zone inset
        inset = ax.child_axes[0]
        self.assertGreaterEqual(len(inset.lines), 14)           # 14 fcc BZ faces + path
        self.assertTrue(any(t.get_text() == r'$\Gamma$' for t in inset.texts))
        edges = band_edges(load_band_data(self.bands))
        self.assertAlmostEqual(fig.sparse_dft_meta['values']['gap_ev'], edges.gap_ev, places=12)
        self.assertTrue(ax.texts)
        self.assertClean(fig)

    def test_bands_spin_side_and_overlay(self):
        fig = make_figure('bands_spin', self.spin)
        self.assertEqual(len(fig.axes), 2)
        self.assertEqual(fig.axes[0].get_ylim(), fig.axes[1].get_ylim())
        self.assertClean(fig)
        fig = make_figure('bands_spin', self.spin, layout='overlay')
        self.assertEqual(len(fig.axes), 1)
        self.assertEqual([t.get_text() for t in fig.axes[0].get_legend().get_texts()], ['spin up', 'spin down'])
        with self.assertRaises(InputError):
            make_figure('bands_spin', self.bands)

    def test_pdos_groupings_integrated_and_stacked(self):
        fig = make_figure('pdos', self.dos)                       # default element+l: one panel per element
        self.assertEqual(len(fig.axes), 2)
        fig = make_figure('pdos', self.dos, group='lm', orbitals='p')
        self.assertEqual([t.get_text() for t in fig.axes[0].get_legend().get_texts()], ['py', 'pz', 'px'])
        self.assertClean(fig)
        fig = make_figure('pdos', self.dos, group='element', integrated=True)
        integrated = max(max(l.get_ydata()) for l in fig.axes[0].lines if len(l.get_xdata()) > 100)
        self.assertAlmostEqual(integrated, 9 * 0.3, delta=2e-3)    # As: 9 lm Gaussians of weight 0.3
        fig = make_figure('pdos', self.dos, group='element', stacked=True)
        self.assertClean(fig)

    def test_bands_pdos_shares_energy_axis(self):
        fig = make_figure('bands_pdos', [self.bands, self.dos])
        self.assertEqual(fig.axes[0].get_ylim(), fig.axes[1].get_ylim())
        self.assertEqual(fig.sparse_dft_meta['values']['pdos_groups'], ['Ga', 'As'])
        self.assertClean(fig)

    def test_dos_compare_identical_is_zero(self):
        other = write_dos(self.root / 'other')
        fig = make_figure('dos_compare', self.dos, reference=other)
        self.assertEqual(fig.sparse_dft_meta['values']['l1_normalised'], {'total': 0.0})
        self.assertClean(fig)

    def test_band_suite_writes_every_figure(self):
        plan = [name for name, _, _ in suite_plan([self.bands, self.dos])]
        self.assertEqual(plan, ['bands', 'fatbands_element', 'fatbands_orbital', 'fatbands_element_orbital',
                                'fatbands_lm', 'pdos_element_orbital', 'pdos_lm', 'bands_pdos'])
        result = band_suite([self.bands, self.dos], self.root / 'suite', formats=('png',), dpi=100)
        self.assertEqual([f['name'] for f in result['figures']], plan)
        self.assertEqual(result['skipped'], [])
        for f in result['figures']:
            self.assertEqual(f['layout_issues'], [], f['name'])
            self.assertTrue(all(Path(p).exists() for p in f['files']))


if __name__ == '__main__':
    unittest.main()
