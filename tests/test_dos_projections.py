"""lm-resolved projections, PDOS grouping and the `sparse-dft dos` workflow."""
import builtins
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from sparse_dft.analysis.electronic import (DosCurve, DosData, GROUPINGS, canonical_orbital, dos_from_payload,
                                      load_band_data, load_dos, load_projections, parse_atom_selection)
from sparse_dft.bands import from_payload
from sparse_dft.errors import InputError
from sparse_dft.scf import gaussian_broaden

ROOT = Path(__file__).resolve().parents[1]
LM = ['s', 'py', 'pz', 'px', 'dxy', 'dyz', 'dz2', 'dxz', 'dx2-y2']


def _bands_with_lm(spin=False):
    """Two atoms (Ga, As), 2 k-points, 2 bands; weight = 0.01*(atom+1)*(orbital+1)*(band+1)."""
    w = [[[[0.01 * (a + 1) * (o + 1) * (b + 1) for o in range(9)] for a in range(2)] for b in range(2)]
         for _ in range(2)]
    flat = [v for k in w for b in k for a in b for v in a]
    payload = {'kpoints': [[0, 0, 0], [0.5, 0, 0]], 'eigenvalues': [[-1.0, 1.0], [-0.5, 1.5]],
               'fermi_energy_ev': 0.0}
    if spin:
        payload['spin_channels'] = {'up': payload['eigenvalues'], 'down': payload['eigenvalues']}
        lm = {'up': w, 'down': [[[[2 * v for v in a] for a in b] for b in k] for k in w]}
    else:
        lm = flat
    payload['projections'] = {'projection_orbitals': ['s', 'p', 'd'], 'projection_atom_symbols': ['Ga', 'As'],
                              'projection_weights': ({'up': [[[[1, 1, 1]] * 2] * 2] * 2,
                                                      'down': [[[[1, 1, 1]] * 2] * 2] * 2} if spin
                                                     else [1.0] * (2 * 2 * 2 * 3)),
                              'projection_lm_orbitals': LM, 'projection_lm_weights': lm,
                              'projection_lm_weight_definition': 'test', 'weight_definition': 'legacy'}
    return from_payload(payload), w


class BroadeningTests(unittest.TestCase):
    def test_gaussian_integrates_to_weight_and_projection(self):
        grid = [-3 + 0.002 * i for i in range(3001)]
        out = gaussian_broaden(grid, [[0.0, 0.5]], [2.0], 0.05, [[[0.25], [1.0]]])
        step = grid[1] - grid[0]
        self.assertAlmostEqual(sum(out[0]) * step, 4.0, places=6)          # 2 states x weight 2
        self.assertAlmostEqual(sum(out[1]) * step, 2.0 * 0.25 + 2.0, places=6)

    def test_pure_python_fallback_matches_numpy(self):
        grid = [-1 + 0.01 * i for i in range(201)]
        args = (grid, [[-0.3, 0.2], [0.0, 0.41]], [0.5, 0.5], 0.04, [[[1, 0], [0.3, 0.7]], [[0.5, 0.5], [0, 1]]])
        fast = gaussian_broaden(*args)
        real_import = builtins.__import__

        def no_numpy(name, *a, **k):
            if name == 'numpy':
                raise ImportError('blocked')
            return real_import(name, *a, **k)
        with mock.patch('builtins.__import__', no_numpy):
            slow = gaussian_broaden(*args)
        self.assertEqual(len(fast), 3)
        self.assertLess(max(abs(x - y) for a, b in zip(fast, slow) for x, y in zip(a, b)), 1e-12)

    def test_rejects_nonpositive_sigma(self):
        with self.assertRaises(InputError):
            gaussian_broaden([0.0, 1.0], [[0.0]], [1.0], 0.0)


class ProjectionTests(unittest.TestCase):
    def test_lm_groups_selection_and_l_consistency(self):
        bands, w = _bands_with_lm()
        p = load_projections(bands)
        self.assertEqual(p.resolution, 'lm')
        self.assertEqual(p.orbitals, LM)
        self.assertEqual(p.weight_definition, 'test')
        g = p.grouped('total', 'element+lm')
        self.assertEqual(list(g)[:3], ['Ga s', 'Ga py', 'Ga pz'])
        self.assertEqual(len(g), 18)
        self.assertAlmostEqual(g['As dxy'][1][1], w[1][1][1][4], places=12)
        # l from lm equals the collapsed l tensor; the legacy l tensor is not used
        by_l, collapsed = p.grouped('total', 'l'), load_projections(bands, resolution='l')
        self.assertEqual(collapsed.resolution, 'l')
        self.assertEqual(collapsed.orbitals, ['s', 'p', 'd'])
        ref = collapsed.grouped('total', 'l')
        for name in ('s', 'p', 'd'):
            for r, s in zip(by_l[name], ref[name]):
                for x, y in zip(r, s):
                    self.assertAlmostEqual(x, y, places=12)
        self.assertAlmostEqual(by_l['p'][0][0], sum(w[0][0][a][o] for a in range(2) for o in (1, 2, 3)), places=12)
        # atom and orbital filters
        only_as_d = p.grouped('total', 'element+l', atoms='As', orbitals='d')
        self.assertEqual(list(only_as_d), ['As d'])
        self.assertAlmostEqual(only_as_d['As d'][0][1], sum(w[0][1][1][o] for o in range(4, 9)), places=12)
        self.assertEqual(list(p.grouped('total', 'atom', atoms='2')), ['As2'])
        self.assertEqual(list(p.grouped('total', 'lm', orbitals='dxy,s')), ['s', 'dxy'])
        with self.assertRaises(InputError):
            p.grouped('total', 'lm', orbitals='fxyz')
        with self.assertRaises(InputError):
            load_projections(load_band_data(ROOT / 'tests/fixtures/plotting/synthetic_bands'),
                             ROOT / 'tests/fixtures/plotting/synthetic_bands/PROCAR').grouped('total', 'lm')

    def test_spin_lm_channels(self):
        bands, w = _bands_with_lm(spin=True)
        p = load_projections(bands)
        self.assertEqual(sorted(p.channels), ['down', 'up'])
        self.assertAlmostEqual(p.grouped('down', 'element')['Ga'][0][0], 2 * sum(w[0][0][0]), places=12)

    def test_every_grouping_is_accepted(self):
        p = load_projections(_bands_with_lm()[0])
        totals = []
        for by in GROUPINGS:
            g = p.grouped('total', by)
            totals.append(sum(v for rows in g.values() for row in rows for v in row))
        for t in totals:
            self.assertAlmostEqual(t, totals[0], places=10)

    def test_atom_selection_and_orbital_names(self):
        species = ['Ga', 'Ga', 'As', 'As']
        self.assertEqual(parse_atom_selection('1-2,4', species, 4), [1, 2, 4])
        self.assertEqual(parse_atom_selection(['As', 1], species, 4), [1, 3, 4])
        self.assertEqual(parse_atom_selection(None, species, 4), [1, 2, 3, 4])
        for bad in ('5', '0', 'Si', '3-1'):
            with self.assertRaises(InputError):
                parse_atom_selection(bad, species, 4)
        self.assertEqual(canonical_orbital('x2-y2'), 'dx2-y2')
        self.assertEqual(canonical_orbital('PX'), 'px')


VASPRUN_SPIN = """<?xml version="1.0"?>
<modeling>
 <atominfo><array name="atoms"><set><rc><c>Ni</c><c>1</c></rc></set></array></atominfo>
 <calculation><dos>
  <i name="efermi">0.5</i>
  <total><array><field>energy</field><field>total</field><field>integrated</field><set>
   <set comment="spin 1"><r>0.0 1.0 0.0</r><r>1.0 2.0 0.0</r></set>
   <set comment="spin 2"><r>0.0 3.0 0.0</r><r>1.0 4.0 0.0</r></set>
  </set></array></total>
  <partial><array><field>energy</field><field>s</field><field>py</field><field>pz</field><field>px</field>
   <field>dxy</field><field>dyz</field><field>dz2</field><field>dxz</field><field>x2-y2</field><set>
   <set comment="ion 1">
    <set comment="spin 1"><r>0.0 .1 .2 .3 .4 .5 .6 .7 .8 .9</r><r>1.0 .1 .1 .1 .1 .1 .1 .1 .1 .1</r></set>
    <set comment="spin 2"><r>0.0 .2 .2 .2 .2 .2 .2 .2 .2 .2</r><r>1.0 .3 .3 .3 .3 .3 .3 .3 .3 .3</r></set>
   </set>
  </set></array></partial>
 </dos></calculation>
</modeling>
"""


class DosLoaderTests(unittest.TestCase):
    def test_vasprun_spin_pdos_keeps_both_spins_and_lm(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'vasprun.xml'
            path.write_text(VASPRUN_SPIN, encoding='utf-8')
            d = load_dos(path)
        self.assertTrue(d.spin_polarized)
        self.assertEqual(d.resolution, 'lm')
        self.assertEqual(len(d.pdos_raw), 18)
        self.assertEqual({c.spin for c in d.pdos_raw}, {'up', 'down'})
        self.assertIn('dx2-y2', {c.orbital for c in d.pdos_raw})
        pd = {(c.label, c.spin): c.values for c in d.pdos}
        self.assertAlmostEqual(pd[('Ni d', 'up')][0], .5 + .6 + .7 + .8 + .9)
        self.assertAlmostEqual(pd[('Ni d', 'down')][1], 5 * .3)
        self.assertEqual([c.label for c in d.grouped('lm', orbitals='p', spin='down')], ['py', 'pz', 'px'])

    def test_native_dos_payload_round_trip(self):
        energies = [0.0, 0.5, 1.0]
        payload = {'schema_version': 'sparse-dft-dos-1', 'energies_ev': energies, 'fermi_energy_ev': 0.4,
                   'total': {'total': [1, 2, 3]}, 'spin_dos': {'total': [1, 2, 3]},
                   'pdos': [{'atom': 1, 'element': 'Si',
                             'curves': [{'orbital': o, 'spin': None, 'values': [0.1, 0.2, 0.3]} for o in LM[:4]]},
                            {'atom': 2, 'element': 'Si',
                             'curves': [{'orbital': o, 'spin': None, 'values': [0.1, 0.1, 0.1]} for o in LM[:4]]}],
                   'metadata': {'format': 'Gaussian DOS from a Sparse-DFT NSCF uniform mesh', 'sigma_ev': 0.1,
                                'sampling': 'uniform_mesh', 'nkpoints': 8, 'complete_below_ev': 0.8}}
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'dos.json'
            path.write_text(json.dumps(payload), encoding='utf-8')
            d = load_dos(Path(td))                     # directory lookup finds dos.json first
        self.assertEqual(d.origin, 'sdft_nscf_mesh')
        self.assertEqual(d.resolution, 'lm')
        self.assertEqual([c.label for c in d.pdos], ['Si s', 'Si p'])
        self.assertEqual(d.pdos[1].atoms, [1, 2])
        self.assertAlmostEqual(d.pdos[1].values[2], 3 * 0.3 + 3 * 0.1)
        self.assertTrue(any('incomplete' in w for w in d.warnings))
        self.assertEqual([c.label for c in d.grouped('atom+l', atoms=[2])], ['Si2 s', 'Si2 p'])

    def test_l_only_dos_refuses_lm_grouping(self):
        raw = [DosCurve('O p', [1.0], None, 'O', 'p', [1], 'p')]
        d = DosData('x', [0.0], None, [DosCurve('total', [1.0])], raw, False, 'DOSCAR', pdos_raw=raw)
        self.assertEqual(d.resolution, 'l')
        with self.assertRaises(InputError):
            d.grouped('lm')
        self.assertEqual([c.label for c in d.grouped('element')], ['O'])


class NativeDosTests(unittest.TestCase):
    def test_si_dos_counts_electrons_and_keeps_lm_pdos(self):
        from sparse_dft.resources import project_paw_archive
        from sparse_dft.scf import run_dos, run_scf
        options = dict(cutoff_ev=200.0, bands=8, max_iterations=480, density_tolerance=1e-6,
                       onecenter_density_tolerance=1e-6, mixing=.2, scf_grid=(2, 2, 2))
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            scf = run_scf(ROOT / 'examples/Si.POSCAR', project_paw_archive(), root / 'scf', **options)
            self.assertTrue(scf['converged'])
            dos = run_dos(ROOT / 'examples/Si.POSCAR', project_paw_archive(), root / 'dos', (3, 3, 3),
                          sigma_ev=0.1, scf_checkpoint=root / 'scf/scf.json', **options)
            meta = dos['metadata']
            self.assertEqual(meta['nkpoints'], 27)
            self.assertEqual(meta['pdos_orbitals'][:4], ['s', 'py', 'pz', 'px'])
            # 8 valence electrons; the rectangle rule on a 5 meV grid is good to ~1e-3
            self.assertLess(abs(meta['integrated_electrons'] - 8.0), 5e-3)
            loaded = load_dos(root / 'dos')
            self.assertEqual(loaded.resolution, 'lm')
            self.assertEqual({c.element for c in loaded.pdos}, {'Si'})
            step = loaded.energies_ev[1] - loaded.energies_ev[0]
            total = sum(loaded.total[0].values) * step
            spheres = sum(sum(c.values) for c in loaded.pdos_raw) * step
            self.assertTrue(math.isfinite(spheres) and 0 < spheres < total)     # spheres miss the interstitial


if __name__ == '__main__':
    unittest.main()
