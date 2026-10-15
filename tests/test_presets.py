import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from preset_fixtures import CASES, al_slab, co_box, graphene, rocksalt  # noqa: E402

from sparse_dft.errors import InputError  # noqa: E402
from sparse_dft.presets import (CATEGORIES, METAL_KT_EV, analyze_structure, nominal_oxidation_states,  # noqa: E402
                          options_from_items, suggest)
from sparse_dft.structure import Structure, write_poscar  # noqa: E402

PAW = os.environ.get('SDFT_PAW_ARCHIVE')
ROOT = Path(__file__).resolve().parents[1]


def profile(name, paw=None):
    s = CASES[name][0]()
    return s, analyze_structure(s, paw, with_symmetry=False)


class ClassificationTest(unittest.TestCase):
    def test_every_fixture_lands_in_its_category(self):
        for name, (factory, expected) in CASES.items():
            with self.subTest(name=name):
                p = analyze_structure(factory(), PAW, with_symmetry=False)
                self.assertEqual(p.category, expected, p.reasons)

    def test_fixtures_cover_every_category_but_other(self):
        covered = {expected for _, expected in CASES.values()}
        self.assertEqual(set(CATEGORIES) - covered, {'other'})

    def test_vacuum_is_measured_along_the_plane_normal(self):
        p = analyze_structure(graphene(), with_symmetry=False)
        self.assertEqual(p.periodic_axes, [True, True, False])
        self.assertAlmostEqual(p.thickness_angstrom, 0.0, places=9)
        thick = analyze_structure(al_slab(layers=7), with_symmetry=False)
        self.assertEqual(thick.category, 'slab')
        self.assertAlmostEqual(thick.thickness_angstrom, 6 * 2.025, places=6)

    def test_small_box_is_not_a_molecule(self):
        # 4 A box: no 5 A gap on any axis, so the cell is treated as a bulk solid.
        dense = Structure(['C', 'O'], [(0, 0, 0), (0, 0, 1.13 / 4)], [[4, 0, 0], [0, 4, 0], [0, 0, 4]], True)
        self.assertEqual(analyze_structure(dense, with_symmetry=False).geometric_dimensionality, 3)

    def test_cell_is_required(self):
        with self.assertRaises(InputError):
            analyze_structure(Structure(['H', 'H'], [(0, 0, 0), (0, 0, .74)]))

    def test_oxidation_states(self):
        self.assertEqual(nominal_oxidation_states({'Fe': 3, 'O': 4})['Fe'], 8 / 3)
        self.assertEqual(nominal_oxidation_states({'Ti': 1, 'O': 2})['Ti'], 4.0)
        self.assertIsNone(nominal_oxidation_states({'Fe': 1, 'Co': 1, 'O': 2}))
        self.assertIsNone(nominal_oxidation_states({'Si': 1}))

    def test_closed_shell_d0_oxide_is_not_magnetic(self):
        _, p = profile('TiO2')
        self.assertEqual(p.open_shell_d, [])
        self.assertNotIn('tm_oxide', p.tags)


class SuggestionTest(unittest.TestCase):
    def test_anchor_rules_reproduced(self):
        """The suggestions repeat the conditions compared with VASP (tools/vasp_anchor.py)."""
        for name, kt, spin in (('Si', 0.0, 1), ('MgO', 0.0, 1), ('LiF', 0.0, 1), ('Al', METAL_KT_EV, 1),
                               ('Al_slab', METAL_KT_EV, 1), ('CO', 0.0, 1),
                               ('Fe', METAL_KT_EV, 2), ('Ni', METAL_KT_EV, 2)):
            with self.subTest(name=name):
                s, p = profile(name, PAW)
                o = suggest(p, 'scf', s)['options']
                self.assertEqual(o['smearing_ev'], kt)
                self.assertEqual(o['spin_channels'], spin)
                self.assertEqual(o['mesh_mode'], 'gamma')
                self.assertNotIn('cutoff', o)   # left to the automatic rule
                self.assertNotIn('bands', o)
        # MoS2 (Mo4+ d2) cannot be told from metallic 1T-MoS2 or NbSe2 by composition:
        # conservative 0.05 eV (the anchor ran kT = 0; the gap of 1.7 eV makes the difference ~exp(-17)).
        s, p = profile('MoS2', PAW)
        self.assertEqual(suggest(p, 'scf', s)['options']['smearing_ev'], 0.05)
        s, p = profile('Fe')
        self.assertEqual(suggest(p, 'scf', s)['options']['initial_moments'], [4.0])
        s, p = profile('Ni')
        self.assertEqual(suggest(p, 'scf', s)['options']['initial_moments'], [2.0])

    def test_vacuum_axes_get_one_k_point(self):
        for name, mask in (('Al_slab', 2), ('MoS2', 2), ('graphene', 2), ('C_chain', 0), ('CO', None)):
            with self.subTest(name=name):
                s, p = profile(name)
                grid = suggest(p, 'scf', s)['options']['scf_grid']
                if mask is None:
                    self.assertEqual(grid, [1, 1, 1])
                elif mask == 2:
                    self.assertEqual(grid[2], 1)
                    self.assertGreater(min(grid[:2]), 1)
                else:
                    self.assertEqual(grid[:2], [1, 1])

    def test_hubbard_is_suggested_but_never_an_option(self):
        s, p = profile('NiO', PAW)
        r = suggest(p, 'scf', s)
        item = next(i for i in r['items'] if i['key'] == 'hubbard')
        self.assertEqual(item['value'], ['Ni:2:6.2'])
        self.assertIsNone(item['option'])
        self.assertNotIn('hubbard', r['options'])
        self.assertIn('10.1103/PhysRevB.73.195107', item['note'])
        self.assertEqual(r['options']['initial_moments'], [2.0, 0.0])
        _, mgo = profile('MgO')
        self.assertFalse(any(i['key'] == 'hubbard' for i in suggest(mgo, 'scf', CASES['MgO'][0]())['items']))

    def test_f_moments_follow_the_oxidation_state(self):
        s, p = profile('CeO2', PAW)
        self.assertEqual(suggest(p, 'scf', s)['options']['spin_channels'], 1)  # Ce4+ is 4f0
        self.assertTrue(any('not validated' in w for w in p.warnings))

    @unittest.skipUnless(PAW, 'SDFT_PAW_ARCHIVE not set')
    def test_open_shell_molecule_is_spin_polarized(self):
        s, p = profile('NO2', PAW)
        self.assertTrue(p.odd_electrons)
        o = suggest(p, 'scf', s)['options']
        self.assertEqual(o['spin_channels'], 2)
        self.assertAlmostEqual(sum(o['initial_moments']), 1.0, places=5)
        s, p = profile('Al', PAW)  # odd electrons but metallic: no spin
        self.assertEqual(suggest(p, 'scf', s)['options']['spin_channels'], 1)

    def test_relax_keeps_the_vacuum_cell_fixed(self):
        s, p = profile('MoS2')
        self.assertIs(suggest(p, 'relax', s)['options']['cell'], False)
        s, p = profile('MgO')
        self.assertNotIn('cell', suggest(p, 'relax', s)['options'])

    def test_large_cells_enable_sparsity(self):
        s, p = profile('Si64')
        self.assertEqual(suggest(p, 'scf', s)['options']['sparsity'], 'auto')
        s, p = profile('Si')
        self.assertEqual(suggest(p, 'scf', s)['options']['sparsity'], 'off')

    def test_every_item_has_a_basis(self):
        for name in CASES:
            s, p = profile(name)
            for task in ('scf', 'relax'):
                for item in suggest(p, task, s)['items']:
                    self.assertIn(item['basis'], ('anchor', 'code', 'sweep', 'heuristic', 'literature'))
                    self.assertTrue(item['note'])

    def test_options_round_trip_through_the_parser(self):
        from sparse_dft.cli import build_parser
        s, p = profile('Ni')
        r = suggest(p, 'scf', s)
        with tempfile.TemporaryDirectory() as d:
            poscar = write_poscar(s, Path(d) / 'POSCAR')
            argv = ['scf', str(poscar), '--paw-root', d]
            for dest, value in options_from_items(r['items']).items():
                argv.append('--' + dest.replace('_', '-'))
                argv += [str(v) for v in value] if isinstance(value, list) else [str(value)]
            ns = build_parser().parse_args(argv)
        self.assertEqual(ns.scf_grid, r['options']['scf_grid'])
        self.assertEqual(ns.initial_moments, [2.0])
        self.assertEqual(ns.smearing_ev, METAL_KT_EV)

    def test_unknown_task(self):
        s, p = profile('Si')
        with self.assertRaises(InputError):
            suggest(p, 'phonons', s)


class SuggestCliTest(unittest.TestCase):
    def test_cli_json_and_argv(self):
        with tempfile.TemporaryDirectory() as d:
            poscar = write_poscar(co_box(), Path(d) / 'POSCAR')
            env = dict(os.environ, PYTHONPATH=str(ROOT))
            out = subprocess.run([sys.executable, '-m', 'sparse_dft.cli', 'suggest', str(poscar)], cwd=d, env=env,
                                 capture_output=True, text=True, check=True).stdout
            payload = json.loads(out)
            self.assertEqual(payload['profile']['category'], 'molecule')
            argv = subprocess.run([sys.executable, '-m', 'sparse_dft.cli', 'suggest', str(poscar), '--argv'], cwd=d,
                                  env=env, capture_output=True, text=True, check=True).stdout.split()
            self.assertEqual(argv[:2], ['sparse-dft', 'scf'])
            self.assertIn('--scf-grid', argv)


if __name__ == '__main__':
    unittest.main()
