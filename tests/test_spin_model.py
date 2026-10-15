import json
import math
import tempfile
import unittest
from pathlib import Path
from sparse_dft.errors import InputError
from sparse_dft.spin_model import validate_spin_model, parse_cli_hubbard
from sparse_dft.bands import density_of_states_from_scf


class SpinModelTests(unittest.TestCase):
    def test_atom_order_and_dudarev_units(self):
        moments, specs = validate_spin_model(['Fe', 'Se', 'Fe'], 2, [2, 0, -2],
            {'Fe': {'l': 2, 'ueff_ev': 4}}, {'Fe': 8, 'Se': 6})
        self.assertEqual(moments, [2, 0, -2])
        self.assertEqual(specs, [{'l': 2, 'ueff_ev': 4}, {'l': -1, 'ueff_ev': 0}, {'l': 2, 'ueff_ev': 4}])

    def test_unsupported_models_are_rejected(self):
        for spin, moments, u in [(1, [2], None), (2, [9], None), (2, [math.nan], None),
                (2, [0], {'Fe': {'l': 2, 'ueff_ev': -1}}),
                (2, [0], {'Fe': {'l': 2, 'u': 4, 'j': 0}}),
                (2, [0], {'Ni': {'l': 2, 'ueff_ev': 4}}),
                (2, [0], {'Fe': {'l': -1, 'ueff_ev': 4}})]:
            with self.subTest(spin=spin, moments=moments, u=u), self.assertRaises(InputError):
                validate_spin_model(['Fe'], spin, moments, u, {'Fe': 8})

    def test_cli_model_is_explicit(self):
        self.assertEqual(parse_cli_hubbard(['Fe:2:4']), {'Fe': {'l': 2, 'ueff_ev': 4}})
        with self.assertRaises(InputError): parse_cli_hubbard(['Fe:2:4', 'Fe:2:3'])

    def test_spin_dos_counts_each_channel_once(self):
        points = [{'fractional': [0, 0, 0], 'weight': 1, 'spin_channel': s,
                   'eigenvalues_ev': [-.5, .5], 'occupations': [1, 0]} for s in (0, 1)]
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'scf.json'
            p.write_text(json.dumps({'converged': True, 'spin_channels': 2,
                'fermi_energy_ev': 0, 'scf_kpoints': points}))
            dos = density_of_states_from_scf(p, npoints=2001)
            xs, ys = dos['energies_ev'], dos['total']['total']
            integral = sum((b-a)*(u+v)/2 for a, b, u, v in zip(xs, xs[1:], ys, ys[1:]))
            self.assertAlmostEqual(integral, 4, places=10)
            self.assertEqual(set(dos['spin_dos']), {'up', 'down'})
            points[1]['weight'] = .5
            p.write_text(json.dumps({'converged': True, 'spin_channels': 2, 'scf_kpoints': points}))
            with self.assertRaises(InputError): density_of_states_from_scf(p)
