"""A requested magnetic/+U model must survive every public JSON API route."""
import unittest
from unittest.mock import patch

from sparse_dft.api import dispatch


class SpinApiRoutingTests(unittest.TestCase):
    def test_same_model_reaches_scf_bands_and_both_relaxations(self):
        model = {'spin_channels': 2, 'initial_moments': [2, -2, 0, 0],
                 'hubbard': {'Fe': {'l': 2, 'ueff_ev': 4}}}
        for method, target, result in (
                ('job.run', 'run_scf', {'converged': True}),
                ('bands.run', 'run_bands', {'metadata': {'scf_converged': True}}),
                ('relax.ionic', 'ionic_relax', {'status': 'CONVERGED'}),
                ('relax.cell', 'cell_relax', {'status': 'CONVERGED'})):
            with self.subTest(method=method), patch('sparse_dft.api.'+target, return_value=result) as native:
                response = dispatch(method, {'structure': 'FeSe4.POSCAR', 'paw_root': 'licensed-paw',
                                             'kpoints': [[0, 0, 0]], 'bands': 32, **model})
                self.assertEqual(response['status'], 'PASS', response)
                for key, expected in model.items():
                    self.assertEqual(native.call_args.kwargs[key], expected)
                self.assertEqual(native.call_args.kwargs['bands'], 32)

    def test_frozen_bands_inherit_an_omitted_spin_model(self):
        with patch('sparse_dft.api.run_bands', return_value={'metadata': {'scf_converged': True}}) as native:
            dispatch('bands.run', {'structure': 'FeSe4.POSCAR', 'paw_root': 'licensed-paw',
                                   'kpoints': [[0, 0, 0]], 'scf_checkpoint': 'U4.json'})
            self.assertIsNone(native.call_args.kwargs['spin_channels'])
            self.assertIsNone(native.call_args.kwargs['hubbard'])

    def test_explicit_invalid_channel_is_not_replaced_with_a_default(self):
        with patch('sparse_dft.api.run_scf', side_effect=ValueError('invalid spin selection')) as native:
            response = dispatch('job.run', {'structure': 'FeSe4.POSCAR', 'paw_root': 'licensed-paw',
                                            'spin_channels': 0})
            self.assertEqual(response['status'], 'FAILED')
            self.assertEqual(native.call_args.kwargs['spin_channels'], 0)

    def test_nonmagnetic_scf_default_is_preserved(self):
        with patch('sparse_dft.api.run_scf', return_value={'converged': True}) as native:
            dispatch('job.run', {'structure': 'Si2.POSCAR', 'paw_root': 'licensed-paw'})
            self.assertEqual(native.call_args.kwargs['spin_channels'], 1)


if __name__ == '__main__':
    unittest.main()
