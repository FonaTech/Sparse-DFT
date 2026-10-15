"""Real common-kernel SCF default, explicit support and frozen NSCF agreement."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from sparse_dft.scf import run_scf, run_bands
from sparse_dft.resources import project_paw_archive


class SharedDensityDefault(unittest.TestCase):
    def test_actual_automatic_and_explicit_fine_support_agree(self):
        source = Path(__file__).resolve().parents[1]
        options = dict(cutoff_ev=100., bands=8, max_iterations=480, mixing=.05,
                       density_tolerance=1e-6, onecenter_density_tolerance=1e-6)
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {'SDFT_PAW_SHARED_KERNEL':'1'}):
            root = Path(directory)
            automatic = run_scf(source/'examples/Si.POSCAR', project_paw_archive(), root/'auto', **options)
            explicit = run_scf(source/'examples/Si.POSCAR', project_paw_archive(), root/'explicit',
                               density_cutoff_ev=1600., **options)
            self.assertTrue(automatic['converged'])
            self.assertTrue(explicit['converged'])
            self.assertEqual(automatic['density_cutoff_ev'], 1600.)
            self.assertEqual(automatic['density_support_index_sha256'], explicit['density_support_index_sha256'])
            for key in ['total_energy_ev', 'eigenvalues_ev', 'hf_total_forces_ev_per_angstrom',
                        'hf_total_stress_ev_per_angstrom3']:
                np.testing.assert_allclose(automatic[key], explicit[key], atol=1e-7, rtol=0)
            bands = run_bands(source/'examples/Si.POSCAR', project_paw_archive(), root/'bands',
                [(0.,0.,0.), (.125,0.,0.), (0.,0.,0.)], scf_checkpoint=root/'auto/scf_result.json', **options)
            self.assertEqual(bands['metadata']['density_cutoff_ev'], 1600.)
            self.assertLessEqual(max(bands['eigensolver_max_residuals_hartree']), 1e-12)


if __name__ == '__main__': unittest.main()
