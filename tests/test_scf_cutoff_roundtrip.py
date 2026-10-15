"""Keep the physical wave/density cutoffs exact across the public/native boundary."""
from pathlib import Path
import json
import tempfile
import unittest

from sparse_dft.scf import _write_input, run_scf, run_bands
from sparse_dft.resources import project_paw_archive
from sparse_dft.structure import Structure


class CutoffRoundtripTests(unittest.TestCase):
    def test_original_li_and_adversarial_fourfold_cutoffs_roundtrip_exactly(self):
        # Li's original recommendation previously passed frontend validation,
        # then failed the native >=4 check solely after 16-digit formatting.
        cutoffs = (1673.6927277190996, 100.00000000000003,
                   799.9999999999999, 1079.2698566792302)
        structure = Structure(['Li'], [(0., 0., 0.)],
                              [[3.49, 0., 0.], [0., 3.49, 0.], [0., 0., 3.49]],
                              True, (True,)*3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'input.sdft'
            choices = {'Li': {'cache': str(Path(directory)/'original.sdftupf'), 'zval': 3.}}
            for wave in cutoffs:
                density = 4*wave
                with self.subTest(wave=wave):
                    _write_input(path, structure, choices, cutoff_ev=wave,
                                 density_cutoff_ev=density, max_iterations=2000,
                                 density_tolerance=1e-10, onecenter_density_tolerance=1e-10,
                                 mixing=.2, bands=8)
                    controls = {row.split()[0]: row.split()[1:]
                                for row in path.read_text().splitlines() if len(row.split()) > 1}
                    saved_wave = float(controls['cutoff_ev'][0])
                    saved_density = float(controls['density_cutoff_ev'][0])
                    self.assertEqual(saved_wave.hex(), wave.hex())
                    self.assertEqual(saved_density.hex(), density.hex())
                    self.assertGreaterEqual(saved_density, 4*saved_wave)

    def test_actual_native_json_and_frozen_nscf_preserve_exact_cutoffs(self):
        wave = 100.00000000000003
        density = 4*wave
        source = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            options = dict(cutoff_ev=wave, density_cutoff_ev=density, bands=8,
                           max_iterations=480, density_tolerance=1e-6,
                           onecenter_density_tolerance=1e-6, mixing=.05)
            result = run_scf(source/'examples/Si.POSCAR', project_paw_archive(),
                             root/'scf', **options)
            self.assertTrue(result['converged'])
            native = json.loads((root/'scf/scf.json').read_text())
            for record in (native, result):
                self.assertEqual(record['cutoff_ev'].hex(), wave.hex())
                self.assertEqual(record['density_cutoff_ev'].hex(), density.hex())
            bands = run_bands(source/'examples/Si.POSCAR', project_paw_archive(),
                              root/'bands', [(0., 0., 0.), (.125, 0., 0.), (0., 0., 0.)],
                              scf_checkpoint=root/'scf/scf_result.json', **options)
            self.assertEqual(bands['metadata']['density_cutoff_ev'].hex(), density.hex())
            self.assertLessEqual(max(bands['eigensolver_max_residuals_hartree']), 1e-12)


if __name__ == '__main__':
    unittest.main()
