import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from sparse_dft.paw import Paw54JsonArchive
from sparse_dft.resources import project_paw_archive


ROOT = project_paw_archive()


class NumericPawTests(unittest.TestCase):
    def test_reference_records_have_strict_tensor_shapes(self):
        archive = Paw54JsonArchive(ROOT)
        expected = {
            'H:standard': (363, 3), 'C:standard': (387, 4),
            'O:standard': (377, 4), 'Si:standard': (351, 4),
            'Fe:standard': (367, 6), 'Ni:standard': (349, 6),
        }
        with tempfile.TemporaryDirectory() as tmp:
            for record_id, (nradial, nproj) in expected.items():
                dataset = archive.restore_dataset(record_id, Path(tmp) / record_id.replace(':', '_'))
                numeric = dataset.numeric_data
                self.assertEqual(numeric.main_grid_size, 1000)
                self.assertEqual(numeric.radial.npoints, nradial)
                self.assertEqual(sum(c.nprojectors for c in numeric.nonlocal_channels), nproj)
                self.assertEqual(len(numeric.radial.pseudo_wavefunctions), nproj)
                self.assertEqual(len(numeric.radial.ae_wavefunctions), nproj)
                self.assertEqual(len(numeric.radial.augmentation_coefficients), nproj * nproj)
                self.assertEqual(len(numeric.kinetic_error_cutoff_ev), len(numeric.kinetic_error_ev))
                self.assertTrue(all(b > a for a, b in zip(numeric.kinetic_error_cutoff_ev,
                                                        numeric.kinetic_error_cutoff_ev[1:])))
                if record_id == 'Si:standard':
                    self.assertEqual(numeric.atomic_reference_energy_ev, 103.0669)
                    self.assertEqual(numeric.core_xc_outside_ev, 0.0)
                    self.assertEqual(len(numeric.kinetic_error_ev), 100)
                    self.assertEqual(numeric.kinetic_error_cutoff_ev[0], 20.0)
                    self.assertEqual(numeric.kinetic_error_ev[0], 10.1)
                self.assertTrue((Path(tmp) / record_id.replace(':', '_') / 'numeric.bin').stat().st_size > 64)

    def test_full_archive_numeric_sweep(self):
        archive = Paw54JsonArchive(ROOT)
        passed = 0
        skipped = 0
        for record in archive.iter_records():
            if record.get('variant', {}).get('family') == 'all_electron_like':
                skipped += 1
                continue
            numeric = archive.parse_numeric_data(record['files']['POTCAR']['text'])
            self.assertGreater(numeric.radial.npoints, 0)
            self.assertGreater(sum(channel.nprojectors for channel in numeric.nonlocal_channels), 0)
            passed += 1
        self.assertEqual((passed, skipped), (324, 3))

    def test_atomic_energy_header_and_kinetic_table_are_required(self):
        from sparse_dft.errors import ArchiveIntegrityError
        archive = Paw54JsonArchive(ROOT)
        text = archive.get_record('Si:standard')['files']['POTCAR']['text']
        with self.assertRaises(ArchiveIntegrityError):
            archive.parse_numeric_data(text.replace('EATOM', 'NOATOM', 1))
        with self.assertRaises(ArchiveIntegrityError):
            archive.parse_numeric_data(text.replace('NDATA  =      100', 'NDATA  =      101', 1))

    def test_cpp_reader_matches_python_cache(self):
        archive = Paw54JsonArchive(ROOT)
        with tempfile.TemporaryDirectory() as tmp:
            bench = Path('build/sdft_paw_bench')
            if not bench.exists():
                self.skipTest('C++ build has not been configured')
            for record_id in ('H:standard', 'C:standard', 'O:standard', 'Si:standard', 'Fe:standard', 'Ni:standard'):
                directory = Path(tmp) / record_id.replace(':', '_')
                dataset = archive.restore_dataset(record_id, directory)
                self.assertTrue(dataset.numeric_data)
                result = subprocess.run([str(bench), str(directory / 'numeric.bin'), '1'], text=True, capture_output=True, check=True)
                self.assertIn('iterations=1', result.stdout)
                self.assertIn('checksum=', result.stdout)
