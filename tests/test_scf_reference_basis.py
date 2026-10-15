"""Frontend wiring guards; real strained-cell numerical gate runs separately."""
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from sparse_dft.scf import run_scf, _validated_reference_lattice
from sparse_dft.structure import Structure
from sparse_dft.errors import InputError


class ReferenceBasisTests(unittest.TestCase):
    def test_rejects_malformed_and_nonfinite_reference_before_launch(self):
        for value in ([[1,0],[0,1]], [[1,0,0],[0,1,0],[0,0,0]],
                      [[float('nan'),0,0],[0,1,0],[0,0,1]], 'not a lattice'):
            with self.subTest(value=value), patch('sparse_dft.scf.subprocess.run') as launch:
                with self.assertRaises(InputError):
                    run_scf(None, '/unused', '/unused', basis_reference_lattice_bohr=value)
                launch.assert_not_called()

    def test_reference_survives_frontend_and_input_serialization(self):
        lattice = [[4.,.7,.2],[0.,3.,.4],[.1,.3,5.]]
        # Reference must differ from the deformed lattice used by the Hamiltonian.
        structure = Structure(['Si'],[(.2,.3,.4)],
                              [[1.03*v for v in row] for row in lattice], True, (True,)*3)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            choices = {'Si': {'cache': str(root/'numeric.bin'), 'zval':4, 'enmax_ev':245.345}}
            archive = types.SimpleNamespace(local_hashes=lambda: {})
            def native(command, **kwargs):
                lines = Path(command[2]).read_text().splitlines()
                reference = next(line for line in lines if line.startswith('basis_reference_lattice_bohr '))
                self.assertEqual([float(x) for x in reference.split()[1:]],
                                 [x for row in lattice for x in row])
                Path(command[3]).write_text(json.dumps({'basis_reference_lattice_bohr':lattice}))
                return types.SimpleNamespace(returncode=0, stdout='', stderr='')
            with patch('sparse_dft.scf._resolve_caches',return_value=(choices, [], archive)), \
                 patch('sparse_dft.scf._executable',return_value=Path('/unused/native')), \
                 patch('sparse_dft.scf.subprocess.run',side_effect=native):
                result = run_scf(structure, root, root/'job', cutoff_ev=500,
                                 basis_reference_lattice_bohr=lattice)
                self.assertEqual(result['provenance']['basis_reference_lattice_bohr'], lattice)
            self.assertEqual(_validated_reference_lattice(lattice), lattice)


if __name__ == '__main__':
    unittest.main()
