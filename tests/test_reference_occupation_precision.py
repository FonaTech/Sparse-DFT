from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
for _scripts in (ROOT/'tools', ROOT/'benchmarks'/'scripts'):  # development / release layout
    sys.path.insert(0,str(_scripts))
from validate_scf_reference import reference_occupations
from sparse_dft.bands import BandStructure

def fixture(directory):
    # Actual Al Gamma degeneracy, with the two VASP output precisions.
    (directory/'EIGENVAL').write_text('1 1 1 1\n0 0 0 0 0\n0\nCAR\nAl fixture\n3 1 4\n\n'
        '0 0 0 1\n1 -3.156081 1.000000\n2 20.462680 0.166667\n'
        '3 20.462680 0.166667\n4 20.462680 0.166667\n')
    return BandStructure('XML precision fixture',[(0.,0.,0.)],[0.],
        [[-3.156081,20.462680,20.462680,20.462680]],[None],
        occupations={'up':[[1.,.1667,.1667,.1667]]})

class ReferenceOccupationPrecisionTests(unittest.TestCase):
    def test_fractional_degeneracy_uses_verified_eigenval(self):
        with tempfile.TemporaryDirectory() as td:
            directory=Path(td);bands=fixture(directory)
            rows,source=reference_occupations(bands,directory)
            self.assertEqual(source.name,'EIGENVAL')
            self.assertLess(abs(2*rows[0][1]-1/3),1e-5)
            self.assertGreater(abs(2*bands.occupations['up'][0][1]-1/3),1e-5)

    def test_stale_eigenval_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            bands=fixture(Path(td))
            target=Path(td)/'EIGENVAL'
            target.write_text(target.read_text().replace('-3.156081','-3.166081'))
            with self.assertRaisesRegex(ValueError,'energies disagree'):
                reference_occupations(bands,Path(td))

if __name__=='__main__':unittest.main()
