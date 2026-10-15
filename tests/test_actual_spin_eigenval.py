"""Real magnetic VASP output cross-checked against its independent XML layout."""
import tempfile
import unittest
from pathlib import Path

from sparse_dft.bands import parse_eigenval, parse_vasprun_bands
from sparse_dft.errors import InputError


class ActualSpinEigenvalTests(unittest.TestCase):
    def test_complete_real_ni_spin_energies_and_occupations_match_xml(self):
        fixture=Path(__file__).parent/'fixtures/vasp/ni1_spin_actual'
        text=parse_eigenval(fixture/'EIGENVAL')
        xml=parse_vasprun_bands(fixture/'bands-only.xml')
        self.assertEqual((text.nkpoints,text.nbands),(1,12))
        self.assertEqual(set(text.spin_channels),{'up','down'})
        for spin in ['up','down']:
            for te,xe,to,xo in zip(text.spin_channels[spin][0],xml.spin_channels[spin][0],
                    text.occupations[spin][0],xml.occupations[spin][0]):
                # Actual XML stores4 decimals and EIGENVAL6 decimals: their
                # maximum rounding difference is5e-5+5e-7eV.
                self.assertAlmostEqual(te,xe,delta=5.1e-5)
                self.assertAlmostEqual(to,xo,delta=6e-7)
        self.assertAlmostEqual(text.spin_channels['down'][0][-1],39.745557,places=6)
        self.assertEqual(text.occupations['down'][0][-1],0.)

    def test_mixed_and_missing_spin_columns_fail_explicitly(self):
        fixture=Path(__file__).parent/'fixtures/vasp/ni1_spin_actual/EIGENVAL'
        lines=fixture.read_text().splitlines()
        for replacement in ['1 0.2 1.0','1 0.2 0.5 1.0']:
            changed=lines.copy();changed[8]=replacement
            with tempfile.TemporaryDirectory() as td:
                p=Path(td)/'EIGENVAL';p.write_text('\n'.join(changed)+'\n')
                with self.assertRaises(InputError):parse_eigenval(p)


if __name__=='__main__':unittest.main()
