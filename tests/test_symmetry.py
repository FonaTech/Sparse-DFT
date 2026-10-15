import json
import tempfile
import unittest
from pathlib import Path

from sparse_dft.errors import InputError
from sparse_dft.structure import Structure, read_poscar, write_poscar

try:
    import spglib  # noqa: F401
    import seekpath  # noqa: F401
    HAVE_SYMMETRY = True
except ImportError:
    HAVE_SYMMETRY = False

A = 5.43
FCC = [[0, A / 2, A / 2], [A / 2, 0, A / 2], [A / 2, A / 2, 0]]


def silicon(shift=0.0):
    return Structure(['Si', 'Si'], [(0, 0, 0), (0.25 + shift, 0.25, 0.25)], FCC, True)


def conventional_silicon():
    base = [(0, 0, 0), (0, .5, .5), (.5, 0, .5), (.5, .5, 0)]
    pos = base + [tuple(x + .25 for x in p) for p in base]
    return Structure(['Si'] * 8, pos, [[A, 0, 0], [0, A, 0], [0, 0, A]], True)


def slab():
    a, d, n, vac = 2.8638, 2.025, 5, 12.0
    c = (n - 1) * d + vac
    pos = [((i % 2) * .5, (i % 2) * .5, i * d / c) for i in range(n)]
    return Structure(['Al'] * n, pos, [[a, 0, 0], [0, a, 0], [0, 0, c]], True)


@unittest.skipUnless(HAVE_SYMMETRY, 'spglib/seekpath not installed (symmetry extra)')
class SymmetryTest(unittest.TestCase):
    def test_fcc_path_labels_and_breaks(self):
        from sparse_dft.symmetry import auto_band_path
        p = auto_band_path(silicon())
        self.assertEqual(p['spacegroup_international'], 'Fd-3m')
        self.assertEqual(p['segments'][0], ['GAMMA', 'X'])
        self.assertEqual(len(p['kpoints']), len(p['labels']))
        self.assertEqual(p['labels'][0], 'GAMMA')
        # X-U | K-Gamma: the U endpoint is immediately followed by K.
        named = [label for label in p['labels'] if label]
        self.assertIn(('U', 'K'), list(zip(named, named[1:])))
        self.assertFalse(p['is_supercell'])
        self.assertEqual(p['warnings'], [])

    def test_slab_keeps_only_in_plane_segments(self):
        from sparse_dft.symmetry import auto_band_path
        p = auto_band_path(slab())
        self.assertEqual(p['periodic_axes'], [True, True, False])
        self.assertTrue(all(abs(k[2]) < 1e-12 for k in p['kpoints']))
        self.assertTrue(p['dropped_segments'])
        full = auto_band_path(slab(), restrict_periodic=False)
        self.assertTrue(any(abs(k[2]) > 1e-6 for k in full['kpoints']))

    def test_molecule_has_no_path(self):
        from sparse_dft.symmetry import auto_band_path
        box = Structure(['C', 'O'], [(0, 0, 0), (0, 0, 0.113)], [[10, 0, 0], [0, 10, 0], [0, 0, 10]], True)
        with self.assertRaises(InputError):
            auto_band_path(box)

    def test_supercell_warning_and_primitive_export(self):
        from sparse_dft.symmetry import auto_band_path
        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / 'prim.vasp'
            p = auto_band_path(conventional_silicon(), export_primitive=target)
            self.assertTrue(p['is_supercell'])
            self.assertTrue(any('supercell' in w for w in p['warnings']))
            prim = read_poscar(target)
            self.assertEqual(prim.natoms, 2)
            self.assertAlmostEqual(prim.volume, conventional_silicon().volume / 4, places=8)
            sibling = json.loads(Path(p['primitive_band_path']).read_text())
            self.assertFalse(sibling['is_supercell'])

    def test_symprec_ladder_reports_hidden_symmetry(self):
        from sparse_dft.symmetry import choose_symprec
        exact = choose_symprec(silicon())
        self.assertEqual(exact['symprec'], 1e-5)
        self.assertEqual(exact['warnings'], [])
        # 0.03 A along x: below 0.1 A the cell still looks like Fd-3m.
        distorted = choose_symprec(silicon(shift=0.03 / A))
        self.assertLess(distorted['operations'], 48)
        self.assertEqual(distorted['loose_probe']['operations'], 48)
        self.assertTrue(any('close to Fd-3m' in w for w in distorted['warnings']))

    def test_wire_path_runs_along_the_periodic_axis(self):
        from sparse_dft.symmetry import auto_band_path
        chain = Structure(['C'], [(0, 0, 0)], [[12, 0, 0], [0, 12, 0], [0, 0, 1.28]], True)
        p = auto_band_path(chain)
        self.assertEqual(p['periodic_axes'], [False, False, True])
        self.assertTrue(all(abs(k[0]) < 1e-12 and abs(k[1]) < 1e-12 for k in p['kpoints']))
        self.assertAlmostEqual(max(k[2] for k in p['kpoints']), 0.5)


class PoscarWriterTest(unittest.TestCase):
    def test_round_trip_groups_species(self):
        s = Structure(['O', 'Mg', 'O'], [(0, 0, 0), (.5, .5, .5), (.25, .25, .25)],
                      [[4, 0, 0], [0, 4, 0], [0, 0, 4]], True)
        with tempfile.TemporaryDirectory() as d:
            back = read_poscar(write_poscar(s, Path(d) / 'POSCAR'))
        self.assertEqual(back.species, ['O', 'O', 'Mg'])
        self.assertEqual([tuple(round(x, 12) for x in p) for p in back.positions],
                         [(0, 0, 0), (.25, .25, .25), (.5, .5, .5)])


if __name__ == '__main__':
    unittest.main()
