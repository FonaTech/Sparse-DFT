import math
import unittest
from sparse_dft.scf import _scf_sampling
from sparse_dft.kpoints import uniform_kpoint_mesh
from sparse_dft.errors import InputError


class ScfSamplingTests(unittest.TestCase):
    def test_mp_and_gamma_have_correct_even_grid_shift(self):
        gamma, weights = uniform_kpoint_mesh((2, 2, 2))
        self.assertEqual(set(x[0] for x in gamma), {0.0, -0.5})
        mp, _ = uniform_kpoint_mesh((2, 2, 2), mode='monkhorst-pack')
        self.assertEqual(set(x[0] for x in mp), {-0.25, 0.25})
        self.assertEqual(len(mp), 8)
        self.assertAlmostEqual(sum(weights), 1.0)

    def test_weights_and_points_are_checked_together(self):
        pts, weights = _scf_sampling((0, 0, 0), [(0, 0, 0), (.1, 0, 0)], [1, 3])
        self.assertEqual(weights, [.25, .75])
        for invalid in ([1], [1, 0], [1, -1], [1, math.nan]):
            with self.assertRaises(InputError):
                _scf_sampling((0, 0, 0), pts, invalid)
        with self.assertRaises(InputError):
            _scf_sampling((0, 0, 0), pts, grid=(2, 2, 2))
