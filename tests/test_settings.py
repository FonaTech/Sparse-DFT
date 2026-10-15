import unittest

from sparse_dft.errors import InputError
from sparse_dft.kpoints import validate_kpoint_grid, validate_kpoint_path, interpolate_band_path
from sparse_dft.settings import (
    DEFAULT_CUTOFF_EV,
    automatic_cutoff,
    validate_cutoff,
    validate_density_cutoff,
    validate_max_iterations,
    validate_timeout,
)


class SettingsValidationTests(unittest.TestCase):
    def test_automatic_cutoff_tracks_paw_enmax(self):
        self.assertEqual(automatic_cutoff([400.0], "standard"), 520.0)
        self.assertEqual(automatic_cutoff([], "standard"), DEFAULT_CUTOFF_EV)

    def test_cutoff_and_iterations_reject_invalid_values(self):
        for value in (0.0, -1.0, 10_000.0):
            with self.assertRaises(InputError):
                validate_cutoff(value)
        for value in (0, -2, 100_001, 2.5):
            with self.assertRaises(InputError):
                validate_max_iterations(value)

    def test_kpoint_limits(self):
        self.assertEqual(validate_kpoint_grid((4, 4, 4)), (4, 4, 4))
        with self.assertRaises(InputError):
            validate_kpoint_grid((0, 1, 1))
        self.assertEqual(len(validate_kpoint_path([(0, 0, 0), (0.5, 0, 0)])), 2)
        with self.assertRaises(InputError):
            validate_kpoint_path([(0, 0, 0)])

    def test_band_path_interpolation_matches_vasp_line_mode(self):
        path = interpolate_band_path([(0, 0, 0), (0.25, 0, 0), (0.25, 0.25, 0)], 81)
        self.assertEqual(len(path), 162)
        self.assertEqual(path[0], (0.0, 0.0, 0.0))
        self.assertEqual(path[80], (0.25, 0.0, 0.0))
        self.assertEqual(path[81], (0.25, 0.0, 0.0))
        self.assertEqual(path[-1], (0.25, 0.25, 0.0))
        self.assertEqual(len(interpolate_band_path([(0, 0, 0), (0.25, 0, 0), (0.25, 0.25, 0)], 81,
                                                    include_duplicate_vertices=True)), 162)

    def test_network_timeout_none_is_unlimited(self):
        self.assertIsNone(validate_timeout(None))
        with self.assertRaises(InputError):
            validate_timeout(0)

    def test_density_cutoff_is_independent_and_cannot_clip_smooth_products(self):
        self.assertIsNone(validate_density_cutoff(None, 500.))
        self.assertEqual(validate_density_cutoff(8000., 500.), 8000.)
        self.assertEqual(validate_density_cutoff(16000., 500.), 16000.)
        for value in (1999., 0., float('nan'), float('inf')):
            with self.assertRaises(InputError):
                validate_density_cutoff(value, 500.)


if __name__ == "__main__":
    unittest.main()
