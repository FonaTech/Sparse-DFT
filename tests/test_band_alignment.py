"""Energy zero for comparing two band structures.

With smearing, E_F of a gapped system is the root where electrons above the
gap balance holes below it, so two codes can place it at different points of
the same gap. A comparison must not turn that offset into a rigid band shift.
"""
import unittest

from sparse_dft.bands import band_zero, common_energy_zero, compare_bands, from_payload, sampled_band_split
from sparse_dft.errors import InputError

KPTS = [[0, 0, 0], [0.25, 0, 0], [0.5, 0, 0]]
GAPPED = [[-2.0, -1.0, 3.0, 5.0], [-1.8, -0.5, 3.4, 5.2], [-1.5, -0.2, 3.1, 5.5]]  # VBM -0.2, CBM 3.0
METAL = [[-2.0, -0.4, 3.0], [-1.8, 0.1, 3.4], [-1.5, 0.6, 3.1]]                     # band 2 crosses 0


def bands(eigenvalues, fermi, offset=0.0, **extra):
    return from_payload({'kpoints': KPTS, 'fermi_energy': fermi + offset,
                         'eigenvalues': [[e + offset for e in row] for row in eigenvalues], **extra},
                        source='synthetic')


class BandAlignmentTests(unittest.TestCase):
    def test_gapped_pair_with_different_in_gap_fermi_levels(self):
        # Same spectrum; one code puts E_F at mid-gap, the other 0.58 eV lower.
        ref, cand = bands(GAPPED, 1.4), bands(GAPPED, 0.82)
        auto = compare_bands(ref, cand, align='auto')
        self.assertEqual(auto['alignment'], 'vbm')
        self.assertAlmostEqual(auto['reference_zero_ev'], -0.2, places=12)
        self.assertAlmostEqual(auto['candidate_zero_ev'], -0.2, places=12)
        self.assertLess(auto['max_abs_error_ev'], 1e-12)
        # Aligning each own E_F reproduces the reported drift as a rigid shift.
        fermi = compare_bands(ref, cand, align='fermi')
        self.assertAlmostEqual(fermi['max_abs_error_ev'], 0.58, places=12)
        self.assertAlmostEqual(fermi['mae_ev'], 0.58, places=12)

    def test_auto_removes_a_true_potential_offset(self):
        # A constant potential shift moves bands and E_F together; VBM alignment removes it.
        ref, cand = bands(GAPPED, 1.4), bands(GAPPED, 1.1, offset=0.37)
        result = compare_bands(ref, cand, align='auto')
        self.assertAlmostEqual(result['energy_shift_ev'], -0.37, places=12)
        self.assertLess(result['max_abs_error_ev'], 1e-12)

    def test_metal_falls_back_to_each_fermi_energy(self):
        ref, cand = bands(METAL, 0.0), bands(METAL, 0.0, offset=0.25)
        zero = common_energy_zero(ref, cand)
        self.assertEqual(zero['kind'], 'fermi')
        self.assertTrue(any('crosses E_F' in r for r in zero['fallback_reasons']))
        result = compare_bands(ref, cand, align='auto')
        self.assertAlmostEqual(result['energy_shift_ev'], -0.25, places=12)
        self.assertLess(result['max_abs_error_ev'], 1e-12)

    def test_vbm_on_a_metal_raises_instead_of_comparing_raw_energies(self):
        with self.assertRaises(InputError):
            compare_bands(bands(METAL, 0.0), bands(METAL, 0.0, offset=0.25), align='vbm')

    def test_unknown_alignment_is_rejected(self):
        with self.assertRaises(InputError):
            band_zero(bands(GAPPED, 1.4), bands(GAPPED, 1.4), align='cbm')

    def test_different_occupied_counts_do_not_use_vbm(self):
        # E_F above band 3 in the candidate: a different filling, not the same gap.
        zero = common_energy_zero(bands(GAPPED, 1.4), bands(GAPPED, 4.0))
        self.assertEqual(zero['kind'], 'fermi')
        self.assertIn('occupied band counts per spin channel differ', zero['fallback_reasons'])

    def test_spin_channels_share_one_zero(self):
        # The VBM sits in the down channel; both channels use it.
        up = [[-2.0, -1.0, 3.0], [-1.8, -0.6, 3.4], [-1.5, -0.4, 3.1]]
        down = [[-2.1, -0.9, 2.6], [-1.7, -0.1, 2.9], [-1.6, -0.3, 2.7]]
        b = from_payload({'kpoints': KPTS, 'fermi_energy': 1.0, 'eigenvalues': up,
                          'spin_channels': {'up': up, 'down': down}}, source='spin')
        split = sampled_band_split(b)
        self.assertTrue(split['gapped'])
        self.assertEqual(split['occupied_bands'], {'up': 2, 'down': 2})
        self.assertAlmostEqual(split['vbm_ev'], -0.1, places=12)
        self.assertAlmostEqual(split['cbm_ev'], 2.6, places=12)
        self.assertAlmostEqual(band_zero(b, b, 'vbm')['reference_ev'], -0.1, places=12)

    def test_single_channel_name_does_not_block_vbm(self):
        # VASP ISPIN=1 output is keyed 'up'; a Sparse-DFT non-spin path has no channels.
        ref = bands(GAPPED, 0.9, spin_channels={'up': GAPPED})
        self.assertEqual(common_energy_zero(ref, bands(GAPPED, 1.4))['kind'], 'vbm')

    def test_auto_refuses_a_refilled_nscf_fermi_energy_for_metals(self):
        # A fixed-density path run refills E_F on its own k list (Al anchor: 7.640 vs 7.983 eV).
        ref = bands(METAL, -0.34, metadata={'fermi_energy_scope': 'nscf_k_list'})
        with self.assertRaises(InputError):
            compare_bands(ref, bands(METAL, 0.0), align='auto')
        self.assertEqual(compare_bands(ref, bands(METAL, 0.0), align='none')['max_abs_error_ev'], 0.0)

    def test_none_keeps_raw_energies_and_the_old_window(self):
        ref, cand = bands(GAPPED, 1.4), bands(GAPPED, 0.82, offset=0.1)
        result = compare_bands(ref, cand, align='none')
        self.assertEqual(result['energy_shift_ev'], 0.0)
        self.assertEqual(result['window_centre_ev'], 1.4)
        self.assertAlmostEqual(result['max_abs_error_ev'], 0.1, places=12)


if __name__ == '__main__':
    unittest.main()
