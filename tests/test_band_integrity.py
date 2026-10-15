import json
import tempfile
import unittest
from pathlib import Path

from sparse_dft.bands import compare_bands, from_json, parse_doscar, parse_eigenval, parse_vasprun_bands
from sparse_dft.errors import InputError


class BandFormatIntegrityTests(unittest.TestCase):
    def test_spin_eigenval_preserves_energy_and_occupancy_columns(self):
        text = """  2  2  2  2
 header
 header
 header
 header
       4       2       2
 0.000000 0.000000 0.000000 0.5
 1  -1.0  -0.5   1.0  1.0
 2   0.1   0.2   0.0  0.5

 0.500000 0.000000 0.000000 0.5
 1  -0.9  -0.4   1.0  1.0
 2   0.2   0.3   0.0  0.0
"""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "EIGENVAL"
            p.write_text(text)
            bands = parse_eigenval(p, fermi_energy=0.0)
        self.assertEqual(bands.spin_channels["down"][0][0], -0.5)
        self.assertEqual(bands.occupations["down"][0][1], 0.5)
        self.assertEqual(bands.gap_status, "metal")
        self.assertIsNone(bands.band_gap)

    def test_doscar_spin_columns_are_not_shifted(self):
        lines = ["h"] * 5 + ["  10.0  -10.0  2  0.0  0.0  0.0"]
        lines += [" -1.0  2.0  3.0  4.0  5.0", " 1.0  6.0  7.0  8.0  9.0"]
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "DOSCAR"
            p.write_text("\n".join(lines) + "\n")
            dos = parse_doscar(p)
        self.assertEqual(dos["total"]["total_up"], [2.0, 6.0])
        self.assertEqual(dos["total"]["total_down"], [3.0, 7.0])
        self.assertEqual(dos["total"]["integrated_up"], [4.0, 8.0])
        self.assertTrue(dos["metadata"]["spin_polarized"])

    def test_vasprun_uses_final_calculation_and_single_2pi_basis(self):
        xml = """<modeling>
 <kpoints><varray name="kpointlist"><v>0 0 0</v><v>0.5 0 0</v></varray><varray name="weights"><v>0.5</v><v>0.5</v></varray></kpoints>
 <structure name="initialpos"><crystal><varray name="rec_basis"><v>1 0 0</v><v>0 1 0</v><v>0 0 1</v></varray></crystal></structure>
 <structure name="finalpos"><crystal><varray name="rec_basis"><v>2 0 0</v><v>0 2 0</v><v>0 0 2</v></varray></crystal></structure>
 <calculation><scstep><i name="efermi">99.0</i></scstep><eigenvalues><array><set><set comment="spin 1"><set comment="kpoint 1"><r>-9 1</r><r>0 0</r></set><set comment="kpoint 2"><r>-8 1</r><r>1 0</r></set></set></set></array></eigenvalues></calculation>
 <calculation><scstep><i name="efermi">0.0</i></scstep><eigenvalues><array><set><set comment="spin 1"><set comment="kpoint 1"><r>-2 1</r><r>2 0</r></set><set comment="kpoint 2"><r>-1 1</r><r>3 0</r></set></set><set comment="spin 2"><set comment="kpoint 1"><r>-3 1</r><r>3 0</r></set><set comment="kpoint 2"><r>-2 1</r><r>4 0</r></set></set></set></array></eigenvalues></calculation>
</modeling>"""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "vasprun.xml"
            p.write_text(xml)
            bands = parse_vasprun_bands(p)
        self.assertEqual(bands.eigenvalues[0], [-2.0, 2.0])
        self.assertEqual(bands.spin_channels["down"][1], [-2.0, 4.0])
        self.assertEqual(bands.fermi_energy, 0.0)
        self.assertAlmostEqual(bands.distances[-1], 2.0 * 3.141592653589793)
        self.assertEqual(bands.metadata["reciprocal_basis_includes_2pi"], True)

    def test_geometry_mismatch_is_rejected_even_with_same_length(self):
        with tempfile.TemporaryDirectory() as td:
            a = Path(td) / "a.json"
            b = Path(td) / "b.json"
            payload = {"eigenvalues": [[0.0, 1.0]] * 3, "fermi_energy": 0.5}
            a.write_text(json.dumps({**payload, "kpoints": [[0, 0, 0], [1, 0, 0], [1, 1, 0]]}))
            b.write_text(json.dumps({**payload, "kpoints": [[0, 0, 0], [0, 1, 0], [1, 1, 0]]}))
            with self.assertRaises(InputError):
                compare_bands(from_json(a), from_json(b))


if __name__ == "__main__":
    unittest.main()
