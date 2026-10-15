import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sparse_dft.backend import backend_info
from sparse_dft.relax import ionic_relax, _structure_from_dict, _json_hash
from sparse_dft.structure import Structure


class RelaxAndBackendTests(unittest.TestCase):
    def test_backend_report_is_explicit(self):
        info = backend_info()
        self.assertIn("cpu", info)
        self.assertIn("openmp", info)
        self.assertIn("cuda", info)
        self.assertIn("mps", info)
        self.assertEqual(info["cuda"]["active"], False)

    def test_openmp_report_has_runtime_probe_contract(self):
        info = backend_info()["openmp"]
        self.assertIn("probe_status", info)
        self.assertIn("team_threads", info)
        self.assertIn("active", info)
        if info["active"]:
            self.assertGreaterEqual(info["team_threads"], 2)

    def test_checkpoint_signature_rejects_changed_controls(self):
        s = Structure(["Si"], [(0.0, 0.0, 0.0)],
                      ((5.4, 0.0, 0.0), (0.0, 5.4, 0.0), (0.0, 0.0, 5.4)),
                      True, (True, True, True)).normalized()
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            # A valid structure is enough to exercise the restart guard; the
            # derivative engine must not be reached when the signature differs.
            checkpoint = out / "relax_checkpoint.json"
            checkpoint.write_text(json.dumps({"schema_version": "sparse-dft-relax-2",
                                               "signature": "stale", "step": 0,
                                               "history": [], "structure": s.as_dict()}))
            with self.assertRaises(Exception) as ctx:
                ionic_relax(s, "/paw", out, force_tolerance=0.01)
            self.assertIn("checkpoint parameters", str(ctx.exception))

    def test_ionic_relax_mocked_derivative_checkpoint(self):
        s = Structure(["Si", "Si"], [(0.0, 0.0, 0.0), (0.25, 0.25, 0.25)],
                      ((5.4, 0.0, 0.0), (0.0, 5.4, 0.0), (0.0, 0.0, 5.4)),
                      True, (True, True, True)).normalized()

        def fake_energy(structure, paw_root, workdir, **kwargs):
            # Translation-invariant quadratic pair proxy; deterministic and
            # converged so the orchestration/checkpoint path can be verified.
            x = structure.positions[1][0]
            return float((x - 0.25) ** 2), {"converged": True, "total_energy_ev": float((x - 0.25) ** 2),
                                           "structure_hash": structure.content_hash()}

        with tempfile.TemporaryDirectory() as td, patch("sparse_dft.relax._energy", side_effect=fake_energy):
            result = ionic_relax(s, "/paw", td, max_steps=1, max_workers=2,
                                 force_tolerance=1e-10, step_angstrom=1e-3,
                                 derivative_method='finite_difference')
            self.assertIn(result["status"], ("CONVERGED", "MAX_STEPS"))
            self.assertTrue((Path(td) / "relax_checkpoint.json").exists())
            doc = json.loads((Path(td) / "relax_checkpoint.json").read_text())
            self.assertEqual(doc["schema_version"], "sparse-dft-relax-2")
            self.assertIn("signature", doc)


if __name__ == "__main__":
    unittest.main()
