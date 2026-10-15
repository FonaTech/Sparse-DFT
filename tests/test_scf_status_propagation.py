"""The Python boundary must preserve numerical and physical validation separately."""
import json
import contextlib
import io
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from sparse_dft.scf import run_bands, run_scf
from sparse_dft.cli import main
from sparse_dft.api import dispatch


ROOT = Path(__file__).resolve().parents[1]


class ScfStatusPropagationTests(unittest.TestCase):
    def _run(self, *, bands, converged):
        validity = {
            "total_energy_valid": False,
            "onecenter_closure_diagnostic": True,
            "onecenter_provenance": {
                "validation_status": "diagnostic_only",
                "double_counting_validated": False,
                "analytic_derivatives_validated": False,
            },
        }
        payload = {"metadata": {"scf_converged": converged, **validity}} if bands else {
            "converged": converged, **validity,
        }

        def execute(command, **kwargs):
            Path(command[-1]).write_text(json.dumps(payload))
            return SimpleNamespace(returncode=0 if converged else 3, stdout="", stderr="")

        choices = {"Si": {"cache": "unused_numeric.bin", "zval": 4, "enmax_ev": 245.345}}
        archive = SimpleNamespace(local_hashes=lambda: {"manifest": "test_manifest_hash"})
        with tempfile.TemporaryDirectory() as td, \
                patch("sparse_dft.scf._resolve_caches", return_value=(choices, {}, archive)), \
                patch("sparse_dft.scf._executable", return_value=Path("unused_sparse_dft_scf")), \
                patch("sparse_dft.scf.subprocess.run", side_effect=execute):
            args = [ROOT / "examples" / "Si.POSCAR", "unused_archive", td]
            if bands:
                result = run_bands(*args, [(0., 0., 0.), (.1, 0., 0.)], cutoff_ev=100.)
                propagated = result["metadata"]
                self.assertEqual(propagated["scf_converged"], converged)
                persisted = json.loads((Path(td) / "bands_result.json").read_text())["metadata"]
            else:
                result = run_scf(*args, cutoff_ev=100.)
                propagated = result
                self.assertEqual(propagated["converged"], converged)
                persisted = json.loads((Path(td) / "scf_result.json").read_text())
            for key, expected in validity.items():
                self.assertEqual(propagated[key], expected)
                self.assertEqual(persisted[key], expected)

    def test_diagnostic_validation_is_not_promoted_by_convergence(self):
        for bands in (False, True):
            for converged in (False, True):
                with self.subTest(bands=bands, converged=converged):
                    self._run(bands=bands, converged=converged)

    def test_bands_cli_uses_explicit_scf_convergence(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "path.json"
            path.write_text(json.dumps([[0., 0., 0.], [.1, 0., 0.]]))
            payload = {"metadata": {"density_source": "converged_scf", "scf_converged": False,
                                    "total_energy_valid": False}}
            with patch("sparse_dft.cli.run_bands", return_value=payload) as calculate, \
                    contextlib.redirect_stdout(io.StringIO()):
                exit_code = main(["bands", "unused.POSCAR", "--paw-root", "unused_archive",
                                  "--path", str(path), "--onecenter-density-tolerance", "2e-7",
                                  "--density-cutoff", "8000"])
            self.assertEqual(exit_code, 3)
            self.assertEqual(calculate.call_args.kwargs["onecenter_density_tolerance"], 2e-7)
            self.assertEqual(calculate.call_args.kwargs["density_cutoff_ev"], 8000.)

    def test_agent_api_preserves_convergence_and_validity(self):
        for method, function in (("job.run", "run_scf"), ("bands.run", "run_bands")):
            for converged in (False, True):
                with self.subTest(method=method, converged=converged):
                    status = {"scf_converged": converged, "total_energy_valid": False}
                    payload = {"metadata": status} if method == "bands.run" else {
                        "converged": converged, "total_energy_valid": False}
                    with patch("sparse_dft.api." + function, return_value=payload) as calculate:
                        result = dispatch(method, {"structure": "unused.POSCAR", "paw_root": "unused",
                                                   "kpoints": [[0., 0., 0.]],
                                                   "onecenter_density_tolerance": 2e-7,
                                                   "density_cutoff_ev": 8000.})
                    self.assertEqual(result["status"], "PASS" if converged else "FAILED")
                    self.assertEqual(result["data"], payload)
                    self.assertEqual(calculate.call_args.kwargs["onecenter_density_tolerance"], 2e-7)
                    self.assertEqual(calculate.call_args.kwargs["density_cutoff_ev"], 8000.)


if __name__ == "__main__":
    unittest.main()
