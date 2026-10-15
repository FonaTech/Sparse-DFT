import json
import tempfile
import unittest
import os
import hashlib
import math
from contextlib import contextmanager
from pathlib import Path

from sparse_dft.scf import run_scf, run_bands, _executable
from sparse_dft.bands import parse_vasprun_bands
from sparse_dft.checkpoint import export_frozen_scf
from sparse_dft.errors import InputError
from sparse_dft.resources import project_paw_archive


ROOT = Path(__file__).resolve().parents[1]
PAW = project_paw_archive()


@contextmanager
def scf_validation_directory():
    configured = os.environ.get("SDFT_SCF_VALIDATION_OUTPUT")
    if configured:
        output = Path(configured)
        output.mkdir(parents=True, exist_ok=True)
        yield str(output)
    else:
        with tempfile.TemporaryDirectory() as temporary:
            yield temporary


class CppNumericalTests(unittest.TestCase):
    def test_diagnostic_onecenter_dij_is_frozen_into_nscf(self):
        if not PAW.exists():
            self.skipTest("PAW54 archive is unavailable")
        previous = os.environ.get("SDFT_ENABLE_PAW_CLOSURE_DIAGNOSTIC")
        os.environ["SDFT_ENABLE_PAW_CLOSURE_DIAGNOSTIC"] = "1"
        try:
            with scf_validation_directory() as td:
                structure = ROOT / "examples" / "Si.POSCAR"
                opts = dict(cutoff_ev=100.0, max_iterations=480,
                            density_tolerance=1e-6, mixing=0.05, bands=8)
                scf = run_scf(structure, PAW, Path(td) / "scf", **opts)
                bands = run_bands(structure, PAW, Path(td) / "bands",
                                  [(0, 0, 0), (0.125, 0, 0)], **opts)
                self.assertTrue(scf["converged"])
                self.assertTrue(bands["metadata"]["scf_converged"])
                self.assertFalse(scf["total_energy_valid"])
                self.assertFalse(bands["metadata"]["total_energy_valid"])
                self.assertEqual(scf["paw_spherical_dij_norm_ev"], 0.0)
                self.assertNotIn("paw_chf_ev", scf)
                self.assertTrue(scf['stress_provenance']['analytic'])
                self.assertEqual(set(scf['stress_components_ev_per_angstrom3']), {
                    'kinetic', 'hartree', 'xc', 'nlcc', 'local', 'local_g0',
                    'augmentation', 'nonlocal', 'ewald'})
                for i in range(3):
                    for j in range(3):
                        ledger_sum=sum(t[i][j] for t in scf['stress_components_ev_per_angstrom3'].values())
                        self.assertAlmostEqual(ledger_sum, scf['hf_total_stress_ev_per_angstrom3'][i][j], delta=1e-12)
                self.assertTrue(math.isfinite(scf["integrated_electron_count"]))
                self.assertTrue(math.isfinite(scf["compensation_electron_count"]))
                self.assertEqual(scf["integrated_electron_count"], bands["metadata"]["scf_integrated_electron_count"])
                self.assertEqual(scf["compensation_electron_count"], bands["metadata"]["scf_compensation_electron_count"])
                self.assertEqual(scf["onecenter_mixing_method"], "joint_regularized_multisecant")
                self.assertEqual(len(scf["onecenter_density_residuals"]), 2)
                self.assertAlmostEqual(scf["onecenter_density_residual"],
                                       max(scf["onecenter_density_residuals"]), places=14)
                self.assertLess(scf["onecenter_density_residual"],
                                scf["onecenter_density_tolerance"])
                self.assertEqual(len(scf["onecenter_states"]), 2)
                self.assertTrue(bands["metadata"]["onecenter_dij_reused"])
                self.assertEqual(bands["metadata"]["onecenter_dij_applications"], 4)
                embedded_scf = json.loads((Path(td) / "bands" / "scf.json").read_text())
                self.assertEqual(embedded_scf["onecenter_states"], bands["metadata"]["onecenter_states"])
                for scf_state, nscf_state in zip(scf["onecenter_states"], bands["metadata"]["onecenter_states"]):
                    # Separate runs restore byte-identical PAW records under
                    # different temporary directories.
                    self.assertNotEqual(scf_state["paw_cache"], nscf_state["paw_cache"])
                    for key in scf_state.keys() - {"paw_cache"}:
                        self.assertEqual(scf_state[key], nscf_state[key])
                self.assertEqual(scf["onecenter_provenance"]["validation_status"], "diagnostic_only")
                for state in scf["onecenter_states"]:
                    dim = state["projector_dimension"]
                    self.assertGreater(dim, 0)
                    for old in ("crhode", "cdij_ev", "rholm_imag_vasp"):
                        self.assertNotIn(old, state)
                    self.assertIsInstance(state["lm_moment_imag"], list)
                    self.assertEqual(state["density_source"], "final_scf_hamiltonian_input_onsite_occupation")
                    for key in ("onsite_occupation", "onsite_dij_ev", "closure_correction_ev", "compensation_correction_ev"):
                        self.assertEqual(len(state[key]), dim)
                        self.assertTrue(all(len(row) == dim for row in state[key]))
                        self.assertTrue(all(len(value) == 2 for row in state[key] for value in row))
                    self.assertEqual(state["paw_record_id"], "Si:standard")
                    self.assertEqual(state["numeric_cache_sha256"],
                                     hashlib.sha256(Path(state["paw_cache"]).read_bytes()).hexdigest())
                    self.assertEqual(state["validation_status"], "diagnostic_only")
                for scf_ev, gamma_ev in zip(scf["eigenvalues_ev"], bands["eigenvalues"][0]):
                    self.assertAlmostEqual(scf_ev, gamma_ev, delta=1e-8)
                replay = run_bands(structure, PAW, Path(td) / 'replay',
                                   [(0, 0, 0), (0.125, 0, 0)],
                                   scf_checkpoint=Path(td) / 'scf' / 'scf_result.json', **opts)
                for expected, actual in zip(scf['eigenvalues_ev'], replay['eigenvalues'][0]):
                    self.assertAlmostEqual(expected, actual, delta=1e-8)
                self.assertTrue(replay['metadata']['onecenter_dij_reused'])
                self.assertEqual(replay['metadata']['density_cutoff_ev'], scf['density_cutoff_ev'])
                self.assertEqual(replay['provenance']['density_cutoff_ev'], scf['density_cutoff_ev'])
                payload = Path(td) / 'replay' / 'frozen_scf.sdft'
                self.assertTrue(payload.read_text().startswith('SDFT_FROZEN_SCF_V3\n'))
                self.assertIn(scf['onecenter_radial_mode'], payload.read_text().splitlines())
                with self.assertRaisesRegex(InputError, 'density_cutoff_ev differs'):
                    export_frozen_scf(Path(td) / 'scf' / 'scf_result.json',
                                      Path(td) / 'mismatch.sdft', density_cutoff_ev=scf['density_cutoff_ev']+1)
                self.assertFalse((Path(td) / 'mismatch.sdft').exists())
                legacy = dict(scf); legacy.pop('density_cutoff_ev'); legacy['provenance'] = {'cutoff_ev':100.}
                (Path(td) / 'missing_density_cutoff.json').write_text(json.dumps(legacy))
                with self.assertRaisesRegex(InputError, 'density_cutoff_ev is missing'):
                    export_frozen_scf(Path(td) / 'missing_density_cutoff.json',
                                      Path(td) / 'missing.sdft', cutoff_ev=100.)
                # Result files written before the key rename must export the
                # same frozen payload, byte for byte.
                old_names = {'onsite_occupation': 'crhode', 'onsite_dij_ev': 'cdij_ev',
                             'lm_moment_imag': 'rholm_imag_vasp'}
                old = json.loads((Path(td) / 'scf' / 'scf_result.json').read_text())
                old['onecenter_states'] = [{old_names.get(k, k): v for k, v in a.items()}
                                           for a in old['onecenter_states']]
                old['onecenter_radial_mode'] = 'vasp_' + old['onecenter_radial_mode']
                old['projector_interpolation'] = 'vasp_' + old['projector_interpolation']
                (Path(td) / 'old_keys.json').write_text(json.dumps(old))
                _, new_digest = export_frozen_scf(Path(td) / 'scf' / 'scf_result.json', Path(td) / 'new_keys.sdft')
                _, old_digest = export_frozen_scf(Path(td) / 'old_keys.json', Path(td) / 'old_keys.sdft')
                self.assertEqual(old_digest, new_digest)
                # Native callers must enforce the same gate independently of
                # the Python exporter, with an otherwise valid signed payload.
                import subprocess
                mismatch_input = Path(td) / 'mismatch_input.sdft'
                mismatch_input.write_text((Path(td) / 'replay' / 'bands_input.sdft').read_text() +
                                           f"density_cutoff_ev {scf['density_cutoff_ev']+1}\n")
                rejected = subprocess.run([str(_executable()), 'nscf',
                    str(mismatch_input), str(Path(td) / 'density_rejected.json'), str(payload)],
                    text=True, capture_output=True)
                self.assertEqual(rejected.returncode, 2)
                self.assertIn('density_cutoff_ev', rejected.stderr)
                self.assertFalse((Path(td) / 'density_rejected.json').exists())
                # A corrupted payload is rejected by the native SHA-256
                # gate, before any frozen density enters a Hamiltonian.
                payload = Path(td) / 'replay' / 'frozen_scf.sdft'
                payload.write_text(payload.read_text() + 'corrupt\n')
                process = subprocess.run([str(_executable()), 'nscf',
                    str(Path(td) / 'replay' / 'bands_input.sdft'),
                    str(Path(td) / 'rejected.json'), str(payload)],
                    text=True, capture_output=True)
                self.assertEqual(process.returncode, 2)
                self.assertIn('SHA-256', process.stderr)
                self.assertFalse((Path(td) / 'rejected.json').exists())
        finally:
            if previous is None:
                os.environ.pop("SDFT_ENABLE_PAW_CLOSURE_DIAGNOSTIC", None)
            else:
                os.environ["SDFT_ENABLE_PAW_CLOSURE_DIAGNOSTIC"] = previous

    def test_onecenter_residual_is_a_convergence_gate(self):
        if not PAW.exists():
            self.skipTest("PAW54 archive is unavailable")
        previous = os.environ.get("SDFT_ENABLE_PAW_CLOSURE_DIAGNOSTIC")
        os.environ["SDFT_ENABLE_PAW_CLOSURE_DIAGNOSTIC"] = "1"
        try:
            with tempfile.TemporaryDirectory() as td:
                result = run_scf(ROOT / "examples" / "Si.POSCAR", PAW, Path(td) / "scf",
                                 cutoff_ev=30.0, max_iterations=240,
                                 density_tolerance=1e-2, onecenter_density_tolerance=1e-20,
                                 mixing=0.05, bands=8)
                self.assertLess(result["density_residual"], 1e-2)
                self.assertGreater(result["onecenter_density_residual"], 1e-20)
                self.assertFalse(result["converged"])
                self.assertFalse(result["total_energy_valid"])
        finally:
            if previous is None:
                os.environ.pop("SDFT_ENABLE_PAW_CLOSURE_DIAGNOSTIC", None)
            else:
                os.environ["SDFT_ENABLE_PAW_CLOSURE_DIAGNOSTIC"] = previous

    def test_gamma_scf_and_nscf_json(self):
        with tempfile.TemporaryDirectory() as td:
            result = run_scf(ROOT / "examples" / "Si.POSCAR", PAW, Path(td) / "scf",
                             # A force translation invariant requires a stationary coupled
                             # density, unlike the loose-outer interface smoke request.
                             cutoff_ev=30.0, max_iterations=120, density_tolerance=1e-10,
                             onecenter_density_tolerance=1e-10, mixing=0.3, bands=8)
            self.assertTrue(result["converged"])
            self.assertLessEqual(result["eigensolver_max_residual_hartree"], 1e-12)
            self.assertEqual(result["electron_count"], 8)
            self.assertGreaterEqual(result["hartree_energy_ev"], 0.0)
            self.assertTrue(abs(result["vxc_density_ev"]) < 1e6)
            self.assertTrue(all(abs(x) < 1e6 for x in result["eigenvalues_ev"]))
            # The C++ path now exposes an electronic HF diagnostic.  It is
            # deliberately marked unvalidated because Ewald and full PAW
            # one-centre terms are still absent from this contraction.
            self.assertEqual(len(result["hf_electronic_forces_ev_per_angstrom"]), 2)
            self.assertFalse(result["force_provenance"]["validated"])
            self.assertLess(max(abs(sum(row[a] for row in result["hf_electronic_forces_ev_per_angstrom"])) for a in range(3)), 1e-8)

            # Disabling both ionic-electron operator pieces must also disable
            # the analytic diagnostic; this catches stale or double-counted
            # derivative paths.
            old_closure = os.environ.get("SDFT_ENABLE_PAW_CLOSURE_DIAGNOSTIC")
            old_local, old_nonlocal = os.environ.get("SDFT_DISABLE_LOCAL"), os.environ.get("SDFT_DISABLE_NONLOCAL")
            try:
                os.environ["SDFT_ENABLE_PAW_CLOSURE_DIAGNOSTIC"] = "0"
                os.environ["SDFT_DISABLE_LOCAL"] = "1"
                os.environ["SDFT_DISABLE_NONLOCAL"] = "1"
                disabled = run_scf(ROOT / "examples" / "Si.POSCAR", PAW, Path(td) / "disabled",
                                   cutoff_ev=30.0, max_iterations=60, density_tolerance=1e-2, mixing=0.3, bands=8)
                self.assertTrue(all(abs(v) < 1e-12 for row in disabled["hf_electronic_forces_ev_per_angstrom"] for v in row))
            finally:
                if old_closure is None: os.environ.pop("SDFT_ENABLE_PAW_CLOSURE_DIAGNOSTIC", None)
                else: os.environ["SDFT_ENABLE_PAW_CLOSURE_DIAGNOSTIC"] = old_closure
                if old_local is None: os.environ.pop("SDFT_DISABLE_LOCAL", None)
                else: os.environ["SDFT_DISABLE_LOCAL"] = old_local
                if old_nonlocal is None: os.environ.pop("SDFT_DISABLE_NONLOCAL", None)
                else: os.environ["SDFT_DISABLE_NONLOCAL"] = old_nonlocal
            bands = run_bands(ROOT / "examples" / "Si.POSCAR", PAW, Path(td) / "bands",
                              [(0, 0, 0), (0.125, 0, 0), (0.25, 0, 0)], cutoff_ev=30.0,
                              max_iterations=120, density_tolerance=1e-2, mixing=0.3, bands=8)
            self.assertEqual(len(bands["kpoints"]), 3)
            self.assertEqual(len(bands["eigenvalues"]), 3)

    def test_vasprun_real_eigenvalue_layout(self):
        sample = ROOT / "tests/fixtures/vasp/si8_gamma/vasprun.xml"
        if not sample.exists():
            self.fail("bundled VASP regression output is unavailable")
        bands = parse_vasprun_bands(sample)
        self.assertEqual(bands.nkpoints, 1)
        self.assertGreater(bands.nbands, 10)
        self.assertAlmostEqual(bands.eigenvalues[0][0], -5.8843, places=4)


if __name__ == "__main__":
    unittest.main()
