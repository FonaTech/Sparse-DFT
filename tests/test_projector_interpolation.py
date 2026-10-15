"""Keep matched PAW projector physics across Python, CLI/API and frozen NSCF."""
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from sparse_dft.api import dispatch
from sparse_dft.cli import main
from sparse_dft.errors import InputError
from sparse_dft.scf import _write_input, run_bands, run_scf
from sparse_dft.structure import read_structure

ROOT = Path(__file__).resolve().parents[1]


class ProjectorInterpolationTests(unittest.TestCase):
    def test_input_and_early_validation(self):
        structure = read_structure(ROOT / 'examples/Si.POSCAR')
        choices = {'Si': {'cache': 'unused.bin', 'zval': 4}}
        with tempfile.TemporaryDirectory() as td:
            # Earlier vasp_* spellings are accepted aliases; inputs are written canonically.
            for selection, written in ((None, 'cubic'), ('cubic', 'cubic'), ('spline', 'spline'),
                                       ('vasp_cubic', 'cubic'), ('vasp_spline', 'spline')):
                path = Path(td) / 'input.sdft'
                _write_input(path, structure, choices, cutoff_ev=500, max_iterations=240,
                             density_tolerance=1e-8, onecenter_density_tolerance=1e-8,
                             mixing=.2, bands=8, projector_interpolation=selection)
                self.assertIn('projector_interpolation ' + written + '\n', path.read_text())
            for value in ('quadratic', True, ['spline']):
                with patch('sparse_dft.scf._resolve_caches') as resolve:
                    with self.assertRaises(InputError):
                        run_scf(structure, 'unused', td, projector_interpolation=value)
                    resolve.assert_not_called()

    def test_frozen_nscf_inherits_and_rejects_conflicts_before_kernel(self):
        choices = {'Si': {'cache': 'unused.bin', 'zval': 4, 'record_id': 'Si:standard'}}
        archive = SimpleNamespace(local_hashes=lambda: {})
        # Frozen results written before the rename carry the vasp_* spelling.
        for frozen_selection, stored in (('cubic', 'vasp_cubic'), ('spline', 'spline')):
            frozen = {'atomic_units': 'codata', 'density_cutoff_ev': 8000,
                      'projector_interpolation': stored,
                      'scf_kpoints': [{'fractional': [0, 0, 0], 'weight': 1}],
                      'onecenter_states': [{'symbol': 'Si', 'paw_record_id': 'Si:standard',
                                           'paw_cache': 'unused.bin'}]}

            def execute(command, **kwargs):
                self.assertEqual(command[1], 'nscf')
                self.assertIn('projector_interpolation ' + frozen_selection, Path(command[2]).read_text())
                Path(command[3]).write_text(json.dumps({'metadata': {'scf_converged': True}}))
                return SimpleNamespace(returncode=0, stdout='', stderr='')

            with tempfile.TemporaryDirectory() as td, \
                    patch('sparse_dft.scf._resolve_caches', return_value=(choices, {}, archive)), \
                    patch('sparse_dft.scf._executable', return_value=Path('unused_kernel')), \
                    patch('sparse_dft.scf.export_frozen_scf', return_value=(frozen, 'test_hash')), \
                    patch('sparse_dft.scf.subprocess.run', side_effect=execute) as kernel:
                args = (ROOT / 'examples/Si.POSCAR', 'unused', td, [(0, 0, 0), (.1, 0, 0)])
                for selection in (None, frozen_selection):
                    result = run_bands(*args, cutoff_ev=500, scf_checkpoint='test_state',
                                       projector_interpolation=selection)
                    self.assertEqual(result['provenance']['projector_interpolation'], frozen_selection)
                kernel.reset_mock()
                conflict = 'vasp_cubic' if frozen_selection == 'spline' else 'vasp_spline'
                with self.assertRaisesRegex(InputError, 'differs from the frozen SCF'):
                    run_bands(*args, cutoff_ev=500, scf_checkpoint='test_state',
                              projector_interpolation=conflict)
                kernel.assert_not_called()

    def test_cli_and_agent_transport_spline_for_scf_bands_and_relax(self):
        for command, function in (('scf', 'run_scf'), ('bands', 'run_bands'), ('relax', 'ionic_relax')):
            payload = {'converged': True, 'metadata': {'scf_converged': True}, 'status': 'CONVERGED'}
            with tempfile.TemporaryDirectory() as td:
                path = Path(td) / 'path.json'
                path.write_text('[[0,0,0],[0.1,0,0]]')
                args = [command, 'unused', '--paw-root', 'unused',
                        '--projector-interpolation', 'vasp_spline']  # accepted alias
                if command == 'bands':
                    args += ['--path', str(path)]
                with patch('sparse_dft.cli.' + function, return_value=payload) as compute, \
                        contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(main(args), 0)
                self.assertEqual(compute.call_args.kwargs['projector_interpolation'], 'spline')
                method = {'scf': 'job.run', 'bands': 'bands.run', 'relax': 'relax.ionic'}[command]
                with patch('sparse_dft.api.' + function, return_value=payload) as compute:
                    result = dispatch(method, {'structure': 'unused', 'paw_root': 'unused',
                                               'kpoints': [[0, 0, 0], [.1, 0, 0]],
                                               'projector_interpolation': 'spline'})
                self.assertEqual(result['status'], 'PASS')
                self.assertEqual(compute.call_args.kwargs['projector_interpolation'], 'spline')


if __name__ == '__main__':
    unittest.main()
