"""Behavior tests with real source interfaces and explicitly synthetic fixtures."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SKILL = Path(os.environ.get('SDFT_TEST_SKILL', str(ROOT / 'skills/sparse-dft-driver'))).resolve()
SCRIPTS = SKILL / 'scripts'
sys.path.insert(0, str(SCRIPTS))
from source_catalog import source_catalog, sha256, verify_manifest
from research_bridge import scaffold, handoff
from case_studies import library, materialize
from sparse_dft_driver import monitor
from eos_workflow import prepare as eos_prepare, analyze as eos_analyze

SOURCE = Path(os.environ['SDFT_TEST_SOURCE']).resolve()
UPSTREAM_DFT = Path(os.environ.get('SDFT_TEST_DFT_SKILLS', str(Path.home() / '.codex'))).resolve()
UPSTREAM_WRITING = Path(os.environ.get('SDFT_TEST_WRITING_SKILLS', str(Path.home() / '.codex'))).resolve()


class DriverBehavior(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='sparse-dft-driver-test-')
        self.project = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def params(self, value, name='request.json'):
        file = self.project / name
        file.write_text(json.dumps(value))
        return str(file)

    def drive(self, *args):
        return subprocess.run([sys.executable, str(SCRIPTS / 'sparse_dft_driver.py'), '--source', str(SOURCE), '--cwd', str(self.project), *args], capture_output=True, text=True)

    def test_source_manifest_and_interface_catalog(self):
        self.assertEqual(verify_manifest(SOURCE)['status'], 'UNAVAILABLE')
        actual = source_catalog(SOURCE)
        frozen = json.loads((SKILL / 'references/capability-catalog.json').read_text())
        self.assertEqual(actual, frozen)
        self.assertEqual(len(actual['cli']), 22)
        self.assertEqual(len(actual['api']), 21)

    def test_every_cli_entry_accepts_actual_help(self):
        for name in source_catalog(SOURCE)['cli']:
            with self.subTest(command=name):
                result = self.drive('cli', '--', name, '--help')
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('usage:', result.stdout)

    def test_real_structure_api_records_process_and_request(self):
        file = self.params({'path': str(SOURCE / 'examples/Si.POSCAR')})
        result = self.drive('api', 'structure.validate', '--params', file)
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data['status'], 'PASS')
        self.assertEqual(data['data']['natoms'], 2)
        records = list((self.project / 'workflow/sparse-dft-driver/runs').glob('*/run.json'))
        self.assertEqual(len(records), 1)
        record = json.loads(records[0].read_text())
        self.assertEqual(record['returncode'], 0)
        self.assertFalse(record['production_validated'])
        self.assertFalse(list(self.project.glob('.sparse-dft-request-*')))

    def test_python_spacing_and_structure_object_are_usable(self):
        file = self.params({'args': [{'$structure': str(SOURCE / 'examples/Si.POSCAR')}], 'kwargs': {'task': 'scf', 'accuracy': 'high', 'spacing_inverse_angstrom': .2, 'metallic': False}})
        result = self.drive('python', 'sparse_dft.kpoints:recommend_kpoints', '--params', file)
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertTrue(data['grid'])

    def test_real_path_generation_has_expected_endpoints(self):
        vertices = self.params([[0, 0, 0], [.5, 0, .5]], 'vertices.json')
        result = self.drive('cli', '--', 'make-band-path', vertices, '--points-per-segment', '5', '--out', 'path.json')
        self.assertEqual(result.returncode, 0, result.stderr)
        points = json.loads((self.project / 'path.json').read_text())
        self.assertEqual(len(points), 5)
        self.assertEqual(points[0], [0, 0, 0])
        self.assertEqual(points[-1], [.5, 0, .5])

    def test_ignored_api_control_is_rejected_before_execution(self):
        file = self.params({'structure': str(SOURCE / 'examples/Si.POSCAR'), 'paw_root': 'missing', 'output': 'run', 'family': 'semicore'})
        result = self.drive('api', 'job.run', '--params', file)
        self.assertEqual(result.returncode, 2)
        self.assertIn('family', result.stderr)
        self.assertFalse((self.project / 'run').exists())

    def test_placeholders_and_unsupported_api_are_non_success(self):
        for method, params in [('job.status', {'job_id': 'any'}), ('results.export', {})]:
            with self.subTest(method=method):
                result = self.drive('api', method, '--params', self.params(params))
                self.assertEqual(result.returncode, 3)
                self.assertIn(json.loads(result.stdout)['status'], ('FAILED', 'UNSUPPORTED'))

    def test_missing_paw_does_not_fabricate_a_success(self):
        file = self.params({'structure': str(SOURCE / 'examples/Si.POSCAR'), 'paw_root': str(self.project / 'missing'), 'output': 'run'})
        result = self.drive('api', 'job.run', '--params', file)
        self.assertEqual(result.returncode, 3)
        self.assertEqual(json.loads(result.stdout)['status'], 'FAILED')
        self.assertFalse((self.project / 'run/scf_result.json').exists())

    def test_cli_outputs_and_python_positional_output_are_protected(self):
        original = self.project / 'path.json'; original.write_text('immutable')
        vertices = self.params([[0, 0, 0], [.5, 0, .5]], 'vertices.json')
        for output_args in (['--out', 'path.json'], ['--out=path.json']):
            result = self.drive('cli', '--', 'make-band-path', vertices, *output_args)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(original.read_text(), 'immutable')
        file = self.params({'args': ['missing-scf.json', 'path.json']})
        result = self.drive('python', 'sparse_dft.checkpoint:export_frozen_scf', '--params', file)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(original.read_text(), 'immutable')

    def test_default_output_is_protected(self):
        (self.project / 'job').mkdir(); (self.project / 'job/sentinel').write_text('keep')
        result = self.drive('cli', '--', 'prepare', str(SOURCE / 'examples/Si.POSCAR'), '--paw-root', 'missing')
        self.assertEqual(result.returncode, 2)
        self.assertIn('Output exists', result.stderr)
        self.assertEqual((self.project / 'job/sentinel').read_text(), 'keep')

    def test_secret_parameter_is_redacted_from_record(self):
        token = 'synthetic-secret-not-a-credential'
        file = self.params({'output': 'mp', 'api_key': token, 'api_root': 'http://127.0.0.1:1', 'timeout': .1, 'count': 1})
        result = self.drive('api', 'mp.fetch_structures', '--params', file)
        self.assertNotEqual(result.returncode, 0)
        for record in (self.project / 'workflow/sparse-dft-driver/runs').glob('*/*'):
            if record.is_file(): self.assertNotIn(token, record.read_text())

    def test_monitor_keeps_partial_lines_and_unknown_liveness(self):
        file = self.project / 'scf.json.iterations.jsonl'
        file.write_text('{"iteration":1}\n{"iteration":')
        data = monitor(self.project)
        self.assertEqual(data['iterations'][0]['complete_records'], 1)
        self.assertEqual(data['iterations'][0]['last'], {'iteration': 1})
        self.assertEqual(data['state'], 'UNKNOWN')
        self.assertTrue(file.read_text().endswith('{"iteration":'))

    def test_existing_workflow_is_never_overwritten(self):
        workflow = self.project / 'workflow'; workflow.mkdir()
        sentinel = workflow / 'experiment_manifest.json'; sentinel.write_text('existing')
        with self.assertRaises(ValueError): scaffold(self.project, 'system', 'claim', 'observable', 'gate')
        self.assertEqual(sentinel.read_text(), 'existing')

    def test_preflight_logs_do_not_block_later_project_scaffold(self):
        log = self.project / 'workflow/sparse-dft-driver/runs/preflight/run.json'
        log.parent.mkdir(parents=True); log.write_text('existing preflight log')
        scaffold(self.project, 'system', 'claim', 'observable', 'gate')
        self.assertEqual(log.read_text(), 'existing preflight log')
        self.assertTrue((self.project / 'workflow/experiment_manifest.json').is_file())

    def test_synthetic_handoff_preserves_flags_and_passes_evidence_schema(self):
        file = self.project / 'synthetic-scf.json'
        file.write_text(json.dumps({'converged': True, 'total_energy_valid': False, 'onecenter_closure_diagnostic': True, 'total_energy_ev': -1.25, 'internal_energy_ev': -1.2, 'entropy_energy_ev': -.05, 'density_residual': 1e-9, 'provenance': {'cutoff_ev': 500, 'fixture': 'synthetic test only'}}))
        output = self.project / 'handoff'
        result = handoff(file, output, 'S01', 'synthetic fixture', 'synthetic', 'user-confirmed')
        self.assertEqual(result['state'], 'converged')
        self.assertFalse(result['validity']['total_energy_valid'])
        self.assertEqual(result['external_accuracy_validation']['basis'], 'user-confirmed')
        records = json.loads((output / 'evidence.json').read_text())
        self.assertEqual(records[0]['result_method'], 'synthetic')
        self.assertEqual(records[0]['verification'], 'pending')
        self.assertEqual(records[0]['provenance']['artifacts'][0]['sha256'], sha256(file))
        self.assertIn('internal_energy_ev', {x['metric'] for x in records})
        self.assertIn('entropy_energy_ev', {x['metric'] for x in records})
        self.assertNotIn('free_energy_ev', {x['metric'] for x in records})
        path = UPSTREAM_WRITING / 'skills/sci-literature-evidence/scripts/evidence_tools.py'
        if path.is_file():
            spec = importlib.util.spec_from_file_location('upstream_evidence', path)
            mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
            for record in records:
                self.assertEqual(mod.verify_record(record, self.project, None, release=False), [])
        else:
            # No legacy evidence-v2 verifier is installed; verify the local
            # immutable extraction contract without claiming upstream validation.
            for record in records:
                self.assertIn('JSON Pointer', record['provenance']['locator'])
                self.assertEqual(record['source_kind'], 'raw_data')
                self.assertEqual(record['result_method'], 'synthetic')
        with self.assertRaises(ValueError): handoff(file, output, 'S01', 'fixture', 'synthetic')

    def test_failed_and_nonfinite_result_does_not_promote_stage(self):
        for i, data in enumerate([{'converged': False, 'total_energy_ev': 1.}, {'converged': True, 'total_energy_ev': float('nan')}, {'status': 'FAILED', 'data': {'converged': True}}]):
            file = self.project / f'fixture-{i}.json'; file.write_text(json.dumps(data))
            result = handoff(file, self.project / f'handoff-{i}', 'S01', 'synthetic fixture', 'synthetic')
            self.assertEqual(result['state'], 'failed')

    def test_all_case_manifests_pass_upstream_design_contract(self):
        validator = UPSTREAM_DFT / 'skills/dft-workflow-orchestrator/scripts/validate_experiment_manifest.py'
        self.assertGreaterEqual(len(library()['cases']), 38)
        for case in library()['cases']:
            with self.subTest(case=case['id']):
                root = self.project / ('case-' + case['id'])
                materialize(case, root, 'real system to be supplied')
                result = subprocess.run([sys.executable, str(validator), '--manifest', str(root / 'workflow/experiment_manifest.json'), '--handoff-register', str(root / 'workflow/handoff_register.csv')], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertTrue(json.loads(result.stdout)['valid'])
                self.assertTrue((root / 'workflow/experiment_matrix.csv').is_file())
                guide = (root / 'workflow/case_method.md').read_text()
                self.assertNotIn('](../scripts/', guide)
                self.assertNotIn('](../references/', guide)
                if case['id'] == '19':
                    stages = json.loads((root / 'workflow/experiment_manifest.json').read_text())['stages']
                    self.assertEqual(stages[0]['kind'], 'analysis')
                    self.assertEqual(stages[2]['kind'], 'analysis')
                    self.assertEqual(stages[2]['engine'], 'project-python')
                    self.assertEqual(stages[1]['kind'], 'dft')

    def test_eos_prepares_actual_scaled_structures_and_recipe(self):
        factors = self.params([.95, 1., 1.05], 'factors.json')
        settings = self.params({'paw_root': '/actual/user-provided/paw', 'smearing_ev': 0.}, 'settings.json')
        result = eos_prepare(SOURCE, SOURCE / 'examples/Si.POSCAR', factors, settings, self.project)
        self.assertFalse(result['calculation_executed'])
        sys.path.insert(0, str(SOURCE))
        from sparse_dft.structure import read_structure
        points = json.loads((self.project / 'volume_points.json').read_text())['points']
        parent_volume = read_structure(SOURCE / 'examples/Si.POSCAR').volume
        for point in points:
            s = read_structure(point['structure'])
            self.assertAlmostEqual(s.volume, parent_volume * point['volume_factor'])
            self.assertEqual(s.natoms, 2)
        recipe = json.loads((self.project / 'eos-recipe.json').read_text())
        self.assertEqual(len(recipe['steps']), 3)
        self.assertEqual(recipe['steps'][0]['method'], 'job.run')

    def test_eos_fits_explicit_synthetic_data_and_rejects_bad_points(self):
        points = []
        for i, volume in enumerate([9., 10., 11.]):
            path = self.project / f'synthetic-{i}.json'
            path.write_text(json.dumps({'converged': True, 'total_energy_ev': -10. + .02 * (volume - 10.) ** 2, 'internal_energy_ev': -9. + .02 * (volume - 10.) ** 2, 'fixture': 'synthetic arithmetic test only'}))
            points.append({'id': f'v{i}', 'volume_angstrom3': volume, 'natoms': 2, 'result': path.name})
        file = self.project / 'synthetic-points.json'; file.write_text(json.dumps({'points': points}))
        report = eos_analyze(file, self.project, self.project / 'fit-f', 'synthetic')
        self.assertEqual(report['result_method'], 'synthetic')
        self.assertAlmostEqual(report['equilibrium_volume_angstrom3'], 10.)
        self.assertAlmostEqual(report['bulk_modulus_ev_per_angstrom3'], .4)
        report = eos_analyze(file, self.project, self.project / 'fit-e', 'synthetic', 'internal_energy_ev')
        self.assertAlmostEqual(report['minimum_energy_ev_per_cell'], -9.)
        path = self.project / 'synthetic-0.json'; path.write_text(json.dumps({'converged': False, 'total_energy_ev': -10.}))
        with self.assertRaises(ValueError): eos_analyze(file, self.project, self.project / 'bad-fit', 'synthetic')

    def test_eos_relax_uses_hash_checked_last_accepted_scf(self):
        points = []
        for i, volume in enumerate([9., 10., 11.]):
            scf = self.project / f'synthetic-accepted-{i}.json'
            scf.write_text(json.dumps({'converged': True, 'total_energy_ev': -10. + .02 * (volume - 10.) ** 2, 'fixture': 'synthetic accepted-state test'}))
            relax = self.project / f'synthetic-relax-{i}.json'
            relax.write_text(json.dumps({'status': 'CONVERGED', 'history': [{'scf_file': scf.name, 'scf_sha256': sha256(scf)}]}))
            points.append({'id': f'v{i}', 'volume_angstrom3': volume, 'natoms': 2, 'result': relax.name, 'relax_ions': True})
        file = self.project / 'synthetic-points.json'; file.write_text(json.dumps({'points': points}))
        report = eos_analyze(file, self.project, self.project / 'fit', 'synthetic')
        self.assertAlmostEqual(report['minimum_energy_ev_per_cell'], -10.)
        (self.project / 'synthetic-accepted-0.json').write_text('{}')
        with self.assertRaises(ValueError): eos_analyze(file, self.project, self.project / 'bad-fit', 'synthetic')

    def recipe(self, value):
        file = self.project / 'recipe.json'; file.write_text(json.dumps(value))
        return file

    def run_recipe(self, file, *args):
        return subprocess.run([sys.executable, str(SCRIPTS / 'run_recipe.py'), '--source', str(SOURCE), '--project', str(self.project), '--recipe', str(file), *args], capture_output=True, text=True)

    def test_recipe_executes_and_resume_checks_immutable_outputs(self):
        script = self.project / 'write-fixture.py'
        script.write_text('from pathlib import Path\nPath("synthetic.txt").write_text("synthetic fixture only")\n')
        file = self.recipe({'schema': 'sparse-dft-recipe-1', 'steps': [{'id': 'fixture', 'mode': 'project-python', 'script': script.name, 'outputs': ['synthetic.txt']}]})
        result = self.run_recipe(file)
        self.assertEqual(result.returncode, 0, result.stderr)
        state = next((self.project / 'workflow/sparse-dft-recipes').glob('*/state.json'))
        result = self.run_recipe(file, '--resume-state', str(state))
        self.assertEqual(result.returncode, 0, result.stderr)
        (self.project / 'synthetic.txt').write_text('changed fixture')
        result = self.run_recipe(file, '--resume-state', str(state))
        self.assertEqual(result.returncode, 2)
        self.assertIn('changed or missing', result.stderr)

    def test_recipe_stops_before_dependent_call_after_failure(self):
        (self.project / 'fail.py').write_text('raise SystemExit(7)\n')
        (self.project / 'never.py').write_text('from pathlib import Path\nPath("never.txt").write_text("should not exist")\n')
        file = self.recipe({'schema': 'sparse-dft-recipe-1', 'steps': [{'id': 'failure', 'mode': 'project-python', 'script': 'fail.py'}, {'id': 'dependent', 'depends_on': ['failure'], 'mode': 'project-python', 'script': 'never.py'}]})
        result = self.run_recipe(file)
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertFalse((self.project / 'never.txt').exists())
        state = json.loads(next((self.project / 'workflow/sparse-dft-recipes').glob('*/state.json')).read_text())
        self.assertEqual(state['status'], 'FAILED')

    def test_recipe_api_and_python_modes_execute_real_calls(self):
        file = self.recipe({'schema': 'sparse-dft-recipe-1', 'steps': [{'id': 'parse', 'mode': 'api', 'method': 'structure.validate', 'params': {'path': str(SOURCE / 'examples/Si.POSCAR')}}, {'id': 'mesh', 'depends_on': ['parse'], 'mode': 'python', 'call': 'sparse_dft.kpoints:recommend_kpoints', 'params': {'args': [{'$structure': str(SOURCE / 'examples/Si.POSCAR')}]}}]})
        result = self.run_recipe(file)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('"natoms": 2', result.stdout)
        self.assertFalse(list((self.project / 'workflow/sparse-dft-recipes').glob('*/*-params.json')))


if __name__ == '__main__':
    unittest.main(verbosity=2)
