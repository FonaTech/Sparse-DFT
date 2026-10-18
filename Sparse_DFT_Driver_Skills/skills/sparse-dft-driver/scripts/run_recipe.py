#!/usr/bin/env python3
"""Execute a reviewed explicit Sparse-DFT recipe, retaining state and output hashes."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

from sparse_dft_driver import protect_skill, resolve_source, run_recorded, write_json
from source_catalog import sha256

SCRIPTS = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recipe', type=Path, required=True)
    p.add_argument('--source', default=os.environ.get('SDFT_SOURCE'))
    p.add_argument('--project', type=Path, required=True)
    p.add_argument('--resume-state', type=Path)
    a = p.parse_args()
    try:
        if sys.version_info < (3, 11): raise ValueError('Python >=3.11 is required')
        source = resolve_source(a.source)
        project = a.project.expanduser().resolve(); protect_skill(project)
        recipe_path = a.recipe.resolve()
        recipe = json.loads(recipe_path.read_text())
        if recipe.get('schema') != 'sparse-dft-recipe-1': raise ValueError('Expected schema sparse-dft-recipe-1')
        if not isinstance(recipe.get('steps'), list) or not recipe['steps']: raise ValueError('Recipe needs nonempty steps')
        names = [s['id'] for s in recipe['steps']]
        if len(names) != len(set(names)) or any(not isinstance(s, str) or not s.replace('-', '').replace('_', '').isalnum() for s in names): raise ValueError('Step IDs must be unique simple names')
        identity = {'recipe_sha256': sha256(recipe_path), 'source_cli_sha256': sha256(source / 'sparse_dft/cli.py'), 'source_api_sha256': sha256(source / 'sparse_dft/api.py'), 'project': str(project), 'source': str(source)}
        for key in ('SDFT_SCF_EXE',):
            exe = recipe.get('environment', {}).get(key, os.environ.get(key))
            if exe and Path(exe).is_file(): identity[key + '_sha256'] = sha256(exe)
        project_scripts = {}
        step_binaries = {}
        for step in recipe['steps']:
            if step.get('mode') == 'project-python':
                script = Path(step['script']); script = script if script.is_absolute() else project / script
                script = script.resolve()
                if not script.is_relative_to(project) or not script.is_file(): raise ValueError('project-python requires an existing script inside the project')
                project_scripts[str(script)] = sha256(script)
            exe = step.get('environment', {}).get('SDFT_SCF_EXE')
            if exe and Path(exe).is_file(): step_binaries[str(Path(exe).resolve())] = sha256(exe)
        identity['project_script_hashes'] = project_scripts
        identity['step_binary_hashes'] = step_binaries
        if a.resume_state:
            state_path = a.resume_state.resolve(); state = json.loads(state_path.read_text())
            if state['identity'] != identity: raise ValueError('Recipe, source, binary or project identity changed; start a new recipe state')
        else:
            project.mkdir(parents=True, exist_ok=True)
            folder = project / 'workflow/sparse-dft-recipes' / (dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex[:8])
            folder.mkdir(parents=True)
            state_path = folder / 'state.json'
            state = {'schema': 'sparse-dft-recipe-state-1', 'identity': identity, 'steps': {}, 'status': 'RUNNING', 'accuracy_validation_basis': recipe.get('accuracy_validation_basis', 'project existing validation'), 'scientific_results_fabricated': False}
        write_json(state_path, state)
        print(f'Recipe state: {state_path}', flush=True)
        completed = set()
        for step in recipe['steps']:
            name = step['id']
            if not set(step.get('depends_on', [])) <= completed: raise ValueError(f'{name}: dependencies must be earlier completed steps')
            old = state['steps'].get(name, {})
            if old.get('status') == 'COMPLETED':
                outputs = old.get('outputs', [])
                if not outputs: raise ValueError(f'{name}: safe resume requires declared file outputs; no output evidence to skip')
                if any(not Path(x['path']).is_file() or sha256(x['path']) != x['sha256'] for x in outputs): raise ValueError(f'{name}: completed output changed or missing; use a new recipe state')
                completed.add(name); continue
            mode = step['mode']
            env = {**recipe.get('environment', {}), **step.get('environment', {})}
            command = [sys.executable, str(SCRIPTS / 'sparse_dft_driver.py'), '--source', str(source), '--cwd', str(project)]
            for key, value in env.items(): command += ['--env', f'{key}={value}']
            if step.get('resume'): command.append('--resume')
            temporary = None
            if mode in ('api', 'python'):
                temporary = state_path.parent / f'{name}-params.json'
                with open(temporary, 'w', opener=lambda file, flags: os.open(file, flags, 0o600)) as f: json.dump(step.get('params', {}), f, allow_nan=False)
                command += [mode, step['method'] if mode == 'api' else step['call'], '--params', str(temporary)]
            elif mode == 'cli': command += ['cli', '--', *step['args']]
            elif mode in ('native', 'tool'):
                command += [mode, '--name', step['name']]
                if mode == 'tool':
                    if not step.get('reviewed'): raise ValueError(f'{name}: review the source tool and set reviewed=true')
                    command += ['--reviewed']
                elif step.get('bin_dir'): command += ['--bin-dir', step['bin_dir']]
                command += ['--', *step.get('args', [])]
            elif mode != 'project-python': raise ValueError(f'Unknown mode: {mode}')
            state['steps'][name] = {'status': 'RUNNING', 'mode': mode}
            state['status'] = 'RUNNING'; write_json(state_path, state)
            try:
                if mode == 'project-python':
                    script = Path(step['script']); script = script if script.is_absolute() else project / script
                    script = script.resolve()
                    if not script.is_relative_to(project) or not script.is_file(): raise ValueError('project-python requires an existing script inside the project')
                    code = run_recorded([sys.executable, str(script), *step.get('args', [])], project, source, env)
                else:
                    code = subprocess.run(command, cwd=project).returncode
            finally:
                if temporary: temporary.unlink(missing_ok=True)
            outputs = []
            missing = []
            for output in step.get('outputs', []):
                file = Path(output); file = file if file.is_absolute() else project / file
                if not file.is_file(): missing.append(str(file))
                else: outputs.append({'path': str(file.resolve()), 'sha256': sha256(file)})
            passed = code == 0 and not missing
            state['steps'][name] = {'status': 'COMPLETED' if passed else 'FAILED', 'returncode': code, 'outputs': outputs, 'missing_outputs': missing}
            if not passed:
                state['status'] = 'FAILED'; write_json(state_path, state); return code if code > 0 else 3
            completed.add(name); write_json(state_path, state)
        state['status'] = 'COMPLETED'; write_json(state_path, state)
        print(json.dumps({'status': 'COMPLETED', 'state': str(state_path), 'steps': len(completed)})); return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        if 'state' in locals() and 'state_path' in locals():
            state['status'] = 'FAILED'; state['error'] = str(exc); write_json(state_path, state)
        print(json.dumps({'status': 'FAILED', 'message': str(exc)}), file=sys.stderr); return 2


if __name__ == '__main__':
    raise SystemExit(main())
