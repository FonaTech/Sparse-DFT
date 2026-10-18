#!/usr/bin/env python3
"""Portable source-aware driver for Sparse-DFT CLI, API, Python and expert tools."""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
import uuid
from pathlib import Path

from source_catalog import sha256, source_catalog, verify_manifest

SKILL = Path(__file__).resolve().parents[1]
RUNTIME_ENV = ('OMP_NUM_THREADS', 'OMP_DYNAMIC', 'VECLIB_MAXIMUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'MPLCONFIGDIR', 'XDG_CACHE_HOME', 'QT_QPA_PLATFORM', 'CUDA_VISIBLE_DEVICES')


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path, data):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def resolve_source(value):
    if value:
        path = Path(value).expanduser().resolve()
    else:
        spec = importlib.util.find_spec('sparse_dft')
        if not spec or not spec.origin:
            raise ValueError('Set --source or SDFT_SOURCE to the Sparse-DFT-source directory')
        path = Path(spec.origin).resolve().parent.parent
    if not (path / 'sparse_dft/cli.py').is_file():
        raise ValueError(f'Sparse-DFT source not found: {path}')
    return path


def redacted(value):
    if isinstance(value, dict):
        return {key: '[REDACTED]' if any(s in key.lower() for s in ('api_key', 'token', 'password', 'secret')) else redacted(item) for key, item in value.items()}
    if isinstance(value, list):
        out, hide = [], False
        for item in value:
            if hide:
                out.append('[REDACTED]'); hide = False; continue
            if isinstance(item, str) and item.startswith(('--api-key=', '--token=', '--password=')):
                out.append(item.split('=', 1)[0] + '=[REDACTED]')
            else:
                out.append(redacted(item))
                hide = item in ('--api-key', '--token', '--password') if isinstance(item, str) else False
        return out
    return value


def protect_skill(path):
    if Path(path).resolve().is_relative_to(SKILL):
        raise ValueError('Calculation outputs and logs must be outside the skill directory')


def output_guard(request, cwd, resume, catalog=None):
    mode = request.get('mode')
    params = dict(request.get('params', {}) if mode == 'api' else request.get('kwargs', {}))
    if mode == 'python' and catalog:
        names = catalog['python'][request['call']]['positional_parameters']
        params = {**dict(zip(names, request.get('args', []))), **params}
    key = next((key for key in ('output', 'output_dir', 'destination', 'workdir') if key in params), 'output')
    output = params.get(key)
    method = request.get('method', request.get('call', ''))
    defaults = {'job.run': 'job', 'job.prepare': 'job', 'bands.run': 'job', 'dos.run': 'dos', 'relax.ionic': 'relax', 'relax.cell': 'relax', 'bands.analyze': 'band_analysis'}
    if output is None and method in defaults:
        output = defaults[method]
    if output:
        path = Path(output)
        if not path.is_absolute():
            path = cwd / path
        protect_skill(path)
        if path.exists() and ((path.is_dir() and any(path.iterdir())) or path.is_file()) and not resume:
            raise ValueError(f'Output exists: {path}. Choose a new output or explicitly use --resume after reviewing restart compatibility.')


def run_recorded(command, cwd, source, env_values, request=None, request_path=None):
    protect_skill(cwd)
    cwd.mkdir(parents=True, exist_ok=True)
    run_dir = cwd / 'workflow/sparse-dft-driver/runs' / (dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex[:8])
    run_dir.mkdir(parents=True)
    env = os.environ.copy()
    env.update(env_values)
    env['PYTHONPATH'] = str(source) + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    record = {'schema': 'sparse-dft-driver-run-1', 'started_at': now(), 'cwd': str(cwd), 'source': str(source), 'command': redacted(command), 'request': redacted(request), 'environment': {k: v for k, v in env.items() if k.startswith('SDFT_') or k in RUNTIME_ENV}, 'state': 'STARTING', 'artifacts': {'stdout': str(run_dir / 'stdout.log'), 'stderr': str(run_dir / 'stderr.log')}, 'identity': {str(source / k): v for k, v in source_catalog(source)['identity'].items()}, 'production_validated': False}
    selected_kernel = env.get('SDFT_SCF_EXE') or next((str(source / d / 'sdft_scf') for d in ('.install/bin', 'build-cpu', 'build-metal', 'build') if (source / d / 'sdft_scf').is_file()), shutil.which('sdft_scf'))
    kernel_related = [selected_kernel, str(Path(selected_kernel).with_name('sdft_relax_step'))] if selected_kernel else []
    for exe in (command[0], *kernel_related):
        if exe and Path(exe).is_file():
            record['identity'][str(Path(exe).resolve())] = sha256(exe)
    inputs = {}
    def input_files(value):
        if isinstance(value, dict):
            for x in value.values(): input_files(x)
        elif isinstance(value, list):
            for x in value: input_files(x)
        elif isinstance(value, str) and '\n' not in value:
            try:
                p = Path(value).expanduser(); p = p if p.is_absolute() else cwd / p
                if p.is_file() and p.resolve() != Path(command[0]).resolve(): inputs[str(p.resolve())] = sha256(p)
            except OSError: pass
    input_files(request or command[1:])
    record['input_file_hashes'] = inputs
    write_json(run_dir / 'run.json', record)
    print(f'Sparse-DFT driver record: {run_dir / "run.json"}', file=sys.stderr, flush=True)
    secrets = []
    if request:
        def collect(value):
            if isinstance(value, dict):
                for k, v in value.items():
                    if any(s in k.lower() for s in ('api_key', 'password', 'token', 'secret')) and isinstance(v, str):
                        secrets.append(v)
                    else: collect(v)
            elif isinstance(value, list):
                for x in value: collect(x)
        collect(request)
    for i, word in enumerate(command):
        if word == '--api-key' and i + 1 < len(command): secrets.append(command[i + 1])
        elif word.startswith('--api-key='): secrets.append(word.split('=', 1)[1])
    try:
        process = subprocess.Popen(command, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1)
        record.update(pid=process.pid, state='RUNNING')
        write_json(run_dir / 'run.json', record)
        def pump(stream, path, target):
            with path.open('w') as file:
                for line in iter(stream.readline, ''):
                    for secret in secrets:
                        if secret: line = line.replace(secret, '[REDACTED]')
                    file.write(line); file.flush()
                    target.write(line); target.flush()
            stream.close()
        threads = [threading.Thread(target=pump, args=(process.stdout, run_dir / 'stdout.log', sys.stdout)), threading.Thread(target=pump, args=(process.stderr, run_dir / 'stderr.log', sys.stderr))]
        for thread in threads: thread.start()
        code = process.wait()
        for thread in threads: thread.join()
        record.update(returncode=code, finished_at=now(), state='PROCESS_COMPLETED' if code == 0 else 'PROCESS_FAILED')
        # A completed helper process is not a declaration of physical validity.
        write_json(run_dir / 'run.json', record)
        return code
    except BaseException as exc:
        record.update(state='INTERRUPTED_OR_LAUNCH_FAILED', error=str(exc), finished_at=now())
        write_json(run_dir / 'run.json', record)
        raise
    finally:
        if request_path: Path(request_path).unlink(missing_ok=True)


def monitor(directory):
    directory = Path(directory).resolve()
    result = {'directory': str(directory), 'state': 'UNKNOWN', 'results': [], 'iterations': [], 'records': []}
    for file in sorted(directory.glob('*.iterations.jsonl')):
        lines = file.read_text().splitlines()
        last = None
        complete = 0
        for line in lines:
            try: last = json.loads(line); complete += 1
            except ValueError: continue
        result['iterations'].append({'path': str(file), 'complete_records': complete, 'last': last})
    for name in ('scf_result.json', 'scf.json', 'bands_result.json', 'bands.json', 'dos.json', 'relax_result.json', 'sparsity_preview.json'):
        file = directory / name
        if file.is_file():
            try:
                obj = json.loads(file.read_text())
                result['results'].append({'kind': name, 'path': str(file), 'sha256': sha256(file), 'converged': obj.get('converged', obj.get('metadata', {}).get('scf_converged')), 'status': obj.get('status'), 'total_energy_valid': obj.get('total_energy_valid', obj.get('metadata', {}).get('total_energy_valid')), 'production_validated': obj.get('production_validated', False)})
            except ValueError: pass
    if (directory / 'run.json').is_file():
        record = json.loads((directory / 'run.json').read_text())
        result['records'].append(record)
        result['state'] = record['state'] if 'returncode' in record else 'UNKNOWN_PROCESS_LIVENESS'
    if result['results']:
        result['state'] = 'RESULT_PRESENT_REVIEW_REQUIRED'
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default=os.environ.get('SDFT_SOURCE'))
    p.add_argument('--cwd', type=Path, default=Path.cwd())
    p.add_argument('--env', action='append', default=[], metavar='NAME=VALUE')
    p.add_argument('--resume', action='store_true', help='allow intentional reuse of an output; Sparse-DFT still validates checkpoint identity')
    sub = p.add_subparsers(dest='mode', required=True)
    q = sub.add_parser('catalog'); q.add_argument('--out', type=Path)
    sub.add_parser('doctor')
    q = sub.add_parser('api'); q.add_argument('method'); q.add_argument('--params', type=Path, required=True)
    q = sub.add_parser('python'); q.add_argument('call'); q.add_argument('--params', type=Path, required=True, help='JSON with args and kwargs; typed object references supported')
    q = sub.add_parser('cli'); q.add_argument('arguments', nargs=argparse.REMAINDER)
    q = sub.add_parser('tool'); q.add_argument('--name', required=True); q.add_argument('--reviewed', action='store_true'); q.add_argument('arguments', nargs=argparse.REMAINDER)
    q = sub.add_parser('native'); q.add_argument('--name', required=True); q.add_argument('--bin-dir', type=Path); q.add_argument('arguments', nargs=argparse.REMAINDER)
    q = sub.add_parser('monitor'); q.add_argument('directory', type=Path)
    a = p.parse_args(argv)
    try:
        if sys.version_info < (3, 11): raise ValueError('Python >=3.11 is required by Sparse-DFT; choose that environment explicitly')
        if a.mode == 'monitor':
            print(json.dumps(monitor(a.directory), indent=2)); return 0
        source = resolve_source(a.source)
        catalog = source_catalog(source)
        cwd = a.cwd.expanduser().resolve()
        env = {}
        for item in a.env:
            key, value = item.split('=', 1)
            if not (key.startswith('SDFT_') or key in RUNTIME_ENV):
                raise ValueError(f'Unsupported runtime environment key: {key}')
            env[key] = value
        if not env.get('SDFT_SCF_EXE', os.environ.get('SDFT_SCF_EXE')):
            candidate = next((source / d / 'sdft_scf' for d in ('.install/bin', 'build', 'build-cpu', 'build-metal') if (source / d / 'sdft_scf').is_file()), None)
            if candidate: env['SDFT_SCF_EXE'] = str(candidate)
        if a.mode == 'catalog':
            if a.out:
                a.out.parent.mkdir(parents=True, exist_ok=True); write_json(a.out, catalog)
            print(json.dumps(catalog if not a.out else {'output': str(a.out), 'counts': {key: len(catalog[key]) for key in ('cli', 'api', 'python', 'tools', 'native')}}, indent=2)); return 0
        if a.mode == 'doctor':
            expected = json.loads((SKILL / 'references/capability-catalog.json').read_text())['identity']
            print(json.dumps({'python': sys.version, 'python_executable': sys.executable, 'source': str(source), 'source_manifest': verify_manifest(source), 'interface_matches_audited_source': catalog['identity'] == expected, 'kernel_configured': env.get('SDFT_SCF_EXE', os.environ.get('SDFT_SCF_EXE')), 'kernel_exists': Path(env.get('SDFT_SCF_EXE', os.environ.get('SDFT_SCF_EXE', '/nonexistent'))).is_file(), 'cmake': shutil.which('cmake'), 'paw_archive': env.get('SDFT_PAW_ARCHIVE', os.environ.get('SDFT_PAW_ARCHIVE')), 'note': 'Call backend.info for compiled capability probes. Active GPU execution must be established from each run performance artifact; this doctor is a file/environment preflight.'}, indent=2)); return 0
        request = None
        request_path = None
        if a.mode in ('api', 'python'):
            params = json.loads(a.params.read_text())
            if not isinstance(params, dict): raise ValueError('Params must be a JSON object')
            if a.mode == 'api':
                contract = catalog['api'].get(a.method)
                if not contract: raise ValueError(f'Unknown API method: {a.method}')
                unknown = set(params) - set(contract['parameters'])
                missing = set(contract['required']) - set(params)
                if unknown or missing: raise ValueError(f'API parameter mismatch: unknown={sorted(unknown)}, missing={sorted(missing)}. Use Python for controls omitted by dispatch.')
                request = {'mode': 'api', 'method': a.method, 'params': params}
            else:
                if a.call not in catalog['python']: raise ValueError('Public Python call not found in actual source catalog')
                if set(params) - {'args', 'kwargs'}: raise ValueError('Python params accepts only args and kwargs')
                request = {'mode': 'python', 'call': a.call, **params}
            output_guard(request, cwd, a.resume, catalog)
            protect_skill(cwd); cwd.mkdir(parents=True, exist_ok=True)
            request_path = cwd / ('.sparse-dft-request-' + uuid.uuid4().hex + '.json')
            with open(request_path, 'x', opener=lambda name, flags: os.open(name, flags, 0o600)) as file:
                json.dump(request, file, allow_nan=False)
            command = [sys.executable, str(SKILL / 'scripts/python_worker.py'), '--source', str(source), '--request', str(request_path)]
        else:
            args = a.arguments[1:] if a.arguments[:1] == ['--'] else a.arguments
            if a.mode == 'cli':
                # argparse in Sparse-DFT is authoritative; preserve every option.
                if '--help' not in args and '-h' not in args:
                    defaults = {'prepare': 'job', 'scf': 'job', 'bands': 'job', 'dos': 'dos', 'relax': 'relax', 'bands-analysis': 'band_analysis', 'mp-fetch': 'data/mp_100', 'mp-fetch-bands': 'data/mp_bands'}
                    output = next((x.split('=', 1)[1] for x in args if x.startswith('--out=')), None)
                    if '--out' in args: output = args[args.index('--out') + 1]
                    if output is None and args: output = defaults.get(args[0])
                    if output: output_guard({'mode': 'api', 'params': {'output': output}}, cwd, a.resume)
                command = [sys.executable, '-m', 'sparse_dft', *args]
            elif a.mode == 'tool':
                if a.name not in catalog['tools']: raise ValueError('Tool name is not present in this Sparse-DFT source')
                if not a.reviewed: raise ValueError('Review tool source, data paths and source-relative writes, then use --reviewed. Some historical tools mutate their source tree even when cwd differs.')
                command = [sys.executable, str(source / catalog['tools'][a.name]['source']), *args]
            else:
                if a.name not in catalog['native']: raise ValueError('Native name is not present in CMakeLists.txt')
                candidates = [a.bin_dir / a.name] if a.bin_dir else [source / '.install/bin' / a.name, source / 'build-cpu' / a.name, source / 'build-metal' / a.name, source / 'build' / a.name]
                found = next((str(x.resolve()) for x in candidates if x.is_file()), shutil.which(a.name))
                if not found: raise ValueError('Native executable missing; specify --bin-dir or build/install it')
                command = [found, *args]
        return run_recorded(command, cwd, source, env, request, request_path)
    except (OSError, ValueError, KeyError, IndexError) as exc:
        print(json.dumps({'status': 'FAILED', 'code': 'DRIVER_INPUT_ERROR', 'message': str(exc)}), file=sys.stderr); return 2


if __name__ == '__main__':
    raise SystemExit(main())
