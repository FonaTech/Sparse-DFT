#!/usr/bin/env python3
"""Build and install a source-only Sparse-DFT delivery without copying host binaries."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
INDEX = "https://pypi.org/simple"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', choices=['cpu', 'metal', 'cuda'], default='cpu')
    p.add_argument('--jobs', type=int, default=2)
    p.add_argument('--prefix', type=Path, default=ROOT / '.install')
    p.add_argument('--build-dir', type=Path)
    p.add_argument('--venv', type=Path, default=ROOT / '.venv')
    p.add_argument('--serial', action='store_true', help='explicitly build without OpenMP')
    p.add_argument('--no-python', action='store_true', help='only install the C++ programs')
    p.add_argument('--no-plots', action='store_true', help='omit the optional matplotlib extra')
    p.add_argument('--index-url', default=INDEX)
    p.add_argument('--cmake-arg', action='append', default=[], help='extra CMake definition; use --cmake-arg=-DNAME=VALUE')
    args = p.parse_args()
    if args.jobs < 1:
        p.error('--jobs must be positive')
    if sys.version_info < (3, 11):
        p.error('Python >=3.11 is required')
    if args.backend == 'metal' and platform.system() != 'Darwin':
        p.error('native Metal requires macOS')
    build = (args.build_dir or ROOT / ('build-' + args.backend)).resolve()
    prefix = args.prefix.resolve()
    venv = args.venv.resolve()
    build.mkdir(parents=True, exist_ok=True)
    report = {'schema_version': 'sparse-dft-source-install-1', 'status': 'RUNNING',
              'started_at': datetime.now(timezone.utc).isoformat(),
              'platform': platform.platform(), 'backend': args.backend,
              'prefix': str(prefix), 'commands': [], 'global_production_validated': False}
    report_path = build / 'installation.json'

    def save():
        report_path.write_text(json.dumps(report, indent=2) + '\n')

    def run(name, cmd, env=None):
        print('+ ' + shlex.join(map(str, cmd)), flush=True)
        start = time.monotonic()
        log = build / (name + '.log')
        with log.open('w') as out:
            result = subprocess.run(list(map(str, cmd)), cwd=ROOT, env=env, stdout=out,
                                    stderr=subprocess.STDOUT)  # no calculation wall timeout
        report['commands'].append({'name': name, 'command': list(map(str, cmd)),
                                   'returncode': result.returncode,
                                   'seconds': time.monotonic() - start, 'log': str(log)})
        save()
        if result.returncode:
            print(log.read_text()[-8000:], file=sys.stderr)
            raise RuntimeError(f'{name} failed; see {log}')

    save()
    try:
        configure = ['cmake', '-S', ROOT, '-B', build, '-DCMAKE_BUILD_TYPE=Release',
                     '-DCMAKE_INSTALL_PREFIX=' + str(prefix),
                     '-DSDFT_USE_SYSTEM_EIGEN=OFF',
                     '-DSDFT_ENABLE_OPENMP=' + ('OFF' if args.serial else 'ON'),
                     '-DSDFT_ENABLE_MPS=' + ('ON' if args.backend == 'metal' else 'OFF'),
                     '-DSDFT_ENABLE_CUDA=' + ('ON' if args.backend == 'cuda' else 'OFF')]
        run('configure', configure + args.cmake_arg)
        cache = (build / 'CMakeCache.txt').read_text()
        if not args.serial and 'SDFT_HAS_OPENMP:BOOL=TRUE' not in cache:
            raise RuntimeError('OpenMP runtime probe failed; install libomp/compiler support '
                               'or explicitly request --serial. No silent parallel-build downgrade.')
        run('compile', ['cmake', '--build', build, '--parallel', args.jobs])
        env = dict(os.environ, SDFT_COMPUTE_BACKEND='cpu', OMP_NUM_THREADS='2',
                   OMP_DYNAMIC='FALSE', VECLIB_MAXIMUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        run('ctest', ['ctest', '--test-dir', build, '--output-on-failure'], env)
        if args.backend != 'cpu':
            env.update(SDFT_COMPUTE_BACKEND='mps' if args.backend == 'metal' else 'cuda',
                       SDFT_GPU_MIN_WORK='0', SDFT_GPU_FFT='1', SDFT_GPU_PBE='1', SDFT_GPU_FFT_MIN_POINTS='0')
            for name in ('sdft_compute_backend_smoke', 'sdft_fft_backend_smoke', 'sdft_pbe_backend_smoke'):
                run(name, [build / (name + ('.exe' if os.name == 'nt' else ''))], env)
            env.update(SDFT_GPU_PAW_CHAIN='1', SDFT_METAL_PACK='gpu',SDFT_BLOCK_ORTHOGONALIZE='1')
            run('sdft_plane_wave_operator_smoke', [build/'sdft_plane_wave_operator_smoke'], env)
        run('native_install', ['cmake', '--install', build])
        if not args.no_python:
            run('venv', [sys.executable, '-m', 'venv', venv])
            python = venv / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
            run('python_install', [python, '-m', 'pip', 'install', '-i', args.index_url,
                                   str(ROOT) + ('' if args.no_plots else '[plots]')])
            run('python_cli', [python, '-m', 'sparse_dft', '--help'])
        # Use the installed binaries when the Python package lives in site-packages.
        if os.name != 'nt':
            lines = ['# Source this file after installation (bash/zsh).',
                     'export PATH=' + shlex.quote(str(prefix / 'bin')) + ':"$PATH"',
                     'export SDFT_SCF_EXE=' + shlex.quote(str(prefix / 'bin/sdft_scf')),
                     'export SDFT_OPENMP_PROBE=' + shlex.quote(str(prefix / 'bin/sdft_openmp_probe')),
                     'export SDFT_ENABLE_PAW_CLOSURE_DIAGNOSTIC=1']
            if not args.no_python:
                lines.insert(1, '. ' + shlex.quote(str(venv / 'bin/activate')))
            (prefix / 'env.sh').write_text('\n'.join(lines) + '\n')
        report['status'] = 'PASS'
        report['scope'] = 'Local clean build, data-independent tests, native install and Python packaging; not all-material scientific certification.'
    except Exception as exc:
        report['status'] = 'FAIL'
        report['error'] = str(exc)
        raise
    finally:
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        save()
    print(json.dumps({'status': 'PASS', 'report': str(report_path), 'prefix': str(prefix)}))


if __name__ == '__main__':
    main()
