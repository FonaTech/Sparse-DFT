#!/usr/bin/env python3
"""Sparse-DFT child process; JSON calls, no eval and no shell interpolation."""
from __future__ import annotations

import argparse
import dataclasses
import importlib
import inspect
import json
import sys
from pathlib import Path


def decode(value):
    if isinstance(value, list):
        return [decode(x) for x in value]
    if isinstance(value, dict):
        if set(value) == {'$structure'}:
            from sparse_dft.structure import read_structure
            return read_structure(value['$structure'])
        if set(value) == {'$bands'}:
            from sparse_dft.bands import load_bands
            return load_bands(value['$bands'])
        if set(value) == {'$paw_archive'}:
            from sparse_dft.paw import Paw54JsonArchive
            return Paw54JsonArchive(value['$paw_archive'])
        if set(value) == {'$kpoint_recommendation'}:
            from sparse_dft.kpoints import KPointRecommendation
            return KPointRecommendation(**value['$kpoint_recommendation'])
        return {key: decode(x) for key, x in value.items()}
    return value


def encode(value):
    if hasattr(value, 'as_dict'):
        return value.as_dict()
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f'Cannot serialize {type(value).__name__}; use the native Python interface for this object')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--request', required=True)
    a = parser.parse_args()
    sys.path.insert(0, a.source)
    request = json.loads(Path(a.request).read_text())
    try:
        if request['mode'] == 'api':
            from sparse_dft.api import dispatch
            result = dispatch(request['method'], request['params'])
            if request['method'] == 'job.status':
                result['status'] = 'UNSUPPORTED'
                result['warnings'].append('Upstream job.status is a placeholder, not live process status; use driver monitor.')
            print(json.dumps(result, default=encode, allow_nan=False), flush=True)
            return 3 if result.get('status') in ('FAILED', 'FAIL', 'UNSUPPORTED') else 0
        module, name = request['call'].split(':', 1)
        if not module.startswith('sparse_dft.') or any(part.startswith('_') for part in name.split('.')):
            raise ValueError('Only public Sparse-DFT Python objects can be called')
        obj = importlib.import_module(module)
        for part in name.split('.'):
            obj = getattr(obj, part)
        args = decode(request.get('args', []))
        kwargs = decode(request.get('kwargs', {}))
        inspect.signature(obj).bind(*args, **kwargs)
        result = obj(*args, **kwargs)
        print(json.dumps(result, default=encode, allow_nan=False), flush=True)
        if isinstance(result, dict):
            if result.get('converged') is False or result.get('status') in ('FAILED', 'FAIL', 'SCF_FAILED', 'MAX_STEPS', 'LINE_SEARCH_FAILED') or result.get('metadata', {}).get('scf_converged') is False:
                return 3
        return 0
    except Exception as exc:
        print(json.dumps({'status': 'FAILED', 'errors': [exc.as_dict() if hasattr(exc, 'as_dict') else {'code': 'DRIVER_CALL_ERROR', 'message': str(exc)}]}), flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
