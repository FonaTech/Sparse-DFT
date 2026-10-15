"""Reproducible Materials Project data acquisition.

The current MP API requires an API key.  This module never embeds a key and
never silently substitutes synthetic data.  It supports API responses saved
as JSON snapshots so a 100-material run can be resumed offline with exact
provenance.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import hashlib
import json
import os
import time

from .errors import InputError
from .settings import validate_timeout


API_ROOT = 'https://api.materialsproject.org'


@dataclass
class MPRecord:
    material_id: str
    formula: str | None
    structure: dict | None
    bandstructure: dict | None
    source_url: str
    retrieved_at: str
    sha256: str

    def as_dict(self):
        return asdict(self)


class MaterialsProjectClient:
    def __init__(self, api_key: str | None = None, api_root: str = API_ROOT, timeout: float | None = None):
        self.api_key = api_key or os.environ.get('MP_API_KEY') or os.environ.get('MAPI_KEY')
        self.api_root = api_root.rstrip('/')
        self.timeout = validate_timeout(timeout)
        if not self.api_key:
            raise InputError('Materials Project API key is required; set MP_API_KEY or pass --api-key')

    def get(self, path: str, params: dict | None = None):
        query = ('?' + urlencode(params)) if params else ''
        url = self.api_root + '/' + path.lstrip('/') + query
        request = Request(url, headers={'X-API-KEY': self.api_key, 'Accept': 'application/json', 'User-Agent': 'Sparse-DFT/0.1 research workflow'})
        # No fixed wall-clock limit: large raw bandstructure responses can
        # legitimately take longer than a short API timeout.  Callers can
        # still supply an explicit timeout for interactive environments.
        with urlopen(request, timeout=self.timeout) as response:
            raw = response.read()
            payload = json.loads(raw)
        if isinstance(payload, dict) and payload.get('message') and payload.get('data') is None:
            raise InputError(f'Materials Project API error: {payload}')
        return url, payload

    def summaries(self, limit=100, criteria=None):
        fields = 'material_id,formula_pretty,structure,band_gap,formation_energy_per_atom'
        url, payload = self.get('/materials/summary/', {'_limit': limit, '_fields': fields, **(criteria or {})})
        rows = payload.get('data', payload if isinstance(payload, list) else [])
        if len(rows) < limit:
            raise InputError(f'Materials Project returned {len(rows)} structures; requested {limit}')
        return url, rows[:limit]

    def bandstructures(self, material_ids: list[str], endpoint='/materials/bandstructure/'):
        """Fetch raw MP band paths when the account/API exposes them.

        MP deployments differ in which electronic-structure endpoint is
        enabled.  We therefore require a response carrying both k-points and
        eigenvalues for every material and fail loudly on summary-only data.
        """
        if not material_ids:
            raise InputError('at least one material id is required')
        url, payload = self.get(endpoint, {'material_ids': ','.join(material_ids), '_fields': 'material_id,bandstructure'})
        rows = payload.get('data', payload if isinstance(payload, list) else [])
        if not isinstance(rows, list) or not rows:
            raise InputError(f'Materials Project band endpoint returned no records: {url}')
        found = {str(r.get('material_id')): r for r in rows if isinstance(r, dict) and r.get('material_id')}
        missing = [x for x in material_ids if x not in found]
        if missing:
            raise InputError(f'Materials Project returned no bandstructure for {len(missing)} ids', details={'missing': missing, 'url': url})
        for mid, row in found.items():
            bs = row.get('bandstructure') or row.get('band_structure')
            if not isinstance(bs, dict) or not (bs.get('kpoints') and (bs.get('eigenvalues') or bs.get('bands'))):
                raise InputError(f'MP response for {mid} lacks raw k-point eigenvalues; summary-only response is not valid', details={'url': url, 'material_id': mid})
        return url, [found[x] for x in material_ids]


def write_snapshot(root: str | Path, rows: list[dict], source_url: str, *, requested_count: int):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    records = []
    for row in rows:
        material_id = row.get('material_id') or row.get('material_id', {}).get('id')
        if not material_id:
            raise InputError('MP summary row has no material_id')
        record = {'material_id': material_id, 'formula': row.get('formula_pretty'), 'summary': row,
                  'source_url': source_url, 'retrieved_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
        records.append(record)
    if len({x['material_id'] for x in records}) != len(records):
        raise InputError('MP snapshot contains duplicate material ids')
    body = json.dumps({'schema_version': 'sparse-dft-mp-snapshot-1', 'requested_count': requested_count,
                       'records': records}, indent=2, sort_keys=True).encode()
    destination = root / 'mp_structures.json'
    destination.write_bytes(body)
    manifest = {'schema_version': 'sparse-dft-mp-manifest-1', 'count': len(records),
                'requested_count': requested_count, 'source_url': source_url,
                'sha256': hashlib.sha256(body).hexdigest(), 'records': [x['material_id'] for x in records]}
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return manifest


def load_snapshot(path: str | Path):
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    if payload.get('schema_version') != 'sparse-dft-mp-snapshot-1':
        raise InputError('unsupported Sparse-DFT MP snapshot schema')
    records = payload.get('records', [])
    if len({x.get('material_id') for x in records}) != len(records):
        raise InputError('duplicate material ids in MP snapshot')
    return payload


def write_band_snapshot(root: str | Path, rows: list[dict], source_url: str, *, requested_ids: list[str]):
    """Persist raw MP band paths with immutable provenance and validation."""
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    records = []
    for row in rows:
        mid = str(row.get('material_id') or '')
        bs = row.get('bandstructure') or row.get('band_structure')
        if not mid or not isinstance(bs, dict):
            raise InputError('raw MP band snapshot row requires material_id and bandstructure')
        if not bs.get('kpoints') or not (bs.get('eigenvalues') or bs.get('bands')):
            raise InputError(f'raw MP band snapshot {mid} lacks kpoints/eigenvalues')
        records.append({'material_id': mid, 'formula': row.get('formula_pretty'), 'bandstructure': bs,
                        'source_url': source_url, 'retrieved_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    ids = [x['material_id'] for x in records]
    if len(set(ids)) != len(ids): raise InputError('duplicate material ids in raw MP band snapshot')
    missing = [x for x in requested_ids if x not in ids]
    if missing: raise InputError('raw MP band snapshot is incomplete', details={'missing': missing})
    body = json.dumps({'schema_version': 'sparse-dft-mp-band-snapshot-1', 'requested_ids': requested_ids,
                       'records': records}, indent=2, sort_keys=True).encode()
    destination = root / 'mp_bandstructures.json'; destination.write_bytes(body)
    manifest = {'schema_version': 'sparse-dft-mp-band-manifest-1', 'count': len(records), 'requested_count': len(requested_ids),
                'source_url': source_url, 'sha256': hashlib.sha256(body).hexdigest(), 'records': ids,
                'raw_eigenvalues': True}
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return manifest


def load_band_snapshot(path: str | Path):
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    if payload.get('schema_version') != 'sparse-dft-mp-band-snapshot-1':
        raise InputError('unsupported Sparse-DFT MP band snapshot schema')
    records = payload.get('records', [])
    if len(records) != len({r.get('material_id') for r in records}):
        raise InputError('duplicate material ids in MP band snapshot')
    for r in records:
        bs = r.get('bandstructure')
        if not isinstance(bs, dict) or not bs.get('kpoints') or not (bs.get('eigenvalues') or bs.get('bands')):
            raise InputError(f'invalid raw band record: {r.get("material_id")}')
    return payload
