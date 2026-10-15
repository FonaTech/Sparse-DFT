"""Inspect and import hash-bound PAW data packages without executing bundle code.

Data integrity and native Hamiltonian support are reported independently. An
imported UPF remains a UPF; it is never relabelled as a VASP numeric cache.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import tempfile

from .errors import InputError, UnsupportedError
from .upf import read_upf_paw
from .upf_storage import decode_upf

SCHEMA = 'sparse-dft-pslibrary-storage-1'
MAX_MEMBER_BYTES = 64 * 1024**2
MAX_MANIFEST_BYTES = 8 * 1024**2


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _safe_name(name):
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or '..' in path.parts or
            '\\' in name or str(path) != name):
        raise InputError('unsafe PAW package member: ' + name)
    return path


def _manifest(raw):
    data = json.loads(raw)
    if data.get('schema_version') != SCHEMA:
        raise UnsupportedError('unsupported PAW library manifest schema')
    records = data.get('records', [])
    if not records or len(records) != data.get('record_count') or len(records) > 5000:
        raise InputError('PAW package record count mismatch')
    ids = set()
    for rec in records:
        for key in ('family', 'filename'):
            if not re.fullmatch(r'[A-Za-z0-9_.+-]+', rec[key]) or rec[key] in ('.', '..'):
                raise InputError('invalid PAW record path component')
        record_id = rec['family'] + '/' + rec['filename']
        if record_id in ids:
            raise InputError('duplicate PAW record: ' + record_id)
        ids.add(record_id)
        if not re.fullmatch(r'[A-Z][a-z]?', rec['element']):
            raise InputError('invalid PAW element')
        for key in ('original_sha256', 'storage_sha256'):
            if not re.fullmatch(r'[0-9a-f]{64}', rec[key]):
                raise InputError('invalid PAW SHA256')
        for key in ('original_bytes', 'storage_bytes'):
            if not isinstance(rec[key], int) or not 0 < rec[key] <= MAX_MEMBER_BYTES:
                raise InputError('PAW record exceeds import size bound')
    if set(data.get('elements', [])) != {r['element'] for r in records}:
        raise InputError('PAW element coverage mismatch')
    if len(data['elements']) != data.get('element_count'):
        raise InputError('PAW element count mismatch')
    return data


class PawLibraryPackage:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.archive = self.path.is_file()
        if self.archive:
            with tarfile.open(self.path, mode='r|*') as bundle:
                for member in bundle:
                    name = _safe_name(member.name)
                    if name.name == 'manifest.json':
                        if not member.isfile() or member.size > MAX_MANIFEST_BYTES or len(name.parts) != 2:
                            raise InputError('invalid PAW package manifest member')
                        self.prefix = name.parts[0]
                        self.manifest = _manifest(bundle.extractfile(member).read())
                        break
                else:
                    raise InputError('PAW package has no manifest')
        else:
            manifest = self.path / 'manifest.json'
            if manifest.is_symlink() or manifest.stat().st_size > MAX_MANIFEST_BYTES:
                raise InputError('invalid installed PAW manifest')
            self.manifest = _manifest(manifest.read_bytes())
            self.prefix = None

    def inventory(self, elements=None):
        requested = set(elements or [])
        records = [dict(r, record_id=r['family']+'/'+r['filename'])
                   for r in self.manifest['records'] if not requested or r['element'] in requested]
        return {'schema_version': 'sparse-dft-paw-library-inventory-1', 'library': str(self.path),
                'tier': self.manifest.get('tier'), 'records': records,
                'record_count': len(records), 'available_elements': self.manifest['elements'],
                'missing_requested_elements': sorted(requested-set(self.manifest['elements'])),
                'absent_elements_from_118': self.manifest.get('absent_elements_from_118', []),
                'source_archive_sha256': self.manifest['source_archive_sha256'],
                'payload_integrity_verified': False, 'native_upf_execution_ready': False,
                'mixed_paw_physics_validated': False}

    def resolve(self, elements, *, family='pbe', variants=None):
        variants = variants or {}
        if set(variants)-set(elements):
            raise InputError('variant map contains unused elements')
        selected = []
        for element in dict.fromkeys(elements):
            records = [r for r in self.inventory([element])['records']
                       if r['family'] == family and r['relativistic'] == 'scalar']
            if element in variants:
                records = [r for r in records if r['record_id'] == variants[element]]
            if len(records) != 1:
                raise InputError('choose one explicit scalar PAW variant for ' + element,
                                 details={'candidates': [r['record_id'] for r in records],
                                          'family': family, 'no_candidate': not records})
            record=dict(records[0])
            if not self.archive:
                restored=self.path/'UPF'/record['family']/record['filename']
                if not restored.is_file() or restored.is_symlink():
                    raise InputError('selected original UPF is missing or is a symbolic link')
                with restored.open('rb') as stream:
                    if hashlib.file_digest(stream,'sha256').hexdigest()!=record['original_sha256']:
                        raise InputError('selected original UPF identity changed after import')
                record['upf_path']=str(restored)
                record['original_sha256_verified']=True
            selected.append(record)
        return {'schema_version': 'sparse-dft-paw-library-selection-1', 'records': selected,
                'native_upf_execution_ready': False, 'mixed_paw_physics_validated': False,
                'selection_rule': 'one explicit variant per species; ambiguity is an error'}

    def import_to(self, destination, elements=None, *, expected_sha256=None):
        """Restore selected original UPFs atomically and verify every byte hash.

        Archive Python files are never executed or installed. Licenses and
        descriptive text are copied; the importer uses this version's codec.
        """
        if not self.archive:
            raise InputError('import requires the original PAW package archive')
        with self.path.open('rb') as stream:
            package_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
        if expected_sha256 is not None and package_sha != expected_sha256:
            raise InputError('PAW package SHA256 does not match the expected release')
        destination = Path(destination).resolve()
        if destination.exists():
            raise InputError('PAW import destination already exists; choose a new directory')
        requested = set(elements or self.manifest['elements'])
        missing = requested - set(self.manifest['elements'])
        if missing:
            raise InputError('PAW package lacks requested elements', details={'missing': sorted(missing)})
        records = [r for r in self.manifest['records'] if r['element'] in requested]
        expected = {f'{self.prefix}/stored/{r["family"]}/{r["filename"]}.bin': r for r in records}
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix='.sparse-dft-paw-import-', dir=destination.parent))
        verified, names, legal = [], set(), 0
        try:
            with tarfile.open(self.path, mode='r|*') as bundle:
                for member in bundle:
                    name = _safe_name(member.name)
                    if name.parts[0] != self.prefix or member.name in names:
                        raise InputError('duplicate or inconsistent PAW package member')
                    names.add(member.name)
                    if not member.isfile() or member.size > MAX_MEMBER_BYTES:
                        raise InputError('PAW package must contain bounded regular files only')
                    if member.name in expected:
                        rec = expected[member.name]
                        payload = bundle.extractfile(member).read()
                        if len(payload) != rec['storage_bytes'] or _sha(payload) != rec['storage_sha256']:
                            raise InputError('stored PAW payload identity mismatch')
                        raw = decode_upf(payload)
                        if len(raw) != rec['original_bytes'] or _sha(raw) != rec['original_sha256']:
                            raise InputError('restored UPF identity mismatch')
                        dataset = read_upf_paw(raw)
                        if (dataset.element != rec['element'] or dataset.relativistic != rec['relativistic'] or
                                dataset.header['functional'].strip() != rec['functional']):
                            raise InputError('UPF header differs from catalog selection')
                        relative = Path('UPF') / rec['family'] / rec['filename']
                        target = staging / relative
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(raw)
                        verified.append(dict(record_id=rec['family']+'/'+rec['filename'],
                                             path=str(destination/relative), **dataset.summary()))
                    elif len(name.parts) == 3 and name.parts[1] == 'licenses':
                        target = staging / 'licenses' / name.name
                        target.parent.mkdir(exist_ok=True)
                        target.write_bytes(bundle.extractfile(member).read())
                        legal += 1
                    elif len(name.parts) == 2 and name.name in ('README.zh.md', 'manifest.json'):
                        payload = bundle.extractfile(member).read()
                        if name.name == 'manifest.json' and _manifest(payload) != self.manifest:
                            raise InputError('PAW archive changed between inventory and import')
                        if name.name == 'README.zh.md': (staging/name.name).write_bytes(payload)
            if len(verified) != len(records) or not legal:
                raise InputError('PAW package missing selected payloads or license')
            imported_manifest = dict(self.manifest, records=records, record_count=len(records),
                                     elements=sorted(requested), element_count=len(requested))
            (staging/'manifest.json').write_text(json.dumps(imported_manifest, indent=2)+'\n', encoding='utf-8')
            report = {'schema_version': 'sparse-dft-paw-library-import-1', 'status': 'PASS_DATA_IMPORT',
                      'archive': str(self.path), 'archive_sha256': package_sha,
                      'destination': str(destination), 'restored_records': len(verified),
                      'restored_elements': len(requested), 'records': verified,
                      'all_restored_original_sha256_verified': True,
                      'native_upf_execution_ready': False, 'mixed_paw_physics_validated': False}
            (staging/'import-report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
            staging.rename(destination)
            return report
        finally:
            if staging.exists(): shutil.rmtree(staging)
