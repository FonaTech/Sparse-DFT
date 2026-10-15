"""Build the indexed PAW archive that Sparse-DFT reads from a user's licensed POTCAR set.

    sparse_dft paw-convert potpaw_PBE.54.tar.gz --out ~/paw/potpaw_PBE.54-json
    sparse_dft paw-convert /path/to/potpaw_PBE.54 --out ~/paw/potpaw_PBE.54-json

The source is a directory or a .tar/.tar.gz/.tgz/.tar.xz archive holding one
sub-directory per dataset (``Si``, ``Ga_d``, ``Ni_pv`` ...) with ``POTCAR`` and
optionally ``PSCTR``. The output directory receives ``manifest.json``,
``index.json``, ``potentials.jsonl.xz`` and ``auxiliary_files.json``. POTCAR
and PSCTR texts are stored byte for byte with their SHA-256; nothing is
rewritten. The archive stays subject to the licence of the POTCAR files: keep
it private, never redistribute it.
"""
from __future__ import annotations

import datetime
import hashlib
import io
import json
import lzma
import re
import tarfile
from pathlib import Path, PurePosixPath

from .errors import InputError
from .geometry import element

SCHEMA = '1.0'
FORMAT = 'VASP PAW JSONL'
_FLAGS = {'sv': 'has_semicore_sv', 'pv': 'has_semicore_pv', 'd': 'has_d_channel', 's': 'has_s_channel',
          'h': 'has_h_channel'}
_VARIANT_KEYS = ('name', 'tokens', 'family', 'is_standard', 'is_gw', 'is_new', 'has_semicore_sv',
                 'has_semicore_pv', 'has_d_channel', 'has_s_channel', 'has_h_channel', 'is_ae')
_PSCTR_KEYS = ('title', 'lpaw', 'r pacor', 'rpacor', 'rwigs', 'icore', 'ne', 'lcor', 'qcut')
_NUMBER = r'([-+]?\d*\.?\d+(?:[EeDd][-+]?\d+)?)'


def classify_variant(directory: str, symbol: str) -> dict:
    """Variant tokens and family from the dataset directory name (``Ga_d`` -> ``d``)."""
    name = directory[len(symbol):].lstrip('_')
    tokens = [t for t in name.split('_') if t]
    v = {'name': name or 'standard', 'tokens': tokens, 'is_standard': not tokens, 'is_gw': 'GW' in tokens,
         'is_new': 'new' in tokens, 'is_ae': 'AE' in tokens}
    for token, key in _FLAGS.items():
        v[key] = token in tokens
    semicore = v['has_semicore_sv'] or v['has_semicore_pv']
    if not tokens:
        family = 'standard'
    elif v['is_ae']:
        family = 'all_electron_like'
    elif v['is_new']:
        family = 'new_grid'
    elif v['is_gw']:
        family = 'gw_semicore' if semicore else 'gw'
    elif semicore:
        family = 'semicore'
    elif any(re.fullmatch(r'[0-9.]+', t) for t in tokens):
        family = 'numeric_variant'
    elif any(t in ('d', 's', 'h') for t in tokens):
        family = 'channel_variant'
    else:
        family = 'other'
    v['family'] = family
    return {k: v[k] for k in _VARIANT_KEYS}


def _float(pattern: str, text: str):
    m = re.search(pattern + r'\s*=\s*' + _NUMBER, text)
    return float(m.group(1).replace('D', 'E').replace('d', 'e')) if m else None


def potcar_metadata(text: str) -> dict:
    head = text.split('Description', 1)[0]
    title = next((ln.split('=', 1)[1].strip() for ln in head.splitlines() if 'TITEL' in ln), '')
    vrhfin = re.search(r'VRHFIN\s*=\s*([^\s:]+:?)', head)
    lexch = re.search(r'LEXCH\s*=\s*(\S+)', head)
    return {'title': title, 'vrhfin': vrhfin.group(1) if vrhfin else None,
            'exchange_correlation': lexch.group(1) if lexch else None,
            'valence_electrons': _float(r'ZVAL', head), 'atomic_mass': _float(r'POMASS', head),
            'enmax_ev': _float(r'ENMAX', head), 'enmin_ev': _float(r'ENMIN', head), 'eaug': _float(r'EAUG', head)}


def psctr_metadata(text: str) -> dict:
    if not text.strip():
        return {}
    head = text.split('Description', 1)[0]
    title = next((ln.split('=', 1)[1].strip() for ln in head.splitlines() if 'TITEL' in ln), '')
    word = lambda key: (re.search(key + r'\s*=\s*(\S+)', head) or [None, None])[1]
    out = {'title': title, 'lpaw': word('LPAW'), 'r pacor': None, 'rpacor': _float('RPACOR', head),
           'rwigs': _float('RWIGS', head), 'icore': _float('ICORE', head), 'ne': _float(r'\bNE', head),
           'lcor': word('LCOR'), 'qcut': _float('QCUT', head)}
    return {k: out[k] for k in _PSCTR_KEYS}


def _file_entry(raw: bytes | None) -> dict:
    raw = raw or b''
    return {'size_bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def _read_source(source: Path):
    """Yield (relative POSIX path, bytes) for every regular file of a directory or tar archive."""
    if source.is_dir():
        for path in sorted(source.rglob('*')):
            if path.is_file() and not path.is_symlink():
                yield PurePosixPath(path.relative_to(source).as_posix()), path.read_bytes()
        return
    if not tarfile.is_tarfile(source):
        raise InputError('PAW source must be a directory or a tar archive', details={'path': str(source)})
    with tarfile.open(source) as tar:
        for member in sorted(tar.getmembers(), key=lambda m: m.name):
            if member.isfile():  # links, devices and absolute paths are never followed
                name = PurePosixPath(member.name)
                if name.is_absolute() or '..' in name.parts:
                    raise InputError('unsafe path in PAW archive', details={'member': member.name})
                yield name, tar.extractfile(member).read()


def _strip_root(files: dict) -> dict:
    """Drop a single common top directory (``potpaw_PBE.54/Si/POTCAR`` -> ``Si/POTCAR``)."""
    tops = {p.parts[0] for p in files if len(p.parts) > 1}
    if len(tops) == 1 and not any(len(p.parts) == 1 for p in files) and \
            not any(p.parts[-1] == 'POTCAR' and len(p.parts) == 2 for p in files):
        return {PurePosixPath(*p.parts[1:]): raw for p, raw in files.items()}
    return files


def convert(source, output, *, overwrite: bool = False) -> dict:
    source, output = Path(source).expanduser(), Path(output).expanduser()
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise InputError('output directory is not empty', details={'path': str(output)})
    files = _strip_root(dict(_read_source(source)))
    datasets = sorted({p.parts[0] for p in files if len(p.parts) == 2 and p.parts[1] == 'POTCAR'})
    if not datasets:
        raise InputError('no <dataset>/POTCAR files found', details={'path': str(source)})
    records, auxiliary = [], []
    for directory in datasets:
        symbol = re.match(r'[A-Z][a-z]?', directory).group(0)
        try:
            el = element(symbol)
        except Exception as exc:
            raise InputError('dataset directory does not start with an element symbol',
                             details={'directory': directory}) from exc
        potcar = files[PurePosixPath(directory, 'POTCAR')]
        psctr = files.get(PurePosixPath(directory, 'PSCTR'))
        potcar_text, psctr_text = potcar.decode('ascii'), (psctr or b'').decode('ascii')
        variant = classify_variant(directory, symbol)
        rec = {'schema_version': SCHEMA, 'id': f"{symbol}:{variant['name']}", 'directory': directory,
               'element': {'symbol': symbol, 'name': el['name'].lower(), 'atomic_number': el['z'],
                           'period': el['period'], 'block': el['block']},
               'variant': variant,
               'metadata': {'potcar': potcar_metadata(potcar_text), 'psctr': psctr_metadata(psctr_text)},
               'files': {'POTCAR': {**_file_entry(potcar), 'text': potcar_text},
                         'PSCTR': {**_file_entry(psctr), 'text': psctr_text, 'empty': not psctr_text}}}
        records.append(rec)
    for path, raw in sorted(files.items()):
        if len(path.parts) == 1:
            try:
                auxiliary.append({'path': str(path), **_file_entry(raw), 'text': raw.decode('utf-8')})
            except UnicodeDecodeError:
                auxiliary.append({'path': str(path), **_file_entry(raw), 'text': None})
    index = {'schema_version': SCHEMA, 'by_element': {}, 'by_family': {}, 'by_tag': {}, 'records': []}
    for rec in records:
        index['by_element'].setdefault(rec['element']['symbol'], []).append(rec['id'])
        index['by_family'].setdefault(rec['variant']['family'], []).append(rec['id'])
        for tag in rec['variant']['tokens']:
            index['by_tag'].setdefault(tag, []).append(rec['id'])
        index['records'].append({**{k: rec[k] for k in ('id', 'directory', 'element', 'variant', 'metadata')},
                                 'files': {k: {kk: vv for kk, vv in v.items() if kk != 'text'}
                                           for k, v in rec['files'].items()}})
    for key in ('by_element', 'by_family', 'by_tag'):
        index[key] = {k: sorted(v) for k, v in sorted(index[key].items())}
    output.mkdir(parents=True, exist_ok=True)
    data = output / 'potentials.jsonl.xz'
    with lzma.open(data, 'wt', encoding='utf-8', preset=1) as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + '\n')
    families = {k: len(v) for k, v in index['by_family'].items()}
    manifest = {'schema_version': SCHEMA, 'format': FORMAT,
                'description': 'Lossless JSON Lines records of a licensed PAW set; POTCAR/PSCTR texts are stored byte for byte.',
                'source': {'path': source.name, 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()
                           if source.is_file() else None},
                'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'record_count': len(records), 'file_count': len(files),
                'auxiliary_file_count': len(auxiliary),
                'records': {'jsonl_primary': data.name, 'compression_primary': 'xz -1', 'encoding': 'UTF-8',
                            'line_format': 'one JSON object per PAW variant'},
                'categories': {'families': dict(sorted(families.items())), 'elements': len(index['by_element']),
                               'tags': {k: len(v) for k, v in index['by_tag'].items()}},
                'integrity': {'source_file_sha256_count': len(files),
                              'empty_psctr_records': [r['id'] for r in records if r['files']['PSCTR']['empty']]},
                'sizes_bytes': {data.name: data.stat().st_size},
                'licence_note': 'Derived from licensed PAW datasets. Keep private; do not redistribute.'}
    (output / 'index.json').write_text(json.dumps(index, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    (output / 'auxiliary_files.json').write_text(json.dumps({'schema_version': SCHEMA, 'files': auxiliary},
                                                            indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    return {'output': str(output), 'records': len(records), 'elements': len(index['by_element']),
            'families': families, 'empty_psctr': manifest['integrity']['empty_psctr_records']}
