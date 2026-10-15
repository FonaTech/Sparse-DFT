"""Select PAW data by content while preserving each format's physical units."""
from pathlib import Path
import hashlib
import json
import tarfile

from .errors import InputError, UnsupportedError
from .upf import read_upf_paw
from .upf_native import export_native_upf
from .paw_library import PawLibraryPackage as PawLibrary


class SourceIdentity:
    def __init__(self, paths):
        self.paths = tuple(sorted(set(Path(p).resolve() for p in paths)))

    def local_hashes(self):
        return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in self.paths}


def resolve_source_map(structure, paw_root, workdir, family='standard'):
    """Content-dispatch a versioned per-element map, including pure limits."""
    root = Path(paw_root).expanduser().resolve()
    path = root / 'paw-sources.json' if root.is_dir() else root
    if not path.is_file() or path.suffix.lower() != '.json':
        return None
    try:
        manifest = json.loads(path.read_text(encoding='utf-8'))
    except (ValueError, OSError):
        return None
    if manifest.get('schema_version') != 'sparse-dft-paw-source-map-1':
        return None
    sources = manifest.get('sources')
    species = list(dict.fromkeys(structure.species))
    if not isinstance(sources, dict) or set(sources) != set(species):
        raise InputError('PAW map must cover exactly the structure species')
    from types import SimpleNamespace
    from .paw import Paw54JsonArchive, PawSelectionPolicy
    from .prepared_paw_cache import prepare
    choices, records, originals = {}, [], [path]
    for element in species:
        source = sources[element]
        if not isinstance(source, dict) or not isinstance(source.get('path'), str):
            raise InputError('PAW map entries require a source path')
        selected = Path(source['path']).expanduser()
        selected = (path.parent / selected).resolve() if not selected.is_absolute() else selected.resolve()
        if selected == path:
            raise InputError('recursive PAW source maps are not supported')
        target = Path(workdir) / 'sources' / element
        resolved = resolve_upf(SimpleNamespace(species=[element]), selected, target)
        if resolved is not None:
            local_choices, local_report, identity = resolved
            choice = local_choices[element]
            if source.get('record_id') and source['record_id'] != choice['record_id']:
                # UPF package record ids are filenames; native ids bind bytes.
                if source['record_id'] != Path(choice['original_upf']).name:
                    raise InputError('UPF record selection does not match original bytes')
            originals.extend(identity.paths)
            record = dict(local_report['records'][0])
            original_sha = record['sha256']
        else:
            archive = Paw54JsonArchive(selected)
            if source.get('record_id'):
                candidates = archive.candidates(element, include_nonstandard=True)
                candidate = next((c for c in candidates if c.record_id == source['record_id']), None)
                if candidate is None:
                    raise InputError('POTCAR map record is not a compatible PBE element')
            else:
                selected_choices, _ = PawSelectionPolicy(family).resolve(archive, [element])
                candidate = selected_choices[element]
            folder = target / 'paw_cache' / element
            prepare(archive, candidate.record_id, folder)
            choice = dict(cache=str((folder/'numeric.bin').resolve()), zval=candidate.zval,
                          enmax_ev=candidate.enmax_ev, record_id=candidate.record_id,
                          element=element, format='vasp-paw54')
            original_sha = candidate.potcar_sha256
            record = dict(element=element, format='vasp-paw54', path=str(selected),
                          record_id=candidate.record_id, sha256=original_sha,
                          native_cache_sha256=hashlib.sha256(Path(choice['cache']).read_bytes()).hexdigest(),
                          functional='PBE', zval=candidate.zval)
            originals.extend(archive.root / name for name in archive.local_hashes())
        if source.get('expected_sha256') and source['expected_sha256'] != original_sha:
            raise InputError('selected original PAW SHA-256 differs from source map')
        choices[element] = choice
        records.append(record)
    formats = {x['format'] for x in records}
    framework = 'mixed_paw' if len(formats) > 1 else 'upf_paw' if formats == {'upf-paw'} else 'potcar_paw'
    return choices, dict(selection_rule='content-verified per-element source map',
        framework=framework, shared_functional='PBE', records=records,
        physics_validated=False, global_production_validated=False), SourceIdentity(originals)


def resolve_upf(structure, paw_root, workdir):
    """Return None for the established POTCAR archive; ambiguity is an error."""
    root = Path(paw_root).expanduser().resolve()
    species = list(dict.fromkeys(structure.species))
    origin_paths = [root] if root.is_file() else []
    files = []
    package = False
    if root.is_file() and root.suffix.lower() in ('.upf', '.upfz'):
        files = [root]
    elif root.is_dir() and (root / 'manifest.json').is_file():
        try:
            manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
        except (ValueError, OSError):
            return None
        if 'records' not in manifest or 'source_archive_sha256' not in manifest:
            return None
        library = PawLibrary(root)
        files = [Path(r['upf_path']) for r in library.resolve(species)['records']]
        origin_paths.append(root / 'manifest.json')
        package = True
    elif root.is_file() and tarfile.is_tarfile(root):
        # Accept only the versioned PAW package manifest. A POTCAR archive is
        # handled by the established adapter; its files are never reinterpreted.
        try:
            library = PawLibrary(root)
        except InputError:
            return None
        restored = Path(workdir) / 'imported-paw-library'
        library.import_to(restored, elements=species)
        installed = PawLibrary(restored)
        files = [Path(r['upf_path']) for r in installed.resolve(species)['records']]
        package = True
    elif root.is_dir():
        files = sorted(root.rglob('*.UPF')) + sorted(root.rglob('*.upf'))
        if not files:
            return None
    else:
        return None
    candidates = {symbol: [] for symbol in species}
    for path in files:
        data = read_upf_paw(path)
        if data.element in candidates:
            candidates[data.element].append((path, data))
    resolved = {}
    records = []
    for symbol, options in candidates.items():
        if len(options) != 1:
            raise InputError('UPF selection needs one explicit variant per element',
                             details={'element': symbol, 'candidates': [str(p) for p, _ in options]})
        path, data = options[0]
        if data.functional != 'PBE' or data.relativistic != 'scalar' or data.summary()['has_soc']:
            raise UnsupportedError('Current UPF physical model requires scalar PBE PAW')
        folder = Path(workdir) / 'paw_cache' / symbol
        folder.mkdir(parents=True, exist_ok=True)
        cache = folder / 'original.sdftupf'
        export_native_upf(path, cache)
        recommendation = float(data.header.get('wfc_cutoff', 0) or 0)
        resolved[symbol] = {'cache': str(cache.resolve()), 'zval': float(data.header['z_valence']),
                            'enmax_ev': recommendation * 13.605693122994 if recommendation > 0 else None,
                            'record_id': data.source_sha256, 'element': symbol,
                            'format': 'upf-paw', 'original_upf': str(path.resolve())}
        records.append({'element': symbol, 'format': 'upf-paw', 'path': str(path.resolve()),
                        'sha256': data.source_sha256, 'native_cache_sha256': hashlib.sha256(cache.read_bytes()).hexdigest(),
                        'functional': data.functional, 'zval': resolved[symbol]['zval']})
        origin_paths.append(path)
    report = {'selection_rule': 'content-verified original UPF, one variant per species',
              'framework': 'upf_paw', 'tier_package': package, 'records': records,
              'physics_validated': False}
    return resolved, report, SourceIdentity(origin_paths)
