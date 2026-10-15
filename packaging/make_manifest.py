"""Checksum manifest for a Sparse-DFT bundle (frozen app folder or staged resource tree).

    python packaging/make_manifest.py dist/Sparse-DFT --write      # write MANIFEST.sha256.json
    python packaging/make_manifest.py dist/Sparse-DFT --check      # verify sizes/hashes, exit 1 on drift
    python packaging/make_manifest.py . --layout-only        # check required layout entries only

The required entries follow ``packaging/layout.json``: the native executables
under ``bin/<platform_tag>/`` (or a CMake ``build/`` tree for a source
checkout), ``sparse_dft/sparsity_rules.json`` and the four GUI catalogs.  Files that
look like VASP POTCAR data are reported as errors: they must never ship.
Only the standard library is used, so the check runs on a clean target machine.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

MANIFEST = 'MANIFEST.sha256.json'
SCHEMA = 'sparse-dft-bundle-manifest-1'
CATALOGS = ('en', 'zh_CN', 'zh_TW', 'ja')
NATIVE = ('sdft_scf', 'sdft_relax_step')
SKIP_DIRS = {'__pycache__', '.git', '.venv'}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def resource_root(bundle: Path) -> Path:
    """PyInstaller 6 one-folder builds keep resources in ``_internal``; macOS apps in Contents/Resources."""
    for candidate in (bundle / '_internal', bundle / 'Contents' / 'Resources', bundle / 'Contents' / 'Frameworks'):
        if (candidate / 'sparse_dft_gui').is_dir() or (candidate / 'bin').is_dir():
            return candidate
    return bundle


def platform_tag() -> str:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    try:
        from sparse_dft.resources import platform_tag as tag
        return tag()
    finally:
        sys.path.pop(0)


def layout_errors(root: Path, tag: str) -> list[str]:
    errors = []
    suffix = '.exe' if tag.startswith('windows') else ''
    bins = [root / 'bin' / tag] + [root.joinpath(*s.split('/')) for s in
                                   ('build', 'build/Release', 'build/RelWithDebInfo', 'build/Debug')]
    if not any(all((d / (n + suffix)).is_file() for n in NATIVE) for d in bins):
        errors.append(f'missing native executables {", ".join(n + suffix for n in NATIVE)} in bin/{tag}/ or build/')
    for rel in ['sparse_dft/sparsity_rules.json'] + [f'sparse_dft_gui/i18n/{lang}.json' for lang in CATALOGS]:
        if not (root / rel).is_file():
            errors.append(f'missing {rel}')
    catalogs = {}
    for lang in CATALOGS:
        path = root / 'sparse_dft_gui' / 'i18n' / f'{lang}.json'
        if path.is_file():
            try:
                catalogs[lang] = set(json.loads(path.read_text(encoding='utf-8')))
            except ValueError as error:
                errors.append(f'{path.name}: invalid JSON ({error})')
    if catalogs and len({frozenset(v) for v in catalogs.values()}) > 1:
        errors.append('translation catalogs have different key sets')
    paw = root / 'data' / 'paw'
    if paw.is_dir():
        potcars = [p for p in paw.rglob('*') if p.is_file() and 'POTCAR' in p.name.upper()]
        if potcars:
            errors.append(f'licensed VASP POTCAR files must not be bundled: {[str(p) for p in potcars[:5]]}')
    return errors


def collect(bundle: Path) -> dict:
    files = {}
    for dirpath, dirnames, filenames in os.walk(bundle):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for name in sorted(filenames):
            path = Path(dirpath) / name
            rel = path.relative_to(bundle).as_posix()
            if rel == MANIFEST or path.is_symlink():
                continue
            files[rel] = {'sha256': sha256(path), 'size': path.stat().st_size}
    return files


def write(bundle: Path, tag: str) -> Path:
    payload = {'schema_version': SCHEMA, 'platform_tag': tag,
               'created': datetime.now(timezone.utc).isoformat(timespec='seconds'),
               'files': collect(bundle)}
    path = bundle / MANIFEST
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + '\n', encoding='utf-8')
    return path


def check(bundle: Path) -> list[str]:
    path = bundle / MANIFEST
    try:
        manifest = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as error:
        return [f'cannot read {path}: {error}']
    if manifest.get('schema_version') != SCHEMA:
        return [f'unexpected manifest schema {manifest.get("schema_version")!r}']
    errors = []
    actual = collect(bundle)
    for rel, entry in manifest.get('files', {}).items():
        found = actual.get(rel)
        if found is None:
            errors.append(f'missing {rel}')
        elif found != entry:
            errors.append(f'changed {rel}')
    for rel in sorted(set(actual) - set(manifest.get('files', {}))):
        errors.append(f'unlisted {rel}')
    return errors


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('bundle', help='frozen app folder (dist/Sparse-DFT, Sparse-DFT.app) or staged resource root')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--write', action='store_true', help=f'write {MANIFEST}')
    mode.add_argument('--check', action='store_true', help=f'verify {MANIFEST}')
    mode.add_argument('--layout-only', action='store_true', help='check required layout entries only')
    parser.add_argument('--platform-tag', help='target platform tag (default: this machine)')
    args = parser.parse_args(argv)
    bundle = Path(args.bundle).resolve()
    tag = args.platform_tag or platform_tag()
    errors = layout_errors(resource_root(bundle), tag)
    if args.write and not errors:
        print(f'wrote {write(bundle, tag)}')
    elif args.check:
        errors += check(bundle)
    for error in errors:
        print(f'ERROR: {error}', file=sys.stderr)
    if not errors:
        print(f'OK: {bundle} ({tag})')
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
