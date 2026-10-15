"""Shared helpers for the Sparse-DFT analysis loaders.

Everything here is pure Python (no numpy, no matplotlib) so the loaders can
be used by the CLI, the GUI worker and headless report tools alike.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import hashlib
import json
import math

from ..errors import InputError

#: 1 eV/Å^3 expressed in GPa (CODATA 2014 elementary charge, exact SI prefixes).
EV_PER_ANGSTROM3_TO_GPA = 160.21766208

#: Schema prefix of the files this package writes, taken from a full id so a
#: release rename of the id ("<prefix>scf-1") renames the prefix with it.
SCHEMA_PREFIX = 'sparse-dft-scf-1'.removesuffix('scf-1')
# Files written before the project rename carry the original prefix.  It is
# assembled from parts so a release rename of SCHEMA_PREFIX leaves it intact.
_LEGACY_SCHEMA_PREFIX = 'd' + 'ftx-'


def schema_ids(name: str) -> tuple[str, ...]:
    """Accepted ``schema_version`` values for ``name`` (e.g. 'scf-1'): current prefix first, then legacy."""
    return tuple(dict.fromkeys((SCHEMA_PREFIX + name, _LEGACY_SCHEMA_PREFIX + name)))


_JSON_CACHE: "OrderedDict[tuple, object]" = OrderedDict()
_SHA_CACHE: "OrderedDict[tuple, str]" = OrderedDict()
_CACHE_SIZE = 8


def _cache_key(path) -> tuple:
    p = Path(path).resolve()
    st = p.stat()
    return (str(p), st.st_mtime_ns, st.st_size)


def _remember(cache, key, value):
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > _CACHE_SIZE:
        cache.popitem(last=False)
    return value


def read_json(path):
    """Parse a JSON file once per (path, mtime, size); callers must not mutate it."""
    try:
        key = _cache_key(path)
    except FileNotFoundError:
        raise InputError(f'file not found: {path}') from None
    if key in _JSON_CACHE:
        _JSON_CACHE.move_to_end(key)
        return _JSON_CACHE[key]
    try:
        with open(key[0], encoding='utf-8') as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise InputError(f'invalid JSON in {path}: {exc}') from None
    return _remember(_JSON_CACHE, key, data)


def read_jsonl(path) -> list[dict]:
    """Read one JSON object per line; blank lines are ignored."""
    records = []
    with open(path, encoding='utf-8') as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise InputError(f'invalid JSON on line {number} of {path}: {exc}') from None
            if not isinstance(value, dict):
                raise InputError(f'line {number} of {path} is not a JSON object')
            records.append(value)
    return records


def sha256_file(path) -> str | None:
    """SHA-256 of a file's bytes (cached per path/mtime/size); None if absent."""
    try:
        key = _cache_key(path)
    except (FileNotFoundError, OSError):
        return None
    if key in _SHA_CACHE:
        _SHA_CACHE.move_to_end(key)
        return _SHA_CACHE[key]
    digest = hashlib.sha256()
    with open(key[0], 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return _remember(_SHA_CACHE, key, digest.hexdigest())


def finite(value) -> float | None:
    """Return ``float(value)`` when it is a finite number, else None."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def finite_list(values) -> list[float | None]:
    return [finite(v) for v in (values or [])]


def resolve_fermi(data_value, override, source, *, purpose='an E - E_F axis'):
    """Return (fermi_ev, origin) or raise a clear error; never substitutes 0 eV."""
    if override is not None:
        value = finite(override)
        if value is None:
            raise InputError(f'fermi override must be a finite energy in eV, got {override!r}')
        return value, 'option'
    value = finite(data_value)
    if value is not None:
        return value, 'data'
    raise InputError(
        f'Fermi energy is unknown for {source}; it is required for {purpose}. '
        "Pass fermi=<eV> (CLI: --fermi <eV>), or use align='vbm' / align='none' "
        '(CLI: --align vbm|none).',
        details={'source': str(source), 'option': 'fermi', 'cli_flag': '--fermi'})
