"""Cheap detection of what a calculation directory or file can plot.

Large result files (10 MB SCF ledgers, big vasprun.xml) are never parsed
here: a bounded head read plus an mmap substring scan is enough.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import mmap
import re

from ..analysis.common import schema_ids

SMALL_JSON_BYTES = 4 << 20
LARGE_BAND_JSON_BYTES = 64 << 20
MAX_DIR_JSON = 64


@dataclass
class SourceInfo:
    path: Path
    roles: set = field(default_factory=set)
    attrs: dict = field(default_factory=dict)


def _head(path: Path, n: int = 1 << 15) -> str:
    with open(path, 'rb') as handle:
        return handle.read(n).decode('utf-8', errors='replace')


def _contains(path: Path, *needles: bytes) -> dict:
    found = {n: False for n in needles}
    try:
        with open(path, 'rb') as handle:
            if path.stat().st_size == 0:
                return found
            with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as m:
                for n in needles:
                    found[n] = m.find(n) >= 0
    except (OSError, ValueError):
        pass
    return found


def _small_json(path: Path):
    if path.stat().st_size > SMALL_JSON_BYTES:
        return None
    try:
        with open(path, encoding='utf-8') as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def _scf_info(info: SourceInfo, head: str):
    p = info.path
    keys = _contains(p, b'"hf_total_forces_ev_per_angstrom"', b'"hf_total_stress_ev_per_angstrom3"',
                     b'"scf_kpoints"', b'"sparsity_ledger"', b'"band_energy_ev"', b'"weight"')
    info.roles.add('scf_result')
    converged = re.search(r'"converged"\s*:\s*(true|false)', head)
    spin = re.search(r'"spin_channels"\s*:\s*(\d)', head)
    info.attrs.update({
        'converged': bool(converged and converged.group(1) == 'true'),
        'spin': bool(spin and spin.group(1) == '2'),
        'forces': keys[b'"hf_total_forces_ev_per_angstrom"'],
        'stress': keys[b'"hf_total_stress_ev_per_angstrom3"'],
        'kpoints': keys[b'"scf_kpoints"'] and keys[b'"weight"'],
        'counters': keys[b'"sparsity_ledger"'],
        'ledger': keys[b'"band_energy_ev"']})
    if info.attrs['converged'] and info.attrs['kpoints']:
        info.roles.add('dos')


def _band_json(info: SourceInfo, data):
    if not isinstance(data, dict):
        return
    kpoints = data.get('kpoints')
    if not isinstance(kpoints, list) or not ('eigenvalues' in data or 'bands' in data):
        return
    # NSCF meshes (sparse-dft dos, SCF k sets) are not band paths
    if (data.get('metadata') or {}).get('sampling') in ('uniform_mesh', 'weighted_scf_kpoints'):
        info.roles.add('bands_mesh')
        return
    info.roles.add('bands_file')
    info.attrs['nkpoints'] = len(kpoints)
    info.attrs['projections'] = bool(data.get('projections') or data.get('projector_weights'))
    info.attrs['lm'] = bool(data.get('projection_lm_orbitals'))
    channels = data.get('spin_channels')
    info.attrs['spin'] = isinstance(channels, dict) and {'up', 'down'} <= set(channels)


def _large_band_json(info: SourceInfo, path: Path):
    """A Sparse-DFT band JSON above the small-JSON limit (lm projections make paths large): full parse up to 64 MB."""
    if path.stat().st_size > LARGE_BAND_JSON_BYTES:
        return
    try:
        with open(path, encoding='utf-8') as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return
    _band_json(info, data)


def classify(path) -> SourceInfo:
    """Roles of one file: scf_result, iterations, performance, bands_file, bands_mesh, dos, pdos,
    procar, relax, sweep, scaling."""
    from ..analysis.relax import is_relax_payload
    from ..analysis.sweep import is_scaling_payload, is_sweep_payload
    p = Path(path)
    info = SourceInfo(p)
    if not p.is_file():
        return info
    name = p.name
    lower = name.lower()
    try:
        if lower.endswith('.iterations.jsonl'):
            info.roles.add('iterations')
            first = _head(p, 1 << 14).splitlines()[:1]
            info.attrs['spin'] = bool(first and '"output_magnetic_moment_muB"' in first[0]
                                      and '"output_magnetic_moment_muB":null' not in first[0].replace(' ', ''))
        elif lower.endswith('.performance.json'):
            data = _small_json(p)
            if isinstance(data, dict) and isinstance(data.get('counters'), dict):
                info.roles.add('performance')
        elif name.upper().startswith('DOSCAR'):
            text = _head(p, 1 << 12).splitlines()
            info.roles.add('dos')
            if len(text) >= 6:
                head = text[5].split()
                try:
                    nedos = int(float(head[2]))
                    with open(p, 'rb') as handle:
                        nlines = sum(1 for _ in handle)
                    info.attrs['pdos'] = nlines > 6 + nedos + 1
                    if info.attrs['pdos']:
                        info.roles.add('pdos')
                except (IndexError, ValueError):
                    pass
        elif name.upper().startswith('PROCAR'):
            if '# of k-points' in _head(p, 1 << 12):
                info.roles.add('procar')
        elif name.upper().startswith('EIGENVAL'):
            lines = _head(p, 1 << 12).splitlines()
            try:
                nk = int(lines[5].split()[1])
            except (IndexError, ValueError):
                nk = 0
            info.roles.add('bands_file')
            info.attrs['nkpoints'] = nk
            try:
                info.attrs['spin'] = int(lines[0].split()[3]) == 2
            except (IndexError, ValueError):
                info.attrs['spin'] = False
        elif lower.endswith('.xml'):
            if '<modeling' not in _head(p, 1 << 12):
                return info
            keys = _contains(p, b'<dos>', b'<partial>', b'<eigenvalues>', b'comment="spin 2"', b'<projected>')
            if keys[b'<eigenvalues>']:
                info.roles.add('bands_file')
                count = len(re.findall(r'<v>', _kpointlist(p)))
                info.attrs['nkpoints'] = count
                info.attrs['spin'] = keys[b'comment="spin 2"']
                info.attrs['projections'] = keys[b'<projected>']
                info.attrs['lm'] = keys[b'<projected>']
            if keys[b'<dos>']:
                info.roles.add('dos')
            if keys[b'<partial>']:
                info.roles.add('pdos')
        elif lower.endswith('.json'):
            size = p.stat().st_size
            head = _head(p)
            if any(f'"{s}"' in head[:4096] for s in schema_ids('dos-1')):   # `sparse-dft dos` output (may be large)
                info.roles.add('dos')
                if _contains(p, b'"curves"')[b'"curves"']:
                    info.roles.add('pdos')
                    info.attrs['lm'] = True
                info.attrs['spin'] = '"total_up"' in head or _contains(p, b'"total_up"')[b'"total_up"']
                return info
            if size > SMALL_JSON_BYTES:
                if any(f'"{s}"' in head for s in schema_ids('scf-1')) or ('"total_energy_ev"' in head and '"iterations"' in head):
                    _scf_info(info, head)
                elif name in ('bands.json', 'bands_result.json'):
                    _large_band_json(info, p)       # lm projections make band paths large
                return info
            data = _small_json(p)
            if not isinstance(data, dict):
                return info
            from ..analysis.scf import is_scf_payload
            if is_scf_payload(data):
                _scf_info(info, head)
            elif is_relax_payload(data):
                info.roles.add('relax')
            elif is_scaling_payload(data):
                info.roles.add('scaling')
            elif is_sweep_payload(data):
                info.roles.add('sweep')
            else:
                _band_json(info, data)
    except OSError:
        pass
    return info


def _kpointlist(path: Path) -> str:
    """Text of the first <varray name="kpointlist"> block (bounded read)."""
    text = _head(path, 1 << 22)
    start = text.find('name="kpointlist"')
    if start < 0:
        return ''
    end = text.find('</varray>', start)
    return text[start:end if end > 0 else start]


CANDIDATE_NAMES = ('dos.json', 'scf_result.json', 'scf.json', 'relax_scf_evidence.json', 'bands_result.json', 'bands.json',
                   'vasprun.xml', 'EIGENVAL', 'DOSCAR', 'PROCAR', 'relax_result.json', 'relax_checkpoint.json',
                   'cell_relax_checkpoint.json', 'summary.json')


def scan(path) -> list[SourceInfo]:
    """Classify a file, or the recognised files directly inside a directory."""
    p = Path(path)
    if p.is_file():
        info = classify(p)
        return [info] if info.roles else []
    if not p.is_dir():
        return []
    files = []
    seen = set()
    for name in CANDIDATE_NAMES:
        if (p / name).is_file():
            files.append(p / name); seen.add(name)
    extra = sorted(x for x in p.iterdir() if x.is_file() and x.name not in seen and (
        x.name.endswith(('.iterations.jsonl', '.performance.json', '.xml')) or
        (x.suffix == '.json' and x.stat().st_size <= SMALL_JSON_BYTES)))
    files += extra[:MAX_DIR_JSON]
    out = []
    for f in files:
        info = classify(f)
        if info.roles:
            out.append(info)
    return out
