"""Band-structure parsing, normalization, and comparison utilities.

The comparison layer deliberately separates path alignment, energy-zero
alignment, and metric calculation.  MP and a local run may use different
Fermi levels, band counts, or path sampling; silently broadcasting arrays in
those cases would produce meaningless validation numbers.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import math
import re
import xml.etree.ElementTree as ET

from .errors import InputError
import hashlib
import csv


@dataclass
class BandStructure:
    source: str
    kpoints: list[tuple[float, float, float]]
    distances: list[float]
    eigenvalues: list[list[float]]
    labels: list[str | None]
    fermi_energy: float | None = None
    vbm: float | None = None
    cbm: float | None = None
    metadata: dict | None = None
    # Optional spin-resolved channels.  Each value is [kpoint][band] in eV.
    spin_channels: dict[str, list[list[float]]] | None = None
    # Occupancies are kept separately from eigenvalues.  This is essential
    # for distinguishing a metal from a tiny numerical gap around E_F.
    occupations: dict[str, list[list[float]]] | None = None
    # VASP k-point weights, when the source format provides them.
    kweights: list[float] | None = None
    # ``insulator`` is only assigned when occupations support that claim;
    # ``inferred`` means a Fermi-level threshold was used, while ``unknown``
    # and ``metal`` deliberately do not expose a fabricated band gap.
    gap_status: str = "unknown"
    projections: dict | None = None

    @property
    def nkpoints(self):
        return len(self.kpoints)

    @property
    def nbands(self):
        return len(self.eigenvalues[0]) if self.eigenvalues else 0

    @property
    def band_gap(self):
        if self.gap_status in ("metal", "unknown", "metal_or_numerical"):
            return None
        if self.vbm is None or self.cbm is None:
            return None
        return self.cbm - self.vbm

    def as_dict(self):
        payload = asdict(self) | {'nkpoints': self.nkpoints, 'nbands': self.nbands, 'band_gap': self.band_gap}
        return payload


_FIXED_FLOAT = re.compile(r'[-+]?(?:\d+\.\d*|\.\d+)(?:[eEdD][-+]?\d+)?')


def _fixed_width_floats(text):
    """Floats from Fortran fixed-width fields, which run together when negative
    ('-0.01315789-0.01315789' in VASP PROCAR k-point lines)."""
    return [float(x.replace('D', 'E').replace('d', 'e')) for x in _FIXED_FLOAT.findall(text)]


def _float_tokens(line):
    values = []
    for token in line.replace('D', 'E').replace('d', 'e').split():
        try:
            values.append(float(token))
        except ValueError:
            pass
    return values


def _distances(kpoints, reciprocal_basis=None):
    """Return cumulative k-path distance.

    Fractional-coordinate Euclidean distance is only a useful fallback.  For
    a real band plot the distance must be measured in reciprocal Cartesian
    space, otherwise skewed cells and non-cubic paths get visibly distorted.
    ``reciprocal_basis`` contains the three reciprocal vectors in inverse
    length units; VASP's XML values are accepted as-is because only the
    relative axis scale matters for plotting and path alignment.
    """
    basis = None
    if reciprocal_basis is not None:
        try:
            rows = [tuple(float(x) for x in row[:3]) for row in reciprocal_basis]
            if len(rows) == 3 and all(len(row) == 3 and all(math.isfinite(x) for x in row) for row in rows):
                basis = rows
        except (TypeError, ValueError, IndexError):
            basis = None

    def cart(point):
        if basis is None:
            return tuple(float(x) for x in point[:3])
        return tuple(sum(float(point[i]) * basis[i][j] for i in range(3)) for j in range(3))

    out = [0.0]
    cartesian = [cart(point) for point in kpoints]
    for left, right in zip(cartesian, cartesian[1:]):
        out.append(out[-1] + math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right))))
    return out


def _infer_edges(eigenvalues, fermi, occupations=None, spin_channels=None, *, gap_tolerance=1e-4):
    """Return (VBM, CBM, status) without turning E_F roundoff into a gap.

    VASP occupations are the authoritative source.  A partially occupied
    state is metallic even when a rounded Fermi level would suggest a small
    positive gap.  Formats without occupations can only report an inferred
    gap, and gaps below ``gap_tolerance`` are treated as numerical/metallic.
    """
    if occupations:
        occupied, empty = [], []
        partial = False
        for channel, ev_rows in _channel_rows(eigenvalues, occupations, spin_channels):
            occ_rows = occupations.get(channel)
            if occ_rows is None or len(occ_rows) != len(ev_rows):
                return None, None, "unknown"
            for erow, orow in zip(ev_rows, occ_rows):
                if len(erow) != len(orow):
                    return None, None, "unknown"
                for energy, occ in zip(erow, orow):
                    if not math.isfinite(occ):
                        return None, None, "unknown"
                    # Spin-resolved VASP occupations are normally [0, 1],
                    # but tolerate the non-spin [0, 2] convention.
                    scale = 1.0 if occ <= 1.0 + 1e-8 else 2.0
                    frac = occ / scale
                    if 1e-7 < frac < 1.0 - 1e-7:
                        partial = True
                    elif frac >= 1.0 - 1e-7:
                        occupied.append(energy)
                    else:
                        empty.append(energy)
        if partial:
            return None, None, "metal"
        if occupied and empty:
            vbm, cbm = max(occupied), min(empty)
            if cbm - vbm > gap_tolerance:
                return vbm, cbm, "insulator"
            return None, None, "metal_or_numerical"
        return None, None, "unknown"
    if fermi is None:
        return None, None, "unknown"
    below = [value for row in eigenvalues for value in row if value <= fermi + 1e-8]
    above = [value for row in eigenvalues for value in row if value > fermi + 1e-8]
    if not below or not above:
        return None, None, "unknown"
    vbm, cbm = max(below), min(above)
    return (vbm, cbm, "inferred") if cbm - vbm > gap_tolerance else (None, None, "metal_or_numerical")


def _channel_rows(eigenvalues, occupations, spin_channels=None):
    """Iterate channels while accepting the legacy single-channel layout."""
    if occupations and all(isinstance(v, list) for v in occupations.values()):
        if len(occupations) == 1:
            key = next(iter(occupations))
            return [(key, eigenvalues)]
        if spin_channels:
            return [(key, spin_channels[key]) for key in occupations if key in spin_channels]
    return [("up", eigenvalues)]


def _labels_from_payload(payload, kpoints):
    labels = payload.get('labels')
    if labels and len(labels) == len(kpoints):
        return labels
    mapping = payload.get('labels_dict') or {}
    if not mapping:
        return [None] * len(kpoints)
    out = [None] * len(kpoints)
    for name, coord in mapping.items():
        try: c = tuple(float(x) for x in coord[:3])
        except Exception: continue
        for i, kp in enumerate(kpoints):
            # Fractional paths often differ by signed zero or roundoff.
            if sum((kp[j] - c[j]) ** 2 for j in range(3)) < 1e-8:
                out[i] = name
    return out


def from_json(path: str | Path) -> BandStructure:
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    # Pymatgen's public BandStructureSymmLine serialization stores bands as
    # {spin: [band][kpoint]} and calls the Fermi level ``efermi``.
    spin_channels = payload.get('spin_channels')
    if 'eigenvalues' not in payload and 'bands' in payload and 'kpoints' in payload:
        spin_keys = sorted(payload['bands'])
        if not spin_keys:
            raise InputError('pymatgen band JSON contains no spin channel')
        spin_channels = {}
        for key in spin_keys:
            band_major = payload['bands'][key]
            if not band_major or not band_major[0]:
                continue
            spin_channels[str(key)] = [list(row) for row in zip(*band_major)]
        band_major = payload['bands'][spin_keys[0]]
        if not band_major or not band_major[0]:
            raise InputError('pymatgen band JSON contains empty bands')
        payload = dict(payload)
        payload['eigenvalues'] = [list(row) for row in zip(*band_major)]
        payload['fermi_energy'] = payload.get('efermi')
    if 'eigenvalues' not in payload or 'kpoints' not in payload:
        raise InputError('band JSON requires kpoints and eigenvalues')
    kpoints = [tuple(float(x) for x in row[:3]) for row in payload['kpoints']]
    eigenvalues = [[float(x) for x in row] for row in payload['eigenvalues']]
    if len(kpoints) != len(eigenvalues) or not eigenvalues:
        raise InputError('band kpoint/eigenvalue dimensions do not match')
    nbands = len(eigenvalues[0])
    if not nbands or any(len(row) != nbands for row in eigenvalues):
        raise InputError('band eigenvalue rows have inconsistent lengths')
    spin_channels = _checked_spin_channels(spin_channels, len(kpoints), nbands)
    fermi = payload.get('fermi_energy', payload.get('fermi_energy_ev'))
    vbm, cbm = payload.get('vbm'), payload.get('cbm')
    if isinstance(vbm, dict): vbm = vbm.get('energy')
    if isinstance(cbm, dict): cbm = cbm.get('energy')
    occupations = payload.get('occupations')
    if occupations is not None:
        occupations = {str(k): [[float(x) for x in row] for row in rows] for k, rows in occupations.items()}
    if occupations is not None:
        vbm, cbm, gap_status = _infer_edges(eigenvalues, float(fermi) if fermi is not None else None,
                                            occupations, spin_channels)
    elif vbm is None or cbm is None:
        vbm, cbm, gap_status = _infer_edges(eigenvalues, float(fermi) if fermi is not None else None)
    else:
        gap_status = 'inferred'
    metadata = (payload.get('metadata') or {}) | ({'spin_channels': list(spin_channels)} if spin_channels else {})
    projections = payload.get('projections')
    if projections is None and 'projector_weights' in payload:
        projections = {'schema_version': 'sparse-dft-projector-1',
                       'projection_orbitals': payload.get('projection_orbitals', []),
                       'projection_atom_symbols': payload.get('projection_atom_symbols', []),
                       'projection_weights': payload.get('projector_weights', []),
                       'spin_projection_available': bool(payload.get('spin_projection_available', False)),
                       'weight_definition': payload.get('projection_weight_definition') or
                       'raw_sum_abs_Bpsi_squared; not normalized probabilities or orbital occupations',
                       'nkpoints': len(kpoints), 'nbands': nbands}
        if payload.get('projection_lm_orbitals') and 'projector_lm_weights' in payload:
            projections.update(projection_lm_orbitals=payload['projection_lm_orbitals'],
                               projection_lm_weights=payload['projector_lm_weights'],
                               projection_lm_weight_definition=payload.get('projection_lm_weight_definition'))
    reciprocal_basis = metadata.get('reciprocal_basis')
    return BandStructure(str(path), kpoints, payload.get('distances') or _distances(kpoints, reciprocal_basis), eigenvalues,
                         _labels_from_payload(payload, kpoints), fermi, vbm, cbm,
                         metadata,
                         spin_channels, occupations, payload.get('kweights'), gap_status,
                         projections)


def from_payload(payload: dict, source: str = '<memory>') -> BandStructure:
    """Parse an in-memory pymatgen/Sparse-DFT band payload with the same checks as JSON."""
    tmp = Path(source)
    if source != '<memory>' and tmp.exists():
        return from_json(tmp)
    data = dict(payload)
    channels = data.get('spin_channels')
    if 'eigenvalues' not in data and 'bands' in data:
        keys = sorted(data['bands'])
        if not keys: raise InputError('band payload has no spin channels')
        channels = {str(k): [list(row) for row in zip(*data['bands'][k])] for k in keys}
        data['eigenvalues'] = channels[keys[0]]
        data['fermi_energy'] = data.get('efermi', data.get('fermi_energy'))
    if 'kpoints' not in data or 'eigenvalues' not in data:
        raise InputError('band payload requires kpoints and eigenvalues')
    kpoints = [tuple(float(x) for x in k[:3]) for k in data['kpoints']]
    ev = [[float(x) for x in row] for row in data['eigenvalues']]
    if not kpoints or len(kpoints) != len(ev): raise InputError('band payload dimensions mismatch')
    nb = len(ev[0]);
    if not nb or any(len(row) != nb for row in ev): raise InputError('band payload has ragged bands')
    channels = _checked_spin_channels(channels,len(kpoints),nb)
    ef = data.get('fermi_energy', data.get('fermi_energy_ev', data.get('efermi')))
    occupations = data.get('occupations')
    if occupations is not None:
        occupations = {str(k): [[float(x) for x in row] for row in rows] for k, rows in occupations.items()}
    vbm, cbm, gap_status = _infer_edges(ev, float(ef) if ef is not None else None, occupations, channels if 'channels' in locals() else None)
    return BandStructure(source, kpoints, data.get('distances') or _distances(kpoints), ev,
                         _labels_from_payload(data, kpoints), ef, vbm, cbm,
                         data.get('metadata') or {}, channels if 'channels' in locals() else None,
                         occupations, data.get('kweights'), gap_status, data.get('projections'))


def _checked_spin_channels(channels, nkpoints, nbands):
    if channels is None:return None
    if not isinstance(channels,dict) or not channels:raise InputError('invalid spin band channels')
    result={}
    for key,rows in channels.items():
        name={'1':'up','+1':'up','-1':'down'}.get(str(key),str(key))
        if name in result:raise InputError('duplicate spin band channel')
        if len(rows)!=nkpoints or any(len(row)!=nbands for row in rows):
            raise InputError('spin band dimensions do not match')
        values=[[float(v) for v in row] for row in rows]
        if not all(math.isfinite(v) for row in values for v in row):raise InputError('nonfinite spin energies')
        result[name]=values
    return result


def parse_eigenval(path: str | Path, fermi_energy: float | None = None) -> BandStructure:
    """Parse VASP EIGENVAL, retaining both spin channels when present."""
    lines = Path(path).read_text(encoding='utf-8', errors='strict').splitlines()
    if len(lines) < 7:
        raise InputError(f'EIGENVAL is too short: {path}')
    header = _float_tokens(lines[5])
    if len(header) < 3:
        raise InputError('EIGENVAL header lacks electron/kpoint/band counts')
    nkpoints, nbands = int(header[1]), int(header[2])
    kpoints, eigenvalues = [], []
    spin_up, spin_down = [], []
    occ_up, occ_down = [], []
    kweights = []
    spin_polarized = None
    i = 6
    while i < len(lines) and len(kpoints) < nkpoints:
        while i < len(lines) and not lines[i].strip():
            i += 1
        if i >= len(lines):
            break
        kp = _float_tokens(lines[i]); i += 1
        if len(kp) < 3:
            raise InputError(f'invalid EIGENVAL k-point line {i}')
        kpoints.append(tuple(kp[:3])); row = []; occ_row = []
        # The fourth value on a VASP k-point line is its integration weight.
        if len(kp) >= 4:
            kweights.append(float(kp[3]))
        for _ in range(nbands):
            if i >= len(lines):
                raise InputError('EIGENVAL truncated in band block')
            values = _float_tokens(lines[i]); i += 1
            if len(values) not in (3, 5):
                raise InputError(f'invalid EIGENVAL band line {i}: expected 3 or 5 columns')
            line_spin = len(values) == 5
            if spin_polarized is not None and spin_polarized != line_spin:
                raise InputError(f'inconsistent EIGENVAL spin columns at line {i}')
            if values[0] != len(row) + 1:
                raise InputError(f'invalid EIGENVAL band index at line {i}')
            if line_spin:
                spin_polarized = True
                row.append(values[1]); occ_row.append(values[3])
                # VASP writes (band, E_up, E_down, occ_up, occ_down).
                # Keep a separate down channel without changing the public
                # non-spin eigenvalue representation.
                if len(spin_down) < len(kpoints):
                    spin_down.append([])
                    occ_down.append([])
                spin_down[-1].append(values[2])
                occ_down[-1].append(values[4])
            else:
                spin_polarized = False
                row.append(values[1])
                occ_row.append(values[2])
        eigenvalues.append(row)
        occ_up.append(occ_row)
    if len(kpoints) != nkpoints:
        raise InputError(f'EIGENVAL has {len(kpoints)} kpoints; expected {nkpoints}')
    occupations = {'up': occ_up}
    channels = None
    if spin_polarized:
        channels = {'up': eigenvalues, 'down': spin_down}
        occupations['down'] = occ_down
    vbm, cbm, gap_status = _infer_edges(eigenvalues, fermi_energy, occupations, channels)
    return BandStructure(str(path), kpoints, _distances(kpoints), eigenvalues,
                         [None] * len(kpoints), fermi_energy, vbm, cbm,
                         {'format': 'VASP EIGENVAL', 'declared_nkpoints': nkpoints, 'declared_nbands': nbands,
                          'spin_polarized': bool(spin_polarized), 'spin_channels': list(channels or {}),
                          'occupations_present': True}, channels, occupations,
                         kweights if len(kweights) == len(kpoints) else None, gap_status)


def parse_vasprun_bands(path: str | Path) -> BandStructure:
    """Read the final VASP calculation, retaining spin and occupations."""
    root = ET.parse(path).getroot()
    calculations = root.findall('./calculation')
    calculations_with_eigs = [c for c in calculations if c.find('.//eigenvalues/array') is not None]
    container = calculations_with_eigs[-1] if calculations_with_eigs else (calculations[-1] if calculations else root)
    fermi_node = container.find('.//i[@name="efermi"]')
    if fermi_node is None:
        fermi_nodes = root.findall('.//i[@name="efermi"]')
        fermi_node = fermi_nodes[-1] if fermi_nodes else None
    fermi = float(fermi_node.text) if fermi_node is not None and fermi_node.text else None

    kpoint_lists = root.findall('./kpoints/varray[@name="kpointlist"]') or container.findall('.//varray[@name="kpointlist"]')
    kpoint_list = kpoint_lists[-1] if kpoint_lists else None
    kpoints = [tuple((_float_tokens(v.text or '') + [0.0, 0.0, 0.0])[:3])
               for v in (kpoint_list.findall('v') if kpoint_list is not None else [])]
    weights_node = root.find('./kpoints/varray[@name="weights"]')
    kweights = [(_float_tokens(v.text or '') or [0.0])[0] for v in weights_node.findall('v')] if weights_node is not None else []

    # Select one final/initial structure.  VASP rec_basis values in this XML
    # are inverse-length values without 2pi; apply the factor exactly once.
    structures = root.findall('./structure')
    structure = next((s for s in reversed(structures) if s.get('name') == 'finalpos'), None)
    if structure is None:
        structure = next((s for s in reversed(structures) if s.get('name') in ('initialpos', 'primitive_cell')), None)
    if structure is None and structures:
        structure = structures[-1]
    rec_nodes = structure.findall('./crystal/varray[@name="rec_basis"]') if structure is not None else []
    reciprocal_basis = []
    for v in (rec_nodes[-1].findall('v') if rec_nodes else []):
        values = _float_tokens(v.text or '')
        if len(values) >= 3:
            reciprocal_basis.append(tuple(2.0 * math.pi * x for x in values[:3]))

    eig_arrays = container.findall('./eigenvalues/array')
    eig_array = eig_arrays[-1] if eig_arrays else container.find('.//eigenvalues/array')
    spin_sets = []
    if eig_array is not None:
        spin_sets = [x for x in eig_array.findall('./set/set') if x.get('comment', '').lower().startswith('spin')]
        if not spin_sets:
            first = eig_array.find('./set')
            if first is not None:
                spin_sets = [first]
    spin_channels, occupations = {}, {}
    for spin_index, spin_set in enumerate(spin_sets, start=1):
        erows, orows = [], []
        ksets = spin_set.findall('./set') or ([spin_set] if spin_set.findall('./r') else [])
        for kset in ksets:
            erow, orow = [], []
            for v in kset.findall('./r'):
                values = _float_tokens(v.text or '')
                if values:
                    erow.append(values[0]); orow.append(values[1] if len(values) > 1 else float('nan'))
            if erow:
                erows.append(erow); orows.append(orow)
        key = 'up' if spin_index == 1 else 'down' if spin_index == 2 else f'spin{spin_index}'
        spin_channels[key] = erows
        occupations[key] = orows
    if not spin_channels:
        raise InputError('vasprun.xml contains no eigenvalue sets in final calculation')
    eigenvalues = spin_channels['up']
    occupations = occupations if all(math.isfinite(x) for rows in occupations.values() for row in rows for x in row) else None
    if not kpoints or len(kpoints) != len(eigenvalues):
        raise InputError(f'vasprun band dimensions mismatch: kpoints={len(kpoints)}, eigenvalue sets={len(eigenvalues)}')
    vbm, cbm, gap_status = _infer_edges(eigenvalues, fermi, occupations, spin_channels)
    metadata = {'format': 'vasprun.xml', 'spin_channels': list(spin_channels), 'final_calculation': bool(calculations)}
    if len(reciprocal_basis) == 3:
        metadata['reciprocal_basis'] = reciprocal_basis
        metadata['reciprocal_basis_includes_2pi'] = True
    return BandStructure(str(path), kpoints, _distances(kpoints, reciprocal_basis), eigenvalues, [None] * len(kpoints),
                         fermi, vbm, cbm, metadata, spin_channels, occupations,
                         kweights if len(kweights) == len(kpoints) else None, gap_status)


def load_bands(path: str | Path) -> BandStructure:
    path = Path(path)
    if path.is_dir():
        for name in ('vasprun.xml', 'bands.json', 'EIGENVAL'):
            candidate = path / name
            if candidate.is_file():
                path = candidate
                break
        else:
            raise InputError(f'no supported band output found in directory: {path}')
    if path.suffix.lower() == '.json':
        return from_json(path)
    if path.name.lower() == 'eigenval':
        fermi = None
        outcar = path.with_name('OUTCAR')
        if outcar.is_file():
            matches = re.findall(r'E-fermi\s*:\s*([-+0-9.eEdD]+)', outcar.read_text(encoding='utf-8', errors='ignore'))
            if matches: fermi = float(matches[-1].replace('D', 'E').replace('d', 'e'))
        return parse_eigenval(path, fermi_energy=fermi)
    if path.suffix.lower() == '.xml':
        return parse_vasprun_bands(path)
    raise InputError(f'unsupported band file: {path}')


def parse_doscar(path: str | Path) -> dict:
    """Parse VASP DOSCAR total DOS and optional atom projected DOS.

    The parser keeps raw columns and labels rather than guessing orbital
    semantics.  This is intentional: LORBIT settings change the PDOS column
    layout and silently relabelling them would corrupt a scientific figure.
    """
    p = Path(path)
    lines = p.read_text(encoding='utf-8', errors='strict').splitlines()
    if len(lines) < 7:
        raise InputError(f'DOSCAR is too short: {p}')
    head = _float_tokens(lines[5])
    if len(head) < 4:
        raise InputError('DOSCAR energy header lacks Emax/Emin/NEDOS/Efermi')
    emax, emin, nedos, efermi = head[:4]
    nedos = int(round(nedos))
    start = 6
    total_rows = []
    for line in lines[start:start + nedos]:
        vals = _float_tokens(line)
        if len(vals) < 2:
            raise InputError('DOSCAR total DOS row has too few columns')
        total_rows.append(vals)
    energies = [r[0] for r in total_rows]
    total = {'total': [r[1] for r in total_rows]}
    spin_total = False
    if total_rows and len(total_rows[0]) >= 5:
        # VASP spin DOSCAR: E, DOS_up, DOS_down, int_up, int_down.
        spin_total = True
        total = {
            'total_up': [r[1] for r in total_rows],
            'total_down': [r[2] for r in total_rows],
            'integrated_up': [r[3] for r in total_rows],
            'integrated_down': [r[4] for r in total_rows],
        }
    elif total_rows and len(total_rows[0]) >= 3:
        total['integrated'] = [r[2] for r in total_rows]
    pdos = []
    i = start + nedos
    # Atom projected blocks start with a six-number header, followed by NEDOS
    # rows.  Preserve columns as orbital_1, orbital_2, ... unless a textual
    # header is available in the file.
    while i < len(lines):
        hdr = _float_tokens(lines[i])
        if len(hdr) < 4:
            i += 1; continue
        block_header = lines[i].strip(); i += 1
        rows = []
        for line in lines[i:i + nedos]:
            vals = _float_tokens(line)
            if vals: rows.append(vals)
        if len(rows) < nedos:
            break
        ncols = len(rows[0]) - 1
        columns = {f'orbital_{j + 1}': [r[j + 1] for r in rows] for j in range(ncols)}
        pdos.append({'atom_index': len(pdos) + 1, 'header': block_header,
                     'energies_ev': [r[0] for r in rows], 'columns': columns})
        i += nedos
    return {'schema_version': 'sparse-dft-dos-1', 'source': str(p),
            'source_sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
            'fermi_energy_ev': efermi, 'emin_ev': emin, 'emax_ev': emax,
            'nedos': nedos, 'energies_ev': energies, 'total': total,
            'pdos': pdos, 'metadata': {'format': 'VASP DOSCAR', 'n_pdos_blocks': len(pdos),
                                        'spin_polarized': spin_total}}


def density_of_states_from_bands(bands: BandStructure, *, emin=None, emax=None,
                                 sigma=0.05, npoints=1201) -> dict:
    """Integrate a weighted Brillouin-zone sample, never a symmetry line.

    Spin-degenerate eigenvalues count twice; explicit collinear channels
    count once each. Sampling convergence is a separate required study.
    """
    if not math.isfinite(sigma) or sigma <= 0:
        raise InputError('DOS Gaussian sigma must be finite and positive')
    if not isinstance(npoints, int) or not 3 <= npoints <= 100000:
        raise InputError('DOS requires 3..100000 energy points')
    if (bands.metadata or {}).get('sampling') not in ('uniform_mesh', 'weighted_scf_kpoints'):
        raise InputError('bulk DOS requires a weighted Brillouin-zone mesh; a symmetry path is insufficient')
    weights = bands.kweights
    if weights is None or len(weights) != bands.nkpoints or any(not math.isfinite(w) or w < 0 for w in weights) or sum(weights) <= 0:
        raise InputError('DOS requires finite nonnegative k-point weights with positive sum')
    weights = [w / sum(weights) for w in weights]
    channels = bands.spin_channels or {'total': bands.eigenvalues}
    degeneracy = 1.0 if len(channels) > 1 else float((bands.metadata or {}).get('spin_degeneracy', 2.0))
    if degeneracy not in (1.0, 2.0): raise InputError('unsupported DOS spin degeneracy')
    for rows in channels.values():
        if len(rows) != bands.nkpoints or any(len(row) != bands.nbands for row in rows):
            raise InputError('DOS spin channel dimensions differ')
    flat = [e for rows in channels.values() for row in rows for e in row]
    if not flat: raise InputError('cannot construct DOS from empty bands')
    if not all(math.isfinite(e) for e in flat): raise InputError('DOS contains nonfinite eigenvalues')
    # An unknown Fermi level stays unknown (None).  Substituting 0 eV would
    # silently shift every E - E_F axis built from this DOS.
    ef = float(bands.fermi_energy) if bands.fermi_energy is not None else None
    lo = min(flat) - 2.0 if emin is None else float(emin)
    hi = max(flat) + 2.0 if emax is None else float(emax)
    if hi <= lo: raise InputError('DOS energy window is empty')
    energies = [lo + (hi - lo) * i / (npoints - 1) for i in range(npoints)]
    norm = 1.0 / (sigma * math.sqrt(2 * math.pi))
    resolved = {}
    for name, rows in channels.items():
        resolved[name] = [sum(degeneracy * w * norm * math.exp(-0.5 * ((x - e) / sigma) ** 2)
                             for w, row in zip(weights, rows) for e in row) for x in energies]
    vals = [sum(channel[i] for channel in resolved.values()) for i in range(npoints)]
    return {'schema_version': 'sparse-dft-dos-1', 'source': bands.source,
            'source_sha256': hashlib.sha256(Path(bands.source).read_bytes()).hexdigest() if Path(bands.source).exists() else None,
            'fermi_energy_ev': ef, 'emin_ev': lo, 'emax_ev': hi, 'nedos': npoints,
            'energies_ev': energies, 'total': {'total': vals}, 'spin_dos': resolved, 'pdos': [],
            'metadata': {'format': 'Gaussian DOS reconstructed from eigenvalues', 'sigma_ev': sigma,
                         'spin_channels': list(channels), 'spin_degeneracy': degeneracy,
                         'sampling': bands.metadata['sampling'], 'kweights': weights,
                         'normalization': 'states/eV/cell', 'mesh_convergence_verified': False}}


def density_of_states_from_scf(path: str | Path, *, sigma=0.05, npoints=1201) -> dict:
    """Read executed native SCF eigenvalues and weights, not recommendations."""
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if not data.get('converged'):
        raise InputError('DOS cannot use an unconverged SCF')
    points = data.get('scf_kpoints') or []
    if not points:
        raise InputError('SCF result lacks executed weighted k-points')
    spin = data.get('spin_channels', 1)
    if spin not in (1, 2): raise InputError('unsupported SCF spin model for DOS')
    if spin == 2:
        pairs = [[p for p in points if p.get('spin_channel') == c] for c in (0, 1)]
        if not pairs[0] or len(pairs[0]) != len(pairs[1]) or any(
            a['fractional'] != b['fractional'] or abs(a['weight'] - b['weight']) > 1e-12
            for a, b in zip(*pairs)):
            raise InputError('spin DOS requires identical physical sampling in both channels')
        channels = {name: [p['eigenvalues_ev'] for p in pair] for name, pair in zip(('up', 'down'), pairs)}
        occupations = {name: [p['occupations'] for p in pair] for name, pair in zip(('up', 'down'), pairs)}
        points = pairs[0]
    else:
        # One spin-degenerate channel: occupations are keyed like every other
        # band payload ({channel: [kpoint][band]}); a bare list crashed here.
        channels = None
        occupations = ({'total': [p['occupations'] for p in points]}
                       if all('occupations' in p for p in points) else None)
    bands = from_payload({'kpoints': [p['fractional'] for p in points],
        'eigenvalues': [p['eigenvalues_ev'] for p in points],
        'spin_channels': channels, 'occupations': occupations,
        'kweights': [p['weight'] for p in points],
        'fermi_energy_ev': data.get('fermi_energy_ev'),
        'metadata': {'sampling': 'weighted_scf_kpoints', 'spin_degeneracy': 1 if spin == 2 else 2}}, source='<memory>')
    bands.source = str(path)
    return density_of_states_from_bands(bands, sigma=sigma, npoints=npoints)


DOS_TAIL_SIGMAS = 6.0      # Gaussian terms beyond 6 sigma (< 1.6e-8 of the peak) are dropped


def gaussian_broaden(energies, eigenvalues, weights, sigma, projections=None):
    """Sum of k-weighted normalised Gaussians on ``energies``; one curve per projection column.

    ``eigenvalues`` is [k][band]; ``weights`` [k] are the full-zone k weights
    including spin degeneracy; ``projections`` is [k][band][column] or None.
    Returns [total] or [total, column_1, ...] as lists of states/eV.
    """
    if not (sigma > 0 and math.isfinite(sigma)):
        raise InputError('DOS broadening must be a finite positive width in eV')
    lo, step, n = energies[0], energies[1] - energies[0], len(energies)
    half = int(math.ceil(DOS_TAIL_SIGMAS * sigma / step)) + 1
    norm = 1.0 / (sigma * math.sqrt(2.0 * math.pi))
    ncol = len(projections[0][0]) if projections else 0
    try:
        import numpy as np
    except ImportError:
        out = [[0.0] * n for _ in range(ncol + 1)]
        for k, (row, w) in enumerate(zip(eigenvalues, weights)):
            for b, e in enumerate(row):
                centre = int(round((e - lo) / step))
                for i in range(max(0, centre - half), min(n, centre + half + 1)):
                    g = w * norm * math.exp(-0.5 * ((energies[i] - e) / sigma) ** 2)
                    out[0][i] += g
                    for c in range(ncol):
                        out[c + 1][i] += g * projections[k][b][c]
        return out
    grid = np.asarray(energies, dtype=float)
    e = np.asarray(eigenvalues, dtype=float).ravel()
    w = np.repeat(np.asarray(weights, dtype=float), len(eigenvalues[0]))
    cols = (np.concatenate([np.ones((e.size, 1)), np.asarray(projections, dtype=float).reshape(e.size, ncol)], axis=1)
            if projections else np.ones((e.size, 1)))
    out = np.zeros((cols.shape[1], n))
    offsets = np.arange(-half, half + 1)
    centre = np.rint((e - lo) / step).astype(int)
    idx = centre[:, None] + offsets[None, :]                    # [state][window]
    valid = (idx >= 0) & (idx < n)
    safe = np.clip(idx, 0, n - 1)
    g = np.where(valid, w[:, None] * norm * np.exp(-0.5 * ((grid[safe] - e[:, None]) / sigma) ** 2), 0.0)
    for c in range(cols.shape[1]):
        np.add.at(out[c], safe.ravel(), (g * cols[:, c:c + 1]).ravel())
    return out.tolist()


def parse_vasprun_projections(path: str | Path) -> dict:
    """``<projected>`` block of vasprun.xml (LORBIT >= 10), same contract as :func:`parse_procar`.

    vasprun prints 4 decimals where PROCAR prints 3. Spin sets map to
    ``total`` (ISPIN=1), ``up``/``down`` (ISPIN=2) or ``total, mx, my, mz``
    (noncollinear).
    """
    p = Path(path)
    root = ET.parse(p).getroot()
    calcs = [c for c in root.findall('./calculation') if c.find('./projected') is not None]
    if not calcs:
        raise InputError(f'{p} has no <projected> block (needs LORBIT >= 10)')
    proj = calcs[-1].find('./projected')
    array = proj.find('./array')
    if array is None:
        raise InputError(f'{p} <projected> lacks the weight array')
    orbitals = [f.text.strip() for f in array.findall('./field') if f.text]
    spin_sets = array.findall('./set/set')
    names = {1: ['total'], 2: ['up', 'down'], 4: ['total', 'mx', 'my', 'mz']}.get(len(spin_sets))
    if names is None:
        raise InputError(f'{p} <projected> has {len(spin_sets)} spin sets')
    weights, energies = {}, {}
    nk = nb = nion = None
    for name, sset in zip(names, spin_sets):
        table = []
        for kset in sset.findall('./set'):
            row = []
            for bset in kset.findall('./set'):
                row.append([_float_tokens(r.text or '') for r in bset.findall('./r')])
            table.append(row)
        nk, nb, nion = len(table), len(table[0]), len(table[0][0])
        if any(len(row) != nb or any(len(ions) != nion or any(len(w) != len(orbitals) for w in ions)
                                     for ions in row) for row in table):
            raise InputError(f'{p} <projected> table {name} is ragged')
        weights[name] = table
    eig = proj.find('./eigenvalues/array')
    if eig is not None:
        esets = eig.findall('./set/set')
        for name, sset in zip(names if len(esets) == len(names) else ['total'] * len(esets), esets):
            energies[name] = [[_float_tokens(r.text or '')[0] for r in kset.findall('./r')]
                              for kset in sset.findall('./set')]
        if len(esets) == 1:  # noncollinear: one eigenvalue table for all four components
            energies = {name: energies['total'] for name in names}
    kpoints = [tuple(_float_tokens(v.text or '')[:3])
               for v in root.findall("./kpoints/varray[@name='kpointlist']/v")]
    spin_mode = {1: 'none', 2: 'collinear', 4: 'noncollinear'}[len(names)]
    return {'schema_version': 'sparse-dft-procar-1', 'source': str(p),
            'source_sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
            'nkpoints': nk, 'nbands': nb, 'nions': nion,
            'projection_orbitals': orbitals, 'projection_weights': weights,
            'projection_energies_ev': energies or None,
            'projection_kpoints': {name: kpoints for name in names} if len(kpoints) == nk else None,
            'spin_mode': spin_mode, 'spin_projection_available': len(names) in (2, 4),
            'weight_definition': 'VASP reported PAW sphere weights (vasprun.xml, 4 decimals)',
            'metadata': {'format': 'VASP vasprun.xml <projected>', 'spin_components': len(names),
                         'spin_mode': spin_mode, 'orbital_labels': orbitals}}


def parse_procar(path: str | Path) -> dict:
    """Strict VASP charge, ISPIN=2, and four-table noncollinear projections."""
    p = Path(path); lines = p.read_text(encoding='utf-8').splitlines()
    counts = re.compile(r'#\s*of k-points:\s*(\d+)\s*#\s*of bands:\s*(\d+)\s*#\s*of ions:\s*(\d+)')
    headers = [counts.search(line) for line in lines if counts.search(line)]
    if not headers: raise InputError(f'PROCAR counts missing: {p}')
    nk, nb, nion = map(int, headers[0].groups())
    if min(nk, nb, nion) < 1: raise InputError('PROCAR dimensions must be positive')
    if any(tuple(map(int, h.groups())) != (nk, nb, nion) for h in headers):
        raise InputError('PROCAR repeated headers disagree')
    records = []; tables = {}; orbital_labels = None
    spin = 1; explicit_spin = False; ik = -1; ib = -1; table_index = 0
    last_k = {}; point = None; energy = None; occupation = None
    i = 0
    while i < len(lines):
        line = lines[i].strip(); i += 1
        if not line: continue
        match = re.match(r'spin component\s+(\d+)', line, re.I)
        if match:
            spin = int(match[1]); explicit_spin = True; ik = -1; ib = -1
            continue
        match = re.match(r'k-point\s+(\d+)\s*:\s*(.*?)\s+weight\s*=\s*(\S+)', line)
        if match:
            index = int(match[1])-1
            # Standard ISPIN=2 PROCAR may repeat the header/k list without a spin marker.
            if not explicit_spin and index == 0 and last_k.get(spin) == nk-1:
                spin += 1
            if index != last_k.get(spin, -1)+1 or not 0 <= index < nk:
                raise InputError('PROCAR k-point order is incomplete or duplicated')
            coordinates = _fixed_width_floats(match[2]); weight = _float_tokens(match[3])
            if len(coordinates) != 3 or len(weight) != 1 or not all(math.isfinite(v) for v in coordinates+weight):
                raise InputError('invalid PROCAR k-point coordinates')
            ik = index; last_k[spin] = ik; point = tuple(coordinates); ib = -1
            continue
        match = re.match(r'band\s+(\d+)\s+#\s*energy\s+(\S+)\s+#\s*occ\.?\s+(\S+)', line)
        if match:
            index = int(match[1])-1
            if ik < 0 or index != ib+1 or not 0 <= index < nb:
                raise InputError('invalid PROCAR band order')
            ib = index; table_index = 0
            vals = _float_tokens(match[2]+' '+match[3])
            if len(vals)!=2 or not all(math.isfinite(v) for v in vals):raise InputError('invalid PROCAR band values')
            energy, occupation = vals
            continue
        if line.startswith('ion '):
            labels = line.split()[1:]
            if not labels or labels[-1] != 'tot' or ib < 0:
                raise InputError('invalid PROCAR projection header')
            labels = labels[:-1]
            if orbital_labels is None: orbital_labels = labels
            if labels != orbital_labels:raise InputError('PROCAR orbital columns change')
            atom_weights = []
            for atom in range(nion):
                while i < len(lines) and not lines[i].strip():i += 1
                if i >= len(lines):raise InputError('PROCAR truncated in ion projections')
                row = _float_tokens(lines[i]); i += 1
                if len(row)!=len(labels)+2 or row[0]!=atom+1 or not all(math.isfinite(v) for v in row):
                    raise InputError('invalid PROCAR ion projection row')
                atom_weights.append(row[1:-1])
            while i < len(lines) and not lines[i].strip():i += 1
            if i < len(lines) and lines[i].strip().startswith('tot '):
                totals = _float_tokens(lines[i]); i += 1
            elif nion == 1:
                # VASP omits the 'tot' row for a single ion: it equals the ion row.
                totals = list(atom_weights[0]) + [sum(atom_weights[0])]
            else:
                raise InputError('PROCAR total row missing')
            if len(totals)!=len(labels)+1 or not all(math.isfinite(v) for v in totals):raise InputError('invalid PROCAR total row')
            key = (spin, ik, ib, table_index)
            if key in tables:raise InputError('duplicate PROCAR projection block')
            record = {'spin':spin,'kpoint_index':ik,'band_index':ib,'component_index':table_index,
                      'kpoint':point,'energy_ev':energy,'occupation':occupation,
                      'orbital_labels':labels,'atom_weights':atom_weights}
            records.append(record); tables[key] = record; table_index += 1
    spins = sorted(last_k)
    if spins not in ([1], [1,2]):raise InputError('unsupported PROCAR spin channel count')
    components = sorted({key[3] for key in tables})
    if components not in ([0], [0,1,2,3]) or (len(spins)>1 and len(components)>1):
        raise InputError('unsupported PROCAR projection components (phase tables are not weights)')
    expected = {(s,k,b,c) for s in spins for k in range(nk) for b in range(nb) for c in components}
    if set(tables)!=expected:raise InputError('PROCAR projection tensor is incomplete')
    if len(components)==4:
        names = {c:n for c,n in enumerate(('charge','mx','my','mz'))}; spin_mode='noncollinear_charge_and_magnetization'
    else:
        names = {0:'total'}; spin_mode='collinear_spin_channels' if len(spins)==2 else 'spin_unpolarized_or_charge'
    projection_weights = {}; projection_energies = {}; kpoints = {}
    for s in spins:
        for c,name in names.items():
            if len(spins)==2:name='up' if s==1 else 'down'
            projection_weights[name] = [[tables[s,k,b,c]['atom_weights'] for b in range(nb)] for k in range(nk)]
            projection_energies[name] = [[tables[s,k,b,c]['energy_ev'] for b in range(nb)] for k in range(nk)]
            kpoints[name] = [tables[s,k,0,c]['kpoint'] for k in range(nk)]
    return {'schema_version': 'sparse-dft-procar-1', 'source': str(p), 'source_sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
            'nkpoints': nk, 'nbands': nb, 'nions': nion, 'records': records,
            'projection_orbitals': orbital_labels, 'projection_weights': projection_weights,
            'projection_energies_ev': projection_energies, 'projection_kpoints': kpoints,
            'spin_mode': spin_mode,
            'spin_projection_available': len(spins)==2 or len(components)==4,
            'weight_definition': 'VASP reported PAW sphere weights',
            'metadata': {'format': 'VASP PROCAR', 'spin_components': len(spins)*len(components),
                         'spin_mode': spin_mode, 'orbital_labels': orbital_labels}}


def _interpolate(values, x, target):
    if target <= x[0]:
        return values[0]
    if target >= x[-1]:
        return values[-1]
    for i, (left, right) in enumerate(zip(x, x[1:])):
        if left <= target <= right:
            if right == left:
                return values[i]
            weight = (target - left) / (right - left)
            return values[i] * (1 - weight) + values[i + 1] * weight
    return values[-1]


def _path_cartesian(bands: BandStructure):
    basis = (bands.metadata or {}).get('reciprocal_basis') if bands.metadata else None
    if basis is None:
        return [tuple(float(x) for x in kp[:3]) for kp in bands.kpoints]
    try:
        rows = [tuple(float(x) for x in row[:3]) for row in basis]
        if len(rows) != 3:
            raise ValueError
        return [tuple(sum(float(kp[i]) * rows[i][j] for i in range(3)) for j in range(3)) for kp in bands.kpoints]
    except (TypeError, ValueError, IndexError):
        return [tuple(float(x) for x in kp[:3]) for kp in bands.kpoints]


def _path_geometry(bands: BandStructure):
    """Return coalesced segment directions and normalized lengths.

    Sampling density and duplicate high-symmetry vertices are discarded only
    after the actual reciprocal-Cartesian geometry has been established.
    This permits scientifically valid resampling while rejecting a different
    k-path that merely happens to have the same total length.
    """
    points = _path_cartesian(bands)
    segments = []
    for left, right in zip(points, points[1:]):
        delta = tuple(b - a for a, b in zip(left, right))
        length = math.sqrt(sum(x * x for x in delta))
        if length <= 1e-10:
            continue
        direction = tuple(x / length for x in delta)
        if segments:
            dot = sum(a * b for a, b in zip(direction, segments[-1][0]))
            if dot >= 1.0 - 1e-7:
                segments[-1] = (segments[-1][0], segments[-1][1] + length)
                continue
        segments.append((direction, length))
    total = sum(length for _, length in segments)
    if total <= 0.0:
        return []
    return [(direction, length / total) for direction, length in segments]


def _same_path_geometry(reference: BandStructure, candidate: BandStructure, tol=2e-5):
    left, right = _path_geometry(reference), _path_geometry(candidate)
    if len(left) != len(right):
        return False, {'reference_segments': len(left), 'candidate_segments': len(right)}
    # A translated copy of a line is not the same high-symmetry path.  Allow
    # only integer reciprocal-lattice translations (Gamma represented as 0 or
    # 1), while rejecting arbitrary offsets that would invalidate comparison.
    starts = []
    for a, b in zip(reference.kpoints[0][:3], candidate.kpoints[0][:3]):
        delta = float(a) - float(b)
        delta -= round(delta)
        starts.append(abs(delta))
    if starts and max(starts) > tol:
        return False, {'reason': 'different_path_origin', 'origin_delta_mod_reciprocal': starts,
                       'reference_segments': len(left), 'candidate_segments': len(right)}
    mismatches = []
    for i, ((d1, l1), (d2, l2)) in enumerate(zip(left, right)):
        cosine = sum(a * b for a, b in zip(d1, d2))
        if cosine < 1.0 - tol or abs(l1 - l2) > tol:
            mismatches.append({'segment': i, 'direction_cosine': cosine,
                               'reference_fraction': l1, 'candidate_fraction': l2})
    return not mismatches, {'reference_segments': len(left), 'candidate_segments': len(right), 'mismatches': mismatches}


def compare_bands(reference: BandStructure, candidate: BandStructure, *, align='vbm', max_energy_ev=10.0) -> dict:
    if reference.nbands != candidate.nbands:
        raise InputError(f'band count mismatch: reference={reference.nbands}, candidate={candidate.nbands}')
    if reference.nkpoints < 2 or candidate.nkpoints < 2:
        raise InputError('band comparison needs at least two kpoints per path')
    # Never trust a serialized display axis for numerical alignment.  Rebuild
    # distances from the fractional k-points and the reciprocal basis after
    # geometry validation; this prevents a stale/incorrect plotting axis from
    # silently biasing interpolation.
    ref_basis = (reference.metadata or {}).get('reciprocal_basis') if reference.metadata else None
    cand_basis = (candidate.metadata or {}).get('reciprocal_basis') if candidate.metadata else None
    ref_distances = _distances(reference.kpoints, ref_basis)
    cand_distances = _distances(candidate.kpoints, cand_basis)
    ref_length, cand_length = ref_distances[-1], cand_distances[-1]
    if ref_length <= 0.0 or cand_length <= 0.0:
        raise InputError('band paths must have positive total length')
    same_geometry, geometry_report = _same_path_geometry(reference, candidate)
    if not same_geometry:
        raise InputError('band paths have different reciprocal-space geometry', details=geometry_report)
    shift = 0.0
    if align == 'vbm' and reference.vbm is not None and candidate.vbm is not None:
        shift = reference.vbm - candidate.vbm
    elif align == 'fermi' and reference.fermi_energy is not None and candidate.fermi_energy is not None:
        shift = reference.fermi_energy - candidate.fermi_energy
    elif align not in ('none', 'vbm', 'fermi'):
        raise InputError(f'unknown band alignment: {align}')
    errors = []
    for x, ref_row in zip(ref_distances, reference.eigenvalues):
        target = (x / ref_length) * cand_length
        cand_row = [_interpolate([row[band] for row in candidate.eigenvalues], cand_distances, target) + shift
                    for band in range(candidate.nbands)]
        for ref_value, cand_value in zip(ref_row, cand_row):
            delta = cand_value - ref_value
            if abs(ref_value - reference.fermi_energy if reference.fermi_energy is not None else ref_value) <= max_energy_ev:
                errors.append(delta)
    if not errors:
        raise InputError('no energies remained within comparison window')
    mae = sum(abs(x) for x in errors) / len(errors)
    rmse = math.sqrt(sum(x * x for x in errors) / len(errors))
    return {
        'reference': reference.source, 'candidate': candidate.source,
        'alignment': align, 'energy_shift_ev': shift,
        'samples': len(errors), 'mae_ev': mae, 'rmse_ev': rmse,
        'max_abs_error_ev': max(abs(x) for x in errors),
        'reference_gap_ev': reference.band_gap, 'candidate_gap_ev': candidate.band_gap,
        'gap_error_ev': (candidate.band_gap - reference.band_gap
                         if reference.band_gap is not None and candidate.band_gap is not None else None),
        'reference_path_length': ref_length, 'candidate_path_length': cand_length,
        'path_geometry': geometry_report,
    }
