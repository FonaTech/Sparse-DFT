"""Band-structure, DOS/PDOS and projection loaders for plotting.

Parsing is delegated to :mod:`sparse_dft.bands`; this module adds what a figure
needs and the parsers do not provide: broken-path segmentation, printable
high-symmetry labels, located band edges (direct/indirect), orbital-resolved
PDOS curves and a Fermi energy that is never silently replaced by 0 eV.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import math
import re
import xml.etree.ElementTree as ET

from ..bands import (BandStructure, density_of_states_from_scf, load_bands, parse_doscar,
                     parse_procar, parse_vasprun_projections, _float_tokens)
from ..errors import InputError
from .common import finite, read_json, resolve_fermi, schema_ids, sha256_file

BAND_FILE_NAMES = ('bands_result.json', 'bands.json', 'vasprun.xml', 'EIGENVAL')

_GREEK = {'GAMMA': r'\Gamma', 'G': r'\Gamma', 'Γ': r'\Gamma', 'DELTA': r'\Delta', 'Δ': r'\Delta',
          'LAMBDA': r'\Lambda', 'Λ': r'\Lambda', 'SIGMA': r'\Sigma', 'Σ': r'\Sigma'}


def pretty_kpoint_label(label) -> str | None:
    """'GAMMA', 'G', 'Γ', '\\Gamma' -> '$\\Gamma$'; 'S_0' -> '$\\mathrm{S}_{0}$'; 'X|U' kept."""
    if label is None:
        return None
    text = str(label).strip()
    if not text:
        return None
    if '|' in text:
        return join_kpoint_labels(pretty_kpoint_label(p) for p in text.split('|'))
    if text.startswith('$') and text.endswith('$') and len(text) > 1:
        return text
    bare = text.lstrip('\\')
    match = re.fullmatch(r'([A-Za-zΓΔΛΣ]+)_?\{?(\w*)\}?', bare) if '_' in bare else None
    base, sub = (match.group(1), match.group(2)) if match else (bare, '')
    key = base.upper() if base.upper() in _GREEK else base
    core = _GREEK.get(key) or _GREEK.get(base)
    if core is None:
        # Roman letters go through math text too, so 'X', 'X|K' and '$\Gamma$'
        # share one font, size and baseline on the k axis.
        core = r'\mathrm{%s}' % base
    return f'${core}_{{{sub}}}$' if sub else f'${core}$'


def join_kpoint_labels(labels) -> str | None:
    """['$\\mathrm{X}$', '$\\mathrm{U}$'] -> '$\\mathrm{X}|\\mathrm{U}$' (one math run, no stray '$')."""
    parts = []
    for label in labels:
        if not label:
            continue
        core = label[1:-1] if label.startswith('$') and label.endswith('$') and len(label) > 1 else r'\mathrm{%s}' % label
        if core not in parts:
            parts.append(core)
    return f"${'|'.join(parts)}$" if parts else None


@dataclass
class KPath:
    """Plot coordinates for a band path with discontinuities removed."""
    x: list[float]
    segments: list[tuple[int, int]]          # inclusive index ranges drawn as connected lines
    ticks: list[tuple[float, str]]           # (x, label) at high-symmetry points / breaks
    breaks: list[int]                        # index i where a jump starts between i-1 and i
    corners: list[float] = field(default_factory=list)   # x of unlabelled direction changes

    @property
    def length(self) -> float:
        return self.x[-1] if self.x else 0.0


def _same_label(a, b) -> bool:
    return pretty_kpoint_label(a) == pretty_kpoint_label(b)


def kpath(bands: BandStructure, *, jump_factor: float = 6.0) -> KPath:
    """Segment a band path.

    A break is declared between consecutive points i-1, i when (1) both carry
    different labels (pymatgen/Sparse-DFT convention: '...U' followed by 'K...'), or
    (2) unlabelled and the step is a jump more than ``jump_factor`` times the
    median step.  Duplicated vertices (zero length, same label) are joined.
    """
    n = bands.nkpoints
    d = [float(v) for v in (bands.distances or [])]
    if len(d) != n:
        from ..bands import _distances
        d = _distances(bands.kpoints, (bands.metadata or {}).get('reciprocal_basis'))
    labels = list(bands.labels or [None] * n)
    if len(labels) != n:
        labels = [None] * n
    steps = [d[i] - d[i - 1] for i in range(1, n)]
    positive = sorted(s for s in steps if s > 1e-12)
    median = positive[len(positive) // 2] if positive else 0.0
    breaks = []
    for i in range(1, n):
        a, b = labels[i - 1], labels[i]
        if a and b and not _same_label(a, b) and steps[i - 1] > 1e-9:
            breaks.append(i)
        elif not (a and b) and median > 0 and steps[i - 1] > jump_factor * median and len(positive) >= 4:
            breaks.append(i)
    x = [0.0]
    for i in range(1, n):
        x.append(x[-1] + (0.0 if i in breaks else max(0.0, steps[i - 1])))
    starts = [0] + breaks
    stops = [b - 1 for b in breaks] + [n - 1]
    segments = [(s, e) for s, e in zip(starts, stops) if e >= s]
    # Tick labels: explicit labels; Gamma is inferred only from (0,0,0)
    # because no other fractional point is convention independent.
    names = [pretty_kpoint_label(l) for l in labels]
    if not any(names):
        names = [r'$\Gamma$' if all(abs(float(c)) < 1e-8 for c in k[:3]) else None for k in bands.kpoints]
    ticks: list[tuple[float, str]] = []
    for i, name in enumerate(names):
        if not name:
            continue
        if ticks and abs(ticks[-1][0] - x[i]) < 1e-9:
            ticks[-1] = (ticks[-1][0], join_kpoint_labels([ticks[-1][1], name]))
            continue
        ticks.append((x[i], name))
    # Unlabelled path vertices (direction changes) get a guide line but no
    # label: a fractional coordinate alone does not name the point.
    basis = (bands.metadata or {}).get('reciprocal_basis')
    def cart(k):
        if basis and len(basis) == 3:
            return [sum(float(k[i]) * float(basis[i][j]) for i in range(3)) for j in range(3)]
        return [float(c) for c in k[:3]]
    pts = [cart(k) for k in bands.kpoints]
    corners = []
    previous = None
    for i in range(1, n):
        if i in breaks:
            previous = None
            continue
        delta = [b - a for a, b in zip(pts[i - 1], pts[i])]
        length = math.sqrt(sum(c * c for c in delta))
        if length <= 1e-10:
            continue
        direction = [c / length for c in delta]
        if previous is not None and sum(a * b for a, b in zip(direction, previous)) < 1.0 - 1e-6:
            xc = x[i - 1]
            if all(abs(xc - t) > 1e-9 for t, _ in ticks) and all(abs(xc - c) > 1e-9 for c in corners):
                corners.append(xc)
        previous = direction
    return KPath(x=x, segments=segments, ticks=ticks, breaks=breaks, corners=corners)


@dataclass
class BandEdges:
    vbm_ev: float | None
    cbm_ev: float | None
    gap_ev: float | None
    status: str                       # insulator | inferred | metal | metal_or_numerical | unknown
    direct: bool | None
    vbm_kindex: list[int] = field(default_factory=list)
    cbm_kindex: list[int] = field(default_factory=list)
    vbm_channel: str | None = None
    cbm_channel: str | None = None
    direct_gap_ev: float | None = None
    direct_gap_kindex: int | None = None

    @property
    def is_metal(self) -> bool:
        return self.status in ('metal', 'metal_or_numerical')


def channels_of(bands: BandStructure) -> dict[str, list[list[float]]]:
    return dict(bands.spin_channels) if bands.spin_channels else {'total': bands.eigenvalues}


def band_edges(bands: BandStructure, tol: float = 1e-6) -> BandEdges:
    """Locate VBM/CBM reported by the parser and classify the gap."""
    vbm, cbm, status = bands.vbm, bands.cbm, bands.gap_status
    if status == 'inferred' and bands.fermi_energy is not None:
        # Without occupations the gap comes from E_F alone. A band that runs from
        # below to above E_F along the path is metallic even if no sample sits
        # exactly at E_F (fcc Ni path: "gap" 0.054 eV between two such samples).
        ef = float(bands.fermi_energy)
        for rows in channels_of(bands).values():
            for b in range(len(rows[0])):
                column = [row[b] for row in rows]
                if min(column) < ef - tol and max(column) > ef + tol:
                    return BandEdges(None, None, None, 'metal', None)
    if status in ('metal', 'metal_or_numerical', 'unknown') or vbm is None or cbm is None or cbm <= vbm:
        return BandEdges(None, None, None, status if status else 'unknown', None)
    where_v, where_c = [], []
    direct_best = None
    for name, rows in channels_of(bands).items():
        for k, row in enumerate(rows):
            occ = [e for e in row if e <= vbm + tol]
            emp = [e for e in row if e >= cbm - tol]
            if occ and abs(max(occ) - vbm) <= tol:
                where_v.append((name, k))
            if emp and abs(min(emp) - cbm) <= tol:
                where_c.append((name, k))
            if occ and emp:
                gap_k = min(emp) - max(occ)
                if direct_best is None or gap_k < direct_best[0] - 1e-12:
                    direct_best = (gap_k, k)
    def key(k):
        # Equivalent k-points (e.g. Gamma at both path ends) compare modulo G.
        return tuple(round((float(c) - math.floor(float(c) + 1e-7)) * 1e6) for c in bands.kpoints[k][:3])
    same = {key(k) for _, k in where_v} & {key(k) for _, k in where_c}
    direct = bool(same) if where_v and where_c else None
    return BandEdges(vbm, cbm, cbm - vbm, status, direct,
                     sorted({k for _, k in where_v}), sorted({k for _, k in where_c}),
                     where_v[0][0] if where_v else None, where_c[0][0] if where_c else None,
                     direct_best[0] if direct_best else None, direct_best[1] if direct_best else None)


# --- band structures -------------------------------------------------------
BandData = BandStructure      # the band container is shared with sparse_dft.bands


def find_band_file(path) -> Path:
    p = Path(path)
    if p.is_dir():
        for name in BAND_FILE_NAMES:
            if (p / name).is_file():
                return p / name
        raise InputError(f'no band output ({", ".join(BAND_FILE_NAMES)}) in {p}')
    if not p.is_file():
        raise InputError(f'file not found: {p}')
    return p


def load_band_data(path, *, fermi=None, structure=None) -> BandStructure:
    """Load a band path; ``fermi`` (eV) overrides the file, ``structure`` fixes the k metric."""
    bands = load_bands(find_band_file(path))
    if fermi is not None:
        bands.fermi_energy = resolve_fermi(None, fermi, bands.source)[0]
    if structure is not None:
        from ..bands import _distances
        from ..structure import read_structure
        s = read_structure(structure) if isinstance(structure, (str, Path)) else structure
        if s.reciprocal_lattice:
            bands.distances = _distances(bands.kpoints, s.reciprocal_lattice)
    return bands


# --- DOS / PDOS ------------------------------------------------------------
# VASP DOSCAR projected-column layouts by (spin_polarized, ncolumns).
_LM_SPD = ['s', 'py', 'pz', 'px', 'dxy', 'dyz', 'dz2', 'dxz', 'dx2-y2']
_LM_F = ['fy3x2', 'fxyz', 'fyz2', 'fz3', 'fxz2', 'fzx2', 'fx3']
DOSCAR_COLUMNS = {
    (False, 3): ['s', 'p', 'd'], (False, 4): ['s', 'p', 'd', 'f'],
    (False, 9): _LM_SPD, (False, 16): _LM_SPD + _LM_F,
    (True, 6): ['s', 'p', 'd'], (True, 8): ['s', 'p', 'd', 'f'],
    (True, 18): _LM_SPD, (True, 32): _LM_SPD + _LM_F,
}
L_ORDER = ('s', 'p', 'd', 'f')


def orbital_l(label) -> str | None:
    """Angular-momentum channel of a VASP/Sparse-DFT orbital label ('px' -> 'p')."""
    if label is None:
        return None
    text = str(label).strip().lower()
    if text in ('x2-y2', 'dx2', 'x2y2', 'dx2-y2'):
        return 'd'
    return text[0] if text and text[0] in L_ORDER else None


def canonical_orbital(label) -> str:
    """One spelling per orbital: VASP prints 'x2-y2' in PROCAR/vasprun, Sparse-DFT 'dx2-y2'."""
    text = str(label).strip().lower()
    return 'dx2-y2' if text in ('x2-y2', 'dx2', 'x2y2') else text


LM_ORDER = tuple(_LM_SPD + _LM_F)
GROUPINGS = ('l', 'lm', 'orbital', 'element', 'element+l', 'element+lm', 'atom', 'atom+l', 'atom+lm')


def parse_atom_selection(selection, species=None, natoms=None) -> list[int]:
    """1-based atom indices from '1-3,5', 'Ga', ['Ga', 4] or ints; sorted, unique."""
    if selection is None:
        return list(range(1, (natoms or 0) + 1))
    tokens = selection.replace(',', ' ').split() if isinstance(selection, str) else list(selection)
    out = set()
    for token in tokens:
        text = str(token).strip()
        if not text:
            continue
        if species and text in species:
            out.update(i + 1 for i, s in enumerate(species) if s == text)
            continue
        match = re.fullmatch(r'(\d+)(?:-(\d+))?', text)
        if not match:
            raise InputError(f"atom selection {text!r} is neither an index, a range 'a-b' nor an element of "
                             f"{sorted(set(species or []))}")
        lo, hi = int(match.group(1)), int(match.group(2) or match.group(1))
        if lo < 1 or hi < lo or (natoms and hi > natoms):
            raise InputError(f'atom selection {text!r} is outside 1..{natoms}')
        out.update(range(lo, hi + 1))
    if not out:
        raise InputError(f'atom selection {selection!r} selects no atom')
    return sorted(out)


def _orbital_matcher(selection, available):
    """Predicate for an orbital filter: exact labels ('dxy') or whole channels ('d')."""
    if selection is None:
        return lambda orb: True
    tokens = selection.replace(',', ' ').split() if isinstance(selection, str) else list(selection)
    wanted = {canonical_orbital(t) for t in tokens if str(t).strip()}
    known = {canonical_orbital(o) for o in available} | {orbital_l(o) for o in available}
    unknown = sorted(wanted - known)
    if unknown:
        raise InputError(f'orbital selection {unknown} is not among {sorted(known - {None})}')
    return lambda orb: canonical_orbital(orb) in wanted or orbital_l(orb) in wanted


def _group_key(by, atom, element, orbital):
    """(label, site, l, orbital) of one atom/orbital column under a grouping."""
    site = element or f'atom {atom}'
    l = orbital_l(orbital)
    tag = l or orbital                     # unrecognised column names stay as they are
    atom_name = f'{element or "atom"}{atom}'
    return {'l': (tag, None, l, None), 'lm': (orbital, None, l, orbital), 'orbital': (orbital, None, l, orbital),
            'element': (site, site, None, None), 'element+l': (f'{site} {tag}', site, l, None),
            'element+lm': (f'{site} {orbital}', site, l, orbital), 'atom': (atom_name, atom_name, None, None),
            'atom+l': (f'{atom_name} {tag}', atom_name, l, None),
            'atom+lm': (f'{atom_name} {orbital}', atom_name, l, orbital)}[by]


def _site_rank(species) -> dict:
    """Element -> 1-based index of its first atom (the POSCAR species order)."""
    rank = {}
    for i, element in enumerate(species or [], start=1):
        if element is not None:
            rank.setdefault(element, i)
    return rank


def _group_rank(by, atom, element, orbital, site_rank):
    l = orbital_l(orbital)
    lrank = L_ORDER.index(l) if l in L_ORDER else 9
    orank = LM_ORDER.index(orbital) if orbital in LM_ORDER else 99
    site = site_rank.get(element, atom) if by.startswith('element') and element is not None else atom
    if by in ('l',):
        return (0, lrank, 0)
    if by in ('lm', 'orbital'):
        return (0, lrank, orank)
    if by in ('element', 'atom'):
        return (site, 0, 0)
    return (site, lrank, orank if by.endswith('lm') else 0)


@dataclass
class DosCurve:
    label: str
    values: list[float]               # states/eV (spin-down kept positive; plots mirror it)
    spin: str | None = None           # 'up' | 'down' | None (spin-degenerate)
    element: str | None = None
    l: str | None = None              # 's' 'p' 'd' 'f', None for totals/unknown columns
    atoms: list[int] = field(default_factory=list)   # 1-based atom indices summed
    orbital: str | None = None        # lm label ('dxy') when the curve is one m component


@dataclass
class DosData:
    source: str
    energies_ev: list[float]
    fermi_energy_ev: float | None
    total: list[DosCurve]
    pdos: list[DosCurve]
    spin_polarized: bool
    origin: str                        # 'DOSCAR' | 'vasprun.xml' | 'scf_gaussian'
    sigma_ev: float | None = None
    warnings: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    units: dict = field(default_factory=lambda: {'energy': 'eV', 'dos': 'states/eV/cell'})
    # One curve per (atom, orbital, spin) column as the source wrote it; ``pdos``
    # is its (element, l, spin) sum. Empty when the source has no PDOS.
    pdos_raw: list[DosCurve] = field(default_factory=list)

    @property
    def resolution(self) -> str | None:
        """'lm' when the raw PDOS keeps m components, 'l' when only s/p/d/f, None without PDOS."""
        orbitals = {c.orbital for c in self.pdos_raw}
        if not orbitals:
            return None
        return 'lm' if any(o not in L_ORDER and orbital_l(o) for o in orbitals) else 'l'

    def grouped(self, by: str = 'element+l', *, atoms=None, orbitals=None, spin=None) -> list[DosCurve]:
        """Sum raw PDOS columns into groups (see GROUPINGS); spins stay separate.

        ``atoms`` takes 1-based indices, ranges or element symbols; ``orbitals``
        takes lm labels or whole channels ('d'). lm groupings need lm columns.
        """
        if by not in GROUPINGS:
            raise InputError(f'PDOS grouping must be one of {", ".join(GROUPINGS)}')
        if not self.pdos_raw:
            raise InputError(f'{self.source} has no per-atom projected DOS')
        if by.endswith('lm') and self.resolution != 'lm':
            raise InputError(f'{self.source} PDOS is l-resolved only; lm groupings need LORBIT=11 '
                             'or a Sparse-DFT dos run')
        return _group_curves(self.pdos_raw, by, atoms=atoms, orbitals=orbitals, spin=spin)

    def integral(self, curve: DosCurve, emax_ev: float | None = None) -> float:
        """Trapezoidal integral of ``curve`` up to ``emax_ev`` (absolute eV)."""
        xs, ys = self.energies_ev, curve.values
        total = 0.0
        for i in range(len(xs) - 1):
            a, b = xs[i], xs[i + 1]
            if emax_ev is not None and a >= emax_ev:
                break
            if emax_ev is not None and b > emax_ev:
                frac = (emax_ev - a) / (b - a)
                yb = ys[i] + frac * (ys[i + 1] - ys[i])
                total += 0.5 * (ys[i] + yb) * (emax_ev - a)
                break
            total += 0.5 * (ys[i] + ys[i + 1]) * (b - a)
        return total

    def electron_count(self, emax_ev: float | None = None) -> float:
        """Sum over total curves (both spins) of the integrated DOS."""
        return sum(self.integral(c, emax_ev) for c in self.total)


def _sibling_species(path: Path, natoms: int | None) -> list[str] | None:
    for name in ('vasprun.xml',):
        candidate = path.parent / name
        if candidate.is_file() and candidate != path:
            try:
                species = _vasprun_species(ET.parse(candidate).getroot())
            except ET.ParseError:
                species = None
            if species and (natoms is None or len(species) == natoms):
                return species
    for name in ('POSCAR', 'CONTCAR'):
        candidate = path.parent / name
        if candidate.is_file():
            try:
                from ..structure import read_structure
                species = list(read_structure(candidate).species)
            except Exception:
                continue
            if natoms is None or len(species) == natoms:
                return species
    return None


def _raw_curve(atom, element, orbital, spin, values) -> DosCurve:
    orbital = canonical_orbital(orbital)
    who = element or f'atom {atom}'
    return DosCurve(f'{who} {orbital}', values, spin, element, orbital_l(orbital), [atom], orbital)


def _raw_doscar_curves(blocks, species, spin_polarized, layout) -> list[DosCurve]:
    """One curve per DOSCAR column; unknown layouts keep the column names unsplit."""
    out = []
    for index, block in enumerate(blocks, start=1):
        element = species[index - 1] if species else None
        for column, (name, values) in enumerate(block):
            if layout is None:
                orbital, spin = name, None
            elif spin_polarized:
                orbital, spin = layout[column // 2], ('up' if column % 2 == 0 else 'down')
            else:
                orbital, spin = layout[column], None
            out.append(_raw_curve(index, element, orbital, spin, [float(v) for v in values]))
    return out


def _group_curves(raw, by='element+l', *, atoms=None, orbitals=None, spin=None) -> list[DosCurve]:
    """Sum raw (atom, orbital, spin) curves into one curve per group and spin; never mixes spins."""
    species = [None] * max(c.atoms[0] for c in raw)
    for c in raw:
        species[c.atoms[0] - 1] = c.element
    chosen = set(parse_atom_selection(atoms, species if all(species) else None, len(species)))
    keep = _orbital_matcher(orbitals, [c.orbital for c in raw])
    site_rank = _site_rank(species)
    groups: dict[tuple, DosCurve] = {}
    ranks = {}
    for c in raw:
        atom = c.atoms[0]
        if atom not in chosen or not keep(c.orbital) or (spin is not None and c.spin != spin):
            continue
        label, _site, l, orbital = _group_key(by, atom, c.element, c.orbital)
        key = (label, c.spin)
        curve = groups.get(key)
        if curve is None:
            # element survives only while every summed column shares it
            curve = groups[key] = DosCurve(label, list(c.values), c.spin, c.element, l, [atom], orbital)
            ranks[key] = _group_rank(by, atom, c.element, c.orbital, site_rank)
            continue
        curve.values = [a + b for a, b in zip(curve.values, c.values)]
        if atom not in curve.atoms:
            curve.atoms.append(atom)
        if curve.element != c.element:
            curve.element = None
    if not groups:
        raise InputError('the atom/orbital/spin selection matches no PDOS column')
    spin_order = {None: 0, 'up': 0, 'down': 1}
    return [groups[k] for k in sorted(groups, key=lambda k: (ranks[k], k[0], spin_order.get(k[1], 2)))]


def _vasprun_species(root) -> list[str] | None:
    rows = root.findall("./atominfo/array[@name='atoms']/set/rc")
    out = []
    for rc in rows:
        cells = rc.findall('c')
        if cells and cells[0].text:
            out.append(cells[0].text.strip())
    return out or None


def dos_from_payload(raw: dict, species=None, *, path=None) -> DosData:
    """Convert a ``sparse-dft-dos-1`` dict (parse_doscar / density_of_states_*) to DosData."""
    source = Path(path or raw.get('source') or '<memory>')
    energies = [float(e) for e in raw['energies_ev']]
    meta = raw.get('metadata') or {}
    t = raw.get('total') or {}
    spin_dos = raw.get('spin_dos') or {}
    gaussian = 'Gaussian' in str(meta.get('format', ''))
    if 'total_up' in t and 'total_down' in t:
        total = [DosCurve('total', list(t['total_up']), 'up'), DosCurve('total', list(t['total_down']), 'down')]
    elif set(spin_dos) == {'up', 'down'}:
        total = [DosCurve('total', list(spin_dos['up']), 'up'), DosCurve('total', list(spin_dos['down']), 'down')]
    elif 'total' in t:
        total = [DosCurve('total', list(t['total']), None)]
    else:
        raise InputError(f'DOS payload from {source} has no total DOS column')
    spin = len(total) == 2
    blocks, raw_curves = [], []
    for block in raw.get('pdos') or []:
        if 'curves' in block:          # sparse-dft dos: labelled (orbital, spin) curves per atom
            for c in block['curves']:
                if len(c['values']) != len(energies):
                    raise InputError(f'{source}: PDOS curve length differs from the energy grid')
                raw_curves.append(_raw_curve(int(block['atom']), block.get('element'), c['orbital'],
                                             c.get('spin'), [float(v) for v in c['values']]))
            continue
        if len(block.get('energies_ev') or []) != len(energies):
            continue
        cols = block['columns']
        blocks.append([(name, cols[name]) for name in sorted(cols, key=lambda c: int(c.split('_')[-1])
                                                             if c.split('_')[-1].isdigit() else 0)])
    natoms = len(blocks) or None
    if not species and blocks and source.exists():
        species = _sibling_species(source, natoms)
    species = list(species) if species else None
    if species and natoms and len(species) != natoms:
        species = None
    warnings = []
    if blocks:
        ncols = len(blocks[0])
        layout = DOSCAR_COLUMNS.get((spin, ncols))
        if layout is None:
            warnings.append(f'PDOS column layout with {ncols} columns is not recognised; columns are shown raw')
        raw_curves = _raw_doscar_curves(blocks, species, spin, layout)
        if not species:
            warnings.append('PDOS atoms are unlabelled: no matching vasprun.xml/POSCAR next to the DOSCAR')
    pdos = _group_curves(raw_curves, 'element+l') if raw_curves else []
    mesh = meta.get('sampling') == 'uniform_mesh' and gaussian
    if mesh:
        warnings.append(f'Gaussian DOS (sigma {meta.get("sigma_ev")} eV) on a {meta.get("nkpoints")}-point '
                        'NSCF mesh; mesh and broadening convergence are unverified')
    elif gaussian:
        warnings.append('Gaussian DOS from the executed SCF k-points; mesh and broadening convergence are unverified')
    top = finite(meta.get('complete_below_ev'))
    if top is not None and top < energies[-1]:
        warnings.append(f'DOS above {top:.2f} eV is incomplete: the highest solved band starts there '
                        '(raise --bands to extend it)')
    return DosData(source=str(source), energies_ev=energies, fermi_energy_ev=finite(raw.get('fermi_energy_ev')),
                   total=total, pdos=pdos, spin_polarized=spin,
                   origin='sdft_nscf_mesh' if mesh else ('scf_gaussian' if gaussian else 'DOSCAR'),
                   sigma_ev=finite(meta.get('sigma_ev')), warnings=warnings,
                   metadata={'sha256': raw.get('source_sha256'), 'nedos': raw.get('nedos'),
                             'spin_degeneracy': meta.get('spin_degeneracy'),
                             'kpoints': meta.get('nkpoints') or len(meta.get('kweights') or []) or None,
                             'complete_below_ev': top, 'grid': meta.get('grid'),
                             'projection_weight_definition': meta.get('projection_weight_definition')},
                   pdos_raw=raw_curves)


def _dos_from_doscar(path: Path, species=None) -> DosData:
    return dos_from_payload(parse_doscar(path), species, path=path)


def _dos_from_vasprun(path: Path) -> DosData:
    root = ET.parse(path).getroot()
    calcs = [c for c in root.findall('./calculation') if c.find('./dos') is not None]
    if not calcs:
        raise InputError(f'{path} contains no <dos> block')
    dos = calcs[-1].find('./dos')
    fermi_node = dos.find("./i[@name='efermi']")
    fermi = finite(fermi_node.text) if fermi_node is not None and fermi_node.text else None

    def spin_sets(node):
        # <total><array> nests spins two levels down (array/set/set); an ion set
        # holds them as direct children. Reading only array/set/set dropped
        # spin 2 of every ion and labelled spin 1 as spin-degenerate.
        for level in ('./set', './set/set'):
            sets = [s for s in node.findall(level) if s.get('comment', '').lower().startswith('spin')]
            if sets:
                return sets
        return [node.find('./set')] if node.find('./set') is not None else []

    total_array = dos.find('./total/array')
    if total_array is None:
        raise InputError(f'{path} <dos> lacks a total array')
    sets = spin_sets(total_array)
    rows = [[_float_tokens(r.text or '') for r in s.findall('./r')] for s in sets]
    energies = [r[0] for r in rows[0]]
    spin = len(rows) == 2
    total = [DosCurve('total', [r[1] for r in rows[i]], ('up', 'down')[i] if spin else None)
             for i in range(len(rows))]
    species = _vasprun_species(root)
    raw = []
    partial = dos.find('./partial/array')
    if partial is not None:
        fields = [f.text.strip() for f in partial.findall('./field') if f.text]
        orbitals = fields[1:]
        ions = [s for s in partial.findall('./set/set') if s.get('comment', '').lower().startswith('ion')]
        for index, ion in enumerate(ions, start=1):
            element = species[index - 1] if species and len(species) == len(ions) else None
            sets_ = spin_sets(ion)
            for si, sset in enumerate(sets_):
                values = [_float_tokens(r.text or '') for r in sset.findall('./r')]
                sp = ('up', 'down')[si] if len(sets_) == 2 else None
                for oi, orb in enumerate(orbitals):
                    raw.append(_raw_curve(index, element, orb, sp, [v[oi + 1] for v in values]))
    pdos = _group_curves(raw, 'element+l') if raw else []
    return DosData(source=str(path), energies_ev=energies, fermi_energy_ev=fermi, total=total, pdos=pdos,
                   spin_polarized=spin, origin='vasprun.xml', warnings=[],
                   metadata={'sha256': sha256_file(path), 'species': species}, pdos_raw=raw)


def _dos_from_scf(path: Path, sigma: float, npoints: int) -> DosData:
    return dos_from_payload(density_of_states_from_scf(path, sigma=sigma, npoints=npoints), path=path)


def load_dos(path, *, sigma: float = 0.05, npoints: int = 2001, species=None) -> DosData:
    """DOS from DOSCAR, vasprun.xml (<dos>) or a converged Sparse-DFT SCF result."""
    p = Path(path)
    if p.is_dir():
        for name in ('dos.json', 'DOSCAR', 'vasprun.xml', 'scf_result.json', 'scf.json'):
            if (p / name).is_file():
                p = p / name
                break
        else:
            raise InputError(f'no DOS source (dos.json, DOSCAR, vasprun.xml, scf_result.json) in {p}')
    if p.name.upper().startswith('DOSCAR'):
        return _dos_from_doscar(p, species)
    if p.suffix.lower() == '.xml':
        return _dos_from_vasprun(p)
    if p.suffix.lower() == '.json':
        data = read_json(p)
        if data.get('schema_version') in schema_ids('dos-1'):      # written by `sparse-dft dos`
            return dos_from_payload(data, species, path=p)
        return _dos_from_scf(p, sigma, npoints)
    raise InputError(f'unsupported DOS source: {p}')


# --- projections (fat bands) -------------------------------------------------
@dataclass
class ProjectionSet:
    source: str
    channels: dict[str, list]          # channel -> [k][band][atom][orbital] weights
    orbitals: list[str]
    nions: int
    species: list[str] | None
    signed: bool                        # True for SOC magnetisation channels mx/my/mz
    weight_definition: str | None
    spin_projection_available: bool
    resolution: str = 'l'               # 'lm' when orbitals are m components (s, py, pz, px, dxy, ...)

    def groups(self, by: str = 'l', *, atoms=None, orbitals=None) -> dict[str, list[tuple[int, int]]]:
        """Group name -> [(atom index 0-based, orbital index)] in display order."""
        if by not in GROUPINGS:
            raise InputError(f'projection grouping must be one of {", ".join(GROUPINGS)}')
        if by.endswith('lm') and self.resolution != 'lm':
            raise InputError(f'{self.source} carries l-resolved projections only; lm groupings need '
                             'LORBIT=11 (PROCAR/vasprun.xml) or a v16 Sparse-DFT band run')
        chosen = set(parse_atom_selection(atoms, self.species, self.nions))
        keep = _orbital_matcher(orbitals, self.orbitals)
        site_rank = _site_rank(self.species)
        members: dict[str, list] = {}
        ranks = {}
        for a in range(self.nions):
            element = self.species[a] if self.species else None
            for o, orb in enumerate(self.orbitals):
                if a + 1 not in chosen or not keep(orb):
                    continue
                name = _group_key(by, a + 1, element, orb)[0]
                members.setdefault(name, []).append((a, o))
                ranks.setdefault(name, _group_rank(by, a + 1, element, orb, site_rank))
        if not members:
            raise InputError('the atom/orbital selection matches no projection')
        return {name: members[name] for name in sorted(members, key=lambda n: (ranks[n], n))}

    def grouped(self, channel: str, by: str = 'l', *, atoms=None, orbitals=None) -> dict[str, list[list[float]]]:
        """Sum weights into groups -> [k][band]; ``by`` is one of GROUPINGS.

        ``atoms`` takes 1-based indices, ranges or element symbols; ``orbitals``
        takes lm labels or whole channels ('d').
        """
        tensor = self.channels[channel]
        groups = self.groups(by, atoms=atoms, orbitals=orbitals)
        try:
            import numpy as np
        except ImportError:                 # the loaders do not require numpy
            return {name: [[sum(float(atoms_[a][o]) for a, o in idx) for atoms_ in row] for row in tensor]
                    for name, idx in groups.items()}
        arr = np.asarray(tensor, dtype=float)              # [k][band][atom][orbital]
        return {name: arr[:, :, [a for a, _ in idx], [o for _, o in idx]].sum(axis=-1).tolist()
                for name, idx in groups.items()}


def _collapse_to_l(norm: dict) -> dict:
    """Sum lm columns into s/p/d/f columns, keeping the order of first appearance."""
    labels = []
    for orb in norm['projection_orbitals']:
        l = orbital_l(orb) or canonical_orbital(orb)
        if l not in labels:
            labels.append(l)
    index = {l: [o for o, orb in enumerate(norm['projection_orbitals'])
                 if (orbital_l(orb) or canonical_orbital(orb)) == l] for l in labels}
    weights = {name: [[[[sum(atom[o] for o in index[l]) for l in labels] for atom in atoms]
                        for atoms in row] for row in tensor]
               for name, tensor in norm['projection_weights'].items()}
    return {**norm, 'projection_orbitals': labels, 'projection_weights': weights}


def load_projections(bands: BandStructure, source=None, *, species=None, resolution: str = 'auto') -> ProjectionSet:
    """Validated projection tensor from PROCAR, vasprun.xml or the native band JSON.

    ``resolution``: 'auto' keeps m components when the source has them (PROCAR
    and vasprun always do; Sparse-DFT bands since v16), 'lm' requires them, 'l'
    sums them into s/p/d/f.
    """
    from .report import _normalize_projections
    if resolution not in ('auto', 'lm', 'l'):
        raise InputError("projection resolution must be 'auto', 'lm' or 'l'")
    if source is not None:
        source = Path(source)
        raw = parse_vasprun_projections(source) if source.suffix.lower() == '.xml' else parse_procar(source)
    else:
        raw = bands.projections
        # Prefer the lm tensor (VASP LORBIT=11 definition) even for 'l': it is
        # collapsed below, so l and lm figures show the same quantity. The
        # legacy l tensor (raw |<p|psi>|^2) is used only when lm is absent.
        if raw and raw.get('projection_lm_orbitals') and 'projection_lm_weights' in raw:
            raw = {**raw, 'projection_orbitals': raw['projection_lm_orbitals'],
                   'projection_weights': raw['projection_lm_weights'],
                   'weight_definition': raw.get('projection_lm_weight_definition') or raw.get('weight_definition')}
    if not raw:
        raise InputError(f'no PROCAR, vasprun.xml or native projections for {bands.source}')
    norm = _normalize_projections(bands, raw)
    norm['projection_orbitals'] = [canonical_orbital(o) for o in norm['projection_orbitals']]
    lm = any(o not in L_ORDER and orbital_l(o) for o in norm['projection_orbitals'])
    if resolution == 'lm' and not lm:
        raise InputError(f'{source or bands.source} has l-resolved projections only; rerun the band '
                         'calculation with this Sparse-DFT version (or VASP LORBIT=11) for lm weights')
    if resolution == 'l' and lm:
        norm, lm = _collapse_to_l(norm), False
    symbols = species or norm.get('projection_atom_symbols') or None
    if symbols and len(symbols) != norm['nions']:
        symbols = None
    return ProjectionSet(source=str(source or bands.source), channels=norm['projection_weights'],
                         orbitals=list(norm['projection_orbitals']), nions=int(norm['nions']),
                         species=list(symbols) if symbols else None,
                         signed=any(c in ('mx', 'my', 'mz') for c in norm['projection_weights']),
                         weight_definition=norm.get('weight_definition'),
                         spin_projection_available=bool(norm.get('spin_projection_available')),
                         resolution='lm' if lm else 'l')


# --- SCF mesh eigenvalues and band comparison ---------------------------------
def bands_from_scf(path) -> BandStructure:
    """Executed SCF k-mesh eigenvalues as a BandStructure (sampling, not a path)."""
    from ..bands import from_payload
    from .scf import find_scf_result
    p = find_scf_result(path)
    data = read_json(p)
    points = data.get('scf_kpoints') or []
    if not points:
        raise InputError(f'{p} has no scf_kpoints eigenvalues')
    spin = int(data.get('spin_channels') or 1)
    if spin == 2:
        pairs = [[q for q in points if q.get('spin_channel') == c] for c in (0, 1)]
        if not pairs[0] or len(pairs[0]) != len(pairs[1]):
            raise InputError(f'{p}: spin channels have different k samplings')
        channels = {name: [q['eigenvalues_ev'] for q in pair] for name, pair in zip(('up', 'down'), pairs)}
        occupations = ({name: [q['occupations'] for q in pair] for name, pair in zip(('up', 'down'), pairs)}
                       if all('occupations' in q for q in points) else None)
        points = pairs[0]
    else:
        channels = None
        occupations = ({'total': [q['occupations'] for q in points]}
                       if all('occupations' in q for q in points) else None)
    bands = from_payload({'kpoints': [q['fractional'] for q in points],
                          'eigenvalues': [q['eigenvalues_ev'] for q in points],
                          'spin_channels': channels, 'occupations': occupations,
                          'kweights': [q['weight'] for q in points],
                          'fermi_energy_ev': data.get('fermi_energy_ev'),
                          'metadata': {'sampling': 'weighted_scf_kpoints',
                                       'spin_degeneracy': 1 if spin == 2 else 2,
                                       **({'reciprocal_basis': (data.get('metadata') or {}).get('reciprocal_basis')}
                                          if (data.get('metadata') or {}).get('reciprocal_basis') else {})}})
    bands.source = str(p)
    return bands


def is_mesh(bands: BandStructure) -> bool:
    return (bands.metadata or {}).get('sampling') in ('weighted_scf_kpoints', 'uniform_mesh')


@dataclass
class BandComparison:
    reference: str
    candidate: str
    metrics: dict                              # sparse_dft.bands.compare_bands output
    kindex: list[int]
    band: list[int]
    reference_ev: list[float]
    candidate_ev: list[float]                  # candidate interpolated onto reference k, aligned
    delta_ev: list[float]                      # candidate - reference
    units: dict = field(default_factory=lambda: {'energy': 'eV'})


COMPARE_TOP_EXCLUDED = 2      # as tools/anchor_report.py: the top Davidson bands are the least converged


def common_band_window(reference: BandStructure, candidate: BandStructure,
                       top_excluded: int = COMPARE_TOP_EXCLUDED) -> tuple[BandStructure, BandStructure, int | None]:
    """Cut both band sets to the lowest min(nbands) - ``top_excluded`` bands when their counts differ.

    VASP band runs solve more bands than Sparse-DFT (ICHARG=11 top-band convergence); the
    comparison then covers the Sparse-DFT window. Equal counts are returned unchanged.
    """
    if reference.nbands == candidate.nbands:
        return reference, candidate, None
    import copy
    n = min(reference.nbands, candidate.nbands) - top_excluded
    if n < 1:
        raise InputError('band sources share no band window', details={'reference': reference.nbands,
                                                                     'candidate': candidate.nbands})
    out = []
    for bands in (reference, candidate):
        c = copy.copy(bands)
        c.eigenvalues = [row[:n] for row in bands.eigenvalues]
        if bands.spin_channels:
            c.spin_channels = {k: [row[:n] for row in rows] for k, rows in bands.spin_channels.items()}
        if bands.occupations:
            c.occupations = {k: [row[:n] for row in rows] for k, rows in bands.occupations.items()}
        c.projections = None
        c.metadata = {**(bands.metadata or {}), 'compared_bands': n, 'solved_bands': bands.nbands}
        out.append(c)
    return out[0], out[1], n


def compare_band_data(reference: BandStructure, candidate: BandStructure, *, align='auto',
                      max_energy_ev: float = 10.0) -> BandComparison:
    """Per-point band differences with the same alignment and window as compare_bands.

    Different band counts are compared on :func:`common_band_window`. The
    energy zero of each source is in ``metrics['reference_zero_ev']`` and
    ``metrics['candidate_zero_ev']`` (see :func:`sparse_dft.bands.band_zero`).
    """
    from ..bands import _distances, _interpolate, compare_bands
    reference, candidate, window = common_band_window(reference, candidate)
    metrics = compare_bands(reference, candidate, align=align, max_energy_ev=max_energy_ev)
    shift = metrics['energy_shift_ev']
    rb = (reference.metadata or {}).get('reciprocal_basis')
    cb = (candidate.metadata or {}).get('reciprocal_basis')
    rd, cd = _distances(reference.kpoints, rb), _distances(candidate.kpoints, cb)
    rl, cl = rd[-1], cd[-1]
    ks, bs, rv, cv, dv = [], [], [], [], []
    centre = metrics['window_centre_ev']
    for k, (x, row) in enumerate(zip(rd, reference.eigenvalues)):
        target = (x / rl) * cl
        for b, ref in enumerate(row):
            cand = _interpolate([r[b] for r in candidate.eigenvalues], cd, target) + shift
            centred = ref - centre if centre is not None else ref
            if abs(centred) <= max_energy_ev:
                ks.append(k); bs.append(b); rv.append(ref); cv.append(cand); dv.append(cand - ref)
    if window is not None:
        metrics = {**metrics, 'compared_bands': window}
    return BandComparison(reference.source, candidate.source, metrics, ks, bs, rv, cv, dv)


# --- DOS rebuilt from band energies (for like-for-like comparison) ------------
def dos_like(template: DosData, bands: BandStructure, proj: ProjectionSet | None = None, *,
             top_excluded: int = 2, source=None) -> DosData:
    """DOS of ``bands`` (+ ``proj``) on the energy grid and Gaussian width of ``template``.

    A VASP DOSCAR uses its own smearing and grid, so a comparison with a Sparse-DFT
    Gaussian DOS rebuilds the reference from its eigenvalues, k weights and lm
    weights instead. The top ``top_excluded`` bands (Davidson tail) are dropped
    and ``metadata['complete_below_ev']`` marks where the DOS stops being complete.
    """
    from ..bands import gaussian_broaden
    if not template.sigma_ev:
        raise InputError(f'{template.source} has no Gaussian width; compare against a sparse-dft dos result')
    if not bands.kweights:
        raise InputError(f'{bands.source} has no k weights (needs a uniform-mesh vasprun.xml or bands.json)')
    chans = (bands.spin_channels if bands.spin_channels and len(bands.spin_channels) == 2
             else {'total': bands.eigenvalues})
    nb = bands.nbands - top_excluded
    if nb < 1:
        raise InputError('too few bands to rebuild a DOS')
    degeneracy = 1.0 if len(chans) == 2 else 2.0
    total_w = sum(bands.kweights)
    weights = [degeneracy * w / total_w for w in bands.kweights]
    top = min(min(row[nb] for row in rows) for rows in chans.values()) if top_excluded else None
    totals, raw = [], []
    for name, rows in chans.items():
        spin = None if name == 'total' else name
        tensor = None
        if proj is not None:
            pch = proj.channels[name if name in proj.channels else next(iter(proj.channels))]
            tensor = [[[v for atom in pch[k][b] for v in atom] for b in range(nb)] for k in range(bands.nkpoints)]
        curves = gaussian_broaden(template.energies_ev, [r[:nb] for r in rows], weights, template.sigma_ev, tensor)
        totals.append(DosCurve('total', curves[0], spin))
        if proj is not None:
            no = len(proj.orbitals)
            for a in range(proj.nions):
                element = proj.species[a] if proj.species else None
                for o, orbital in enumerate(proj.orbitals):
                    raw.append(_raw_curve(a + 1, element, orbital, spin, curves[1 + a * no + o]))
    return DosData(source=str(source or bands.source), energies_ev=list(template.energies_ev),
                   fermi_energy_ev=bands.fermi_energy, total=totals,
                   pdos=_group_curves(raw, 'element+l') if raw else [], spin_polarized=len(chans) == 2,
                   origin='rebuilt_gaussian', sigma_ev=template.sigma_ev,
                   warnings=[f'rebuilt from {nb} of {bands.nbands} bands with the Gaussian of {template.source}'],
                   metadata={'complete_below_ev': top, 'bands_used': nb}, pdos_raw=raw)


def dos_difference(candidate: DosData, reference: DosData, *, window=None) -> dict:
    """Normalised L1 of total DOS (and of matching raw PDOS columns) on a shared grid."""
    if len(candidate.energies_ev) != len(reference.energies_ev) or any(
            abs(a - b) > 1e-9 for a, b in zip(candidate.energies_ev, reference.energies_ev)):
        raise InputError('DOS comparison needs the same energy grid (rebuild the reference with dos_like)')
    energies = candidate.energies_ev
    step = energies[1] - energies[0]
    tops = [x for x in (candidate.metadata.get('complete_below_ev'), reference.metadata.get('complete_below_ev'))
            if x is not None]
    lo, hi = window or (energies[0], (min(tops) - 0.5) if tops else energies[-1])
    mask = [lo <= x <= hi for x in energies]

    def l1(a, b):
        ref = sum(y for y, m in zip(b, mask) if m) * step
        diff = sum(abs(x - y) for x, y, m in zip(a, b, mask) if m) * step
        return (diff / ref if ref else None), ref

    out = {'window_ev': [lo, hi], 'sigma_ev': candidate.sigma_ev, 'total': {}, 'pdos': {}}
    for c in candidate.total:
        r = next((x for x in reference.total if x.spin == c.spin), None)
        if r is not None:
            out['total'][c.spin or 'total'] = l1(c.values, r.values)[0]
    index = {(x.atoms[0], x.orbital, x.spin): x for x in reference.pdos_raw}
    for c in candidate.pdos_raw:
        r = index.get((c.atoms[0], c.orbital, c.spin))
        if r is not None:
            value, integral = l1(c.values, r.values)
            out['pdos'][f'{c.atoms[0]}:{c.label}' + (f' ({c.spin})' if c.spin else '')] = {
                'l1_normalised': value, 'integral_reference': integral}
    return out
