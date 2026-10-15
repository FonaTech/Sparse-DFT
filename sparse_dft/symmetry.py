"""Space groups (spglib) and standard high-symmetry band paths (seekpath).

The k path is the Hinuma et al. convention (Comput. Mater. Sci. 128, 140
(2017), seekpath ``recipe='hpkot'``). seekpath returns its vertices in the
reciprocal basis of the *user* cell (``get_path_orig_cell``), so the path
runs on the structure the user loaded; a supercell input is flagged
because the path then folds bands (seekpath ``is_supercell``).

Sampling follows ``sparse_dft.kpoints.interpolate_band_path``: every segment
keeps both labelled endpoints (VASP line-mode convention), the number of
points per segment is proportional to its Cartesian length, and a
discontinuity ('X|U') is two consecutive points with different labels.
"""
from __future__ import annotations

import math

from .errors import InputError, UnsupportedError
from .structure import Structure, _inverse, _norm

DEFAULT_SYMPREC = 1e-5
# Points per 1/A of path (with 2 pi): 0.025 1/A spacing is seekpath's default.
DEFAULT_SPACING = 0.025


def _require():
    try:
        import spglib  # noqa: F401
        import seekpath  # noqa: F401
    except ImportError as exc:
        raise UnsupportedError('symmetry analysis needs spglib and seekpath; install the "symmetry" extra',
                               details={'missing': exc.name}) from exc


def _numbers(species):
    from .geometry import element
    return [int(element(s)['z']) for s in species]


def cell_tuple(structure: Structure):
    if structure.lattice is None:
        raise InputError('symmetry analysis needs a periodic cell')
    s = structure.normalized()
    return ([list(r) for r in s.lattice], [list(p) for p in s.positions], _numbers(s.species))


def analyze(structure: Structure, symprec: float = DEFAULT_SYMPREC) -> dict:
    """Space group, Bravais lattice and primitive-cell size for the user cell."""
    _require()
    import spglib
    cell = cell_tuple(structure)
    dataset = spglib.get_symmetry_dataset(cell, symprec=symprec)
    if dataset is None:
        raise InputError('spglib could not determine the space group', details={'symprec': symprec})
    primitive = spglib.standardize_cell(cell, to_primitive=True, no_idealize=False, symprec=symprec)
    nprim = len(primitive[2]) if primitive is not None else None
    return {
        'spacegroup_number': int(dataset.number),
        'spacegroup_international': str(dataset.international),
        'hall_symbol': str(dataset.hall),
        'point_group': str(dataset.pointgroup),
        'crystal_system': crystal_system(int(dataset.number)),
        'natoms': len(cell[2]),
        'primitive_natoms': nprim,
        'is_primitive': nprim == len(cell[2]),
        'symprec': symprec,
        'spglib_version': spglib.__version__,
    }


def crystal_system(number: int) -> str:
    for upper, name in ((2, 'triclinic'), (15, 'monoclinic'), (74, 'orthorhombic'), (142, 'tetragonal'),
                        (167, 'trigonal'), (194, 'hexagonal'), (230, 'cubic')):
        if number <= upper:
            return name
    raise InputError('space group number out of range', details={'number': number})


def standard_primitive(structure: Structure, symprec: float = DEFAULT_SYMPREC) -> Structure:
    """spglib standardized primitive cell (rotated to the standard setting)."""
    _require()
    import spglib
    from .geometry import element_table
    lattice, positions, numbers = spglib.standardize_cell(cell_tuple(structure), to_primitive=True,
                                                          no_idealize=False, symprec=symprec)
    symbols = {row['z']: sym for sym, row in element_table().items()}
    order = sorted(range(len(numbers)), key=lambda i: (list(dict.fromkeys(numbers)).index(numbers[i]), i))
    return Structure([symbols[int(numbers[i])] for i in order],
                     [tuple(float(x) for x in positions[i]) for i in order],
                     tuple(tuple(float(x) for x in row) for row in lattice), True, (True, True, True),
                     source_format='spglib-standard-primitive')


SYMPREC_LADDER = (1e-5, 1e-4, 1e-3, 1e-2)
# Probed only to report "nearly symmetric" cells; never selected silently.
LOOSE_SYMPREC = 0.1


def _operation_count(cell, symprec):
    import spglib
    dataset = spglib.get_symmetry_dataset(cell, symprec=symprec)
    return (0, None) if dataset is None else (len(dataset.rotations), str(dataset.international))


def choose_symprec(structure: Structure, ladder=SYMPREC_LADDER) -> dict:
    """Tightest tolerance on ``ladder`` that reaches the highest symmetry found on it.

    A looser LOOSE_SYMPREC probe is reported (with a warning) but never used.
    """
    _require()
    cell = cell_tuple(structure)
    rungs = [(p, *_operation_count(cell, p)) for p in ladder]
    best = max(r[1] for r in rungs)
    chosen = next(r for r in rungs if r[1] == best)
    loose = _operation_count(cell, LOOSE_SYMPREC)
    warnings = []
    if chosen[0] > ladder[0]:
        warnings.append(f'symmetry {chosen[2]} needs symprec {chosen[0]:g} A; at {ladder[0]:g} A the cell is '
                        f'{rungs[0][2]} (slightly distorted or rounded coordinates)')
    if loose[0] > best:
        warnings.append(f'within {LOOSE_SYMPREC:g} A the cell is close to {loose[1]} ({loose[0]} operations vs {best}); '
                        f'if the distortion is numerical noise, symmetrize it or pass --symprec {LOOSE_SYMPREC:g}')
    return {'symprec': chosen[0], 'spacegroup_international': chosen[2], 'operations': best,
            'ladder': [{'symprec': p, 'operations': n, 'spacegroup': g} for p, n, g in rungs],
            'loose_probe': {'symprec': LOOSE_SYMPREC, 'operations': loose[0], 'spacegroup': loose[1]},
            'warnings': warnings}


def auto_band_path(structure: Structure, *, symprec: float | None = None, spacing: float = DEFAULT_SPACING,
                   restrict_periodic: bool = True, export_primitive=None) -> dict:
    """``high_symmetry_path`` with tolerance choice, vacuum masking and primitive export."""
    from .geometry import axis_vacuum
    report = choose_symprec(structure) if symprec is None else {
        'symprec': float(symprec), 'warnings': [], 'ladder': None, 'loose_probe': None}
    periodic = None
    if restrict_periodic:
        vacuum = axis_vacuum(structure).vacuum_axes
        periodic = tuple(not v for v in vacuum)
        if not any(periodic):
            raise InputError('every axis is vacuum (molecule or cluster): there is no band dispersion; '
                             'run a Gamma-only SCF and plot levels or DOS instead')
    try:
        path = high_symmetry_path(structure, spacing=spacing, symprec=report['symprec'], periodic_axes=periodic)
    except InputError:
        if periodic is None or sum(periodic) != 1:
            raise
        path = _wire_path(structure, periodic, spacing, report['symprec'])
    path['symprec'] = report['symprec']
    path['symprec_report'] = report
    path['periodic_axes'] = list(periodic) if periodic is not None else [True, True, True]
    path['warnings'] = report['warnings'] + path['warnings']
    if export_primitive:
        primitive = standard_primitive(structure, symprec=report['symprec'])
        from .structure import write_poscar
        target = write_poscar(primitive, export_primitive,
                              f"spglib standard primitive of {structure.source_path or 'input'} (symprec {report['symprec']:g})")
        prim_path = high_symmetry_path(primitive, spacing=spacing, symprec=report['symprec'],
                                       periodic_axes=None if periodic is None else
                                       tuple(not v for v in axis_vacuum(primitive).vacuum_axes))
        sibling = target.with_name(target.name + '.band_path.json')
        import json
        sibling.write_text(json.dumps(prim_path, indent=1) + '\n', encoding='utf-8')
        path['primitive_poscar'] = str(target)
        path['primitive_band_path'] = str(sibling)
        path['primitive_natoms'] = primitive.natoms
    return path


def _wire_path(structure: Structure, periodic, spacing, symprec) -> dict:
    """1D fallback: Gamma -> zone boundary along the single periodic axis."""
    axis = periodic.index(True)
    end = [0.0, 0.0, 0.0]
    end[axis] = 0.5
    rec = _reciprocal(cell_tuple(structure)[0])
    length = 0.5 * _norm(rec[axis])
    n = max(5, int(math.ceil(length / float(spacing))) + 1)
    kpoints = [[i / (n - 1) * end[j] for j in range(3)] for i in range(n)]
    labels = ['GAMMA'] + [None] * (n - 2) + ['X']
    return {'schema_version': 'sparse-dft-band-path-1', 'kpoints': kpoints, 'labels': labels,
            'vertices': [{'from': 'GAMMA', 'to': 'X', 'start': [0.0] * 3, 'end': end, 'points': n,
                          'length_inverse_angstrom': length}],
            'segments': [['GAMMA', 'X']], 'dropped_segments': [], 'point_coords': {'GAMMA': [0.0] * 3, 'X': end},
            'bravais_lattice_extended': None, 'spacegroup_number': None, 'spacegroup_international': None,
            'has_inversion_symmetry': None, 'is_supercell': False,
            'warnings': ['wire: straight Gamma-X path along the periodic axis (no seekpath segment lies on it)'],
            'spacing_inverse_angstrom': float(spacing),
            'source': {'library': 'sparse_dft', 'recipe': 'wire Gamma-X', 'coordinates': 'fractional, reciprocal basis of the input cell'}}


def _reciprocal(lattice):
    inv = _inverse(lattice)
    return [[2.0 * math.pi * x for x in row] for row in inv]


def high_symmetry_path(structure: Structure, *, spacing: float = DEFAULT_SPACING,
                       min_points: int = 5, symprec: float = DEFAULT_SYMPREC,
                       with_time_reversal: bool = True, periodic_axes=None) -> dict:
    """Explicit labelled k path on the user cell.

    Returns ``{'kpoints', 'labels', 'vertices', 'segments', 'path', 'source', ...}``;
    ``kpoints`` and ``labels`` go straight to ``run_bands(kpoints=..., path_labels=...)``.
    ``periodic_axes`` (from the preset classifier) drops segments that move
    along a non-periodic (vacuum) axis, so slabs and 2D sheets get the
    in-plane path only.
    """
    _require()
    import seekpath
    if not 0.001 <= float(spacing) <= 0.5:
        raise InputError('band path spacing must be in [0.001, 0.5] 1/A', details={'spacing': spacing})
    cell = cell_tuple(structure)
    res = seekpath.get_path_orig_cell(cell, with_time_reversal=with_time_reversal, symprec=symprec)
    coords = {k: [float(x) for x in v] for k, v in res['point_coords'].items()}
    rec = _reciprocal(cell[0])

    def cart(f):
        return [sum(f[i] * rec[i][j] for i in range(3)) for j in range(3)]

    segments = []
    dropped = []
    for a, b in res['path']:
        # A non-periodic axis has no dispersion: keep only the k_i = 0 plane.
        if periodic_axes is not None and any(
                not periodic_axes[i] and max(abs(coords[a][i]), abs(coords[b][i])) > 1e-9 for i in range(3)):
            dropped.append([a, b])
            continue
        segments.append((a, b))
    if not segments:
        raise InputError('no high-symmetry segment lies in the periodic directions',
                         details={'dropped': dropped})
    kpoints, labels, vertices = [], [], []
    for a, b in segments:
        ka, kb = coords[a], coords[b]
        length = _norm([x - y for x, y in zip(cart(ka), cart(kb))])
        n = max(int(min_points), int(math.ceil(length / float(spacing))) + 1)
        for i in range(n):
            t = i / (n - 1)
            kpoints.append(tuple((1.0 - t) * ka[j] + t * kb[j] for j in range(3)))
            labels.append(a if i == 0 else b if i == n - 1 else None)
        vertices.append({'from': a, 'to': b, 'start': list(ka), 'end': list(kb), 'points': n,
                         'length_inverse_angstrom': length})
    return {
        'schema_version': 'sparse-dft-band-path-1',
        'kpoints': [list(k) for k in kpoints],
        'labels': labels,
        'vertices': vertices,
        'segments': [list(s) for s in segments],
        'dropped_segments': dropped,
        'point_coords': coords,
        'bravais_lattice_extended': res['bravais_lattice_extended'],
        'spacegroup_number': int(res['spacegroup_number']),
        'spacegroup_international': res['spacegroup_international'],
        'has_inversion_symmetry': bool(res['has_inversion_symmetry']),
        'is_supercell': bool(res.get('is_supercell', False)),
        'warnings': (['the cell is a supercell of its primitive cell: bands are folded; '
                      'export the standard primitive cell for an unfolded path']
                     if res.get('is_supercell') else []),
        'spacing_inverse_angstrom': float(spacing),
        'source': {'library': 'seekpath', 'version': seekpath.__version__, 'recipe': 'hpkot',
                   'reference': 'Hinuma et al., Comput. Mater. Sci. 128, 140 (2017)',
                   'coordinates': 'fractional, reciprocal basis of the input cell'},
    }
