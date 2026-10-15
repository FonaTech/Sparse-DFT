from __future__ import annotations
from dataclasses import dataclass, asdict
import math
from .structure import Structure, _norm
from .errors import InputError
from .settings import KPOINT_GRID_LIMITS, KPOINT_SPACING_LIMITS


def interpolate_band_path(vertices, points_per_segment=81, *, include_duplicate_vertices=True):
    """Generate an explicit line-mode path from fractional vertices.

    ``points_per_segment`` is the number of samples *including both segment
    endpoints*, matching VASP line-mode ``divisions`` behavior.  Keeping the
    shared endpoint by default preserves the exact convention used by VASP's
    EIGENVAL/vasprun.xml and makes per-segment labels unambiguous.
    """
    try:
        points_per_segment = int(points_per_segment)
    except (TypeError, ValueError) as exc:
        raise InputError('points_per_segment must be an integer') from exc
    if points_per_segment < 2:
        raise InputError('points_per_segment must be at least 2')
    try:
        verts = [tuple(float(x) for x in row[:3]) for row in vertices]
    except (TypeError, ValueError, IndexError) as exc:
        raise InputError('band path vertices must contain 3-component points') from exc
    if len(verts) < 2 or any(len(v) != 3 or any(not math.isfinite(x) for x in v) for v in verts):
        raise InputError('band path requires at least two finite 3-component vertices')
    out = []
    for seg, (left, right) in enumerate(zip(verts, verts[1:])):
        start = 0 if include_duplicate_vertices or seg == 0 else 1
        for i in range(start, points_per_segment):
            t = i / (points_per_segment - 1)
            out.append(tuple((1.0 - t) * left[j] + t * right[j] for j in range(3)))
    return validate_kpoint_path(out, task='bands')

@dataclass
class KPointRecommendation:
    grid: tuple[int,int,int]
    shift: tuple[float,float,float]
    mode: str
    spacing_inverse_angstrom: float
    periodic_axes: tuple[bool,bool,bool]
    reason: list[str]
    convergence_scan: list[float]

    def as_dict(self): return asdict(self)

def validate_kpoint_grid(grid, *, task='scf') -> tuple[int, int, int]:
    """Validate a Monkhorst-Pack/Gamma grid before it reaches the core."""
    if isinstance(grid, (str, bytes)):
        raise InputError('k-point grid must contain three positive integers')
    try:
        raw = tuple(grid)
        if len(raw) != 3 or any(isinstance(x, bool) or int(x) != x for x in raw):
            raise ValueError
        values = tuple(int(x) for x in raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InputError('k-point grid must contain three positive integers', details={'grid': grid}) from exc
    if len(values) != 3 or any(x < 1 or x > 128 for x in values):
        raise InputError('k-point grid must contain three positive integers', details={'grid': list(values)})
    limit = KPOINT_GRID_LIMITS.get(str(task).lower(), KPOINT_GRID_LIMITS['nscf'])
    count = values[0] * values[1] * values[2]
    if count > limit:
        raise InputError(
            f'{task} k-point grid has {count} points; maximum supported is {limit}',
            details={'grid': values, 'points': count, 'maximum': limit},
        )
    return values


def uniform_kpoint_mesh(grid, *, mode='gamma'):
    """Explicit full-Brillouin-zone points and weights; no assumed symmetry.

    Gamma uses i/N, Monkhorst-Pack uses (2*i+1-N)/(2*N). All points are
    folded into [-1/2, 1/2); weights sum to one in the C++ SCF.
    """
    grid = validate_kpoint_grid(grid, task='scf')
    if mode not in ('gamma', 'monkhorst-pack'):
        raise InputError('mesh mode must be gamma or monkhorst-pack')
    axes = []
    for n in grid:
        axis = [(i / n if mode == 'gamma' else (2*i+1-n)/(2*n)) for i in range(n)]
        axes.append([x-math.floor(x+0.5) for x in axis])
    points = [(x, y, z) for x in axes[0] for y in axes[1] for z in axes[2]]
    return points, [1.0 / len(points)] * len(points)


def validate_kpoint_path(kpoints, *, task='nscf', minimum=2):
    """Validate an explicit fractional NSCF/band path.

    Fractional coordinates are deliberately allowed outside [0, 1] because
    equivalent translated points are common in band-path files.
    """
    if isinstance(kpoints, (str, bytes)):
        raise InputError('k-point path must be a sequence of three-component points')
    try:
        points = [tuple(float(x) for x in point[:3]) for point in kpoints]
    except (TypeError, ValueError, IndexError) as exc:
        raise InputError('k-point path must contain finite 3-component points') from exc
    if len(points) < minimum:
        raise InputError(f'{task} k-point path requires at least {minimum} points', details={'count': len(points)})
    limit = KPOINT_GRID_LIMITS.get(str(task).lower(), KPOINT_GRID_LIMITS['nscf'])
    if len(points) > limit:
        raise InputError(f'{task} k-point path has {len(points)} points; maximum supported is {limit}',
                         details={'count': len(points), 'maximum': limit})
    import math
    if any(len(point) != 3 or any(not math.isfinite(value) for value in point) for point in points):
        raise InputError('k-point path contains a non-finite or malformed point')
    return points

def recommend_kpoints(structure: Structure, task='scf', accuracy='standard', spacing_inverse_angstrom=None, metallic=None):
    s=structure.normalized()
    task = str(task).lower()
    if task not in ('scf', 'nscf', 'bands', 'relax', 'ionic_relax', 'cell_relax'):
        raise InputError('unsupported k-point task', details={'task': task})
    if s.lattice is None:
        return KPointRecommendation((1,1,1),(0.,0.,0.),'gamma',0.0,s.periodic,['non-periodic structure'],[])
    spacing = ({'low':0.30,'standard':0.18,'high':0.12,'production':0.10}.get(accuracy,0.18)
               if spacing_inverse_angstrom is None else spacing_inverse_angstrom)
    if not (KPOINT_SPACING_LIMITS[0] <= float(spacing) <= KPOINT_SPACING_LIMITS[1]):
        raise InputError(
            f'k-point spacing must be in [{KPOINT_SPACING_LIMITS[0]}, {KPOINT_SPACING_LIMITS[1]}] 1/A',
            details={'spacing_inverse_angstrom': spacing, 'minimum': KPOINT_SPACING_LIMITS[0], 'maximum': KPOINT_SPACING_LIMITS[1]},
        )
    spacing = float(spacing)
    rec=s.reciprocal_lattice; grid=[]
    grid_task = 'scf' if task in ('scf', 'relax', 'ionic_relax', 'cell_relax') else task
    limit = KPOINT_GRID_LIMITS.get(grid_task, KPOINT_GRID_LIMITS['nscf'])
    original_count = None
    for _ in range(8):
        grid = [1 if not s.periodic[i] else max(1,math.ceil(_norm(rec[i])/spacing)) for i in range(3)]
        count = math.prod(grid)
        if original_count is None:
            original_count = count
        if count <= limit:
            break
        # Relax the spacing just enough to keep automatic meshes bounded.  A
        # caller asking for an explicit grid still gets a hard validation error
        # through validate_kpoint_grid().
        scale = (count / limit) ** (1.0 / 3.0)
        spacing *= max(1.01, scale)
    grid = list(validate_kpoint_grid(grid, task=grid_task))
    reason=[f'inverse-space spacing target {spacing:.4f} 1/A',f'task={task}',f'accuracy={accuracy}']
    if original_count is not None and original_count > limit:
        reason.append(f'automatic mesh capped at {limit} points by adjusting spacing')
    if not all(s.periodic): reason.append('non-periodic axes fixed at 1')
    if metallic: reason.append('metallic system: retain dense mesh and use smearing')
    mode='gamma' if (not metallic and max(grid)>=4) else 'monkhorst-pack'
    shift=(0.,0.,0.) if mode=='gamma' else tuple(0.5 if n%2==0 else 0.0 for n in grid)
    scan = [x for x in (spacing*1.25, spacing, spacing*0.8) if KPOINT_SPACING_LIMITS[0] <= x <= KPOINT_SPACING_LIMITS[1]]
    return KPointRecommendation(tuple(grid),shift,mode,spacing,s.periodic,reason,scan)
