"""Sparsity-sweep tables and parallel-scaling sets.

``SweepTable`` reads ``tools/report_sparsity.py`` output
(schema ``sparse-dft-sparsity-report-1``) or a raw ``tools/sweep_sparsity.py``
``summary.json``.  Gates are whitepaper Table 1.  Work ratios are counter
based (reference / run, > 1 means less work) and are kept apart from wall
time, which is only meaningful on an idle machine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import math

from ..errors import InputError
from .common import finite, read_json, schema_ids

SWEEP_REPORT_SCHEMA = 'sparse-dft-sparsity-report-1'
SCALING_SCHEMA = 'sparse-dft-scaling-1'
SWEEP_REPORT_SCHEMAS = schema_ids('sparsity-report-1')
SCALING_SCHEMAS = schema_ids('scaling-1')
# Whitepaper Table 1 acceptance gates.
GATES = {'energy_mev_per_atom': 1.0, 'force_ev_per_angstrom': 1e-3, 'stress_gpa': 0.01, 'band_window_mev': 5.0}
WORK_METRICS = ('auto', 'operator_applications', 'overlap_applications', 'density_modes', 'xc_grid_points',
                'density_functional_seconds', 'diagonalization_seconds', 'total_seconds')


def gate_use(row: dict) -> float | None:
    """max over Table-1 gates of error / limit (1 = at the limit); None when no metric exists."""
    uses = [abs(v) / limit for key, limit in GATES.items() if (v := finite(row.get(key))) is not None]
    return max(uses) if uses else None


def component_of(variant: str) -> str:
    """'S1_f4' -> 'S1', 'E_bundle_f8' -> 'E_bundle', 'auto' -> 'auto', 'reference' -> 'reference'."""
    if variant.startswith('E_bundle'):
        return 'E_bundle'
    head = variant.split('_', 1)[0]
    return head if head[:1] in ('S',) or head in ('auto', 'reference') else variant


@dataclass
class SweepRow:
    system: str
    variant: str
    component: str
    converged: bool
    passed: bool | None
    gate_use: float | None
    errors: dict[str, float | None]
    work: dict[str, float]                    # reference/run ratios (> 1 = less work)
    natoms: int | None = None
    wall_speedup: float | None = None

    def work_reduction(self, metric: str = 'auto') -> tuple[float | None, str | None]:
        """Work reduction for ``metric``.

        'auto' takes the largest of the H+S operator-application, density-mode
        and XC-grid ratios.  The overlap-only ratio is excluded because overlap
        applications are already part of the H+S count.
        """
        counters = ('operator_applications', 'density_modes', 'xc_grid_points')
        if metric == 'auto':
            options = [(self.work[k], k) for k in counters if self.work.get(k)]
            return max(options) if options else (None, None)
        if metric == 'total_seconds' and self.work.get('total_seconds') is None and self.wall_speedup:
            return self.wall_speedup, 'total_seconds'
        value = self.work.get(metric)
        return (value, metric) if value else (None, None)


@dataclass
class SweepTable:
    source: str
    rows: list[SweepRow]
    systems: list[str]
    gates: dict = field(default_factory=lambda: dict(GATES))
    origin: str = 'report'                     # 'report' | 'summary'
    units: dict = field(default_factory=lambda: {
        'energy_mev_per_atom': 'meV/atom', 'force_ev_per_angstrom': 'eV/angstrom',
        'stress_gpa': 'GPa', 'band_window_mev': 'meV', 'gate_use': 'dimensionless (1 = limit)',
        'work': 'reference/run ratio'})

    def system_rows(self, system: str) -> list[SweepRow]:
        return [r for r in self.rows if r.system == system]


def is_sweep_payload(data) -> bool:
    if not isinstance(data, dict):
        return False
    if data.get('schema_version') in SWEEP_REPORT_SCHEMAS:
        return True
    values = [v for v in data.values() if isinstance(v, dict)]
    return bool(values) and len(values) == len(data) and all(
        isinstance(v.get('rows'), list) and all(isinstance(r, dict) and 'variant' in r for r in v['rows'])
        for v in values)


def _summary_work(row: dict, system: dict) -> dict:
    """Ratios derivable from summary.json alone (no reference operator calls are stored)."""
    out = {}
    modes, xc = finite(row.get('density_modes')), finite(row.get('xc_points'))
    ref_modes, ref_xc = finite(system.get('reference_modes')), finite(system.get('reference_xc_points'))
    if modes and ref_modes:
        out['density_modes'] = ref_modes / modes
    if xc and ref_xc:
        out['xc_grid_points'] = ref_xc / xc
    return out


def _reference_calls(directory: Path | None, system: str) -> float | None:
    """operator_hamiltonian + operator_overlap calls of the reference run (performance sidecar)."""
    if directory is None:
        return None
    from .scf import load_timing
    path = directory / system / 'reference'
    if not path.is_dir():
        return None
    try:
        timing = load_timing(path)
    except InputError:
        return None
    calls = {e.name: e.calls for e in timing.entries}
    total = calls.get('operator_hamiltonian', 0) + calls.get('operator_overlap', 0)
    return float(total) if total else None


def load_sweep(path) -> SweepTable:
    p = Path(path)
    if p.is_dir():
        for name in ('report.json', 'summary.json'):
            if (p / name).is_file():
                p = p / name
                break
        else:
            raise InputError(f'no sweep report.json/summary.json in {p}')
    data = read_json(p)
    if not is_sweep_payload(data):
        raise InputError(f'{p} is neither a {SWEEP_REPORT_SCHEMA} report nor a sweep summary.json')
    rows, systems = [], []
    if data.get('schema_version') in SWEEP_REPORT_SCHEMAS:
        origin = 'report'
        for system, entry in data.get('systems', {}).items():
            if not entry.get('reference_converged', False):
                continue
            systems.append(system)
            for r in entry.get('rows', []):
                work = {k: float(v) for k, v in (r.get('work') or {}).items() if finite(v)}
                rows.append(SweepRow(system=system, variant=r['variant'], component=component_of(r['variant']),
                                     converged=bool(r.get('converged')), passed=r.get('pass'),
                                     gate_use=(finite(r.get('gate_use')) if r.get('gate_use') is not None else gate_use(r))
                                     if r.get('converged') else None,
                                     errors={k: finite(r.get(k)) for k in GATES}, work=work,
                                     natoms=entry.get('natoms')))
    else:
        origin = 'summary'
        directory = p.parent
        for system, entry in data.items():
            if not entry.get('reference_converged', True):
                continue
            systems.append(system)
            ref_calls = _reference_calls(directory, system)
            for r in entry.get('rows', []):
                work = _summary_work(r, entry)
                calls = (r.get('operator_hamiltonian_calls') or 0) + (r.get('operator_overlap_calls') or 0)
                if ref_calls and calls:
                    work['operator_applications'] = ref_calls / calls
                converged = bool(r.get('converged'))
                rows.append(SweepRow(system=system, variant=r['variant'], component=component_of(r['variant']),
                                     converged=converged, passed=r.get('pass'),
                                     gate_use=gate_use(r) if converged else None,
                                     errors={k: finite(r.get(k)) for k in GATES}, work=work,
                                     natoms=entry.get('natoms'), wall_speedup=finite(r.get('speedup'))))
    if not rows:
        raise InputError(f'{p} has no sweep rows with a converged reference')
    return SweepTable(source=str(p), rows=rows, systems=systems, origin=origin)


# --- parallel scaling --------------------------------------------------------
@dataclass
class ScalingRun:
    label: str
    build: str
    threads: int
    ranks: int
    wall_seconds: float
    user_seconds: float | None = None
    sys_seconds: float | None = None
    source: str | None = None

    @property
    def workers(self) -> int:
        return self.threads * self.ranks

    @property
    def occupancy(self) -> float | None:
        """(user + sys) / (wall * threads * ranks); 1 = every worker busy all the time."""
        if self.user_seconds is None or self.wall_seconds <= 0:
            return None
        return (self.user_seconds + (self.sys_seconds or 0.0)) / (self.wall_seconds * self.workers)


@dataclass
class ScalingPoint:
    build: str
    threads: int
    ranks: int
    workers: int
    wall_seconds: float                      # median over repeats
    wall_min_seconds: float
    wall_max_seconds: float
    repeats: int
    occupancy: float | None
    speedup: float | None                    # T(1 worker, same build) / T
    efficiency: float | None                 # speedup / workers


@dataclass
class ScalingSet:
    sources: list[str]
    system: str | None
    runs: list[ScalingRun]
    units: dict = field(default_factory=lambda: {'time': 's', 'occupancy': 'fraction', 'speedup': 'x',
                                                 'efficiency': 'fraction'})

    @property
    def builds(self) -> list[str]:
        return list(dict.fromkeys(r.build for r in self.runs))

    @property
    def axis(self) -> str:
        """'threads', 'ranks' or 'workers' (threads × ranks) depending on what varies."""
        if all(r.ranks == 1 for r in self.runs):
            return 'threads'
        if all(r.threads == 1 for r in self.runs):
            return 'ranks'
        return 'workers'

    def points(self, build: str) -> list[ScalingPoint]:
        groups: dict[tuple[int, int], list[ScalingRun]] = {}
        for r in self.runs:
            if r.build == build:
                groups.setdefault((r.threads, r.ranks), []).append(r)
        def median(values):
            v = sorted(values)
            m = len(v) // 2
            return v[m] if len(v) % 2 else 0.5 * (v[m - 1] + v[m])
        baseline = groups.get((1, 1))
        t1 = median([r.wall_seconds for r in baseline]) if baseline else None
        out = []
        for (threads, ranks), runs in sorted(groups.items(), key=lambda kv: (kv[0][0] * kv[0][1], kv[0])):
            wall = median([r.wall_seconds for r in runs])
            occ = [r.occupancy for r in runs if r.occupancy is not None]
            speed = t1 / wall if t1 and wall > 0 else None
            out.append(ScalingPoint(build, threads, ranks, threads * ranks, wall,
                                    min(r.wall_seconds for r in runs), max(r.wall_seconds for r in runs), len(runs),
                                    median(occ) if occ else None, speed,
                                    speed / (threads * ranks) if speed is not None else None))
        return out


def is_scaling_payload(data) -> bool:
    return isinstance(data, dict) and data.get('schema_version') in SCALING_SCHEMAS and isinstance(data.get('runs'), list)


def load_scaling(paths) -> ScalingSet:
    """Merge one or more ``sparse-dft-scaling-1`` files (several builds may be compared)."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    runs, systems, sources = [], [], []
    for path in paths:
        p = Path(path)
        data = read_json(p)
        if not is_scaling_payload(data):
            raise InputError(f'{p} is not a {SCALING_SCHEMA} file')
        sources.append(str(p))
        if data.get('system'):
            systems.append(str(data['system']))
        for i, r in enumerate(data['runs']):
            threads, ranks = r.get('threads', 1), r.get('ranks', 1)
            wall = finite(r.get('wall_seconds'))
            if not isinstance(threads, int) or not isinstance(ranks, int) or threads < 1 or ranks < 1:
                raise InputError(f'{p} run {i}: threads and ranks must be positive integers')
            if wall is None or wall <= 0:
                raise InputError(f'{p} run {i}: wall_seconds must be positive')
            runs.append(ScalingRun(label=str(r.get('label') or f'{threads}x{ranks}'),
                                   build=str(r.get('build') or data.get('build') or 'default'),
                                   threads=threads, ranks=ranks, wall_seconds=wall,
                                   user_seconds=finite(r.get('user_seconds')), sys_seconds=finite(r.get('sys_seconds')),
                                   source=str(p)))
    if not runs:
        raise InputError('scaling input contains no runs')
    unique = list(dict.fromkeys(systems))
    return ScalingSet(sources=sources, system=unique[0] if len(unique) == 1 else (', '.join(unique) or None), runs=runs)
