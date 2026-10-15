"""Loaders for Sparse-DFT SCF outputs: result ledger, iteration history, timers.

Every quantity keeps its unit in the field name (``*_ev``, ``*_muB``,
``*_ev_per_angstrom``) or in the ``units`` mapping of its container.
Nothing here invents a value: absent fields stay ``None``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import math
import re

from ..errors import InputError
from .common import EV_PER_ANGSTROM3_TO_GPA, finite, read_json, read_jsonl, schema_ids

# Result files in order of preference inside one calculation directory.
SCF_RESULT_NAMES = ('scf_result.json', 'scf.json', 'relax_scf_evidence.json')
SCF_SCHEMA = 'sparse-dft-scf-1'
SCF_SCHEMAS = schema_ids('scf-1')

# Signed terms of the native total-energy ledger, exactly as assembled in
# cpp/scf_core.cpp (EBANDS - E_H + E_xc - int v_xc rho + PAW one-centre
# - <D^1> - E_xc^core + Ewald + local G=0 + EATOM + cutoff correction - TS).
ENERGY_LEDGER = (
    ('band_energy_ev', +1, 'band'),
    ('hartree_energy_ev', -1, 'hartree'),
    ('xc_energy_ev', +1, 'xc'),
    ('vxc_density_ev', -1, 'vxc_density'),
    ('paw_onecenter_energy_ev', +1, 'paw_onecenter'),
    ('paw_onecenter_potential_expectation_ev', -1, 'paw_expectation'),
    ('paw_onecenter_core_xc_ev', -1, 'paw_core_xc'),
    ('ion_ion_energy_ev', +1, 'ion_ion'),
    ('local_g0_energy_ev', +1, 'local_g0'),
    ('atomic_reference_energy_ev', +1, 'atomic_reference'),
    ('atomic_cutoff_correction_ev', +1, 'cutoff_correction'),
    ('entropy_energy_ev', +1, 'entropy'),
)
LEDGER_TOLERANCE_EV = 1e-6


@dataclass
class EnergyTerm:
    key: str            # result-file field, e.g. 'hartree_energy_ev'
    name: str           # short identifier used for labels, e.g. 'hartree'
    sign: int           # sign with which the field enters the ledger
    value_ev: float     # raw field value (eV)

    @property
    def contribution_ev(self) -> float:
        return self.sign * self.value_ev


@dataclass
class EnergyTerms:
    source: str
    terms: list[EnergyTerm]
    total_energy_ev: float
    internal_energy_ev: float | None
    total_energy_valid: bool
    natoms: int | None
    ledger_sum_ev: float
    ledger_residual_ev: float          # ledger_sum_ev - total_energy_ev
    tolerance_ev: float = LEDGER_TOLERANCE_EV
    missing_terms: list[str] = field(default_factory=list)
    units: dict = field(default_factory=lambda: {'energy': 'eV'})

    @property
    def additive(self) -> bool:
        """True only when every ledger term exists and the sum reproduces the total."""
        return not self.missing_terms and abs(self.ledger_residual_ev) <= self.tolerance_ev


@dataclass
class ForceSet:
    source: str
    forces_ev_per_angstrom: list[list[float]]
    species: list[str] | None
    validated: bool
    status: str | None
    components_ev_per_angstrom: dict[str, list[list[float]]] = field(default_factory=dict)
    total_energy_valid: bool | None = None
    units: dict = field(default_factory=lambda: {'force': 'eV/angstrom'})

    @property
    def natoms(self) -> int:
        return len(self.forces_ev_per_angstrom)

    @property
    def magnitudes_ev_per_angstrom(self) -> list[float]:
        return [math.sqrt(sum(c * c for c in row)) for row in self.forces_ev_per_angstrom]

    @property
    def max_ev_per_angstrom(self) -> float:
        return max(self.magnitudes_ev_per_angstrom, default=0.0)

    @property
    def net_force_ev_per_angstrom(self) -> list[float]:
        return [sum(row[i] for row in self.forces_ev_per_angstrom) for i in range(3)]

    def atom_labels(self) -> list[str]:
        if self.species and len(self.species) == self.natoms:
            return [f'{s}{i + 1}' for i, s in enumerate(self.species)]
        return [str(i + 1) for i in range(self.natoms)]


@dataclass
class StressTensor:
    source: str
    stress_ev_per_angstrom3: list[list[float]]
    validated: bool
    convention: str | None
    components_ev_per_angstrom3: dict[str, list[list[float]]] = field(default_factory=dict)
    total_energy_valid: bool | None = None
    units: dict = field(default_factory=lambda: {'stress': 'eV/angstrom^3', 'stress_gpa': 'GPa'})

    @property
    def stress_gpa(self) -> list[list[float]]:
        return [[v * EV_PER_ANGSTROM3_TO_GPA for v in row] for row in self.stress_ev_per_angstrom3]

    @property
    def mean_normal_stress_gpa(self) -> float:
        g = self.stress_gpa
        return (g[0][0] + g[1][1] + g[2][2]) / 3.0

    @property
    def pressure_gpa(self) -> float | None:
        """-tr(sigma)/3; only defined for the documented positive-tension convention."""
        if self.convention and 'positive_tension' in self.convention:
            return -self.mean_normal_stress_gpa
        return None


@dataclass
class ScfRun:
    source: str
    schema_version: str | None
    converged: bool
    total_energy_ev: float | None
    total_energy_valid: bool
    fermi_energy_ev: float | None
    iterations: int | None
    spin_channels: int
    magnetic_moment_muB: float | None
    natoms: int | None
    species: list[str] | None
    electron_count: float | None
    smearing_ev: float | None
    effective_density_tolerance: float | None
    effective_onecenter_density_tolerance: float | None
    raw: dict = field(default_factory=dict, repr=False, compare=False)
    units: dict = field(default_factory=lambda: {
        'energy': 'eV', 'magnetic_moment': 'muB', 'smearing': 'eV'})


@dataclass
class ScfHistory:
    source: str
    iteration: list[int]
    series: dict[str, list[float | None]]
    tolerances: dict[str, float]
    eigensolver_tolerance_hartree: float | None
    spin_polarized: bool
    converged: bool | None
    result_source: str | None = None
    units: dict = field(default_factory=lambda: {
        'density_residual': 'max |rho_out - rho_in| (native density units)',
        'onecenter_density_residual': 'dimensionless (relative Frobenius)',
        'eigensolver_max_residual_hartree': 'hartree',
        'output_magnetic_moment_muB': 'muB', 'mixed_magnetic_moment_muB': 'muB'})

    @property
    def niterations(self) -> int:
        return len(self.iteration)


@dataclass
class TimerEntry:
    name: str
    seconds: float
    calls: int


@dataclass
class TimingProfile:
    source: str
    entries: list[TimerEntry]
    timings_overlap: bool
    backend: str | None = None
    device: str | None = None
    openmp_max_threads: int | None = None
    wall_seconds: float | None = None       # process_compute_total (or scf_total)
    origin: str = 'performance.json'
    units: dict = field(default_factory=lambda: {'time': 's', 'calls': 'count'})

    def top(self, n: int = 15) -> list[TimerEntry]:
        """Largest timed counters; zero-time entries are pure call counters."""
        timed = [e for e in self.entries if e.seconds > 0.0]
        return sorted(timed, key=lambda e: (-e.seconds, e.name))[:max(0, int(n))]


def find_scf_result(path) -> Path:
    p = Path(path)
    if p.is_dir():
        for name in SCF_RESULT_NAMES:
            if (p / name).is_file():
                return p / name
        raise InputError(f'no SCF result ({", ".join(SCF_RESULT_NAMES)}) in {p}')
    if not p.is_file():
        raise InputError(f'file not found: {p}')
    return p


def is_scf_payload(data) -> bool:
    return isinstance(data, dict) and (data.get('schema_version') in SCF_SCHEMAS or (
        'total_energy_ev' in data and ('scf_kpoints' in data or 'band_energy_ev' in data)))


def _scf_payload(path):
    p = find_scf_result(path)
    data = read_json(p)
    if not is_scf_payload(data):
        raise InputError(f'{p} is not a Sparse-DFT SCF result (schema {SCF_SCHEMA})')
    return p, data


_ATOM_LINE = re.compile(r'^atom\s+([A-Z][a-z]?)\s')


def _species(data, path: Path) -> list[str] | None:
    states = data.get('onecenter_states') or []
    if states and all(isinstance(s, dict) and 'symbol' in s for s in states):
        ordered = sorted(states, key=lambda s: s.get('atom_index', 0))
        indices = [s.get('atom_index') for s in ordered]
        if indices == list(range(len(ordered))):
            return [str(s['symbol']) for s in ordered]
    for name in ('scf_input.sdft', 'bands_input.sdft'):
        candidate = path.parent / name
        if candidate.is_file():
            with open(candidate, encoding='utf-8', errors='replace') as handle:
                symbols = [m.group(1) for m in (_ATOM_LINE.match(line) for line in handle) if m]
            if symbols:
                return symbols
    return None


def load_scf_run(path) -> ScfRun:
    p, data = _scf_payload(path)
    forces = data.get('hf_total_forces_ev_per_angstrom')
    species = _species(data, p)
    natoms = len(forces) if isinstance(forces, list) and forces else (len(species) if species else None)
    spin = data.get('spin_channels') or 1
    return ScfRun(
        source=str(p), schema_version=data.get('schema_version'),
        converged=bool(data.get('converged')),
        total_energy_ev=finite(data.get('total_energy_ev')),
        total_energy_valid=bool(data.get('total_energy_valid', False)),
        fermi_energy_ev=finite(data.get('fermi_energy_ev')),
        iterations=data.get('iterations'), spin_channels=int(spin),
        magnetic_moment_muB=finite(data.get('magnetic_moment_muB')) if int(spin) == 2 else None,
        natoms=natoms, species=species,
        electron_count=finite(data.get('electron_count')),
        smearing_ev=finite((data.get('provenance') or {}).get('smearing_ev')),
        effective_density_tolerance=finite(data.get('effective_density_tolerance')),
        effective_onecenter_density_tolerance=finite(data.get('effective_onecenter_density_tolerance')),
        raw=data)


def load_energy_terms(path) -> EnergyTerms:
    p, data = _scf_payload(path)
    total = finite(data.get('total_energy_ev'))
    if total is None:
        raise InputError(f'{p} has no finite total_energy_ev')
    terms, missing = [], []
    for key, sign, name in ENERGY_LEDGER:
        value = finite(data.get(key))
        if value is None:
            missing.append(key)
            continue
        terms.append(EnergyTerm(key, name, sign, value))
    ledger = math.fsum(t.contribution_ev for t in terms)
    forces = data.get('hf_total_forces_ev_per_angstrom')
    return EnergyTerms(source=str(p), terms=terms, total_energy_ev=total,
                       internal_energy_ev=finite(data.get('internal_energy_ev')),
                       total_energy_valid=bool(data.get('total_energy_valid', False)),
                       natoms=len(forces) if isinstance(forces, list) and forces else None,
                       ledger_sum_ev=ledger, ledger_residual_ev=ledger - total,
                       missing_terms=missing)


def _matrix(value, rows, cols=3):
    if not isinstance(value, list) or (rows is not None and len(value) != rows):
        return None
    out = []
    for row in value:
        if not isinstance(row, list) or len(row) != cols:
            return None
        numbers = [finite(x) for x in row]
        if any(x is None for x in numbers):
            return None
        out.append(numbers)
    return out


def load_forces(path) -> ForceSet:
    p, data = _scf_payload(path)
    forces = _matrix(data.get('hf_total_forces_ev_per_angstrom'), None)
    if not forces:
        raise InputError(f'{p} has no finite hf_total_forces_ev_per_angstrom (natoms x 3)')
    components = {}
    for key, name in (('hf_electronic_forces_ev_per_angstrom', 'electronic'),
                      ('ewald_forces_ev_per_angstrom', 'ewald')):
        m = _matrix(data.get(key), len(forces))
        if m:
            components[name] = m
    prov = data.get('force_provenance') or {}
    species = _species(data, p)
    return ForceSet(source=str(p), forces_ev_per_angstrom=forces,
                    species=species if species and len(species) == len(forces) else None,
                    validated=bool(prov.get('validated', False)), status=prov.get('status'),
                    components_ev_per_angstrom=components,
                    total_energy_valid=bool(data.get('total_energy_valid', False)))


def load_stress(path) -> StressTensor:
    p, data = _scf_payload(path)
    stress = _matrix(data.get('hf_total_stress_ev_per_angstrom3'), 3)
    if not stress:
        raise InputError(f'{p} has no finite 3x3 hf_total_stress_ev_per_angstrom3')
    components = {}
    for name, value in (data.get('stress_components_ev_per_angstrom3') or {}).items():
        m = _matrix(value, 3)
        if m:
            components[str(name)] = m
    prov = data.get('stress_provenance') or {}
    return StressTensor(source=str(p), stress_ev_per_angstrom3=stress,
                        validated=bool(prov.get('validated', False)),
                        convention=prov.get('convention'), components_ev_per_angstrom3=components,
                        total_energy_valid=bool(data.get('total_energy_valid', False)))


# --- iteration history -----------------------------------------------------
HISTORY_SERIES = ('density_residual', 'onecenter_density_residual', 'eigensolver_max_residual_hartree',
                  'output_magnetic_moment_muB', 'mixed_magnetic_moment_muB', 'mixing_step_scale',
                  'multisecant_rank')
_SCALAR = re.compile(r'"(effective_density_tolerance|effective_onecenter_density_tolerance|'
                     r'density_tolerance|onecenter_density_tolerance|converged|spin_channels)"\s*:\s*'
                     r'(true|false|null|[-+0-9.eE]+)')


def _sidecar(path, suffix) -> Path | None:
    """Locate ``<result>.<suffix>`` next to an SCF result or inside a directory."""
    p = Path(path)
    if p.is_file() and p.name.endswith(suffix):
        return p
    directory = p if p.is_dir() else p.parent
    stems = ([p.name] if p.is_file() else []) + list(SCF_RESULT_NAMES)
    for stem in stems:
        candidate = directory / f'{stem}.{suffix}'
        if candidate.is_file():
            return candidate
    matches = sorted(directory.glob(f'*.{suffix}'))
    return matches[0] if len(matches) == 1 else None


def iterations_file(path) -> Path | None:
    return _sidecar(path, 'iterations.jsonl')


def performance_file(path) -> Path | None:
    return _sidecar(path, 'performance.json')


def _result_scalars(directory: Path) -> tuple[dict, Path | None]:
    """Small top-level fields of the sibling result without parsing 10 MB arrays."""
    for name in SCF_RESULT_NAMES:
        p = directory / name
        if not p.is_file():
            continue
        with open(p, encoding='utf-8', errors='replace') as handle:
            head = handle.read(1 << 16)
        found = {}
        for m in _SCALAR.finditer(head):
            raw = m.group(2)
            found.setdefault(m.group(1), True if raw == 'true' else False if raw == 'false'
                             else None if raw == 'null' else float(raw))
        if found:
            return found, p
    return {}, None


def load_scf_history(path) -> ScfHistory:
    src = iterations_file(path)
    if src is None:
        raise InputError(f'no *.iterations.jsonl SCF history found for {path}')
    records = read_jsonl(src)
    if not records:
        raise InputError(f'{src} contains no SCF iterations')
    iteration = [int(r.get('iteration', i + 1)) for i, r in enumerate(records)]
    series = {}
    for key in HISTORY_SERIES:
        values = [finite(r.get(key)) for r in records]
        if any(v is not None for v in values):
            series[key] = values
    scalars, result = _result_scalars(src.parent)
    tolerances = {}
    for key, fallback in (('density_residual', 'density_tolerance'),
                          ('onecenter_density_residual', 'onecenter_density_tolerance')):
        value = finite(scalars.get('effective_' + fallback, scalars.get(fallback)))
        if value is not None:
            tolerances[key] = value
    eig_tol = finite(records[-1].get('eigensolver_final_tolerance_hartree',
                                     records[-1].get('eigensolver_requested_tolerance_hartree')))
    spin = 'output_magnetic_moment_muB' in series or scalars.get('spin_channels') == 2.0
    return ScfHistory(source=str(src), iteration=iteration, series=series, tolerances=tolerances,
                      eigensolver_tolerance_hartree=eig_tol, spin_polarized=bool(spin),
                      converged=scalars.get('converged'), result_source=str(result) if result else None)


# --- timers ----------------------------------------------------------------
def _entries(counters) -> list[TimerEntry]:
    out = []
    for name, value in (counters or {}).items():
        if not isinstance(value, dict):
            continue
        seconds, calls = finite(value.get('seconds')), value.get('calls')
        if seconds is None:
            continue
        out.append(TimerEntry(str(name), seconds, int(calls) if isinstance(calls, (int, float)) else 0))
    return out


def _wall(entries) -> float | None:
    by_name = {e.name: e.seconds for e in entries}
    for key in ('process_compute_total', 'scf_total'):
        if by_name.get(key):
            return by_name[key]
    return None


def load_timing(path) -> TimingProfile:
    src = performance_file(path)
    if src is not None:
        data = read_json(src)
        if not isinstance(data, dict) or not isinstance(data.get('counters'), dict):
            raise InputError(f'{src} has no counters mapping')
        entries = _entries(data['counters'])
        return TimingProfile(source=str(src), entries=entries,
                             timings_overlap=bool(data.get('timings_overlap', True)),
                             backend=data.get('backend'), device=data.get('device') or None,
                             openmp_max_threads=data.get('openmp_max_threads'),
                             wall_seconds=_wall(entries), origin='performance.json')
    p, data = _scf_payload(path)
    ledger = data.get('sparsity_ledger') or {}
    if not isinstance(ledger.get('counters'), dict):
        raise InputError(f'no performance counters for {path}')
    entries = _entries(ledger['counters'])
    # The ledger is a subset of the same nested timers; it overlaps too.
    return TimingProfile(source=str(p), entries=entries, timings_overlap=True,
                         wall_seconds=_wall(entries), origin='sparsity_ledger')
