"""Relaxation trajectory loader for sparse_dft.relax / sparse_dft.analytic_relax outputs.

Supported files (schemas written by this code base):
* ``relax_result.json`` (analytic L-BFGS result: status, history, tolerance)
* ``relax_checkpoint.json`` / ``cell_relax_checkpoint.json`` with schema
  ``sparse-dft-analytic-relax-3`` (history + accepted_state), ``sparse-dft-relax-2``
  (finite-difference ionic) or ``sparse-dft-cell-relax-2`` (finite-difference cell).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import math

from ..errors import InputError
from .common import EV_PER_ANGSTROM3_TO_GPA, finite, read_json, schema_ids

RELAX_FILE_NAMES = ('relax_result.json', 'relax_checkpoint.json', 'cell_relax_checkpoint.json')
RELAX_SCHEMAS = schema_ids('analytic-relax-3') + schema_ids('relax-2') + schema_ids('cell-relax-2')
# Defaults of sparse_dft.analytic_relax.analytic_relax / sparse_dft.relax (used only when
# a checkpoint does not record its tolerance; flagged in ``tolerance_origin``).
DEFAULT_TOLERANCE = {'ionic': 0.02, 'cell': 0.00031207545}


@dataclass
class RelaxTrajectory:
    source: str
    kind: str                                  # 'ionic' | 'cell'
    status: str | None
    steps: list[int]
    energy_ev: list[float | None]
    residual: list[float | None]               # max |F| (eV/Å) or max |sigma| (eV/Å^3)
    step_length: list[float | None]            # |Δx| between accepted states (Å or strain)
    line_search_trials: list[int | None]
    tolerance: float | None
    tolerance_origin: str                      # 'file' | 'default' | 'option' | 'none'
    total_energy_valid: list[bool | None] = field(default_factory=list)
    natoms: int | None = None
    units: dict = field(default_factory=dict)

    @property
    def residual_display(self) -> list[float | None]:
        """Residual in plotting units: eV/Å for ionic, GPa for cell."""
        if self.kind == 'cell':
            return [None if r is None else r * EV_PER_ANGSTROM3_TO_GPA for r in self.residual]
        return list(self.residual)

    @property
    def tolerance_display(self) -> float | None:
        if self.tolerance is None:
            return None
        return self.tolerance * EV_PER_ANGSTROM3_TO_GPA if self.kind == 'cell' else self.tolerance


def find_relax_file(path) -> Path:
    p = Path(path)
    if p.is_dir():
        for name in RELAX_FILE_NAMES:
            if (p / name).is_file():
                return p / name
        raise InputError(f'no relaxation output ({", ".join(RELAX_FILE_NAMES)}) in {p}')
    if not p.is_file():
        raise InputError(f'file not found: {p}')
    return p


def is_relax_payload(data) -> bool:
    if not isinstance(data, dict) or not isinstance(data.get('history'), list):
        return False
    return data.get('schema_version') in RELAX_SCHEMAS or (
        'status' in data and any('energy_ev' in h for h in data['history'] if isinstance(h, dict)))


def _max_abs(matrix) -> float | None:
    try:
        values = [abs(float(v)) for row in matrix for v in row]
    except (TypeError, ValueError):
        return None
    return max(values) if values else None


def _max_norm(matrix) -> float | None:
    try:
        return max(math.sqrt(sum(float(v) ** 2 for v in row)) for row in matrix)
    except (TypeError, ValueError):
        return None


def load_relax(path, *, tolerance=None) -> RelaxTrajectory:
    p = find_relax_file(path)
    data = read_json(p)
    if not is_relax_payload(data):
        raise InputError(f'{p} is not a Sparse-DFT relaxation result/checkpoint')
    schema = data.get('schema_version')
    history = [h for h in data['history'] if isinstance(h, dict)]
    if not history:
        raise InputError(f'{p} has an empty relaxation history')
    units = data.get('derivative_units') or ''
    cell = (schema in schema_ids('cell-relax-2') or 'angstrom^3' in units or p.name.startswith('cell_')
            or data.get('cell_ions') == 'fixed_fractional_coordinates'
            or any('stress_ev_per_angstrom3' in h for h in history))
    kind = 'cell' if cell else 'ionic'
    steps, energy, residual, length, trials, valid = [], [], [], [], [], []
    previous = None
    for i, h in enumerate(history):
        steps.append(int(h.get('step', i)))
        energy.append(finite(h.get('energy_ev')))
        if 'residual' in h:
            residual.append(finite(h.get('residual')))
        elif 'max_force_ev_per_angstrom' in h:
            residual.append(finite(h.get('max_force_ev_per_angstrom')))
        elif 'stress_ev_per_angstrom3' in h:
            residual.append(_max_abs(h['stress_ev_per_angstrom3']))
        else:
            residual.append(None)
        coords = h.get('coordinates')
        if isinstance(coords, list) and previous is not None and len(previous) == len(coords):
            length.append(math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(coords, previous))))
        else:
            length.append(None)
        previous = coords if isinstance(coords, list) else None
        t = h.get('line_search_trials')
        trials.append(int(t) if isinstance(t, (int, float)) else None)
        v = h.get('total_energy_valid')
        valid.append(bool(v) if isinstance(v, bool) else None)
    if tolerance is not None:
        tol, origin = float(tolerance), 'option'
    elif finite(data.get('tolerance')) is not None:
        tol, origin = float(data['tolerance']), 'file'
    elif schema in RELAX_SCHEMAS:
        tol, origin = DEFAULT_TOLERANCE[kind], 'default'
    else:
        tol, origin = None, 'none'
    natoms = None
    structure = history[-1].get('structure') or data.get('structure')
    if isinstance(structure, dict) and isinstance(structure.get('species'), list):
        natoms = len(structure['species'])
    return RelaxTrajectory(source=str(p), kind=kind, status=data.get('status'), steps=steps,
                           energy_ev=energy, residual=residual, step_length=length,
                           line_search_trials=trials, tolerance=tol, tolerance_origin=origin,
                           total_energy_valid=valid, natoms=natoms,
                           units={'energy': 'eV', 'residual': 'eV/angstrom^3' if cell else 'eV/angstrom',
                                  'residual_display': 'GPa' if cell else 'eV/angstrom',
                                  'step_length': 'strain' if cell else 'angstrom'})
