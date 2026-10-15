"""Validated numerical defaults shared by the Python Sparse-DFT front end.

The C++ core still receives explicit values in its input file.  This module
keeps the Python entry points from silently accepting nonsensical values and
provides a PAW-aware default plane-wave cutoff when no value is supplied.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

from .errors import InputError

# A PBE PAW calculation normally uses at least 1.3*max(ENMAX).  The floor is
# deliberately conservative for light elements whose ENMAX is unusually low.
DEFAULT_CUTOFF_EV = 520.0
DEFAULT_SCF_MAX_ITERATIONS = 240
DEFAULT_SCF_MIXING = 0.2
DEFAULT_NSCF_MAX_ITERATIONS = DEFAULT_SCF_MAX_ITERATIONS

# These are input sanity limits, not convergence claims.  Low-cutoff values
# remain usable for fast tests, but production callers should use the PAW
# recommendation returned by ``recommended_cutoff``.
MIN_CUTOFF_EV = 20.0
MAX_CUTOFF_EV = 5000.0
MIN_ITERATIONS = 2
MAX_ITERATIONS = 100_000

KPOINT_SPACING_LIMITS = (0.03, 2.0)  # inverse Angstrom
KPOINT_GRID_LIMITS = {"scf": 4096, "nscf": 32768, "bands": 32768}


def _finite(value, name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise InputError(f"{name} must be a finite number", details={"value": value}) from exc
    if not math.isfinite(out):
        raise InputError(f"{name} must be finite", details={"value": value})
    return out


def validate_cutoff(value: float | None, *, name: str = "cutoff_ev") -> float | None:
    if value is None:
        return None
    out = _finite(value, name)
    if not MIN_CUTOFF_EV <= out <= MAX_CUTOFF_EV:
        raise InputError(
            f"{name} is outside the supported range [{MIN_CUTOFF_EV:g}, {MAX_CUTOFF_EV:g}] eV",
            details={"value": out, "minimum": MIN_CUTOFF_EV, "maximum": MAX_CUTOFF_EV},
        )
    return out


def validate_density_cutoff(value: float | None, cutoff_ev: float) -> float | None:
    """Independent PAW fine support; None uses the C++ 16*ENCUT starting value.

    The smooth product requires at least 4*ENCUT. No arbitrary upper bound
    is imposed on the fine density cutoff; it must be converged separately.
    """
    if value is None:
        return None
    out = _finite(value, 'density_cutoff_ev')
    if out < 4 * cutoff_ev:
        raise InputError('density_cutoff_ev must be at least 4*cutoff_ev',
                         details={'value': out, 'minimum': 4 * cutoff_ev})
    return out


def recommended_cutoff(candidates: Mapping | None = None) -> float:
    """Return the PAW-aware production starting cutoff in eV.

    ``candidates`` may contain resolved PAW records with ``enmax_ev``.  A
    30% margin follows the usual PAW convergence starting point.  The result
    is rounded up to a 5 eV boundary for reproducible input files.
    """
    enmax = []
    for candidate in (candidates or {}).values():
        raw = candidate.get("enmax_ev") if isinstance(candidate, Mapping) else getattr(candidate, "enmax_ev", None)
        if raw is None:
            continue
        value = _finite(raw, "PAW ENMAX")
        if value > 0:
            enmax.append(value)
    if not enmax:
        return DEFAULT_CUTOFF_EV
    target = max(300.0, 1.30 * max(enmax))
    # Keep automatic values in the accepted range while preserving a clear
    # failure for a corrupt PAW metadata value.
    target = min(MAX_CUTOFF_EV, max(MIN_CUTOFF_EV, target))
    return float(5.0 * math.ceil(target / 5.0))


def resolve_cutoff(value: float | None, candidates: Mapping | None = None) -> tuple[float, bool]:
    """Resolve an optional user value; return ``(cutoff, was_automatic)``."""
    if value is None:
        return recommended_cutoff(candidates), True
    return validate_cutoff(value), False


def automatic_cutoff(enmax_values, accuracy: str = "standard") -> float:
    """Compatibility helper for callers that already extracted PAW ENMAX."""
    margin = {"low": 1.15, "standard": 1.30, "high": 1.40, "production": 1.40}.get(str(accuracy).lower())
    if margin is None:
        raise InputError("unsupported accuracy level", details={"accuracy": accuracy})
    values = []
    for raw in enmax_values:
        value = _finite(raw, "PAW ENMAX")
        if value <= 0:
            raise InputError("PAW ENMAX must be positive", details={"value": value})
        values.append(value)
    if not values:
        return DEFAULT_CUTOFF_EV
    target = min(MAX_CUTOFF_EV, max(MIN_CUTOFF_EV, max(300.0, margin * max(values))))
    return float(5.0 * math.ceil(target / 5.0))


def validate_nscf_path(kpoints, *, maximum: int = KPOINT_GRID_LIMITS["nscf"]):
    """Validate an explicit fractional NSCF/band path without imposing a box."""
    if isinstance(kpoints, (str, bytes)):
        raise InputError("NSCF k-point path must be a sequence of points")
    try:
        points = [tuple(float(x) for x in point[:3]) for point in kpoints]
    except (TypeError, ValueError, IndexError) as exc:
        raise InputError("NSCF k-point path must contain 3-component points") from exc
    if len(points) < 2:
        raise InputError("NSCF k-point path requires at least two points", details={"count": len(points)})
    if len(points) > maximum:
        raise InputError("NSCF k-point path is too large", details={"count": len(points), "maximum": maximum})
    if any(len(point) != 3 or any(not math.isfinite(v) for v in point) for point in points):
        raise InputError("NSCF k-point path contains non-finite coordinates")
    return points


def validate_max_iterations(value: int, *, name: str = "max_iterations") -> int:
    try:
        # Reject 3.5 instead of silently truncating it to 3.
        if isinstance(value, bool) or int(value) != value:
            raise ValueError
        out = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InputError(f"{name} must be an integer", details={"value": value}) from exc
    if not MIN_ITERATIONS <= out <= MAX_ITERATIONS:
        raise InputError(
            f"{name} is outside the supported range [{MIN_ITERATIONS}, {MAX_ITERATIONS}]",
            details={"value": out, "minimum": MIN_ITERATIONS, "maximum": MAX_ITERATIONS},
        )
    return out


def validate_timeout(value: float | None) -> float | None:
    """Validate an optional network timeout; ``None`` means no hard timeout."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise InputError("timeout must be positive or null for no timeout", details={"value": value})
    out = _finite(value, "timeout")
    if out <= 0:
        raise InputError("timeout must be positive or null for no timeout", details={"value": value})
    return out
