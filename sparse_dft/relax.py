"""Analytic PAW relaxation and independent finite-difference verification.

The force/stress path intentionally consumes only converged C++ SCF results.
This makes incomplete electronic convergence an actionable error rather than a
silent source of bogus geometry updates.
"""
from __future__ import annotations

from pathlib import Path
import copy
import json
import math
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from .errors import InputError
from .scf import run_scf
from .structure import Structure

HARTREE_TO_EV = 27.211386245988
BOHR_PER_ANGSTROM = 1.8897261254578281


def _inv(m):
    a, b, c = m
    cross = lambda x, y: (x[1]*y[2]-x[2]*y[1], x[2]*y[0]-x[0]*y[2], x[0]*y[1]-x[1]*y[0])
    dot = lambda x, y: sum(i*j for i, j in zip(x, y))
    rows = (cross(b, c), cross(c, a), cross(a, b)); d = dot(a, rows[0])
    if abs(d) < 1e-14: raise InputError("singular lattice")
    return tuple(tuple(x / d for x in row) for row in rows)


def _mat_vec(m, v):
    return tuple(sum(m[i][j] * v[j] for j in range(3)) for i in range(3))


def _apply_strain(lattice, eps):
    """Return lattice with Cartesian deformation ``F = I + eps`` applied.

    Lattice vectors are stored as rows.  Applying the deformation to their
    Cartesian components avoids the component-wise scaling bug that made
    shear perturbations depend on the chosen coordinate representation.
    """
    return tuple(tuple(sum(lattice[r][d] * ((1.0 if c == d else 0.0) + eps[c][d])
                           for d in range(3)) for c in range(3)) for r in range(3))


def _shift_fractional(s: Structure, atom: int, axis: int, delta_angstrom: float):
    inv = _inv(s.lattice)
    cart = [0.0, 0.0, 0.0]; cart[axis] = delta_angstrom
    frac_delta = _mat_vec(inv, cart)
    positions = list(s.positions); positions[atom] = tuple(positions[atom][j] + frac_delta[j] for j in range(3))
    return Structure(list(s.species), positions, s.lattice, True, s.periodic, s.selective_dynamics,
                     s.source_format, s.source_path, list(s.warnings)).normalized()


def _energy(structure, paw_root, workdir, **kwargs):
    result = run_scf(structure, paw_root, workdir, **kwargs)
    if not result.get("converged"):
        raise InputError("geometry derivative requires a converged SCF",
                         details={"scf": result})
    return float(result["total_energy_ev"]), result


def _json_hash(payload):
    """Stable hash for restart compatibility checks."""
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(raw).hexdigest()


def _atomic_json_write(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding='utf-8')
    os.replace(tmp, path)


def _structure_from_dict(payload):
    return Structure(payload["species"], [tuple(x) for x in payload["positions"]],
                     tuple(tuple(x) for x in payload["lattice"]) if payload.get("lattice") else None,
                     bool(payload.get("fractional")), tuple(payload.get("periodic", (False, False, False))),
                     payload.get("selective_dynamics"), payload.get("source_format", "restart"),
                     payload.get("source_path"), payload.get("warnings", [])).normalized()


def finite_difference_forces(structure, paw_root, workdir, *, step_angstrom=1e-3,
                             max_workers=1, cache=None, **kwargs):
    """Central finite-difference force estimator with reproducible caching.

    This remains a verification/reference path until analytic PAW force terms are
    validated.  Independent +/- displacements may be evaluated concurrently;
    each SCF receives its own directory so subprocesses never share state.
    """
    s = structure.normalized()
    workdir = Path(workdir); workdir.mkdir(parents=True, exist_ok=True)
    forces = [[0.0, 0.0, 0.0] for _ in range(s.natoms)]
    samples = {}
    jobs = []
    for atom in range(s.natoms):
        for axis in range(3):
            plus = _shift_fractional(s, atom, axis, step_angstrom)
            minus = _shift_fractional(s, atom, axis, -step_angstrom)
            for sign, ss in (("plus", plus), ("minus", minus)):
                key = f"{atom}:{axis}:{sign}"
                jobs.append((key, atom, axis, sign, ss))

    def evaluate(item):
        key, atom, axis, sign, ss = item
        subdir = workdir / f"fd_{atom}_{axis}_{sign}"
        # A completed JSON result is immutable evidence and can be reused after
        # an interrupted relax.  The structure hash prevents stale reuse.
        result_file = subdir / "scf_result.json"
        if result_file.exists():
            try:
                previous = json.loads(result_file.read_text(encoding='utf-8'))
                if previous.get("structure_hash") == ss.content_hash() and previous.get("converged"):
                    return key, float(previous["total_energy_ev"]), previous, "cached"
            except (OSError, ValueError, KeyError):
                pass
        energy, result = _energy(ss, paw_root, subdir, **kwargs)
        return key, energy, result, "computed"

    if max_workers and max_workers > 1:
        with ThreadPoolExecutor(max_workers=min(int(max_workers), len(jobs))) as pool:
            futures = [pool.submit(evaluate, job) for job in jobs]
            for fut in as_completed(futures):
                key, energy, result, source = fut.result(); samples[key] = (energy, source)
    else:
        for job in jobs:
            key, energy, result, source = evaluate(job); samples[key] = (energy, source)
    energies = []
    for key, atom, axis, sign, _ in jobs:
        energies.append(samples[key][0])
    for atom in range(s.natoms):
        for axis in range(3):
            ep = samples[f"{atom}:{axis}:plus"][0]
            em = samples[f"{atom}:{axis}:minus"][0]
            forces[atom][axis] = -(ep - em) / (2.0 * step_angstrom)
    # Preserve the measured translation sum rule; subtracting a mean would
    # conceal SCF/derivative errors in an independent validation path.
    net_force = [sum(row[a] for row in forces) for a in range(3)]
    return {"forces_ev_per_angstrom": forces, "step_angstrom": step_angstrom,
            "energy_samples_ev": energies, "method": "central_finite_difference",
            "analytic": False, "cached_samples": sum(v[1] == "cached" for v in samples.values()),
            "sample_count": len(samples), "net_force_ev_per_angstrom": net_force}


def _finite_difference_ionic_relax(structure: str | Path | Structure, paw_root: str | Path, output: str | Path,
                *, max_steps=20, force_tolerance=0.02, step_angstrom=1e-3,
                move_scale=0.05, max_displacement=0.15, max_workers=1, **kwargs):
    from .structure import read_structure
    s = read_structure(structure) if isinstance(structure, (str, Path)) else structure
    s = s.normalized(); out = Path(output); out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "relax_checkpoint.json"
    signature = _json_hash({"kind": "ionic", "paw_root": str(Path(paw_root).resolve()),
                            "force_tolerance": force_tolerance, "step_angstrom": step_angstrom,
                            "move_scale": move_scale, "max_displacement": max_displacement,
                            "kwargs": kwargs})
    history = []
    start_step = 0
    if checkpoint.exists():
        saved = json.loads(checkpoint.read_text(encoding='utf-8'))
        if saved.get("schema_version") in ("sparse-dft-relax-1", "sparse-dft-relax-2") and saved.get("structure"):
            if saved.get("schema_version") == "sparse-dft-relax-2" and saved.get("signature") != signature:
                raise InputError("ionic relaxation checkpoint parameters do not match current request",
                                 details={"checkpoint": str(checkpoint), "expected_signature": signature,
                                          "found_signature": saved.get("signature")})
            s = _structure_from_dict(saved["structure"])
            history = saved.get("history", [])
            start_step = int(saved.get("step", -1)) + 1
    for step in range(start_step, start_step + max_steps):
        step_dir = out / f"step_{step:04d}"
        energy, scf = _energy(s, paw_root, step_dir, **kwargs)
        derivative = finite_difference_forces(s, paw_root, out / f"derivatives_{step:04d}",
                                              step_angstrom=step_angstrom, max_workers=max_workers, **kwargs)
        forces = derivative["forces_ev_per_angstrom"]
        max_force = max(abs(x) for row in forces for x in row)
        history.append({"step": step, "energy_ev": energy, "max_force_ev_per_angstrom": max_force, "structure": s.as_dict()})
        _atomic_json_write(checkpoint, {"schema_version": "sparse-dft-relax-2", "signature": signature,
                                        "step": step, "history": history, "structure": s.as_dict()})
        if max_force <= force_tolerance:
            return {"status": "CONVERGED", "steps": step + 1, "structure": s.as_dict(), "history": history,
                    "forces": derivative, "checkpoint": str(checkpoint), "analytic": False,
                    "validation": "finite_difference_only"}
        positions = list(s.positions)
        inv = _inv(s.lattice)
        for atom, force in enumerate(forces):
            if s.selective_dynamics and not all(s.selective_dynamics[atom]):
                force = [force[i] if s.selective_dynamics[atom][i] else 0.0 for i in range(3)]
            # Convert a force direction into a bounded Cartesian displacement.
            # The old implementation could take arbitrarily large steps when
            # the SCF energy was noisy, causing repeated cell crossings.
            norm_force = math.sqrt(sum(x * x for x in force))
            scale = min(float(max_displacement), float(move_scale) * norm_force)
            if norm_force > 1e-14:
                disp = tuple(scale * x / norm_force for x in force)
            else:
                disp = (0.0, 0.0, 0.0)
            frac_force = _mat_vec(inv, disp)
            positions[atom] = tuple(positions[atom][j] + frac_force[j] for j in range(3))
        s = Structure(list(s.species), positions, s.lattice, True, s.periodic, s.selective_dynamics,
                      s.source_format, s.source_path, list(s.warnings)).normalized()
    return {"status": "MAX_STEPS", "steps": max_steps, "structure": s.as_dict(), "history": history,
            "checkpoint": str(checkpoint)}


def _finite_difference_cell_relax(structure: str | Path | Structure, paw_root: str | Path, output: str | Path,
               *, max_steps=10, stress_tolerance=0.05, step_angstrom=1e-3,
               move_scale=0.01, max_strain=0.02, max_workers=1, **kwargs):
    """Variable-cell relaxation from six finite-difference *dimensionless* strains.

    Fractional coordinates are preserved.  A checkpoint contains the input
    signature and is atomically replaced after each converged SCF/derivative
    step, allowing ``sparse-dft resume`` style interruption recovery.
    """
    from .structure import read_structure
    s = read_structure(structure) if isinstance(structure, (str, Path)) else structure
    s = s.normalized(); out = Path(output); out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "cell_relax_checkpoint.json"
    signature = _json_hash({"kind": "cell", "paw_root": str(Path(paw_root).resolve()),
                            "stress_tolerance": stress_tolerance, "step_angstrom": step_angstrom,
                            "move_scale": move_scale, "max_strain": max_strain, "kwargs": kwargs})
    history = []
    start_step = 0
    if checkpoint.exists():
        saved = json.loads(checkpoint.read_text(encoding='utf-8'))
        if saved.get("schema_version") == "sparse-dft-cell-relax-2" and saved.get("signature") != signature:
            raise InputError("cell relaxation checkpoint parameters do not match current request",
                             details={"checkpoint": str(checkpoint), "expected_signature": signature,
                                      "found_signature": saved.get("signature")})
        if saved.get("structure"):
            s = _structure_from_dict(saved["structure"]); history = saved.get("history", [])
            start_step = int(saved.get("step", -1)) + 1
    for step in range(start_step, start_step + max_steps):
        energy, _ = _energy(s, paw_root, out / f"step_{step:04d}", **kwargs)
        stress = [[0.0] * 3 for _ in range(3)]
        jobs = []
        for i in range(3):
            for j in range(i, 3):
                # Symmetric strain epsilon_ij, in Cartesian lattice basis.
                # ``step_angstrom`` is retained as a public compatibility
                # parameter and converted to a dimensionless perturbation
                # relative to the shortest lattice vector.
                h = float(step_angstrom) / max(min(math.sqrt(sum(x*x for x in row)) for row in s.lattice), 1e-8)
                eps = [[0.0] * 3 for _ in range(3)]
                eps[i][j] = h; eps[j][i] = h
                lp = _apply_strain(s.lattice, eps)
                epsm = [[-x for x in row] for row in eps]
                lm = _apply_strain(s.lattice, epsm)
                sp = Structure(list(s.species), list(s.positions), lp, True, s.periodic).normalized()
                sm = Structure(list(s.species), list(s.positions), lm, True, s.periodic).normalized()
                jobs.append((i, j, h, sp, sm))
        def stress_job(item):
            i, j, h, sp, sm = item
            ep, _ = _energy(sp, paw_root, out / f"cell_{step}_{i}_{j}_plus", **kwargs)
            em, _ = _energy(sm, paw_root, out / f"cell_{step}_{i}_{j}_minus", **kwargs)
            # Both eps_ij and eps_ji vary: dE/dh = 2*volume*sigma_ij.
            shear_factor = 2.0 if i != j else 1.0
            return i, j, (ep - em) / (2.0 * h * shear_factor * max(s.volume or 1.0, 1e-12))
        if max_workers and max_workers > 1:
            with ThreadPoolExecutor(max_workers=min(int(max_workers), len(jobs))) as pool:
                results = list(pool.map(stress_job, jobs))
        else:
            results = [stress_job(job) for job in jobs]
        for i, j, value in results: stress[i][j] = stress[j][i] = value
        max_stress = max(abs(x) for row in stress for x in row)
        history.append({"step": step, "energy_ev": energy, "stress_ev_per_angstrom3": stress, "structure": s.as_dict()})
        _atomic_json_write(checkpoint, {"schema_version": "sparse-dft-cell-relax-2", "signature": signature,
                                        "step": step, "history": history, "structure": s.as_dict()})
        if max_stress <= stress_tolerance:
            return {"status": "CONVERGED", "steps": step + 1, "structure": s.as_dict(), "history": history,
                    "checkpoint": str(checkpoint), "analytic": False,
                    "validation": "finite_difference_only"}
        # Stress descent with a per-component strain cap.  The cap prevents a
        # noisy derivative from producing a singular lattice.
        eps = [[0.0] * 3 for _ in range(3)]
        for i in range(3):
            for j in range(3):
                eps[i][j] = max(-max_strain, min(max_strain, -float(move_scale) * stress[i][j]))
        lattice = _apply_strain(s.lattice, eps)
        s = Structure(list(s.species), list(s.positions), lattice, True, s.periodic).normalized()
    return {"status": "MAX_STEPS", "steps": len(history), "structure": s.as_dict(), "history": history,
            "checkpoint": str(checkpoint), "analytic": False}


def ionic_relax(structure, paw_root, output, *, derivative_method='analytic', max_workers=1,
                step_angstrom=1e-3, **kwargs):
    if derivative_method == 'finite_difference':
        return _finite_difference_ionic_relax(structure,paw_root,output,max_workers=max_workers,
                                             step_angstrom=step_angstrom,**kwargs)
    if derivative_method != 'analytic': raise InputError('unknown relaxation derivative method')
    from .analytic_relax import analytic_relax
    return analytic_relax(structure,paw_root,output,**kwargs)


def cell_relax(structure, paw_root, output, *, derivative_method='analytic', max_workers=1,
               step_angstrom=1e-3, **kwargs):
    if derivative_method == 'finite_difference':
        return _finite_difference_cell_relax(structure,paw_root,output,max_workers=max_workers,
                                            step_angstrom=step_angstrom,**kwargs)
    if derivative_method != 'analytic': raise InputError('unknown relaxation derivative method')
    from .analytic_relax import analytic_relax
    return analytic_relax(structure,paw_root,output,cell=True,**kwargs)
