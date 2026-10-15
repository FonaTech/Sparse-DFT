"""Python orchestration for the C++ plane-wave PAW SCF executable.

The wrapper owns provenance, cache preparation, and stable JSON I/O.  It does
not turn a non-converged process into a successful calculation.
"""
from __future__ import annotations

from pathlib import Path
import json
import hashlib
import math
import os
import shutil
import subprocess
import tempfile

from .errors import InputError
from .paw import Paw54JsonArchive, PawSelectionPolicy
from .structure import Structure, read_structure
from .kpoints import uniform_kpoint_mesh, validate_kpoint_path
from .bands import DOS_TAIL_SIGMAS, gaussian_broaden  # noqa: F401  (re-exported)
from .checkpoint import export_frozen_scf, canonical_projector_interpolation
from .spin_model import validate_spin_model
from .parallel import coerce_layout, launch_command, layout_environment
from .sparsity import (PROBE_FACTORS, SparsityProfile, auto_candidates, coerce_profile, profile_environment,
                       resolved_density_cutoff, resolve_auto, run_probe, solver_environment, system_features,
                       vacuum_fraction, validate_solver_controls, with_overrides, sparsity_ledger)
from .settings import (automatic_cutoff, validate_cutoff, validate_density_cutoff, validate_max_iterations,
                       validate_nscf_path, DEFAULT_SCF_MIXING)

BOHR_PER_ANGSTROM = 1.8897261254578281


def _projector_interpolation(value):
    # The earlier spellings vasp_cubic/vasp_spline are accepted aliases.
    value = 'cubic' if value is None else canonical_projector_interpolation(value)
    if value not in ('cubic', 'spline', 'upf_original_radial_quadrature', 'per_site_original_paw'):
        raise InputError('projector_interpolation must be cubic, spline or upf_original_radial_quadrature',
                         details={'value': value})
    return value


def _bohr_per_angstrom(convention):
    if convention not in ('codata', 'vasp_legacy'):
        raise InputError('atomic_units must be codata or vasp_legacy')
    return 1.0 / 0.529177249 if convention == 'vasp_legacy' else BOHR_PER_ANGSTROM


def _scf_sampling(kpoint, points=None, weights=None, grid=None, mode='gamma'):
    if grid is not None:
        if points is not None or weights is not None:
            raise InputError('specify SCF grid or explicit points/weights, not both')
        return uniform_kpoint_mesh(grid, mode=mode)
    points = validate_kpoint_path([kpoint] if points is None else points, task='scf', minimum=1)
    if any(abs(x) > 2.0 for point in points for x in point):
        raise InputError('SCF fractional k-points must be in [-2, 2]')
    try:
        weights = [1.0 / len(points)] * len(points) if weights is None else [float(x) for x in weights]
    except (ValueError, TypeError) as exc:
        raise InputError('SCF weights must be finite positive numbers') from exc
    if len(weights) != len(points) or any(not math.isfinite(x) or x <= 0 for x in weights):
        raise InputError('SCF weights must be finite positive numbers, one per k-point')
    total = sum(weights)
    if not math.isfinite(total):
        raise InputError('SCF weight sum overflow')
    return points, [x / total for x in weights]


def _record_process(output: Path, mode: str, process: subprocess.CompletedProcess) -> None:
    """Retain numerical-kernel diagnostics even when no result JSON is emitted."""
    (output / f"{mode}.stdout.log").write_text(process.stdout or "", encoding="utf-8")
    (output / f"{mode}.stderr.log").write_text(process.stderr or "", encoding="utf-8")
    (output / f"{mode}.process.json").write_text(json.dumps({
        "mode": mode, "returncode": process.returncode,
        "stdout": f"{mode}.stdout.log", "stderr": f"{mode}.stderr.log",
    }, indent=2) + "\n", encoding="utf-8")


def _executable() -> Path:
    """sdft_scf: SDFT_SCF_EXE, bundled bin/<platform>/, build trees (incl.
    multi-config Release/Debug), then PATH; '.exe' is implied on Windows."""
    from .resources import find_executable
    return find_executable("sdft_scf")


def _input_path_token(value) -> str:
    """Solver-input path token. Bare when safe (keeps old binaries working);
    otherwise C++ std::quoted syntax so spaces and non-ASCII names survive."""
    text = str(value)
    if text and not any(c.isspace() or c in '"\\' for c in text):
        return text
    return '"' + text.replace('\\', '\\\\').replace('"', '\\"') + '"'


def _write_input(path: Path, structure: Structure, choices: dict, *, cutoff_ev: float,
                 max_iterations: int, density_tolerance: float, onecenter_density_tolerance: float,
                 mixing: float,
                 bands: int, kpoint=(0.0, 0.0, 0.0), path_kpoints=None,
                 scf_kpoints=None, scf_weights=None, smearing_ev=0.0, atomic_units='codata',
                 density_cutoff_ev=None, basis_reference_lattice_bohr=None,
                 projector_interpolation='cubic', geometry_guess=None, geometry_guess_output=None,
                 spin_channels=1, initial_moments=None, hubbard=None):
    projector_interpolation = _projector_interpolation(projector_interpolation)
    cutoff_ev = validate_cutoff(cutoff_ev)
    if cutoff_ev is None:
        raise InputError("resolved cutoff_ev cannot be null")
    density_cutoff_ev = validate_density_cutoff(density_cutoff_ev, cutoff_ev)
    max_iterations = validate_max_iterations(max_iterations)
    if not math.isfinite(float(smearing_ev)) or smearing_ev < 0:
        raise InputError('smearing_ev must be finite and nonnegative')
    if not math.isfinite(float(density_tolerance)) or not 0.0 < float(density_tolerance) < 1.0:
        raise InputError("density_tolerance must be in (0, 1)", details={"value": density_tolerance})
    if not math.isfinite(float(mixing)) or not 0.0 < float(mixing) <= 1.0:
        raise InputError("mixing must be in (0, 1]", details={"value": mixing})
    if not math.isfinite(float(onecenter_density_tolerance)) or not 0.0 < float(onecenter_density_tolerance) < 1.0:
        raise InputError("onecenter_density_tolerance must be in (0, 1)",
                         details={"value": onecenter_density_tolerance})
    s = structure.normalized()
    if s.lattice is None or not all(s.periodic):
        raise InputError("periodic SCF requires a three-dimensional lattice")
    moments, specs = validate_spin_model(s.species, spin_channels, initial_moments, hubbard,
        {symbol: float(c['zval'] if isinstance(c, dict) else c.zval) for symbol, c in choices.items()})
    lattice = [[x * _bohr_per_angstrom(atomic_units) for x in row] for row in s.lattice]
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("SDFT_SCF_INPUT_V1\n")
        f.write("lattice_bohr " + " ".join(str(x) for row in lattice for x in row) + "\n")
        f.write(f"atoms {s.natoms}\n")
        for symbol, frac in zip(s.species, s.positions):
            candidate = choices[symbol]
            cache = Path(candidate["cache"] if isinstance(candidate, dict) else candidate)
            zval = float(candidate["zval"] if isinstance(candidate, dict) else candidate.zval)
            f.write(f"atom {symbol} {frac[0]:.16g} {frac[1]:.16g} {frac[2]:.16g} {zval:.16g} {_input_path_token(cache)}\n")
        for name, value in (('geometry_guess_input', geometry_guess), ('geometry_guess_output', geometry_guess_output)):
            if value is not None:
                resolved_path = str(Path(value).resolve())
                f.write(f'{name} {_input_path_token(resolved_path)}\n')
        f.write(f'spin_channels {spin_channels}\n')
        if spin_channels == 2:
            f.write(f'initial_moments {s.natoms} ' + ' '.join(f'{x:.17g}' for x in moments) + '\n')
            f.write(f'hubbard_ueff {s.natoms} ' + ' '.join(f"{x['l']} {x['ueff_ev']:.17g}" for x in specs) + '\n')
        f.write(f"cutoff_ev {cutoff_ev:.17g}\nmax_iterations {int(max_iterations)}\n")
        if density_cutoff_ev is not None:
            f.write(f"density_cutoff_ev {density_cutoff_ev:.17g}\n")
        if basis_reference_lattice_bohr is not None:
            f.write('basis_reference_lattice_bohr ' + ' '.join(
                f'{v:.17g}' for row in basis_reference_lattice_bohr for v in row) + '\n')
        f.write(f"density_tolerance {density_tolerance:.16g}\n")
        f.write(f"onecenter_density_tolerance {onecenter_density_tolerance:.16g}\n")
        f.write(f"mixing {mixing:.16g}\n")
        f.write(f"bands {int(bands)}\nkpoint_fractional {kpoint[0]:.16g} {kpoint[1]:.16g} {kpoint[2]:.16g}\n")
        f.write(f"smearing_ev {smearing_ev:.16g}\n")
        f.write(f"atomic_units {atomic_units}\n")
        f.write(f"projector_interpolation {projector_interpolation}\n")
        if scf_kpoints is not None:
            f.write(f"scf_kpoints {len(scf_kpoints)}\n")
            for point, weight in zip(scf_kpoints, scf_weights):
                f.write(' '.join(f'{x:.16g}' for x in (*point, weight)) + '\n')
        if path_kpoints is not None:
            f.write(f"path {len(path_kpoints)}\n")
            for kp in path_kpoints:
                f.write(f"{kp[0]:.16g} {kp[1]:.16g} {kp[2]:.16g}\n")


def _resolve_caches(structure: Structure, paw_root: str | Path, workdir: Path, family="standard"):
    from .automatic_paw import resolve_upf, resolve_source_map
    mapped = resolve_source_map(structure, paw_root, workdir, family)
    if mapped is not None:
        return mapped
    upf = resolve_upf(structure, paw_root, workdir)
    if upf is not None:
        return upf
    archive = Paw54JsonArchive(paw_root)
    choices, report = PawSelectionPolicy(family).resolve(archive, structure.species)
    cache_dir = workdir / "paw_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    resolved = {}
    for symbol, candidate in choices.items():
        dataset_dir = cache_dir / symbol
        from .prepared_paw_cache import prepare
        prepare(archive, candidate.record_id, dataset_dir)
        resolved[symbol] = {"cache": str(Path(dataset_dir) / "numeric.bin"), "zval": candidate.zval,
                            "enmax_ev": candidate.enmax_ev,
                            "record_id": candidate.record_id, "element": symbol}
    return resolved, report, archive


class PreparedPawSession:
    """Reuse parsed immutable PAW files, with full source/cache hash checks.

    No Hamiltonian, density, projection, or operator image is retained here.
    A relaxation still runs a complete new SCF at every trial geometry.
    """
    def __init__(self, structure, paw_root, workdir, family="standard"):
        self.root = Path(paw_root).expanduser().resolve()
        self.family = family
        self.species = tuple(sorted(set(structure.species)))
        self.choices, self.report, self.archive = _resolve_caches(
            structure, self.root, Path(workdir), family)
        self.source_hashes = self.archive.local_hashes()
        self.cache_hashes = {}
        for choice in self.choices.values():
            directory = Path(choice['cache']).parent
            names = ('original.sdftupf',) if choice.get('format') == 'upf-paw' else ('POTCAR', 'PSCTR', 'numeric.bin', 'metadata.json', 'numeric.json')
            for name in names:
                path = directory / name
                if not path.is_file():
                    raise InputError('prepared PAW cache is incomplete', details={'path': str(path)})
                self.cache_hashes[path] = self._sha(path)

    @staticmethod
    def _sha(path):
        with Path(path).open('rb') as stream:
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1048576), b''):
                digest.update(chunk)
            return digest.hexdigest()

    def resolve(self, structure, paw_root, family):
        if (Path(paw_root).expanduser().resolve() != self.root or family != self.family or
                tuple(sorted(set(structure.species))) != self.species):
            raise InputError('prepared PAW session element/source/policy mismatch')
        if self.archive.local_hashes() != self.source_hashes:
            raise InputError('PAW archive changed during the prepared session')
        for path, digest in self.cache_hashes.items():
            if not path.is_file() or self._sha(path) != digest:
                raise InputError('prepared PAW cache changed', details={'path': str(path)})
        # Return copies so downstream provenance edits cannot mutate the session.
        import copy
        return copy.deepcopy(self.choices), copy.deepcopy(self.report), self.archive


def _validated_reference_lattice(value):
    if value is None:
        return None
    try:
        if len(value) != 3 or any(len(row) != 3 for row in value):
            raise ValueError('expected a 3 by 3 lattice')
        lattice = [[float(x) for x in row] for row in value]
        if not all(math.isfinite(x) for row in lattice for x in row):
            raise ValueError('nonfinite reference lattice')
        a, b, c = lattice
        determinant = (a[0]*(b[1]*c[2]-b[2]*c[1])
                       - a[1]*(b[0]*c[2]-b[2]*c[0])
                       + a[2]*(b[0]*c[1]-b[1]*c[0]))
        if not math.isfinite(determinant) or abs(determinant) < 1e-14:
            raise ValueError('singular reference lattice')
        return lattice
    except (ValueError, TypeError, OverflowError) as error:
        raise InputError('basis_reference_lattice_bohr must be a finite nonsingular 3 by 3 lattice') from error


def preview_sparsity(structure: str | Path | Structure, paw_root: str | Path, output: str | Path, *,
                     cutoff_ev=None, accuracy='standard', scf_grid=None, mesh_mode='gamma', smearing_ev=0.0,
                     spin_channels=1, initial_moments=None, atomic_units='codata', family='standard',
                     controls=None) -> dict:
    """Run the density spectrum probe once and report what ``--sparsity auto`` would choose per level.

    No SCF is run. The S1 work ratio is the (factor/16)^(3/2) scaling of density modes and FFT
    points at a fixed cell; measured wall-time savings depend on the system (see the ledger).
    """
    from .sparsity import LEVELS
    s = read_structure(structure) if isinstance(structure, (str, Path)) else structure
    s = s.normalized()
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    choices, _report, _archive = _resolve_caches(s, paw_root, out, family)
    cutoff = validate_cutoff(cutoff_ev)
    cutoff = cutoff if cutoff is not None else automatic_cutoff([c.get('enmax_ev') for c in choices.values()], accuracy)
    points, weights = _scf_sampling((0.0, 0.0, 0.0), None, None, scf_grid, mesh_mode)
    # the probe runs no SCF; iteration and tolerance values only satisfy the input validator
    options = dict(cutoff_ev=cutoff, max_iterations=2, density_tolerance=1e-6, onecenter_density_tolerance=1e-6,
                   mixing=DEFAULT_SCF_MIXING, bands=0, kpoint=(0.0, 0.0, 0.0), scf_kpoints=points,
                   scf_weights=weights, smearing_ev=smearing_ev, atomic_units=atomic_units,
                   projector_interpolation=None, basis_reference_lattice_bohr=None, geometry_guess=None,
                   geometry_guess_output=None, spin_channels=spin_channels, initial_moments=initial_moments,
                   hubbard=None)
    base = dict(controls or {})
    base.pop('level', None)
    probe_input = out / 'probe_input.sdft'
    _write_input(probe_input, s, choices, density_cutoff_ev=None, **options)
    probe = run_probe(_executable(), probe_input, out / 'density_probe.json',
                      candidates=sorted(set(PROBE_FACTORS) | set(auto_candidates(base))))
    features = system_features(probe, natoms=s.natoms, smearing_ev=smearing_ev, spin_channels=spin_channels,
                               vacuum_fraction=vacuum_fraction(s))
    levels, metric = {}, None
    for level in LEVELS:
        chosen = resolve_auto(probe, features, controls={**base, 'level': level})
        factor = chosen.density_cutoff_factor
        metric = chosen.estimates.get('density_tail_metric')
        levels[level] = {'components': {k: v for k, v in chosen.as_dict().items()
                                        if k not in ('mode', 'rule_ids', 'estimates', 'auto_controls') and v is not None},
                         'rule_ids': list(chosen.rule_ids), 'tail_budget': chosen.estimates.get('density_tail_budget'),
                         'tail_at_choice': chosen.estimates.get('density_tail'),
                         'fallback': bool(chosen.estimates.get('density_fallback')),
                         'density_modes_ratio_vs_16x_estimate': (factor / 16.0) ** 1.5 if factor else None}
    return {'schema_version': 'sparse-dft-sparsity-preview-1', 'cutoff_ev': cutoff, 'features': features,
            'tail_metric': metric, 'levels': levels, 'probe': str(out / 'density_probe.json')}


def _resolve_sparsity(sparsity, structure: Structure, choices: dict, out: Path, input_options: dict,
                      *, density_cutoff_ev=None):
    """Return the effective profile and, for auto, the spectrum probe."""
    profile = coerce_profile(sparsity)
    if profile.mode != 'auto':
        return profile, None
    probe_input = out / "probe_input.sdft"
    options = dict(input_options, geometry_guess=None, geometry_guess_output=None)
    _write_input(probe_input, structure, choices, density_cutoff_ev=None, **options)
    controls = dict(profile.auto_controls)
    # The probe reports the default factors plus every requested candidate,
    # so a candidate can never be missing (a missing tail is an error).
    factors = sorted(set(PROBE_FACTORS) | set(auto_candidates(controls)))
    probe = run_probe(_executable(), probe_input, out / "density_probe.json", candidates=factors)
    features = system_features(probe, natoms=structure.natoms, smearing_ev=input_options['smearing_ev'],
                               spin_channels=input_options['spin_channels'],
                               vacuum_fraction=vacuum_fraction(structure))
    chosen = resolve_auto(probe, features, controls=controls)
    explicit = {k: v for k, v in profile.as_dict().items()
                if k not in ('mode', 'rule_ids', 'estimates', 'auto_controls') and v is not None}
    if density_cutoff_ev is not None:
        # An explicit absolute density cutoff always wins over the rule.
        chosen = SparsityProfile(**{**chosen.as_dict(), 'density_cutoff_factor': None,
                                    'rule_ids': tuple(chosen.rule_ids)})
    chosen = with_overrides(chosen, **explicit)
    chosen = SparsityProfile(**{**chosen.as_dict(), 'rule_ids': tuple(chosen.rule_ids),
                                'estimates': {**chosen.estimates, 'features': features,
                                              'overrides': sorted(explicit)}})
    return chosen, probe


def run_scf(structure: str | Path | Structure, paw_root: str | Path, output: str | Path,
            *, cutoff_ev=None, max_iterations=240, density_tolerance=1e-6,
            onecenter_density_tolerance=1e-6, mixing=DEFAULT_SCF_MIXING, bands=0, family="standard", accuracy="standard",
            kpoint=(0.0, 0.0, 0.0), scf_kpoints=None, scf_weights=None,
            scf_grid=None, mesh_mode='gamma', smearing_ev=0.0, atomic_units='codata',
            density_cutoff_ev=None, projector_interpolation='cubic',
            basis_reference_lattice_bohr=None, _paw_session=None,
            geometry_guess=None, save_geometry_guess=False,
            spin_channels=1, initial_moments=None, hubbard=None, sparsity=None, parallel=None,
            solver=None) -> dict:
    solver = validate_solver_controls(solver)
    basis_reference_lattice_bohr = _validated_reference_lattice(basis_reference_lattice_bohr)
    projector_interpolation = _projector_interpolation(projector_interpolation)
    s = read_structure(structure) if isinstance(structure, (str, Path)) else structure
    s = s.normalized()
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    if _paw_session is not None:
        if not isinstance(_paw_session, PreparedPawSession):
            raise InputError('invalid prepared PAW session')
        choices, report, archive = _paw_session.resolve(s, paw_root, family)
    else:
        choices, report, archive = _resolve_caches(s, paw_root, out, family)
    max_iterations = validate_max_iterations(max_iterations)
    requested_cutoff = validate_cutoff(cutoff_ev)
    cutoff_ev = requested_cutoff if requested_cutoff is not None else automatic_cutoff(
        [candidate.get("enmax_ev") for candidate in choices.values()], accuracy)
    kpoint = tuple(float(x) for x in kpoint)
    if len(kpoint) != 3 or not all(math.isfinite(x) and abs(x) <= 2.0 for x in kpoint):
        raise InputError("SCF fractional k-point is outside [-2, 2]", details={"kpoint": kpoint})
    points, weights = _scf_sampling(kpoint, scf_kpoints, scf_weights, scf_grid, mesh_mode)
    input_path = out / "scf_input.sdft"
    result_path = out / "scf.json"
    result_path.unlink(missing_ok=True)
    input_options = dict(cutoff_ev=cutoff_ev, max_iterations=max_iterations,
                 density_tolerance=density_tolerance, onecenter_density_tolerance=onecenter_density_tolerance,
                 mixing=mixing, bands=bands or 0,
                 kpoint=kpoint, scf_kpoints=points, scf_weights=weights, smearing_ev=smearing_ev,
                 atomic_units=atomic_units,
                 projector_interpolation=projector_interpolation,
                 basis_reference_lattice_bohr=basis_reference_lattice_bohr,
                 geometry_guess=geometry_guess,
                 geometry_guess_output=out/"geometry_guess.sdft" if save_geometry_guess else None,
                 spin_channels=spin_channels, initial_moments=initial_moments, hubbard=hubbard)
    profile, probe = _resolve_sparsity(sparsity, s, choices, out, input_options,
                                       density_cutoff_ev=density_cutoff_ev)
    density_cutoff_ev = resolved_density_cutoff(profile, cutoff_ev, density_cutoff_ev)
    _write_input(input_path, s, choices, density_cutoff_ev=density_cutoff_ev, **input_options)
    layout = coerce_layout(parallel)
    proc = subprocess.run(launch_command([str(_executable()), "scf", str(input_path), str(result_path)], layout),
                          text=True, encoding="utf-8", errors="replace", capture_output=True,
                          env=layout_environment(layout, solver_environment(solver, profile_environment(profile))))
    _record_process(out, "scf", proc)
    if not result_path.exists():
        raise InputError("C++ SCF did not produce an output", details={"stdout": proc.stdout, "stderr": proc.stderr, "returncode": proc.returncode})
    native_text = result_path.read_text(encoding="utf-8")
    result = json.loads(native_text)
    native_keys = set(result)
    result.setdefault("metadata", {})["reciprocal_basis"] = [list(row) for row in s.reciprocal_lattice]
    result["metadata"]["reciprocal_basis_includes_2pi"] = True
    result["provenance"] = {"engine": "Sparse-DFT C++", "input": str(input_path), "paw": report,
                            "cutoff_ev": cutoff_ev, "cutoff_source": "user" if requested_cutoff is not None else "automatic_enmax",
                            "accuracy": accuracy, "scf_kpoint_fractional": kpoint,
                            "scf_kpoints_fractional": points, "scf_weights": weights,
                            "scf_grid": scf_grid, "mesh_mode": mesh_mode, "smearing_ev": smearing_ev,
                            "atomic_units": atomic_units,
                            "spin_channels": result.get("spin_channels", spin_channels),
                            "hubbard": hubbard, "initial_moments": initial_moments,
                            "projector_interpolation": projector_interpolation,
                            "density_cutoff_ev": result.get('density_cutoff_ev', density_cutoff_ev),
                            "basis_reference_lattice_bohr": basis_reference_lattice_bohr,
                            "onecenter_density_tolerance": onecenter_density_tolerance,
                            "archive_hashes": archive.local_hashes(), "stdout": proc.stdout, "stderr": proc.stderr,
                            "sparsity": profile.as_dict(),
                            "solver_controls": solver,
                            "parallel": layout.as_dict(),
                            "density_spectrum_probe": str(out / "density_probe.json") if probe is not None else None}
    result["sparsity_ledger"] = sparsity_ledger(result, profile, result_path.with_name(result_path.name + ".performance.json"))
    if save_geometry_guess and result.get('converged'):
        from .prepared_paw_cache import sha
        guess = out/'geometry_guess.sdft'
        if not guess.is_file() or not guess.with_name(guess.name+'.sha256').is_file():
            raise InputError('converged SCF did not write requested geometry guess')
        digest = sha(guess)
        if guess.with_name(guess.name+'.sha256').read_text(encoding="utf-8").strip() != digest:
            raise InputError('geometry guess output hash differs')
        result['geometry_guess'] = {'path':str(guess.resolve()),'sha256':digest,'kind':'initial_guess_only'}
    result["structure_hash"] = s.content_hash()
    from .result_io import write_augmented_result
    write_augmented_result(out / "scf_result.json", native_text, native_keys, result,
                           ('metadata', 'provenance', 'structure_hash', 'geometry_guess', 'sparsity_ledger'))
    if proc.returncode not in (0, 3):
        raise InputError("C++ SCF failed", details={"returncode": proc.returncode, "stderr": proc.stderr, "result": result})
    return result


def run_bands(structure: str | Path | Structure, paw_root: str | Path, output: str | Path,
              kpoints: list[tuple[float, float, float]], *, cutoff_ev=None,
              max_iterations=240, density_tolerance=1e-6, onecenter_density_tolerance=1e-6,
              mixing=DEFAULT_SCF_MIXING, bands=0,
              family="standard", accuracy="standard", scf_kpoint=(0.0, 0.0, 0.0),
              scf_kpoints=None, scf_weights=None, scf_grid=None,
              mesh_mode='gamma', smearing_ev=0.0, atomic_units='codata', scf_checkpoint=None,
              path_labels=None, density_cutoff_ev=None, projector_interpolation=None,
              spin_channels=None, initial_moments=None, hubbard=None, sparsity=None, parallel=None,
              solver=None) -> dict:
    solver = validate_solver_controls(solver)
    profile = coerce_profile(sparsity)
    if profile.mode == 'auto':
        raise InputError('bands/NSCF take an explicit sparsity profile; reuse the profile recorded by the SCF')
    requested_interpolation = projector_interpolation
    projector_interpolation = _projector_interpolation(projector_interpolation)
    kpoints = validate_nscf_path(kpoints)
    if path_labels is not None and (len(path_labels) != len(kpoints) or any(
            label is not None and not isinstance(label,str) for label in path_labels)):
        raise InputError('path_labels must contain a string or null for every executed path point')
    s = read_structure(structure) if isinstance(structure, (str, Path)) else structure
    s = s.normalized()
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    choices, report, archive = _resolve_caches(s, paw_root, out, family)
    max_iterations = validate_max_iterations(max_iterations)
    requested_cutoff = validate_cutoff(cutoff_ev)
    cutoff_ev = requested_cutoff if requested_cutoff is not None else automatic_cutoff(
        [candidate.get("enmax_ev") for candidate in choices.values()], accuracy)
    scf_kpoint = tuple(float(x) for x in scf_kpoint)
    if len(scf_kpoint) != 3 or not all(math.isfinite(x) and abs(x) <= 2.0 for x in scf_kpoint):
        raise InputError("SCF fractional k-point is outside [-2, 2]", details={"kpoint": scf_kpoint})
    points, weights = _scf_sampling(scf_kpoint, scf_kpoints, scf_weights, scf_grid, mesh_mode)
    if scf_checkpoint is None:
        density_cutoff_ev = resolved_density_cutoff(profile, cutoff_ev, density_cutoff_ev)
    elif profile.density_cutoff_factor is not None:
        raise InputError('NSCF inherits the frozen density cutoff; omit density_cutoff_factor')
    input_path = out / "bands_input.sdft"; result_path = out / "bands.json"
    result_path.unlink(missing_ok=True)
    checkpoint_digest = None
    frozen_path = out / 'frozen_scf.sdft'
    if scf_checkpoint is not None:
        frozen, checkpoint_digest = export_frozen_scf(scf_checkpoint, frozen_path, cutoff_ev=cutoff_ev,
                                                     density_cutoff_ev=density_cutoff_ev)
        frozen_spin = frozen.get('spin_channels', 1)
        if spin_channels is not None and spin_channels != frozen_spin:
            raise InputError('NSCF spin model differs from the frozen SCF')
        spin_channels = frozen_spin
        frozen_hubbard = frozen.get('hubbard_specs')
        if not frozen_hubbard and frozen_spin == 1:
            frozen_hubbard = [{'l': -1, 'ueff_ev': 0.0} for _ in s.species]
        if hubbard is not None and validate_spin_model(s.species, spin_channels, None, hubbard)[1] != frozen_hubbard:
            raise InputError('NSCF Hubbard model differs from the frozen SCF')
        hubbard = frozen_hubbard
        frozen_interpolation = _projector_interpolation(frozen.get('projector_interpolation'))
        if requested_interpolation is not None and projector_interpolation != frozen_interpolation:
            raise InputError('NSCF projector interpolation differs from the frozen SCF',
                             details={'requested': projector_interpolation,
                                      'frozen': frozen_interpolation})
        projector_interpolation = frozen_interpolation
        if frozen['atomic_units'] != atomic_units:
            raise InputError('NSCF atomic units differ from the frozen SCF')
        density_cutoff_ev = frozen['density_cutoff_ev']
        if frozen.get('structure_hash') and frozen['structure_hash'] != s.content_hash():
            raise InputError('NSCF structure differs from the frozen SCF')
        frozen_points = [p for p in frozen['scf_kpoints'] if p.get('spin_channel', 0) == 0]
        points = [tuple(point['fractional']) for point in frozen_points]
        weights = [point['weight'] for point in frozen_points]
        for atom in frozen['onecenter_states']:
            symbol = atom['symbol']
            if choices[symbol]['record_id'] != atom['paw_record_id']:
                raise InputError('NSCF PAW selection differs from the frozen SCF')
            choices[symbol]['cache'] = atom['paw_cache']
        (out / 'scf.json').write_text(json.dumps(frozen, indent=2)+'\n', encoding="utf-8")
    _write_input(input_path, s, choices, cutoff_ev=cutoff_ev, max_iterations=max_iterations,
                 density_tolerance=density_tolerance, onecenter_density_tolerance=onecenter_density_tolerance,
                 mixing=mixing, bands=bands or 0,
                 kpoint=scf_kpoint, path_kpoints=kpoints, scf_kpoints=points,
                 scf_weights=weights, smearing_ev=smearing_ev, atomic_units=atomic_units,
                 density_cutoff_ev=density_cutoff_ev,
                 projector_interpolation=projector_interpolation,
                 basis_reference_lattice_bohr=frozen.get('basis_reference_lattice_bohr') if scf_checkpoint else None,
                 spin_channels=1 if spin_channels is None else spin_channels,
                 initial_moments=initial_moments, hubbard=hubbard)
    command = [str(_executable()), 'nscf' if scf_checkpoint is not None else 'bands', str(input_path), str(result_path)]
    if scf_checkpoint is not None:
        command.append(str(frozen_path))
    layout = coerce_layout(parallel)
    proc = subprocess.run(launch_command(command, layout),
                          text=True, encoding="utf-8", errors="replace", capture_output=True,
                          env=layout_environment(layout, solver_environment(solver, profile_environment(profile))))
    _record_process(out, "bands", proc)
    if not result_path.exists():
        raise InputError("C++ bands run did not produce an output", details={"stdout": proc.stdout, "stderr": proc.stderr, "returncode": proc.returncode})
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if path_labels is not None:
        result['labels'] = list(path_labels)
    result.setdefault("metadata", {})["reciprocal_basis"] = [list(row) for row in s.reciprocal_lattice]
    result["metadata"]["reciprocal_basis_includes_2pi"] = True
    result["provenance"] = {"engine": "Sparse-DFT C++", "input": str(input_path), "paw": report,
                            "cutoff_ev": cutoff_ev, "cutoff_source": "user" if requested_cutoff is not None else "automatic_enmax",
                            "accuracy": accuracy, "scf_kpoint_fractional": scf_kpoint,
                            "scf_kpoints_fractional": points, "scf_weights": weights,
                            "scf_grid": scf_grid, "mesh_mode": mesh_mode, "smearing_ev": smearing_ev,
                            "atomic_units": atomic_units,
                            "spin_channels": result.get("spin_channels", spin_channels),
                            "hubbard": hubbard, "initial_moments": initial_moments,
                            "projector_interpolation": projector_interpolation,
                            "density_cutoff_ev": result.get('metadata', {}).get('density_cutoff_ev', density_cutoff_ev),
                            "frozen_scf_source": str(scf_checkpoint) if scf_checkpoint else None,
                            "frozen_scf_sha256": checkpoint_digest,
                            "nscf_kpoint_count": len(kpoints),
                            "onecenter_density_tolerance": onecenter_density_tolerance,
                            "sparsity": profile.as_dict(), "solver_controls": solver,
                            "archive_hashes": archive.local_hashes(), "stdout": proc.stdout, "stderr": proc.stderr}
    result["structure_hash"] = s.content_hash()
    # Keep the public band artifact self-contained.  The C++ executable writes
    # a minimal JSON first; Python adds reciprocal-space provenance needed for
    # physically valid path comparison and plotting.
    (out / "bands.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "bands_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if proc.returncode not in (0, 3):
        raise InputError("C++ bands run failed", details={"returncode": proc.returncode, "stderr": proc.stderr, "result": result})
    return result


def run_dos(structure: str | Path | Structure, paw_root: str | Path, output: str | Path, dos_grid, *,
            dos_mesh_mode='gamma', sigma_ev=0.05, npoints=None, energy_step_ev=0.005, emin_ev=None, emax_ev=None,
            **bands_options) -> dict:
    """NSCF on a full uniform mesh, then Gaussian total, spin and lm-resolved PDOS.

    The mesh carries every point of the zone (no symmetry reduction) with equal
    weights, as the SCF mesh does. The PDOS uses the band output's lm weights
    (VASP LORBIT=11 definition: augmentation-sphere charge), so it is a
    sphere-projected DOS whose sum is below the total by the interstitial part.
    Other options are those of :func:`run_bands` (``scf_checkpoint`` reuses a
    converged SCF; it is strongly recommended).
    """
    from .kpoints import validate_kpoint_grid
    grid = validate_kpoint_grid(dos_grid, task='nscf')
    if dos_mesh_mode not in ('gamma', 'monkhorst-pack'):
        raise InputError('DOS mesh mode must be gamma or monkhorst-pack')
    axes = []
    for n in grid:
        axis = [(i / n if dos_mesh_mode == 'gamma' else (2 * i + 1 - n) / (2 * n)) for i in range(n)]
        axes.append([x - math.floor(x + 0.5) for x in axis])
    points = [(x, y, z) for x in axes[0] for y in axes[1] for z in axes[2]]
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    (out / 'dos.json').unlink(missing_ok=True)
    result = run_bands(structure, paw_root, out, points, **bands_options)
    meta = result.get('metadata') or {}
    if not meta.get('scf_converged'):
        raise InputError('DOS needs a converged SCF; the NSCF band run reports scf_converged=false',
                         details={'bands_json': str(out / 'bands.json')})
    from .bands import load_bands
    # Mark both band files as a uniform mesh: plotting must not read them as a k path.
    for name in ('bands.json', 'bands_result.json'):
        path = out / name
        if path.exists():
            payload = json.loads(path.read_text(encoding='utf-8'))
            payload.setdefault('metadata', {}).update(sampling='uniform_mesh', dos_grid=list(grid),
                                                      dos_mesh_mode=dos_mesh_mode)
            payload['kweights'] = [1.0 / len(points)] * len(points)
            path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    bands = load_bands(out / 'bands.json')
    channels = bands.spin_channels or {'total': bands.eigenvalues}
    degeneracy = 1.0 if bands.spin_channels else 2.0
    weights = [degeneracy / len(points)] * len(points)
    flat = [e for rows in channels.values() for row in rows for e in row]
    tops = [min(row[-1] for row in rows) for rows in channels.values()]
    complete_below = min(tops)
    lo = (min(flat) - 1.0) if emin_ev is None else float(emin_ev)
    hi = (complete_below + 3.0 * sigma_ev) if emax_ev is None else float(emax_ev)
    if not hi > lo:
        raise InputError('DOS energy window is empty', details={'emin_ev': lo, 'emax_ev': hi})
    if npoints is None:
        npoints = int(math.ceil((hi - lo) / float(energy_step_ev))) + 1
    if not 2 <= int(npoints) <= 200001:
        raise InputError('DOS grid needs between 2 and 200001 points')
    npoints = int(npoints)
    energies = [lo + (hi - lo) * i / (npoints - 1) for i in range(npoints)]
    if (hi - lo) / (npoints - 1) > sigma_ev / 3.0:
        raise InputError('DOS energy step is coarser than sigma/3; the broadened DOS would be undersampled',
                         details={'step_ev': (hi - lo) / (npoints - 1), 'sigma_ev': sigma_ev})

    proj = (bands.projections or {})
    lm_orbitals = proj.get('projection_lm_orbitals') or []
    lm = proj.get('projection_lm_weights')
    symbols = proj.get('projection_atom_symbols') or []
    nions = len(symbols)
    spin_dos, pdos_cols = {}, {}
    for name, rows in channels.items():
        tensor = None
        if lm_orbitals and lm is not None and nions:
            nk, nb, no = len(rows), len(rows[0]), len(lm_orbitals)
            if isinstance(lm, dict):
                tensor = [[[v for atom in band for v in atom] for band in k] for k in lm[name]]
            else:
                it = iter(lm)
                tensor = [[[next(it) for _ in range(nions * no)] for _ in range(nb)] for _ in range(nk)]
        curves = gaussian_broaden(energies, rows, weights, sigma_ev, tensor)
        spin_dos[name] = curves[0]
        if tensor is not None:
            pdos_cols[name] = curves[1:]
    pdos = []
    if pdos_cols:
        no = len(lm_orbitals)
        for a in range(nions):
            block = {'atom': a + 1, 'element': symbols[a], 'curves': []}
            for o, orbital in enumerate(lm_orbitals):
                for name, cols in pdos_cols.items():
                    block['curves'].append({'orbital': orbital, 'spin': None if name == 'total' else name,
                                            'values': cols[a * no + o]})
            pdos.append(block)
    total = ({'total_up': spin_dos['up'], 'total_down': spin_dos['down']} if set(spin_dos) == {'up', 'down'}
             else {'total': spin_dos['total']})
    step = energies[1] - energies[0]
    ef = bands.fermi_energy
    # Gapped: count electrons up to midgap. At E_F = VBM half of the top
    # Gaussian lies above the cut (Si 6x6x6, sigma 0.1 eV: 7.9946 at the VBM;
    # Si 3x3x3 at 200 eV, gap 0.63 eV: 7.888). At midgap the VBM tail lost and
    # the CBM tail gained largely cancel.
    # The SCF occupations decide insulator vs metal: on a finite mesh a metal
    # also shows a small eigenvalue gap around E_F.
    count_to, count_at = ef, 'fermi'
    scf_json = out / 'scf.json'
    occ = []
    if scf_json.exists():
        occ = [x for p in json.loads(scf_json.read_text(encoding='utf-8')).get('scf_kpoints') or []
               for x in p.get('occupations') or []]
    full = max(occ, default=0.0)
    insulating = bool(occ) and all(x <= 1e-8 or x >= full * (1 - 1e-8) for x in occ)
    if insulating and bands.vbm is not None and bands.cbm is not None and bands.cbm > bands.vbm:
        count_to, count_at = 0.5 * (bands.vbm + bands.cbm), 'midgap'
    electrons = None
    if count_to is not None:
        electrons = sum(sum(v for x, v in zip(energies, curve) if x <= count_to) * step for curve in spin_dos.values())
    payload = {
        'schema_version': 'sparse-dft-dos-1', 'source': str(out / 'bands.json'),
        'source_sha256': hashlib.sha256((out / 'bands.json').read_bytes()).hexdigest(),
        'fermi_energy_ev': ef, 'emin_ev': lo, 'emax_ev': hi, 'nedos': npoints, 'energies_ev': energies,
        'total': total, 'spin_dos': spin_dos, 'pdos': pdos,
        'metadata': {'format': 'Gaussian DOS from a Sparse-DFT NSCF uniform mesh', 'sigma_ev': sigma_ev,
                     'sampling': 'uniform_mesh', 'grid': list(grid), 'mesh_mode': dos_mesh_mode,
                     'nkpoints': len(points), 'spin_channels': list(channels),
                     'spin_degeneracy': int(degeneracy), 'normalization': 'states/eV/cell',
                     'complete_below_ev': complete_below,
                     'pdos_orbitals': lm_orbitals,
                     'projection_weight_definition': proj.get('projection_lm_weight_definition'),
                     'pdos_note': 'augmentation-sphere projections; their sum is below the total DOS',
                     'integrated_electrons': electrons, 'integrated_to': count_at, 'integrated_to_ev': count_to,
                     'kweights': weights, 'tail_sigmas': DOS_TAIL_SIGMAS,
                     'mesh_convergence_verified': False}}
    (out / 'dos.json').write_text(json.dumps(payload) + '\n', encoding='utf-8')
    return payload
