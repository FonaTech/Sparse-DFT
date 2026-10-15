"""Per-component sparsification profiles for the Sparse-DFT SCF.

Each component has its own switch and ratio. Components are of two kinds:

* ``search`` (class E): they only change how the SCF reaches its fixed
  point. The final FP64 eigen certificate and the original density
  thresholds still decide convergence, so the converged result keeps its
  precision. Wall time and operator counts change.
* ``discretization`` (class A): they change the discrete functional
  (density support, XC quadrature). Their error is controlled by the
  a-priori spectrum probe and by the sweep-calibrated rules in
  ``sparsity_rules.json`` (validated per system by tools/sweep_sparsity.py
  against a 16xENCUT reference).

``mode='off'`` sets no environment variable and no density cutoff: the
discretization stays exact (16xENCUT, padding 2). Since v15 the native
defaults include the certificate-protected solver components (block CGS2,
blocks-scope PAW compensation); results agree with the pre-v15 build to
<=2.4e-12 eV. ``SDFT_DETERMINISTIC=1`` (CLI ``--deterministic``) restores
the pre-v15 path bit for bit.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from .errors import InputError, UnsupportedError

MODES = ('off', 'custom', 'auto')
PAW_COMPENSATION = ('full', 'blocks', 'off')
PROJECTOR_SPACE = ('reciprocal', 'real-search')
SEARCH_PRECISION = ('fp64', 'fp32')
RULES_PATH = Path(__file__).with_name('sparsity_rules.json')
# Values whose C++ implementation exists in this build. Anything else is
# refused rather than passed as an environment variable nobody reads.
# S6 real-search (cpp/plane_wave_operator.cpp build_real_space_projectors)
# is search-only: the FP64 reciprocal certificate still decides convergence.
# It is slower than reciprocal projectors on every swept cell up to 64 atoms
# (v15 sweep), so no auto rule selects it. FP32 search has no C++ kernel.
IMPLEMENTED = {'paw_compensation': {'full', 'blocks', 'off'}, 'projector_space': {'reciprocal', 'real-search'},
               'search_precision': {'fp64'}}
# Density factors the native probe reports by default; the probe's spectrum
# ends at PROBE_TOP_FACTOR, so a factor at or above it has no measured tail.
PROBE_FACTORS = (4, 5, 6, 8, 10, 12, 16)
PROBE_TOP_FACTOR = 24.0
DENSITY_FACTOR_RANGE = (4.0, 24.0)
# auto levels: tail budget relative to the rule file (balanced = the fitted
# budget, 2.5e-7). aggressive = the largest measured passing tail (4.25e-7,
# MgO f6) over the balanced budget; conservative = a quarter of balanced.
LEVELS = {'conservative': 0.25, 'balanced': 1.0, 'aggressive': 4.25e-7 / 2.5e-7}
METRICS = ('weighted_compensation_l2', 'shape_hartree')

# Environment variables owned by a profile. A user-set value that disagrees
# with an explicit profile value is an error, never silently overridden.
_ENV_KEYS = ('SDFT_FFT_OVERSAMPLING', 'SDFT_ADAPTIVE_SEARCH', 'SDFT_SCF_SEARCH_FACTOR',
             'SDFT_BLOCK_ORTHOGONALIZE', 'SDFT_PAW_COMPENSATED_ACCUMULATION',
             'SDFT_PAW_COMPENSATION_SCOPE', 'SDFT_PROJECTOR_SPACE',
             'SDFT_PROJECTOR_RADIUS_FACTOR', 'SDFT_SEARCH_PRECISION', 'SDFT_SPARSITY_PROBE_FACTORS')


@dataclass(frozen=True)
class SparsityProfile:
    mode: str = 'off'
    # S1 (discretization): density support as a multiple of ENCUT.
    density_cutoff_factor: float | None = None
    # S2 (discretization): XC quadrature padding per axis (code default 2).
    xc_padding: int | None = None
    # S3 (search): inner eigen tolerance = factor * outer residual.
    eigen_search_factor: float | None = None
    # S4 (search): block CGS2 orthogonalization of appended vectors.
    block_ortho: bool | None = None
    # S5: scope of compensated PAW accumulation.
    paw_compensation: str | None = None
    # S6 (search): real-space projectors during inexact search only.
    projector_space: str | None = None
    projector_radius_factor: float | None = None
    # S7 (search): working precision of inexact search.
    search_precision: str | None = None
    # auto only: how S1 is chosen (see AUTO_CONTROL_KEYS / validate_auto_controls).
    auto_controls: dict = field(default_factory=dict)
    rule_ids: tuple[str, ...] = field(default_factory=tuple)
    estimates: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        out = asdict(self)
        out['rule_ids'] = list(self.rule_ids)
        return out


# level: conservative|balanced|aggressive; tail_budget: absolute S1 budget
# (replaces level and rule budget); candidates: S1 factors to try, all
# reported by the probe; min_factor/max_factor: clip the candidates;
# metric: probe tail metric; rules: path of an alternative rules JSON.
AUTO_CONTROL_KEYS = ('level', 'tail_budget', 'candidates', 'min_factor', 'max_factor', 'metric', 'rules')


def validate_auto_controls(controls: dict) -> dict:
    unknown = sorted(set(controls) - set(AUTO_CONTROL_KEYS))
    if unknown:
        raise InputError('unknown sparsity auto controls', details={'keys': unknown})
    c = {k: v for k, v in controls.items() if v is not None}
    if 'level' in c and c['level'] not in LEVELS:
        raise InputError('sparsity level must be conservative, balanced or aggressive', details={'value': c['level']})
    if 'tail_budget' in c:
        b = float(c['tail_budget'])
        if not math.isfinite(b) or not 0.0 < b <= 1e-3:
            raise InputError('sparsity tail budget must be in (0, 1e-3]', details={'value': b})
        c['tail_budget'] = b
    lo, hi = DENSITY_FACTOR_RANGE
    for key in ('min_factor', 'max_factor'):
        if key in c:
            f = float(c[key])
            if not math.isfinite(f) or not lo <= f <= hi:
                raise InputError(f'sparsity {key} must be in [4, 24]', details={'value': f})
            c[key] = f
    if c.get('min_factor', lo) > c.get('max_factor', hi):
        raise InputError('sparsity min_factor exceeds max_factor', details=c)
    if 'candidates' in c:
        values = sorted({float(x) for x in c['candidates']})
        if not values or any(not math.isfinite(x) or not lo <= x < PROBE_TOP_FACTOR for x in values):
            raise InputError('sparsity candidates must be factors in [4, 24)', details={'value': c['candidates']})
        c['candidates'] = values
    if 'metric' in c and c['metric'] not in METRICS:
        raise InputError('sparsity metric must be weighted_compensation_l2 or shape_hartree',
                         details={'value': c['metric']})
    if 'rules' in c:
        c['rules'] = str(c['rules'])
    return c


def validate_profile(profile: SparsityProfile) -> SparsityProfile:
    if profile.mode not in MODES:
        raise InputError('sparsity mode must be off, custom or auto', details={'value': profile.mode})
    p = profile
    if p.density_cutoff_factor is not None:
        f = float(p.density_cutoff_factor)
        lo, hi = DENSITY_FACTOR_RANGE
        if not math.isfinite(f) or not lo <= f <= hi:
            raise InputError('density_cutoff_factor must be in [4, 24] (the probe spectrum ends at 24xENCUT;'
                             ' use --density-cutoff for a larger absolute support)', details={'value': f})
    if p.xc_padding is not None and p.xc_padding not in (1, 2, 3, 4):
        raise InputError('xc_padding must be 1, 2, 3 or 4', details={'value': p.xc_padding})
    if p.eigen_search_factor is not None:
        f = float(p.eigen_search_factor)
        if not math.isfinite(f) or not 0.0 < f <= 0.1:
            raise InputError('eigen_search_factor must be in (0, 0.1]', details={'value': f})
    if p.paw_compensation is not None and p.paw_compensation not in PAW_COMPENSATION:
        raise InputError('paw_compensation must be full, blocks or off', details={'value': p.paw_compensation})
    if p.projector_space is not None and p.projector_space not in PROJECTOR_SPACE:
        raise InputError('projector_space must be reciprocal or real-search', details={'value': p.projector_space})
    if p.projector_radius_factor is not None:
        f = float(p.projector_radius_factor)
        if not math.isfinite(f) or not 1.0 <= f <= 2.0:
            raise InputError('projector_radius_factor must be in [1, 2]', details={'value': f})
    if p.search_precision is not None and p.search_precision not in SEARCH_PRECISION:
        raise InputError('search_precision must be fp64 or fp32', details={'value': p.search_precision})
    for name, allowed in IMPLEMENTED.items():
        value = getattr(p, name)
        if value is not None and value not in allowed:
            raise UnsupportedError(f'{name}={value} is not implemented in this build',
                                   details={'implemented': sorted(allowed)})
    if p.auto_controls:
        if p.mode != 'auto':
            raise InputError('sparsity level/budget/candidates/metric/rules only apply to --sparsity auto')
        p = replace(p, auto_controls=validate_auto_controls(p.auto_controls))
    if p.mode == 'off' and any(v is not None for v in _component_values(p)):
        raise InputError('sparsity mode off does not accept component settings; use custom')
    return p


def _component_values(p: SparsityProfile):
    return (p.density_cutoff_factor, p.xc_padding, p.eigen_search_factor, p.block_ortho,
            p.paw_compensation, p.projector_space, p.projector_radius_factor, p.search_precision)


def profile_environment(profile: SparsityProfile, base: dict | None = None) -> dict:
    """Return the child-process environment for ``profile``.

    ``off`` returns ``base`` unchanged. Explicit settings that conflict with
    a user-set variable raise instead of being overridden.
    """
    env = dict(os.environ if base is None else base)
    p = validate_profile(profile)
    if p.mode == 'off':
        return env
    wanted: dict[str, str] = {}
    if p.xc_padding is not None:
        wanted['SDFT_FFT_OVERSAMPLING'] = str(int(p.xc_padding))
    if p.eigen_search_factor is not None:
        wanted['SDFT_ADAPTIVE_SEARCH'] = '1'
        wanted['SDFT_SCF_SEARCH_FACTOR'] = f'{float(p.eigen_search_factor):.6g}'
    if p.block_ortho is not None:
        wanted['SDFT_BLOCK_ORTHOGONALIZE'] = '1' if p.block_ortho else '0'
    if p.paw_compensation is not None:
        wanted['SDFT_PAW_COMPENSATED_ACCUMULATION'] = '0' if p.paw_compensation == 'off' else '1'
        wanted['SDFT_PAW_COMPENSATION_SCOPE'] = 'blocks' if p.paw_compensation == 'blocks' else 'full'
    if p.projector_space is not None:
        wanted['SDFT_PROJECTOR_SPACE'] = p.projector_space
    if p.projector_radius_factor is not None:
        wanted['SDFT_PROJECTOR_RADIUS_FACTOR'] = f'{float(p.projector_radius_factor):.6g}'
    if p.search_precision is not None:
        wanted['SDFT_SEARCH_PRECISION'] = p.search_precision
    for key, value in wanted.items():
        if key in env and env[key] != value:
            raise InputError('environment variable conflicts with the sparsity profile',
                             details={'variable': key, 'environment': env[key], 'profile': value})
        env[key] = value
    return env


# Certificate-protected solver knobs: each sets one native variable whose
# parser (file named) enforces the same range. They change the route to the
# fixed point, never the convergence certificate. None = native default.
SOLVER_CONTROLS = {
    # name: (variable, kind, allowed, native default, parser)
    'davidson_subspace_factor': ('SDFT_DAVIDSON_SUBSPACE_FACTOR', int, (4, 16), 8, 'cpp/pw_core.cpp'),
    'search_operator_floor': ('SDFT_SEARCH_OPERATOR_FLOOR', float, (0.0, 1e-2), 1e-6, 'cpp/scf_core.cpp'),
    'ritz_solver': ('SDFT_RITZ_SOLVER', str, ('auto', 'qr', 'divide_conquer', 'subset'), 'auto', 'cpp/pw_core.cpp'),
    'incremental_ritz': ('SDFT_INCREMENTAL_RITZ', bool, None, None, 'cpp/pw_core.cpp'),
    'warm_guard_vectors': ('SDFT_WARM_GUARD_VECTORS', int, (1, None), None, 'cpp/pw_core.cpp (<= min(n, 2*bands))'),
    'fast_density_grid': ('SDFT_FAST_DENSITY_GRID', bool, None, None, 'cpp/density_functionals.cpp'),
    'scf_acceptance_margin': ('SDFT_SCF_ACCEPTANCE_MARGIN', float, (0.0, 1.0), 1.0, 'cpp/scf_core.cpp'),
}


def validate_solver_controls(controls: dict | None) -> dict:
    controls = {k: v for k, v in (controls or {}).items() if v is not None}
    unknown = sorted(set(controls) - set(SOLVER_CONTROLS))
    if unknown:
        raise InputError('unknown solver controls', details={'keys': unknown})
    out = {}
    for name, value in controls.items():
        _, kind, allowed, _, _ = SOLVER_CONTROLS[name]
        if kind is bool:
            if isinstance(value, str):
                if value not in ('0', '1', 'true', 'false'):
                    raise InputError(f'{name} must be 0 or 1', details={'value': value})
                value = value in ('1', 'true')
            out[name] = bool(value)
        elif kind is str:
            if value not in allowed:
                raise InputError(f'{name} must be one of {", ".join(allowed)}', details={'value': value})
            out[name] = value
        else:
            v = kind(value)
            lo, hi = allowed
            if kind is float:
                # (lo, hi]: both float knobs exclude zero and include the top.
                if not math.isfinite(v) or not lo < v <= hi:
                    raise InputError(f'{name} must be in ({lo:g}, {hi:g}]', details={'value': value})
            elif v < lo or (hi is not None and v > hi):
                raise InputError(f'{name} must be in [{lo}, {hi if hi is not None else "inf"}]', details={'value': value})
            out[name] = v
    return out


def solver_environment(controls: dict | None, base: dict | None = None) -> dict:
    env = dict(os.environ if base is None else base)
    for name, value in validate_solver_controls(controls).items():
        variable, kind = SOLVER_CONTROLS[name][:2]
        text = ('1' if value else '0') if kind is bool else (f'{value:.6g}' if kind is float else str(value))
        if variable in env and env[variable] != text:
            raise InputError('environment variable conflicts with a solver control',
                             details={'variable': variable, 'environment': env[variable], 'control': text})
        env[variable] = text
    return env


def resolved_density_cutoff(profile: SparsityProfile, cutoff_ev: float,
                            explicit_density_cutoff_ev: float | None) -> float | None:
    if profile.density_cutoff_factor is None:
        return explicit_density_cutoff_ev
    if explicit_density_cutoff_ev is not None:
        raise InputError('give either density_cutoff_ev or density_cutoff_factor, not both')
    return float(profile.density_cutoff_factor) * float(cutoff_ev)


def load_rules(path: Path = RULES_PATH) -> dict:
    p = Path(path)
    if not p.is_file():
        raise InputError('sparsity rules file not found', details={'path': str(p)})
    try:
        rules = json.loads(p.read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        raise InputError('sparsity rules file is not valid JSON', details={'path': str(p), 'error': str(exc)}) from exc
    if rules.get('schema_version') != 'sparse-dft-sparsity-rules-1':
        raise InputError('unsupported sparsity rules schema', details={'path': str(p)})
    s1 = rules.get('density_cutoff') or {}
    for key in ('id', 'candidates', 'tail_budget', 'fallback_factor'):
        if key not in s1:
            raise InputError('sparsity rules lack density_cutoff.' + key, details={'path': str(p)})
    lo, hi = DENSITY_FACTOR_RANGE
    if not lo <= float(s1['fallback_factor']) <= hi or any(not lo <= float(c) < PROBE_TOP_FACTOR for c in s1['candidates']):
        raise InputError('sparsity rule factors must lie in [4, 24)', details={'path': str(p)})
    return rules


def run_probe(executable: Path, input_path: Path, output_path: Path, candidates=None) -> dict:
    env = probe_environment(candidates) if candidates else None
    proc = subprocess.run([str(executable), 'probe', str(input_path), str(output_path)],
                          text=True, capture_output=True, env=env)
    if proc.returncode != 0 or not output_path.exists():
        raise InputError('density spectrum probe failed',
                         details={'returncode': proc.returncode, 'stderr': proc.stderr})
    return json.loads(output_path.read_text())


def probe_factors(probe: dict) -> list[float]:
    """Factors every species of the probe reports a tail for."""
    sets = [{float(row['factor']) for row in species['tails']} for species in probe['species']]
    return sorted(set.intersection(*sets)) if sets else []


def _species_tail(species: dict, factor: float, key: str) -> float:
    for row in species['tails']:
        if abs(row['factor'] - factor) < 1e-9:
            if key not in row:
                raise InputError('density probe lacks the requested tail metric',
                                 details={'metric': key, 'species': species.get('record_id')})
            return float(row[key])
    # A missing factor is never a zero tail (that would pass every budget).
    raise InputError('density probe has no tail at this factor; add it to the probe factors',
                     details={'factor': factor, 'species': species.get('record_id')})


def _worst_tail(probe: dict, factor: float, key: str) -> float:
    return max(_species_tail(s, factor, key) for s in probe['species'])


def _summed_tail(probe: dict, factor: float, key: str) -> float:
    """Sum over distinct species of an absolute (charge-weighted) tail."""
    return sum(_species_tail(s, factor, key) for s in probe['species'])


def probe_environment(candidates, base: dict | None = None) -> dict:
    """Environment that makes the native probe report exactly ``candidates``."""
    env = dict(os.environ if base is None else base)
    wanted = sorted({float(c) for c in candidates})
    for c in wanted:
        if not 1.0 <= c < PROBE_TOP_FACTOR:
            raise InputError('sparsity candidate factors must be in [1, 24)', details={'factor': c})
    if wanted == [float(f) for f in PROBE_FACTORS]:
        return env
    value = ','.join(f'{c:g}' for c in wanted)
    if 'SDFT_SPARSITY_PROBE_FACTORS' in env and env['SDFT_SPARSITY_PROBE_FACTORS'] != value:
        raise InputError('environment variable conflicts with the sparsity candidates',
                         details={'variable': 'SDFT_SPARSITY_PROBE_FACTORS',
                                  'environment': env['SDFT_SPARSITY_PROBE_FACTORS'], 'profile': value})
    env['SDFT_SPARSITY_PROBE_FACTORS'] = value
    return env


def system_features(probe: dict, *, natoms: int, smearing_ev: float, spin_channels: int,
                    vacuum_fraction: float) -> dict:
    return {
        'natoms': int(natoms),
        'metallic': bool(smearing_ev > 0),
        'spin': int(spin_channels) == 2,
        'vacuum_fraction': float(vacuum_fraction),
        'max_zval': max(float(s['zval']) for s in probe['species']),
        'species': [s['record_id'] for s in probe['species']],
    }


def auto_candidates(controls: dict, rules: dict | None = None) -> list[float]:
    """S1 factors the probe must report for these controls (before rule minima)."""
    controls = validate_auto_controls(controls or {})
    rules = rules or load_rules(controls.get('rules', RULES_PATH))
    s1 = rules['density_cutoff']
    base = controls.get('candidates', s1['candidates'])
    lo, hi = controls.get('min_factor', DENSITY_FACTOR_RANGE[0]), controls.get('max_factor', DENSITY_FACTOR_RANGE[1])
    picked = sorted(float(c) for c in base if lo <= float(c) <= hi)
    if not picked:
        raise InputError('no sparsity candidate factor lies in [min_factor, max_factor]',
                         details={'candidates': list(base), 'min_factor': lo, 'max_factor': hi})
    return picked


def resolve_auto(probe: dict, features: dict, rules: dict | None = None,
                 controls: dict | None = None) -> SparsityProfile:
    """Choose every component from the probe and the calibrated rule table.

    ``controls`` (see AUTO_CONTROL_KEYS) moves S1 along the measured
    accuracy/cost curve; S2-S7 rules are unchanged by them.
    """
    controls = validate_auto_controls(controls or {})
    if rules is None:
        rules = load_rules(controls.get('rules', RULES_PATH))
    ids: list[str] = []
    estimates: dict = {}
    # S1: smallest factor whose worst compensation/core Hartree tail meets
    # the calibrated budget. Larger cells and metals use the stricter budget.
    s1 = rules['density_cutoff']
    budget = s1['tail_budget']
    candidates = auto_candidates(controls, rules)
    minimum = min(candidates)
    for condition in s1.get('stricter_when', []):
        if features.get(condition['feature']) and condition.get('equals', True) == features[condition['feature']]:
            budget = min(budget, condition.get('tail_budget', budget))
            minimum = max(minimum, condition.get('minimum_factor', minimum))
            ids.append(condition['id'])
    level = controls.get('level', 'balanced')
    if 'tail_budget' in controls:
        budget = controls['tail_budget']
        estimates['density_tail_budget_source'] = 'user'
    else:
        budget *= LEVELS[level]
        estimates['density_tail_budget_source'] = f'rules x level {level} ({LEVELS[level]:.4g})'
    metric = controls.get('metric', s1.get('metric', 'shape_hartree'))
    if metric != s1.get('metric', 'shape_hartree') and 'tail_budget' not in controls:
        raise InputError('a non-default sparsity metric needs an explicit tail budget (budgets are metric-specific)',
                         details={'metric': metric, 'rules_metric': s1.get('metric')})
    reported = probe_factors(probe)
    missing = [c for c in candidates if not any(abs(c - r) < 1e-9 for r in reported)]
    if missing:
        raise InputError('sparsity candidates are not in the probe factor list', details={
            'missing': missing, 'probe_factors': reported})
    eligible = [c for c in candidates if c >= minimum]
    if not eligible:
        raise InputError('the sparsity candidate range lies below the rule minimum factor for this system',
                         details={'minimum_factor': minimum, 'candidates': candidates, 'rules': ids})
    # Fallback: the rule file's, unless the user's range excludes it; a
    # fallback means no candidate met the budget and is reported as such.
    factor = float(s1['fallback_factor'])
    if 'max_factor' in controls or 'candidates' in controls:
        factor = min(factor, max(eligible))
    tails = {}
    for f in eligible:
        if metric == 'weighted_compensation_l2':
            tail = _summed_tail(probe, f, 'weighted_compensation_l2')
        else:
            tail = max(_worst_tail(probe, f, 'compensation_hartree'), _worst_tail(probe, f, 'core_hartree'))
        tails[f] = tail
        if tail <= budget:
            factor = f
            estimates['density_tail'] = tail
            break
    else:
        estimates['density_tail'] = None
        estimates['density_fallback'] = True
        estimates['warnings'] = [f'no candidate met the S1 tail budget {budget:.3g}; using factor {factor:g}'
                                 + ('' if factor >= float(s1['fallback_factor']) else
                                    ' (below the rule fallback because of max_factor/candidates)')]
    ids.append(s1['id'])
    estimates['density_tail_budget'] = budget
    estimates['density_tail_metric'] = metric
    estimates['density_tails_tried'] = {f'{f:g}': t for f, t in tails.items()}
    estimates['level'] = level
    chosen = {'density_cutoff_factor': float(factor)}
    # S2-S7: first matching rule per component wins; rules are ordered.
    for component in ('xc_padding', 'eigen_search_factor', 'block_ortho', 'paw_compensation',
                      'projector_space', 'projector_radius_factor', 'search_precision'):
        for rule in rules.get(component, []):
            if _matches(rule.get('when', {}), features, chosen):
                chosen[component] = rule['value']
                ids.append(rule['id'])
                break
    return validate_profile(SparsityProfile(mode='auto', rule_ids=tuple(ids), estimates=estimates,
                                            auto_controls=controls, **chosen))


def _matches(when: dict, features: dict, chosen: dict) -> bool:
    for key, expected in when.items():
        if key.endswith('_max'):
            name = key[:-4]
            value = features.get(name, chosen.get(name))
            if value is None or value > expected:
                return False
        elif key.endswith('_min'):
            name = key[:-4]
            value = features.get(name, chosen.get(name))
            if value is None or value < expected:
                return False
        elif features.get(key, chosen.get(key)) != expected:
            return False
    return True


def with_overrides(profile: SparsityProfile, **overrides) -> SparsityProfile:
    """Apply explicit component values on top of an auto choice."""
    given = {k: v for k, v in overrides.items() if v is not None}
    if not given:
        return profile
    mode = profile.mode if profile.mode != 'off' else 'custom'
    return validate_profile(replace(profile, mode=mode, **given))


def coerce_profile(value) -> SparsityProfile:
    if value is None:
        return SparsityProfile()
    if isinstance(value, SparsityProfile):
        return validate_profile(value)
    if isinstance(value, str):
        return validate_profile(SparsityProfile(mode=value))
    if isinstance(value, dict):
        known = set(SparsityProfile.__dataclass_fields__)
        unknown = sorted(set(value) - known)
        if unknown:
            raise InputError('unknown sparsity settings', details={'keys': unknown})
        data = dict(value)
        data.setdefault('mode', 'custom')
        if 'rule_ids' in data:
            data['rule_ids'] = tuple(data['rule_ids'])
        return validate_profile(SparsityProfile(**data))
    raise InputError('sparsity must be a mode string, a mapping or a SparsityProfile')


def vacuum_fraction(structure) -> float:
    """Largest empty slab fraction along any lattice axis (0 for bulk)."""
    s = structure.normalized()
    worst = 0.0
    for axis in range(3):
        coords = sorted((p[axis] % 1.0) for p in s.positions)
        gaps = [b - a for a, b in zip(coords, coords[1:])] + [coords[0] + 1.0 - coords[-1]]
        length = math.sqrt(sum(x * x for x in s.lattice[axis]))
        # A gap below 5 Angstrom is ordinary bonding geometry, not vacuum.
        gap = max(gaps)
        if gap * length > 5.0:
            worst = max(worst, gap)
    return worst


# Counters whose per-call work is fixed for a given cell, used to express the
# computation each component removes.
_LEDGER_COUNTERS = ('operator_hamiltonian', 'operator_overlap', 'paw_block_projectors',
                    'paw_compensated_factor_product', 'fft_local_block', 'density_functional',
                    'scf_diagonalization', 'scf_total', 'process_compute_total',
                    'paw_compensation_density_batch', 'paw_compensation_adjoint',
                    'real_space_projector_points', 'search_fp32_blocks')


def sparsity_ledger(result: dict, profile: SparsityProfile, performance_path: Path) -> dict:
    """Measured counts for this run; ratios need a baseline run (see tools)."""
    counters = {}
    if Path(performance_path).exists():
        raw = json.loads(Path(performance_path).read_text()).get('counters', {})
        counters = {k: raw[k] for k in _LEDGER_COUNTERS if k in raw}
    cutoff = float(result.get('cutoff_ev') or 0.0)
    density = float(result.get('density_cutoff_ev') or 0.0)
    factor = density / cutoff if cutoff else None
    est = profile.estimates or {}
    return {
        'schema_version': 'sparse-dft-sparsity-ledger-2',
        'mode': profile.mode,
        'density_cutoff_factor': factor,
        # density modes and FFT points scale as (cutoff)^(3/2) at fixed cell: the
        # S1 work relative to the 16x exact discretization, before FFT rounding
        'density_modes_ratio_vs_16x_estimate': (factor / 16.0) ** 1.5 if factor else None,
        'density_reciprocal_modes': result.get('density_reciprocal_modes'),
        'max_xc_grid_points': result.get('max_xc_grid_points'),
        'max_fft_grid_points': result.get('max_fft_grid_points'),
        'plane_waves': result.get('plane_waves'),
        'iterations': result.get('iterations'),
        'components': {k: v for k, v in profile.as_dict().items()
                       if k not in ('mode', 'rule_ids', 'estimates', 'auto_controls') and v is not None},
        'rule_ids': list(profile.rule_ids),
        'auto': ({'level': est.get('level'), 'controls': dict(profile.auto_controls or {}),
                  'tail_metric': est.get('density_tail_metric'), 'tail_budget': est.get('density_tail_budget'),
                  'tail_budget_source': est.get('density_tail_budget_source'),
                  'tail_at_choice': est.get('density_tail'), 'tails_tried': est.get('density_tails_tried'),
                  'fallback': bool(est.get('density_fallback')), 'warnings': est.get('warnings', [])}
                 if profile.mode == 'auto' else None),
        'counters': counters,
    }
