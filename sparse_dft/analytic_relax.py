"""Analytic PAW relaxation with native L-BFGS and evaluated energy line search.

Optimizer convergence is reported independently of material-wide validation.
No unconverged SCF or incomplete PAW derivative may advance the geometry.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

from .errors import InputError
from .structure import Structure, read_structure
from .scf import _executable, PreparedPawSession
from .resources import find_executable


def _dot(a, b): return sum(x*y for x, y in zip(a, b))
def _norm(a): return math.sqrt(_dot(a, a))
def _sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _optimizer_executable():
    # Resolved like sdft_scf (env override, bundle, build tree, PATH) so the
    # Windows .exe suffix and frozen layouts are honoured.
    try:
        return find_executable('sdft_relax_step')
    except InputError as error:
        raise InputError('native L-BFGS executable is missing; rebuild Sparse-DFT', details=error.details) from error


def _geometry_predictor_setting():
    # Absence uses the same enabled PAW closure default as the native core.
    closure = os.environ.get('SDFT_ENABLE_PAW_CLOSURE_DIAGNOSTIC', '1')
    if closure not in ('0', '1'):
        raise InputError('SDFT_ENABLE_PAW_CLOSURE_DIAGNOSTIC must be 0 or 1')
    setting = os.environ.get('SDFT_GEOMETRY_PREDICTOR', closure)
    if setting not in ('0', '1'):
        raise InputError('SDFT_GEOMETRY_PREDICTOR must be 0 or 1')
    if setting == '1' and closure == '0':
        raise InputError('geometry prediction requires the complete PAW closure')
    return setting


def native_direction(gradient, pairs, initial_scale, max_step):
    executable = _optimizer_executable()
    lines = [f'SDFT_LBFGS_V1 {len(gradient)} {len(pairs)} {initial_scale:.17g} {max_step:.17g}',
             ' '.join(f'{v:.17g}' for v in gradient)]
    for pair in pairs:
        lines += [' '.join(f'{v:.17g}' for v in pair[key]) for key in ('s', 'y')]
    result = subprocess.run([str(executable)], input='\n'.join(lines)+'\n',
                            text=True, capture_output=True, encoding='utf-8', errors='replace')
    if result.returncode:
        raise InputError('native L-BFGS failed', details={'stderr': result.stderr,
                                                       'returncode': result.returncode})
    direction = json.loads(result.stdout)
    if len(direction['step']) != len(gradient) or not all(math.isfinite(x) for x in direction['step']):
        raise InputError('invalid native optimizer step')
    return direction


def _ionic_modes(structure):
    """Orthonormal allowed displacement spans; POSCAR flags refer to Direct axes."""
    modes = []
    for atom in range(structure.natoms):
        flags = structure.selective_dynamics[atom] if structure.selective_dynamics else (True,)*3
        directions = structure.lattice if structure.selective_dynamics else ((1.,0.,0.),(0.,1.,0.),(0.,0.,1.))
        local = []
        for enabled, vector in zip(flags, directions):
            if not enabled: continue
            v = list(vector)
            for u in local:
                d = _dot(v, u); v = [a-d*b for a,b in zip(v,u)]
            length = _norm(v)
            if length < 1e-12: raise InputError('linearly dependent selective-dynamics directions')
            local.append([a/length for a in v])
        modes.extend((atom, u) for u in local)
    return modes


def _cell_deformation(x):
    # Engineering shear coordinates: x=(xx,yy,zz,2yz,2xz,2xy).
    return ((1+x[0],x[5]/2,x[4]/2), (x[5]/2,1+x[1],x[3]/2),
            (x[4]/2,x[3]/2,1+x[2]))


def _cell_gradient(x, stress, volume):
    from .relax import _inv
    # dE/dF = volume * sigma * F^-T. _inv returns F^-T.
    fit = _inv(_cell_deformation(x))
    g = [[volume*sum(stress[i][k]*fit[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    return [g[0][0],g[1][1],g[2][2],(g[1][2]+g[2][1])/2,
            (g[0][2]+g[2][0])/2,(g[0][1]+g[1][0])/2]


def _derivatives(result, natoms, cell):
    if not result.get('converged') or not result.get('onecenter_closure_diagnostic'):
        raise InputError('analytic relaxation requires converged complete PAW closure SCF',
                         details={'converged': result.get('converged'),
                                  'onecenter_closure': result.get('onecenter_closure_diagnostic')})
    key = 'hf_total_stress_ev_per_angstrom3' if cell else 'hf_total_forces_ev_per_angstrom'
    provenance = result.get('stress_provenance' if cell else 'force_provenance', {})
    values = result.get(key)
    if not provenance.get('analytic') or not values or len(values) != (3 if cell else natoms) or any(
            len(row) != 3 or not all(math.isfinite(v) for v in row) for row in values):
        raise InputError('complete finite analytic PAW derivatives are unavailable', details={'field': key})
    return values


def _freeze_sparsity(frozen, result):
    """Record the profile the first SCF actually used, as a fixed custom profile."""
    used = (result.get('provenance') or {}).get('sparsity') or {}
    if used.get('mode') != 'auto':
        return
    from .sparsity import SparsityProfile, coerce_profile
    components = {k: used.get(k) for k in SparsityProfile.__dataclass_fields__
                  if k not in ('mode', 'rule_ids', 'estimates', 'auto_controls') and used.get(k) is not None}
    frozen['profile'] = coerce_profile({'mode': 'custom', **components,
                                        'rule_ids': tuple(used.get('rule_ids') or ()),
                                        'estimates': {**(used.get('estimates') or {}),
                                                      'frozen_from': 'relaxation starting geometry'}})


def analytic_relax(structure, paw_root, output, *, cell=False, max_steps=20,
                   force_tolerance=.02, stress_tolerance=.00031207545,
                   move_scale=.05, max_displacement=.15, max_strain=.02,
                   history_size=8, line_search_steps=16, **kwargs):
    from .relax import _energy, _inv, _mat_vec, _apply_strain, _json_hash, _atomic_json_write, _structure_from_dict
    original = read_structure(structure) if isinstance(structure, (str, Path)) else structure
    original = original.normalized()
    if original.lattice is None or not all(original.periodic):
        raise InputError('analytic periodic relaxation requires a three-dimensional cell')
    for name, value in [('force_tolerance',force_tolerance),('stress_tolerance',stress_tolerance),
                        ('move_scale',move_scale),('max_displacement',max_displacement),('max_strain',max_strain)]:
        if not math.isfinite(value) or value <= 0: raise InputError(f'{name} must be finite and positive')
    if not isinstance(max_steps,int) or max_steps < 1 or not 1 <= history_size <= 64 or not 1 <= line_search_steps <= 64:
        raise InputError('invalid relaxation iteration/history/line search limits')
    if max_strain >= .5: raise InputError('max_strain must be less than 0.5')
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / ('cell_relax_checkpoint.json' if cell else 'relax_checkpoint.json')
    if checkpoint.exists() and json.loads(checkpoint.read_text(encoding='utf-8')).get('schema_version')!='sparse-dft-analytic-relax-3':
        raise InputError('relaxation checkpoint parameters and derivative method do not match current request')
    modes = [] if cell else _ionic_modes(original)
    n = 6 if cell else len(modes)
    base_cart = original.cartesian_positions()
    archive = Path(paw_root)
    source_hashes = {name:_sha(archive/name) for name in ('manifest.json','index.json','potentials.jsonl.xz') if (archive/name).exists()}
    binary_hash = _sha(_executable())
    optimizer_hash = _sha(_optimizer_executable())
    signature = _json_hash({'kind':'cell' if cell else 'ionic', 'initial_structure':original.as_dict(),
        'paw_root':str(archive.resolve()), 'source_hashes':source_hashes,
        'binary_sha256':binary_hash, 'optimizer_sha256':optimizer_hash,
        'force_tolerance':force_tolerance,'stress_tolerance':stress_tolerance,
        'move_scale':move_scale,'max_displacement':max_displacement,'max_strain':max_strain,
        'history_size':history_size,'line_search_steps':line_search_steps,'kwargs':kwargs,
        'geometry_predictor':_geometry_predictor_setting(),
        'kernel_environment':{key:val for key,val in os.environ.items() if key.startswith('SDFT_') and key != 'SDFT_SCF_VALIDATION_OUTPUT'}})

    def make_structure(x):
        if cell:
            f = _cell_deformation(x)
            # A Cholesky-style principal-minor check for symmetric positive F.
            if f[0][0] <= .1 or f[0][0]*f[1][1]-f[0][1]**2 <= .01:
                raise InputError('cell trial deformation is not positive definite')
            det = f[0][0]*(f[1][1]*f[2][2]-f[1][2]**2)-f[0][1]*(f[0][1]*f[2][2]-f[1][2]*f[0][2])+f[0][2]*(f[0][1]*f[1][2]-f[1][1]*f[0][2])
            if det <= .01: raise InputError('cell trial deformation is singular or inverted')
            eps = [[f[i][j]-(1. if i==j else 0.) for j in range(3)] for i in range(3)]
            lattice = _apply_strain(original.lattice,eps); positions = original.positions
        else:
            cart = [list(v) for v in base_cart]
            for value,(atom,u) in zip(x,modes):
                for j in range(3): cart[atom][j] += value*u[j]
            lattice = original.lattice; inv = _inv(lattice)
            positions = [_mat_vec(inv,point) for point in cart]
        return Structure(list(original.species),positions,lattice,True,original.periodic,
                         original.selective_dynamics,original.source_format,original.source_path,
                         list(original.warnings)).normalized()

    reuse_paw = os.environ.get('SDFT_REUSE_PAW_PREPARATION', '0')
    if reuse_paw not in ('0', '1'):
        raise InputError('SDFT_REUSE_PAW_PREPARATION must be 0 or 1')
    paw_session = PreparedPawSession(original, paw_root, out/'shared_paw',
                                    kwargs.get('family', 'standard')) if reuse_paw == '1' else None

    reuse_geometry = _geometry_predictor_setting()
    accepted_guess = None
    # sparsity='auto' is resolved once, on the starting geometry; every later
    # SCF reuses that frozen custom profile so all steps share one
    # discretization (an energy surface that changes basis between line
    # search trials would break the optimizer's energy comparisons).
    frozen_sparsity = {}

    def evaluate(x, directory):
        if _sha(_executable()) != binary_hash or _sha(_optimizer_executable()) != optimizer_hash:
            raise InputError('relaxation binaries changed during the calculation')
        if any(_sha(archive/name) != digest for name,digest in source_hashes.items()):
            raise InputError('PAW archive changed during the calculation')
        s = make_structure(x)
        geometry_options = {}
        if reuse_geometry == '1':
            geometry_options['save_geometry_guess'] = True
            if accepted_guess is not None:
                if _sha(accepted_guess['path']) != accepted_guess['sha256']:
                    raise InputError('accepted geometry initial guess changed')
                geometry_options['geometry_guess'] = accepted_guess['path']
        scf_options = dict(kwargs)
        if frozen_sparsity:
            scf_options['sparsity'] = frozen_sparsity['profile']
        energy,result = _energy(s,paw_root,directory,**scf_options, **geometry_options,
                               **({'_paw_session':paw_session} if paw_session else {}))
        if not frozen_sparsity:
            _freeze_sparsity(frozen_sparsity, result)
        if not math.isfinite(energy): raise InputError('nonfinite SCF energy during relaxation')
        derivative = _derivatives(result,s.natoms,cell)
        if cell:
            gradient = _cell_gradient(x,derivative,s.volume)
            residual = max(abs(v) for row in derivative for v in row)
        else:
            gradient = [-_dot(derivative[atom],u) for atom,u in modes]
            projected = [[0.]*3 for _ in range(s.natoms)]
            for g,(atom,u) in zip(gradient,modes):
                for j in range(3): projected[atom][j] -= g*u[j]
            residual = max((_norm(row) for row in projected),default=0.)
        evidence = Path(directory)/'relax_scf_evidence.json'
        _atomic_json_write(evidence,result)
        return {'coordinates':list(x),'energy_ev':energy,'gradient':gradient,'residual':residual,
                'structure':s.as_dict(),'scf_file':str(evidence),'scf_sha256':_sha(evidence),
                'geometry_guess':result.get('geometry_guess'),
                'geometry_guess_used':result.get('geometry_guess_used',False),
                'scf_iterations':result.get('iterations'),'total_energy_valid':result.get('total_energy_valid',False),
                'derivatives_validated':bool(result.get('stress_provenance' if cell else 'force_provenance',{}).get('validated'))}

    history=[]; pairs=[]; accepted=0; evaluations=0
    if checkpoint.exists():
        saved=json.loads(checkpoint.read_text(encoding='utf-8'))
        if saved.get('schema_version')!='sparse-dft-analytic-relax-3' or saved.get('signature')!=signature:
            raise InputError('relaxation checkpoint parameters, structure, PAW or binaries do not match current request')
        state=saved['accepted_state']; history=saved['history']; pairs=saved['pairs']
        accepted=int(saved['accepted_steps']);evaluations=int(saved['scf_evaluations'])
        if _sha(state['scf_file']) != state['scf_sha256']: raise InputError('relaxation SCF checkpoint hash differs')
        result=json.loads(Path(state['scf_file']).read_text(encoding='utf-8'));_derivatives(result,original.natoms,cell)
        if len(state['coordinates'])!=n: raise InputError('relaxation coordinate dimension differs')
        first=Path(history[0]['scf_file'])
        if _sha(first)!=history[0]['scf_sha256']: raise InputError('relaxation initial SCF hash differs')
        _freeze_sparsity(frozen_sparsity,json.loads(first.read_text(encoding='utf-8')))
    else:
        state=evaluate([0.]*n,out/'initial');evaluations=1
        history=[dict(step=0,**state)]

    accepted_guess = state.get('geometry_guess') if reuse_geometry == '1' else None

    def save(status):
        _atomic_json_write(checkpoint,{'schema_version':'sparse-dft-analytic-relax-3','signature':signature,
            'status':status,'accepted_steps':accepted,'scf_evaluations':evaluations,
            'accepted_state':state,'history':history,'pairs':pairs})

    threshold=stress_tolerance if cell else force_tolerance
    save('RUNNING')
    status='MAX_STEPS'
    for _ in range(max_steps):
        if state['residual'] <= threshold: status='CONVERGED';break
        trust=max_strain if cell else max_displacement
        # Cell scale is supplied in strain/(eV/A^3); the native gradient is eV.
        scale=move_scale/(original.volume if cell else 1.)
        direction=native_direction(state['gradient'],pairs,scale,trust)
        if direction['directional_derivative']>=0: raise InputError('relaxation could not construct a descent direction')
        trial=None
        for attempt in range(line_search_steps):
            alpha=2.**(-attempt)
            x=[a+alpha*b for a,b in zip(state['coordinates'],direction['step'])]
            directory=out/f'step_{accepted+1:04d}'/f'trial_{attempt:02d}'
            candidate=evaluate(x,directory);evaluations+=1
            armijo=state['energy_ev']+1e-4*alpha*direction['directional_derivative']
            audit={'alpha':alpha,'energy_ev':candidate['energy_ev'],'armijo_limit_ev':armijo,
                   'accepted':candidate['energy_ev']<=armijo+1e-12,'direction':direction}
            _atomic_json_write(directory/'line_search.json',audit)
            if audit['accepted']: trial=candidate;break
        if trial is None:
            save('LINE_SEARCH_FAILED')
            raise InputError('PAW energy line search failed; accepted geometry preserved',
                             details={'checkpoint':str(checkpoint),'residual':state['residual'],
                                      'trials':line_search_steps})
        pairs.append({'s':[a-b for a,b in zip(trial['coordinates'],state['coordinates'])],
                      'y':[a-b for a,b in zip(trial['gradient'],state['gradient'])]})
        pairs=pairs[-history_size:];state=trial;accepted+=1
        accepted_guess = state.get('geometry_guess') if reuse_geometry == '1' else None
        history.append(dict(step=accepted,line_search_trials=attempt+1,**state));save('RUNNING')
    if state['residual']<=threshold: status='CONVERGED'
    save(status)
    result={'status':status,'steps':accepted,'scf_evaluations':evaluations,
        'structure':state['structure'],'history':history,'checkpoint':str(checkpoint),
        'analytic':True,'optimizer':'C++ L-BFGS with evaluated Armijo line search',
        'residual':state['residual'],'tolerance':threshold,
        'derivative_units':'eV/angstrom^3' if cell else 'eV/angstrom',
        'total_energy_valid':state['total_energy_valid'], 'production_validated':False,
        'validation':'optimizer convergence; PAW material/cutoff/k-mesh validation remains separate',
        'cell_ions':'fixed_fractional_coordinates' if cell else 'fixed_cell'}
    _atomic_json_write(out/'relax_result.json',result)
    return result
