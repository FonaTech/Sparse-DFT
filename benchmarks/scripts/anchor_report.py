"""Compare Sparse-DFT anchor runs with the same-condition VASP runs (v16 E2).

Gates are the plan's Table 1: energy 1 meV/atom, force 1e-3 eV/A, stress
0.01 GPa, eigenvalues 5 meV in the window, electron count 1e-8 relative.
Moment 0.01 muB and DOS normalised L1 1 % are this plan's own gates.

Window: every band whose maximum over the mesh lies below
E_ref + WINDOW_EV, excluding the top TOP_EXCLUDED computed bands (empty
top bands of both Davidson solvers are converged to a looser criterion).
E_ref is the Fermi level for metals and the VBM otherwise. Eigenvalues are
compared unaligned (both codes share the PAW energy zero).
"""
from __future__ import annotations

import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

GATES = {'energy_ev_per_atom': 1e-3, 'force_ev_per_angstrom': 1e-3, 'stress_gpa': 0.01,
         'eigenvalue_ev': 5e-3, 'electron_relative': 1e-8, 'moment_muB': 0.01, 'dos_l1': 0.01}
GATE_SOURCE = {'energy_ev_per_atom': 'Table 1', 'force_ev_per_angstrom': 'Table 1', 'stress_gpa': 'Table 1',
               'eigenvalue_ev': 'Table 1', 'electron_relative': 'Table 1', 'moment_muB': 'plan (own gate)',
               'dos_l1': 'plan (own gate)'}
WINDOW_EV = 5.0
TOP_EXCLUDED = 2
DOS_SIGMA_EV = 0.1
GPA_PER_EV_A3 = 160.2176634
KBAR_PER_EV_A3 = 1602.176634


def _values(calc, name):
    return [[float(x) for x in row.text.split()] for row in calc.findall(f"varray[@name='{name}']/v")]


def vasp_result(directory: Path) -> dict:
    from sparse_dft.bands import parse_eigenval
    from validate_scf_reference import reference_convergence
    root = ET.parse(directory / 'vasprun.xml').getroot()
    calc = root.findall('calculation')[-1]
    par = lambda name: root.find(f".//i[@name='{name}']")
    bands = parse_eigenval(directory / 'EIGENVAL')
    channels = bands.spin_channels or {'up': bands.eigenvalues}
    occupations = bands.occupations or {}
    mag = None
    oszicar = directory / 'OSZICAR'
    if oszicar.exists():
        hits = re.findall(r'mag=\s*([-+0-9.Ee]+)', oszicar.read_text())
        mag = float(hits[-1]) if hits else None
    outcar = (directory / 'OUTCAR').read_text(errors='replace') if (directory / 'OUTCAR').exists() else ''
    fermi = re.findall(r'E-fermi\s*:\s*([-+0-9.Ee]+)', outcar)
    return {
        'energy_ev': float(calc.find("energy/i[@name='e_fr_energy']").text),
        'forces': _values(calc, 'forces'),
        # VASP kbar, compression positive -> eV/A^3, tension positive (Sparse-DFT sign).
        'stress_ev_a3': [[-v / KBAR_PER_EV_A3 for v in row] for row in _values(calc, 'stress')],
        'kpoints': bands.kpoints, 'weights': bands.kweights, 'channels': channels, 'occupations': occupations,
        'magnetization_muB': mag, 'fermi_ev': float(fermi[-1]) if fermi else None,
        'nelect': float(par('NELECT').text), 'encut': float(par('ENCUT').text), 'nbands': int(par('NBANDS').text),
        'ispin': int(par('ISPIN').text), 'convergence': reference_convergence(root, calc, directory),
        'positions': [[float(x) for x in r.text.split()] for r in
                      root.find("structure[@name='finalpos']").findall("varray[@name='positions']/v")],
        'lattice': [[float(x) for x in r.text.split()] for r in
                    root.find("structure[@name='finalpos']").findall("crystal/varray[@name='basis']/v")],
    }


def sparse_dft_result(path: Path) -> dict:
    d = json.loads(path.read_text())
    keys = sorted({p['spin_channel'] for p in d['scf_kpoints']})
    names = {0: 'up', 1: 'down'} if len(keys) == 2 else {keys[0]: 'up'}
    channels, occupations, kpoints, weights = {}, {}, [], []
    for p in d['scf_kpoints']:
        name = names[p['spin_channel']]
        channels.setdefault(name, []).append(p['eigenvalues_ev'])
        occupations.setdefault(name, []).append(p['occupations'])
        if name == 'up':
            kpoints.append(tuple(p['fractional']))
            weights.append(p['weight'])
    return {'raw': d, 'energy_ev': d['total_energy_ev'], 'forces': d['hf_total_forces_ev_per_angstrom'],
            'stress_ev_a3': d['hf_total_stress_ev_per_angstrom3'], 'kpoints': kpoints, 'weights': weights,
            'channels': channels, 'occupations': occupations, 'magnetization_muB': d.get('magnetic_moment_muB'),
            'fermi_ev': d.get('fermi_energy_ev'), 'electrons': d.get('electron_count'),
            'integrated_electrons': d.get('integrated_electron_count'), 'cutoff_ev': d.get('cutoff_ev'),
            'converged': bool(d.get('converged')), 'closure': bool(d.get('onecenter_closure_diagnostic')),
            'iterations': d.get('iterations'), 'atomic_units': d.get('atomic_units'),
            'projector_interpolation': d.get('projector_interpolation'), 'density_cutoff_ev': d.get('density_cutoff_ev')}


def _map_kpoints(candidate, reference):
    used, mapping = set(), []
    for k in candidate:
        hits = [i for i, r in enumerate(reference) if i not in used and
                all(abs((a - b + .5) % 1 - .5) < 1e-6 for a, b in zip(k, r))]
        if len(hits) != 1:
            return None
        mapping.append(hits[0])
        used.add(hits[0])
    return mapping


def _max_abs(a, b):
    return max(abs(x - y) for ra, rb in zip(a, b) for x, y in zip(ra, rb))


def _reference_energy(ref, metal):
    if metal and ref.get('fermi_ev') is not None:
        return ref['fermi_ev']
    best = -math.inf
    for name, rows in ref['channels'].items():
        occ = ref['occupations'].get(name) or [[1.0] * len(r) for r in rows]
        for row, orow in zip(rows, occ):
            best = max([best] + [e for e, o in zip(row, orow) if o > 0.5 * max(orow)])
    return best


def eigen_window(cand_channels, ref_channels, mapping, e_ref):
    """Max/MAE |dE| over the window bands of every channel and k point."""
    diffs, total, nb_used = [], [], {}
    for name, rrows in ref_channels.items():
        crows = cand_channels.get(name)
        if crows is None:
            return None
        rows = [rrows[i] for i in mapping]
        nb = min(len(crows[0]), len(rows[0]))
        usable = max(0, nb - TOP_EXCLUDED)
        window = [b for b in range(usable) if max(r[b] for r in rows) <= e_ref + WINDOW_EV]
        nb_used[name] = len(window)
        for cr, rr in zip(crows, rows):
            diffs += [cr[b] - rr[b] for b in window]
            total += [cr[b] - rr[b] for b in range(nb)]
    if not diffs:
        return None
    return {'max_ev': max(abs(x) for x in diffs), 'mae_ev': sum(abs(x) for x in diffs) / len(diffs),
            'mean_signed_ev': sum(diffs) / len(diffs), 'all_bands_max_ev': max(abs(x) for x in total),
            'window_bands': nb_used, 'reference_energy_ev': e_ref}


def gaussian_dos(channels, weights, grid, sigma=DOS_SIGMA_EV):
    norm = 1.0 / (sigma * math.sqrt(2 * math.pi))
    out = [0.0] * len(grid)
    wsum = sum(weights)
    for rows in channels.values():
        for row, w in zip(rows, weights):
            for e in row:
                for i, x in enumerate(grid):
                    t = (x - e) / sigma
                    if t * t < 50:
                        out[i] += w / wsum * norm * math.exp(-0.5 * t * t)
    return out


def dos_l1(cand, ref, mapping, e_ref):
    rch = {n: [rows[i] for i in mapping] for n, rows in ref['channels'].items()}
    lo = min(min(r) for rows in rch.values() for r in rows) - 1.0
    hi = e_ref + 3.0
    n = 1201
    grid = [lo + (hi - lo) * i / (n - 1) for i in range(n)]
    w = [ref['weights'][i] for i in mapping] if ref.get('weights') else [1.0] * len(mapping)
    a = gaussian_dos(cand['channels'], w, grid)
    b = gaussian_dos(rch, w, grid)
    step = grid[1] - grid[0]
    l1 = sum(abs(x - y) for x, y in zip(a, b)) * step
    integral = sum(b) * step
    return {'l1_normalised': l1 / integral if integral else None, 'sigma_ev': DOS_SIGMA_EV,
            'window_ev': [lo, hi], 'grid_points': n}


def compare_scf(cand_path: Path, vasp_dir: Path, settings: dict) -> dict:
    c, r = sparse_dft_result(cand_path), vasp_result(vasp_dir)
    natoms = len(settings['species'])
    metal = settings['smearing_ev'] > 0
    mapping = _map_kpoints(c['kpoints'], r['kpoints'])
    conditions = {
        'sparse_dft_converged': c['converged'], 'paw_closure_diagnostic': c['closure'],
        'vasp_converged_by_ediff': bool(r['convergence']['converged']),
        'same_cutoff': abs(c['cutoff_ev'] - r['encut']) < 1e-8,
        'same_kpoints': mapping is not None and len(mapping) == len(r['kpoints']),
        'same_bands': all(len(rows[0]) == r['nbands'] for rows in c['channels'].values()),
        'same_spin': len(c['channels']) == r['ispin'],
        'same_nominal_electrons': abs(float(c['electrons']) - r['nelect']) < 1e-9,
    }
    metrics = {'atomic_units': c['atomic_units'], 'projector_interpolation': c['projector_interpolation'],
               'density_cutoff_ev': c['density_cutoff_ev'], 'sparse_dft_iterations': c['iterations'],
               'vasp_iterations': r['convergence']['electronic_iterations']}
    checks = {}
    if all(conditions.values()):
        e_ref = _reference_energy(r, metal)
        metrics.update({
            'energy_diff_ev_per_atom': (c['energy_ev'] - r['energy_ev']) / natoms,
            'force_max_ev_per_angstrom': _max_abs(c['forces'], r['forces']),
            'force_scale_ev_per_angstrom': max(abs(x) for row in r['forces'] for x in row),
            'stress_max_gpa': GPA_PER_EV_A3 * _max_abs(c['stress_ev_a3'], r['stress_ev_a3']),
            'stress_scale_gpa': GPA_PER_EV_A3 * max(abs(x) for row in r['stress_ev_a3'] for x in row),
            'eigenvalues': eigen_window(c['channels'], r['channels'], mapping, e_ref),
            'electron_relative': abs(c['integrated_electrons'] - r['nelect']) / r['nelect'],
            'dos': dos_l1(c, r, mapping, e_ref),
        })
        if metal:
            metrics['fermi_diff_ev'] = c['fermi_ev'] - r['fermi_ev'] if r['fermi_ev'] is not None else None
        checks = {
            'energy': abs(metrics['energy_diff_ev_per_atom']) <= GATES['energy_ev_per_atom'],
            'forces': metrics['force_max_ev_per_angstrom'] <= GATES['force_ev_per_angstrom'],
            'stress': metrics['stress_max_gpa'] <= GATES['stress_gpa'],
            'eigenvalues': metrics['eigenvalues'] is not None and
            metrics['eigenvalues']['max_ev'] <= GATES['eigenvalue_ev'],
            'electrons': metrics['electron_relative'] <= GATES['electron_relative'],
            'dos': metrics['dos']['l1_normalised'] is not None and metrics['dos']['l1_normalised'] <= GATES['dos_l1'],
        }
        if settings['spin_channels'] == 2:
            metrics['moment_diff_muB'] = c['magnetization_muB'] - r['magnetization_muB']
            metrics['moment_sparse_dft_muB'] = c['magnetization_muB']
            metrics['moment_vasp_muB'] = r['magnetization_muB']
            checks['moment'] = abs(metrics['moment_diff_muB']) <= GATES['moment_muB']
    status = 'PASS' if all(conditions.values()) and checks and all(checks.values()) else (
        'CONDITION_MISMATCH' if not all(conditions.values()) else 'FAIL')
    return {'status': status, 'conditions': conditions, 'checks': checks, 'metrics': metrics,
            'files': {'sparse_dft': str(cand_path), 'vasp': str(vasp_dir / 'vasprun.xml')}}


def compare_bands(cand_path: Path, vasp_dir: Path, e_ref: float) -> dict:
    from sparse_dft.bands import load_bands, parse_eigenval
    c = load_bands(cand_path)
    r = parse_eigenval(vasp_dir / 'EIGENVAL')
    if c.nkpoints != r.nkpoints or any(max(abs(a - b) for a, b in zip(p, q)) > 1e-7
                                       for p, q in zip(c.kpoints, r.kpoints)):
        return {'status': 'CONDITION_MISMATCH', 'reason': 'band path k points differ'}
    cch = c.spin_channels or {'up': c.eigenvalues}
    rch = r.spin_channels or {'up': r.eigenvalues}
    window = eigen_window(cch, rch, list(range(r.nkpoints)), e_ref)
    ok = window is not None and window['max_ev'] <= GATES['eigenvalue_ev']
    return {'status': 'PASS' if ok else 'FAIL', 'kpoints': r.nkpoints, 'eigenvalues': window,
            'files': {'sparse_dft': str(cand_path), 'vasp': str(vasp_dir / 'EIGENVAL')}}


# PROCAR prints weights with 3 decimals, so each entry carries <= 5e-4 rounding.
# Plan gate (own): group-summed |dw| <= 5e-4 * group size + 5e-4.
PROCAR_ROUNDING = 5e-4
DEGENERATE_EV = 2e-3
# Diagnostic ladder for close-pair merging (not a gate).
MERGE_WINDOWS_EV = (5e-3, 1e-2, 2e-2, 5e-2, 0.1)


def compare_projections(cand_path: Path, procar: Path) -> dict:
    """Sparse-DFT AE-sphere lm weights vs VASP LORBIT=11 PROCAR, summed per degenerate group.

    Single-band lm splits inside a degenerate shell depend on the eigenvector
    gauge; the group sum does not. The top TOP_EXCLUDED bands are skipped as
    in the eigenvalue window.
    """
    from sparse_dft.bands import parse_procar
    d = json.loads(cand_path.read_text())
    lm = d.get('projection_lm_orbitals') or []
    if not lm:
        return {'status': 'NOT_AVAILABLE', 'reason': 'Sparse-DFT output has no lm weights'}
    p = parse_procar(procar)
    labels = [{'x2-y2': 'dx2-y2'}.get(n, n) for n in p['projection_orbitals']]
    nat, nk, nb = len(d['projection_atom_symbols']), len(d['kpoints']), len(d['eigenvalues'][0])
    spin = d.get('spin_channel_count') == 2

    def check(window_ev):
        worst_excess, worst, groups, info = -math.inf, 0.0, 0, None
        for name, ref in p['projection_weights'].items():
            energies = p['projection_energies_ev'][name]
            tensor = d['projections']['projection_lm_weights'][name] if spin else None

            def weight(k, b, a, o):
                if spin:
                    return tensor[k][b][a][o]
                return d['projector_lm_weights'][((k * nb + b) * nat + a) * len(lm) + o]
            top = min(nb, p['nbands']) - TOP_EXCLUDED
            for k in range(nk):
                b = 0
                while b < top:
                    g = [b]
                    while g[-1] + 1 < p['nbands'] and abs(energies[k][g[-1] + 1] - energies[k][g[-1]]) < window_ev:
                        g.append(g[-1] + 1)
                    b = g[-1] + 1
                    if g[-1] >= top:
                        continue
                    groups += 1
                    gate = PROCAR_ROUNDING * len(g) + PROCAR_ROUNDING
                    for a in range(nat):
                        for j, label in enumerate(labels):
                            if label not in lm:
                                continue
                            o = lm.index(label)
                            diff = abs(sum(weight(k, x, a, o) for x in g) - sum(ref[k][x][a][j] for x in g))
                            worst = max(worst, diff)
                            if diff - gate > worst_excess:
                                worst_excess = diff - gate
                                info = {'channel': name, 'k': k, 'bands': g, 'atom': a, 'lm': label,
                                        'diff': diff, 'gate': gate,
                                        'gap_to_next_ev': (energies[k][g[-1] + 1] - energies[k][g[-1]]
                                                           if g[-1] + 1 < p['nbands'] else None),
                                        'gap_to_previous_ev': (energies[k][g[0]] - energies[k][g[0] - 1]
                                                               if g[0] > 0 else None)}
        return worst_excess <= 0, groups, worst, info

    ok, groups, worst, info = check(DEGENERATE_EV)
    out = {'status': 'PASS' if ok else 'FAIL', 'groups': groups, 'max_group_diff': worst, 'worst': info,
           'degenerate_window_ev': DEGENERATE_EV, 'definition': d.get('projection_lm_weight_definition'),
           'gate': 'plan (own): PROCAR 3-decimal rounding 5e-4 per band in group + 5e-4'}
    if not ok:
        # Diagnostic only, the status above stays strict: the smallest pair
        # window at which the group sums pass. A band pair split by Delta
        # mixes by ~dV/Delta for an off-diagonal difference dV, so single-band
        # weights of close pairs are ill-conditioned while the pair sum is not.
        out['merge_window_to_pass_ev'] = next((w for w in MERGE_WINDOWS_EV if check(w)[0]), None)
    return out


# Relax gates derived from the Table 1 force/stress gates (plan, own):
# position <= 2e-3 A (force gate / ~0.5 eV/A^2 bond stiffness), lattice
# metric relative <= 2e-4 (stress gate / ~50 GPa bulk modulus, x1 margin).
RELAX_GATES = {'position_angstrom': 2e-3, 'lattice_metric_relative': 2e-4}


def _metric(lattice):
    return [[sum(a * b for a, b in zip(u, v)) for v in lattice] for u in lattice]


def compare_relax(sparse_dft_dir: Path, vasp_dir: Path, task: str) -> dict:
    from sparse_dft.structure import read_poscar
    r = json.loads((sparse_dft_dir / 'relax_result.json').read_text())
    final = r['history'][-1]
    scf = Path(final['scf_file'])
    scf = scf if scf.is_absolute() else sparse_dft_dir / scf
    cand = sparse_dft_result(scf)
    ref = vasp_result(vasp_dir / 'static')
    vs = read_poscar(vasp_dir / ('cell2' if task == 'cell' else 'ionic') / 'CONTCAR').normalized()
    ds = r['structure']
    if not ds.get('fractional'):
        raise ValueError('Sparse-DFT relax structure positions are not fractional')
    lat_d, lat_v = ds['lattice'], [list(v) for v in vs.lattice]
    natoms = len(ds['species'])
    disp = 0.0
    for fd, fv in zip(ds['positions'], [list(x) for x in vs.positions]):
        df = [(a - b) - round(a - b) for a, b in zip(fd, fv)]
        cart = [sum(df[i] * lat_v[i][c] for i in range(3)) for c in range(3)]
        disp = max(disp, math.sqrt(sum(x * x for x in cart)))
    gd, gv = _metric(lat_d), _metric(lat_v)
    metric_rel = max(abs(gd[i][j] - gv[i][j]) for i in range(3) for j in range(3)) / max(gv[i][i] for i in range(3))
    energy = (cand['energy_ev'] - ref['energy_ev']) / natoms
    # Relaxation energy (same composition): the codata offset
    # 9.77e-6*(E - sum EATOM) cancels to 9.77e-6*dE here, so this tests the
    # unit-independent physics of the default tier.
    first_vasp = ET.parse(vasp_dir / task / 'vasprun.xml').getroot().findall('calculation')[0]
    vasp_relax = ref['energy_ev'] - float(first_vasp.find("energy/i[@name='e_fr_energy']").text)
    sparse_dft_relax = float(r['history'][-1]['energy_ev']) - float(r['history'][0]['energy_ev'])
    relax_diff = (sparse_dft_relax - vasp_relax) / natoms
    residual_force = max(math.sqrt(sum(x * x for x in f)) for f in cand['forces'])
    residual_stress = max(abs(x) for row in cand['stress_ev_a3'] for x in row) * GPA_PER_EV_A3
    ok = {'converged': r['status'] == 'CONVERGED',
          'position': disp <= RELAX_GATES['position_angstrom'],
          'lattice': task != 'cell' or metric_rel <= RELAX_GATES['lattice_metric_relative'],
          'energy': abs(energy) <= GATES['energy_ev_per_atom'],
          'relaxation_energy': abs(relax_diff) <= GATES['energy_ev_per_atom']}
    return {'status': 'PASS' if all(ok.values()) else 'FAIL', 'checks': ok,
            'metrics': {'max_displacement_angstrom': disp, 'lattice_metric_relative': metric_rel,
                        'energy_diff_ev_per_atom': energy,
                        'relaxation_energy_sparse_dft_ev': sparse_dft_relax, 'relaxation_energy_vasp_ev': vasp_relax,
                        'relaxation_energy_diff_ev_per_atom': relax_diff,
                        'sparse_dft_residual_force_ev_per_angstrom': residual_force,
                        'sparse_dft_residual_stress_gpa': residual_stress, 'sparse_dft_steps': r['steps']},
            'gates': RELAX_GATES | {'energy_ev_per_atom': GATES['energy_ev_per_atom']}}


def report(args) -> int:
    out = args.out
    manifest = json.loads((out / 'manifest.json').read_text())
    rows = {'schema_version': 'sparse-dft-vasp-anchor-report-1', 'gates': GATES, 'gate_source': GATE_SOURCE,
            'window': {'above_reference_ev': WINDOW_EV, 'top_bands_excluded': TOP_EXCLUDED},
            'dos_sigma_ev': DOS_SIGMA_EV, 'systems': {}}
    for name in manifest['systems']:
        st = json.loads((out / 'systems' / name / 'settings.json').read_text())
        vdir = out / 'vasp' / name / 'scf'
        entry = {'settings': st, 'tiers': {}}
        if not (vdir / 'vasprun.xml').exists():
            entry['status'] = 'VASP_MISSING'
            rows['systems'][name] = entry
            continue
        for tier_dir in sorted((out / 'sparse_dft').glob('*')):
            cand = tier_dir / name / 'scf' / 'scf.json'
            if not cand.exists():
                continue
            try:
                result = compare_scf(cand, vdir, st)
            except Exception as exc:  # report, never hide
                result = {'status': 'ERROR', 'error': f'{type(exc).__name__}: {exc}'}
            bands = tier_dir / name / 'bands' / 'bands.json'
            if bands.exists() and (out / 'vasp' / name / 'bands' / 'EIGENVAL').exists() and \
                    result.get('metrics', {}).get('eigenvalues'):
                e_ref = result['metrics']['eigenvalues']['reference_energy_ev']
                result['bands'] = compare_bands(bands, out / 'vasp' / name / 'bands', e_ref)
                procar = out / 'vasp' / name / 'bands' / 'PROCAR'
                if procar.exists():
                    try:
                        result['projections'] = compare_projections(bands, procar)
                    except Exception as exc:
                        result['projections'] = {'status': 'ERROR', 'error': f'{type(exc).__name__}: {exc}'}
            entry['tiers'][tier_dir.name] = result
        rows['systems'][name] = entry
    rows['relax'] = {}
    for key, info in manifest.get('relax', {}).items():
        for tier_dir in sorted((out / 'sparse_dft').glob('*')):
            d = tier_dir / 'relax' / key
            v = out / 'vasp' / 'relax' / key
            if not (d / 'relax_result.json').exists() or not (v / 'static' / 'vasprun.xml').exists():
                continue
            try:
                res = compare_relax(d, v, info['task'])
            except Exception as exc:
                res = {'status': 'ERROR', 'error': f'{type(exc).__name__}: {exc}'}
            rows['relax'].setdefault(key, {})[tier_dir.name] = res
    path = out / 'report.json'
    path.write_text(json.dumps(rows, indent=1, default=str) + '\n')
    for name, entry in rows['systems'].items():
        for tier, res in entry.get('tiers', {}).items():
            m = res.get('metrics', {})
            ev = m.get('eigenvalues') or {}
            print(f"{name:8s} {tier:14s} {res['status']:18s} dE/atom={m.get('energy_diff_ev_per_atom', float('nan')):+.2e} "
                  f"dF={m.get('force_max_ev_per_angstrom', float('nan')):.1e} dS={m.get('stress_max_gpa', float('nan')):.1e} "
                  f"dEig={ev.get('max_ev', float('nan')):.1e} dos={(m.get('dos') or {}).get('l1_normalised', float('nan')):.1e} "
                  f"dM={m.get('moment_diff_muB', float('nan')):+.1e} bands={res.get('bands', {}).get('status', '-')}"
                  f"/{(res.get('bands', {}).get('eigenvalues') or {}).get('max_ev', float('nan')):.1e}"
                  + ('' if res['status'] != 'CONDITION_MISMATCH' else f" {[k for k, v in res['conditions'].items() if not v]}")
                  + (f" {res.get('error')}" if res['status'] == 'ERROR' else '')
                  + (f" proj={res['projections']['status']}/{res['projections'].get('max_group_diff', float('nan')):.1e}"
                     + (f" (passes with pairs < {res['projections']['merge_window_to_pass_ev'] * 1e3:g} meV merged)"
                        if res['projections'].get('merge_window_to_pass_ev') else '')
                     if res.get('projections') else ''))
    for key, tiers in rows['relax'].items():
        for tier, res in tiers.items():
            m = res.get('metrics', {})
            print(f"{key:8s} {tier:14s} {res['status']:18s} dE/atom={m.get('energy_diff_ev_per_atom', float('nan')):+.2e} "
                  f"dErelax/atom={m.get('relaxation_energy_diff_ev_per_atom', float('nan')):+.1e} "
                  f"(VASP {m.get('relaxation_energy_vasp_ev', float('nan')):+.4f} eV) "
                  f"dx={m.get('max_displacement_angstrom', float('nan')):.1e} dG={m.get('lattice_metric_relative', float('nan')):.1e} "
                  f"steps={m.get('sparse_dft_steps')}" + (f" {res.get('error')}" if res['status'] == 'ERROR' else ''))
    print(str(path))
    return 0
