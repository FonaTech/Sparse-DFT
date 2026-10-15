#!/usr/bin/env python3
"""Same-condition VASP vs Sparse-DFT anchor runs (v16 E2).

    vasp_anchor.py build  --out RUN            # structures, settings, VASP+Sparse-DFT inputs
    vasp_anchor.py vasp   --out RUN [--ranks 6] [--systems Si,MgO]
    vasp_anchor.py sparse_dft   --out RUN --tier default|matched [--threads 10]
    vasp_anchor.py sync-bands --out RUN       # NBANDS := Sparse-DFT solved_bands, archive stale
    vasp_anchor.py report --out RUN

Every system is a nominal test geometry (lattice constants are rounded
textbook values, not sourced reference data), perturbed by seeded random
displacements and +0.5 % strain so forces and stresses are non-zero. Both
codes see the same cell, the same POTCAR bytes (exported from the Sparse-DFT PAW
archive), the same ENCUT, explicit full k mesh (no symmetry), number of
bands, Fermi-Dirac width and initial moments.

Tiers:
* ``default``: Sparse-DFT factory defaults for everything the comparison does not
  pin (atomic units codata, cubic projector interpolation, density
  tolerance and density cutoff defaults, automatic cutoff and bands). VASP runs NLSPLINE=.FALSE.
  (cubic, as Sparse-DFT's default).
* ``matched``: Sparse-DFT ``--atomic-units vasp_legacy`` (VASP's own constants),
  so the residual is the numerical method alone.

VASP runs once per system and task; both tiers compare with it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PAW_ROOT = Path(os.environ.get('SDFT_PAW_ARCHIVE', ROOT.parent / 'PAW Raw Data'))
VASP = Path.home() / '.local/bin/vasp_std'
MPIRUN = Path('/opt/homebrew/opt/open-mpi/bin/mpirun')

# Nominal geometries. a in Angstrom; 'basis' fractional in the given lattice.
FCC = lambda a: [[0, a / 2, a / 2], [a / 2, 0, a / 2], [a / 2, a / 2, 0]]
BCC = lambda a: [[-a / 2, a / 2, a / 2], [a / 2, -a / 2, a / 2], [a / 2, a / 2, -a / 2]]


def _hex(a, c):
    return [[a, 0, 0], [-a / 2, a * math.sqrt(3) / 2, 0], [0, 0, c]]


SYSTEMS = {
    # name: lattice, species, fractional basis, k mesh, kind, extras
    'Si': dict(lattice=FCC(5.43), species=['Si', 'Si'], basis=[[0, 0, 0], [.25, .25, .25]],
               grid=[6, 6, 6], kind='semiconductor'),
    'GaAs': dict(lattice=FCC(5.65), species=['Ga', 'As'], basis=[[0, 0, 0], [.25, .25, .25]],
                 grid=[6, 6, 6], kind='semiconductor'),
    'MgO': dict(lattice=FCC(4.21), species=['Mg', 'O'], basis=[[0, 0, 0], [.5, .5, .5]],
                grid=[6, 6, 6], kind='ionic'),
    'LiF': dict(lattice=FCC(4.03), species=['Li', 'F'], basis=[[0, 0, 0], [.5, .5, .5]],
                grid=[6, 6, 6], kind='ionic'),
    'Al': dict(lattice=FCC(4.05), species=['Al'], basis=[[0, 0, 0]], grid=[10, 10, 10], kind='metal'),
    'Ni': dict(lattice=FCC(3.52), species=['Ni'], basis=[[0, 0, 0]], grid=[10, 10, 10], kind='magnetic_metal',
               moments=[1.0]),
    'Fe': dict(lattice=BCC(2.83), species=['Fe'], basis=[[0, 0, 0]], grid=[10, 10, 10], kind='magnetic_metal',
               moments=[3.0]),
    # Al(001) 5-layer slab, 1x1 in-plane (a/sqrt2), 12 A vacuum along c.
    'Al_slab': dict(lattice=[[2.8638, 0, 0], [0, 2.8638, 0], [0, 0, 4 * 2.025 + 12.0]], species=['Al'] * 5,
                    basis=[[(i % 2) * .5, (i % 2) * .5, i * 2.025 / (4 * 2.025 + 12.0)] for i in range(5)],
                    grid=[8, 8, 1], kind='slab_metal'),
    # MoS2 2H monolayer, 15 A vacuum along c; S-S height 3.13 A.
    'MoS2': dict(lattice=_hex(3.19, 18.0), species=['Mo', 'S', 'S'],
                 basis=[[1 / 3, 2 / 3, .5], [2 / 3, 1 / 3, .5 + 1.565 / 18.0], [2 / 3, 1 / 3, .5 - 1.565 / 18.0]],
                 grid=[6, 6, 1], kind='monolayer'),
    # CO in a 10 A box, Gamma only; bond 1.13 A.
    'CO': dict(lattice=[[10., 0, 0], [0, 10., 0], [0, 0, 10.]], species=['C', 'O'],
               basis=[[.5, .5, .5 - .0565], [.5, .5, .5 + .0565]], grid=[1, 1, 1], kind='molecule'),
}
DISPLACEMENT_ANGSTROM = 0.03
STRAIN = 0.005
SEED = 20261015


def sha(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=1, ensure_ascii=False, default=str) + '\n', encoding='utf-8')


def perturbed(name: str, spec: dict) -> dict:
    """Seeded displacement of every atom plus a symmetric +0.5 % strain."""
    rng = random.Random(f'{SEED}:{name}')
    lat = [[float(x) for x in row] for row in spec['lattice']]
    # Symmetric strain with off-diagonal shear so stress has all components.
    eps = [[STRAIN, .4 * STRAIN, .2 * STRAIN], [.4 * STRAIN, -.5 * STRAIN, .3 * STRAIN],
           [.2 * STRAIN, .3 * STRAIN, .8 * STRAIN]]
    if spec['kind'] in ('slab_metal', 'monolayer', 'molecule'):
        # Do not strain the vacuum axis; in-plane only (molecule: none).
        eps = [[0.0] * 3 for _ in range(3)] if spec['kind'] == 'molecule' else \
            [[STRAIN, .4 * STRAIN, 0], [.4 * STRAIN, -.5 * STRAIN, 0], [0, 0, 0]]
    strained = [[sum(lat[i][k] * ((1.0 if k == j else 0.0) + eps[k][j]) for k in range(3)) for j in range(3)]
                for i in range(3)]
    from sparse_dft.structure import _inverse
    inv = _inverse(strained)
    frac = []
    for p in spec['basis']:
        cart = [sum(p[k] * strained[k][j] for k in range(3)) for j in range(3)]
        d = [rng.uniform(-1, 1) * DISPLACEMENT_ANGSTROM for _ in range(3)]
        cart = [c + x for c, x in zip(cart, d)]
        frac.append([sum(inv[j][k] * cart[k] for k in range(3)) % 1.0 for j in range(3)])
    return {'lattice': strained, 'positions': frac, 'species': list(spec['species'])}


def poscar_text(cell: dict, title: str) -> str:
    groups = []
    for s in cell['species']:
        if groups and groups[-1][0] == s:
            groups[-1][1] += 1
        elif s in [g[0] for g in groups]:
            raise ValueError('species must be grouped')
        else:
            groups.append([s, 1])
    rows = [title, '1.0'] + [' '.join(f'{v:.16f}' for v in r) for r in cell['lattice']]
    rows += [' '.join(g[0] for g in groups), ' '.join(str(g[1]) for g in groups), 'Direct']
    rows += [' '.join(f'{v:.16f}' for v in p) for p in cell['positions']]
    return '\n'.join(rows) + '\n'


METAL_KINDS = ('metal', 'magnetic_metal', 'slab_metal')
# Fermi-Dirac kBT for metals in both codes. 0.1 eV is the width used by the
# v15 sparsity sweeps; the E2 report records it as a chosen condition.
METAL_KT_EV = 0.1
TIERS = {
    'default': [],
    'matched': ['--atomic-units', 'vasp_legacy'],
    'auto': ['--sparsity', 'auto'],
    'tight': ['--density-tolerance', '1e-8', '--onecenter-density-tolerance', '1e-8'],
    'matched_tight': ['--atomic-units', 'vasp_legacy', '--density-tolerance', '1e-8',
                      '--onecenter-density-tolerance', '1e-8'],
}
# Tiers that also run the band path (NSCF) and both relaxations.
FULL_TASK_TIERS = ('default', 'matched', 'auto')


def system_settings(name: str, spec: dict) -> dict:
    """Sparse-DFT defaults for cutoff and bands; mesh, kT and moments pinned for both codes."""
    from sparse_dft.paw import Paw54JsonArchive, PawSelectionPolicy
    from sparse_dft.settings import automatic_cutoff
    from sparse_dft.geometry import element
    archive = Paw54JsonArchive(PAW_ROOT)
    elements = list(dict.fromkeys(spec['species']))
    choices, _ = PawSelectionPolicy('standard').resolve(archive, elements)
    cutoff = automatic_cutoff([choices[e].enmax_ev for e in elements], 'standard')
    electrons = sum(choices[s].zval for s in spec['species'])
    spin = 'moments' in spec or spec['kind'] == 'magnetic_metal'
    moments = None
    if spin:
        moments = [float(element(s)['free_atom_moment_muB'] or 0.0) for s in spec['species']]
    # sparse_dft/cpp/scf_core.cpp default: occupied + 4, polarized ceil((N+|m|)/2) + 8.
    if spin:
        bands = math.ceil((electrons + sum(abs(m) for m in moments)) / 2.0) + 8
    else:
        bands = math.ceil(electrons / 2.0) + 4
    return {
        'name': name, 'kind': spec['kind'], 'species': spec['species'], 'elements': elements,
        'paw_records': {e: choices[e].record_id for e in elements},
        'paw_titles': {e: choices[e].title for e in elements},
        'enmax_ev': {e: choices[e].enmax_ev for e in elements},
        'cutoff_ev': cutoff, 'cutoff_rule': 'sparse_dft.settings.automatic_cutoff(standard) = max(300, 1.30*ENMAX)',
        'electrons': electrons, 'bands': bands, 'bands_rule': 'cpp/scf_core.cpp:2534 default',
        'grid': spec['grid'], 'smearing_ev': METAL_KT_EV if spec['kind'] in METAL_KINDS else 0.0,
        'spin_channels': 2 if spin else 1, 'initial_moments': moments,
        'moments_rule': 'free-atom moment from sparse_dft/data/elements.json' if spin else None,
        'lmaxmix': 4 if any(element(e)['block'] == 'd' for e in elements) else 2,
    }


def export_potcar(settings: dict, dest: Path) -> list[str]:
    from sparse_dft.paw import Paw54JsonArchive
    archive = Paw54JsonArchive(PAW_ROOT)
    parts = []
    for e in settings['elements']:
        tmp = dest.with_name(f'.POTCAR.{e}')
        archive.restore(settings['paw_records'][e], 'POTCAR', tmp)
        parts.append(tmp.read_bytes())
        tmp.unlink()
    dest.write_bytes(b''.join(parts))
    return [hashlib.sha256(p).hexdigest() for p in parts]


def kpoints_text(points, weights, title='explicit') -> str:
    rows = [title, str(len(points)), 'Reciprocal']
    rows += [' '.join(f'{x:.16f}' for x in k) + f' {w:.16f}' for k, w in zip(points, weights)]
    return '\n'.join(rows) + '\n'


def nscf_vasp_bands(settings: dict) -> int:
    return max(2 * settings['bands'], settings['bands'] + 8)


def incar(settings: dict, task: str, ranks: int, nk: int) -> dict:
    metal = settings['smearing_ev'] > 0
    kpar = ranks if nk >= ranks else 1
    tags = {
        'SYSTEM': f"anchor {settings['name']} {task}", 'PREC': 'Accurate', 'ENCUT': settings['cutoff_ev'],
        'EDIFF': '1E-9', 'NELM': 300, 'ALGO': 'Normal', 'ISMEAR': -1,
        'SIGMA': settings['smearing_ev'] if metal else 0.001,
        'ISPIN': settings['spin_channels'], 'NBANDS': settings['bands'], 'LASPH': '.TRUE.',
        'LREAL': '.FALSE.', 'ISYM': -1, 'NLSPLINE': '.FALSE.', 'LMAXMIX': settings['lmaxmix'],
        # One rank per k group and NPAR=1, so VASP never rounds NBANDS up.
        'KPAR': kpar, 'NCORE': ranks // kpar, 'LORBIT': 11, 'NEDOS': 3001,
        'LWAVE': '.FALSE.', 'LCHARG': '.TRUE.' if task in ('scf',) else '.FALSE.',
    }
    if settings['spin_channels'] == 2:
        tags['MAGMOM'] = ' '.join(f'{m:g}' for m in settings['initial_moments'])
    if task in ('ionic', 'cell', 'cell2') or settings.get('relax_static'):
        # A relaxed cell carries ~1e-6 lattice noise; VASP's Bravais check
        # (applied even with ISYM=-1) then aborts. Tighten its tolerance.
        tags['SYMPREC'] = '1E-8'
    if task == 'bands':
        # VASP's ICHARG=11 Davidson leaves the top bands of a tight NBANDS
        # under-converged (LiF, k 129, band 6 Li px weight: 0.782 at NBANDS 8,
        # 0.792 at NBANDS 16, Sparse-DFT 0.7921). Solve extra bands; the report still
        # compares min(NBANDS) - TOP_EXCLUDED bands, i.e. the Sparse-DFT band window.
        tags.update(ICHARG=11, LCHARG='.FALSE.', NBANDS=nscf_vasp_bands(settings))
    elif task == 'ionic':
        tags.update(IBRION=2, ISIF=2, NSW=120, POTIM=0.3, EDIFFG='-1E-3', LCHARG='.FALSE.', LORBIT=0)
    elif task in ('cell', 'cell2'):
        # ISIF=6: cell shape and volume with fixed fractional ions (as Sparse-DFT
        # cell relax). Two passes reduce VASP's fixed-basis Pulay stress.
        tags.update(IBRION=2, ISIF=6, NSW=80, POTIM=0.3, EDIFFG='1E-8', LCHARG='.FALSE.', LORBIT=0)
    return tags


def write_vasp_inputs(directory: Path, settings: dict, cell: dict, task: str, points, weights, ranks: int):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'POSCAR').write_text(poscar_text(cell, f"anchor {settings['name']}"))
    hashes = export_potcar(settings, directory / 'POTCAR')
    (directory / 'KPOINTS').write_text(kpoints_text(points, weights))
    tags = incar(settings, task, ranks, len(points))
    (directory / 'INCAR').write_text(''.join(f'{k} = {v}\n' for k, v in tags.items()))
    return {'incar': tags, 'potcar_sha256': hashes}


# Relaxation anchors: (system, VASP task). Si: ions only, in the strained
# perturbed cell. MgO: cell only, ideal rock-salt sites in the strained cell.
RELAX = {'Si_ionic': ('Si', 'ionic'), 'MgO_cell': ('MgO', 'cell')}
FORCE_TOL = 1e-3            # eV/A, both codes
STRESS_TOL_GPA = 0.01       # both codes
GPA_PER_EV_A3 = 160.2176634


def _cell_for(name: str, spec: dict, which: str) -> dict:
    if which == 'ideal':
        return {'lattice': spec['lattice'], 'positions': spec['basis'], 'species': list(spec['species'])}
    cell = perturbed(name, spec)
    if which == 'strained_ideal_sites':
        cell['positions'] = [list(p) for p in spec['basis']]
    return cell


def build(args):
    from sparse_dft.kpoints import uniform_kpoint_mesh
    from sparse_dft.structure import Structure
    from sparse_dft import symmetry
    out = args.out
    manifest = {'schema_version': 'sparse-dft-vasp-anchor-1', 'seed': SEED, 'displacement_angstrom': DISPLACEMENT_ANGSTROM,
                'strain': STRAIN, 'metal_kT_ev': METAL_KT_EV, 'paw_root': str(PAW_ROOT), 'systems': {},
                'relax': {}, 'note': 'nominal test geometries, not relaxed or sourced structures'}
    for name, spec in SYSTEMS.items():
        if args.systems and name not in args.systems:
            continue
        d = out / 'systems' / name
        d.mkdir(parents=True, exist_ok=True)
        st = system_settings(name, spec)
        cell = _cell_for(name, spec, 'perturbed')
        (d / 'POSCAR').write_text(poscar_text(cell, f'anchor {name} perturbed'))
        (d / 'POSCAR.ideal').write_text(poscar_text(_cell_for(name, spec, 'ideal'), f'anchor {name} ideal'))
        points, weights = uniform_kpoint_mesh(spec['grid'], mode='gamma')
        if spec['kind'] != 'molecule':
            ideal = Structure(list(spec['species']), [tuple(p) for p in spec['basis']],
                              tuple(tuple(float(x) for x in r) for r in spec['lattice']), True, (True, True, True))
            periodic = (True, True, False) if spec['kind'] in ('slab_metal', 'monolayer') else None
            path = symmetry.high_symmetry_path(ideal, spacing=0.03, periodic_axes=periodic)
            write_json(d / 'band_path.json', path)
            st['band_path_points'] = len(path['kpoints'])
            st['band_path_segments'] = path['segments']
        write_json(d / 'settings.json', st)
        write_json(d / 'mesh.json', {'grid': spec['grid'], 'mode': 'gamma', 'kpoints': points, 'weights': weights})
        manifest['systems'][name] = {'settings': st, 'poscar_sha256': sha(d / 'POSCAR')}
    for key, (system, task) in RELAX.items():
        if args.systems and system not in args.systems:
            continue
        spec = SYSTEMS[system]
        cell = _cell_for(system, spec, 'perturbed' if task == 'ionic' else 'strained_ideal_sites')
        d = out / 'relax' / key
        d.mkdir(parents=True, exist_ok=True)
        (d / 'POSCAR').write_text(poscar_text(cell, f'anchor {key} start'))
        manifest['relax'][key] = {'system': system, 'task': task, 'poscar_sha256': sha(d / 'POSCAR')}
    write_json(out / 'manifest.json', manifest)
    print(json.dumps({'systems': list(manifest['systems']), 'relax': list(manifest['relax'])}))


def _vasp_env(ranks: int) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith('SDFT_')}
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1',
               VASP_MPI_RANKS=str(ranks), PRTE_MCA_hwloc_default_binding_policy='none')
    return env


def run_vasp_stage(directory: Path, ranks: int) -> dict:
    done = directory / 'done.json'
    if done.exists():
        return json.loads(done.read_text())
    from validate_scf_reference import reference_convergence
    import xml.etree.ElementTree as ET
    start = time.perf_counter()
    with open(directory / 'stdout.log', 'w') as o, open(directory / 'stderr.log', 'w') as e:
        code = subprocess.call([str(Path.home() / '.local/bin/vasp')], cwd=directory, env=_vasp_env(ranks),
                               stdout=o, stderr=e)
    record = {'exit_code': code, 'seconds': time.perf_counter() - start, 'ranks': ranks,
              'binary_sha256': sha(VASP.resolve())}
    if code == 0 and (directory / 'vasprun.xml').exists():
        root = ET.parse(directory / 'vasprun.xml').getroot()
        calcs = root.findall('calculation')
        status = [reference_convergence(root, c, directory) for c in calcs]
        record.update(ionic_steps=len(calcs), converged=bool(calcs) and all(s['converged'] for s in status),
                      electronic_iterations=sum(s['electronic_iterations'] for s in status))
    else:
        record['converged'] = False
    write_json(done, record)
    return record


def vasp(args):
    from sparse_dft.kpoints import uniform_kpoint_mesh
    out, ranks = args.out, args.ranks
    manifest = json.loads((out / 'manifest.json').read_text())
    log = []
    for name in manifest['systems']:
        if args.systems and name not in args.systems:
            continue
        st = json.loads((out / 'systems' / name / 'settings.json').read_text())
        cell = _read_cell(out / 'systems' / name / 'POSCAR')
        mesh = json.loads((out / 'systems' / name / 'mesh.json').read_text())
        scf = out / 'vasp' / name / 'scf'
        if not (scf / 'done.json').exists():
            shutil.rmtree(scf, ignore_errors=True)
            write_vasp_inputs(scf, st, cell, 'scf', mesh['kpoints'], mesh['weights'], ranks)
        log.append((name, 'scf', run_vasp_stage(scf, ranks)))
        path_file = out / 'systems' / name / 'band_path.json'
        if path_file.exists() and log[-1][2].get('converged'):
            path = json.loads(path_file.read_text())
            bands = out / 'vasp' / name / 'bands'
            if not (bands / 'done.json').exists():
                shutil.rmtree(bands, ignore_errors=True)
                n = len(path['kpoints'])
                write_vasp_inputs(bands, st, cell, 'bands', path['kpoints'], [1.0 / n] * n, ranks)
                shutil.copyfile(scf / 'CHGCAR', bands / 'CHGCAR')
            log.append((name, 'bands', run_vasp_stage(bands, ranks)))
        print(json.dumps({'system': name, 'stages': [(t, r.get('converged'), round(r.get('seconds', 0), 1))
                                                     for n, t, r in log if n == name]}), flush=True)
    for key, info in manifest['relax'].items():
        if args.systems and info['system'] not in args.systems:
            continue
        st = json.loads((out / 'systems' / info['system'] / 'settings.json').read_text())
        mesh = json.loads((out / 'systems' / info['system'] / 'mesh.json').read_text())
        cell = _read_cell(out / 'relax' / key / 'POSCAR')
        stages = [info['task']] + (['cell2', 'static'] if info['task'] == 'cell' else ['static'])
        for task in stages:
            d = out / 'vasp' / 'relax' / key / task
            if not (d / 'done.json').exists():
                shutil.rmtree(d, ignore_errors=True)
                write_vasp_inputs(d, st | {'relax_static': True}, cell, 'scf' if task == 'static' else task, mesh['kpoints'],
                                  mesh['weights'], ranks)
            r = run_vasp_stage(d, ranks)
            print(json.dumps({'relax': key, 'stage': task, 'converged': r.get('converged'),
                              'seconds': round(r.get('seconds', 0), 1)}), flush=True)
            if not r.get('converged') or not (d / 'CONTCAR').exists():
                break
            cell = _read_cell(d / 'CONTCAR')


def _read_cell(path: Path) -> dict:
    from sparse_dft.structure import read_poscar
    s = read_poscar(path).normalized()
    return {'lattice': [list(r) for r in s.lattice], 'positions': [list(p) for p in s.positions],
            'species': list(s.species)}


def _sparse_dft_env(threads: int) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith('SDFT_')}
    env.update(OMP_NUM_THREADS=str(threads), SDFT_PAW_ARCHIVE=str(PAW_ROOT), PYTHONPATH=str(ROOT),
               SDFT_ENABLE_PAW_CLOSURE_DIAGNOSTIC='1')
    return env


_PROFILE_FLAGS = {'density_cutoff_factor': '--density-cutoff-factor', 'xc_padding': '--xc-padding',
                  'eigen_search_factor': '--eigen-search-factor', 'block_ortho': '--block-ortho',
                  'paw_compensation': '--paw-compensation', 'projector_space': '--projector-space',
                  'projector_radius_factor': '--projector-radius-factor', 'search_precision': '--search-precision'}


def _recorded_sparsity_flags(scf_dir: Path) -> list[str]:
    """NSCF must reuse the SCF's resolved profile (run_bands refuses 'auto').

    The density cutoff (S1) is not passed: NSCF inherits it from the frozen
    checkpoint and sparse_dft.scf.run_bands rejects an explicit factor.
    """
    used = json.loads((scf_dir / 'scf_result.json').read_text())['provenance']['sparsity']
    if used['mode'] == 'off':
        return []
    flags = ['--sparsity', 'custom']
    for key, flag in _PROFILE_FLAGS.items():
        if key == 'density_cutoff_factor':
            continue
        if used.get(key) is not None:
            value = used[key]
            flags += [flag, str(int(value) if isinstance(value, bool) else value)]
    return flags


def _common_flags(st: dict, *, explicit_cutoff: bool = True) -> list[str]:
    flags = ['--scf-grid', *map(str, st['grid']), '--mesh-mode', 'gamma', '--smearing-ev', str(st['smearing_ev'])]
    if explicit_cutoff:
        flags += ['--cutoff', str(st['cutoff_ev'])]
    if st['spin_channels'] == 2:
        flags += ['--spin-channels', '2', '--initial-moments', *map(str, st['initial_moments'])]
    return flags


def _run_sparse_dft(command: list[str], directory: Path, threads: int) -> dict:
    done = directory / 'done.json'
    if done.exists():
        record = json.loads(done.read_text())
        if record.get('exit_code') == 0:
            return record
        # A failed stage is never cached: rerun it (e.g. after a code fix).
        shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with open(directory / 'cli.stdout.log', 'w') as o, open(directory / 'cli.stderr.log', 'w') as e:
        code = subprocess.call([sys.executable, '-m', 'sparse_dft.cli', *command], cwd=ROOT, env=_sparse_dft_env(threads),
                               stdout=o, stderr=e)
    record = {'exit_code': code, 'seconds': time.perf_counter() - start, 'threads': threads, 'command': command}
    write_json(done, record)
    return record


def sparse_dft(args):
    out, tier, threads = args.out, args.tier, args.threads
    manifest = json.loads((out / 'manifest.json').read_text())
    extra = TIERS[tier]
    for name in manifest['systems']:
        if args.systems and name not in args.systems:
            continue
        sd = out / 'systems' / name
        st = json.loads((sd / 'settings.json').read_text())
        base = out / 'sparse_dft' / tier / name
        # Default tier leaves cutoff and bands to Sparse-DFT's own rules; the report
        # checks both equal the VASP ENCUT/NBANDS (sync-bands copies a solved
        # automatic band extension into NBANDS). Other tiers pin both.
        scf = base / 'scf'
        pinned = [] if tier == 'default' else ['--bands', str(st['bands'])]
        cmd = ['scf', str(sd / 'POSCAR'), '--paw-root', str(PAW_ROOT), '--out', str(scf),
               *pinned, *_common_flags(st, explicit_cutoff=tier != 'default'), *extra]
        r = _run_sparse_dft(cmd, scf, threads)
        print(json.dumps({'tier': tier, 'system': name, 'task': 'scf', 'exit': r['exit_code'],
                          'seconds': round(r['seconds'], 1)}), flush=True)
        if tier in FULL_TASK_TIERS and (sd / 'band_path.json').exists() and r['exit_code'] == 0:
            bands = base / 'bands'
            cmd = ['bands', str(sd / 'POSCAR'), '--paw-root', str(PAW_ROOT), '--out', str(bands),
                   '--path', str(sd / 'band_path.json'), '--scf-checkpoint', str(scf / 'scf.json'),
                   '--bands', str(st['bands']), *_common_flags(st),
                   *(_recorded_sparsity_flags(scf) if extra == TIERS['auto'] else extra)]
            r = _run_sparse_dft(cmd, bands, threads)
            print(json.dumps({'tier': tier, 'system': name, 'task': 'bands', 'exit': r['exit_code'],
                              'seconds': round(r['seconds'], 1)}), flush=True)
    if tier not in FULL_TASK_TIERS:
        return
    for key, info in manifest['relax'].items():
        if args.systems and info['system'] not in args.systems:
            continue
        st = json.loads((out / 'systems' / info['system'] / 'settings.json').read_text())
        d = out / 'sparse_dft' / tier / 'relax' / key
        cmd = ['relax', str(out / 'relax' / key / 'POSCAR'), '--paw-root', str(PAW_ROOT), '--out', str(d),
               '--max-steps', '60', '--force-tolerance', str(FORCE_TOL),
               '--stress-tolerance', repr(STRESS_TOL_GPA / GPA_PER_EV_A3), *_common_flags(st), *extra]
        if info['task'] == 'cell':
            cmd.append('--cell')
        r = _run_sparse_dft(cmd, d, threads)
        print(json.dumps({'tier': tier, 'relax': key, 'exit': r['exit_code'], 'seconds': round(r['seconds'], 1)}),
              flush=True)


def solved_bands(scf_json: Path) -> int:
    """Bands Sparse-DFT actually solved; older outputs lack the field, so count eigenvalues."""
    d = json.loads(scf_json.read_text())
    if d.get('solved_bands'):
        return int(d['solved_bands'])
    return len(d['scf_kpoints'][0]['eigenvalues_ev'])


def sync_bands(args):
    """Copy Sparse-DFT's automatic band count into the VASP NBANDS where they differ.

    The stale VASP stages and pinned-band Sparse-DFT tiers move to an archive
    directory; rerun `vasp` and `sparse_dft --tier ...` afterwards.
    """
    out = args.out
    manifest = json.loads((out / 'manifest.json').read_text())
    stamp = time.strftime('%Y%m%d-%H%M%S')
    for name in manifest['systems']:
        if args.systems and name not in args.systems:
            continue
        scf_json = out / 'sparse_dft' / 'default' / name / 'scf' / 'scf.json'
        if not scf_json.exists():
            continue
        path = out / 'systems' / name / 'settings.json'
        st = json.loads(path.read_text())
        solved = solved_bands(scf_json)
        if solved == st['bands']:
            continue
        archive = out / 'archive' / f'{name}-nbands{st["bands"]}-{stamp}'
        archive.mkdir(parents=True)
        for stale in [out / 'vasp' / name, *[out / 'sparse_dft' / t / name for t in TIERS if t != 'default']]:
            if stale.exists():
                shutil.move(str(stale), str(archive / '_'.join(stale.relative_to(out).parts)))
        bands_json = out / 'sparse_dft' / 'default' / name / 'bands'
        if bands_json.exists():
            shutil.move(str(bands_json), str(archive / 'sparse_dft_default_bands'))
        st |= {'bands': solved, 'bands_static_rule': st['bands'],
               'bands_rule': 'Sparse-DFT automatic empty-band extension (scf.json solved_bands), copied to NBANDS'}
        write_json(path, st)
        print(json.dumps({'system': name, 'nbands': [st['bands_static_rule'], solved], 'archive': str(archive)}))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=['build', 'vasp', 'sparse_dft', 'sync-bands', 'report'])
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--systems', type=lambda s: [x for x in s.split(',') if x])
    ap.add_argument('--ranks', type=int, default=6)
    ap.add_argument('--threads', type=int, default=10)
    ap.add_argument('--tier', choices=sorted(TIERS), default='default')
    args = ap.parse_args(argv)
    args.out = args.out.resolve()
    sys.path.insert(0, str(ROOT / 'tools'))
    if args.command == 'report':
        from anchor_report import report
        return report(args)
    return {'build': build, 'vasp': vasp, 'sparse_dft': sparse_dft, 'sync-bands': sync_bands}[args.command](args) or 0


if __name__ == '__main__':
    raise SystemExit(main())
