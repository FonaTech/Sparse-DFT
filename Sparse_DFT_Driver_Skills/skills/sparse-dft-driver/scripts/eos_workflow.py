#!/usr/bin/env python3
"""Runnable EOS structure/recipe generation and quadratic analysis, stdlib only."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

from source_catalog import sha256


def new_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as file: json.dump(data, file, ensure_ascii=False, indent=2, allow_nan=False); file.write('\n')


def poscar(structure, path):
    structure = structure.normalized()
    symbols, counts = [], []
    for symbol in structure.species:
        if symbols and symbols[-1] == symbol: counts[-1] += 1
        else: symbols.append(symbol); counts.append(1)
    lines = ['Sparse-DFT EOS scaled structure; actual parent recorded in volume_points.json', '1.0']
    lines += [' '.join(f'{v:.17g}' for v in row) for row in structure.lattice]
    lines += [' '.join(symbols), ' '.join(map(str, counts))]
    if structure.selective_dynamics: lines.append('Selective dynamics')
    lines.append('Direct')
    for i, point in enumerate(structure.positions):
        line = ' '.join(f'{v:.17g}' for v in point)
        if structure.selective_dynamics: line += ' ' + ' '.join('T' if v else 'F' for v in structure.selective_dynamics[i])
        lines.append(line)
    with Path(path).open('x') as file: file.write('\n'.join(lines) + '\n')


def prepare(source, structure_path, factors_path, settings_path, project, relax_ions=False):
    sys.path.insert(0, str(source))
    from sparse_dft.structure import Structure, read_structure
    parent = read_structure(structure_path).normalized()
    if parent.lattice is None or not all(parent.periodic): raise ValueError('EOS requires a real periodic lattice')
    factors = json.loads(Path(factors_path).read_text())
    if not isinstance(factors, list) or len(factors) < 3 or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0 for x in factors) or len(set(factors)) != len(factors): raise ValueError('Supply at least three distinct positive finite volume factors')
    settings = json.loads(Path(settings_path).read_text())
    if not isinstance(settings, dict) or 'paw_root' not in settings: raise ValueError('SCF settings need actual paw_root and chosen numerical controls')
    if set(settings) & {'structure', 'output'}: raise ValueError('Structure/output are generated per point; omit them from settings')
    project = Path(project).resolve()
    folder = project / 'structures/eos'
    if folder.exists() or (project / 'eos-recipe.json').exists(): raise ValueError('EOS structures/recipe already exist; select a new project')
    folder.mkdir(parents=True)
    points, steps = [], []
    for i, factor in enumerate(factors):
        name = f'v{i:03d}'; scale = factor ** (1 / 3)
        structure = Structure(list(parent.species), list(parent.positions), tuple(tuple(v * scale for v in row) for row in parent.lattice), True, parent.periodic, parent.selective_dynamics)
        path = folder / f'{name}.POSCAR'; poscar(structure, path)
        output = f'runs/eos/{name}'
        params = {**settings, 'structure': str(path), 'output': output}
        if relax_ions:
            steps.append({'id': name, 'mode': 'api', 'method': 'relax.ionic', 'params': params, 'outputs': [output + '/relax_result.json']})
            # Last accepted state's SCF artifact is read by analysis below.
            result = output + '/relax_result.json'
        else:
            steps.append({'id': name, 'mode': 'api', 'method': 'job.run', 'params': params, 'outputs': [output + '/scf_result.json']})
            result = output + '/scf_result.json'
        points.append({'id': name, 'volume_factor': factor, 'volume_angstrom3': structure.volume, 'natoms': structure.natoms, 'structure': str(path), 'structure_sha256': sha256(path), 'result': result, 'relax_ions': relax_ions})
    new_json(project / 'volume_points.json', {'schema': 'sparse-dft-eos-points-1', 'parent': {'path': str(Path(structure_path).resolve()), 'sha256': sha256(structure_path)}, 'points': points})
    recipe = {'schema': 'sparse-dft-recipe-1', 'accuracy_validation_basis': 'project existing user-confirmed Sparse-DFT accuracy validation', 'environment': {'SDFT_ENABLE_PAW_CLOSURE_DIAGNOSTIC': '1'}, 'steps': steps}
    new_json(project / 'eos-recipe.json', recipe)
    return {'status': 'PREPARED', 'calculation_executed': False, 'points': len(points), 'recipe': str(project / 'eos-recipe.json')}


def solve(matrix, vector):
    a = [list(row) + [y] for row, y in zip(matrix, vector)]
    n = len(vector)
    for i in range(n):
        pivot = max(range(i, n), key=lambda k: abs(a[k][i]))
        a[i], a[pivot] = a[pivot], a[i]
        if abs(a[i][i]) < 1e-12: raise ValueError('Fit matrix is rank deficient')
        factor = a[i][i]; a[i] = [v / factor for v in a[i]]
        for j in range(n):
            if j != i:
                factor = a[j][i]; a[j] = [x - factor * y for x, y in zip(a[j], a[i])]
    return [row[-1] for row in a]


def analyze(points_path, project, output, result_method, energy_field='total_energy_ev'):
    data = json.loads(Path(points_path).read_text())
    rows, artifacts = [], [{'path': str(Path(points_path).resolve()), 'sha256': sha256(points_path)}]
    for point in data['points']:
        path = Path(project) / point['result']
        result = json.loads(path.read_text())
        if point.get('relax_ions'):
            if result.get('status') != 'CONVERGED': raise ValueError(f'{point["id"]}: ionic optimization did not converge')
            # The top-level relax result has no energy field. Follow its last
            # accepted SCF artifact and verify the saved identity.
            accepted = result['history'][-1]
            scf_path = Path(accepted['scf_file'])
            if not scf_path.is_absolute(): scf_path = Path(project) / scf_path
            if sha256(scf_path) != accepted['scf_sha256']: raise ValueError('Last accepted SCF artifact hash mismatch')
            artifacts.append({'path': str(scf_path.resolve()), 'sha256': sha256(scf_path)})
            scf = json.loads(scf_path.read_text())
            if scf.get('converged') is not True: raise ValueError('Last accepted SCF is not converged')
            energy = scf.get(energy_field)
        else:
            if result.get('converged') is not True: raise ValueError(f'{point["id"]}: SCF did not converge')
            energy = result.get(energy_field)
        if isinstance(energy, bool) or not isinstance(energy, (int, float)) or not math.isfinite(energy): raise ValueError(f'{point["id"]}: missing/nonfinite energy')
        volume = point['volume_angstrom3']
        rows.append({'id': point['id'], 'volume_angstrom3': volume, 'energy_ev': energy, 'energy_ev_per_atom': energy / point['natoms'], 'result': str(path.resolve()), 'sha256': sha256(path)})
        artifacts.append({'path': str(path.resolve()), 'sha256': sha256(path)})
    if len(rows) < 3: raise ValueError('At least three actual completed volume points are required')
    center = sum(x['volume_angstrom3'] for x in rows) / len(rows)
    scale = max(abs(x['volume_angstrom3'] - center) for x in rows)
    if scale <= 0: raise ValueError('All volumes are equal')
    offset = sum(x['energy_ev'] for x in rows) / len(rows)
    design = [[1., (x['volume_angstrom3'] - center) / scale, ((x['volume_angstrom3'] - center) / scale) ** 2] for x in rows]
    coefficients = solve([[sum(row[i] * row[j] for row in design) for j in range(3)] for i in range(3)], [sum(row[i] * (x['energy_ev'] - offset) for row, x in zip(design, rows)) for i in range(3)])
    constant, linear, quadratic = coefficients
    if quadratic <= 0: raise ValueError('No stable positive-curvature minimum; expand/reconsider actual volume branch')
    v0 = center - scale * linear / (2 * quadratic)
    if not min(x['volume_angstrom3'] for x in rows) < v0 < max(x['volume_angstrom3'] for x in rows): raise ValueError('Fitted minimum lies outside the sampled volume interval')
    e0 = offset + constant - linear * linear / (4 * quadratic)
    bulk = v0 * 2 * quadratic / (scale * scale)
    residuals = [x['energy_ev'] - (offset + sum(a * b for a, b in zip(row, coefficients))) for row, x in zip(design, rows)]
    report = {'schema': 'sparse-dft-eos-quadratic-1', 'result_method': result_method, 'fit': 'E=offset+c0+c1*x+c2*x^2; x=(V-center)/scale', 'center_angstrom3': center, 'scale_angstrom3': scale, 'offset_ev': offset, 'coefficients_ev': coefficients, 'equilibrium_volume_angstrom3': v0, 'minimum_energy_ev_per_cell': e0, 'bulk_modulus_ev_per_angstrom3': bulk, 'bulk_modulus_gpa': bulk * 160.2176634, 'bulk_formula': 'B=V0*d2E/dV2; 1 eV/angstrom3=160.2176634 GPa', 'rmse_ev': math.sqrt(sum(x*x for x in residuals) / len(rows)), 'uncertainty': 'Fit-range/model uncertainty not estimated by this quadratic estimator', 'accounting_scope': 'one simulation cell; atoms unchanged across points', 'provenance': {'artifacts': artifacts + [{'path': str(Path(__file__).resolve()), 'sha256': sha256(__file__)}], 'locator': 'volume_points.json and each source result energy field'}}
    report['energy_field'] = energy_field
    report['energy_definition'] = 'F=E-TS, equals E at zero entropy' if energy_field == 'total_energy_ev' else 'internal energy E before entropy contribution'
    output = Path(output)
    if output.exists(): raise ValueError('Output exists; choose a new analysis directory')
    output.mkdir(parents=True)
    new_json(output / 'eos.json', report)
    with (output / 'eos.csv').open('x', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='mode', required=True)
    q = sub.add_parser('prepare'); q.add_argument('--source', type=Path, required=True); q.add_argument('--structure', type=Path, required=True); q.add_argument('--volume-factors', type=Path, required=True); q.add_argument('--settings', type=Path, required=True); q.add_argument('--project', type=Path, required=True); q.add_argument('--relax-ions', action='store_true')
    q = sub.add_parser('analyze'); q.add_argument('--points', type=Path, required=True); q.add_argument('--project', type=Path, required=True); q.add_argument('--out', type=Path, required=True); q.add_argument('--result-method', choices=['simulated', 'synthetic'], default='simulated'); q.add_argument('--energy-field', choices=['total_energy_ev', 'internal_energy_ev'], default='total_energy_ev')
    a = p.parse_args()
    try:
        result = prepare(a.source.resolve(), a.structure, a.volume_factors, a.settings, a.project, a.relax_ions) if a.mode == 'prepare' else analyze(a.points, a.project, a.out, a.result_method, a.energy_field)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)); return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({'status': 'FAILED', 'message': str(exc)})); return 2


if __name__ == '__main__':
    raise SystemExit(main())
