#!/usr/bin/env python3
"""Summarize sweep_sparsity.py output: per-component error, pass rate and
measured work reduction against the same-system 16xENCUT reference.

Usage: python3.11 -I tools/report_sparsity.py SWEEP_DIR [SWEEP_DIR ...] --out report.json

Work reduction uses counters, not only wall time:
  S1  density_reciprocal_modes and max_xc_grid_points (reference / run)
  S3  operator_hamiltonian + operator_overlap calls
  S4  operator_overlap calls
  S5  paw_compensated_factor_product seconds
Wall-time ratios are reported separately and are only meaningful for runs
timed on an otherwise idle machine (see the 'timing' field).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _counter(run, key, field='seconds'):
    return run.get('sparsity_ledger', {}).get('counters', {}).get(key, {}).get(field)


def _ratio(a, b):
    return a / b if a and b else None


def work_ratios(ref, run):
    lr, lv = ref.get('sparsity_ledger', {}), run.get('sparsity_ledger', {})
    applications = lambda r: ((_counter(r, 'operator_hamiltonian', 'calls') or 0) +
                              (_counter(r, 'operator_overlap', 'calls') or 0))
    return {
        'density_modes': _ratio(lr.get('density_reciprocal_modes'), lv.get('density_reciprocal_modes')),
        'xc_grid_points': _ratio(lr.get('max_xc_grid_points'), lv.get('max_xc_grid_points')),
        'operator_applications': _ratio(applications(ref), applications(run)),
        'overlap_applications': _ratio(_counter(ref, 'operator_overlap', 'calls'), _counter(run, 'operator_overlap', 'calls')),
        'density_functional_seconds': _ratio(_counter(ref, 'density_functional'), _counter(run, 'density_functional')),
        'diagonalization_seconds': _ratio(_counter(ref, 'scf_diagonalization'), _counter(run, 'scf_diagonalization')),
        'total_seconds': _ratio(_counter(ref, 'process_compute_total'), _counter(run, 'process_compute_total')),
        'outer_iterations': (ref.get('iterations'), run.get('iterations')),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('sweeps', nargs='+')
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    report = {'schema_version': 'sparse-dft-sparsity-report-1', 'systems': {}}
    for sweep in map(Path, a.sweeps):
        summary = _load(sweep / 'summary.json') or {}
        for system, data in summary.items():
            ref = _load(sweep / system / 'reference' / 'scf_result.json')
            if not ref or not ref.get('converged'):
                report['systems'][system] = {'reference_converged': False}
                continue
            rows = []
            for row in data['rows']:
                run = _load(sweep / system / row['variant'] / 'scf_result.json') or {}
                entry = {k: row.get(k) for k in ('variant', 'pass', 'energy_mev_per_atom', 'force_ev_per_angstrom',
                                                  'stress_gpa', 'band_window_mev', 'converged')}
                entry['gate_use'] = max((row.get(k) or 0) / v for k, v in
                                        {'energy_mev_per_atom': 1.0, 'force_ev_per_angstrom': 1e-3,
                                         'stress_gpa': 0.01, 'band_window_mev': 5.0}.items()) if row.get('converged') else None
                if run.get('converged'):
                    entry['work'] = work_ratios(ref, run)
                    entry['chosen'] = run.get('provenance', {}).get('sparsity')
                rows.append(entry)
            report['systems'][system] = {'reference_converged': True, 'natoms': data.get('natoms'),
                                         'reference_seconds': _counter(ref, 'process_compute_total'),
                                         'rows': rows}
    Path(a.out).write_text(json.dumps(report, indent=1) + '\n')
    for system, data in report['systems'].items():
        if not data.get('reference_converged'):
            print(f'{system}: reference not converged'); continue
        for r in data['rows']:
            w = r.get('work') or {}
            fmt = lambda x: f'{x:5.2f}' if isinstance(x, (int, float)) else '  -  '
            print(f"{system:8s} {r['variant']:16s} pass={str(r['pass']):5s} use={fmt(r['gate_use'])} "
                  f"modes x{fmt(w.get('density_modes'))} xc x{fmt(w.get('xc_grid_points'))} "
                  f"ops x{fmt(w.get('operator_applications'))} S x{fmt(w.get('overlap_applications'))} "
                  f"T x{fmt(w.get('total_seconds'))} it={w.get('outer_iterations')}")


if __name__ == '__main__':
    main()
