#!/usr/bin/env python3
"""Sweep sparsification components per system and judge them against the accuracy gates.

Usage: python3 -I sweep_sparsity.py CAMPAIGN.json OUT_DIR [--only NAME ...]

CAMPAIGN.json:
  {"paw_root": "...", "systems": [{"name": "Si2", "structure": "...", "cutoff": 500,
    "scf_grid": [2,2,2], "bands": 8, "smearing_ev": 0, "spin_channels": 1,
    "initial_moments": null, "paw_map": null}], "variants": [{"name": "...", "sparsity": {...}}]}

Each system runs a reference (sparsity off, 16xENCUT) and every variant. A
finished run is reused only if its recorded settings hash matches. Results
go to OUT_DIR/summary.json. Gates: 1 meV/atom energy, 1e-3 eV/A force,
0.01 GPa stress, 5 meV band window (docs/ACCURACY.md, section 1).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sparse_dft.scf import run_scf  # noqa: E402

GATES = {'energy_mev_per_atom': 1.0, 'force_ev_per_angstrom': 1e-3, 'stress_gpa': 0.01,
         'band_window_mev': 5.0}
EV_PER_A3_TO_GPA = 160.21766208
REFERENCE = {'name': 'reference', 'sparsity': None}


def _flat(m):
    return [x for row in m for x in row]


def _settings_hash(system, variant, paw_root):
    keys = {k: os.environ[k] for k in sorted(os.environ) if k.startswith('SDFT_')}
    from sparse_dft.scf import _executable
    exe = _executable()
    rules = (ROOT / 'sparse_dft' / 'sparsity_rules.json').read_bytes() if variant['sparsity'] == 'auto' else b''
    blob = json.dumps({'system': system, 'variant': variant, 'paw_root': str(paw_root), 'env': keys,
                       'exe': hashlib.sha256(exe.read_bytes()).hexdigest(),
                       'rules': hashlib.sha256(rules).hexdigest()}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def _run(system, variant, paw_root, out_dir):
    out = out_dir / system['name'] / variant['name']
    out.mkdir(parents=True, exist_ok=True)
    digest = _settings_hash(system, variant, paw_root)
    stamp = out / 'sweep_settings.sha256'
    result_path = out / 'scf_result.json'
    if stamp.exists() and stamp.read_text().strip() == digest and result_path.exists():
        return json.loads(result_path.read_text())
    stamp.unlink(missing_ok=True)
    paw = system.get('paw_map') or paw_root
    started = time.perf_counter()
    try:
        result = run_scf(ROOT / system['structure'] if not Path(system['structure']).is_absolute() else system['structure'],
                         paw, out, cutoff_ev=system['cutoff'], scf_grid=system.get('scf_grid'),
                         bands=system['bands'], smearing_ev=system.get('smearing_ev', 0.0),
                         spin_channels=system.get('spin_channels', 1),
                         initial_moments=system.get('initial_moments'),
                         density_tolerance=system.get('density_tolerance', 1e-8),
                         onecenter_density_tolerance=system.get('density_tolerance', 1e-8),
                         mixing=system.get('mixing', 0.2),
                         max_iterations=system.get('max_iterations', 2000),
                         density_cutoff_ev=(16 * system['cutoff'] if variant['sparsity'] is None else None),
                         sparsity=variant['sparsity'])
    except Exception as error:  # recorded, never treated as a pass
        result = {'converged': False, 'error': repr(error)}
        result_path.write_text(json.dumps(result) + '\n')
    result['sweep_wall_seconds'] = time.perf_counter() - started
    stamp.write_text(digest + '\n')
    return result


def _band_window(ref, run, window_ev=5.0):
    """Max |de| of bands within window_ev above the lowest band, aligned at E_F."""
    worst = 0.0
    for kr, kc in zip(ref['scf_kpoints'], run['scf_kpoints']):
        e0 = kr['eigenvalues_ev']
        lo = min(e0)
        for a, b in zip(e0, kc['eigenvalues_ev']):
            if a - ref['fermi_energy_ev'] <= window_ev and a >= lo:
                worst = max(worst, abs((a - ref['fermi_energy_ev']) - (b - run['fermi_energy_ev'])))
    return 1000 * worst


def compare(ref, run, natoms):
    if not run.get('converged'):
        return {'converged': False, 'error': run.get('error'), 'pass': False}
    d = {
        'converged': True,
        'energy_mev_per_atom': 1000 * abs(run['total_energy_ev'] - ref['total_energy_ev']) / natoms,
        'force_ev_per_angstrom': max(abs(a - b) for a, b in zip(_flat(run['hf_total_forces_ev_per_angstrom']),
                                                                 _flat(ref['hf_total_forces_ev_per_angstrom']))),
        'stress_gpa': EV_PER_A3_TO_GPA * max(abs(a - b) for a, b in zip(
            _flat(run['hf_total_stress_ev_per_angstrom3']), _flat(ref['hf_total_stress_ev_per_angstrom3']))),
        'band_window_mev': _band_window(ref, run),
        'electron_count_relative': abs(run['electron_count'] - ref['electron_count']) / ref['electron_count'],
    }
    d['pass'] = all(d[k] <= v for k, v in GATES.items()) and d['electron_count_relative'] <= 1e-8
    return d


def _seconds(run, key):
    return run.get('sparsity_ledger', {}).get('counters', {}).get(key, {}).get('seconds')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('campaign'); ap.add_argument('out'); ap.add_argument('--only', nargs='*')
    a = ap.parse_args(argv)
    campaign = json.loads(Path(a.campaign).read_text())
    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / 'summary.json'
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for system in campaign['systems']:
        if a.only and system['name'] not in a.only:
            continue
        natoms = system['natoms']
        ref = _run(system, REFERENCE, campaign['paw_root'], out_dir)
        rows = []
        t_ref = _seconds(ref, 'process_compute_total')
        for variant in campaign['variants']:
            run = _run(system, variant, campaign['paw_root'], out_dir)
            row = {'variant': variant['name'], 'sparsity': variant['sparsity']}
            if ref.get('converged'):
                row.update(compare(ref, run, natoms))
            t = _seconds(run, 'process_compute_total')
            row['seconds'] = t
            row['speedup'] = (t_ref / t) if t and t_ref else None
            led = run.get('sparsity_ledger', {})
            row['density_modes'] = led.get('density_reciprocal_modes')
            row['xc_points'] = led.get('max_xc_grid_points')
            for key in ('operator_hamiltonian', 'operator_overlap', 'paw_block_projectors'):
                row[key + '_calls'] = led.get('counters', {}).get(key, {}).get('calls')
            row['iterations'] = run.get('iterations')
            row['auto_choice'] = run.get('provenance', {}).get('sparsity')
            rows.append(row)
            print(f"{system['name']:10s} {variant['name']:22s} pass={row.get('pass')} "
                  f"dE={row.get('energy_mev_per_atom', float('nan')):.4f} dF={row.get('force_ev_per_angstrom', float('nan')):.1e} "
                  f"dS={row.get('stress_gpa', float('nan')):.4f} band={row.get('band_window_mev', float('nan')):.3f} "
                  f"x{row['speedup'] or float('nan'):.2f}", flush=True)
        summary[system['name']] = {'natoms': natoms, 'reference_seconds': t_ref,
                                   'reference_converged': ref.get('converged'),
                                   'reference_modes': ref.get('density_reciprocal_modes'),
                                   'reference_xc_points': ref.get('max_xc_grid_points'), 'rows': rows}
        summary_path.write_text(json.dumps(summary, indent=1) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
