"""Numeric tests of the sparse_dft.analysis loaders (pure Python, no matplotlib)."""
import json
import math
import tempfile
import unittest
from pathlib import Path

from sparse_dft.analysis.common import EV_PER_ANGSTROM3_TO_GPA, resolve_fermi
from sparse_dft.analysis.electronic import (band_edges, bands_from_scf, compare_band_data, kpath, load_band_data,
                                      load_dos, load_projections, pretty_kpoint_label)
from sparse_dft.analysis.relax import load_relax
from sparse_dft.analysis.scf import (load_energy_terms, load_forces, load_scf_history, load_scf_run, load_stress,
                               load_timing)
from sparse_dft.analysis.sweep import GATES, gate_use, load_scaling, load_sweep
from sparse_dft.bands import density_of_states_from_bands, density_of_states_from_scf, from_payload
from sparse_dft.errors import InputError

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / 'tests' / 'fixtures'
PF = FIX / 'plotting'
NI = PF / 'ni_spin_scf'
SWEEP = Path('/tmp/sparse_dft_sweep')


class ScfLoaderTests(unittest.TestCase):
    def test_run_summary_from_trimmed_real_result(self):
        run = load_scf_run(NI)
        self.assertTrue(run.converged)
        self.assertFalse(run.total_energy_valid)
        self.assertEqual(run.spin_channels, 2)
        self.assertEqual(run.species, ['Ni'])
        self.assertAlmostEqual(run.magnetic_moment_muB, 0.5829974348447797, places=12)
        self.assertAlmostEqual(run.fermi_energy_ev, 9.36348724614469, places=12)

    def test_energy_ledger_reproduces_total(self):
        terms = load_energy_terms(NI)
        self.assertEqual(terms.missing_terms, [])
        self.assertTrue(terms.additive)
        self.assertLess(abs(terms.ledger_residual_ev), 1e-9)
        self.assertAlmostEqual(terms.ledger_sum_ev, terms.total_energy_ev, places=9)

    def test_non_additive_ledger_is_detected(self):
        with tempfile.TemporaryDirectory() as td:
            data = json.loads((NI / 'scf_result.json').read_text(encoding='utf-8'))
            data['hartree_energy_ev'] += 0.5
            (Path(td) / 'scf_result.json').write_text(json.dumps(data), encoding='utf-8')
            terms = load_energy_terms(td)
            self.assertFalse(terms.additive)
            self.assertAlmostEqual(terms.ledger_residual_ev, -0.5, places=9)

    def test_stress_unit_conversion_and_pressure_sign(self):
        st = load_stress(NI)
        raw = json.loads((NI / 'scf_result.json').read_text(encoding='utf-8'))['hf_total_stress_ev_per_angstrom3']
        self.assertEqual(EV_PER_ANGSTROM3_TO_GPA, 160.21766208)
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(st.stress_gpa[i][j], raw[i][j] * 160.21766208, places=12)
        trace = sum(raw[i][i] for i in range(3)) / 3 * 160.21766208
        self.assertAlmostEqual(st.pressure_gpa, -trace, places=12)

    def test_forces_magnitudes(self):
        f = load_forces(NI)
        self.assertEqual(f.natoms, 1)
        row = f.forces_ev_per_angstrom[0]
        self.assertAlmostEqual(f.magnitudes_ev_per_angstrom[0], math.sqrt(sum(c * c for c in row)), places=15)
        self.assertEqual(f.atom_labels(), ['Ni1'])

    def test_history_and_tolerances(self):
        h = load_scf_history(NI)
        self.assertEqual(h.niterations, 23)
        self.assertEqual(h.iteration[0], 1)
        self.assertTrue(h.spin_polarized)
        self.assertEqual(h.tolerances, {'density_residual': 1e-08, 'onecenter_density_residual': 1e-08})
        self.assertAlmostEqual(h.series['output_magnetic_moment_muB'][-1], 0.5829974356904134, places=12)
        self.assertTrue(h.converged)

    def test_timers_are_flagged_overlapping(self):
        t = load_timing(NI)
        self.assertTrue(t.timings_overlap)
        self.assertEqual(t.openmp_max_threads, 8)
        self.assertAlmostEqual(t.wall_seconds, 73.9308124575764, places=9)
        top = t.top(3)
        self.assertEqual([e.name for e in top], ['scf_diagonalization', 'davidson_total', 'davidson_iteration'])
        self.assertTrue(all(e.seconds > 0 for e in t.top(50)))


class ElectronicLoaderTests(unittest.TestCase):
    def test_gamma_labels(self):
        for raw in ('GAMMA', 'G', 'Γ', '\\Gamma', 'gamma'):
            self.assertEqual(pretty_kpoint_label(raw), r'$\Gamma$', raw)
        self.assertEqual(pretty_kpoint_label('X|U'), r'$\mathrm{X}|\mathrm{U}$')
        self.assertEqual(pretty_kpoint_label('S_0'), r'$\mathrm{S}_{0}$')
        self.assertIsNone(pretty_kpoint_label(''))

    def test_broken_path_segments_and_ticks(self):
        b = load_band_data(PF / 'synthetic_bands')
        kp = kpath(b)
        self.assertEqual(kp.segments, [(0, 6), (7, 13)])
        self.assertEqual(kp.breaks, [7])
        self.assertEqual(kp.x[6], kp.x[7])                     # no gap drawn for the jump
        self.assertEqual([t for _, t in kp.ticks], [r'$\Gamma$', r'$\mathrm{X}|\mathrm{K}$', r'$\Gamma$'])

    def test_indirect_gap_from_occupations(self):
        e = band_edges(load_band_data(PF / 'synthetic_bands'))
        self.assertEqual(e.status, 'insulator')
        self.assertAlmostEqual(e.vbm_ev, 0.0, places=12)
        self.assertAlmostEqual(e.cbm_ev, 1.1, places=12)
        self.assertAlmostEqual(e.gap_ev, 1.1, places=12)
        self.assertFalse(e.direct)
        self.assertEqual(e.vbm_kindex, [0, 13])                 # Gamma at both path ends
        self.assertEqual(e.cbm_kindex, [6, 7])                  # X and K
        # Minimum over k of c1 - v2 = 1.5 - 0.4 t + 0.5 t^2 on the sampled t = i/6.
        expected = min(1.5 - 0.4 * (i / 6) + 0.5 * (i / 6) ** 2 for i in range(7))
        self.assertAlmostEqual(e.direct_gap_ev, expected, places=12)

    def test_direct_gap_on_real_vasprun(self):
        e = band_edges(load_band_data(FIX / 'vasp' / 'si8_gamma' / 'vasprun.xml'))
        self.assertTrue(e.direct)
        self.assertAlmostEqual(e.gap_ev, 6.6384 - 6.2153, places=9)

    def test_metal_has_no_gap(self):
        e = band_edges(bands_from_scf(NI))
        self.assertTrue(e.is_metal)
        self.assertIsNone(e.gap_ev)

    def test_dos_integral_counts_electrons(self):
        # Two k-points, two bands, spin-degenerate: 2 * 2 = 4 states in total.
        payload = {'kpoints': [[0, 0, 0], [.5, 0, 0]], 'eigenvalues': [[-1, 1], [-.5, 1.5]], 'kweights': [.25, .75],
                   'fermi_energy': 0.0, 'metadata': {'sampling': 'uniform_mesh'}}
        from sparse_dft.analysis.electronic import dos_from_payload
        dos = dos_from_payload(density_of_states_from_bands(from_payload(payload), sigma=.05, npoints=4001))
        self.assertAlmostEqual(dos.electron_count(), 4.0, places=6)
        self.assertAlmostEqual(dos.electron_count(emax_ev=0.0), 2.0, places=6)

    def test_spin_scf_dos_both_channels(self):
        dos = load_dos(NI, sigma=0.1)
        self.assertTrue(dos.spin_polarized)
        self.assertEqual([c.spin for c in dos.total], ['up', 'down'])
        self.assertAlmostEqual(dos.electron_count(), 24.0, places=6)      # 12 bands x 2 spins
        self.assertTrue(all(v >= 0 for c in dos.total for v in c.values))

    def test_nonspin_scf_dos_no_longer_crashes(self):
        # Regression: spin-unpolarized SCF occupations were passed as a bare list.
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'scf.json'
            p.write_text(json.dumps({'converged': True, 'spin_channels': 1, 'fermi_energy_ev': 0.0, 'scf_kpoints': [
                {'fractional': [0, 0, 0], 'weight': 1.0, 'spin_channel': 0, 'eigenvalues_ev': [-1.0, 1.0],
                 'occupations': [2.0, 0.0]}]}), encoding='utf-8')
            dos = density_of_states_from_scf(p, sigma=0.05, npoints=2001)
            xs, ys = dos['energies_ev'], dos['total']['total']
            self.assertAlmostEqual(sum((xs[i + 1] - xs[i]) * (ys[i] + ys[i + 1]) / 2 for i in range(len(xs) - 1)), 4.0, places=6)

    def test_unknown_fermi_stays_unknown(self):
        payload = {'kpoints': [[0, 0, 0]], 'eigenvalues': [[-1, 1]], 'kweights': [1.0],
                   'metadata': {'sampling': 'uniform_mesh'}}
        dos = density_of_states_from_bands(from_payload(payload), sigma=.1, npoints=101)
        self.assertIsNone(dos['fermi_energy_ev'])
        with self.assertRaisesRegex(InputError, '--fermi'):
            resolve_fermi(None, None, 'x.json')
        self.assertEqual(resolve_fermi(None, 1.5, 'x.json'), (1.5, 'option'))

    def test_pdos_split_by_element_and_l(self):
        dos = load_dos(PF / 'synthetic_spin_dos')
        keys = [(c.element, c.l, c.spin) for c in dos.pdos]
        self.assertEqual(len(keys), 12)                          # 2 elements x 3 l x 2 spins
        self.assertEqual(len(set(keys)), 12)
        self.assertEqual({c.element for c in dos.pdos}, {'Mg', 'O'})
        self.assertEqual({c.l for c in dos.pdos}, {'s', 'p', 'd'})

    def test_projection_groups(self):
        b = load_band_data(PF / 'synthetic_bands')
        proj = load_projections(b, PF / 'synthetic_bands' / 'PROCAR')
        groups = proj.grouped('total', by='l')
        self.assertEqual(list(groups), ['s', 'p', 'd'])
        self.assertAlmostEqual(groups['s'][0][0], 0.8, places=6)
        self.assertAlmostEqual(groups['d'][0][2], 0.6, places=6)

    def test_band_comparison_identity(self):
        b = bands_from_scf(NI)
        cmp = compare_band_data(b, b, align='fermi')
        self.assertLess(cmp.metrics['max_abs_error_ev'], 1e-12)      # interpolation roundoff only
        self.assertTrue(all(abs(d) < 1e-12 for d in cmp.delta_ev))


class WorkflowLoaderTests(unittest.TestCase):
    def test_relax_trajectory(self):
        r = load_relax(PF / 'synthetic_relax')
        self.assertEqual(r.kind, 'ionic')
        self.assertEqual((r.tolerance, r.tolerance_origin), (0.02, 'file'))
        self.assertEqual(r.steps, list(range(6)))
        self.assertIsNone(r.step_length[0])
        self.assertAlmostEqual(r.step_length[1], 0.05 * 0.65, places=12)
        self.assertLess(r.residual[-1], r.tolerance)

    def test_relax_checkpoint_schema_defaults(self):
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / 'cell_relax_checkpoint.json').write_text(json.dumps({
                'schema_version': 'sparse-dft-cell-relax-2', 'step': 1, 'history': [
                    {'step': 0, 'energy_ev': -1.0, 'stress_ev_per_angstrom3': [[0.01, 0, 0], [0, 0.01, 0], [0, 0, -0.02]]},
                    {'step': 1, 'energy_ev': -1.1, 'stress_ev_per_angstrom3': [[1e-4, 0, 0], [0, 1e-4, 0], [0, 0, 0]]}]}),
                encoding='utf-8')
            r = load_relax(td)
            self.assertEqual(r.kind, 'cell')
            self.assertEqual(r.tolerance_origin, 'default')
            self.assertAlmostEqual(r.residual[0], 0.02)
            self.assertAlmostEqual(r.residual_display[0], 0.02 * 160.21766208)

    def test_scaling_math(self):
        s = load_scaling(PF / 'synthetic_scaling.json')
        self.assertEqual(s.builds, ['v1', 'v2'])
        self.assertEqual(s.axis, 'threads')
        pts = {p.threads: p for p in s.points('v2')}
        t1 = 100.0
        for n in (1, 2, 4, 8):
            wall = t1 * (0.05 + 0.95 / n)
            self.assertAlmostEqual(pts[n].wall_seconds, wall, places=5)
            self.assertAlmostEqual(pts[n].speedup, t1 / wall, places=5)
            self.assertAlmostEqual(pts[n].efficiency, t1 / wall / n, places=5)
            self.assertAlmostEqual(pts[n].occupancy, t1 / (wall * n), places=5)

    def test_scaling_without_baseline_has_no_speedup(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 's.json'
            p.write_text(json.dumps({'schema_version': 'sparse-dft-scaling-1', 'system': 'x', 'runs': [
                {'label': 'a', 'threads': 2, 'ranks': 1, 'wall_seconds': 5.0, 'user_seconds': 9.0, 'sys_seconds': 0.5}]}),
                encoding='utf-8')
            pt = load_scaling(p).points('default')[0]
            self.assertIsNone(pt.speedup)
            self.assertAlmostEqual(pt.occupancy, 9.5 / 10.0)
            p.write_text(json.dumps({'schema_version': 'sparse-dft-scaling-1', 'runs': [{'threads': 0, 'wall_seconds': 1}]}),
                         encoding='utf-8')
            with self.assertRaises(InputError):
                load_scaling(p)

    def test_gate_use_is_max_ratio(self):
        row = {'energy_mev_per_atom': 0.5, 'force_ev_per_angstrom': 2e-3, 'stress_gpa': 0.001, 'band_window_mev': 1.0}
        self.assertAlmostEqual(gate_use(row), 2.0)
        self.assertEqual(GATES, {'energy_mev_per_atom': 1.0, 'force_ev_per_angstrom': 1e-3, 'stress_gpa': 0.01,
                                 'band_window_mev': 5.0})

    @unittest.skipUnless((SWEEP / 'report_small.json').is_file(), 'sweep report not present')
    def test_real_sweep_report_matches_summary(self):
        report = load_sweep(SWEEP / 'report_small.json')
        summary = load_sweep(SWEEP / 'small' / 'summary.json')
        a = {(r.system, r.variant): r for r in report.rows}
        b = {(r.system, r.variant): r for r in summary.rows}
        for key, row in a.items():
            if row.gate_use is not None:
                self.assertAlmostEqual(b[key].gate_use, row.gate_use, places=12)
            if 'density_modes' in row.work:
                self.assertAlmostEqual(b[key].work['density_modes'], row.work['density_modes'], places=9)


if __name__ == '__main__':
    unittest.main()


class SchemaAliasTests(unittest.TestCase):
    """Loaders accept the current schema prefix and the one files carried before the project rename."""

    def test_schema_ids_cover_current_and_legacy(self):
        from sparse_dft.analysis.common import SCHEMA_PREFIX, schema_ids
        from sparse_dft.analysis.scf import SCF_SCHEMA, SCF_SCHEMAS
        ids = schema_ids('scf-1')
        self.assertEqual(ids[0], SCHEMA_PREFIX + 'scf-1')
        self.assertIn(SCF_SCHEMA, ids)                       # the id this package writes
        self.assertIn('d' + 'ftx-scf-1', ids)                # files written before the rename
        self.assertEqual(SCF_SCHEMAS, ids)

    def test_legacy_scaling_file_loads(self):
        payload = {'schema_version': 'd' + 'ftx-scaling-1', 'runs': [
            {'workers': 1, 'wall_s': 10.0, 'label': 'a'}, {'workers': 2, 'wall_s': 6.0, 'label': 'a'}]}
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'scaling.json'
            path.write_text(json.dumps(payload), encoding='utf-8')
            from sparse_dft.analysis.sweep import is_scaling_payload
            self.assertTrue(is_scaling_payload(payload))
