import json
import tempfile
import unittest
from pathlib import Path

from sparse_dft.errors import InputError, UnsupportedError
from sparse_dft.sparsity import (SparsityProfile, coerce_profile, profile_environment, resolve_auto,
                           resolved_density_cutoff, system_features, vacuum_fraction, with_overrides,
                           sparsity_ledger, load_rules)
from sparse_dft.structure import Structure


def _probe(tails_by_factor):
    return {'species': [{'record_id': 'X:standard', 'zval': 4.0, 'tails': [
        {'factor': f, 'compensation_hartree': t, 'core_hartree': t / 10} for f, t in tails_by_factor.items()]}]}


RULES = {
    'schema_version': 'sparse-dft-sparsity-rules-1',
    'density_cutoff': {'id': 'S1', 'candidates': [4, 6, 8, 16], 'tail_budget': 1e-7, 'fallback_factor': 16,
                       'stricter_when': [{'id': 'S1.metal', 'feature': 'metallic', 'tail_budget': 1e-8}]},
    'eigen_search_factor': [{'id': 'S3.spin', 'when': {'spin': True}, 'value': 1e-5},
                            {'id': 'S3.all', 'when': {}, 'value': 1e-4}],
    'block_ortho': [{'id': 'S4.large', 'when': {'natoms_min': 4}, 'value': True}],
}


class SparsityProfileTest(unittest.TestCase):
    def test_off_sets_no_environment(self):
        base = {'PATH': '/bin'}
        self.assertEqual(profile_environment(SparsityProfile(), base), base)

    def test_off_rejects_components(self):
        with self.assertRaises(InputError):
            coerce_profile({'mode': 'off', 'xc_padding': 1})

    def test_custom_maps_each_component(self):
        env = profile_environment(coerce_profile({'xc_padding': 1, 'eigen_search_factor': 1e-3,
                                                  'block_ortho': False, 'paw_compensation': 'off'}), {})
        self.assertEqual(env['SDFT_FFT_OVERSAMPLING'], '1')
        self.assertEqual(env['SDFT_ADAPTIVE_SEARCH'], '1')
        self.assertEqual(env['SDFT_SCF_SEARCH_FACTOR'], '0.001')
        self.assertEqual(env['SDFT_BLOCK_ORTHOGONALIZE'], '0')
        self.assertEqual(env['SDFT_PAW_COMPENSATED_ACCUMULATION'], '0')

    def test_conflicting_user_environment_is_an_error(self):
        with self.assertRaises(InputError):
            profile_environment(coerce_profile({'xc_padding': 1}), {'SDFT_FFT_OVERSAMPLING': '2'})
        # Agreement is fine.
        profile_environment(coerce_profile({'xc_padding': 2}), {'SDFT_FFT_OVERSAMPLING': '2'})

    def test_ranges_are_validated(self):
        for bad in ({'density_cutoff_factor': 3.0}, {'eigen_search_factor': 0.5},
                    {'projector_radius_factor': 0.5}, {'xc_padding': 7}):
            with self.assertRaises(InputError):
                coerce_profile(bad)

    def test_unimplemented_values_are_refused(self):
        with self.assertRaises(UnsupportedError):
            coerce_profile({'search_precision': 'fp32'})

    def test_density_factor_and_absolute_cutoff_are_exclusive(self):
        p = coerce_profile({'density_cutoff_factor': 6})
        self.assertEqual(resolved_density_cutoff(p, 500.0, None), 3000.0)
        with self.assertRaises(InputError):
            resolved_density_cutoff(p, 500.0, 4000.0)
        self.assertEqual(resolved_density_cutoff(SparsityProfile(), 500.0, 4000.0), 4000.0)


class AutoRuleTest(unittest.TestCase):
    def features(self, **kw):
        base = dict(natoms=2, smearing_ev=0.0, spin_channels=1, vacuum_fraction=0.0)
        base.update(kw)
        return system_features(_probe({4: 1e-6, 6: 5e-8, 8: 5e-9, 16: 1e-10}), **base)

    def test_smallest_factor_meeting_budget(self):
        p = resolve_auto(_probe({4: 1e-6, 6: 5e-8, 8: 5e-9, 16: 1e-10}), self.features(), RULES)
        self.assertEqual(p.density_cutoff_factor, 6.0)
        self.assertIn('S1', p.rule_ids)

    def test_metal_uses_stricter_budget(self):
        p = resolve_auto(_probe({4: 1e-6, 6: 5e-8, 8: 5e-9, 16: 1e-10}), self.features(smearing_ev=0.1), RULES)
        self.assertEqual(p.density_cutoff_factor, 8.0)
        self.assertIn('S1.metal', p.rule_ids)

    def test_fallback_when_no_candidate_meets_budget(self):
        p = resolve_auto(_probe({4: 1e-3, 6: 1e-3, 8: 1e-3, 16: 1e-3}), self.features(), RULES)
        self.assertEqual(p.density_cutoff_factor, 16.0)

    def test_ordered_rules_and_feature_thresholds(self):
        p = resolve_auto(_probe({4: 1e-9, 6: 1e-9, 8: 1e-9, 16: 1e-9}), self.features(spin_channels=2), RULES)
        self.assertEqual(p.eigen_search_factor, 1e-5)
        self.assertIsNone(p.block_ortho)
        p = resolve_auto(_probe({4: 1e-9, 6: 1e-9, 8: 1e-9, 16: 1e-9}), self.features(natoms=8), RULES)
        self.assertEqual(p.eigen_search_factor, 1e-4)
        self.assertTrue(p.block_ortho)

    def test_explicit_component_overrides_auto(self):
        p = resolve_auto(_probe({4: 1e-9, 6: 1e-9, 8: 1e-9, 16: 1e-9}), self.features(), RULES)
        q = with_overrides(p, density_cutoff_factor=12.0)
        self.assertEqual(q.density_cutoff_factor, 12.0)
        self.assertEqual(q.mode, 'auto')

    def test_shipped_rules_load(self):
        self.assertEqual(load_rules()['schema_version'], 'sparse-dft-sparsity-rules-1')

    def test_shipped_rules_keep_padding_in_vacuum(self):
        # Al(001) slab hold-out: f6 + padding 1 failed the force gate.
        probe = {'species': [{'record_id': 'Al:standard', 'zval': 3.0, 'tails': [
            {'factor': f, 'weighted_compensation_l2': 1e-9} for f in (4, 5, 6, 8, 10, 12, 16)]}]}
        slab = system_features(probe, natoms=5, smearing_ev=0.1, spin_channels=1, vacuum_fraction=0.6)
        bulk = system_features(probe, natoms=1, smearing_ev=0.1, spin_channels=1, vacuum_fraction=0.0)
        self.assertEqual(resolve_auto(probe, slab).xc_padding, 2)
        self.assertEqual(resolve_auto(probe, bulk).xc_padding, 1)
        self.assertEqual(resolve_auto(probe, slab).density_cutoff_factor, 6.0)


class FeatureTest(unittest.TestCase):
    def test_vacuum_fraction(self):
        bulk = Structure(species=['Si', 'Si'], positions=[(0, 0, 0), (.25, .25, .25)],
                         lattice=[[0, 2.715, 2.715], [2.715, 0, 2.715], [2.715, 2.715, 0]], fractional=True)
        self.assertEqual(vacuum_fraction(bulk), 0.0)
        slab = Structure(species=['Al', 'Al'], positions=[(0, 0, 0), (.5, .5, .1)],
                         lattice=[[2.86, 0, 0], [0, 2.86, 0], [0, 0, 20.0]], fractional=True)
        self.assertAlmostEqual(vacuum_fraction(slab), 0.9)

    def test_ledger_reads_counters(self):
        with tempfile.TemporaryDirectory() as d:
            perf = Path(d) / 'p.json'
            perf.write_text(json.dumps({'counters': {'scf_total': {'seconds': 1.5, 'calls': 1},
                                                     'unrelated': {'seconds': 9, 'calls': 1}}}))
            ledger = sparsity_ledger({'cutoff_ev': 500, 'density_cutoff_ev': 3000, 'iterations': 7},
                                     coerce_profile({'density_cutoff_factor': 6}), perf)
            self.assertEqual(ledger['density_cutoff_factor'], 6.0)
            self.assertEqual(set(ledger['counters']), {'scf_total'})


class AutoControlTest(unittest.TestCase):
    TAILS = {4: 1e-6, 6: 5e-8, 8: 5e-9, 16: 1e-10}

    def features(self, **kw):
        base = dict(natoms=2, smearing_ev=0.0, spin_channels=1, vacuum_fraction=0.0)
        base.update(kw)
        return system_features(_probe(self.TAILS), **base)

    def resolve(self, tails=None, **controls):
        return resolve_auto(_probe(tails or self.TAILS), self.features(), RULES, controls=controls)

    def test_missing_probe_factor_is_an_error_not_a_zero_tail(self):
        # Before v16 a candidate absent from the probe read as tail 0 and won.
        with self.assertRaises(InputError):
            resolve_auto(_probe({6: 5e-8, 8: 5e-9, 16: 1e-10}), self.features(), RULES)

    def test_levels_move_the_budget(self):
        self.assertEqual(self.resolve(level='balanced').density_cutoff_factor, 6.0)
        # conservative: 1e-7 / 4 = 2.5e-8 < 5e-8 at f6 -> f8.
        self.assertEqual(self.resolve(level='conservative').density_cutoff_factor, 8.0)
        p = self.resolve(level='aggressive', tail_budget=2e-6)
        self.assertEqual(p.density_cutoff_factor, 4.0)
        self.assertEqual(p.estimates['density_tail_budget_source'], 'user')

    def test_candidates_and_range(self):
        self.assertEqual(self.resolve(candidates=[8, 16]).density_cutoff_factor, 8.0)
        self.assertEqual(self.resolve(min_factor=8).density_cutoff_factor, 8.0)
        with self.assertRaises(InputError):
            self.resolve(min_factor=20)  # no candidate in [20, 24]
        with self.assertRaises(InputError):
            self.resolve(candidates=[7])  # not reported by this probe

    def test_max_factor_caps_the_fallback_and_warns(self):
        bad = {4: 1e-3, 6: 1e-3, 8: 1e-3, 16: 1e-3}
        p = self.resolve(tails=bad, max_factor=8)
        self.assertEqual(p.density_cutoff_factor, 8.0)
        self.assertTrue(p.estimates['density_fallback'])
        self.assertIn('below the rule fallback', p.estimates['warnings'][0])

    def test_metric_change_needs_explicit_budget(self):
        with self.assertRaises(InputError):
            self.resolve(metric='weighted_compensation_l2')

    def test_controls_are_validated_and_auto_only(self):
        for bad in ({'level': 'extreme'}, {'tail_budget': 0}, {'min_factor': 30}, {'candidates': [2]},
                    {'min_factor': 12, 'max_factor': 8}, {'bogus': 1}):
            with self.assertRaises(InputError):
                coerce_profile({'mode': 'auto', 'auto_controls': bad})
        with self.assertRaises(InputError):
            coerce_profile({'mode': 'custom', 'xc_padding': 1, 'auto_controls': {'level': 'balanced'}})

    def test_density_factor_range_is_the_probe_range(self):
        coerce_profile({'density_cutoff_factor': 24})
        with self.assertRaises(InputError):
            coerce_profile({'density_cutoff_factor': 32})

    def test_probe_environment(self):
        from sparse_dft.sparsity import PROBE_FACTORS, probe_environment
        self.assertNotIn('SDFT_SPARSITY_PROBE_FACTORS', probe_environment(PROBE_FACTORS, {}))
        self.assertEqual(probe_environment([7, 4, 16], {})['SDFT_SPARSITY_PROBE_FACTORS'], '4,7,16')
        with self.assertRaises(InputError):
            probe_environment([4, 30], {})
        with self.assertRaises(InputError):
            probe_environment([4, 7], {'SDFT_SPARSITY_PROBE_FACTORS': '4,8'})

    def test_real_search_projectors_are_accepted(self):
        env = profile_environment(coerce_profile({'projector_space': 'real-search',
                                                  'projector_radius_factor': 1.5}), {})
        self.assertEqual(env['SDFT_PROJECTOR_SPACE'], 'real-search')
        self.assertEqual(env['SDFT_PROJECTOR_RADIUS_FACTOR'], '1.5')

    def test_rules_file_is_checked(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'rules.json'
            rules = load_rules()
            rules['density_cutoff']['fallback_factor'] = 32
            path.write_text(json.dumps(rules))
            with self.assertRaises(InputError):
                load_rules(path)
            with self.assertRaises(InputError):
                load_rules(Path(d) / 'missing.json')


class SolverControlTest(unittest.TestCase):
    def test_mapping_and_ranges(self):
        from sparse_dft.sparsity import solver_environment, validate_solver_controls
        env = solver_environment({'davidson_subspace_factor': 12, 'search_operator_floor': 1e-5,
                                  'ritz_solver': 'qr', 'incremental_ritz': 0, 'warm_guard_vectors': 10,
                                  'fast_density_grid': 1, 'scf_acceptance_margin': 0.5}, {})
        self.assertEqual(env['SDFT_DAVIDSON_SUBSPACE_FACTOR'], '12')
        self.assertEqual(env['SDFT_SEARCH_OPERATOR_FLOOR'], '1e-05')
        self.assertEqual(env['SDFT_RITZ_SOLVER'], 'qr')
        self.assertEqual(env['SDFT_INCREMENTAL_RITZ'], '0')
        self.assertEqual(env['SDFT_WARM_GUARD_VECTORS'], '10')
        self.assertEqual(env['SDFT_FAST_DENSITY_GRID'], '1')
        self.assertEqual(env['SDFT_SCF_ACCEPTANCE_MARGIN'], '0.5')
        self.assertEqual(solver_environment(None, {'PATH': '/bin'}), {'PATH': '/bin'})
        for bad in ({'davidson_subspace_factor': 3}, {'davidson_subspace_factor': 17},
                    {'search_operator_floor': 0}, {'search_operator_floor': 0.1},
                    {'ritz_solver': 'jacobi'}, {'warm_guard_vectors': 0},
                    {'scf_acceptance_margin': 1.5}, {'unknown_knob': 1}):
            with self.assertRaises(InputError):
                validate_solver_controls(bad)
        with self.assertRaises(InputError):
            solver_environment({'ritz_solver': 'qr'}, {'SDFT_RITZ_SOLVER': 'subset'})


if __name__ == '__main__':
    unittest.main()
