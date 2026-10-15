import os
import unittest
from unittest import mock

from sparse_dft.errors import InputError
from sparse_dft.parallel import ParallelLayout, coerce_layout, launch_command, layout_environment


class ParallelLayoutTest(unittest.TestCase):
    def test_single_rank_is_the_plain_command(self):
        self.assertEqual(launch_command(['sdft_scf', 'scf', 'a', 'b'], ParallelLayout()), ['sdft_scf', 'scf', 'a', 'b'])
        base = {'PATH': '/bin'}
        self.assertEqual(layout_environment(ParallelLayout(), base), base)

    def test_groups_must_factor_the_rank_count(self):
        self.assertEqual(coerce_layout({'ranks': 4, 'bandgroups': 2}).kgroups, 2)
        for bad in ({'ranks': 4, 'bandgroups': 3}, {'ranks': 4, 'kgroups': 3}, {'ranks': 0}, {'ranks': 2, 'x': 1}):
            with self.assertRaises(InputError):
                coerce_layout(bad)

    def test_launcher_template_and_environment(self):
        layout = coerce_layout({'ranks': 2, 'bandgroups': 2})
        with mock.patch.dict(os.environ, {'SDFT_MPI_LAUNCHER': '/bin/echo -n {ranks}'}):
            self.assertEqual(launch_command(['exe'], layout), ['/bin/echo', '-n', '2', 'exe'])
        env = layout_environment(layout, {'OMP_NUM_THREADS': '8'})
        self.assertEqual((env['OMP_NUM_THREADS'], env['SDFT_KGROUPS'], env['SDFT_BANDGROUPS']), ('4', '1', '2'))
        self.assertEqual(env['PRTE_MCA_hwloc_default_binding_policy'], 'none')

    def test_deterministic_switch(self):
        env = layout_environment(coerce_layout({'deterministic': True}), {})
        self.assertEqual(env, {'SDFT_DETERMINISTIC': '1'})

    def test_missing_launcher_is_an_error(self):
        with mock.patch.dict(os.environ, {'SDFT_MPI_LAUNCHER': 'no-such-mpiexec-xyz -n {ranks}'}):
            with self.assertRaises(InputError):
                launch_command(['exe'], coerce_layout({'ranks': 2}))

    def test_cli_exposes_mpi_options(self):
        from sparse_dft.cli import build_parser
        a = build_parser().parse_args(['scf', 'x.vasp', '--paw-root', 'p', '--mpi-ranks', '4', '--bandgroups', '2'])
        from sparse_dft.cli import _cli_parallel
        self.assertEqual(_cli_parallel(a).as_dict(), {'ranks': 4, 'kgroups': 2, 'bandgroups': 2, 'threads_per_rank': None, 'deterministic': False})


if __name__ == '__main__':
    unittest.main()
