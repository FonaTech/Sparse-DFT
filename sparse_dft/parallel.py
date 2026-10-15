"""Process-level parallel launch for the native solver (MPI k groups and
band groups) and the per-rank OpenMP budget.

A ``ParallelLayout`` with one rank launches the executable directly, exactly
as before. With ``ranks > 1`` the command is prefixed by an MPI launcher
(``SDFT_MPI_LAUNCHER``, default ``mpiexec -n {ranks}``; MS-MPI's
``mpiexec`` uses the same syntax). The executable must be built with
``-DSDFT_ENABLE_MPI=ON``; a serial build refuses ``SDFT_KGROUPS``/
``SDFT_BANDGROUPS`` instead of silently running serially.
"""
from __future__ import annotations

import os
import shlex
import shutil
from dataclasses import dataclass

from .errors import InputError

DEFAULT_LAUNCHER = 'mpiexec -n {ranks}'


@dataclass(frozen=True)
class ParallelLayout:
    ranks: int = 1
    kgroups: int | None = None
    bandgroups: int = 1
    threads_per_rank: int | None = None
    # SDFT_DETERMINISTIC=1: historical summation order, one-vector CGS2,
    # full PAW compensation scope, FFTW_MEASURE and single-level k teams.
    deterministic: bool = False

    def validated(self) -> 'ParallelLayout':
        for name in ('ranks', 'bandgroups'):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise InputError(f'{name} must be a positive integer', details={name: value})
        if self.ranks % self.bandgroups:
            raise InputError('bandgroups must divide the MPI rank count',
                             details={'ranks': self.ranks, 'bandgroups': self.bandgroups})
        kgroups = self.kgroups if self.kgroups is not None else self.ranks // self.bandgroups
        if kgroups * self.bandgroups != self.ranks:
            raise InputError('kgroups x bandgroups must equal the MPI rank count',
                             details={'ranks': self.ranks, 'kgroups': kgroups, 'bandgroups': self.bandgroups})
        if self.threads_per_rank is not None and self.threads_per_rank < 1:
            raise InputError('threads_per_rank must be a positive integer')
        return ParallelLayout(self.ranks, kgroups, self.bandgroups, self.threads_per_rank, bool(self.deterministic))

    def as_dict(self) -> dict:
        return {'ranks': self.ranks, 'kgroups': self.kgroups, 'bandgroups': self.bandgroups,
                'threads_per_rank': self.threads_per_rank, 'deterministic': self.deterministic}


def coerce_layout(value) -> ParallelLayout:
    if value is None:
        return ParallelLayout()
    if isinstance(value, ParallelLayout):
        return value.validated()
    if isinstance(value, dict):
        allowed = {'ranks', 'kgroups', 'bandgroups', 'threads_per_rank', 'deterministic'}
        unknown = set(value) - allowed
        if unknown:
            raise InputError('unknown parallel layout keys', details={'keys': sorted(unknown)})
        return ParallelLayout(**{k: value[k] for k in allowed if value.get(k) is not None}).validated()
    raise InputError('parallel layout must be a dict or ParallelLayout')


def _threads_available(env: dict) -> int:
    configured = env.get('OMP_NUM_THREADS')
    if configured and configured.isdigit() and int(configured) > 0:
        return int(configured)
    return os.cpu_count() or 1


def launch_command(executable_args: list[str], layout: ParallelLayout) -> list[str]:
    """Full argv: the executable alone, or prefixed by the MPI launcher."""
    layout = layout.validated()
    if layout.ranks == 1:
        return list(executable_args)
    template = os.environ.get('SDFT_MPI_LAUNCHER', DEFAULT_LAUNCHER)
    # Windows paths in the template keep their backslashes.
    prefix = [part.format(ranks=layout.ranks) for part in shlex.split(template, posix=os.name != 'nt')]
    if not prefix or shutil.which(prefix[0]) is None and not os.path.isfile(prefix[0]):
        raise InputError('MPI launcher not found; install Open MPI/MPICH/MS-MPI or set SDFT_MPI_LAUNCHER',
                         details={'launcher': template})
    if '{ranks}' not in template and str(layout.ranks) not in prefix:
        raise InputError('SDFT_MPI_LAUNCHER must contain {ranks}', details={'launcher': template})
    return prefix + list(executable_args)


def layout_environment(layout: ParallelLayout, base: dict | None = None) -> dict:
    """Per-rank OpenMP budget and the group layout read by the native code."""
    env = dict(os.environ if base is None else base)
    layout = layout.validated()
    if layout.deterministic:
        env['SDFT_DETERMINISTIC'] = '1'
    if layout.ranks == 1 and layout.threads_per_rank is None:
        return env
    threads = layout.threads_per_rank or max(1, _threads_available(env) // layout.ranks)
    env['OMP_NUM_THREADS'] = str(threads)
    if layout.ranks > 1:
        env['SDFT_KGROUPS'] = str(layout.kgroups)
        env['SDFT_BANDGROUPS'] = str(layout.bandgroups)
        # Open MPI on Linux binds each rank to one core by default, which
        # pins all OpenMP threads of a rank there; let the OS place threads.
        # PRTE_* is read by Open MPI 5 (verified: "Rank 0 is not bound"),
        # OMPI_* by Open MPI 4. Other MPI implementations ignore both.
        env.setdefault('PRTE_MCA_hwloc_default_binding_policy', 'none')
        env.setdefault('OMPI_MCA_hwloc_base_binding_policy', 'none')
    return env
