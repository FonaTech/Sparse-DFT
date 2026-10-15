"""Typed QSettings wrapper holding the user preferences."""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QSettings

DEFAULTS = {
    'language': 'auto', 'theme': 'light', 'font_px': 13,
    'scf_exe': '', 'paw_root': '', 'output_root': str(Path.home() / 'Sparse-DFT' / 'runs'),
    'omp_threads': 0, 'mpi_launcher': '', 'mpi_ranks': 0,
    'extra_env': 'SDFT_ENABLE_PAW_CLOSURE_DIAGNOSTIC=1',
    'plot_width': 'single', 'plot_lang': 'en', 'plot_palette': 'academic',
    'plot_formats': 'pdf,png', 'plot_dpi': 600,
    # Structure page: apply sparse_dft.presets recommendations right after a file is
    # inspected (1) or only from the "Apply recommended" button (0).
    'auto_apply_presets': 0, 'preset_accuracy': 'standard',
}
# Commands whose --out is a directory that defaults under the output root.
OUTPUT_COMMANDS = ('scf', 'bands', 'dos', 'relax', 'prepare', 'bands-analysis')


class Preferences:
    def __init__(self, settings: QSettings | None = None):
        self.qsettings = settings or QSettings('Sparse-DFT', 'Sparse-DFT Studio')

    def get(self, key):
        default = DEFAULTS.get(key)
        value = self.qsettings.value(key, default)
        if isinstance(default, int) and not isinstance(default, bool):
            try:
                return int(value)
            except (TypeError, ValueError):
                return default
        return '' if value is None else str(value)

    def set(self, key, value):
        self.qsettings.setValue(key, value)
        self.qsettings.sync()

    def extra_env(self) -> dict:
        env = {}
        for line in self.get('extra_env').splitlines():
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, value = line.split('=', 1)
            if key.strip():
                env[key.strip()] = value.strip()
        return env

    def job_env(self) -> dict:
        env = self.extra_env()
        if self.get('omp_threads') > 0:
            env['OMP_NUM_THREADS'] = str(self.get('omp_threads'))
        if self.get('scf_exe'):
            env['SDFT_SCF_EXE'] = self.get('scf_exe')
        if self.get('mpi_launcher'):
            env['SDFT_MPI_LAUNCHER'] = self.get('mpi_launcher')
        return env

    def output_root(self) -> Path:
        return Path(self.get('output_root') or DEFAULTS['output_root']).expanduser()

    def form_defaults(self, command: str) -> dict:
        out = {}
        paw = self.get('paw_root')
        if paw:
            out['paw_root'] = paw
            out['root'] = paw
        root = self.get('output_root')
        if root and command in OUTPUT_COMMANDS:
            out['out'] = str(self.output_root() / command)
        if self.get('mpi_ranks') > 0:
            out['mpi_ranks'] = str(self.get('mpi_ranks'))
        return out
