"""Locate project-local data, bundled executables and per-user directories.

The same lookups serve a source checkout, an installed wheel and a frozen
(PyInstaller) application.  The frozen layout is documented in
``packaging/layout.json``::

    <resource_root>/bin/<platform_tag>/sdft_scf[.exe]
    <resource_root>/bin/<platform_tag>/sdft_relax_step[.exe]
    <resource_root>/data/paw/        (reserved; users import licensed PAW sets)
    <resource_root>/data/fonts/      (optional .ttf/.otf loaded by the GUI)
    <resource_root>/sparse_dft_gui/i18n/   (JSON translation catalogs)
"""
from __future__ import annotations

import os
import platform
import shutil
import sys
from pathlib import Path

from .errors import InputError

_REPO_ROOT = Path(__file__).resolve().parents[1]
# Build-tree subdirectories searched in order (single- and multi-config CMake).
BUILD_SUBDIRS = ('build', 'build/Release', 'build/RelWithDebInfo', 'build/Debug')
# Historical names of the per-executable overrides.
_ENV_OVERRIDES = {'sdft_scf': 'SDFT_SCF_EXE', 'sdft_relax_step': 'SDFT_RELAX_STEP_EXE'}


def project_paw_archive():
    """An explicit override must be valid; never silently skip PAW tests."""
    configured = os.environ.get('SDFT_PAW_ARCHIVE')
    root = Path(configured).expanduser() if configured else (
        Path(__file__).resolve().parents[1] / 'data/paw/potpaw_PBE.54-json')
    missing = [name for name in ('manifest.json', 'index.json', 'potentials.jsonl.xz')
               if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f'PAW archive {root} is missing {missing}; '
                                'convert a POTCAR set with paw-convert or set SDFT_PAW_ARCHIVE')
    return root


def is_frozen() -> bool:
    """True inside a PyInstaller (or similar) frozen application."""
    return bool(getattr(sys, 'frozen', False))


def resource_root() -> Path:
    """Root of bundled resources: ``sys._MEIPASS`` when frozen, else the repo root."""
    if is_frozen():
        meipass = getattr(sys, '_MEIPASS', None)
        return Path(meipass) if meipass else Path(sys.executable).resolve().parent
    return _REPO_ROOT


def _is_windows() -> bool:
    return sys.platform.startswith('win')


def _is_macos() -> bool:
    return sys.platform == 'darwin'


def platform_tag() -> str:
    """``<os>-<arch>``, e.g. ``macos-arm64``, ``windows-x86_64``, ``linux-x86_64``."""
    system = 'windows' if _is_windows() else 'macos' if _is_macos() else (
        'linux' if sys.platform.startswith('linux') else sys.platform)
    machine = platform.machine().lower()
    arch = {'amd64': 'x86_64', 'x64': 'x86_64', 'x86-64': 'x86_64', 'aarch64': 'arm64',
            'arm64': 'arm64', 'i386': 'x86', 'i686': 'x86'}.get(machine, machine or 'unknown')
    return f'{system}-{arch}'


def executable_name(name: str) -> str:
    """Platform file name of an executable (``.exe`` appended on Windows)."""
    if _is_windows() and not name.lower().endswith('.exe'):
        return name + '.exe'
    return name


def _env_override_name(name: str) -> str:
    stem = name[:-4] if name.lower().endswith('.exe') else name
    return _ENV_OVERRIDES.get(stem, 'SDFT_' + stem.upper().replace('-', '_') + '_EXE')


def executable_candidates(name: str) -> list[tuple[str, Path]]:
    """Ordered ``(origin, path)`` locations searched by :func:`find_executable`."""
    stem = name[:-4] if name.lower().endswith('.exe') else name
    filename = executable_name(stem)
    out: list[tuple[str, Path]] = []
    root = resource_root()
    out.append(('bundled', root / 'bin' / platform_tag() / filename))
    if is_frozen():
        # One-folder builds may also place helpers next to the launcher.
        out.append(('bundled', Path(sys.executable).resolve().parent / filename))
    if stem != 'sdft_scf':
        configured = os.environ.get('SDFT_SCF_EXE')
        if configured:
            out.append(('sibling of SDFT_SCF_EXE', Path(configured).expanduser().with_name(filename)))
    for sub in BUILD_SUBDIRS:
        out.append(('build', _REPO_ROOT.joinpath(*sub.split('/')) / filename))
    return out


def find_executable(name: str) -> Path:
    """Resolve a native executable such as ``sdft_scf`` or ``sdft_relax_step``.

    Order: environment override (``SDFT_SCF_EXE``, ``SDFT_RELAX_STEP_EXE`` or
    generic ``SDFT_<NAME>_EXE``) -> bundled ``bin/<platform_tag>/`` -> sibling
    of the configured ``sdft_scf`` -> ``build/``, ``build/Release``,
    ``build/RelWithDebInfo``, ``build/Debug`` -> ``PATH``.  An explicit
    environment override is authoritative: if it names a missing file the
    lookup fails instead of silently using another binary.
    """
    stem = name[:-4] if name.lower().endswith('.exe') else name
    variable = _env_override_name(stem)
    configured = os.environ.get(variable)
    if configured:
        path = Path(configured).expanduser()
        if path.is_file():
            return path
        if _is_windows() and not path.suffix and path.with_suffix('.exe').is_file():
            return path.with_suffix('.exe')
        raise InputError(f'{variable} points to a missing executable: {path}',
                         details={'variable': variable, 'searched': [str(path)]})
    searched = []
    for _origin, candidate in executable_candidates(stem):
        searched.append(str(candidate))
        if candidate.is_file():
            return candidate
    found = shutil.which(executable_name(stem))
    if found:
        return Path(found)
    searched.append(f'PATH ({executable_name(stem)})')
    raise InputError(f'{executable_name(stem)} executable is not available; build the C++ core '
                     f'or set {variable}. Searched: ' + '; '.join(searched),
                     details={'variable': variable, 'searched': searched})


def user_data_dir() -> Path:
    """Per-user writable data directory (created lazily by callers)."""
    override = os.environ.get('SDFT_USER_DATA_DIR')
    if override:
        return Path(override).expanduser()
    home = Path(os.path.expanduser('~'))
    if _is_windows():
        appdata = os.environ.get('APPDATA')
        return (Path(appdata) if appdata else home / 'AppData' / 'Roaming') / 'Sparse-DFT'
    if _is_macos():
        return home / 'Library' / 'Application Support' / 'Sparse-DFT'
    xdg = os.environ.get('XDG_DATA_HOME')
    return (Path(xdg) if xdg else home / '.local' / 'share') / 'sparse_dft'


def paw_search_paths() -> list[Path]:
    """PAW roots in priority order: user ``paw/`` -> bundled ``data/paw/`` -> ``SDFT_PAW_ROOT``."""
    paths = [user_data_dir() / 'paw', resource_root() / 'data' / 'paw']
    configured = os.environ.get('SDFT_PAW_ROOT')
    if configured:
        paths.append(Path(configured).expanduser())
    return paths


def fonts_dir() -> Path:
    """Optional bundled fonts (``.ttf``/``.otf``) loaded by the GUI."""
    return resource_root() / 'data' / 'fonts'


def translations_dir() -> Path:
    """JSON translation catalogs of the GUI."""
    return resource_root() / 'sparse_dft_gui' / 'i18n'
