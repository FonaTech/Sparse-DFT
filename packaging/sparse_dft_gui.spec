# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the Sparse-DFT GUI on Windows, macOS and Linux.

Build from the repository root after building the C++ core:

    pyinstaller packaging/sparse_dft_gui.spec --noconfirm --clean
    python packaging/make_manifest.py dist/Sparse-DFT --write   # then --check on the target machine

Native executables are taken from bin/<platform_tag>/ when present, else
from build/ (Release/RelWithDebInfo/Debug).  data/paw/ is bundled only if it
contains a PAW root the distributor may redistribute (never VASP POTCARs).
The frozen executable runs workers as ``<exe> --worker JOB.json``.
Environment: SDFT_BUNDLE_ONEFILE=1 builds a single file (slower start-up);
SDFT_BUNDLE_CONSOLE=1 keeps a console window for debugging.
"""
import os
import sys
from pathlib import Path

ROOT = Path(SPECPATH).resolve().parent  # noqa: F821 - SPECPATH is defined by PyInstaller
sys.path.insert(0, str(ROOT))
from sparse_dft.resources import BUILD_SUBDIRS, platform_tag  # noqa: E402

WINDOWS = sys.platform.startswith('win')
MACOS = sys.platform == 'darwin'
TAG = platform_tag()
EXE_SUFFIX = '.exe' if WINDOWS else ''
ONEFILE = os.environ.get('SDFT_BUNDLE_ONEFILE') == '1'
CONSOLE = os.environ.get('SDFT_BUNDLE_CONSOLE') == '1'
NATIVE = ('sdft_scf', 'sdft_relax_step')
LIB_PATTERNS = ('*.dll',) if WINDOWS else (('*.dylib',) if MACOS else ('*.so', '*.so.*'))


def native_binaries():
    """(source, destination) pairs: bin/<tag>/ first, else the CMake build tree."""
    bundled = ROOT / 'bin' / TAG
    candidates = [bundled] + [ROOT.joinpath(*sub.split('/')) for sub in BUILD_SUBDIRS]
    for directory in candidates:
        found = [directory / (name + EXE_SUFFIX) for name in NATIVE]
        if all(path.is_file() for path in found):
            libraries = [p for pattern in LIB_PATTERNS for p in sorted(directory.glob(pattern))] \
                if directory == bundled else []
            return [(str(p), f'bin/{TAG}') for p in found + libraries]
    raise SystemExit(f'sdft_scf/sdft_relax_step{EXE_SUFFIX} not found in bin/{TAG}/ or build/; build the C++ core first')


def data_files():
    out = [(str(ROOT / 'sparse_dft' / 'sparsity_rules.json'), 'sparse_dft'),
           (str(ROOT / 'sparse_dft_gui' / 'i18n'), 'sparse_dft_gui/i18n'),
           (str(ROOT / 'sparse_dft_gui' / 'assets'), 'sparse_dft_gui/assets'),
           (str(ROOT / 'packaging' / 'layout.json'), 'packaging')]
    for name in ('paw', 'fonts'):
        directory = ROOT / 'data' / name
        if directory.is_dir():
            out.append((str(directory), f'data/{name}'))
    return out


def refuse_potcar():
    """Fail the build if anything that looks like raw VASP POTCAR data would be bundled."""
    paw = ROOT / 'data' / 'paw'
    if not paw.is_dir():
        return
    offenders = [p for p in paw.rglob('*') if p.is_file() and 'POTCAR' in p.name.upper()]
    if offenders:
        raise SystemExit(f'refusing to bundle POTCAR files: {offenders[:5]}')


refuse_potcar()
hidden = ['sparse_dft.runner', 'sparse_dft.cli', 'sparse_dft.api', 'sparse_dft_gui.pages.analysis', 'sparse_dft_gui.pages.commands',
          'sparse_dft_gui.pages.settings', 'sparse_dft_gui.pages.sparsity', 'sparse_dft_gui.pages.structure',
          'sparse_dft_gui.pages.system', 'matplotlib.backends.backend_qtagg', 'matplotlib.backends.backend_agg',
          'matplotlib.backends.backend_pdf', 'matplotlib.backends.backend_svg']
# Optional modules: listed only when importable so a missing plotting/analysis package is not fatal.
for optional in ('sparse_dft.plotting', 'sparse_dft.plotting.figures', 'sparse_dft.plotting.export', 'sparse_dft.plotting.cli',
                 'sparse_dft.analysis', 'sparse_dft.parallel'):
    try:
        __import__(optional)
        hidden.append(optional)
    except Exception:  # noqa: BLE001
        pass

a = Analysis(  # noqa: F821
    [str(ROOT / 'packaging' / 'sparse_dft_gui_entry.py')],
    pathex=[str(ROOT)],
    binaries=native_binaries(),
    datas=data_files(),
    hiddenimports=hidden,
    excludes=['tkinter', 'PyQt5', 'PySide2', 'PySide6', 'IPython', 'pytest'],
    noarchive=False,
)
pyz = PYZ(a.pure)  # noqa: F821
icon = None
for candidate in ('sparse_dft.ico' if WINDOWS else 'sparse_dft.icns' if MACOS else 'sparse_dft.png',):
    path = ROOT / 'packaging' / candidate
    icon = str(path) if path.is_file() else None

exe_kwargs = dict(name='sparse-dft-gui', debug=False, strip=False, upx=False, console=CONSOLE,
                  disable_windowed_traceback=False, icon=icon,
                  argv_emulation=False, target_arch=None, codesign_identity=None, entitlements_file=None)
if WINDOWS:
    version_file = ROOT / 'packaging' / 'version_info.txt'
    exe_kwargs['version'] = str(version_file) if version_file.is_file() else None

# Console-subsystem twin used for job workers (sparse_dft.runner.frozen_worker_executable):
# native console programs started by a worker inherit its hidden console instead
# of opening windows (Windows), and its std streams are always real pipes.
worker_kwargs = dict(exe_kwargs, name='sparse-dft-worker', console=True)
if ONEFILE:
    exe = EXE(pyz, a.scripts, a.binaries, a.datas, [], **exe_kwargs)  # noqa: F821
    # One-file builds keep a single executable; workers run as "sparse-dft-gui --worker".
else:
    exe = EXE(pyz, a.scripts, [], exclude_binaries=True, **exe_kwargs)  # noqa: F821
    worker_exe = EXE(pyz, a.scripts, [], exclude_binaries=True, **worker_kwargs)  # noqa: F821
    coll = COLLECT(exe, worker_exe, a.binaries, a.datas, strip=False, upx=False, name='Sparse-DFT')  # noqa: F821
    if MACOS:
        app = BUNDLE(coll, name='Sparse-DFT.app', icon=icon,  # noqa: F821
                     bundle_identifier='org.sparsedft.studio',
                     info_plist={'CFBundleShortVersionString': '0.1.0', 'NSHighResolutionCapable': True,
                                 'LSMinimumSystemVersion': '12.0',
                                 'CFBundleDevelopmentRegion': 'en',
                                 'CFBundleLocalizations': ['en', 'zh_CN', 'zh_TW', 'ja']})
