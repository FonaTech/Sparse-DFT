"""Guarded adapter for the publication plotting API (``sparse_dft.plotting``).

The GUI keeps working when the module is missing or raises; every call
returns a value or raises :class:`PlottingUnavailable` with a reason.
"""
from __future__ import annotations

import importlib
import threading
from pathlib import Path

WIDTHS = ('single', 'double')
LANGS = ('en', 'zh_CN', 'zh_TW', 'ja')
FALLBACK_PALETTES = ('academic', 'okabe_ito', 'macaron')
FORMATS = ('pdf', 'svg', 'png', 'tiff')


# matplotlib swaps global rcParams inside rc_context: figures are built, drawn and
# saved one at a time across the GUI's worker threads.
PLOT_LOCK = threading.RLock()


class PlottingUnavailable(RuntimeError):
    pass


def _modules():
    try:
        figures = importlib.import_module('sparse_dft.plotting.figures')
    except Exception as error:  # noqa: BLE001 - missing or broken optional module
        raise PlottingUnavailable(f'sparse_dft.plotting.figures: {error}') from error
    try:
        export = importlib.import_module('sparse_dft.plotting.export')
    except Exception:  # noqa: BLE001
        export = None
    return figures, export


def available() -> bool:
    try:
        _modules()
        return True
    except PlottingUnavailable:
        return False


def kinds_for(path) -> list[str]:
    figures, _ = _modules()
    return list(figures.available_kinds(Path(path)))


def kind_info(kind: str) -> dict:
    try:
        figures, _ = _modules()
        return dict(getattr(figures, 'KINDS', {}).get(kind, {}))
    except PlottingUnavailable:
        return {}


_PALETTES: list[str] | None = None


def palettes() -> list[str]:
    """Palette names offered by the plotting package (cached; it imports matplotlib)."""
    global _PALETTES
    if _PALETTES is None:
        _PALETTES = _find_palettes()
    return list(_PALETTES)


def _find_palettes() -> list[str]:
    for name in ('sparse_dft.plotting.palettes', 'sparse_dft.plotting.style', 'sparse_dft.plotting.figures'):
        try:
            module = importlib.import_module(name)
        except Exception:  # noqa: BLE001
            continue
        for attr in ('PALETTES', 'palettes'):
            value = getattr(module, attr, None)
            if callable(value):
                try:
                    value = value()
                except Exception:  # noqa: BLE001
                    value = None
            if isinstance(value, dict) and value:
                return list(value)
            if isinstance(value, (list, tuple)) and value:
                return [str(v) for v in value]
    return list(FALLBACK_PALETTES)


def make_figure(kind: str, source, *, width='single', lang='en', palette='academic', **options):
    """A ``matplotlib.figure.Figure`` from ``source`` (path) for ``kind``."""
    figures, _ = _modules()
    with PLOT_LOCK:
        return figures.make_figure(kind, [Path(source)], width=width, lang=lang, palette=palette, **options)


def is_multi_panel(kind: str) -> bool:
    try:
        figures, _ = _modules()
    except PlottingUnavailable:
        return False
    return kind in getattr(figures, 'MULTI_PANEL', ())


def make_figures(kind: str, source, *, panels='both', width='single', lang='en', palette='academic',
                 **options) -> dict:
    """{'combined': fig, <panel>: fig, ...} for multi-panel kinds; {'combined': fig} otherwise."""
    figures, _ = _modules()
    if hasattr(figures, 'make_figures') and kind in getattr(figures, 'MULTI_PANEL', ()):
        with PLOT_LOCK:
            return dict(figures.make_figures(kind, [Path(source)], panels=panels, lang=lang, palette=palette,
                                             **options))
    return {'combined': make_figure(kind, source, width=width, lang=lang, palette=palette, **options)}


def export_figures(figs: dict, stem, formats=('pdf', 'png'), *, dpi=600, provenance=None) -> list[Path]:
    """'combined' to ``stem``, each panel to ``stem_<panel>``; the flat list of written paths."""
    out = []
    stem = Path(stem)
    for name, fig in figs.items():
        target = stem if name == 'combined' else stem.with_name(f'{stem.name}_{name}')
        out += export(fig, target, formats, dpi=dpi, provenance={**(provenance or {}), 'panel': name})
    return out


def export(fig, stem, formats=('pdf', 'png'), *, dpi=600, provenance=None) -> list[Path]:
    _, module = _modules()
    if module is not None and hasattr(module, 'export'):
        with PLOT_LOCK:
            return [Path(p) for p in module.export(fig, stem, formats=tuple(formats), dpi=dpi,
                                                   provenance=provenance)]
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for fmt in formats:
        path = stem.with_suffix('.' + fmt)
        fig.savefig(path, dpi=dpi if fmt in ('png', 'tiff') else None)
        out.append(path)
    return out


def preview_source(path) -> Path:
    """Directory to search for figures: a result file's folder, or the folder itself."""
    path = Path(path)
    return path if path.is_dir() else path.parent
