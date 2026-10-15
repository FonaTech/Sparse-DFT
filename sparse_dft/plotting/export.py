"""Export figures with a provenance record next to them."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import logging
import platform

import matplotlib
from matplotlib.backend_bases import FigureCanvasBase

from ..analysis.common import sha256_file
from .style import style_of, use_spec

FORMATS = {'pdf': 'pdf', 'svg': 'svg', 'png': 'png', 'tif': 'tiff', 'tiff': 'tiff', 'eps': 'eps'}
RASTER = {'png', 'tiff'}
PROVENANCE_SCHEMA = 'sparse-dft-figure-provenance-1'


def normalize_formats(formats) -> list[str]:
    if isinstance(formats, str):
        formats = [f for f in formats.replace(';', ',').split(',')]
    out = []
    for f in formats:
        key = str(f).strip().lower().lstrip('.')
        if not key:
            continue
        if key not in FORMATS:
            raise ValueError(f'unsupported figure format {f!r}; choose from {sorted(FORMATS)}')
        if FORMATS[key] not in out:
            out.append(FORMATS[key])
    if not out:
        raise ValueError('no figure format requested')
    return out


def provenance_path(stem) -> Path:
    return _stem(stem).with_name(_stem(stem).name + '.provenance.json')


def _stem(stem) -> Path:
    p = Path(stem)
    if p.suffix.lower().lstrip('.') in FORMATS:
        p = p.with_suffix('')
    return p


def _ensure_canvas(fig):
    if fig.canvas is None or type(fig.canvas) is FigureCanvasBase:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        FigureCanvasAgg(fig)


def _inputs(paths) -> list[dict]:
    out = []
    for item in paths:
        p = Path(item)
        out.append({'path': str(p.resolve()) if p.exists() else str(item),
                    'sha256': sha256_file(p) if p.is_file() else None,
                    'bytes': p.stat().st_size if p.is_file() else None})
    return out


def _save_all(fig, base, fmts, dpi, kind, written, spec):
    with use_spec(spec):
        for fmt in fmts:
            target = base.with_name(f"{base.name}.{fmt}")
            kwargs = {'format': fmt, 'dpi': dpi, 'facecolor': 'white'}
            if fmt == 'pdf':
                kwargs['metadata'] = {'Creator': f'Sparse-DFT plotting (matplotlib {matplotlib.__version__})',
                                      'Title': str(kind or base.name)}
            elif fmt == 'svg':
                kwargs['metadata'] = {'Creator': 'Sparse-DFT plotting', 'Title': str(kind or base.name)}
            elif fmt == 'png':
                kwargs['metadata'] = {'Software': f'Sparse-DFT plotting (matplotlib {matplotlib.__version__})'}
            elif fmt == 'tiff':
                kwargs['pil_kwargs'] = {'compression': 'tiff_lzw'}
            fig.savefig(target, **kwargs)
            written.append(target)


def export(fig, stem, formats=('pdf', 'png'), *, dpi=600, provenance=None) -> list[Path]:
    """Save ``fig`` as ``<stem>.<fmt>`` for every format plus ``<stem>.provenance.json``.

    Returns the written paths: figure files in the requested order, then the
    provenance file.  The figure's journal style is re-applied while saving
    (Type-42 PDF/PS fonts, SVG text kept as text).
    """
    fmts = normalize_formats(formats)
    if not isinstance(dpi, (int, float)) or not 72 <= dpi <= 2400:
        raise ValueError('dpi must be 72-2400')
    base = _stem(stem)
    base.parent.mkdir(parents=True, exist_ok=True)
    _ensure_canvas(fig)
    spec = style_of(fig)
    meta = dict(getattr(fig, 'sparse_dft_meta', None) or {})
    extra = dict(provenance or {})
    kind = extra.pop('kind', None) or meta.get('kind')
    written = []
    # fontTools logs a harmless notice when subsetting CFF-flavoured CJK
    # fonts (e.g. PingFang) into PDF; silence it for this call only.
    quiet = [logging.getLogger(name) for name in ('fontTools.cffLib', 'fontTools.subset')]
    levels = [lg.level for lg in quiet]
    for lg in quiet:
        lg.setLevel(logging.ERROR)
    try:
        _save_all(fig, base, fmts, dpi, kind, written, spec)
    finally:
        for lg, level in zip(quiet, levels):
            lg.setLevel(level)
    inputs = list(dict.fromkeys([str(x) for x in (meta.get('inputs') or [])] +
                                [str(x) for x in (extra.pop('inputs', None) or [])]))
    parameters = dict(meta.get('parameters') or {})
    parameters.update(extra.pop('parameters', None) or {})
    width_in, height_in = fig.get_size_inches()
    record = {
        'schema_version': PROVENANCE_SCHEMA,
        'kind': kind,
        'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'files': [{'path': str(p.resolve()), 'format': p.suffix.lstrip('.'), 'sha256': sha256_file(p),
                   'bytes': p.stat().st_size} for p in written],
        'inputs': _inputs(inputs),
        'parameters': parameters,
        'values': meta.get('values') or {},
        'notes': list(dict.fromkeys((meta.get('notes') or []) + list(extra.pop('notes', None) or []))),
        'style': {'width_mm': round(width_in * 25.4, 3), 'height_mm': round(height_in * 25.4, 3),
                  'lang': spec.lang, 'palette': spec.palette, 'base_fontsize_pt': spec.base_fontsize,
                  'font_family': list(spec.rc.get('font.family', [])),
                  'pdf.fonttype': spec.rc.get('pdf.fonttype'), 'svg.fonttype': spec.rc.get('svg.fonttype')},
        'dpi': dpi,
        'software': {'matplotlib': matplotlib.__version__, 'python': platform.python_version(),
                     'platform': platform.platform()},
    }
    if extra:
        record['extra'] = extra
    prov = provenance_path(base)
    prov.write_text(json.dumps(record, indent=2, sort_keys=True, default=str) + '\n', encoding='utf-8')
    written.append(prov)
    return written


def export_figures(figures: dict, stem, formats=('pdf', 'png'), *, dpi=600, provenance=None) -> dict:
    """Export {name: Figure}: 'combined' goes to ``stem``, each panel to ``stem_<name>``; returns {name: paths}."""
    out = {}
    base = _stem(stem)
    for name, fig in figures.items():
        target = base if name == 'combined' else base.with_name(f'{base.name}_{name}')
        out[name] = export(fig, target, formats, dpi=dpi, provenance=provenance)
    return out
