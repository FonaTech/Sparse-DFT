"""Internal helpers shared by the per-kind plotting functions."""
from __future__ import annotations

import math

import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.ticker import AutoMinorLocator, MaxNLocator

from ..errors import InputError
from .style import NEUTRAL, current_style, new_figure, style_of

GUIDE_KW = dict(color=NEUTRAL['guide'], lw=0.4, zorder=0.5)
ZERO_KW = dict(color=NEUTRAL['zero'], lw=0.5, ls=(0, (3.0, 2.0)), zorder=0.6)


def axes_or_new(ax, *, nrows=1, ncols=1, height_mm=None, **subplots_kw):
    """Return (fig, axes 2-D array). A caller-supplied Axes/array is used as is."""
    if ax is None:
        spec = current_style()
        return new_figure(spec.width if spec else 'single', height_mm=height_mm,
                          nrows=nrows, ncols=ncols, **subplots_kw)
    arr = np.asarray(ax, dtype=object)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(-1, 1) if nrows > 1 else arr.reshape(1, -1)
    return arr.flat[0].figure, arr


def provided_count(ax) -> int:
    return 0 if ax is None else int(np.asarray(ax, dtype=object).size)


def attach_meta(fig, kind, data_sources, parameters=None, notes=None, values=None):
    """Record what a figure shows; :func:`sparse_dft.plotting.export.export` writes it out."""
    meta = getattr(fig, 'sparse_dft_meta', None) or {}
    meta.setdefault('kind', kind)
    sources = [str(s) for s in data_sources if s]
    meta['inputs'] = list(dict.fromkeys(meta.get('inputs', []) + sources))
    meta.setdefault('parameters', {}).update({k: v for k, v in (parameters or {}).items()})
    meta.setdefault('notes', []).extend(notes or [])
    meta.setdefault('values', {}).update(values or {})
    fig.sparse_dft_meta = meta
    return fig


def note(ax, text, *, loc='upper right', color=None, fontsize=None, zorder=5):
    spec = style_of(ax.figure)
    positions = {'upper right': (0.99, 0.99, 'right', 'top'), 'upper left': (0.01, 0.99, 'left', 'top'),
                 'lower right': (0.99, 0.01, 'right', 'bottom'), 'lower left': (0.01, 0.01, 'left', 'bottom'),
                 'above right': (1.0, 1.01, 'right', 'bottom'), 'above left': (0.0, 1.01, 'left', 'bottom')}
    x, y, ha, va = positions[loc]
    return ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, color=color or NEUTRAL['muted'],
                   fontsize=fontsize or spec.small_fontsize, zorder=zorder)


def integer_xaxis(ax):
    """Integer ticks limited to the data range (no '-1' tick before step 0)."""
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6, min_n_ticks=3))
    xs = [x for line in ax.lines for x in np.asarray(line.get_xdata(), dtype=float).ravel() if np.isfinite(x)]
    if xs:
        lo, hi = min(xs), max(xs)
        pad = max(0.4, 0.03 * (hi - lo))
        ax.set_xlim(lo - pad, hi + pad)


def minor(ax, x=True, y=True):
    if x and ax.get_xscale() == 'linear':
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    if y and ax.get_yscale() == 'linear':
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))


def text_color_for(rgba) -> str:
    r, g, b = to_rgb(rgba)
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return NEUTRAL['ink'] if lum > 0.55 else 'white'


def sci(value, digits=3) -> str:
    """Compact math-text number: 1.5e-08 -> '1.5\\times10^{-8}' (no $)."""
    if value is None or not math.isfinite(value):
        return 'n/a'
    if value == 0:
        return '0'
    exponent = int(math.floor(math.log10(abs(value))))
    if -3 <= exponent <= 4:
        text = f'{value:.{digits}g}'
        return text.replace('-', '−')
    mantissa = value / 10 ** exponent
    m = f'{mantissa:.{max(1, digits - 1)}g}'
    sign = '−' if exponent < 0 else ''
    head = '' if m == '1' else (m.replace('-', '−') + r'\times')
    return f'{head}10^{{{sign}{abs(exponent)}}}'


def signed(value, fmt='{:.3f}') -> str:
    text = fmt.format(abs(value))
    return ('−' if value < 0 else '+') + text


def symmetric_limit(values, floor=1e-300) -> float:
    finite_values = [abs(v) for v in values if v is not None and math.isfinite(v)]
    peak = max(finite_values, default=0.0)
    return peak if peak > floor else 1.0


def require(condition, message, **details):
    if not condition:
        raise InputError(message, details=details or None)


def energy_window(values, emin, emax, default=(-10.0, 10.0), pad=0.3):
    finite_values = [v for v in values if v is not None and math.isfinite(v)]
    lo = min(finite_values, default=default[0]) - pad
    hi = max(finite_values, default=default[1]) + pad
    lo = max(lo, default[0]) if emin is None else float(emin)
    hi = min(hi, default[1]) if emax is None else float(emax)
    if hi <= lo:
        raise InputError(f'empty energy window: emin={lo} >= emax={hi}')
    return lo, hi
