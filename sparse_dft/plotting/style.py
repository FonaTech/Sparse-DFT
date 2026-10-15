"""Journal figure style for Sparse-DFT plots.

Design rules (Nature-family artwork guidelines):
* widths 89 / 120 / 183 mm (250 mm 'atlas' for multi-panel overviews); 5-8 pt
  text; TrueType fonts embedded (Type 42).
* the default 'academic' palette draws a closed frame with inward, mirrored
  ticks, slate ink and light grids; 'okabe_ito' and 'macaron' keep the open
  (no top/right spine, outward tick) journal frame.
* everything is scoped by :func:`matplotlib.rc_context`; this module never
  calls ``matplotlib.use`` and never edits the global ``rcParams``.
* figures are :class:`matplotlib.figure.Figure` objects (no pyplot), so they
  are safe to embed in a Qt application.  :class:`StyledFigure` re-enters its
  style whenever it is drawn, so a later Qt repaint looks like the export.
"""
from __future__ import annotations

import contextlib
import contextvars
from dataclasses import dataclass, field
from functools import lru_cache

import matplotlib
from matplotlib.figure import Figure

MM_PER_INCH = 25.4
WIDTHS_MM = {'single': 89.0, 'onehalf': 120.0, 'double': 183.0}
ATLAS_WIDTH_MM = 250.0           # multi-panel overview figures (screen/poster, still within width_mm limits)
# Name of the program in figure text.  The release build renames it; never
# hard-code the product name elsewhere in figure text.
PRODUCT = 'Sparse-DFT'

# Colour-blind-safe categorical palettes, fixed order (never cycled for
# identity; >6 series use colour + marker shape).  Both were checked with the
# dataviz palette validator on a white surface: lightness band, chroma floor,
# adjacent-pair CVD separation >= 8 (OKLab x100) and normal-vision floor pass.
# Okabe-Ito yellow (#F0E442) is omitted: 1.2:1 contrast on white paper.
PALETTES: dict[str, list[str]] = {
    'okabe_ito': ['#0072B2', '#D55E00', '#009E73', '#E69F00', '#CC79A7', '#56B4E9'],
    # Macaron: blueberry, peach, pistachio, apricot, lavender, rose.  Line
    # colours are darkened pastels so they survive print; use tint() for fills.
    'macaron': ['#3A6DB5', '#F08C6C', '#2B8A57', '#E0A23E', '#7B5BB5', '#E58FB6'],
    # Academic: royal blue, ochre, teal, crimson, violet, cyan, rose, olive.  The
    # slate/ink family of the GaAs reference figure; 8 slots, all six validator
    # checks pass on white (adjacent CVD dE >= 10.3, normal-vision floor 24.3).
    'academic': ['#1D4ED8', '#D97706', '#0D9488', '#B91C1C', '#7C3AED', '#0891B2', '#DB2777', '#65A30D'],
}
DEFAULT_PALETTE = 'academic'
FRAMED_PALETTES = {'academic'}
# Diverging (negative pole, neutral midpoint, positive pole).
DIVERGING = {
    'okabe_ito': ('#2166AC', '#F7F7F7', '#B2182B'),
    'macaron': ('#3A6DB5', '#F4F1EE', '#C9503A'),
    'academic': ('#1E3A8A', '#F1F5F9', '#B91C1C'),
}
NEUTRAL = {'ink': '#0F172A', 'text': '#334155', 'muted': '#64748B', 'guide': '#CBD5E1', 'zero': '#475569',
           'band': '#94A3B8', 'grid': '#E2E8F0', 'gap': '#F1F5F9', 'panel': '#F8FAFC', 'frame': '#CBD5E1',
           'fermi': '#B91C1C', 'fail_fill': '#F6DADA', 'pass_fill': '#E3F1E8'}
# Continuous maps of the reference figure: orbital character (0 = p/d-like ->
# 1 = s-like) and Brillouin-zone energy landscapes.  Both are monotonic in
# lightness except the deliberate warm end of the orbital map.
ORBITAL_CMAP_STOPS = ((0.00, '#172554'), (0.25, '#0369A1'), (0.50, '#0D9488'), (0.75, '#D97706'), (1.00, '#B91C1C'))
BZ_CMAP_STOPS = ((0.00, '#0F172A'), (0.20, '#1E3A8A'), (0.45, '#0D9488'), (0.72, '#EAB308'), (1.00, '#FEF08A'))
MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '<', '>', 'h']

LATIN_FONTS = ('Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans')
CJK_FONTS = {
    'zh_CN': ('Noto Sans CJK SC', 'Source Han Sans SC', 'Source Han Sans CN', 'Source Han Sans',
              'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', 'Heiti SC', 'STHeiti',
              'WenQuanYi Micro Hei', 'Arial Unicode MS'),
    'zh_TW': ('Noto Sans CJK TC', 'Source Han Sans TC', 'Source Han Sans TW', 'Source Han Sans',
              'PingFang TC', 'Microsoft JhengHei', 'Heiti TC', 'Arial Unicode MS'),
    'ja': ('Noto Sans CJK JP', 'Source Han Sans JP', 'Source Han Sans', 'Hiragino Sans',
           'Hiragino Kaku Gothic ProN', 'Yu Gothic', 'YuGothic', 'MS Gothic', 'Noto Sans JP',
           'Arial Unicode MS'),
}
LANGS = ('en', 'zh_CN', 'zh_TW', 'ja')


@lru_cache(maxsize=1)
def _installed_fonts() -> frozenset:
    from matplotlib import font_manager
    return frozenset(f.name for f in font_manager.fontManager.ttflist)


def pick_font(candidates) -> str | None:
    """First installed font family among ``candidates`` (None if none)."""
    installed = _installed_fonts()
    return next((name for name in candidates if name in installed), None)


def font_stack(lang: str = 'en') -> list[str]:
    """Font family list for ``lang``.

    English: Arial -> Helvetica -> Liberation Sans -> DejaVu Sans.  CJK
    languages put the first installed CJK gothic *first*: matplotlib's
    mathtext draws the plain-text part of a mixed label such as
    '能量 $E - E_F$ (eV)' with the first family only (no per-glyph fallback),
    so a Latin-first stack would render those CJK characters as tofu.  The
    CJK candidates all carry Latin, Greek and U+2212 glyphs.  Never raises:
    with no CJK font installed the Latin stack is returned.
    """
    latin = pick_font(LATIN_FONTS) or 'DejaVu Sans'
    stack = [latin]
    if lang in CJK_FONTS:
        cjk = pick_font(CJK_FONTS[lang])
        if cjk:
            stack.insert(0, cjk)
    if 'DejaVu Sans' not in stack:
        stack.append('DejaVu Sans')       # ships with matplotlib: never tofu for Latin/Greek
    return stack


def latin_font() -> str:
    return pick_font(LATIN_FONTS) or 'DejaVu Sans'


def mm(value_mm: float) -> float:
    return float(value_mm) / MM_PER_INCH


def width_mm(width) -> float:
    if isinstance(width, (int, float)):
        if not 20.0 <= float(width) <= 250.0:
            raise ValueError(f'figure width must be 20-250 mm, got {width}')
        return float(width)
    if str(width) == 'atlas':
        return ATLAS_WIDTH_MM
    try:
        return WIDTHS_MM[str(width)]
    except KeyError:
        raise ValueError(f'unknown width {width!r}; choose one of {sorted(WIDTHS_MM)} or mm') from None


def palette_colors(palette) -> list[str]:
    if isinstance(palette, (list, tuple)):
        return [str(c) for c in palette]
    try:
        return list(PALETTES[str(palette)])
    except KeyError:
        raise ValueError(f'unknown palette {palette!r}; choose one of {sorted(PALETTES)}') from None


def tint(color, fraction: float = 0.6) -> tuple:
    """Blend ``color`` toward white (fraction 0 = colour, 1 = white)."""
    from matplotlib.colors import to_rgb
    r, g, b = to_rgb(color)
    f = min(max(float(fraction), 0.0), 1.0)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)


@lru_cache(maxsize=4)
def _listed(name, stops):
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(name, list(stops), N=256)


def orbital_cmap():
    """Orbital-character map (0: p/d-like, 1: s-like) of the academic style."""
    return _listed('sparse_dft_orbital', ORBITAL_CMAP_STOPS)


def bz_cmap():
    """Sequential map for Brillouin-zone energy landscapes."""
    return _listed('sparse_dft_bz', BZ_CMAP_STOPS)


def diverging_cmap(palette='okabe_ito'):
    from matplotlib.colors import LinearSegmentedColormap
    if isinstance(palette, str) and palette in DIVERGING:
        lo, mid, hi = DIVERGING[palette]
    else:
        lo, mid, hi = DIVERGING['okabe_ito']
    return LinearSegmentedColormap.from_list(f'sparse_dft_div_{palette}', [lo, mid, hi], N=256)


@dataclass
class StyleSpec:
    width: object
    width_mm: float
    lang: str
    palette: str
    colors: list[str]
    base_fontsize: float
    rc: dict = field(repr=False)
    frame: bool = False                 # closed frame, inward mirrored ticks, light grids (academic)

    @property
    def tick_fontsize(self) -> float:
        return max(5.0, self.base_fontsize - 1.0)

    @property
    def small_fontsize(self) -> float:
        return max(5.0, self.base_fontsize - 1.5)

    def color(self, index: int) -> str:
        return self.colors[index % len(self.colors)]

    @property
    def diverging(self):
        return diverging_cmap(self.palette)


def _rc(lang: str, colors: list[str], base: float, frame: bool = False) -> dict:
    from cycler import cycler
    if lang not in LANGS:
        raise ValueError(f'unknown language {lang!r}; choose one of {list(LANGS)}')
    if not 5.0 <= float(base) <= 9.0:
        raise ValueError('base_fontsize must be 5-9 pt (journal artwork range)')
    stack = font_stack(lang)
    latin = latin_font()          # math symbols stay in the Latin face in every language
    tick = max(5.0, base - 1.0)
    line = 0.55
    rc = {
        'font.family': stack, 'font.size': base,
        'axes.labelsize': base, 'axes.titlesize': base, 'axes.titleweight': 'normal',
        'xtick.labelsize': tick, 'ytick.labelsize': tick, 'legend.fontsize': tick,
        'figure.titlesize': base + 1,
        'axes.linewidth': line, 'axes.edgecolor': NEUTRAL['ink'], 'axes.labelcolor': NEUTRAL['ink'],
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.labelpad': 2.0, 'axes.titlepad': 3.0, 'axes.grid': False,
        'axes.prop_cycle': cycler(color=colors), 'axes.unicode_minus': True,
        'axes.formatter.use_mathtext': True, 'axes.formatter.limits': (-3, 4),
        'axes.xmargin': 0.02, 'axes.ymargin': 0.04, 'axes.facecolor': 'white',
        'xtick.direction': 'out', 'ytick.direction': 'out',
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.minor.size': 1.4, 'ytick.minor.size': 1.4,
        'xtick.major.width': line, 'ytick.major.width': line, 'xtick.minor.width': 0.45, 'ytick.minor.width': 0.45,
        'xtick.major.pad': 1.8, 'ytick.major.pad': 1.8, 'xtick.color': NEUTRAL['ink'], 'ytick.color': NEUTRAL['ink'],
        'xtick.top': False, 'ytick.right': False,
        'lines.linewidth': 0.9, 'lines.markersize': 3.0, 'lines.markeredgewidth': 0.5,
        'patch.linewidth': 0.5, 'hatch.linewidth': 0.5, 'errorbar.capsize': 1.5,
        'legend.frameon': False, 'legend.handlelength': 1.6, 'legend.handletextpad': 0.4,
        'legend.labelspacing': 0.3, 'legend.columnspacing': 1.0, 'legend.borderaxespad': 0.3,
        'legend.borderpad': 0.2,
        'figure.facecolor': 'white', 'figure.dpi': 150,
        'figure.constrained_layout.h_pad': 0.02, 'figure.constrained_layout.w_pad': 0.02,
        'figure.constrained_layout.hspace': 0.03, 'figure.constrained_layout.wspace': 0.03,
        'savefig.facecolor': 'white', 'savefig.bbox': None, 'savefig.pad_inches': 0.02,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'svg.hashsalt': 'sparse_dft',
        'image.interpolation': 'nearest',
    }
    if frame:
        # The academic frame: closed box, inward ticks mirrored on all sides,
        # bold axis labels, slate ink, recessive grid drawn by the plotters.
        rc.update({
            'axes.spines.top': True, 'axes.spines.right': True, 'axes.linewidth': 0.7,
            'axes.edgecolor': NEUTRAL['ink'], 'axes.labelcolor': NEUTRAL['ink'], 'axes.labelweight': 'bold',
            'axes.titleweight': 'bold', 'axes.titlelocation': 'left', 'axes.titlepad': 4.0,
            'axes.titlesize': base + 0.5, 'axes.labelpad': 2.4,
            'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True,
            'xtick.major.size': 3.0, 'ytick.major.size': 3.0, 'xtick.minor.size': 1.6, 'ytick.minor.size': 1.6,
            'xtick.major.width': 0.7, 'ytick.major.width': 0.7, 'xtick.minor.width': 0.45,
            'ytick.minor.width': 0.45, 'xtick.major.pad': 3.6, 'ytick.major.pad': 3.0,
            'xtick.color': NEUTRAL['ink'], 'ytick.color': NEUTRAL['ink'], 'text.color': NEUTRAL['ink'],
            'grid.color': NEUTRAL['grid'], 'grid.linewidth': 0.45, 'grid.alpha': 0.8, 'grid.linestyle': '-',
            'legend.frameon': True, 'legend.framealpha': 0.95, 'legend.edgecolor': NEUTRAL['frame'],
            'legend.facecolor': 'white', 'legend.fancybox': True, 'legend.borderpad': 0.35,
            'lines.linewidth': 1.0, 'figure.titleweight': 'bold', 'figure.titlesize': min(8.0, base + 1.0),
        })
    if latin == 'DejaVu Sans':
        rc['mathtext.fontset'] = 'dejavusans'
    else:
        rc.update({'mathtext.fontset': 'custom', 'mathtext.rm': latin, 'mathtext.sf': latin,
                   'mathtext.it': f'{latin}:italic', 'mathtext.bf': f'{latin}:bold',
                   'mathtext.cal': latin, 'mathtext.tt': 'DejaVu Sans Mono',
                   'mathtext.fallback': 'stixsans'})
    return rc


def make_style(width='single', *, lang='en', palette=DEFAULT_PALETTE, base_fontsize=7.0,
               frame: bool | None = None) -> StyleSpec:
    colors = palette_colors(palette)
    if frame is None:
        frame = isinstance(palette, str) and palette in FRAMED_PALETTES
    return StyleSpec(width=width, width_mm=width_mm(width), lang=lang,
                     palette=palette if isinstance(palette, str) else 'custom', colors=colors,
                     base_fontsize=float(base_fontsize), rc=_rc(lang, colors, float(base_fontsize), bool(frame)),
                     frame=bool(frame))


_ACTIVE: contextvars.ContextVar = contextvars.ContextVar('sparse_dft_plot_style', default=None)


def current_style() -> StyleSpec:
    """The style of the innermost :func:`figure_style` block (default style outside)."""
    return _ACTIVE.get() or make_style()


@contextlib.contextmanager
def figure_style(width='single', *, lang='en', palette=DEFAULT_PALETTE, base_fontsize=7.0, frame=None):
    """Scope journal rcParams to a block; yields the :class:`StyleSpec`.

    Only :func:`matplotlib.rc_context` is used, so global ``rcParams`` are
    restored on exit even if plotting raises.
    """
    spec = make_style(width, lang=lang, palette=palette, base_fontsize=base_fontsize, frame=frame)
    token = _ACTIVE.set(spec)
    try:
        with matplotlib.rc_context(spec.rc):
            yield spec
    finally:
        _ACTIVE.reset(token)


@contextlib.contextmanager
def use_spec(spec: StyleSpec | None):
    if spec is None:
        yield None
        return
    token = _ACTIVE.set(spec)
    try:
        with matplotlib.rc_context(spec.rc):
            yield spec
    finally:
        _ACTIVE.reset(token)


class StyledFigure(Figure):
    """A Figure that re-applies its journal style each time it is drawn."""

    def __init__(self, *args, sparse_dft_style: StyleSpec | None = None, **kwargs):
        self.sparse_dft_style = sparse_dft_style
        with use_spec(sparse_dft_style):
            super().__init__(*args, **kwargs)

    def draw(self, renderer):
        with use_spec(self.sparse_dft_style):
            return super().draw(renderer)


def style_of(fig) -> StyleSpec:
    return getattr(fig, 'sparse_dft_style', None) or current_style()


def new_figure(width='single', height_mm=None, nrows=1, ncols=1, **subplots_kw):
    """Create a styled Figure (no pyplot) and its axes.

    Returns ``(fig, axes)`` where ``axes`` is always a numpy array; by default
    2-D ``(nrows, ncols)``.  ``squeeze=True`` gives a 1-D array instead.
    ``layout`` (default 'constrained') goes to the Figure, everything else to
    :meth:`Figure.subplots`.  Inside :func:`figure_style` the active language
    and palette are kept; ``width`` always sets the physical width.
    """
    import numpy as np
    active = _ACTIVE.get()
    if active is not None and width_mm(width) == active.width_mm:
        spec = active
    elif active is not None:
        spec = make_style(width, lang=active.lang, palette=active.colors, base_fontsize=active.base_fontsize,
                          frame=active.frame)
        spec.palette = active.palette
    else:
        spec = make_style(width)
    w = spec.width_mm
    if height_mm is None:
        height_mm = min(230.0, max(40.0, w * 0.68 * nrows / max(1, ncols) ** 0.5))
    layout = subplots_kw.pop('layout', 'constrained')
    squeeze = bool(subplots_kw.pop('squeeze', False))
    fig = StyledFigure(figsize=(mm(w), mm(float(height_mm))), layout=layout, sparse_dft_style=spec)
    with use_spec(spec):
        axes = fig.subplots(nrows, ncols, squeeze=False, **subplots_kw)
    axes = np.asarray(axes, dtype=object)
    if squeeze:
        axes = np.atleast_1d(axes.squeeze())
    return fig, axes



def blank_figure(width='single', height_mm=None, *, layout='constrained'):
    """A styled Figure without axes, for custom GridSpec layouts (same style rules as :func:`new_figure`)."""
    active = _ACTIVE.get()
    if active is not None and width_mm(width) == active.width_mm:
        spec = active
    elif active is not None:
        spec = make_style(width, lang=active.lang, palette=active.colors, base_fontsize=active.base_fontsize,
                          frame=active.frame)
        spec.palette = active.palette
    else:
        spec = make_style(width)
    height = float(height_mm) if height_mm is not None else spec.width_mm * 0.68
    return StyledFigure(figsize=(mm(spec.width_mm), mm(height)), layout=layout, sparse_dft_style=spec)

def panel_labels(axes, start='a', *, offset_pt=(-14.0, 2.0), fontsize=None) -> None:
    """Bold lowercase panel letters just outside each panel's top-left corner."""
    import numpy as np
    flat = [ax for ax in np.asarray(axes, dtype=object).ravel() if ax is not None]
    first = ord(str(start)[0])
    for i, ax in enumerate(flat):
        spec = style_of(ax.figure)
        ax.annotate(chr(first + i), xy=(0.0, 1.0), xycoords='axes fraction',
                    xytext=offset_pt, textcoords='offset points', ha='right', va='bottom',
                    fontsize=fontsize or spec.base_fontsize + 1.0, fontweight='bold',
                    color=NEUTRAL['ink'], annotation_clip=False)


def frame_axes(ax, *, grid='y') -> None:
    """Academic frame details for one Axes: mirrored inward ticks and a recessive grid.

    ``grid`` is 'x', 'y', 'both' or None.  A no-op for the open journal frame.
    """
    spec = style_of(ax.figure)
    if not spec.frame:
        return
    ax.tick_params(which='both', direction='in', top=True, right=True)
    if grid:
        ax.grid(True, axis=grid, which='major', color=NEUTRAL['grid'], lw=0.45, alpha=0.8, zorder=0)
        ax.set_axisbelow(True)


def panel_title(ax, letter, text=None, *, loc='left', fontsize=None):
    """Bold ``letter`` + bold title above an Axes ("a   Projected DOS").

    The open journal frame keeps the plain bold letter outside the corner (no
    in-panel titles); the academic frame writes letter and title as the Axes
    title, so constrained layout reserves room for it.
    """
    spec = style_of(ax.figure)
    if not spec.frame:
        if letter:
            ax.annotate(letter, xy=(0.0, 1.0), xycoords='axes fraction', xytext=(-14.0, 2.0),
                        textcoords='offset points', ha='right', va='bottom', fontweight='bold',
                        fontsize=fontsize or spec.base_fontsize + 1.0, color=NEUTRAL['ink'], annotation_clip=False)
        return None
    label = '   '.join(s for s in (letter, text) if s)
    return ax.set_title(label, loc=loc, fontweight='bold', fontsize=fontsize or min(8.0, spec.base_fontsize + 0.5),
                        color=NEUTRAL['ink'])


def figure_header(fig, title, subtitle=None, *, top=0.985, gap=0.022):
    """Bold title and italic subtitle at the top of a manually laid-out figure.

    ``top``/``gap`` are figure fractions (the atlas layouts reserve that strip).
    Returns the (title, subtitle) Text artists.  Constrained-layout figures use
    ``fig.suptitle`` instead.
    """
    spec = style_of(fig)
    size = min(8.0, spec.base_fontsize + 1.0)
    head = fig.text(0.5, top, title, ha='center', va='top', fontsize=size, fontweight='bold', color=NEUTRAL['ink'])
    sub = None
    if subtitle:
        sub = fig.text(0.5, top - gap, subtitle, ha='center', va='top', fontsize=max(5.0, spec.base_fontsize - 0.5),
                       style='italic', color=NEUTRAL['text'])
    return head, sub
