"""Electronic-structure atlas: a multi-panel overview drawn from one band run.

Panels (each omitted, with the reason recorded, when its inputs are missing):

a  bands coloured by an l-orbital fraction; reference-engine markers at shared k,
   gap shading and arrow, conduction-valley call-outs
b  projected DOS (``sparse-dft dos`` mesh result) or the SCF-mesh total DOS
c  band-edge curvature: path samples, Richardson c2 k² curves, m*/m0
d  k3 = 0 plane of a uniform mesh: direct transition E_c - E_v (insulators) or
   Fermi-level cuts (metals); mesh points drawn, interpolation labelled
e  BZ-weighted element+l composition per band (mesh weights, else path average)
f  candidate - reference band energies at shared k (matched conditions only)
g  lm projection parity against y = x with near-degenerate pairs flagged

Every printed number comes from :mod:`sparse_dft.analysis.electronic_atlas` and is
stored in the figure's provenance ``values`` with its definition.
``render_atlas(data, panels='combined'|'split')`` returns one figure or one
figure per panel with the same styling.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import Bbox

from ..analysis import electronic as E
from ..analysis import electronic_atlas as A
from ..errors import InputError
from ._util import attach_meta
from .export import _ensure_canvas
from .labels import tr
from .style import (NEUTRAL, PRODUCT, blank_figure, bz_cmap, frame_axes, mm, orbital_cmap, panel_title, style_of,
                    tint)

PANEL_KEYS = ('bands', 'pdos', 'edges', 'bz', 'composition', 'residual', 'parity')
LETTERS = dict(zip(PANEL_KEYS, 'abcdefg'))
PANEL_MODES = ('combined', 'split', 'both')
# split-figure sizes (mm): width, height
SPLIT_SIZE_MM = {'bands': (120.0, 92.0), 'pdos': (70.0, 92.0), 'edges': (89.0, 74.0), 'bz': (89.0, 80.0),
                 'composition': (120.0, 68.0), 'residual': (89.0, 68.0), 'parity': (89.0, 74.0)}
_trapz = getattr(np, "trapezoid", None) or np.trapz
SUBSCRIPT = str.maketrans('0123456789', '₀₁₂₃₄₅₆₇₈₉')


# --- inputs ------------------------------------------------------------------------
@dataclass
class AtlasData:
    """Everything the atlas panels draw, loaded once."""
    bands: E.BandStructure
    projections: E.ProjectionSet | None
    reference: E.BandStructure | None = None
    reference_projections: E.ProjectionSet | None = None
    mesh: E.BandStructure | None = None
    mesh_projections: E.ProjectionSet | None = None
    dos: E.DosData | None = None
    conditions: dict | None = None            # compare_conditions(candidate, reference)
    mesh_conditions: dict | None = None       # path vs mesh run (same frozen SCF density?)
    species: list = field(default_factory=list)
    sources: dict = field(default_factory=dict)
    load_notes: list = field(default_factory=list)


def _frozen_hash(path) -> str | None:
    """sha256 recorded next to the frozen SCF density a band/mesh run was started from."""
    folder = Path(path).resolve().parent         # the run folder, also through a symlinked file
    for p in sorted(folder.glob('frozen_scf.*.sha256')):
        text = p.read_text(encoding='utf-8', errors='replace').split()
        if text:
            return text[0]
    return None


def formula(species) -> str:
    """Reduced formula with Unicode subscripts in first-appearance order ('GaAs', 'MoS₂', 'Al')."""
    from math import gcd
    counts = {}
    for s in species or []:
        counts[s] = counts.get(s, 0) + 1
    if not counts:
        return ''
    g = 0
    for n in counts.values():
        g = gcd(g, n)
    return ''.join(el + (str(n // g).translate(SUBSCRIPT) if n // g > 1 else '') for el, n in counts.items())


def load_atlas(bands, *, reference=None, mesh=None, dos=None, structure=None) -> AtlasData:
    """Load a band path and its optional companions (paths of files)."""
    b = E.load_band_data(bands, structure=structure)
    notes = []
    proj = None
    try:
        proj = E.load_projections(b)
    except InputError as exc:
        notes.append(f'no projections on the band path: {exc}')
    data = AtlasData(bands=b, projections=proj, sources={'bands': str(bands)})
    cond = A.run_conditions(bands, b)
    data.species = list(cond.get('species') or (proj.species if proj and proj.species else []))
    if reference is not None:
        ref = E.load_band_data(reference)
        data.reference = ref
        data.sources['reference'] = str(reference)
        try:
            data.reference_projections = E.load_projections(ref, reference)
        except InputError as exc:
            notes.append(f'reference has no projections: {exc}')
        data.conditions = A.compare_conditions(cond, A.run_conditions(reference, ref), b, ref)
    if mesh is not None:
        info = Path(mesh)
        m = E.bands_from_scf(info) if _is_scf(info) else E.load_band_data(info)
        if E.is_mesh(m):
            data.mesh = m
            data.sources['mesh'] = str(mesh)
            try:
                data.mesh_projections = E.load_projections(m)
            except InputError:
                pass
            same = None
            h1, h2 = _frozen_hash(bands), _frozen_hash(mesh)
            if h1 and h2:
                same = h1 == h2
            data.mesh_conditions = {'same_frozen_scf_density': same, 'path_frozen_sha256': h1, 'mesh_frozen_sha256': h2,
                                    'mesh_kind': 'scf mesh' if _is_scf(info) else 'nscf mesh',
                                    'grid_points': m.nkpoints}
        else:
            notes.append(f'{mesh} is not a uniform k mesh; panels d and e use the path')
    if dos is not None:
        data.dos = E.load_dos(dos, sigma=0.1)
        data.sources['dos'] = str(dos)
    data.load_notes = notes
    return data


def _is_scf(path) -> bool:
    from .detect import scan
    info = scan(path)
    return bool(info) and 'scf_result' in info[0].roles and 'bands_file' not in info[0].roles and \
        'bands_mesh' not in info[0].roles


# --- analysis context -------------------------------------------------------------
L_TINT = {'s': 0.0, 'p': 0.38, 'd': 0.62, 'f': 0.78}
L_DASH = {'s': '-', 'p': (0, (3.2, 1.4)), 'd': (0, (1.1, 1.1)), 'f': (0, (4.0, 1.2, 1.2, 1.2))}


@dataclass
class AtlasContext:
    data: AtlasData
    pa: A.PathAnalysis
    kp: E.KPath
    channels: dict                       # name -> [k][band] (eV, absolute)
    zero: float
    zero_key: str                        # axis-label key of the energy zero
    lo: float
    hi: float
    top_complete: float                  # bands are complete below this energy (relative)
    orbital: str | None = None
    fractions: dict = field(default_factory=dict)      # channel -> (fraction, sphere weight)
    groups: list = field(default_factory=list)         # [(element, l)] colour identities, display order
    pdos: list = field(default_factory=list)           # [(element, l, DosCurve)] drawn in panel b
    residuals: dict | None = None
    parity: dict | None = None
    planes: list = field(default_factory=list)         # [(channel, bz_plane result)]
    composition: list = field(default_factory=list)    # [(channel, band_composition result)]
    composition_source: str = ''
    omitted: dict = field(default_factory=dict)        # panel -> reason
    notes: list = field(default_factory=list)
    title: str = ''
    subtitle: str = ''

    @property
    def available(self) -> list:
        return [k for k in PANEL_KEYS if k not in self.omitted]

    @property
    def formula(self) -> str:
        return formula(self.data.species) or Path(self.data.bands.source).parent.name

    @property
    def ref_engine(self) -> str:
        cond = self.data.conditions or {}
        return (cond.get('reference') or {}).get('engine') or tr('reference')

    def rel(self, values):
        return np.asarray(values, dtype=float) - self.zero


def _group_key(label, element, l):
    if element and l:
        return element, l
    parts = str(label).replace('$', '').split()
    return (parts[0], parts[-1]) if len(parts) >= 2 else (str(label), None)


def _composition_bands(pa, src_bands, channel):
    rows = np.asarray(E.channels_of(src_bands)[channel], dtype=float)
    if pa.cb is not None:
        return int(min(rows.shape[1], pa.cb + 1 + 4))
    occupied = [b for b in range(rows.shape[1]) if rows[:, b].min() - pa.zero_ev < 3.0]
    return int(min(rows.shape[1], (max(occupied) + 1) if occupied else rows.shape[1], 16))


def analyse_atlas(data: AtlasData, *, emin=None, emax=None) -> AtlasContext:
    """Compute every panel's content once; panels without inputs get an omission reason."""
    b = data.bands
    pa = A.analyse_path(b)
    chans = {n: np.asarray(r, dtype=float) for n, r in E.channels_of(b).items()}
    zero = pa.zero_ev
    if pa.edges.vbm_ev is None and b.fermi_energy is None:
        zero_key = 'energy_abs'
    else:
        zero_key = 'energy_rel_ef' if pa.zero_label == 'F' else 'energy_rel_vbm'
    low = min(float(r.min()) for r in chans.values()) - zero
    top = min(float(r[:, -1].min()) for r in chans.values()) - zero
    if pa.is_metal:
        lo, hi = max(-12.0, low - 0.8), min(6.0, top - 0.05)
    else:
        lo, hi = max(-20.0, low - 0.8), min(float(pa.edges.gap_ev) + 4.5, top - 0.05)
    lo = float(emin) if emin is not None else lo
    hi = float(emax) if emax is not None else hi
    if hi <= lo:
        raise InputError(f'empty energy window [{lo:g}, {hi:g}] eV')
    ctx = AtlasContext(data, pa, pa.kp, chans, zero, zero_key, lo, hi, top, notes=list(data.load_notes) + list(pa.notes))
    _orbital_context(ctx)
    _pdos_context(ctx)
    _edges_context(ctx)
    _engine_context(ctx)
    _plane_context(ctx)
    _composition_context(ctx)
    _header_text(ctx)
    return ctx


def _orbital_context(ctx):
    proj = ctx.data.projections
    if proj is None:
        return
    first = next(iter(proj.channels))
    pc = ctx.pa.channel if ctx.pa.channel in proj.channels else first
    ctx.orbital = A.dominant_character(proj, pc, ctx.pa.rows, ctx.zero, (ctx.lo, ctx.hi))
    for name in ctx.channels:
        source = name if name in proj.channels else (first if len(proj.channels) == 1 else None)
        if source is not None:
            ctx.fractions[name] = A.orbital_fraction(proj, source, ctx.orbital)


def _pdos_context(ctx):
    dos = ctx.data.dos
    if dos is None:
        ctx.omitted['pdos'] = 'no DOS source (sparse-dft dos result or SCF k mesh)'
        return
    curves = []
    if dos.pdos_raw:
        try:
            curves = dos.grouped('element+l')
        except InputError:
            curves = list(dos.pdos)
    else:
        curves = list(dos.pdos)
    e = ctx.rel(dos.energies_ev)
    inside = (e >= ctx.lo) & (e <= ctx.hi)
    weights = {}
    for c in curves:
        key = _group_key(c.label, c.element, c.l)
        weights[key] = weights.get(key, 0.0) + float(_trapz(np.asarray(c.values)[inside], e[inside]))
    total = sum(weights.values()) or 1.0
    keep = [k for k, w in weights.items() if w / total >= 0.025]
    ctx.pdos = [(*_group_key(c.label, c.element, c.l), c) for c in curves if _group_key(c.label, c.element, c.l) in keep]
    ctx.groups = _ordered_groups(keep, ctx.data.species)


def _ordered_groups(keys, species):
    order = {el: i for i, el in enumerate(dict.fromkeys(species or []))}
    ls = ('s', 'p', 'd', 'f')
    return sorted(dict.fromkeys(keys), key=lambda k: (order.get(k[0], 99), ls.index(k[1]) if k[1] in ls else 9))


def _edges_context(ctx):
    pa = ctx.pa
    if pa.is_metal:
        ctx.omitted['edges'] = 'metal: no band edges to fit'
    elif not pa.electron_fits and not pa.hole_fits:
        ctx.omitted['edges'] = '; '.join(pa.notes) or 'no reliable band-edge fit'


def _engine_context(ctx):
    data = ctx.data
    if data.reference is None:
        ctx.omitted['residual'] = ctx.omitted['parity'] = 'no reference band run (--reference)'
        return
    cond = data.conditions or {}
    if not cond.get('matched'):
        why = 'reference run conditions differ: ' + (', '.join(cond.get('differences') or []) or 'no shared k points')
        ctx.omitted['residual'] = ctx.omitted['parity'] = why
        return
    ctx.residuals = A.engine_residuals(data.bands, data.reference, align='none', zero_ev=ctx.zero)
    if data.projections is None or data.reference_projections is None:
        ctx.omitted['parity'] = 'lm projections missing in the candidate or the reference'
        return
    try:
        ctx.parity = A.projection_parity(data.bands, data.projections, data.reference, data.reference_projections)
    except InputError as exc:
        ctx.omitted['parity'] = str(exc)


def _plane_context(ctx):
    mesh, pa = ctx.data.mesh, ctx.pa
    if mesh is None:
        ctx.omitted['bz'] = 'no uniform k mesh (sparse-dft dos/mesh run or the SCF k mesh)'
        return
    mchans = E.channels_of(mesh)
    try:
        if pa.is_metal:
            for name in mchans:
                try:
                    ctx.planes.append((name, A.bz_plane(mesh, quantity='fermi', channel=name, zero_ev=ctx.zero)))
                except InputError as exc:
                    if 'crosses' not in str(exc):
                        raise
            if not ctx.planes:
                raise InputError('no band crosses E_F on the k3 = 0 plane')
        else:
            name = pa.channel if pa.channel in mchans else next(iter(mchans))
            rows = np.asarray(mchans[name], dtype=float)
            if rows.shape[1] <= pa.cb or rows[:, pa.vb].max() > rows[:, pa.cb].min() + 1e-6:
                raise InputError(f'bands {pa.vb + 1}/{pa.cb + 1} are not the valence/conduction edge on the mesh')
            ctx.planes.append((name, A.bz_plane(mesh, quantity='direct', vb=pa.vb, cb=pa.cb, channel=name,
                                                zero_ev=ctx.zero)))
    except InputError as exc:
        ctx.planes = []
        ctx.omitted['bz'] = str(exc)


def _composition_context(ctx):
    data = ctx.data
    if data.mesh is not None and data.mesh_projections is not None:
        src, proj, ctx.composition_source = data.mesh, data.mesh_projections, 'mesh'
    elif data.projections is not None:
        src, proj, ctx.composition_source = data.bands, data.projections, 'path'
    else:
        ctx.omitted['composition'] = 'no lm/l projections'
        return
    for name in E.channels_of(src):
        pc = name if name in proj.channels else (next(iter(proj.channels)) if len(proj.channels) == 1 else None)
        if pc is None:
            continue
        nb = _composition_bands(ctx.pa, src, name)
        ctx.composition.append((name, A.band_composition(proj, src, pc, nbands=nb)))
    if not ctx.composition:
        ctx.omitted['composition'] = 'projection channels do not match the band channels'
        return
    keys = []
    for _, comp in ctx.composition:
        keys += [_group_key(lab, None, None) for lab in comp['order'] if lab != 'other']
    ctx.groups = _ordered_groups(list(ctx.groups) + keys, data.species)


def _classification_text(pa) -> str:
    cls = pa.classification
    if cls == 'metal':
        return tr('classification_metal')
    parts = cls.replace('-gap', '').split()
    key = f'classification_{parts[0]}_{parts[-1]}' if len(parts) == 2 else None
    return tr(key) if key and key in _labels() else cls


def _labels():
    from .labels import LABELS
    return LABELS


def _mass_range(fits) -> str | None:
    if not fits:
        return None
    ms = sorted(abs(f.mass) for f in fits)
    lo, hi = ms[0], ms[-1]
    return f'{lo:.3g}' if abs(hi - lo) < 0.0005 else f'{lo:.3g}–{hi:.3g}'


def _header_text(ctx):
    pa = ctx.pa
    ctx.title = tr('atlas_title', formula=ctx.formula, product=PRODUCT)
    parts = [_classification_text(pa)]
    if not pa.is_metal:
        parts.append(r'$E_\mathrm{g}$ = ' + f'{pa.edges.gap_ev:.3f} eV')
        parts.append(f'VBM {pa.k_name(ctx.data.bands, pa.vbm_k)}, CBM {pa.k_name(ctx.data.bands, pa.cbm_k)}')
        me, mh = _mass_range(pa.electron_fits), _mass_range(pa.hole_fits)
        if me:
            parts.append(r'$m_e^*$ ' + me + r' $m_0$')
        if mh:
            parts.append(r'$m_h^*$ ' + mh + r' $m_0$')
    else:
        n = sum(len([b for b in range(r.shape[1]) if (r[:, b] - ctx.zero).min() < 0 < (r[:, b] - ctx.zero).max()])
                for r in ctx.channels.values())
        parts.append(tr('fermi_crossing', n=n))
    parts.append(tr('cells_atoms', n=len(ctx.data.species)) if ctx.data.species else None)
    cond = ctx.data.conditions or {}
    if cond.get('matched'):
        ref = cond.get('reference') or {}
        parts.append(f"{PRODUCT} vs {ref.get('engine', '')} {ref.get('version') or ''}".strip()
                     + f" ({tr('matched_conditions')})")
    ctx.subtitle = '  ·  '.join(p for p in parts if p)


def format_mass(m, u) -> str:
    """'0.0316(8)': value with the uncertainty in the last digits (|order 2 - order 1|)."""
    m = abs(float(m))
    u = abs(float(u))
    digits = 3
    if u > 0:
        digits = int(min(4, max(2, -np.floor(np.log10(u)))))
    text = f'{m:.{digits}f}'
    last = int(round(u * 10 ** digits))
    return f'{text}({last})' if last >= 1 else text


# --- callouts placed away from data and from each other ------------------------------
def _avoid_points(ax):
    from .qa import _data_points
    return _data_points(ax)


def place_callout(ax, text, xy, *, placed, points, fontsize, color=None, arrow=True, prefer=(0.0, 1.0),
                  pad_px=3.0, zorder=8):
    """Annotate ``xy`` with a boxed ``text`` at the offset with the least data under it.

    Positions on a 23 x 23 grid over the axes are tried (closest clean one wins);
    boxes must stay inside the axes and clear of ``placed`` boxes.  The chosen box is appended
    to ``placed``.  Call after the axes limits are final and the figure has been
    laid out once (``fig.canvas.draw()``).
    """
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    trans = ax.transData
    target = trans.transform(xy)
    ann = ax.annotate(text, xy=xy, xytext=xy, textcoords='data', ha='center', va='center', fontsize=fontsize,
                      color=color or NEUTRAL['ink'], fontweight='bold', zorder=zorder, annotation_clip=False,
                      bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor=NEUTRAL['muted'], lw=0.5),
                      arrowprops=dict(arrowstyle='-', color=NEUTRAL['muted'], lw=0.5, shrinkA=0.0, shrinkB=2.0)
                      if arrow else None)
    axbox = ax.get_window_extent(renderer)
    size = min(axbox.width, axbox.height)
    ann.arrow_patch and ann.arrow_patch.set_visible(False)
    box = ann.get_window_extent(renderer).padded(pad_px)        # size only: the box does not change shape
    w, h = box.width, box.height
    ann.arrow_patch and ann.arrow_patch.set_visible(True)
    px, py = (points[:, 0], points[:, 1]) if len(points) else (np.zeros(0), np.zeros(0))
    best, best_cost = None, None
    for cx in np.linspace(axbox.x0 + w / 2 + 2, axbox.x1 - w / 2 - 2, 31):
        for cy in np.linspace(axbox.y0 + h / 2 + 2, axbox.y1 - h / 2 - 2, 31):
            off = np.array([cx, cy]) - target
            r = float(np.hypot(*off))
            if r < 0.06 * size:
                continue
            bb = Bbox.from_extents(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
            if any(bb.overlaps(other) for other in placed):
                continue
            inside = int(((px > bb.x0) & (px < bb.x1) & (py > bb.y0) & (py < bb.y1)).sum())
            cost = inside * 1000.0 + 60.0 * r / size - 8.0 * float(np.dot(off / r, prefer))
            if best_cost is None or cost < best_cost:
                best, best_cost = (np.array([cx, cy]), bb), cost
    if best is None:
        ann.remove()
        return None
    ann.xyann = tuple(trans.inverted().transform(best[0]))
    placed.append(best[1])
    return ann


# --- colour identity and shared axes -------------------------------------------------
def group_colors(ctx, spec) -> dict:
    """(element, l) -> colour in a fixed order (species order, then s, p, d, f); 'other' is neutral."""
    out = {g: spec.color(i) for i, g in enumerate(ctx.groups)}
    out[('other', None)] = NEUTRAL['band']
    return out


def _group_label(key) -> str:
    el, l = key
    return tr('other') if el == 'other' else (f'{el} ${l}$' if l else str(el))


def _kaxis(ax, kp):
    ax.set_xlim(0.0, kp.length if kp.length > 0 else 1.0)
    ax.set_xticks([t for t, _ in kp.ticks], [label for _, label in kp.ticks])
    for t, name in kp.ticks:
        if 1e-9 < t < kp.length - 1e-9:
            broken = '|' in name
            ax.axvline(t, color=NEUTRAL['ink'] if broken else NEUTRAL['guide'], lw=0.8 if broken else 0.45,
                       zorder=0.6)
    ax.tick_params(axis='x', which='both', length=0, pad=3.0)
    ax.tick_params(axis='x', labelsize=style_of(ax.figure).base_fontsize)


def _zero_lines(ax, ctx, *, vertical=False):
    line = ax.axvline if vertical else ax.axhline
    span = ax.axvspan if vertical else ax.axhspan
    if ctx.pa.is_metal:
        line(0.0, color=NEUTRAL['fermi'], lw=0.7, ls=(0, (4.0, 2.0)), zorder=1.0)
        return
    gap = float(ctx.pa.edges.gap_ev)
    span(0.0, gap, color=NEUTRAL['gap'], lw=0, zorder=0.3)
    line(0.0, color=NEUTRAL['zero'], lw=0.6, ls=(0, (4.0, 2.0)), zorder=1.0)
    line(gap, color=NEUTRAL['zero'], lw=0.6, ls=(0, (1.0, 1.4)), zorder=1.0)


def _incomplete(ax, ctx, *, vertical=False):
    if ctx.top_complete < ctx.hi:
        (ax.axvspan if vertical else ax.axhspan)(ctx.top_complete, ctx.hi, color=NEUTRAL['fail_fill'], lw=0,
                                                 zorder=0.35)


def _segments(kp, x_values, rows, band):
    pieces, weights = [], []
    for s, e in kp.segments:
        for i in range(s, e):
            pieces.append(((x_values[i], rows[i, band]), (x_values[i + 1], rows[i + 1, band])))
            weights.append((i, i + 1))
    return pieces, weights


# --- panel a: bands -------------------------------------------------------------------
def draw_bands(ax, ctx, spec, *, header=None):
    kp = ctx.kp
    x = np.asarray(kp.x, dtype=float)
    ax.set_ylim(ctx.lo, ctx.hi)
    _kaxis(ax, kp)
    _zero_lines(ax, ctx)
    _incomplete(ax, ctx)
    frame_axes(ax, grid='y')
    cmap, norm = orbital_cmap(), _fraction_norm(ctx)
    handles = []
    spin = len(ctx.channels) > 1
    for ci, (name, rows) in enumerate(ctx.channels.items()):
        rel = rows - ctx.zero
        frac = ctx.fractions.get(name)
        ls = '-' if name != 'down' else (0, (3.0, 1.6))
        for b in range(rel.shape[1]):
            col = rel[:, b]
            if col.max() < ctx.lo - 0.5 or col.min() > ctx.hi + 0.5:
                continue
            segs, idx = _segments(kp, x, rel, b)
            if frac is not None:
                f, w = frac
                c = np.array([0.5 * (f[i, b] + f[j, b]) for i, j in idx])
                ww = np.clip(np.array([0.5 * (w[i, b] + w[j, b]) for i, j in idx]), 0.0, 1.0)
                halo = LineCollection(segs, cmap=cmap, norm=norm, linewidths=0.9 + 2.4 * ww, alpha=0.18,
                                      capstyle='round', zorder=2.6, linestyles=ls)
                halo.set_array(c)
                ax.add_collection(halo)
                core = LineCollection(segs, cmap=cmap, norm=norm, linewidths=0.75 + 0.55 * ww, capstyle='round',
                                      joinstyle='round', zorder=3.0, linestyles=ls)
                core.set_array(c)
                ax.add_collection(core)
            else:
                ax.add_collection(LineCollection(segs, colors=[spec.color(ci) if spin else NEUTRAL['ink']],
                                                 linewidths=0.8, zorder=3.0, linestyles=ls))
    handles.append(Line2D([], [], color=NEUTRAL['ink'] if ctx.fractions else spec.color(0), lw=1.2,
                          label=tr('path_dataset', product=PRODUCT, n=ctx.data.bands.nkpoints)))
    shown = _reference_markers(ax, ctx, x)
    if shown:
        n, stride = shown
        key = 'ref_dataset' if stride == 1 else 'ref_dataset_stride'
        handles.append(Line2D([], [], ls='none', marker='o', ms=3.0, mfc='white', mec=NEUTRAL['ink'], mew=0.6,
                              label=tr(key, ref=ctx.ref_engine, n=n, s=stride)))
    if spin:
        handles.append(Line2D([], [], color=NEUTRAL['muted'], lw=0.8, ls=(0, (3.0, 1.6)),
                              label=tr('spin_solid_dashed')))
    ax.set_ylabel(tr(ctx.zero_key))
    values = _gap_marks(ax, ctx, x, spec)
    return {'handles': handles, 'mappable': (cmap, norm) if ctx.fractions else None, **values}


def _fraction_norm(ctx):
    vals = [f[(w > 1e-6)] for f, w in ctx.fractions.values()]
    top = float(np.percentile(np.concatenate(vals), 98)) if vals and sum(v.size for v in vals) else 1.0
    return Normalize(0.0, float(np.clip(top, 0.5, 1.0)))


def _reference_markers(ax, ctx, x):
    """Reference eigenvalues at the shared k points, thinned to about one marker per 3 mm."""
    ref = ctx.data.reference
    cond = ctx.data.conditions or {}
    if ref is None or not cond.get('matched'):
        return None
    pairs = A.match_kpoints(ctx.data.bands, ref)
    width_mm = ax.get_position().width * ax.figure.get_figwidth() * 25.4
    stride = max(1, int(np.ceil(len(pairs) / max(1.0, width_mm / 2.6))))
    rc = E.channels_of(ref)
    nb = A.compared_band_count(ctx.data.bands, ref)
    for cn, rn in A._channel_pairs(ctx.data.bands, ref):
        rows = np.asarray(rc[rn], dtype=float) - ctx.zero
        xs, ys = [], []
        for i, j in pairs[::stride]:
            for n in range(nb):
                if ctx.lo <= rows[j, n] <= ctx.hi:
                    xs.append(x[i]); ys.append(rows[j, n])
        ax.plot(xs, ys, ls='none', marker='o', ms=2.2, mfc='white', mec=NEUTRAL['ink'], mew=0.45, zorder=4.0,
                alpha=0.9)
    return len(pairs), stride


def _gap_marks(ax, ctx, x, spec):
    pa = ctx.pa
    if pa.is_metal:
        return {}
    gap = float(pa.edges.gap_ev)
    vk = list(pa.edges.vbm_kindex or [pa.vbm_k])
    ck = list(pa.edges.cbm_kindex or [pa.cbm_k])
    mid = 0.5 * (x[0] + x[-1])
    kv = min(vk, key=lambda k: abs(x[k] - mid))
    kc = min(ck, key=lambda k: abs(x[k] - x[kv]))
    for k in vk:
        ax.plot(x[k], 0.0, marker='o', ms=3.4, mfc='white', mec=NEUTRAL['fermi'], mew=0.9, zorder=6, ls='none',
                clip_on=False)
    for k in ck:
        ax.plot(x[k], gap, marker='o', ms=3.4, mfc=NEUTRAL['fermi'], mec=NEUTRAL['fermi'], mew=0.9, zorder=6,
                ls='none', clip_on=False)
    ax.annotate('', xy=(x[kc], gap), xytext=(x[kv], 0.0), zorder=6.5,
                arrowprops=dict(arrowstyle='<|-|>', lw=0.7, color=NEUTRAL['fermi'], shrinkA=2.0, shrinkB=2.0,
                                mutation_scale=5))
    return {'gap_anchor': ((x[kv] + x[kc]) / 2.0, gap / 2.0), 'valley_points': [(x[v.kindex], v.energy_rel_ev, v)
                                                                               for v in pa.valleys]}


def annotate_bands(ax, ctx, info, placed):
    """Gap and valley call-outs; run after layout so the free space is measured on the final axes."""
    pa = ctx.pa
    if pa.is_metal or 'gap_anchor' not in info:
        return []
    fs = style_of(ax.figure).small_fontsize
    points = _avoid_points(ax)
    kind = tr('gap_direct') if pa.edges.direct else tr('gap_indirect')
    vname, cname = pa.k_name(ctx.data.bands, pa.vbm_k), pa.k_name(ctx.data.bands, pa.cbm_k)
    head = tr('gap_at', kind=kind, k=vname) if pa.edges.direct else tr('gap_between', kind=kind, a=vname, b=cname)
    text = head + '\n' + r'$E_\mathrm{g}$ = ' + f'{pa.edges.gap_ev:.3f} eV'
    out = []
    ann = place_callout(ax, text, info['gap_anchor'], placed=placed, points=points, fontsize=fs, prefer=(1.0, 0.3))
    out.append(ann)
    for xv, ev, v in info.get('valley_points', [])[1:3]:
        ax.plot(xv, ev, marker='o', ms=3.0, mfc='white', mec=NEUTRAL['fermi'], mew=0.8, zorder=6, ls='none')
        label = tr('valley_callout', label=v.label) + '\n' + tr('above_cbm', v=f'{ev - pa.edges.gap_ev:.3f}')
        out.append(place_callout(ax, label, (xv, ev), placed=placed, points=points, fontsize=fs, prefer=(0.0, 1.0)))
    return [a for a in out if a is not None]


# --- panel b: projected DOS -----------------------------------------------------------
def draw_pdos(ax, ctx, spec, colors):
    dos = ctx.data.dos
    e = ctx.rel(dos.energies_ev)
    inside = (e >= ctx.lo - 0.5) & (e <= ctx.hi + 0.5)
    spin = dos.spin_polarized
    peak = 0.0
    handles = []
    for c in dos.total:
        sign = -1.0 if c.spin == 'down' else 1.0
        v = sign * np.asarray(c.values, dtype=float)
        ax.fill_betweenx(e[inside], 0.0, v[inside], color=NEUTRAL['grid'], lw=0, zorder=1.2)
        ax.plot(v[inside], e[inside], color=NEUTRAL['ink'], lw=0.75, zorder=3.2)
        peak = max(peak, float(np.abs(v[(e >= ctx.lo) & (e <= ctx.hi)]).max(initial=0.0)))
    total_label = tr('sigma_total', s=f'{dos.sigma_ev:g}') if dos.sigma_ev else tr('atlas_dos_total')
    handles.append(Patch(facecolor=NEUTRAL['grid'], edgecolor=NEUTRAL['ink'], lw=0.6, label=total_label))
    seen = set()
    for el, l, c in ctx.pdos:
        key = (el, l)
        color = colors.get(key, NEUTRAL['muted'])
        sign = -1.0 if c.spin == 'down' else 1.0
        v = sign * np.asarray(c.values, dtype=float)
        ax.fill_betweenx(e[inside], 0.0, v[inside], color=tint(color, 0.8), alpha=0.55, lw=0, zorder=2.0)
        ax.plot(v[inside], e[inside], color=color, lw=0.8, ls=L_DASH.get(l, '-'), zorder=3.4)
        if key not in seen:
            seen.add(key)
            handles.append(Line2D([], [], color=color, lw=1.0, ls=L_DASH.get(l, '-'), label=_group_label(key)))
    _zero_lines(ax, ctx)
    _incomplete(ax, ctx)
    complete = (dos.metadata or {}).get('complete_below_ev')
    if complete is not None and complete - ctx.zero < ctx.hi:
        ax.axhspan(complete - ctx.zero, ctx.hi, color=NEUTRAL['fail_fill'], lw=0, zorder=0.35)
    peak = peak or 1.0
    ax.set_xlim((-1.08 * peak, 1.08 * peak) if spin else (0.0, 1.12 * peak))
    if spin:
        ax.axvline(0.0, color=NEUTRAL['ink'], lw=0.4, zorder=3.5)
        from matplotlib.ticker import FuncFormatter
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{abs(v):g}'))
    ax.set_ylim(ctx.lo, ctx.hi)
    ax.set_xlabel(tr('dos'))
    frame_axes(ax, grid='both')
    integral = {}
    for el, l, c in ctx.pdos:
        mask = e <= 0.0
        integral[f'{el} {l}' + (f' ({c.spin})' if c.spin else '')] = float(_trapz(np.asarray(c.values)[mask], e[mask]))
    return {'handles': handles,
            'values': {'origin': dos.origin, 'sigma_ev': dos.sigma_ev, 'source': dos.source,
                       'groups_shown': [f'{el} {l}' for el, l in dict.fromkeys((el, l) for el, l, _ in ctx.pdos)],
                       'group_threshold': 'element+l groups holding >= 2.5 % of the PDOS integral in the window',
                       'pdos_integral_below_zero_states': integral,
                       'complete_below_ev': complete, 'warnings': list(dos.warnings)}}


# --- panel c: band-edge curvature -----------------------------------------------------
def _edge_series(ctx, fits):
    """Samples of each fitted band on both sides of its extremum: (q signed, E - E0) along each fit line."""
    out = []
    kc = A.cartesian_k(ctx.data.bands)
    rows = ctx.pa.rows
    for f in fits:
        seg = [(s, e) for s, e in ctx.kp.segments if s <= f.kindex <= e]
        if not seg:
            continue
        s, e = seg[0]
        x0, e0 = kc[f.kindex], rows[f.kindex, f.band]
        q, en = [], []
        for j in range(s, e + 1):
            d = float(np.linalg.norm(kc[j] - x0))
            if d <= 0.32:
                q.append(d if j >= f.kindex else -d)
                en.append(rows[j, f.band] - e0)
        out.append((f, np.asarray(q), np.asarray(en)))
    return out


def _fit_side(f):
    """+1/-1: the side of the extremum the fit's direction leaves to (trim fits are one-sided)."""
    if not f.kx:
        return 0
    k = np.asarray(f.kx, dtype=float)
    if (k > 0).all():
        return 1
    if (k < 0).all():
        return -1
    return 0


def draw_edges(ax_e, ax_h, ctx, spec):
    """Electron and hole curvatures in two stacked axes (shared q axis, own energy zero each)."""
    pa = ctx.pa
    out = {'electron': [], 'hole': []}
    for ax, fits, carrier, sign in ((ax_e, pa.electron_fits, 'electron', 1.0), (ax_h, pa.hole_fits, 'hole', -1.0)):
        ax.axhline(0.0, color=NEUTRAL['zero'], lw=0.5, ls=(0, (4.0, 2.0)), zorder=1.0)
        ax.axvline(0.0, color=NEUTRAL['guide'], lw=0.5, zorder=0.6)
        frame_axes(ax, grid='both')
        if not fits:
            ax.text(0.5, 0.5, tr('masses_none'), transform=ax.transAxes, ha='center', va='center',
                    fontsize=spec.small_fontsize, color=NEUTRAL['muted'])
            continue
        series = _edge_series(ctx, fits)
        # one colour per fit direction, in a fixed order (palette slots after the orbital groups)
        top = 0.0
        for i, (f, q, en) in enumerate(series):
            color = spec.color((i + 4) % len(spec.colors))
            side = _fit_side(f)
            keep = (q * side >= -1e-12) if side else np.ones_like(q, dtype=bool)
            ax.plot(q[keep], en[keep], ls='none', marker='o', ms=2.0, mfc='white', mec=color, mew=0.55, zorder=4)
            qq = np.linspace(min(0.0, q[keep].min()), max(0.0, q[keep].max()), 120)
            lim = 1.6 * f.k_window_inv_a
            qq = qq[np.abs(qq) <= max(lim, 0.06)]
            ax.plot(qq, f.curve(qq), color=color, lw=0.9, zorder=5,
                    label=f'{f.direction}  {format_mass(f.mass, f.uncertainty)}')
            fitted = np.asarray(f.kx)
            ax.plot(fitted, np.asarray(f.ey), ls='none', marker='o', ms=2.6, mfc=color, mec=color, zorder=6)
            top = max(top, float(np.abs(en[keep][np.abs(q[keep]) <= 0.32]).max(initial=0.0)))
            out[carrier].append(f.as_dict())
        span = min(max(0.15, 1.15 * top), 1.5)
        ax.set_ylim((-0.06 * span, span) if sign > 0 else (-span, 0.06 * span))
        ax.set_xlim(-0.33, 0.33)
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, fontsize=spec.small_fontsize,
                  title=r'$m_e^*/m_0$' if carrier == 'electron' else r'$|m_h^*|/m_0$',
                  title_fontsize=spec.small_fontsize, handlelength=1.0, labelspacing=0.25, borderpad=0.3)
    k_e = pa.k_name(ctx.data.bands, pa.cbm_k)
    k_h = pa.k_name(ctx.data.bands, pa.vbm_k)
    ax_e.set_ylabel(tr('electrons_label'))
    ax_h.set_ylabel(tr('holes_label'))
    ax_h.set_xlabel(tr('k_offset'))
    ax_e.tick_params(labelbottom=False)
    return {'values': {'electron_masses': out['electron'], 'hole_masses': out['hole'], 'cbm_k': k_e, 'vbm_k': k_h,
                       'definition': pa.values().get('effective_mass_definition'),
                       'uncertainty_format': '0.0316(8) = 0.0316 +/- 0.0008: |order 2 - order 1| Richardson',
                       'plotted_samples': 'all path points within 0.32 1/A of the extremum; filled = fitted points'}}


# --- free-space text boxes ------------------------------------------------------------
ANCHORS = ((0.03, 0.97, 'left', 'top'), (0.97, 0.97, 'right', 'top'), (0.03, 0.03, 'left', 'bottom'),
           (0.97, 0.03, 'right', 'bottom'), (0.5, 0.97, 'center', 'top'), (0.5, 0.03, 'center', 'bottom'),
           (0.03, 0.5, 'left', 'center'), (0.97, 0.5, 'right', 'center'))


def place_text_box(ax, text, *, placed, fontsize, anchors=ANCHORS, color=None):
    """A boxed text at the axes anchor with the fewest data samples underneath (corners first)."""
    renderer = ax.figure.canvas.get_renderer()
    points = _avoid_points(ax)
    t = ax.text(0, 0, text, transform=ax.transAxes, fontsize=fontsize, color=color or NEUTRAL['ink'], zorder=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=NEUTRAL['frame'], lw=0.5))
    best = None
    for i, (x, y, ha, va) in enumerate(anchors):
        t.set_position((x, y)); t.set_ha(ha); t.set_va(va)
        bb = t.get_window_extent(renderer).padded(2.0)
        if any(bb.overlaps(o) for o in placed):
            continue
        n = 0
        if len(points):
            px, py = points[:, 0], points[:, 1]
            n = int(((px > bb.x0) & (px < bb.x1) & (py > bb.y0) & (py < bb.y1)).sum())
        cost = n * 100 + i
        if best is None or cost < best[0]:
            best = (cost, (x, y, ha, va), bb)
    if best is None:
        t.remove()
        return None
    x, y, ha, va = best[1]
    t.set_position((x, y)); t.set_ha(ha); t.set_va(va)
    placed.append(best[2])
    return t


# --- panel d: k3 = 0 plane ------------------------------------------------------------
def _plane_basis(plane):
    b = np.asarray(plane['reciprocal_basis'], dtype=float) if plane.get('reciprocal_basis') else np.eye(3)
    b1, b2 = b[0], b[1]
    e1 = b1 / np.linalg.norm(b1)
    e2 = b2 - np.dot(b2, e1) * e1
    e2 = e2 / np.linalg.norm(e2)
    return b1, b2, e1, e2


def _to_plane(plane, f1, f2):
    b1, b2, e1, e2 = _plane_basis(plane)
    k = np.multiply.outer(f1, b1) + np.multiply.outer(f2, b2) if np.ndim(f1) else f1 * b1 + f2 * b2
    return k @ e1, k @ e2


def draw_plane(ax, ctx, spec, *, cax=None):
    planes = ctx.planes
    name, plane = planes[0]
    F1, F2 = np.meshgrid(plane['fine_k1'], plane['fine_k2'], indexing='ij')
    X, Y = _to_plane(plane, F1, F2)
    S1, S2 = np.meshgrid(plane['k1'], plane['k2'], indexing='ij')
    SX, SY = _to_plane(plane, S1, S2)
    values = {'plane': 'k3 = 0 (fractional), one reciprocal cell', 'grid': plane['grid'], 'n_points': plane['n_points'],
              'interpolation': plane['interpolation'], 'channel': name, 'quantity': plane['quantity']}
    handles = []
    mappable = None
    if plane['quantity'] == 'direct':
        Z = np.asarray(plane['fine'][0])
        mappable = ax.pcolormesh(X, Y, Z, cmap=bz_cmap(), shading='gouraud', zorder=1.0, rasterized=True)
        levels = np.linspace(float(Z.min()), float(Z.max()), 7)[1:-1]
        ax.contour(X, Y, Z, levels=levels, colors='white', linewidths=0.4, alpha=0.75, zorder=1.5)
        m = plane['sampled_min']
        mx, my = _to_plane(plane, m['k1'], m['k2'])
        ax.plot(mx, my, marker='o', ms=4.0, mfc=NEUTRAL['fermi'], mec='white', mew=0.7, ls='none', zorder=6)
        values.update({'sampled_min_ev': m['value_ev'], 'sampled_min_k_fractional': [m['k1'], m['k2'], 0.0],
                       'definition': f"E_cb - E_vb per mesh point, bands {ctx.pa.cb + 1}/{ctx.pa.vb + 1}; the minimum "
                                     'is a mesh value, never interpolated'})
        handles.append(Line2D([], [], ls='none', marker='o', ms=3.6, mfc=NEUTRAL['fermi'], mec='white', mew=0.6,
                              label=tr('sampled_min', v=f"{m['value_ev']:.3f}")))
    else:
        colors = [spec.color(i) for i in range(len(spec.colors))]
        idx = 0
        crossing = []
        for cn, pl in planes:
            ls = '-' if cn != 'down' else (0, (3.0, 1.6))
            for lab, g in zip(pl['labels'], pl['fine']):
                G = np.asarray(g)
                if G.min() < 0 < G.max():
                    ax.contour(X, Y, G, levels=[0.0], colors=[colors[idx % len(colors)]], linewidths=1.0,
                               linestyles=[ls], zorder=3)
                    band = lab.split()[-1]
                    text = tr('band_n', n=band) + (f" ({tr('spin_' + cn)})" if cn in ('up', 'down') else '')
                    handles.append(Line2D([], [], color=colors[idx % len(colors)], lw=1.0, ls=ls, label=text))
                    crossing.append({'channel': cn, 'band': int(band)})
                    idx += 1
        values.update({'fermi_cuts': crossing, 'definition': 'E_n(k) - E_F = 0 contour of the trigonometric '
                       'interpolation of band-index-ordered mesh energies; approximate near band crossings'})
    ax.plot(SX.ravel(), SY.ravel(), ls='none', marker='o', ms=1.8, mfc='none', mec=NEUTRAL['ink'] if
            plane['quantity'] != 'direct' else 'white', mew=0.45, zorder=5, alpha=0.9)
    handles.insert(0, Line2D([], [], ls='none', marker='o', ms=2.4, mfc='none', mec=NEUTRAL['ink'], mew=0.5,
                             label=tr('mesh_points', n=plane['n_points'])))
    gx, gy = _to_plane(plane, 0.0, 0.0)
    ax.plot(gx, gy, marker='+', ms=4.5, mew=0.8, color=NEUTRAL['ink'] if plane['quantity'] != 'direct' else 'white',
            ls='none', zorder=6)
    corners = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5], [-0.5, -0.5]])
    a1, a2 = plane['k1'][0], plane['k2'][0]
    corners = corners + np.array([a1 + 0.5, a2 + 0.5])
    cx, cy = _to_plane(plane, corners[:, 0], corners[:, 1])
    ax.plot(cx, cy, color=NEUTRAL['ink'], lw=0.5, zorder=4)
    # the reciprocal cell plus a free strip on top for the key (no data under it)
    xr, yr = (cx.min(), cx.max()), (cy.min(), cy.max())
    padx, pady = 0.04 * (xr[1] - xr[0]), 0.04 * (yr[1] - yr[0])
    ax.set_xlim(xr[0] - padx, xr[1] + padx)
    if mappable is not None:
        # transition map: colour bar on the right, two-entry key in a free strip on top
        ax.set_ylim(yr[0] - pady, yr[1] + 0.20 * (yr[1] - yr[0]))
        ax.legend(handles=handles, loc='upper center', ncol=len(handles), fontsize=spec.small_fontsize,
                  handlelength=1.0, borderpad=0.25, columnspacing=0.8, borderaxespad=0.25)
    else:
        # Fermi cuts: one key entry per crossing band and spin, in a column right of the map
        ax.set_ylim(yr[0] - pady, yr[1] + pady)
        ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.03, 1.0), borderaxespad=0.0,
                  fontsize=spec.small_fontsize, handlelength=1.4, labelspacing=0.25, borderpad=0.3)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(tr('kx_plane'))
    ax.set_ylabel(tr('ky_plane'))
    frame_axes(ax, grid=None)
    if mappable is not None:
        fig = ax.figure
        cb = fig.colorbar(mappable, cax=cax) if cax is not None else fig.colorbar(mappable, ax=ax, fraction=0.05,
                                                                                    pad=0.02)
        cb.set_label(tr('transition_ev'))
        cb.outline.set_linewidth(0.5)
        cb.ax.tick_params(direction='in', length=2.0)
    return {'handles': handles, 'values': values, 'cell_xy': (np.asarray(X).ravel(), np.asarray(Y).ravel())}


# --- panel e: band composition --------------------------------------------------------
def draw_composition(ax, ctx, spec, colors):
    comps = ctx.composition
    nch = len(comps)
    width = 0.78 / nch
    handles, seen = [], set()
    values = {}
    for ci, (name, comp) in enumerate(comps):
        bands = np.asarray(comp['bands'])
        x = bands + (ci - (nch - 1) / 2.0) * width
        bottom = np.zeros(len(bands))
        hatch = '////' if name == 'down' else None
        for lab in comp['order']:
            key = ('other', None) if lab == 'other' else _group_key(lab, None, None)
            share = np.asarray(comp['shares_percent'][lab])
            color = colors.get(key, NEUTRAL['band'])
            ax.bar(x, share, width=width * 0.92, bottom=bottom, color=color, edgecolor='white', lw=0.4,
                   hatch=hatch, zorder=2.5)
            bottom += share
            if key not in seen:
                seen.add(key)
                handles.append(Patch(facecolor=color, edgecolor='white', lw=0.4, label=_group_label(key)))
        sphere = 100.0 * np.asarray(comp['sphere_weight'])
        ax.plot(x, sphere, ls='none', marker='D', ms=2.6, mfc='white', mec=NEUTRAL['ink'], mew=0.6, zorder=4)
        values[name] = {'bands': comp['bands'], 'shares_percent': comp['shares_percent'],
                        'sphere_weight': comp['sphere_weight'], 'k_average': comp['k_average']}
    handles.append(Line2D([], [], ls='none', marker='D', ms=2.6, mfc='white', mec=NEUTRAL['ink'], mew=0.6,
                          label=tr('sphere_weight') + ' (%)'))
    if nch > 1:
        handles.append(Patch(facecolor='white', edgecolor=NEUTRAL['ink'], hatch='////', lw=0.4,
                             label=tr('spin_down_hatched')))
    if ctx.pa.cb is not None:
        ax.axvline(ctx.pa.vb + 1.5, color=NEUTRAL['zero'], lw=0.6, ls=(0, (4.0, 2.0)), zorder=3)
    nb = max(len(c['bands']) for _, c in comps)
    ax.set_xlim(0.4, nb + 0.6)
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=min(nb, 12), min_n_ticks=2))
    ax.set_ylim(0.0, 100.0)
    ax.set_xlabel(tr('band_index'))
    ax.set_ylabel(tr('share_percent'))
    frame_axes(ax, grid='y')
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0,
              fontsize=spec.small_fontsize, handlelength=1.0, labelspacing=0.3)
    first = comps[0][1]
    return {'handles': handles,
            'values': {'channels': values, 'source': ctx.composition_source, 'definition': first['definition'],
                       'sphere_weight_definition': 'BZ-weighted sum of lm weights over atoms and orbitals per band '
                                                   '(< 1: part of the state lies outside the PAW spheres)',
                       'valence_conduction_divider': (ctx.pa.vb + 1.5) if ctx.pa.cb is not None else None}}


# --- panel f: engine residuals -------------------------------------------------------
def draw_residuals(ax, ctx, spec):
    r = ctx.residuals
    e = np.asarray(r['energy_rel_ev'])
    d = np.asarray(r['delta_mev'])
    ch = np.asarray(r['channel'])
    handles = []
    styles = {'up': ('o', NEUTRAL['ink']), 'down': ('^', NEUTRAL['muted'])}
    for name in dict.fromkeys(ch.tolist()):
        m = ch == name
        marker, color = styles.get(name, ('o', NEUTRAL['ink']))
        ax.plot(e[m], d[m], ls='none', marker=marker, ms=1.6, mfc=color, mec='none', alpha=0.55, zorder=3,
                rasterized=True)
        if name in ('up', 'down'):
            handles.append(Line2D([], [], ls='none', marker=marker, ms=3.0, mfc=color, mec='none',
                                  label=tr('spin_' + name)))
    ax.axhline(0.0, color=NEUTRAL['zero'], lw=0.5, zorder=2)
    lim = 1.2 * max(abs(d).max(), 1e-3)
    ax.set_ylim(-lim, 2.25 * lim)          # free strip above the data for the statistics box
    pad = 0.03 * (e.max() - e.min() or 1.0)
    ax.set_xlim(e.min() - pad, e.max() + pad)
    ax.set_xlabel(f'{ctx.ref_engine} ' + tr(ctx.zero_key))
    ax.set_ylabel(tr('band_error_mev'))
    frame_axes(ax, grid='both')
    cond = ctx.data.conditions or {}
    return {'handles': handles,
            'values': {**{k: v for k, v in r.items() if k not in ('energy_rel_ev', 'delta_mev', 'band', 'channel')},
                       'conditions': cond}}


def residual_note(ctx):
    r = ctx.residuals
    spins = f" × {r['n_channels']} spins" if r['n_channels'] > 1 else ''
    return '\n'.join([tr('residual_stats', mae=f"{r['mae_mev']:.3f}", mx=f"{r['max_abs_mev']:.3f}"),
                      tr('residual_count', n=len(r['delta_mev']), k=r['n_kpoints'], b=r['n_bands'], s=spins),
                      tr('no_alignment') if r['align'] == 'none' else 'VBM aligned'])


# --- panel g: projection parity -------------------------------------------------------
L_MARKERS = {'s': 'o', 'p': 's', 'd': '^', 'f': 'D'}


def draw_parity(ax, ctx, spec):
    p = ctx.parity
    x, y = np.asarray(p['reference']), np.asarray(p['candidate'])
    ls, near = np.asarray(p['l']), np.asarray(p['near_degenerate'], dtype=bool)
    top = max(float(x.max()), float(y.max()), 0.05) * 1.06
    ax.plot([0, top], [0, top], color=NEUTRAL['zero'], lw=0.6, ls=(0, (4.0, 2.0)), zorder=2)
    handles = [Line2D([], [], color=NEUTRAL['zero'], lw=0.6, ls=(0, (4.0, 2.0)), label=tr('identity'))]
    for l in ('s', 'p', 'd', 'f'):
        m = (ls == l) & ~near
        if not m.any():
            continue
        shade = {'s': NEUTRAL['ink'], 'p': '#334155', 'd': '#64748B', 'f': '#94A3B8'}[l]
        ax.plot(x[m], y[m], ls='none', marker=L_MARKERS[l], ms=1.8, mfc='none', mec=shade, mew=0.4, alpha=0.7,
                zorder=3, rasterized=True)
        handles.append(Line2D([], [], ls='none', marker=L_MARKERS[l], ms=3.0, mfc='none', mec=shade, mew=0.6,
                              label=f'${l}$'))
    if near.any():
        ax.plot(x[near], y[near], ls='none', marker='o', ms=2.4, mfc='none', mec=NEUTRAL['fermi'], mew=0.55,
                zorder=4, rasterized=True)
        handles.append(Line2D([], [], ls='none', marker='o', ms=3.0, mfc='none', mec=NEUTRAL['fermi'], mew=0.6,
                              label=tr('near_degenerate', mev=f"{p['near_degenerate_ev'] * 1000:.0f}")))
    ax.set_xlim(-0.02 * top, top)
    ax.set_ylim(-0.02 * top, top)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(tr('ref_weight', ref=ctx.ref_engine))
    ax.set_ylabel(tr('cand_weight', cand=PRODUCT))
    frame_axes(ax, grid='both')
    return {'handles': handles,
            'values': {k: v for k, v in p.items() if k not in ('reference', 'candidate', 'l', 'near_degenerate')}}


def parity_note(ctx):
    p = ctx.parity
    nd = p['max_abs_dw_non_degenerate']
    lines = [tr('states_n', n=f"{p['n_values']:,}"), tr('parity_r2', r2=f"{p['r2_identity']:.6f}"),
             tr('parity_max', v=f"{p['max_abs_dw']:.4f}")]
    if nd is not None:
        lines.append(tr('parity_nd', v=f'{nd:.4f}'))
    if p.get('max_abs_dw_pair_summed') is not None:
        lines.append(tr('parity_pair', v=f"{p['max_abs_dw_pair_summed']:.4f}"))
    return '\n'.join(lines)


# --- legends placed where they cover the least data ----------------------------------
LEGEND_LOCS = ('upper left', 'upper right', 'lower right', 'lower left', 'upper center', 'lower center',
               'center left', 'center right')


def best_legend(ax, handles, *, fontsize, ncol=1, locs=LEGEND_LOCS, outside='right', placed=None,
                inside_only=False, extra_points=None, **kw):
    """Legend at the inside location covering the fewest data samples; outside the axes if every one does."""
    if not handles:
        return None
    renderer = ax.figure.canvas.get_renderer()
    points = _avoid_points(ax)
    if extra_points is not None and len(extra_points):
        points = np.concatenate([points.reshape(-1, 2), np.asarray(extra_points, dtype=float)])
    best = None
    for loc in locs:
        leg = ax.legend(handles=handles, loc=loc, fontsize=fontsize, ncol=ncol, **kw)
        bb = leg.get_window_extent(renderer)
        n = 0
        if len(points):
            px, py = points[:, 0], points[:, 1]
            n = int(((px > bb.x0) & (px < bb.x1) & (py > bb.y0) & (py < bb.y1)).sum())
        if placed and any(bb.overlaps(o) for o in placed):
            n += 10 ** 6
        if best is None or n < best[0]:
            best = (n, loc, bb)
        if n == 0:
            break
    if best[0] == 0 or inside_only:
        leg = ax.legend(handles=handles, loc=best[1], fontsize=fontsize, ncol=ncol, **kw)
        if placed is not None:
            placed.append(best[2])
        return leg
    if outside == 'right':
        return ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0,
                         fontsize=fontsize, ncol=ncol, **kw)
    return ax.legend(handles=handles, loc='lower left', bbox_to_anchor=(0.0, 1.02), borderaxespad=0.0,
                     fontsize=fontsize, ncol=max(ncol, min(4, len(handles))), **kw)


# --- rendering ------------------------------------------------------------------------
TITLES = {'bands': 'atlas_bands', 'pdos': 'atlas_pdos', 'edges': 'atlas_edges', 'composition': 'atlas_composition',
          'parity': 'atlas_parity'}


def _title(ctx, key):
    if key == 'bz':
        return tr('atlas_bz_fermi' if ctx.pa.is_metal else 'atlas_bz_direct')
    if key == 'residual':
        return tr('atlas_residual', cand=PRODUCT, ref=ctx.ref_engine)
    if key == 'pdos' and ctx.data.dos is not None and not ctx.pdos:
        return tr('atlas_dos_total')
    return tr(TITLES[key])


def _letters(ctx):
    """Letters a, b, c ... over the panels that are present (no gaps after omissions)."""
    return {k: chr(ord('a') + i) for i, k in enumerate(ctx.available)}



def _draw_panel(fig, slot, key, ctx, spec, colors, *, combined, share=None):
    """Create the axes of one panel in ``slot`` (a SubplotSpec) and draw it; returns (axes list, result)."""
    axes, res = _draw_panel_axes(fig, slot, key, ctx, spec, colors, share=share)
    for i, a in enumerate(axes):
        a.set_label(f'atlas:{key}' + (f':{i}' if i else ''))
    return axes, res


def _draw_panel_axes(fig, slot, key, ctx, spec, colors, *, share=None):
    if key == 'bands':
        ax = fig.add_subplot(slot)
        return [ax], draw_bands(ax, ctx, spec)
    if key == 'pdos':
        ax = fig.add_subplot(slot, sharey=share)
        res = draw_pdos(ax, ctx, spec, colors)
        if share is not None:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel(tr(ctx.zero_key))
        return [ax], res
    if key == 'edges':
        sub = slot.subgridspec(2, 1, hspace=0.08)
        ax_e = fig.add_subplot(sub[0, 0])
        ax_h = fig.add_subplot(sub[1, 0], sharex=ax_e)
        return [ax_e, ax_h], draw_edges(ax_e, ax_h, ctx, spec)
    if key == 'bz':
        ax = fig.add_subplot(slot)
        return [ax], draw_plane(ax, ctx, spec)
    if key == 'composition':
        ax = fig.add_subplot(slot)
        return [ax], draw_composition(ax, ctx, spec, colors)
    if key == 'residual':
        ax = fig.add_subplot(slot)
        return [ax], draw_residuals(ax, ctx, spec)
    if key == 'parity':
        ax = fig.add_subplot(slot)
        return [ax], draw_parity(ax, ctx, spec)
    raise KeyError(key)


def _finish_panel(key, axes, res, ctx, spec, *, placed, legend_inside=True):
    """Text boxes, call-outs and the panel legend (after the layout is final)."""
    ax = axes[0]
    fs = spec.small_fontsize
    if key == 'bands':
        annotate_bands(ax, ctx, res, placed)
    elif key == 'residual':
        place_text_box(ax, residual_note(ctx), placed=placed, fontsize=fs, anchors=ANCHORS[:2] + ANCHORS[4:5])
        if res['handles']:
            best_legend(ax, res['handles'], fontsize=fs, placed=placed)
    elif key == 'parity':
        place_text_box(ax, parity_note(ctx), placed=placed, fontsize=fs,
                       anchors=ANCHORS[:1] + ANCHORS[4:5] + ANCHORS[1:2])
        best_legend(ax, res['handles'], fontsize=fs, placed=placed, locs=('lower right', 'center right',
                                                                          'upper left', 'lower left'))


def _record(ctx, results):
    values = {'analysis': ctx.pa.values(), 'energy_window_ev': [ctx.lo, ctx.hi],
              'bands_complete_below_ev': ctx.top_complete, 'energy_zero': ctx.zero_key,
              'orbital_colour': ({'l': ctx.orbital, 'definition': f'f_{ctx.orbital} = w_{ctx.orbital} / sum_l w_l of '
                                  'the PAW-sphere lm weights per state; line width ~ sphere weight'}
                                 if ctx.orbital else None),
              'omitted_panels': dict(ctx.omitted), 'panels': {}}
    for key, res in results.items():
        values['panels'][key] = res.get('values', {}) if isinstance(res, dict) else {}
    if ctx.residuals is not None:
        values['panels'].setdefault('residual', {})
    if ctx.data.mesh_conditions:
        values['mesh_conditions'] = ctx.data.mesh_conditions
    return values


def _layout(ctx):
    keys = ctx.available
    left = [k for k in ('bands', 'pdos') if k in keys]
    right = [k for k in ('edges', 'bz') if k in keys]
    bottom = [k for k in ('composition', 'residual', 'parity') if k in keys]
    if len(right) == 1 and 'composition' in bottom and len(bottom) > 1:
        # one panel on the right (metals: no band edges): composition fills the column
        right.append('composition')
        bottom.remove('composition')
    return left, right, bottom


def render_combined(ctx: AtlasContext, *, width='atlas', height_mm=None):
    """All present panels in one figure: title strip, shared legends, panels reflowed around omissions."""
    left, right, bottom = _layout(ctx)
    if not left:
        raise InputError('the atlas needs the band panel')
    main_h = 112.0 if right else 92.0
    bottom_h = 58.0 if bottom else 0.0
    height = height_mm or (20.0 + main_h + bottom_h)
    fig = blank_figure(width, height)
    spec = style_of(fig)
    colors = group_colors(ctx, spec)
    letters = _letters(ctx)
    rows = [1.0, main_h / 20.0] + ([bottom_h / 20.0] if bottom else [])
    gs = fig.add_gridspec(len(rows), 1, height_ratios=rows, hspace=0.02)
    head = fig.add_subplot(gs[0, 0])
    head.set_axis_off()
    widths = {'bands': 2.35, 'pdos': 0.95}
    cols = [widths[k] for k in left] + ([1.55] if right else [])
    main = gs[1, 0].subgridspec(1, len(cols), width_ratios=cols, wspace=0.05)
    results, axes_of = {}, {}
    share = None
    for i, key in enumerate(left):
        axes, res = _draw_panel(fig, main[0, i], key, ctx, spec, colors, combined=True,
                                share=share if key == 'pdos' else None)
        share = axes[0] if key == 'bands' else share
        results[key], axes_of[key] = res, axes
    if right:
        rsub = main[0, len(left)].subgridspec(len(right), 1, height_ratios=[1.25 if k == 'edges' else 1.0
                                                                           for k in right], hspace=0.14)
        for i, key in enumerate(right):
            axes, res = _draw_panel(fig, rsub[i, 0], key, ctx, spec, colors, combined=True)
            results[key], axes_of[key] = res, axes
    if bottom:
        bw = {'composition': 1.45, 'residual': 1.25, 'parity': 0.9}
        bsub = gs[2, 0].subgridspec(1, len(bottom), width_ratios=[bw[k] for k in bottom], wspace=0.06)
        for i, key in enumerate(bottom):
            axes, res = _draw_panel(fig, bsub[0, i], key, ctx, spec, colors, combined=True)
            results[key], axes_of[key] = res, axes
    for key, axes in axes_of.items():
        panel_title(axes[0], letters[key], _title(ctx, key))
    _header(fig, head, ctx, spec, results)
    _ensure_canvas(fig)
    fig.canvas.draw()                    # fix the constrained layout, then place text in the free space
    fig.set_layout_engine('none')
    placed = []
    for key in axes_of:
        _finish_panel(key, axes_of[key], results[key], ctx, spec, placed=placed, legend_inside=True)
    attach_meta(fig, 'electronic_atlas', [v for v in ctx.data.sources.values()],
                notes=ctx.notes + [f'omitted {k}: {v}' for k, v in ctx.omitted.items()],
                values=_record(ctx, results))
    return fig


def _orbital_label(ctx):
    return tr('orbital_fraction', l=ctx.orbital or 's')


def _header(fig, head, ctx, spec, results):
    """Title, subtitle and one strip of shared keys: datasets | orbital colour bar | PDOS groups."""
    fs = spec.small_fontsize
    head.text(0.5, 1.0, ctx.title, transform=head.transAxes, ha='center', va='top', fontweight='bold',
              fontsize=min(8.0, spec.base_fontsize + 1.0), color=NEUTRAL['ink'])
    head.text(0.5, 0.60, ctx.subtitle, transform=head.transAxes, ha='center', va='top', style='italic',
              fontsize=fs, color=NEUTRAL['text'])
    legend_kw = dict(fontsize=fs, borderaxespad=0.0, handlelength=1.4, columnspacing=1.0)
    pd = (results.get('pdos') or {}).get('handles')
    if pd:
        head.add_artist(head.legend(handles=pd, loc='lower right', bbox_to_anchor=(1.0, 0.0),
                                    ncol=min(len(pd), 4), **legend_kw))
    band_res = results.get('bands') or {}
    if band_res.get('handles'):
        head.legend(handles=band_res['handles'], loc='lower left', bbox_to_anchor=(0.0, 0.0),
                    ncol=1 if len(band_res['handles']) > 2 else len(band_res['handles']), **legend_kw)
    if band_res.get('mappable'):
        from matplotlib.cm import ScalarMappable
        cmap, norm = band_res['mappable']
        cax = head.inset_axes([0.42, 0.04, 0.16, 0.11])
        cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal')
        cb.set_label(_orbital_label(ctx), fontsize=fs, labelpad=1.0)
        cb.ax.xaxis.set_label_position('top')
        cb.ax.tick_params(labelsize=fs, length=1.6, direction='in', pad=1.0)
        cb.outline.set_linewidth(0.5)


def render_split(ctx: AtlasContext):
    """One figure per present panel, same styling; each carries its own legend/colour bar and record."""
    out = {}
    letters = _letters(ctx)
    for key in ctx.available:
        w, h = SPLIT_SIZE_MM[key]
        fig = blank_figure(w, h)
        spec = style_of(fig)
        colors = group_colors(ctx, spec)
        gs = fig.add_gridspec(1, 1)
        axes, res = _draw_panel(fig, gs[0, 0], key, ctx, spec, colors, combined=False)
        panel_title(axes[0], letters[key], _title(ctx, key))
        fs = spec.small_fontsize
        if key == 'bands':
            if res.get('mappable'):
                from matplotlib.cm import ScalarMappable
                cmap, norm = res['mappable']
                cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes[0], location='right', fraction=0.04,
                                  pad=0.02, aspect=30, shrink=0.6)
                cb.set_label(_orbital_label(ctx), fontsize=fs, labelpad=1.0)
                cb.ax.tick_params(labelsize=fs, length=1.6, direction='in', pad=1.0)
                cb.outline.set_linewidth(0.5)
            axes[0].legend(handles=res['handles'], loc='lower right', bbox_to_anchor=(1.0, 1.0), fontsize=fs,
                           borderaxespad=0.3, ncol=1, handlelength=1.4)
        elif key == 'pdos' and res.get('handles'):
            axes[0].legend(handles=res['handles'], loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=fs,
                           borderaxespad=0.0, handlelength=1.4)
        _ensure_canvas(fig)
        fig.canvas.draw()
        fig.set_layout_engine('none')
        _finish_panel(key, axes, res, ctx, spec, placed=[], legend_inside=True)
        attach_meta(fig, f'electronic_atlas.{key}', list(ctx.data.sources.values()),
                    notes=ctx.notes, values={'panel': key, 'letter': letters[key], **(res.get('values') or {}),
                                             'analysis': ctx.pa.values() if key in ('bands', 'edges') else None,
                                             'energy_window_ev': [ctx.lo, ctx.hi]})
        out[key] = fig
    return out


def render_atlas(ctx: AtlasContext, panels='combined', *, width='atlas'):
    """{'combined': Figure} / {key: Figure, ...} / both, for ``panels`` in PANEL_MODES."""
    if panels not in PANEL_MODES:
        raise InputError(f'panels must be one of {", ".join(PANEL_MODES)}')
    out = {}
    if panels in ('combined', 'both'):
        out['combined'] = render_combined(ctx, width=width)
    if panels in ('split', 'both'):
        out.update(render_split(ctx))
    return out


def plot_electronic_atlas(data, ax=None, *, emin=None, emax=None, **_opts):
    """Figure-registry entry: the combined atlas (``ax`` is not supported: the atlas owns its layout)."""
    if ax is not None:
        raise InputError('electronic_atlas draws its own multi-panel layout; ax is not supported')
    ctx = analyse_atlas(data, emin=emin, emax=emax)
    return render_combined(ctx)
