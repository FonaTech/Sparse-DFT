"""Research-grade band figures: fixed element/orbital colours, fat bands in four
modes, band structure with a PDOS side panel, DOS comparison and a Brillouin-zone
path inset.

Colour identity is global, never per figure: s/p/d/f take palette slots 0-3, an
m component is a lighter shade of its l with its own dash, and an element keeps
one colour (by atomic number) in every figure of a band suite.
"""
from __future__ import annotations

import math

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from ..analysis.electronic import (BandStructure, DosData, ProjectionSet, LM_ORDER, L_ORDER, band_edges,
                                   channels_of, kpath, orbital_l)
from ..errors import InputError
from ._util import ZERO_KW, attach_meta, axes_or_new, minor, note
from .labels import tr
from .style import NEUTRAL, style_of, tint

FAT_MODES = ('width', 'scatter', 'dominant', 'rgb')
# Shade (blend toward white) and dash per m within one l, in VASP order.
_M_SHADE = (0.0, 0.32, 0.55, 0.15, 0.42, 0.65, 0.25, 0.50)
_M_DASH = ('-', (0, (4.0, 1.4)), (0, (1.2, 1.2)), (0, (4.0, 1.2, 1.2, 1.2)), (0, (2.4, 1.2)),
           (0, (6.0, 1.6)), (0, (1.2, 2.4)), (0, (3.0, 1.0, 1.0, 1.0, 1.0, 1.0)))
# Element colours beyond the palette: a fixed, colour-blind-checked extension
# (Paul Tol "muted"), indexed by atomic number so identities never shuffle.
_ELEMENT_EXTRA = ('#332288', '#88CCEE', '#44AA99', '#117733', '#999933', '#DDCC77', '#CC6677', '#882255',
                  '#AA4499')
# Weights in the RGB mode: red, green, blue channels.
_RGB = ((0.84, 0.15, 0.10), (0.10, 0.62, 0.20), (0.10, 0.30, 0.85))


# --- colour identity ----------------------------------------------------------
def _atomic_number(symbol) -> int:
    try:
        from ..symmetry import _numbers
        return int(_numbers([symbol])[0])
    except Exception:  # unknown or synthetic label: a stable hash
        return 200 + sum(ord(c) for c in str(symbol))


def element_colors(elements, spec) -> dict:
    """Element -> colour, stable across figures: sorted by Z, palette slots after s/p/d/f."""
    unique = sorted(set(e for e in elements if e), key=_atomic_number)
    pool = list(spec.colors[4:]) + list(_ELEMENT_EXTRA) + list(spec.colors[:4])
    return {e: pool[i % len(pool)] for i, e in enumerate(unique)}


def orbital_style(orbital, spec) -> tuple:
    """(colour, dash) of an l channel ('p') or an m component ('py')."""
    l = orbital_l(orbital)
    base = spec.color(L_ORDER.index(l)) if l in L_ORDER else NEUTRAL['muted']
    if orbital in L_ORDER or orbital not in LM_ORDER:
        return base, '-'
    same_l = [o for o in LM_ORDER if orbital_l(o) == l]
    m = same_l.index(orbital)
    shade = _M_SHADE[m % len(_M_SHADE)]
    return (tint(base, shade) if shade else base), _M_DASH[m % len(_M_DASH)]


def group_style(name, by, spec, elements) -> tuple:
    """(colour, dash) of a projection/PDOS group under a grouping (see analysis GROUPINGS)."""
    parts = str(name).split(' ')
    if by in ('l', 'lm', 'orbital'):
        return orbital_style(parts[-1], spec)
    if by in ('element', 'atom'):
        site = parts[0].rstrip('0123456789')
        return element_colors(elements, spec).get(site, NEUTRAL['muted']), '-'
    # element+orbital: one element -> orbital colours; several -> element colour, orbital dash
    site = parts[0].rstrip('0123456789')
    colour, dash = orbital_style(parts[-1], spec)
    if len(set(e for e in elements if e)) > 1:
        colour = element_colors(elements, spec).get(site, colour)
        dash = ('-', (0, (4.0, 1.4)), (0, (1.2, 1.2)), (0, (4.0, 1.2, 1.2, 1.2)))[
            L_ORDER.index(orbital_l(parts[-1])) if orbital_l(parts[-1]) in L_ORDER else 0] \
            if by == 'element+l' or by == 'atom+l' else dash
    return colour, dash


# --- shared helpers ---------------------------------------------------------------
def _energy_zero(bands, dos=None, *, align='fermi', fermi=None):
    from .electronic_plots import energy_reference
    if align == 'fermi' and fermi is None and dos is not None and dos.fermi_energy_ev is not None:
        return energy_reference(None, dos, align='fermi')
    return energy_reference(bands, dos, align=align, fermi=fermi)


def _window(bands, shift, emin, emax, align):
    from .electronic_plots import _band_window
    values = [e - shift for rows in channels_of(bands).values() for row in rows for e in row]
    return _band_window(values, emin, emax) if align != 'none' else _band_window(values, emin, emax, (-1e9, 1e9))


def _segments_xy(kp, rows, band, shift):
    """[(x0, e0), (x1, e1)] pieces of one band, path breaks not joined."""
    out = []
    for s, e in kp.segments:
        for i in range(s, e):
            out.append(((kp.x[i], rows[i][band] - shift), (kp.x[i + 1], rows[i + 1][band] - shift)))
    return out


def _segment_weights(kp, w, band):
    return [0.5 * (w[i][band] + w[i + 1][band]) for s, e in kp.segments for i in range(s, e)]


def _visible_peak(groups, energies, shift, lo, hi):
    """Largest weight on bands inside the energy window (high bands above it can exceed 1)."""
    peak = 0.0
    for w in groups.values():
        for k, row in enumerate(w):
            for b, v in enumerate(row):
                if lo <= energies[k][b] - shift <= hi:
                    peak = max(peak, v)
    return peak or 1.0


def _projection_qa(proj, channel, energies, shift, lo, hi):
    """Sum of weights over atoms and orbitals per state inside the window (<= 1 for sphere weights)."""
    tensor = proj.channels[channel]
    worst = 0.0
    for k, row in enumerate(tensor):
        for b, atoms in enumerate(row):
            if lo <= energies[k][b] - shift <= hi:
                worst = max(worst, sum(sum(a) for a in atoms))
    return worst


def _gap_annotation(ax, bands, kp, shift, spec, lo, hi, arrow=True):
    """VBM/CBM markers, a shaded gap and a direct/indirect arrow; returns the edges."""
    edges = band_edges(bands)
    if edges.vbm_ev is None:
        if edges.is_metal:
            note(ax, tr('metal'), loc='above right')
        return edges
    v, c = edges.vbm_ev - shift, edges.cbm_ev - shift
    if lo < v < hi or lo < c < hi:
        ax.axhspan(v, c, color=tint(spec.color(5), 0.88), lw=0, zorder=0.4)
    for k in edges.vbm_kindex:
        ax.plot(kp.x[k], v, marker='o', ms=3.4, mfc='white', mec=spec.color(0), mew=0.9, zorder=7, ls='none',
                clip_on=False)
    for k in edges.cbm_kindex:
        ax.plot(kp.x[k], c, marker='o', ms=3.4, mfc=spec.color(1), mec=spec.color(1), mew=0.9, zorder=7, ls='none',
                clip_on=False)
    if arrow and edges.vbm_kindex and edges.cbm_kindex:
        x0, x1 = kp.x[edges.vbm_kindex[0]], kp.x[edges.cbm_kindex[0]]
        ax.annotate('', xy=(x1, c), xytext=(x0, v), zorder=8,
                    arrowprops=dict(arrowstyle='-|>', lw=0.7, color=NEUTRAL['ink'], shrinkA=2.2, shrinkB=2.2,
                                    mutation_scale=6))
    kind = tr('gap_direct') if edges.direct else tr('gap_indirect')
    text = r'$E_\mathrm{g}$' + f' = {edges.gap_ev:.3f} eV ({kind})'
    if edges.status == 'inferred':
        text += f'; {tr("inferred")}'
    note(ax, text, loc='above right', color=NEUTRAL['ink'])
    return edges


def _fermi_crossing(ax, bands, kp, shift, lo, hi, spec):
    """Highlight bands that cross the energy zero (metals)."""
    count = 0
    for name, rows in channels_of(bands).items():
        for b in range(len(rows[0])):
            values = [r[b] - shift for r in rows]
            if min(values) < 0.0 < max(values):
                segs = [seg for seg in _segments_xy(kp, rows, b, shift)]
                ax.add_collection(LineCollection(segs, colors=[spec.color(1)], linewidths=1.1, zorder=4,
                                                 capstyle='round'))
                count += 1
    return count


# --- Brillouin-zone inset -------------------------------------------------------
_VIEW = (np.radians(22.0), np.radians(-58.0))     # elevation, azimuth of the inset projection


def _project(points, view=_VIEW):
    elev, azim = view
    rz = np.array([[math.cos(azim), -math.sin(azim), 0], [math.sin(azim), math.cos(azim), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, math.cos(elev), -math.sin(elev)], [0, math.sin(elev), math.cos(elev)]])
    p = np.asarray(points, dtype=float) @ (rx @ rz).T
    return p[:, [0, 2]], p[:, 1]                        # screen (x, z), depth y (larger = farther)


def brillouin_inset(ax, bands: BandStructure, spec, *, bounds=(0.62, 0.56, 0.38, 0.44)):
    """First Brillouin zone of the band file's reciprocal cell with the k path and labels.

    Needs ``metadata.reciprocal_basis`` (Sparse-DFT bands, vasprun.xml) and scipy (seekpath's
    Voronoi construction). The zone is that of the cell the path was computed in.
    """
    basis = (bands.metadata or {}).get('reciprocal_basis')
    if not basis:
        raise InputError(f'{bands.source} has no reciprocal basis; the Brillouin-zone inset needs Sparse-DFT '
                         'bands.json or vasprun.xml')
    try:
        from seekpath.brillouinzone.brillouinzone import BZ
    except ImportError as exc:
        raise InputError('the Brillouin-zone inset needs seekpath and scipy (pip install "sparse_dft[bz]")') from exc
    import warnings
    b = np.asarray(basis, dtype=float)
    if not (bands.metadata or {}).get('reciprocal_basis_includes_2pi', True):
        b = 2 * math.pi * b
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        zone = BZ(*b)
    inset = ax.inset_axes(bounds, zorder=9)
    inset.set_axis_off()
    inset.set_aspect('equal')
    depth_mid = float(np.mean(_project(np.concatenate(zone.faces))[1]))
    for face in zone.faces:
        xy, depth = _project(face)
        hidden = float(np.mean(depth)) > depth_mid
        closed = np.vstack([xy, xy[:1]])
        inset.plot(closed[:, 0], closed[:, 1], color=NEUTRAL['muted'], lw=0.35 if hidden else 0.55,
                   ls=(0, (1.2, 1.2)) if hidden else '-', zorder=1)
    cart = np.asarray(bands.kpoints, dtype=float) @ b
    kp = kpath(bands)
    xy, _ = _project(cart)
    for s, e in kp.segments:
        inset.plot(xy[s:e + 1, 0], xy[s:e + 1, 1], color=spec.color(1), lw=0.9, zorder=3, solid_capstyle='round')
    seen = set()
    for i, label in enumerate(bands.labels or []):
        if not label:
            continue
        key = (round(float(xy[i, 0]), 4), round(float(xy[i, 1]), 4))
        inset.plot(*xy[i], marker='o', ms=1.6, color=spec.color(1), zorder=4)
        if key in seen:
            continue
        seen.add(key)
        from ..analysis.electronic import pretty_kpoint_label
        inset.annotate(pretty_kpoint_label(label) or str(label), xy[i], xytext=(1.5, 1.5),
                       textcoords='offset points', fontsize=spec.small_fontsize - 0.5, color=NEUTRAL['ink'], zorder=5)
    inset.autoscale_view()
    return inset


def decorate_bands(ax, bands, kp, shift, spec, lo, hi, *, gap_arrow=True, fermi_bands=False, bz_inset=False):
    """Gap shading/arrow, E_F-crossing bands and the BZ inset on an existing band axis."""
    edges = _gap_annotation(ax, bands, kp, shift, spec, lo, hi, arrow=gap_arrow)
    crossing = _fermi_crossing(ax, bands, kp, shift, lo, hi, spec) if fermi_bands else 0
    if bz_inset:
        brillouin_inset(ax, bands, spec)
    return edges, crossing


def plot_bands_spin(data: BandStructure, ax=None, *, emin=None, emax=None, fermi=None, align='fermi',
                    layout='side', gap_arrow=True, **_opts):
    """Collinear spin bands: majority and minority side by side (``layout='side'``) or overlaid."""
    from .electronic_plots import _kaxis
    channels = channels_of(data)
    if set(channels) != {'up', 'down'}:
        raise InputError(f'{data.source} is not spin polarised (channels {list(channels)})')
    if layout not in ('side', 'overlay'):
        raise InputError("bands_spin layout must be 'side' or 'overlay'")
    ncols = 2 if layout == 'side' else 1
    if ax is None:
        from .style import current_style, new_figure
        fig, axes = new_figure(current_style().width, height_mm=62.0, nrows=1, ncols=ncols, sharey=True,
                               **({'gridspec_kw': {'wspace': 0.04}} if ncols == 2 else {}))
    else:
        fig, axes = axes_or_new(ax, nrows=1, ncols=ncols)
    spec = style_of(fig)
    kp = kpath(data)
    shift, ylabel, info = _energy_zero(data, align=align, fermi=fermi)
    lo, hi = _window(data, shift, emin, emax, align)
    colours = {'up': spec.color(0), 'down': spec.color(1)}
    dashes = {'up': '-', 'down': '-' if layout == 'side' else (0, (3.2, 1.6))}
    handles = []
    for i, name in enumerate(('up', 'down')):
        a = axes[0, i if ncols == 2 else 0]
        rows = channels[name]
        segs = [seg for b in range(len(rows[0])) for seg in _segments_xy(kp, rows, b, shift)]
        a.add_collection(LineCollection(segs, colors=[colours[name]], linewidths=0.8, linestyles=dashes[name],
                                        zorder=3 + i * 0.1, capstyle='round'))
        handles.append(Line2D([], [], color=colours[name], lw=0.8, ls=dashes[name],
                              label=tr('spin_up' if name == 'up' else 'spin_down')))
        if ncols == 2 or i == 0:
            a.set_ylim(lo, hi)
            _kaxis(a, kp)
            if align != 'none':
                a.axhline(0.0, **ZERO_KW)
            minor(a, x=False)
        if ncols == 2:
            # caption under the k axis: above the panel it would meet the gap note
            a.set_xlabel(tr('spin_up' if name == 'up' else 'spin_down'))
    axes[0, 0].set_ylabel(tr(ylabel))
    if ncols == 2:
        axes[0, 1].tick_params(axis='y', labelleft=False)
    else:
        axes[0, 0].legend(handles=handles, loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol=2,
                          fontsize=spec.small_fontsize, borderaxespad=0.1)
    edges = band_edges(data)
    if edges.vbm_ev is not None:
        _gap_annotation(axes[0, -1], data, kp, shift, spec, lo, hi, arrow=gap_arrow)
    values = {'nkpoints': data.nkpoints, 'nbands': data.nbands, 'layout': layout, 'energy_window_ev': [lo, hi],
              'vbm_ev': edges.vbm_ev, 'cbm_ev': edges.cbm_ev, 'gap_ev': edges.gap_ev, **info}
    attach_meta(fig, 'bands_spin', [data.source], parameters={'align': align, 'emin': emin, 'emax': emax,
                                                              'layout': layout}, values=values)
    return fig


# --- fat bands ------------------------------------------------------------------
def _draw_fat_width(a, kp, energies, grouped, styles, shift, peak, nbands, max_width):
    handles = []
    for gi, (gname, w) in enumerate(grouped.items()):
        colour, dash = styles[gname]
        segs, widths = [], []
        for b in range(nbands):
            segs += _segments_xy(kp, energies, b, shift)
            widths += [max_width * v / peak for v in _segment_weights(kp, w, b)]
        a.add_collection(LineCollection(segs, colors=[colour], linewidths=widths, alpha=0.75, zorder=3 + gi * 0.01,
                                        capstyle='round', label=str(gname)))
        handles.append(Line2D([], [], color=colour, lw=2.0, ls=dash, label=str(gname)))
    return handles


def _draw_fat_scatter(a, kp, energies, grouped, styles, shift, peak, nbands, max_width):
    handles = []
    xs = np.repeat(np.asarray(kp.x, dtype=float)[:, None], nbands, axis=1)
    ys = np.asarray(energies, dtype=float)[:, :nbands] - shift
    for gi, (gname, w) in enumerate(grouped.items()):
        colour, _ = styles[gname]
        size = (np.asarray(w, dtype=float)[:, :nbands] / peak) * (2.2 * max_width) ** 2
        keep = size > 0.02
        a.scatter(xs[keep], ys[keep], s=size[keep], color=colour, alpha=0.55, lw=0, zorder=3 + gi * 0.01)
        handles.append(Line2D([], [], color=colour, ls='none', marker='o', ms=4.0, label=str(gname)))
    return handles


def _draw_fat_dominant(a, kp, energies, grouped, styles, shift, peak, nbands, max_width):
    names = list(grouped)
    stack = np.asarray([grouped[n] for n in names], dtype=float)          # [group][k][band]
    segs, colours = [], []
    for b in range(nbands):
        pieces = _segments_xy(kp, energies, b, shift)
        seg_w = np.asarray([_segment_weights(kp, stack[g], b) for g in range(len(names))])   # [group][segment]
        total = seg_w.sum(axis=0)
        best = seg_w.argmax(axis=0)
        for piece, g, t, wmax in zip(pieces, best, total, seg_w.max(axis=0)):
            share = float(wmax / t) if t > 0 else 0.0
            colour = styles[names[g]][0] if t > 1e-3 * peak else NEUTRAL['guide']
            segs.append(piece)
            colours.append(tint(colour, 0.65 * (1.0 - share)))
    a.add_collection(LineCollection(segs, colors=colours, linewidths=0.4 * max_width, zorder=3, capstyle='round'))
    return [Line2D([], [], color=styles[n][0], lw=2.0, label=str(n)) for n in names]


def _draw_fat_rgb(a, kp, energies, grouped, styles, shift, peak, nbands, max_width):
    names = list(grouped)
    if not 2 <= len(names) <= 3:
        raise InputError(f'rgb fat bands need 2 or 3 groups, got {len(names)} ({names}); narrow them with '
                         '--group, --atoms or --orbitals')
    stack = np.asarray([grouped[n] for n in names], dtype=float)
    base = np.asarray(_RGB[:len(names)])
    segs, colours = [], []
    for b in range(nbands):
        pieces = _segments_xy(kp, energies, b, shift)
        seg_w = np.asarray([_segment_weights(kp, stack[g], b) for g in range(len(names))]).T     # [segment][group]
        total = seg_w.sum(axis=1)
        for piece, w, t in zip(pieces, seg_w, total):
            mix = (w / t) @ base if t > 0 else np.full(3, 0.75)
            strength = min(1.0, float(t / peak)) if t > 0 else 0.0
            segs.append(piece)
            colours.append(tuple(np.clip(mix * strength + 0.82 * (1.0 - strength), 0.0, 1.0)))
    a.add_collection(LineCollection(segs, colors=colours, linewidths=0.4 * max_width, zorder=3, capstyle='round'))
    return [Line2D([], [], color=tuple(base[i]), lw=2.0, label=str(n)) for i, n in enumerate(names)]


_FAT_DRAW = {'width': _draw_fat_width, 'scatter': _draw_fat_scatter, 'dominant': _draw_fat_dominant,
             'rgb': _draw_fat_rgb}


def plot_fatbands(data, ax=None, *, emin=None, emax=None, fermi=None, align='fermi', group='l', mode='width',
                  atoms=None, orbitals=None, channel=None, max_width=3.2, gap_arrow=False, bz_inset=False, **_opts):
    """Projected bands. ``group`` is any analysis GROUPINGS ('l', 'lm', 'element', 'element+l', ...);
    ``mode`` is 'width' (line width), 'scatter' (marker area), 'dominant' (colour of the largest
    group, paler when mixed) or 'rgb' (2-3 groups blended as colour). ``atoms`` / ``orbitals``
    restrict the projection ('1-4', 'Ga', 'd', 'dxy,dyz'). SOC magnetisation channels use a
    signed diverging scale.
    """
    from .electronic_plots import _band_window, _kaxis
    bands, proj = data
    if mode not in FAT_MODES:
        raise InputError(f'fat-band mode must be one of {FAT_MODES}')
    channels = [channel] if channel else list(proj.channels)
    for c in channels:
        if c not in proj.channels:
            raise InputError(f'projection channel {c!r} not in {list(proj.channels)}')
    n = len(channels)
    fig, axes = axes_or_new(ax, nrows=n, ncols=1, sharex=True, height_mm=8.0 + 54.0 * n)
    spec = style_of(fig)
    kp = kpath(bands)
    shift, label, info = _energy_zero(bands, align=align, fermi=fermi)
    elements = proj.species or []
    groups_meta, qa_sum = {}, {}
    for row, name in enumerate(channels):
        a = axes[row, 0]
        energies = (bands.spin_channels or {}).get(name, bands.eigenvalues)
        values = [e - shift for r in energies for e in r]
        lo, hi = _band_window(values, emin, emax)
        base = [seg for b in range(bands.nbands) for seg in _segments_xy(kp, energies, b, shift)]
        a.add_collection(LineCollection(base, colors=[NEUTRAL['guide']], linewidths=0.4, zorder=2))
        if name in ('mx', 'my', 'mz'):
            _signed_channel(a, fig, kp, energies, proj.channels[name], shift, spec, bands.nbands, name)
            groups_meta[name] = ['signed']
        else:
            grouped = proj.grouped(name, by=group, atoms=atoms, orbitals=orbitals)
            styles = {g: group_style(g, group, spec, elements) for g in grouped}
            peak = _visible_peak(grouped, energies, shift, lo, hi)
            handles = _FAT_DRAW[mode](a, kp, energies, grouped, styles, shift, peak, bands.nbands, max_width)
            ncol = min(5 if len(handles) > 6 else 4, len(handles))
            a.legend(handles=handles, loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol=ncol,
                     fontsize=spec.small_fontsize, borderaxespad=0.1, handlelength=1.4, columnspacing=0.8)
            groups_meta[name] = list(grouped)
            qa_sum[name] = _projection_qa(proj, name, energies, shift, lo, hi)
        a.set_ylim(lo, hi)
        _kaxis(a, kp)
        if align != 'none':
            a.axhline(0.0, **ZERO_KW)
        a.set_ylabel(tr(label))
        minor(a, x=False)
        if n > 1:
            a.text(0.01, 0.97, {'up': tr('spin_up'), 'down': tr('spin_down')}.get(name, name), transform=a.transAxes,
                   ha='left', va='top', fontsize=spec.small_fontsize, color=NEUTRAL['ink'])
        if row < n - 1:
            a.tick_params(labelbottom=False)
        if row == 0 and (gap_arrow or bz_inset):
            decorate_bands(a, bands, kp, shift, spec, lo, hi, gap_arrow=gap_arrow, bz_inset=bz_inset)
    meta_notes = [proj.weight_definition] if proj.weight_definition else []
    if any(v > 1.0 + 1e-3 for v in qa_sum.values()):
        meta_notes.append('sum of sphere weights exceeds 1 inside the window: check the projection source')
    attach_meta(fig, 'fatbands', [bands.source, proj.source],
                parameters={'group': group, 'mode': mode, 'atoms': atoms, 'orbitals': orbitals, 'channel': channel,
                            'align': align, 'emin': emin, 'emax': emax},
                notes=meta_notes,
                values={**info, 'groups': groups_meta, 'resolution': proj.resolution,
                        'peak_weight_scale': 'per panel, bands inside the energy window',
                        'max_state_weight_sum_in_window': qa_sum})
    return fig


def _signed_channel(a, fig, kp, energies, tensor, shift, spec, nbands, name):
    w = [[sum(sum(atom) for atom in tensor[k][b]) for b in range(nbands)] for k in range(len(tensor))]
    peak = max((abs(v) for r in w for v in r), default=1.0) or 1.0
    segs, vals = [], []
    for b in range(nbands):
        segs += _segments_xy(kp, energies, b, shift)
        vals += _segment_weights(kp, w, b)
    lc = LineCollection(segs, cmap=spec.diverging, linewidths=1.6, zorder=3, capstyle='round')
    lc.set_array(np.asarray(vals)); lc.set_clim(-peak, peak)
    a.add_collection(lc)
    cb = fig.colorbar(lc, ax=a, fraction=0.04, pad=0.02, aspect=14)
    cb.set_label(f'{name} ' + tr('weight'))
    cb.outline.set_linewidth(0.4)


# --- PDOS -----------------------------------------------------------------------
def _default_dos_group(dos: DosData) -> str:
    elements = {c.element for c in dos.pdos_raw if c.element}
    return 'element' if len(elements) > 1 else 'l'


def _site_and_tag(label, by):
    """('Ga', 'dxy') from 'Ga dxy' for element+/atom+ groupings; (None, label) otherwise."""
    if '+' in by and ' ' in label:
        site, tag = label.rsplit(' ', 1)
        return site, tag
    return None, label


def _cumulative(values, step):
    out, acc = [], 0.0
    for i, v in enumerate(values):
        if i:
            acc += 0.5 * (v + values[i - 1]) * step
        out.append(acc)
    return out


def _dos_series(dos, curve, shift, integrated):
    e = np.asarray(dos.energies_ev, dtype=float) - shift
    v = np.asarray(_cumulative(curve.values, dos.energies_ev[1] - dos.energies_ev[0]) if integrated
                   else curve.values, dtype=float)
    return e, (-v if curve.spin == 'down' else v)


def plot_pdos_resolved(data: DosData, ax=None, *, emin=None, emax=None, fermi=None, align='fermi', group=None,
                       atoms=None, orbitals=None, integrated=False, stacked=False, show_total=True, **_opts):
    """PDOS for any grouping ('lm', 'element+lm', 'atom+l', ...). Element+ and atom+ groupings get one
    panel per site; others one panel. ``integrated`` plots the running integral (states);
    ``stacked`` fills cumulative areas. Spin down is mirrored below zero.
    """
    from .electronic_plots import _abs_ticks, _dos_window, energy_reference
    group = group or 'element+l'
    curves = data.grouped(group, atoms=atoms, orbitals=orbitals)
    panels: dict[str, list] = {}
    for c in curves:
        site, _ = _site_and_tag(c.label, group)
        panels.setdefault(site or '', []).append(c)
    n = len(panels)
    fig, axes = axes_or_new(ax, nrows=n, ncols=1, sharex=True, height_mm=14.0 + 26.0 * n)
    spec = style_of(fig)
    elements = [c.element for c in data.pdos_raw]
    shift, label, info = energy_reference(None, data, align=align if align != 'vbm' else 'fermi', fermi=fermi)
    lo, hi = _dos_window(data, shift, emin, emax)
    legend_seen = {}
    for row, (site, members) in enumerate(panels.items()):
        a = axes[row, 0]
        peak = 0.0
        if show_total and not integrated and not atoms and not orbitals and n == 1:
            for t in data.total:
                e, v = _dos_series(data, t, shift, False)
                a.plot(e, v, color=NEUTRAL['band'], lw=0.6, zorder=2.5,
                       label=tr('total_dos') if t.spin != 'down' else None)
                m = (e >= lo) & (e <= hi)
                peak = max(peak, float(np.abs(v[m]).max()) if m.any() else 0.0)
        base = {}
        for c in members:
            _, tag = _site_and_tag(c.label, group)
            # one site per panel: colours carry the orbital
            colour, dash = orbital_style(tag, spec) if '+' in group else group_style(c.label, group, spec, elements)
            e, v = _dos_series(data, c, shift, integrated)
            if stacked:
                prev = base.get(c.spin, np.zeros_like(v))
                top = prev + v
                a.fill_between(e, prev, top, color=tint(colour, 0.35), lw=0, zorder=2.6)
                base[c.spin] = top
                v = top
            lab = tag if (tag not in legend_seen and c.spin != 'down') else None
            if lab:
                legend_seen[tag] = True
            a.plot(e, v, color=colour, lw=0.8, ls=dash, label=lab, zorder=3)
            m = (e >= lo) & (e <= hi)
            peak = max(peak, float(np.abs(v[m]).max()) if m.any() else 0.0)
        peak = peak or 1.0
        head = 1.3 if site else 1.1          # room for the site label above the tallest curve
        a.set_ylim((-head * peak, head * peak) if data.spin_polarized else (0.0, head * peak))
        if data.spin_polarized:
            a.axhline(0.0, color=NEUTRAL['ink'], lw=0.4, zorder=4)
            _abs_ticks(a.yaxis)
        if align != 'none':
            a.axvline(0.0, **ZERO_KW)
        if site:
            a.text(0.01, 0.97, site, transform=a.transAxes, ha='left', va='top', fontsize=spec.base_fontsize,
                   color=NEUTRAL['ink'])
        minor(a)
        if row < n - 1:
            a.tick_params(labelbottom=False)
    axes[-1, 0].set_xlim(lo, hi)
    axes[-1, 0].set_xlabel(tr(label))
    ylab = tr('integrated_states') if integrated else tr('pdos')
    if n == 1:
        axes[0, 0].set_ylabel(ylab)
    else:
        fig.supylabel(ylab, fontsize=spec.base_fontsize)
    handles, labels = [], []
    for a in axes.ravel():
        for hh, ll in zip(*a.get_legend_handles_labels()):
            if ll not in labels:
                handles.append(hh); labels.append(ll)
    axes[0, 0].legend(handles, labels, loc='lower right', bbox_to_anchor=(1.0, 1.0),
                      ncol=min(5 if len(labels) > 6 else 4, len(labels)), handlelength=1.6, borderaxespad=0.1,
                      columnspacing=0.8)
    if data.spin_polarized:
        note(axes[0, 0], f'{tr("spin_up")} > 0, {tr("spin_down")} < 0', loc='above left')
    integrals = {c.label + (f' ({c.spin})' if c.spin else ''): data.integral(c, info.get('fermi_energy_ev'))
                 for c in curves} if info.get('fermi_energy_ev') is not None else {}
    attach_meta(fig, 'pdos_resolved', [data.source],
                parameters={'group': group, 'atoms': atoms, 'orbitals': orbitals, 'integrated': integrated,
                            'stacked': stacked, 'emin': emin, 'emax': emax, 'align': align},
                notes=list(data.warnings) + ([data.metadata['projection_weight_definition']]
                                             if data.metadata.get('projection_weight_definition') else []),
                values={**info, 'resolution': data.resolution, 'panels': {k or 'all': [c.label for c in v]
                                                                          for k, v in panels.items()},
                        'integrals_to_fermi': integrals})
    return fig


def plot_bands_pdos(data, ax=None, *, emin=None, emax=None, fermi=None, align='fermi', group=None, atoms=None,
                    orbitals=None, stacked=True, gap_arrow=True, fermi_bands=False, bz_inset=False, **opts):
    """Band structure (left) and PDOS side panel (right) on one energy axis; ``data`` = (bands, dos).

    The PDOS groups (default: per element for compounds, per l otherwise) are filled and stacked;
    the total DOS is the outline. Spin-polarised data mirror spin down to the left of zero.
    """
    from .electronic_plots import _kaxis
    bands, dos = data
    if dos is None or not dos.pdos_raw:
        raise InputError('bands_pdos needs a projected DOS (sparse-dft dos, DOSCAR with LORBIT, vasprun.xml <partial>)')
    group = group or _default_dos_group(dos)
    if ax is None:
        from .style import current_style, new_figure
        fig, axes = new_figure(current_style().width, height_mm=66.0, nrows=1, ncols=2, sharey=True,
                               gridspec_kw={'width_ratios': [2.6, 1.0], 'wspace': 0.03})
    else:
        fig, axes = axes_or_new(ax, nrows=1, ncols=2)
    spec = style_of(fig)
    ab, ad = axes[0, 0], axes[0, 1]
    shift, label, info = _energy_zero(bands, dos, align=align, fermi=fermi)
    lo, hi = _window(bands, shift, emin, emax, align)
    kp = kpath(bands)
    for ci, (name, rows) in enumerate(channels_of(bands).items()):
        segs = [seg for b in range(len(rows[0])) for seg in _segments_xy(kp, rows, b, shift)]
        colour = spec.color(ci) if len(channels_of(bands)) > 1 else NEUTRAL['ink']
        ab.add_collection(LineCollection(segs, colors=[colour], linewidths=0.8, zorder=3, capstyle='round',
                                         linestyles='-' if name != 'down' else (0, (3.2, 1.6))))
    ab.set_ylim(lo, hi)
    _kaxis(ab, kp)
    ab.set_ylabel(tr(label))
    minor(ab, x=False)
    edges, _ = decorate_bands(ab, bands, kp, shift, spec, lo, hi, gap_arrow=gap_arrow, fermi_bands=fermi_bands,
                              bz_inset=bz_inset)
    curves = dos.grouped(group, atoms=atoms, orbitals=orbitals)
    elements = [c.element for c in dos.pdos_raw]
    e = np.asarray(dos.energies_ev, dtype=float) - shift
    m = (e >= lo) & (e <= hi)
    base, peak, handles, seen = {}, 0.0, [], set()
    for c in curves:
        colour, _ = group_style(c.label, group, spec, elements)
        v = np.asarray(c.values, dtype=float) * (-1.0 if c.spin == 'down' else 1.0)
        prev = base.get(c.spin, np.zeros_like(v))
        top = prev + v if stacked else v
        if stacked:
            ad.fill_betweenx(e, prev, top, color=tint(colour, 0.3), lw=0, zorder=2.6)
            base[c.spin] = top
        ad.plot(top, e, color=colour, lw=0.6, zorder=3)
        peak = max(peak, float(np.abs(top[m]).max()) if m.any() else 0.0)
        if c.label not in seen:
            seen.add(c.label)
            handles.append(Line2D([], [], color=colour, lw=2.0, label=c.label))
    for t in dos.total:
        v = np.asarray(t.values, dtype=float) * (-1.0 if t.spin == 'down' else 1.0)
        ad.plot(v, e, color=NEUTRAL['ink'], lw=0.5, zorder=3.5)
        peak = max(peak, float(np.abs(v[m]).max()) if m.any() else 0.0)
    handles.append(Line2D([], [], color=NEUTRAL['ink'], lw=0.8, label=tr('total_dos')))
    peak = peak or 1.0
    ad.set_xlim((-1.08 * peak, 1.08 * peak) if dos.spin_polarized else (0.0, 1.08 * peak))
    if dos.spin_polarized:
        ad.axvline(0.0, color=NEUTRAL['ink'], lw=0.4, zorder=4)
    for a in (ab, ad):
        if align != 'none':
            a.axhline(0.0, **ZERO_KW)
    ad.set_ylim(lo, hi)
    ad.tick_params(axis='y', labelleft=False)
    ad.set_xticks([])
    ad.set_xlabel(tr('dos_short'))
    ad.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=spec.small_fontsize,
              borderaxespad=0.0, handlelength=1.2)
    values = {**info, 'group': group, 'pdos_groups': [c.label for c in curves], 'dos_origin': dos.origin,
              'dos_sigma_ev': dos.sigma_ev, 'energy_window_ev': [lo, hi], 'vbm_ev': edges.vbm_ev,
              'cbm_ev': edges.cbm_ev, 'gap_ev': edges.gap_ev, 'gap_direct': edges.direct}
    attach_meta(fig, 'bands_pdos', [bands.source, dos.source],
                parameters={'group': group, 'atoms': atoms, 'orbitals': orbitals, 'stacked': stacked,
                            'align': align, 'emin': emin, 'emax': emax},
                notes=list(dos.warnings) + ['PDOS and bands share the DOS (uniform mesh) energy zero'],
                values=values)
    return fig


def plot_dos_compare(data, ax=None, *, emin=None, emax=None, fermi=None, align='fermi', labels=None, **_opts):
    """Reference vs candidate total DOS (top) and their difference (bottom) on one grid.

    ``data`` = (reference, candidate) DosData on the same energy grid (see
    :func:`sparse_dft.analysis.electronic.dos_like`). Both use the reference Fermi energy.
    """
    from ..analysis.electronic import dos_difference
    from .electronic_plots import _abs_ticks, _dos_window, energy_reference
    ref, cand = data
    metrics = dos_difference(cand, ref)
    if ax is None:
        from .style import current_style, new_figure
        fig, axes = new_figure(current_style().width, height_mm=66.0, nrows=2, ncols=1, sharex=True,
                               gridspec_kw={'height_ratios': [2.4, 1.0], 'hspace': 0.06})
    else:
        fig, axes = axes_or_new(ax, nrows=2, ncols=1)
    spec = style_of(fig)
    top, bottom = axes[0, 0], axes[1, 0]
    shift, label, info = energy_reference(None, ref, align=align if align != 'vbm' else 'fermi', fermi=fermi)
    lo, hi = _dos_window(ref, shift, emin, emax)
    names = labels or (tr('reference'), tr('candidate'))
    e = np.asarray(ref.energies_ev, dtype=float) - shift
    m = (e >= lo) & (e <= hi)
    peak, dpeak = 0.0, 0.0
    for c in cand.total:
        r = next((x for x in ref.total if x.spin == c.spin), None)
        if r is None:
            continue
        sign = -1.0 if c.spin == 'down' else 1.0
        rv, cv = sign * np.asarray(r.values, dtype=float), sign * np.asarray(c.values, dtype=float)
        top.fill_between(e, 0.0, rv, color=tint(NEUTRAL['ink'], 0.85), lw=0, zorder=2.4)
        top.plot(e, rv, color=NEUTRAL['ink'], lw=0.9, zorder=3, label=names[0] if c.spin != 'down' else None)
        top.plot(e, cv, color=spec.color(1), lw=0.7, ls=(0, (3.0, 1.6)), zorder=4,
                 label=names[1] if c.spin != 'down' else None)
        bottom.plot(e, cv - rv, color=spec.color(0), lw=0.7, zorder=3)
        if m.any():
            peak = max(peak, float(np.abs(rv[m]).max()))
            dpeak = max(dpeak, float(np.abs((cv - rv)[m]).max()))
    peak = peak or 1.0
    top.set_ylim((-1.1 * peak, 1.1 * peak) if ref.spin_polarized else (0.0, 1.1 * peak))
    if ref.spin_polarized:
        top.axhline(0.0, color=NEUTRAL['ink'], lw=0.4, zorder=4)
        _abs_ticks(top.yaxis)
    lim = 1.2 * (dpeak or 1e-6)
    bottom.set_ylim(-lim, lim)
    bottom.axhline(0.0, color=NEUTRAL['ink'], lw=0.4, zorder=2)
    complete = metrics['window_ev'][1] - shift
    for a in (top, bottom):
        if align != 'none':
            a.axvline(0.0, **ZERO_KW)
        if complete < hi:
            a.axvspan(complete, hi, color=NEUTRAL['fail_fill'], lw=0, zorder=0.3)
        minor(a)
    top.set_ylabel(tr('dos'))
    bottom.set_ylabel(r'$\Delta$DOS')
    bottom.set_xlim(lo, hi)
    bottom.set_xlabel(tr(label))
    top.legend(loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=spec.small_fontsize, borderaxespad=0.1)
    l1 = metrics['total']
    text = 'L1 ' + ', '.join(f'{k} {v:.2e}' for k, v in l1.items() if v is not None)
    note(top, text, loc='above right', color=NEUTRAL['ink'])
    attach_meta(fig, 'dos_compare', [ref.source, cand.source],
                parameters={'emin': emin, 'emax': emax, 'align': align},
                notes=list(ref.warnings) + list(cand.warnings),
                values={**info, 'l1_normalised': l1, 'window_ev': metrics['window_ev'], 'sigma_ev': ref.sigma_ev,
                        'pdos_l1_normalised': {k: v['l1_normalised'] for k, v in metrics['pdos'].items()}})
    return fig


# --- one-command suite ----------------------------------------------------------
def suite_plan(sources, *, reference=None) -> list[tuple[str, str, dict]]:
    """(name, kind, options) of every figure the sources support; projections decide the rest."""
    from .figures import _Pool, _path_bands
    pool = _Pool(sources)
    paths = _path_bands(pool)
    band = paths[0] if paths else None
    pdos = pool.first('pdos')
    plan = []
    if band is not None:
        plan.append(('bands', 'bands', {'gap_arrow': True, 'fermi_bands': True}))
        if band.attrs.get('spin'):
            plan.append(('bands_spin', 'bands_spin', {}))
        if band.attrs.get('projections') or pool.first('procar'):
            lm = bool(band.attrs.get('lm') or pool.first('procar'))
            plan += [('fatbands_element', 'fatbands', {'group': 'element'}),
                     ('fatbands_orbital', 'fatbands', {'group': 'l'}),
                     ('fatbands_element_orbital', 'fatbands', {'group': 'element+l'})]
            if lm:
                plan.append(('fatbands_lm', 'fatbands', {'group': 'lm'}))
    if pdos is not None:
        plan += [('pdos_element_orbital', 'pdos', {'group': 'element+l'}),
                 ('pdos_lm', 'pdos', {'group': 'element+lm'})]
        if band is not None:
            plan.append(('bands_pdos', 'bands_pdos', {}))
    if reference is not None:
        if band is not None:
            # Unaligned: Sparse-DFT and VASP share the PAW energy zero, while a VASP NSCF
            # path recomputes E-fermi from the path eigenvalues (fcc Ni: 9.068 eV on
            # the path vs 9.262 eV in the SCF), which would shift every band.
            plan.append(('bands_compare', 'bands_compare', {'reference': reference, 'align': 'none'}))
        if pool.first('dos', lambda i: i.path.name == 'dos.json'):
            plan.append(('dos_compare', 'dos_compare', {'reference': reference}))
    return plan


def band_suite(sources, out, *, reference=None, dos_reference=None, formats=('pdf', 'png'), dpi=600,
               width='single', lang='en', palette='academic', **common) -> dict:
    """Render the band toolkit for ``sources`` into ``out``; one entry per figure with its QA report.

    ``reference`` is a VASP (or Sparse-DFT) band path for bands_compare; ``dos_reference`` a uniform-mesh
    vasprun.xml or DOS for dos_compare (default: ``reference``).
    """
    from pathlib import Path
    from .export import export
    from .figures import make_figure
    from .qa import layout_report
    out = Path(out)
    plan = suite_plan(sources, reference=reference or dos_reference)
    if not plan:
        raise InputError('band_suite: the sources contain no band path or projected DOS')
    written, skipped = [], []
    for name, kind, opts in plan:
        if kind == 'dos_compare':
            opts = {**opts, 'reference': dos_reference or reference}
        try:
            merged = {**common, **opts}
            if kind == 'bands_compare':
                merged['align'] = 'none'
                merged.pop('fermi', None)
            fig = make_figure(kind, sources, width=width, lang=lang, palette=palette, **merged)
            report = layout_report(fig)
            files = export(fig, out / name, formats=formats, dpi=dpi)
            meta = getattr(fig, 'sparse_dft_meta', {}) or {}
            written.append({'name': name, 'kind': kind, 'options': {k: str(v) for k, v in opts.items()},
                            'files': [str(f) for f in files], 'layout_issues': report['issues'],
                            'values': meta.get('values', {})})
        except InputError as exc:
            skipped.append({'name': name, 'kind': kind, **exc.as_dict()})
    return {'out': str(out), 'figures': written, 'skipped': skipped}
