"""Band structure, DOS/PDOS, fat-band and band-comparison figures."""
from __future__ import annotations

import math

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from ..analysis.common import resolve_fermi
from ..analysis.electronic import (BandStructure, DosData, ProjectionSet, L_ORDER, band_edges,
                                   channels_of, kpath)
from ..errors import InputError
from ._util import GUIDE_KW, ZERO_KW, attach_meta, axes_or_new, minor, note, provided_count
from .labels import tr
from .style import NEUTRAL, style_of, tint

SPIN_STYLE = {'up': '-', 'down': (0, (3.2, 1.6)), 'total': '-'}
ALIGNS = ('fermi', 'vbm', 'none')


def energy_reference(bands: BandStructure | None = None, dos: DosData | None = None, *,
                     align='fermi', fermi=None):
    """Return (shift_ev, ylabel_key, info); plotted energy = E - shift_ev."""
    if align not in ALIGNS:
        raise InputError(f"align must be one of {ALIGNS}, got {align!r}")
    source = bands.source if bands is not None else (dos.source if dos is not None else '<data>')
    if align == 'none':
        return 0.0, 'energy_abs', {'align': 'none'}
    if align == 'vbm':
        edges = band_edges(bands) if bands is not None else None
        if edges is None or edges.vbm_ev is None:
            raise InputError(f'no valence-band maximum for {source} (metal, or no occupations/E_F); '
                             "use align='fermi' with fermi=<eV> (CLI: --align fermi --fermi <eV>)")
        return edges.vbm_ev, 'energy_rel_vbm', {'align': 'vbm', 'vbm_ev': edges.vbm_ev}
    data_value = None
    origin_source = source
    if bands is not None and bands.fermi_energy is not None:
        data_value = bands.fermi_energy
    elif dos is not None and dos.fermi_energy_ev is not None:
        data_value = dos.fermi_energy_ev
        origin_source = dos.source
    value, origin = resolve_fermi(data_value, fermi, source)
    return value, 'energy_rel_ef', {'align': 'fermi', 'fermi_energy_ev': value, 'fermi_origin': origin,
                                    'fermi_source': origin_source if origin == 'data' else 'option'}


def _band_window(values, emin, emax, default=(-6.0, 6.0)):
    finite_values = [v for v in values if math.isfinite(v)]
    lo = default[0] if emin is None else float(emin)
    hi = default[1] if emax is None else float(emax)
    if emin is None and finite_values:
        lo = max(lo, min(finite_values) - 0.5)
    if emax is None and finite_values:
        hi = min(hi, max(finite_values) + 0.5)
    if hi <= lo:
        raise InputError(f'empty energy window [{lo}, {hi}] eV')
    return lo, hi


def _kaxis(ax, kp):
    ax.set_xlim(0.0, kp.length if kp.length > 0 else 1.0)
    if kp.ticks:
        ax.set_xticks([t for t, _ in kp.ticks], [label for _, label in kp.ticks])
    else:
        ax.set_xticks([])
    for t, _ in kp.ticks:
        if 1e-9 < t < kp.length - 1e-9:
            ax.axvline(t, **GUIDE_KW)
    for c in kp.corners:
        ax.axvline(c, color=NEUTRAL['guide'], lw=0.35, ls=(0, (1.5, 1.5)), zorder=0.5)
    ax.tick_params(axis='x', length=0, pad=2.5)
    if not kp.ticks or len(kp.ticks) < 2:
        ax.set_xlabel(tr('kpath'))


def _band_segments(kp, rows, band, shift):
    out = []
    for s, e in kp.segments:
        pts = [(kp.x[i], rows[i][band] - shift) for i in range(s, e + 1)]
        if len(pts) == 1:
            pts = [pts[0], pts[0]]
        out.append(pts)
    return out


def _draw_bands(ax, bands, kp, shift, spec, *, color=None, lw=0.8, zorder=3, alpha=1.0, label_suffix=''):
    channels = channels_of(bands)
    spin = len(channels) > 1
    handles = []
    for ci, (name, rows) in enumerate(channels.items()):
        c = color or (spec.color(ci) if spin else NEUTRAL['ink'])
        segs = [seg for b in range(len(rows[0])) for seg in _band_segments(kp, rows, b, shift)]
        ax.add_collection(LineCollection(segs, colors=[c], linewidths=lw, linestyles=SPIN_STYLE.get(name, '-'),
                                         zorder=zorder, alpha=alpha, capstyle='round', joinstyle='round'))
        if spin:
            key = {'up': 'spin_up', 'down': 'spin_down'}.get(name)
            handles.append(Line2D([], [], color=c, lw=lw, ls=SPIN_STYLE.get(name, '-'),
                                  label=(tr(key) if key else name) + label_suffix))
    return handles


def _mark_edges(ax, bands, kp, shift, spec, lo, hi):
    edges = band_edges(bands)
    if edges.vbm_ev is None:
        if edges.is_metal:
            note(ax, tr('metal'), loc='above right')
        return edges, None
    for k in edges.vbm_kindex:
        ax.plot(kp.x[k], edges.vbm_ev - shift, marker='o', ms=3.2, mfc='white', mec=spec.color(0), mew=0.8,
                zorder=6, clip_on=False, ls='none')
    for k in edges.cbm_kindex:
        ax.plot(kp.x[k], edges.cbm_ev - shift, marker='o', ms=3.2, mfc=spec.color(1), mec=spec.color(1), mew=0.8,
                zorder=6, clip_on=False, ls='none')
    kind = tr('gap_direct') if edges.direct else tr('gap_indirect')
    text = r'$E_\mathrm{g}$' + f' = {edges.gap_ev:.3f} eV ({kind})'
    if edges.status == 'inferred':
        text += f'; {tr("inferred")}'
    label = note(ax, text, loc='above right', color=NEUTRAL['ink'])
    return edges, label


def plot_bands(data: BandStructure, ax=None, *, emin=None, emax=None, fermi=None, align='fermi',
               show_edges=True, color=None, gap_arrow=False, fermi_bands=False, bz_inset=False, **_opts):
    """Band structure along the path: broken segments not joined, Γ labels, E − E_F axis.

    ``gap_arrow`` shades the gap and draws the VBM→CBM arrow, ``fermi_bands`` highlights
    bands crossing the energy zero, ``bz_inset`` adds the Brillouin-zone path inset.
    """
    fig, axes = axes_or_new(ax, height_mm=62.0)
    spec = style_of(fig)
    a = axes[0, 0]
    kp = kpath(data)
    shift, ylabel, info = energy_reference(data, align=align, fermi=fermi)
    handles = _draw_bands(a, data, kp, shift, spec, color=color)
    values = [e - shift for rows in channels_of(data).values() for row in rows for e in row]
    lo, hi = _band_window(values, emin, emax) if align != 'none' else _band_window(values, emin, emax, (-1e9, 1e9))
    a.set_ylim(lo, hi)
    _kaxis(a, kp)
    if align != 'none':
        a.axhline(0.0, **ZERO_KW)
    a.set_ylabel(tr(ylabel))
    minor(a, x=False)
    edges = None
    if gap_arrow or fermi_bands or bz_inset:
        from .band_toolkit import decorate_bands
        edges, _ = decorate_bands(a, data, kp, shift, spec, lo, hi, gap_arrow=gap_arrow and show_edges,
                                  fermi_bands=fermi_bands, bz_inset=bz_inset)
    elif show_edges:
        edges, _ = _mark_edges(a, data, kp, shift, spec, lo, hi)
    if handles:
        # above the panel: spin bands fill every inside corner (fcc Ni)
        a.legend(handles=handles, loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol=len(handles),
                 fontsize=spec.small_fontsize, borderaxespad=0.1)
    values_meta = {'nkpoints': data.nkpoints, 'nbands': data.nbands, 'segments': kp.segments,
                   'breaks': kp.breaks, 'ticks': kp.ticks, 'energy_window_ev': [lo, hi], **info}
    if edges is not None:
        values_meta.update({'vbm_ev': edges.vbm_ev, 'cbm_ev': edges.cbm_ev, 'gap_ev': edges.gap_ev,
                            'gap_direct': edges.direct, 'gap_status': edges.status})
    attach_meta(fig, 'bands', [data.source], parameters={'align': align, 'emin': emin, 'emax': emax,
                                                         'fermi': fermi}, values=values_meta)
    return fig


# --- DOS -------------------------------------------------------------------
L_COLOR_INDEX = {'s': 0, 'p': 1, 'd': 2, 'f': 3}


def _dos_arrays(dos: DosData, curve, shift):
    e = np.asarray(dos.energies_ev, dtype=float) - shift
    v = np.asarray(curve.values, dtype=float)
    sign = -1.0 if curve.spin == 'down' else 1.0
    return e, sign * v


def _dos_limit(dos, curves, shift, lo, hi):
    peak = 0.0
    for c in curves:
        e, v = _dos_arrays(dos, c, shift)
        mask = (e >= lo) & (e <= hi)
        if mask.any():
            peak = max(peak, float(np.nanmax(np.abs(v[mask]))))
    return peak if peak > 0 else 1.0


def _draw_curve(ax, e, v, *, vertical, color, lw=0.8, fill=None, ls='-', label=None, zorder=3):
    if vertical:
        if fill is not None:
            ax.fill_betweenx(e, 0.0, v, color=fill, lw=0, zorder=zorder - 1)
        return ax.plot(v, e, color=color, lw=lw, ls=ls, label=label, zorder=zorder)[0]
    if fill is not None:
        ax.fill_between(e, 0.0, v, color=fill, lw=0, zorder=zorder - 1)
    return ax.plot(e, v, color=color, lw=lw, ls=ls, label=label, zorder=zorder)[0]


def _draw_total_dos(ax, dos, shift, spec, *, vertical, lo, hi):
    spin = dos.spin_polarized
    for c in dos.total:
        e, v = _dos_arrays(dos, c, shift)
        color = spec.color(0 if c.spin != 'down' else 1) if spin else NEUTRAL['ink']
        key = {'up': 'spin_up', 'down': 'spin_down'}.get(c.spin)
        _draw_curve(ax, e, v, vertical=vertical, color=color, lw=0.75, fill=tint(color, 0.82),
                    label=tr(key) if key else tr('total_dos'))
    peak = _dos_limit(dos, dos.total, shift, lo, hi)
    limits = (-1.08 * peak, 1.08 * peak) if spin else (0.0, 1.08 * peak)
    if vertical:
        ax.set_xlim(*limits)
        if spin:
            ax.axvline(0.0, color=NEUTRAL['ink'], lw=0.4, zorder=4)
    else:
        ax.set_ylim(*limits)
        if spin:
            ax.axhline(0.0, color=NEUTRAL['ink'], lw=0.4, zorder=4)
    return peak


def _abs_ticks(axis):
    from matplotlib.ticker import FuncFormatter
    axis.set_major_formatter(FuncFormatter(lambda v, _: f'{abs(v):g}'))


def _dos_window(dos, shift, emin, emax, default=(-8.0, 6.0)):
    e = [x - shift for x in dos.energies_ev]
    lo = default[0] if emin is None else float(emin)
    hi = default[1] if emax is None else float(emax)
    if emin is None:
        lo = max(lo, min(e))
    if emax is None:
        hi = min(hi, max(e))
    if hi <= lo:
        raise InputError(f'empty DOS energy window [{lo}, {hi}] eV')
    return lo, hi


def _dos_notes(dos):
    notes = []
    if dos.origin == 'scf_gaussian' and dos.sigma_ev:
        notes.append(tr('gaussian_note', sigma=dos.sigma_ev))
    return notes


def plot_dos(data: DosData, ax=None, *, emin=None, emax=None, fermi=None, align='fermi', **_opts):
    """Total DOS; spin-polarised channels mirrored (down negative)."""
    fig, axes = axes_or_new(ax, height_mm=50.0)
    spec = style_of(fig)
    a = axes[0, 0]
    shift, label, info = energy_reference(None, data, align=align if align != 'vbm' else 'fermi', fermi=fermi)
    lo, hi = _dos_window(data, shift, emin, emax)
    _draw_total_dos(a, data, shift, spec, vertical=False, lo=lo, hi=hi)
    a.set_xlim(lo, hi)
    if align != 'none':
        a.axvline(0.0, **ZERO_KW)
    a.set_xlabel(tr(label))
    a.set_ylabel(tr('dos'))
    if data.spin_polarized:
        _abs_ticks(a.yaxis)
        a.legend(loc='upper left')
    minor(a)
    notes = _dos_notes(data)
    if notes:
        note(a, '\n'.join(notes), loc='above right')
    count = data.electron_count(info.get('fermi_energy_ev')) if info.get('fermi_energy_ev') is not None else None
    attach_meta(fig, 'dos', [data.source], parameters={'emin': emin, 'emax': emax, 'fermi': fermi, 'align': align},
                notes=data.warnings, values={**info, 'origin': data.origin, 'spin_polarized': data.spin_polarized,
                                             'integrated_electrons_to_fermi': count, 'sigma_ev': data.sigma_ev})
    return fig


def _pdos_groups(data: DosData):
    groups = {}
    for c in data.pdos:
        key = c.element or (f'atom {c.atoms[0]}' if c.atoms else c.label)
        groups.setdefault(key, []).append(c)
    return groups


def plot_pdos(data: DosData, ax=None, *, emin=None, emax=None, fermi=None, align='fermi', **_opts):
    """PDOS: one panel per element, one curve per l (s, p, d, f); spin down mirrored."""
    if not data.pdos:
        raise InputError(f'{data.source} has no projected DOS blocks')
    groups = _pdos_groups(data)
    n = len(groups)
    fig, axes = axes_or_new(ax, nrows=n, ncols=1, sharex=True, height_mm=14.0 + 24.0 * n)
    spec = style_of(fig)
    shift, label, info = energy_reference(None, data, align=align if align != 'vbm' else 'fermi', fermi=fermi)
    lo, hi = _dos_window(data, shift, emin, emax)
    seen_l = []
    raw_index = 0
    for row, (name, curves) in enumerate(groups.items()):
        a = axes[row, 0]
        peak = _dos_limit(data, curves, shift, lo, hi)
        for c in curves:
            e, v = _dos_arrays(data, c, shift)
            if c.l in L_COLOR_INDEX:
                color = spec.color(L_COLOR_INDEX[c.l])
                tag = c.l
            else:
                color = spec.color(4 + raw_index % 2); raw_index += 1
                tag = c.label.split(' ')[-1]
            lab = tag if (tag not in seen_l and c.spin != 'down') else None
            if lab:
                seen_l.append(tag)
            _draw_curve(a, e, v, vertical=False, color=color, lw=0.8, label=lab)
        a.set_ylim((-1.1 * peak, 1.1 * peak) if data.spin_polarized else (0.0, 1.1 * peak))
        if data.spin_polarized:
            a.axhline(0.0, color=NEUTRAL['ink'], lw=0.4, zorder=4)
            _abs_ticks(a.yaxis)
        if align != 'none':
            a.axvline(0.0, **ZERO_KW)
        a.text(0.01, 0.97, name, transform=a.transAxes, ha='left', va='top', fontsize=spec.base_fontsize,
               color=NEUTRAL['ink'])
        minor(a)
        if row < n - 1:
            a.tick_params(labelbottom=False)
    axes[-1, 0].set_xlim(lo, hi)
    axes[-1, 0].set_xlabel(tr(label))
    # One y label for the stack, placed on the middle panel(s) with constrained
    # layout so it never collides with tick labels (fig.supylabel does).
    if n == 1:
        axes[0, 0].set_ylabel(tr('pdos'))
    else:
        fig.supylabel(tr('pdos'), fontsize=spec.base_fontsize)
    handles, labels = [], []
    for a in axes.ravel():
        h, l = a.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in labels:
                handles.append(hh); labels.append(ll)
    # Legend above the stack: PDOS peaks can fill any inside corner.
    axes[0, 0].legend(handles, labels, loc='lower right', bbox_to_anchor=(1.0, 1.0), ncol=min(4, len(labels)),
                      handlelength=1.4, borderaxespad=0.1)
    if data.spin_polarized:
        note(axes[0, 0], f'{tr("spin_up")} > 0, {tr("spin_down")} < 0', loc='above left')
    attach_meta(fig, 'pdos', [data.source], parameters={'emin': emin, 'emax': emax, 'fermi': fermi, 'align': align},
                notes=data.warnings, values={**info, 'groups': {k: [c.label + (f' ({c.spin})' if c.spin else '')
                                                                    for c in v] for k, v in groups.items()}})
    return fig


def plot_bands_dos(data, ax=None, *, emin=None, emax=None, fermi=None, align='fermi', **opts):
    """Bands (left) and DOS (right) on one shared energy axis; ``data`` = (bands, dos)."""
    bands, dos = data
    if dos is None:
        raise InputError('bands_dos needs a DOS source (DOSCAR, vasprun.xml <dos> or a converged SCF result)')
    if ax is None:
        from .style import current_style, new_figure
        fig, axes = new_figure(current_style().width, height_mm=62.0, nrows=1, ncols=2, sharey=True,
                               gridspec_kw={'width_ratios': [3.0, 1.0], 'wspace': 0.02})
    else:
        fig, axes = axes_or_new(ax, nrows=1, ncols=2)
    spec = style_of(fig)
    ab, ad = axes[0, 0], axes[0, 1]
    # One energy zero for both panels: explicit fermi, else the DOS (mesh)
    # Fermi energy, else the band file's.  A path-only NSCF rarely defines
    # E_F; the uniform-mesh value is the physical one (as in pymatgen).
    if align == 'fermi' and fermi is None and dos.fermi_energy_ev is not None:
        shift, label, info = energy_reference(None, dos, align='fermi')
    else:
        shift, label, info = energy_reference(bands, dos, align=align, fermi=fermi)
    if align == 'fermi' and fermi is None and bands.fermi_energy is not None and info.get('fermi_energy_ev') is not None \
            and abs(bands.fermi_energy - info['fermi_energy_ev']) > 1e-3:
        info['band_file_fermi_energy_ev'] = bands.fermi_energy
    plot_bands(bands, ab, emin=emin, emax=emax, fermi=shift if align == 'fermi' else fermi, align=align, **opts)
    lo, hi = ab.get_ylim()
    _draw_total_dos(ad, dos, shift, spec, vertical=True, lo=lo, hi=hi)
    ad.set_ylim(lo, hi)
    if align != 'none':
        ad.axhline(0.0, **ZERO_KW)
    ad.tick_params(axis='y', labelleft=False)
    ad.set_xlabel(tr('dos_short'))
    ad.set_xticks([])          # DOS magnitude is relative here; the dos kind gives the scale
    meta = dict(getattr(fig, 'sparse_dft_meta', {}) or {})
    fig.sparse_dft_meta = {}
    attach_meta(fig, 'bands_dos', [bands.source, dos.source],
                parameters={'emin': emin, 'emax': emax, 'fermi': fermi, 'align': align},
                notes=list(dos.warnings) + ['DOS uses the band-structure energy reference'],
                values={**(meta.get('values') or {}), **info, 'dos_origin': dos.origin, 'dos_sigma_ev': dos.sigma_ev})
    if 'band_file_fermi_energy_ev' in info:
        note(ab, r'zero: DOS $E_\mathrm{F}$ (band file $E_\mathrm{F}$ differs)', loc='above left')
    return fig


def plot_fatbands(data, ax=None, *, emin=None, emax=None, fermi=None, align='fermi', group='l',
                  channel=None, max_width=3.2, **_opts):
    """Projection weights as line widths: one LineCollection per (channel, group).

    ``data`` = (bands, projections).  Groups are l (s, p, d, f) by default,
    or 'element' / 'orbital'.  SOC magnetisation channels use a signed
    diverging colour scale instead of widths.
    """
    bands, proj = data
    channels = [channel] if channel else list(proj.channels)
    for c in channels:
        if c not in proj.channels:
            raise InputError(f'projection channel {c!r} not in {list(proj.channels)}')
    n = len(channels)
    fig, axes = axes_or_new(ax, nrows=n, ncols=1, sharex=True, height_mm=8.0 + 54.0 * n)
    spec = style_of(fig)
    kp = kpath(bands)
    shift, label, info = energy_reference(bands, align=align, fermi=fermi)
    groups_meta = {}
    for row, name in enumerate(channels):
        a = axes[row, 0]
        energies = (bands.spin_channels or {}).get(name, bands.eigenvalues)
        base = [seg for b in range(bands.nbands) for seg in _band_segments(kp, energies, b, shift)]
        a.add_collection(LineCollection(base, colors=[NEUTRAL['guide']], linewidths=0.4, zorder=2))
        if name in ('mx', 'my', 'mz'):
            tensor = proj.channels[name]
            w = [[sum(sum(atom) for atom in tensor[k][b]) for b in range(bands.nbands)] for k in range(bands.nkpoints)]
            peak = max((abs(v) for r in w for v in r), default=1.0) or 1.0
            segs, vals = [], []
            for b in range(bands.nbands):
                for s, e in kp.segments:
                    for i in range(s, e):
                        segs.append([(kp.x[i], energies[i][b] - shift), (kp.x[i + 1], energies[i + 1][b] - shift)])
                        vals.append(0.5 * (w[i][b] + w[i + 1][b]))
            lc = LineCollection(segs, cmap=spec.diverging, linewidths=1.6, zorder=3, capstyle='round')
            lc.set_array(np.asarray(vals)); lc.set_clim(-peak, peak)
            a.add_collection(lc)
            cb = fig.colorbar(lc, ax=a, fraction=0.04, pad=0.02, aspect=14)
            cb.set_label(f'{name} ' + tr('weight'))
            cb.outline.set_linewidth(0.4)
            groups_meta[name] = ['signed']
        else:
            grouped = proj.grouped(name, by=group)
            peak = max((v for g in grouped.values() for r in g for v in r), default=0.0) or 1.0
            handles = []
            for gi, (gname, w) in enumerate(grouped.items()):
                color = spec.color(L_COLOR_INDEX.get(gname, gi) if group == 'l' else gi)
                segs, widths = [], []
                for b in range(bands.nbands):
                    for s, e in kp.segments:
                        for i in range(s, e):
                            segs.append([(kp.x[i], energies[i][b] - shift), (kp.x[i + 1], energies[i + 1][b] - shift)])
                            widths.append(max_width * 0.5 * (w[i][b] + w[i + 1][b]) / peak)
                a.add_collection(LineCollection(segs, colors=[color], linewidths=widths, alpha=0.75, zorder=3 + gi * 0.01,
                                                capstyle='round', label=str(gname)))
                handles.append(Line2D([], [], color=color, lw=2.0, label=str(gname)))
            a.legend(handles=handles, loc='upper right', ncol=min(4, len(handles)), fontsize=spec.small_fontsize)
            groups_meta[name] = list(grouped)
        values = [e - shift for r in energies for e in r]
        lo, hi = _band_window(values, emin, emax)
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
    msg = tr('width_note')
    if proj.weight_definition and 'raw' in proj.weight_definition:
        msg += '; raw PAW projector intensities'
    note(axes[0, 0], msg, loc='above left')
    attach_meta(fig, 'fatbands', [bands.source, proj.source], parameters={'group': group, 'channel': channel, 'align': align,
                                                                         'emin': emin, 'emax': emax},
                notes=[proj.weight_definition] if proj.weight_definition else [],
                values={**info, 'groups': groups_meta, 'peak_weight_scale': 'per panel'})
    return fig


def plot_bands_compare(data, ax=None, *, emin=None, emax=None, align='fermi', max_energy_ev=10.0,
                       labels=None, **_opts):
    """Reference vs candidate bands (overlay) and ΔE vs energy on a shared energy axis.

    ``data`` = (reference, candidate) BandStructures; differences use the
    alignment and interpolation of :func:`sparse_dft.bands.compare_bands`.
    """
    from ..analysis.electronic import compare_band_data, is_mesh
    ref, cand = data
    cmp = compare_band_data(ref, cand, align=align, max_energy_ev=max_energy_ev)
    if ax is None:
        from .style import current_style, new_figure
        fig, axes = new_figure(current_style().width, height_mm=62.0, nrows=1, ncols=2, sharey=True,
                               gridspec_kw={'width_ratios': [2.2, 1.0], 'wspace': 0.03})
    else:
        fig, axes = axes_or_new(ax, nrows=1, ncols=2)
    spec = style_of(fig)
    ab, ae = axes[0, 0], axes[0, 1]
    zero = ref.fermi_energy if align == 'fermi' else (ref.vbm if align == 'vbm' else 0.0)
    ylabel = {'fermi': 'energy_rel_ef', 'vbm': 'energy_rel_vbm', 'none': 'energy_abs'}[align]
    shift_c = cmp.metrics['energy_shift_ev']
    names = labels or (tr('reference'), tr('candidate'))
    ref_color, cand_color = NEUTRAL['ink'], spec.color(1)
    mesh = is_mesh(ref) or is_mesh(cand)
    if mesh:
        # A k-mesh is not a path: show eigenvalues per k-point as markers.
        for k, row in enumerate(ref.eigenvalues):
            ab.plot([k] * len(row), [e - zero for e in row], ls='none', marker='_', ms=7, mew=0.9, color=ref_color, zorder=3)
        for k, row in enumerate(cand.eigenvalues):
            ab.plot([k] * len(row), [e + shift_c - zero for e in row], ls='none', marker='o', ms=2.0, mew=0.6,
                    mfc='none', mec=cand_color, zorder=4)
        ab.set_xlim(-0.6, ref.nkpoints - 0.4)
        ab.set_xticks(range(ref.nkpoints)) if ref.nkpoints <= 16 else None
        ab.set_xlabel('SCF k-point')
        handles = [Line2D([], [], color=ref_color, ls='none', marker='_', ms=7, mew=0.9, label=names[0]),
                   Line2D([], [], color=cand_color, ls='none', marker='o', ms=2.5, mfc='none', label=names[1])]
    else:
        kp_r, kp_c = kpath(ref), kpath(cand)
        _draw_bands(ab, ref, kp_r, zero, spec, color=ref_color, lw=0.9, zorder=3)
        scale = kp_r.length / kp_c.length if kp_c.length > 0 else 1.0
        kp_c.x = [x * scale for x in kp_c.x]
        _draw_bands(ab, cand, kp_c, zero - shift_c, spec, color=cand_color, lw=0.7, zorder=4)
        for coll in ab.collections[-len(channels_of(cand)):]:
            coll.set_linestyle((0, (3.0, 1.6)))
        _kaxis(ab, kp_r)
        handles = [Line2D([], [], color=ref_color, lw=0.9, label=names[0]),
                   Line2D([], [], color=cand_color, lw=0.7, ls=(0, (3.0, 1.6)), label=names[1])]
    energies_rel = [e - zero for e in cmp.reference_ev]
    lo, hi = _band_window(energies_rel, emin, emax, default=(-max_energy_ev, max_energy_ev) if align != 'none' else (-1e9, 1e9))
    ab.set_ylim(lo, hi)
    if align != 'none':
        ab.axhline(0.0, **ZERO_KW)
    ab.set_ylabel(tr(ylabel))
    # Legend above the band panel: overlaid bands fill every inside corner.
    ab.legend(handles=handles, loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=spec.small_fontsize,
              borderaxespad=0.1)
    minor(ab, x=False)
    delta_mev = [1e3 * d for d in cmp.delta_ev]
    ae.axvline(0.0, color=NEUTRAL['ink'], lw=0.4, zorder=2)
    ae.plot(delta_mev, energies_rel, ls='none', marker='o', ms=2.0, mew=0, color=spec.color(0), alpha=0.85, zorder=3)
    ae.set_xlabel(tr('band_error_mev'))
    ae.tick_params(axis='y', labelleft=False)
    lim = max([abs(v) for v in delta_mev] + [1e-6]) * 1.15
    ae.set_xlim(-lim, lim)
    minor(ae)
    m = cmp.metrics
    summary = (f'{tr("mae")} {1e3 * m["mae_ev"]:.3g} meV\n{tr("max_error")} {1e3 * m["max_abs_error_ev"]:.3g} meV')
    ae.text(0.0, 1.01, summary, transform=ae.transAxes, ha='left', va='bottom', fontsize=spec.small_fontsize,
            color=NEUTRAL['ink'])
    attach_meta(fig, 'bands_compare', [ref.source, cand.source],
                parameters={'align': align, 'max_energy_ev': max_energy_ev, 'emin': emin, 'emax': emax},
                values={'mae_ev': m['mae_ev'], 'rmse_ev': m['rmse_ev'], 'max_abs_error_ev': m['max_abs_error_ev'],
                        'energy_shift_ev': m['energy_shift_ev'], 'samples': m['samples'],
                        'reference_gap_ev': m.get('reference_gap_ev'), 'candidate_gap_ev': m.get('candidate_gap_ev'),
                        'sampling': 'scf_mesh' if mesh else 'path'})
    return fig
