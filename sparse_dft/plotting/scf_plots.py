"""SCF figures: convergence, magnetisation, energy ledger, forces, stress, timers."""
from __future__ import annotations

import math

import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import LogLocator, NullFormatter

from ..analysis.scf import EnergyTerms, ForceSet, ScfHistory, StressTensor, TimingProfile
from ..errors import InputError
from ._util import (GUIDE_KW, ZERO_KW, attach_meta, axes_or_new, integer_xaxis, minor, note,
                    provided_count, sci, symmetric_limit, text_color_for)
from .labels import ENERGY_TERM_LABELS, tr
from .style import NEUTRAL, MARKERS, style_of, tint


def _log_axis(ax, min_decades=1.2):
    """Log y axis spanning at least ``min_decades`` so it always shows >= 2 labelled ticks."""
    ax.set_yscale('log')
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2.0, 10.0), numticks=30))
    ax.yaxis.set_minor_formatter(NullFormatter())
    lo, hi = ax.get_ylim()
    if lo > 0 and hi > 0 and math.log10(hi / lo) < min_decades:
        centre = 0.5 * (math.log10(hi) + math.log10(lo))
        ax.set_ylim(10 ** (centre - min_decades / 2), 10 ** (centre + min_decades / 2))


def _positive(values):
    return [v if v is not None and v > 0 else np.nan for v in values]


def plot_scf_convergence(data: ScfHistory, ax=None, *, show_eigensolver=True, **_opts):
    """Log residuals vs iteration with tolerance lines; eigensolver in its own panel."""
    has_eig = show_eigensolver and 'eigensolver_max_residual_hartree' in data.series
    rows = 2 if has_eig else 1
    if provided_count(ax) == 1 and has_eig:
        has_eig, rows = False, 1         # a single supplied Axes gets the density panel only
    fig, axes = axes_or_new(ax, nrows=rows, ncols=1, sharex=True,
                            height_mm=72.0 if rows == 2 else 50.0,
                            gridspec_kw={'height_ratios': [1.6, 1.0]} if rows == 2 else None)
    spec = style_of(fig)
    top = axes[0, 0]
    x = data.iteration
    series = [('density_residual', tr('density_residual'), 0, 'o'),
              ('onecenter_density_residual', tr('onecenter_residual'), 1, 's')]
    drawn = []
    for key, label, ci, marker in series:
        if key not in data.series:
            continue
        color = spec.color(ci)
        top.plot(x, _positive(data.series[key]), color=color, marker=marker, ms=2.4, mew=0,
                 lw=0.9, label=label, zorder=3)
        tol = data.tolerances.get(key)
        if tol:
            top.axhline(tol, color=color, lw=0.6, ls=(0, (4, 2)), zorder=2)
        drawn.append(key)
    if not drawn:
        raise InputError(f'{data.source} has no density residual series')
    _log_axis(top)
    top.set_ylabel(tr('residual'))
    tols = sorted(set(data.tolerances.values()))
    if tols:
        tol_text = ', '.join(f'${sci(t)}$' for t in tols)
        # Residuals start high and fall through the tolerance at the right, so
        # the label sits at the left end of the line, clear of the data.
        top.annotate(f'{tr("tolerance")} {tol_text}', xy=(0.0, tols[0]), xycoords=('axes fraction', 'data'),
                     xytext=(2, -1.5), textcoords='offset points', ha='left', va='top',
                     fontsize=spec.small_fontsize, color=NEUTRAL['muted'])
    top.legend(loc='upper right', handlelength=1.8)
    if data.converged is False:
        note(top, tr('not_converged'), loc='lower left', color='#B2182B')
    if has_eig:
        bottom = axes[1, 0]
        bottom.plot(x, _positive(data.series['eigensolver_max_residual_hartree']), color=spec.color(2),
                    marker='^', ms=2.4, mew=0, lw=0.9, label=tr('eigensolver'))
        if data.eigensolver_tolerance_hartree:
            bottom.axhline(data.eigensolver_tolerance_hartree, color=spec.color(2), lw=0.6, ls=(0, (4, 2)))
        _log_axis(bottom)
        bottom.set_ylabel(tr('eig_residual'))
        bottom.set_xlabel(tr('iteration'))
    else:
        top.set_xlabel(tr('iteration'))
    for a in axes.ravel():
        integer_xaxis(a)
    if rows == 2:
        axes[0, 0].tick_params(labelbottom=False)
    attach_meta(fig, 'scf_convergence', [data.source, data.result_source],
                values={'iterations': data.niterations, 'tolerances': data.tolerances,
                        'final': {k: v[-1] for k, v in data.series.items() if k in
                                  ('density_residual', 'onecenter_density_residual', 'eigensolver_max_residual_hartree')}})
    return fig


def plot_magnetization(data: ScfHistory, ax=None, **_opts):
    """Output and mixed moments per iteration, and |Δm| between iterations (log)."""
    if 'output_magnetic_moment_muB' not in data.series:
        raise InputError(f'{data.source} has no magnetic-moment history (spin-unpolarized run)')
    fig, axes = axes_or_new(ax, nrows=2, ncols=1, sharex=True, height_mm=70.0,
                            gridspec_kw={'height_ratios': [1.5, 1.0]})
    spec = style_of(fig)
    top = axes[0, 0]
    x = data.iteration
    out = data.series['output_magnetic_moment_muB']
    top.plot(x, out, color=spec.color(0), marker='o', ms=2.4, mew=0, lw=0.9, label=tr('moment_output'))
    if 'mixed_magnetic_moment_muB' in data.series:
        top.plot(x, data.series['mixed_magnetic_moment_muB'], color=spec.color(1), marker='s', ms=2.2,
                 mew=0, lw=0.8, ls=(0, (3, 1.5)), label=tr('moment_mixed'))
    top.set_ylabel(tr('moment'))
    top.legend(loc='best')
    minor(top, x=False)
    final = next((v for v in reversed(out) if v is not None), None)
    if final is not None:
        top.annotate(f'${final:.4f}\\,\\mu_\\mathrm{{B}}$', xy=(x[-1], final), xytext=(-2, 6),
                     textcoords='offset points', ha='right', va='bottom', fontsize=spec.small_fontsize,
                     color=NEUTRAL['ink'])
    bottom = axes[1, 0]
    delta = [abs(b - a) if a is not None and b is not None else None for a, b in zip(out, out[1:])]
    bottom.plot(x[1:], _positive(delta), color=spec.color(0), marker='o', ms=2.2, mew=0, lw=0.8)
    _log_axis(bottom)
    bottom.set_ylabel(tr('moment_change'))
    bottom.set_xlabel(tr('iteration'))
    for a in axes.ravel():
        integer_xaxis(a)
    top.tick_params(labelbottom=False)
    attach_meta(fig, 'magnetization', [data.source, data.result_source], values={'final_output_moment_muB': final})
    return fig


def plot_energy_terms(data: EnergyTerms, ax=None, *, mode='auto', per_atom=False, **_opts):
    """Energy ledger: verified waterfall, else a signed non-additive bar chart.

    ``mode``: 'auto' (waterfall only if the ledger sums to total_energy_ev
    within 1e-6 eV), 'waterfall' (raises if not additive) or 'bars'.
    """
    if mode not in ('auto', 'waterfall', 'bars'):
        raise InputError("energy_terms mode must be 'auto', 'waterfall' or 'bars'")
    if mode == 'waterfall' and not data.additive:
        raise InputError('energy terms do not sum to total_energy_ev; a waterfall would misstate the ledger',
                         details={'residual_ev': data.ledger_residual_ev, 'missing': data.missing_terms})
    waterfall = data.additive and mode != 'bars'
    scale = 1.0
    if per_atom:
        if not data.natoms:
            raise InputError('per_atom requires the atom count (forces array) in the SCF result')
        scale = 1.0 / data.natoms
    terms = list(data.terms)
    n = len(terms)
    fig, axes = axes_or_new(ax, height_mm=max(52.0, 7.0 + 4.6 * (n + 1)))
    spec = style_of(fig)
    a = axes[0, 0]
    pos_c, neg_c = spec.color(1), spec.color(0)
    y = np.arange(n + (1 if waterfall else 0))
    labels = [ENERGY_TERM_LABELS.get(t.name, t.name) for t in terms]
    contributions = [t.contribution_ev * scale for t in terms]
    total = data.total_energy_ev * scale
    span = max([abs(c) for c in contributions] + [abs(total)])
    if waterfall:
        running = 0.0
        for i, c in enumerate(contributions):
            left = running if c >= 0 else running + c
            a.barh(i, abs(c), left=left, height=0.62, color=tint(pos_c if c >= 0 else neg_c, 0.15),
                   edgecolor='white', linewidth=0.4, zorder=3)
            nxt = running + c
            if i < n - 1:
                a.plot([nxt, nxt], [i + 0.31, i + 0.69], color=NEUTRAL['muted'], lw=0.35, zorder=2)
            running = nxt
        a.plot([running, running], [n - 1 + 0.31, n - 0.31], color=NEUTRAL['muted'], lw=0.35, zorder=2)
        a.barh(n, total, left=0.0, height=0.62, color=NEUTRAL['ink'], zorder=3)
        labels.append(tr('total'))
        values = contributions + [total]
        ends = []
        running = 0.0
        for c in contributions:
            ends.append(max(running, running + c))
            running += c
        ends.append(max(0.0, total))
        starts = []
        running = 0.0
        for c in contributions:
            starts.append(min(running, running + c))
            running += c
        starts.append(min(0.0, total))
    else:
        for i, c in enumerate(contributions):
            a.barh(i, c, height=0.62, color=tint(pos_c if c >= 0 else neg_c, 0.15), zorder=3)
        values = contributions
        ends = [max(0.0, c) for c in contributions]
        starts = [min(0.0, c) for c in contributions]
        a.axvline(0.0, **ZERO_KW)
    a.set_yticks(y, labels)
    a.invert_yaxis()
    a.tick_params(axis='y', length=0, pad=3)
    a.spines['left'].set_visible(False)
    lo = min(starts + [0.0]); hi = max(ends + [0.0])
    pad = 0.30 * (hi - lo)
    a.set_xlim(lo - 0.04 * (hi - lo), hi + pad)
    for i, v in enumerate(values):
        text = f'{v:+.4g}'.replace('-', '−') if abs(v) >= 1e-3 or v == 0 else f'${sci(v, 2)}$'
        if i == len(values) - 1 and waterfall:
            text = f'{v:.6g}'.replace('-', '−')
        a.annotate(text, xy=(ends[i], i), xytext=(2, 0), textcoords='offset points', ha='left', va='center',
                   fontsize=spec.small_fontsize, color=NEUTRAL['ink'], zorder=4)
    a.set_xlabel(tr('contribution_ev_atom') if per_atom else tr('contribution_ev'))
    minor(a, y=False)
    a.grid(axis='x', color=NEUTRAL['guide'], lw=0.3, alpha=0.6, zorder=0)
    from matplotlib.patches import Patch
    handles = [Patch(color=tint(pos_c, 0.15), label=tr('raises_total')),
               Patch(color=tint(neg_c, 0.15), label=tr('lowers_total'))]
    a.legend(handles=handles, loc='lower right', bbox_to_anchor=(1.0, 1.0), ncol=2, fontsize=spec.small_fontsize,
             handlelength=1.0, borderaxespad=0.1)
    notes = []
    if not data.total_energy_valid:
        notes.append(tr('diagnostic_note'))
    if not waterfall:
        notes.append(tr('non_additive'))
    if notes:
        note(a, '\n'.join(notes), loc='above left', color='#B2182B' if not data.total_energy_valid else None)
    attach_meta(fig, 'energy_terms', [data.source],
                parameters={'mode': 'waterfall' if waterfall else 'bars', 'per_atom': per_atom},
                notes=['total_energy_valid=false: energies are diagnostic'] if not data.total_energy_valid else [],
                values={'total_energy_ev': data.total_energy_ev, 'ledger_residual_ev': data.ledger_residual_ev,
                        'additive': data.additive, 'missing_terms': data.missing_terms,
                        'terms_ev': {t.key: t.value_ev for t in terms}})
    return fig


def _annotate_cells(ax, matrix, norm, cmap, fmt, fontsize):
    for i, row in enumerate(matrix):
        for j, v in enumerate(row):
            ax.text(j, i, fmt(v), ha='center', va='center', fontsize=fontsize,
                    color=text_color_for(cmap(norm(v))))


def plot_forces(data: ForceSet, ax=None, *, tolerance=None, annotate=None, **_opts):
    """Per-atom |F| bars over a Cartesian-component heatmap (eV/Å)."""
    n = data.natoms
    fig, axes = axes_or_new(ax, nrows=2, ncols=1, sharex=True, height_mm=70.0,
                            gridspec_kw={'height_ratios': [1.25, 1.0]})
    spec = style_of(fig)
    bars, heat = axes[0, 0], axes[1, 0]
    mags = data.magnitudes_ev_per_angstrom
    x = np.arange(n)
    bars.bar(x, mags, width=0.7 if n > 1 else 0.4, color=spec.color(0), zorder=3)
    if tolerance is not None:
        bars.axhline(float(tolerance), color=spec.color(1), lw=0.6, ls=(0, (4, 2)), zorder=4)
        bars.annotate(f'{tr("tolerance")} {float(tolerance):g}', xy=(1.0, float(tolerance)),
                      xycoords=('axes fraction', 'data'), xytext=(0, 2), textcoords='offset points',
                      ha='right', va='bottom', fontsize=spec.small_fontsize, color=spec.color(1))
    bars.set_ylabel(tr('force_abs'))
    bars.set_ylim(0, max(mags + [float(tolerance or 0)]) * 1.18 or 1.0)
    minor(bars, x=False)
    bars.tick_params(labelbottom=False)
    matrix = np.array(data.forces_ev_per_angstrom, dtype=float).T          # 3 x natoms
    limit = symmetric_limit(matrix.ravel().tolist())
    cmap = spec.diverging
    from matplotlib.colors import Normalize
    norm = Normalize(-limit, limit)
    image = heat.imshow(matrix, cmap=cmap, norm=norm, aspect='auto', interpolation='nearest')
    heat.set_yticks([0, 1, 2], ['$F_x$', '$F_y$', '$F_z$'])
    heat.tick_params(axis='y', length=0)
    for side in ('left', 'bottom'):
        heat.spines[side].set_visible(False)
    labels = data.atom_labels()
    step = max(1, int(math.ceil(n / 24)))
    heat.set_xticks(x[::step], labels[::step], rotation=90 if n > 10 else 0)
    heat.tick_params(axis='x', length=0)
    heat.set_xlabel(tr('atom'))
    if annotate is None:
        annotate = n <= 10
    if annotate:
        _annotate_cells(heat, matrix, norm, cmap, lambda v: f'{v:.3g}'.replace('-', '−'),
                        max(5.0, spec.small_fontsize - 0.5))
    cb = fig.colorbar(image, ax=heat, location='right', fraction=0.05, pad=0.02, aspect=12)
    cb.set_label(tr('force_component'))
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(width=0.4, length=2)
    notes = []
    if not data.validated:
        notes.append(tr('unvalidated_note'))
    net = data.net_force_ev_per_angstrom
    notes.append(r'$\max|F|$ = ' + f'${sci(data.max_ev_per_angstrom)}$ eV ' + r'$\mathrm{\AA}^{-1}$')
    note(bars, '\n'.join(notes), loc='above right')
    attach_meta(fig, 'forces', [data.source], parameters={'tolerance': tolerance},
                values={'max_force_ev_per_angstrom': data.max_ev_per_angstrom, 'net_force_ev_per_angstrom': net,
                        'validated': data.validated, 'status': data.status})
    return fig


def plot_stress(data: StressTensor, ax=None, **_opts):
    """3×3 stress tensor in GPa on a diverging scale centred at zero."""
    fig, axes = axes_or_new(ax, height_mm=66.0)
    spec = style_of(fig)
    a = axes[0, 0]
    gpa = np.array(data.stress_gpa, dtype=float)
    limit = symmetric_limit(gpa.ravel().tolist())
    from matplotlib.colors import Normalize
    norm = Normalize(-limit, limit)
    cmap = spec.diverging
    image = a.imshow(gpa, cmap=cmap, norm=norm, interpolation='nearest')
    names = ['$x$', '$y$', '$z$']
    a.set_xticks(range(3), names)
    a.set_yticks(range(3), names)
    a.xaxis.tick_top()
    a.tick_params(axis='both', length=0, top=False)
    for side in a.spines.values():
        side.set_visible(False)
    def fmt(v):
        if abs(v) < 5e-4 * limit or abs(v) < 1e-9:
            return f'${sci(v, 2)}$' if v != 0 and abs(v) >= 1e-12 else '0'
        return f'{v:.4g}'.replace('-', '−')
    _annotate_cells(a, gpa, norm, cmap, fmt, spec.small_fontsize)
    cb = fig.colorbar(image, ax=a, fraction=0.046, pad=0.04, aspect=16)
    cb.set_label(tr('stress_gpa'))
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(width=0.4, length=2)
    lines = []
    if data.pressure_gpa is not None:
        lines.append(f'{tr("pressure")} = {data.pressure_gpa:.4g} GPa'.replace('-', '−'))
        lines.append(tr('tension_positive'))
    if not data.validated:
        lines.append(tr('unvalidated_note'))
    if lines:
        a.text(0.5, -0.06, '   '.join(lines), transform=a.transAxes, ha='center', va='top',
               fontsize=spec.small_fontsize, color=NEUTRAL['muted'])
    attach_meta(fig, 'stress', [data.source],
                notes=['1 eV/angstrom^3 = 160.21766208 GPa'],
                values={'stress_gpa': gpa.tolist(), 'pressure_gpa': data.pressure_gpa,
                        'convention': data.convention, 'validated': data.validated})
    return fig


def plot_timing(data: TimingProfile, ax=None, *, top=15, log=False, **_opts):
    """Top-N timers with call counts; explicitly marked as nested (non-additive)."""
    entries = data.top(int(top))
    if not entries:
        raise InputError(f'{data.source} has no timed counters')
    n = len(entries)
    fig, axes = axes_or_new(ax, height_mm=max(45.0, 12.0 + 4.0 * n))
    spec = style_of(fig)
    a = axes[0, 0]
    y = np.arange(n)
    seconds = [e.seconds for e in entries]
    a.barh(y, seconds, height=0.66, color=spec.color(0), zorder=3)
    a.set_yticks(y, [e.name.replace('_', ' ') for e in entries], fontsize=spec.small_fontsize)
    a.invert_yaxis()
    a.tick_params(axis='y', length=0, pad=2)
    a.spines['left'].set_visible(False)
    if log:
        a.set_xscale('log')
    peak = max(seconds)
    a.set_xlim(0 if not log else min(seconds) * 0.5, peak * 1.42 if not log else peak * 8)
    for i, e in enumerate(entries):
        a.annotate(f'{e.seconds:.3g} s · {e.calls:,} {tr("calls")}', xy=(e.seconds, i), xytext=(2, 0),
                   textcoords='offset points', ha='left', va='center', fontsize=max(5.0, spec.small_fontsize - 0.5),
                   color=NEUTRAL['ink'])
    if data.wall_seconds:
        a.axvline(data.wall_seconds, color=spec.color(1), lw=0.6, ls=(0, (4, 2)), zorder=4)
        # Label the wall-clock line on the side with more room.
        right = data.wall_seconds < 0.55 * a.get_xlim()[1]
        a.annotate(f'process_compute_total {data.wall_seconds:.3g} s', xy=(data.wall_seconds, 1.0),
                   xycoords=('data', 'axes fraction'), xytext=(2 if right else -2, -1), textcoords='offset points',
                   ha='left' if right else 'right', va='top', fontsize=spec.small_fontsize, color=spec.color(1),
                   bbox={'boxstyle': 'square,pad=0.1', 'fc': 'white', 'ec': 'none'})
    a.set_xlabel(tr('time_s'))
    minor(a, y=False)
    header = tr('nested_note')
    if data.openmp_max_threads:
        header += f'; {tr("openmp_threads")} = {data.openmp_max_threads}'
    note(a, header, loc='above left')
    attach_meta(fig, 'timing', [data.source], parameters={'top': int(top), 'log': bool(log)},
                notes=['timers are nested and overlap; never stack or sum them'],
                values={'timers': {e.name: {'seconds': e.seconds, 'calls': e.calls} for e in entries},
                        'wall_seconds': data.wall_seconds, 'origin': data.origin})
    return fig
