"""Workflow figures: relaxation trajectory, sparsity Pareto front, parallel scaling."""
from __future__ import annotations

import math

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

from ..analysis.relax import RelaxTrajectory
from ..analysis.sweep import GATES, ScalingSet, SweepTable
from ..errors import InputError
from ._util import ZERO_KW, attach_meta, axes_or_new, integer_xaxis, minor, note, sci
from .labels import tr
from .style import MARKERS, NEUTRAL, style_of, tint


def _fmt_log(v, _=None):
    if v <= 0:
        return ''
    return f'{v:g}' if 1e-3 <= v <= 1e4 else f'$10^{{{int(round(math.log10(v)))}}}$'


def _plain_log(axis, lo=None, hi=None):
    """Decade ticks; an axis spanning < 2.5 decades also labels the 2 and 5 minors."""
    axis.set_major_locator(LogLocator(base=10, numticks=8))
    axis.set_major_formatter(FuncFormatter(_fmt_log))
    short = lo is not None and hi is not None and lo > 0 and math.log10(hi / lo) < 2.5
    if short:
        axis.set_minor_locator(LogLocator(base=10, subs=(2.0, 5.0), numticks=40))
        axis.set_minor_formatter(FuncFormatter(_fmt_log))
    else:
        axis.set_minor_locator(LogLocator(base=10, subs=np.arange(2.0, 10.0), numticks=40))
        axis.set_minor_formatter(NullFormatter())


def plot_relax(data: RelaxTrajectory, ax=None, **_opts):
    """Energy, max force (or stress) and step length per accepted relaxation step."""
    has_step = any(v is not None for v in data.step_length)
    rows = 3 if has_step else 2
    fig, axes = axes_or_new(ax, nrows=rows, ncols=1, sharex=True, height_mm=30.0 + 26.0 * rows)
    spec = style_of(fig)
    x = data.steps
    final = next((e for e in reversed(data.energy_ev) if e is not None), None)
    a0 = axes[0, 0]
    if final is not None:
        rel = [1e3 * (e - final) if e is not None else np.nan for e in data.energy_ev]
        a0.plot(x, rel, color=spec.color(0), marker='o', ms=2.6, mew=0, lw=0.9)
        a0.axhline(0.0, **ZERO_KW)
        a0.set_ylabel(tr('energy_change_mev'))
        a0.annotate(f'$E_\\mathrm{{final}}$ = {final:.6f} eV'.replace('-', '−'), xy=(1, 1), xycoords='axes fraction',
                    xytext=(0, 1), textcoords='offset points', ha='right', va='bottom', fontsize=spec.small_fontsize,
                    color=NEUTRAL['muted'])
    minor(a0, x=False)
    a1 = axes[1, 0]
    resid = [v if v is not None and v > 0 else np.nan for v in data.residual_display]
    a1.plot(x, resid, color=spec.color(1), marker='s', ms=2.4, mew=0, lw=0.9)
    a1.set_yscale('log')
    a1.yaxis.set_minor_formatter(NullFormatter())
    tol = data.tolerance_display
    if tol:
        a1.axhline(tol, color=NEUTRAL['ink'], lw=0.6, ls=(0, (4, 2)))
        origin = '' if data.tolerance_origin == 'file' else f' ({data.tolerance_origin})'
        a1.annotate(f'{tr("tolerance")} {tol:.3g}{origin}', xy=(1.0, tol), xycoords=('axes fraction', 'data'),
                    xytext=(0, 2), textcoords='offset points', ha='right', va='bottom',
                    fontsize=spec.small_fontsize, color=NEUTRAL['muted'])
    a1.set_ylabel(tr('max_stress') if data.kind == 'cell' else tr('max_force'))
    if has_step:
        a2 = axes[2, 0]
        a2.plot(x, [v if v is not None else np.nan for v in data.step_length], color=spec.color(2), marker='^',
                ms=2.6, mew=0, lw=0.9)
        a2.set_ylabel(tr('step_length_strain') if data.kind == 'cell' else tr('step_length'))
        a2.set_ylim(bottom=0)
        minor(a2, x=False)
    axes[-1, 0].set_xlabel(tr('relax_step'))
    for a in axes.ravel():
        integer_xaxis(a)
    for a in axes.ravel()[:-1]:
        a.tick_params(labelbottom=False)
    notes = []
    if data.status:
        notes.append(str(data.status))
    if data.total_energy_valid and any(v is False for v in data.total_energy_valid):
        notes.append(tr('diagnostic_note'))
    if notes:
        note(a0, '; '.join(notes), loc='above left')
    attach_meta(fig, 'relax', [data.source], values={
        'kind': data.kind, 'status': data.status, 'steps': len(x), 'final_energy_ev': final,
        'final_residual': data.residual[-1] if data.residual else None, 'tolerance': data.tolerance,
        'tolerance_origin': data.tolerance_origin, 'units': data.units})
    return fig


def plot_sparsity_pareto(data: SweepTable, ax=None, *, metric='auto', systems=None, label_points=False,
                         floor=1e-4, **_opts):
    """Gate use (max over Table-1 gates; 1 = limit) vs counter-based work reduction."""
    if metric not in ('auto', 'operator_applications', 'overlap_applications', 'density_modes', 'xc_grid_points',
                      'density_functional_seconds', 'diagonalization_seconds', 'total_seconds'):
        raise InputError(f'unknown work metric {metric!r}')
    chosen = [s for s in data.systems if not systems or s in systems]
    fig, axes = axes_or_new(ax, height_mm=68.0)
    spec = style_of(fig)
    a = axes[0, 0]
    points, omitted, unconverged, used_metrics = [], 0, 0, set()
    for si, system in enumerate(chosen):
        color = spec.color(si)
        marker = MARKERS[si % len(MARKERS)]
        xs, ys, passed = [], [], []
        for r in data.system_rows(system):
            if not r.converged or r.gate_use is None:
                unconverged += 1
                continue
            w, used = r.work_reduction(metric)
            if w is None or w <= 1.02:
                omitted += 1
                continue
            used_metrics.add(used)
            xs.append(w); ys.append(max(r.gate_use, floor)); passed.append(r.gate_use <= 1.0)
            points.append({'system': system, 'variant': r.variant, 'work': w, 'work_metric': used,
                           'gate_use': r.gate_use, 'pass': r.gate_use <= 1.0})
        if not xs:
            continue
        xs, ys, passed = np.array(xs), np.array(ys), np.array(passed)
        a.scatter(xs[passed], ys[passed], s=14, marker=marker, facecolor=color, edgecolor='white', linewidth=0.4,
                  zorder=4, label=system)
        if (~passed).any():
            a.scatter(xs[~passed], ys[~passed], s=14, marker=marker, facecolor='white', edgecolor=color,
                      linewidth=0.8, zorder=4)
        if label_points:
            for p in points:
                if p['system'] == system:
                    a.annotate(p['variant'], (p['work'], max(p['gate_use'], floor)), xytext=(2, 2),
                               textcoords='offset points', fontsize=5, color=NEUTRAL['muted'])
    if not points:
        raise InputError('no converged sweep rows with a work reduction > 1.02 for the chosen metric')
    a.set_xscale('log'); a.set_yscale('log')
    ys_all = [max(p['gate_use'], floor) for p in points]
    clamped = sum(1 for p in points if p['gate_use'] < floor)
    ylo = min(ys_all) / 2.0
    yhi = max(max(ys_all) * 2.5, 3.0)
    a.set_ylim(ylo, yhi)
    xs_all = [p['work'] for p in points]
    xlo, xhi = min(xs_all) / 1.25, max(xs_all) * 1.3
    a.set_xlim(xlo, xhi)
    _plain_log(a.xaxis, xlo, xhi)
    _plain_log(a.yaxis, ylo, yhi)
    a.axhspan(1.0, yhi, color=NEUTRAL['fail_fill'], lw=0, zorder=0)
    a.axhspan(ylo, 1.0, color=NEUTRAL['pass_fill'], lw=0, zorder=0, alpha=0.6)
    a.axhline(1.0, color=NEUTRAL['ink'], lw=0.7, zorder=3)
    a.text(0.99, 0.985, tr('fail'), transform=a.transAxes, ha='right', va='top', fontsize=spec.small_fontsize,
           color='#9E2A2A')
    a.text(0.99, 0.015, tr('pass'), transform=a.transAxes, ha='right', va='bottom', fontsize=spec.small_fontsize,
           color='#2E6B45')
    metric_name = metric if metric != 'auto' else 'max of H+S, density, XC counters'
    a.set_xlabel(tr('work_reduction', metric=metric_name.replace('_', ' ')))
    a.set_ylabel(tr('gate_use'))
    handles, labels = a.get_legend_handles_labels()
    handles.append(Line2D([], [], ls='none', marker='o', mfc='white', mec=NEUTRAL['ink'], mew=0.8, ms=3.5))
    labels.append(tr('fail'))
    handles.append(Line2D([], [], color=NEUTRAL['ink'], lw=0.7))
    labels.append(tr('gate_limit'))
    # Outside on the right: the scatter fills the whole plane, so no inside
    # corner is reliably free of points.
    a.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.01, 1.0), ncol=1, fontsize=spec.small_fontsize,
             handletextpad=0.3, borderaxespad=0.0, handlelength=1.2)
    msgs = []
    if clamped:
        a.axhline(floor, color=NEUTRAL['guide'], lw=0.4, ls=(0, (1.5, 1.5)), zorder=1)
        msgs.append(f'{clamped} ' + r'at $<' + sci(floor) + r'$ (floor)')
    if omitted:
        msgs.append(tr('omitted_rows', n=omitted))
    if unconverged:
        msgs.append(tr('not_converged_rows', n=unconverged))
    if msgs:
        note(a, '\n'.join(msgs), loc='above left')
    attach_meta(fig, 'sparsity_pareto', [data.source], parameters={'metric': metric, 'systems': chosen},
                notes=['gate_use = max(error/limit) over Table-1 gates: 1 meV/atom, 1e-3 eV/A, 0.01 GPa, 5 meV band window',
                       'work = reference/run counter ratio (> 1: less work); wall time reported separately'],
                values={'points': points, 'gates': GATES, 'work_metrics_used': sorted(used_metrics),
                        'omitted_low_reduction': omitted, 'unconverged': unconverged,
                        'clamped_below_floor': clamped, 'floor': floor})
    return fig


def plot_scaling(data: ScalingSet, ax=None, *, ideal=True, **_opts):
    """Wall time, speed-up, parallel efficiency and CPU occupancy vs workers, per build."""
    fig, axes = axes_or_new(ax, nrows=2, ncols=2, sharex=True, height_mm=74.0)
    spec = style_of(fig)
    t_ax, s_ax, e_ax, o_ax = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    axis_key = data.axis
    xkey = {'threads': 'threads', 'ranks': 'ranks', 'workers': 'workers'}[axis_key]
    xs_all, summary, missing_baseline = [], {}, []
    for bi, build in enumerate(data.builds):
        pts = data.points(build)
        color, marker = spec.color(bi), MARKERS[bi % len(MARKERS)]
        x = [getattr(p, xkey) if xkey != 'workers' else p.workers for p in pts]
        xs_all += x
        kw = dict(color=color, marker=marker, ms=3.0, mew=0, lw=0.9, label=build)
        wall = [p.wall_seconds for p in pts]
        t_ax.plot(x, wall, **kw)
        if any(p.repeats > 1 for p in pts):
            t_ax.vlines(x, [p.wall_min_seconds for p in pts], [p.wall_max_seconds for p in pts], color=color, lw=0.6)
        if all(p.speedup is None for p in pts):
            missing_baseline.append(build)
        s_ax.plot(x, [p.speedup if p.speedup is not None else np.nan for p in pts], **kw)
        e_ax.plot(x, [p.efficiency if p.efficiency is not None else np.nan for p in pts], **kw)
        o_ax.plot(x, [p.occupancy if p.occupancy is not None else np.nan for p in pts], **kw)
        summary[build] = [{'threads': p.threads, 'ranks': p.ranks, 'wall_seconds': p.wall_seconds,
                           'repeats': p.repeats, 'speedup': p.speedup, 'efficiency': p.efficiency,
                           'occupancy': p.occupancy} for p in pts]
    if not xs_all:
        raise InputError('no scaling points to plot')
    xmin, xmax = min(xs_all), max(xs_all)
    for a in (t_ax, s_ax):
        a.set_xscale('log', base=2)
    t_ax.set_yscale('log')
    s_ax.set_yscale('log', base=2)
    ticks = [2 ** i for i in range(int(math.floor(math.log2(xmin))), int(math.ceil(math.log2(xmax))) + 1)]
    for a in axes.ravel():
        a.set_xticks(ticks, [str(t) for t in ticks])
        a.xaxis.set_minor_formatter(NullFormatter())
        a.set_xlim(xmin / 1.3, xmax * 1.3)
    tlo, thi = t_ax.get_ylim()
    _plain_log(t_ax.yaxis, tlo, thi)
    s_ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    s_ax.yaxis.set_minor_formatter(NullFormatter())
    if ideal:
        line = [t for t in ticks if xmin <= t <= xmax] or [xmin, xmax]
        s_ax.plot(line, line, color=NEUTRAL['muted'], lw=0.6, ls=(0, (4, 2)), zorder=1, label=tr('ideal'))
        e_ax.axhline(1.0, color=NEUTRAL['muted'], lw=0.6, ls=(0, (4, 2)), zorder=1)
        o_ax.axhline(1.0, color=NEUTRAL['muted'], lw=0.6, ls=(0, (4, 2)), zorder=1)
    t_ax.set_ylabel(tr('wall_time_s'))
    s_ax.set_ylabel(tr('speedup'))
    e_ax.set_ylabel(tr('efficiency'))
    o_ax.set_ylabel(tr('occupancy'))
    e_ax.set_ylim(0, max(1.1, np.nanmax([p['efficiency'] or 0 for v in summary.values() for p in v]) * 1.1))
    o_ax.set_ylim(0, max(1.1, np.nanmax([p['occupancy'] or 0 for v in summary.values() for p in v]) * 1.1))
    label = tr({'threads': 'threads', 'ranks': 'ranks', 'workers': 'workers'}[axis_key])
    for a in (e_ax, o_ax):
        a.set_xlabel(label)
        minor(a, x=False)
    for a in (t_ax, s_ax):
        a.tick_params(labelbottom=False)
    handles, labels = s_ax.get_legend_handles_labels()
    t_ax.legend(handles, labels, loc='upper right', fontsize=spec.small_fontsize)
    if missing_baseline:
        note(s_ax, tr('baseline_missing') + ': ' + ', '.join(missing_baseline), loc='above right')
    attach_meta(fig, 'scaling', data.sources, values={'system': data.system, 'axis': axis_key, 'builds': summary},
                notes=['occupancy = (user+sys)/(wall*threads*ranks); speed-up and efficiency vs the 1-worker run '
                       'of the same build'])
    return fig
