#!/usr/bin/env python3
"""Draw the documentation figures from benchmarks/data.

    python -I benchmarks/plot_docs.py [--data benchmarks/data] [--out docs/assets]
                                      [--lang en zh_CN] [--package-root .]

Only the files in benchmarks/data are read. Every plotted value comes from
them; nothing is typed in by hand except the documented gates and constants.
Each figure is written as PNG (600 dpi) and PDF, once per language, and a
JSON sidecar records the values drawn and the layout check result.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--data', type=Path, default=HERE / 'data')
    ap.add_argument('--out', type=Path, default=HERE.parent / 'docs' / 'assets')
    ap.add_argument('--lang', nargs='+', default=['en', 'zh_CN'], choices=['en', 'zh_CN'])
    ap.add_argument('--package-root', type=Path, default=HERE.parent,
                    help='directory that contains the sparse_dft package (release root)')
    return ap.parse_args(argv)


# ------------------------------------------------------------------ style
def figure_style(lang: str):
    """The single place that decides fonts, sizes and colours.

    It uses the package style when it is importable, so the documentation
    figures follow sparse_dft.plotting.style; otherwise a compatible fallback.
    """
    try:
        from sparse_dft.plotting.style import make_style, PALETTES, NEUTRAL
        spec = make_style('double', lang=lang, palette='okabe_ito', base_fontsize=6.5)
        rc = dict(spec.rc)
        colors = list(PALETTES['okabe_ito'])
        neutral = dict(NEUTRAL)
        source = 'sparse_dft.plotting.style'
    except Exception:  # package not importable: same rules, local copy
        colors = ['#0072B2', '#D55E00', '#009E73', '#E69F00', '#CC79A7', '#56B4E9']
        neutral = {'ink': '#222222', 'muted': '#6E6E6E', 'guide': '#BDBDBD', 'zero': '#8C8C8C',
                   'pass_fill': '#E3F1E8', 'fail_fill': '#F6DADA'}
        fonts = ['Arial', 'DejaVu Sans'] if lang == 'en' else ['PingFang SC', 'Hiragino Sans GB',
                                                             'Noto Sans CJK SC', 'Arial', 'DejaVu Sans']
        rc = {'font.family': 'sans-serif', 'font.sans-serif': fonts, 'font.size': 6.5,
              'axes.linewidth': 0.55, 'axes.spines.top': False, 'axes.spines.right': False,
              'xtick.direction': 'out', 'ytick.direction': 'out', 'xtick.major.width': 0.55,
              'ytick.major.width': 0.55, 'xtick.labelsize': 6.0, 'ytick.labelsize': 6.0,
              'axes.labelsize': 6.5, 'legend.fontsize': 5.8, 'legend.frameon': False,
              'pdf.fonttype': 42, 'svg.fonttype': 'none', 'axes.unicode_minus': False}
        source = 'fallback'
    rc.update({'figure.dpi': 150, 'savefig.dpi': 600, 'axes.titlesize': 6.8})
    return {'rc': rc, 'colors': colors, 'neutral': neutral, 'source': source}


# ----------------------------------------------------------------- labels
# Every visible string in both languages; numbers are formatted at draw time.
L = {
    'factor': ('Density support factor $f = E_\\rho / E_\\mathrm{cut}$', '密度支持倍数 $f = E_\\rho / E_\\mathrm{cut}$'),
    'wl2': ('Weighted compensation $L^2$ tail $T_{L^2}(f)$', '加权补偿电荷 $L^2$ 尾部 $T_{L^2}(f)$'),
    'budget': ('budget $2.5\\times10^{-7}$ (balanced)', '预算 $2.5\\times10^{-7}$（balanced）'),
    'gate_use': ('Gate use $u = \\max_i |\\Delta_i| / g_i$', '门槛占用 $u = \\max_i |\\Delta_i| / g_i$'),
    'gate_line': ('gate ($u = 1$)', '门槛（$u = 1$）'),
    'pass': ('pass', '通过'), 'fail': ('fail', '失败'),
    'chosen': ('auto choice', 'auto 选择'),
    'modes': ('Density modes, ratio to $16\\times$', '密度模式数（相对 $16\\times$）'),
    'modes_est': ('$(f/16)^{3/2}$', '$(f/16)^{3/2}$'),
    'speedup_wall': ('Wall-time speed-up vs off', '相对 off 的墙钟加速比'),
    'calls_ratio': ('Call-count ratio, off / auto', '调用次数比（off / auto）'),
    'H': ('$H$ applications', '$H$ 作用次数'), 'S': ('$S$ applications', '$S$ 作用次数'),
    'P': ('PAW projector blocks', 'PAW 投影块'),
    'wall': ('wall time', '墙钟时间'),
    'fit': ('fit set', '拟合集'), 'holdout': ('hold-out', '留出集'),
    'tier_default': ('default (codata)', 'default（codata）'),
    'tier_matched': ('matched (vasp_legacy)', 'matched（vasp_legacy）'),
    'tier_auto': ('auto (sparsity, codata)', 'auto（稀疏化，codata）'),
    'energy': ('Energy', '能量'), 'force': ('Force', '力'), 'stress': ('Stress', '应力'),
    'eig': ('SCF eigenvalues', 'SCF 本征值'), 'bands': ('Band path', '能带路径'),
    'dos': ('DOS $L^1$', 'DOS $L^1$'), 'proj': ('lm projections', 'lm 投影'),
    'ratio_to_gate': ('Deviation / gate', '偏差 / 门槛'),
    'bias': ('codata $-$ vasp_legacy energy (meV/atom)', 'codata $-$ vasp_legacy 能量（meV/atom）'),
    'eref': ('Atomic reference energy $\\sum E_\\mathrm{atom}/N$ (eV/atom)',
             '原子参考能 $\\sum E_\\mathrm{atom}/N$（eV/atom）'),
    'estimate': ('Hartree-ratio estimate', 'Hartree 比值估计'),
    'measured': ('measured', '实测'),
    'fitline': ('least-squares line', '最小二乘直线'),
    'gate_e': ('energy gate 1 meV/atom', '能量门槛 1 meV/atom'),
    'threads': ('OpenMP threads', 'OpenMP 线程数'), 'seconds': ('Wall time (s)', '墙钟时间（s）'),
    'v2': ('baseline', '基线'), 'v15': ('this release', '本版本'),
    'replay_stage': ('Si64 Davidson replay: solver configuration', 'Si64 Davidson 回放：求解器配置'),
    'cpu': ('CPU', 'CPU'), 'metal': ('Metal', 'Metal'),
    'ranks': ('MPI ranks x threads', 'MPI 进程 x 线程'),
    'dos_col': ('$L^1$ difference to VASP', '与 VASP 的 $L^1$ 差'),
    'mae': ('MAE', 'MAE'), 'max': ('max', '最大'),
    'band_err': ('Band-path error vs VASP (meV)', '能带路径相对 VASP 的误差（meV）'),
    'relax_dx': ('Max position deviation (Å)', '最大位置偏差（Å）'),
    'relax_de': ('Relaxation-energy deviation (eV/atom)', '弛豫能偏差（eV/atom）'),
}


def T(key: str, lang: str) -> str:
    en, zh = L[key]
    return zh if lang == 'zh_CN' else en


# ------------------------------------------------------------------ data
class Data:
    def __init__(self, root: Path):
        self.root = root
        self.read = {}

    def __call__(self, name: str) -> dict:
        path = self.root / f'{name}.json'
        payload = json.loads(path.read_text(encoding='utf-8'))
        if payload.get('schema_version') != 'sparse-dft-release-data-1':
            raise SystemExit(f'{path}: unexpected schema {payload.get("schema_version")!r}')
        import hashlib
        self.read[f'benchmarks/data/{path.name}'] = hashlib.sha256(path.read_bytes()).hexdigest()
        return payload


SYSTEM_ORDER = ['Si', 'GaAs', 'MgO', 'LiF', 'Al', 'Ni', 'Fe', 'Al_slab', 'MoS2', 'CO']
SWEEP_ORDER = ['Si2', 'MgO', 'MgO_h', 'LiF', 'Al', 'Ni', 'Si8', 'Al_slab', 'Ni4pv', 'Si64']
LABEL = {'Al_slab': 'Al(001) slab', 'MgO_h': 'MgO (O_h)', 'Ni4pv': 'Ni$_4$ (Ni_pv)', 'MoS2': 'MoS$_2$',
         'Si64': 'Si$_{64}$', 'Si8': 'Si$_8$', 'Si2': 'Si$_2$'}


def name(s: str) -> str:
    return LABEL.get(s, s)


# ------------------------------------------------------------------ output
class Saver:
    def __init__(self, out: Path, data: Data, style_source: str):
        self.out = out
        self.data = data
        self.style_source = style_source
        self.records = []

    def save(self, fig, stem: str, lang: str, values: dict) -> None:
        suffix = '' if lang == 'en' else '.zh-CN'
        self.out.mkdir(parents=True, exist_ok=True)
        issues = layout_issues(fig)
        files = []
        for ext in ('png', 'pdf'):
            path = self.out / f'{stem}{suffix}.{ext}'
            fig.savefig(path, dpi=600 if ext == 'png' else None, bbox_inches='tight', pad_inches=0.02,
                        metadata={'Creator': 'Sparse-DFT benchmarks/plot_docs.py'} if ext == 'pdf' else None)
            files.append(path.name)
        w, h = fig.get_size_inches() * 25.4
        self.records.append({'figure': stem, 'lang': lang, 'files': files, 'size_mm': [round(w, 1), round(h, 1)],
                             'layout_issues': issues, 'values': values})
        import matplotlib.pyplot as plt
        plt.close(fig)

    def write_manifest(self) -> Path:
        path = self.out / 'figures.provenance.json'
        payload = {'schema_version': 'sparse-dft-doc-figures-1', 'generator': 'benchmarks/plot_docs.py',
                   'style': self.style_source, 'inputs_sha256': dict(sorted(self.data.read.items())),
                   'figures': self.records}
        path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
        return path


def layout_issues(fig) -> list[str]:
    """Overlap checks without viewing the image: package QA when available."""
    try:
        from sparse_dft.plotting.qa import layout_report
        return list(layout_report(fig)['issues'])
    except Exception:
        pass
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = []
    for ax in fig.axes:
        for t in [ax.xaxis.label, ax.yaxis.label, *ax.texts, *(ax.get_xticklabels()), *(ax.get_yticklabels())]:
            if t.get_visible() and t.get_text().strip():
                boxes.append((t.get_text(), t.get_window_extent(renderer)))
    issues = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i][1], boxes[j][1]
            if min(a.x1, b.x1) - max(a.x0, b.x0) > 1.5 and min(a.y1, b.y1) - max(a.y0, b.y0) > 1.5:
                issues.append(f'text overlap: {boxes[i][0]!r} / {boxes[j][0]!r}')
    return issues


def panel(ax, letter: str) -> None:
    ax.text(-0.13, 1.04, letter, transform=ax.transAxes, fontsize=8, fontweight='bold', va='bottom', ha='left')


def new_fig(style, width_mm: float, height_mm: float, **kw):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4), layout='constrained', **kw)
    return fig


# ------------------------------------------------------------- figure 1
def fig_concept(style, lang, D, S):
    """Reciprocal-space supports, drawn to scale for one cutoff (schematic geometry)."""
    import numpy as np
    from matplotlib.patches import Circle
    c, n = style['colors'], style['neutral']
    fig = new_fig(style, 120, 70)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_aspect('equal')
    radii = {1: 1.0, 4: 2.0, 6: math.sqrt(6), 16: 4.0, 24: math.sqrt(24)}   # |G| ~ sqrt(E)
    fills = [(24, n['guide'], 0.10), (16, c[1], 0.10), (6, c[0], 0.16), (4, c[2], 0.22), (1, c[3], 0.45)]
    for f, col, alpha in fills:
        ax.add_patch(Circle((0, 0), radii[f], facecolor=col, alpha=alpha, edgecolor=col, lw=0.7))
    text = {
        1: ('$|\\mathbf{G}+\\mathbf{k}|^2/2 \\leq E_\\mathrm{cut}$: wavefunctions',
            '$|\\mathbf{G}+\\mathbf{k}|^2/2 \\leq E_\\mathrm{cut}$：波函数'),
        4: ('$f = 4$: exact support of $|\\tilde\\psi|^2$', '$f = 4$：$|\\tilde\\psi|^2$ 的精确支持'),
        6: ('$f = 6$: typical auto choice (metal floor)', '$f = 6$：典型 auto 选择（金属下限）'),
        16: ('$f = 16$: sparsity off (reference)', '$f = 16$：关闭稀疏化（参考）'),
        24: ('$f = 24$: probe top $q_\\mathrm{top}$', '$f = 24$：探针上限 $q_\\mathrm{top}$'),
    }
    angles = {1: 80, 4: 55, 6: 32, 16: 8, 24: -18}
    for f, ang in angles.items():
        r = radii[f]
        x, y = r * math.cos(math.radians(ang)), r * math.sin(math.radians(ang))
        ax.plot([x], [y], 'o', ms=2.2, color=n['ink'])
        tx, ty = 5.6, 4.6 - 2.3 * list(angles).index(f)
        # leader line and label drawn separately, so the label box is only the text
        ax.plot([x, tx - 0.15], [y, ty], '-', lw=0.45, color=n['muted'])
        ax.text(tx, ty, text[f][1 if lang == 'zh_CN' else 0], fontsize=6, va='center', ha='left')
    ax.annotate('', (radii[16], 0), (0, 0), arrowprops=dict(arrowstyle='->', lw=0.6, color=n['ink']))
    ax.text(radii[16] / 2, -0.35, '$G_\\rho = \\sqrt{f}\\,G_\\mathrm{cut}$', ha='center', va='top', fontsize=6)
    ax.set_xlim(-5.2, 14.5)
    ax.set_ylim(-5.4, 5.4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    modes = {f: (f / 16) ** 1.5 for f in (4, 6, 8, 12, 16)}
    S.save(fig, 'sparsity_concept', lang, {'radius_over_Gcut': {str(k): v for k, v in radii.items()},
                                           'modes_ratio_estimate': {str(k): v for k, v in modes.items()},
                                           'kind': 'schematic geometry drawn by code, to scale in |G|'})


# ------------------------------------------------------------- figure 2
def fig_tail(style, lang, D, S):
    import numpy as np
    c, n = style['colors'], style['neutral']
    tails = D('sparsity_probe_tails')['rows']
    sweep = D('sparsity_sweep')
    chosen = {r['system']: r['density_cutoff_factor'] for r in sweep['fitted_auto']}
    budget = 2.5e-7
    systems = [s for s in SWEEP_ORDER if any(t['system'] == s for t in tails)]
    fig = new_fig(style, 183, 72)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1])
    ax, bx = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    markers = ['o', 's', '^', 'D', 'v', 'P']
    values = {}
    for i, s in enumerate(systems):
        rows = sorted((t for t in tails if t['system'] == s), key=lambda t: t['factor'])
        f = [t['factor'] for t in rows]
        y = [t['weighted_compensation_l2'] for t in rows]
        col = c[i % len(c)]
        ax.plot(f, y, '-', lw=0.8, color=col, label=name(s))
        for t in rows:
            if t['sweep_pass'] is None:
                ax.plot(t['factor'], t['weighted_compensation_l2'], marker=markers[i], ms=2.2, color=col, mfc='white',
                        mew=0.5, ls='none')
            else:
                ax.plot(t['factor'], t['weighted_compensation_l2'], marker=markers[i], ms=3.6, ls='none',
                        color=col, mfc=col if t['sweep_pass'] else 'white', mew=0.8)
        if s in chosen:
            t = next(t for t in rows if t['factor'] == chosen[s])
            ax.plot(t['factor'], t['weighted_compensation_l2'], marker='*', ms=7.5, color=col, mec=n['ink'],
                    mew=0.4, ls='none', zorder=5)
        values[s] = {'factors': f, 'tails': y, 'auto_factor': chosen.get(s)}
        if not s in chosen:
            pass
        for t in rows:
            if t['sweep_pass'] is not None:
                bx.plot(t['weighted_compensation_l2'], t['sweep_gate_use'], marker=markers[i], ms=3.8, ls='none',
                        color=col, mfc=col if t['sweep_pass'] else 'white', mew=0.8)
                dx, dy, ha = {4: (3, 2, 'left'), 6: (3, -6, 'left'), 8: (-3, 2, 'right'),
                              12: (-3, 4, 'right')}.get(t['factor'], (3, 2, 'left'))
                bx.annotate(f"{t['factor']:g}", (t['weighted_compensation_l2'], t['sweep_gate_use']),
                            xytext=(dx, dy), textcoords='offset points', fontsize=5.0, color=col, ha=ha)
    ax.axhline(budget, color=n['ink'], lw=0.7, ls='--')
    ax.set_yscale('log')
    ax.set_xlabel(T('factor', lang))
    ax.set_ylabel(T('wl2', lang))
    ax.set_xticks([4, 5, 6, 8, 10, 12, 16])
    ax.set_xlim(3.5, 16.6)
    from matplotlib.lines import Line2D
    keys = [Line2D([], [], color=c[i % len(c)], marker=markers[i], ms=3, lw=0.8, label=name(s))
            for i, s in enumerate(systems)]
    keys += [Line2D([], [], color=n['ink'], marker='o', ms=3.4, ls='none', label=T('pass', lang)),
             Line2D([], [], color=n['ink'], marker='o', ms=3.4, mfc='white', ls='none', label=T('fail', lang)),
             Line2D([], [], color=n['ink'], marker='*', ms=6.5, ls='none', label=T('chosen', lang)),
             Line2D([], [], color=n['ink'], lw=0.7, ls='--', label=T('budget', lang))]
    ax.legend(handles=keys, loc='upper right', ncol=3, fontsize=5.2, handlelength=1.6, columnspacing=0.9,
              borderaxespad=0.2)
    ax.set_ylim(1e-12, 3e-3)
    bx.axvline(budget, color=n['ink'], lw=0.7, ls='--')
    bx.axhline(1.0, color=style['colors'][1], lw=0.7, ls=':')
    bx.text(1.3e-12, 1.12, T('gate_line', lang), fontsize=5.6, color=style['colors'][1], va='bottom')
    bx.set_xscale('log')
    bx.set_yscale('log')
    bx.set_xlabel(T('wl2', lang))
    bx.set_ylabel(T('gate_use', lang))
    bx.set_xlim(1e-12, 1e-3)
    bx.set_ylim(1e-3, 20)
    panel(ax, 'a')
    panel(bx, 'b')
    fails = [t for t in tails if t['sweep_pass'] is False]
    passes = [t for t in tails if t['sweep_pass'] is True]
    values['largest_passing_tail'] = max(passes, key=lambda t: t['weighted_compensation_l2'])
    values['smallest_failing_tail'] = min(fails, key=lambda t: t['weighted_compensation_l2'])
    values['budget'] = budget
    S.save(fig, 'sparsity_tail_budget', lang, values)


# ------------------------------------------------------------- figure 3
VARIANTS = ['S1_f4', 'S1_f6', 'S1_f8', 'S1_f12', 'S2_p1', 'S3_1e-2', 'S3_1e-3', 'S3_1e-4', 'S3_1e-5',
            'S4_block', 'S5_blocks', 'S5_off', 'E_bundle', 'E_bundle_f6', 'E_bundle_f8_p1']


def fig_components(style, lang, D, S):
    import numpy as np
    from matplotlib.colors import LogNorm
    n = style['neutral']
    sweep = D('sparsity_sweep')
    systems = ['Si2', 'MgO', 'MgO_h', 'LiF', 'Al', 'Ni', 'Si8', 'Al_slab']
    grid = np.full((len(systems), len(VARIANTS)), np.nan)
    passed = np.zeros_like(grid, dtype=bool)
    for r in sweep['rows']:
        if r['system'] in systems and r['variant'] in VARIANTS and r['gate_use'] is not None:
            i, j = systems.index(r['system']), VARIANTS.index(r['variant'])
            grid[i, j] = max(r['gate_use'], 1e-7)
            passed[i, j] = bool(r['pass'])
    fig = new_fig(style, 183, 66)
    ax = fig.add_subplot(1, 1, 1)
    import matplotlib
    cmap = matplotlib.colormaps['cividis'].copy()
    cmap.set_bad('#FFFFFF')
    im = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, norm=LogNorm(1e-7, 10), aspect='auto')
    for i in range(len(systems)):
        for j in range(len(VARIANTS)):
            v = grid[i, j]
            if np.isnan(v):
                ax.text(j, i, '–', ha='center', va='center', fontsize=5, color=n['muted'])
                continue
            label = f'{v:.2g}' if v >= 1e-3 else f'{v:.0e}'.replace('e-0', 'e-')
            ax.text(j, i, label, ha='center', va='center', fontsize=5.0,
                    color='white' if v < 3e-3 else '#111111', fontweight='bold' if not passed[i, j] else 'normal')
            if not passed[i, j]:
                ax.add_patch(matplotlib.patches.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec='#B2182B', lw=1.1))
    ax.set_xticks(range(len(VARIANTS)), [v.replace('E_bundle', 'E').replace('_', ' ') for v in VARIANTS],
                  rotation=40, ha='right', rotation_mode='anchor')
    ax.set_yticks(range(len(systems)), [name(s) for s in systems])
    ax.tick_params(length=0)
    for side in ('left', 'bottom'):
        ax.spines[side].set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.01)
    cb.set_label(T('gate_use', lang))
    cb.ax.axhline(1.0, color='#B2182B', lw=1.0)
    S.save(fig, 'sparsity_components', lang, {
        'systems': systems, 'variants': VARIANTS, 'gate_use': [[None if np.isnan(v) else float(v) for v in row]
                                                               for row in grid],
        'pass': passed.tolist(), 'floor_for_display': 1e-7})


# ------------------------------------------------------------- figure 4
def fig_speedup(style, lang, D, S):
    import numpy as np
    c, n = style['colors'], style['neutral']
    sweep = D('sparsity_sweep')
    rows = []
    for r in sweep['fitted_auto']:
        rows.append({'system': r['system'], 'role': sweep['systems'][r['system']]['role'], 'wall': r['speedup'],
                     'H': r['operator_hamiltonian_ratio'], 'S': r['operator_overlap_ratio'],
                     'P': r['paw_block_projectors_ratio'], 'f': r['density_cutoff_factor'], 'u': r['gate_use']})
    for r in sweep['rows']:
        if r['variant'] == 'auto' and r['system'] in ('Ni4pv', 'Si64'):
            ref = sweep['systems'][r['system']]['reference_counters']
            rows.append({'system': r['system'], 'role': 'holdout', 'wall': r['speedup'],
                         'H': ref['operator_hamiltonian_calls'] / r['operator_hamiltonian_calls'],
                         'S': ref['operator_overlap_calls'] / r['operator_overlap_calls'],
                         'P': ref['paw_block_projectors_calls'] / r['paw_block_projectors_calls'],
                         'f': r['auto_density_cutoff_factor'], 'u': r['gate_use']})
    order = [s for s in SWEEP_ORDER if any(r['system'] == s for r in rows)]
    rows.sort(key=lambda r: order.index(r['system']))
    fig = new_fig(style, 183, 70)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1])
    ax, bx = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    x = np.arange(len(rows))
    w = 0.2
    series = [('wall', T('wall', lang), n['ink']), ('H', T('H', lang), c[0]), ('S', T('S', lang), c[1]),
              ('P', T('P', lang), c[2])]
    for k, (key, label, col) in enumerate(series):
        vals = [r[key] for r in rows]
        ax.bar(x + (k - 1.5) * w, vals, w, color=col, label=label, lw=0)
        if key == 'wall':
            for xi, v in zip(x, vals):
                ax.text(xi - 1.5 * w, v * 1.06, f'{v:.2f}', ha='center', va='bottom', fontsize=5.0, rotation=90)
    ax.set_yscale('log')
    ax.set_ylim(0.8, 120)
    ax.axhline(1, color=n['zero'], lw=0.5)
    short = {'Al_slab': 'Al slab', 'Ni4pv': 'Ni$_4$', 'MgO_h': 'MgO$_h$'}
    ax.set_xticks(x, [f"{short.get(r['system'], name(r['system']))}\n$f$={r['f']:g}" for r in rows], fontsize=5.2)
    for tick, r in zip(ax.get_xticklabels(), rows):
        if r['role'] == 'holdout':
            tick.set_fontstyle('italic')
    ax.set_ylabel(T('calls_ratio', lang) + ' / ' + T('speedup_wall', lang).lower() if lang == 'en'
                  else T('calls_ratio', lang) + ' / 墙钟加速比')
    ax.legend(loc='upper left', ncol=4, fontsize=5.4)
    # b: Pareto, every sweep variant of the fit and hold-out systems
    mark = {'S1': ('o', c[0]), 'S2': ('s', c[1]), 'S3': ('^', c[2]), 'S4': ('D', c[3]), 'S5': ('v', c[4]),
            'E_': ('P', c[5])}
    seen = set()
    for r in sweep['rows']:
        if r['variant'] == 'auto' or r['gate_use'] is None or r['speedup'] is None:
            continue
        key = r['variant'][:2]
        m, col = mark.get(key, ('x', n['muted']))
        label = None if key in seen else {'E_': 'E bundle'}.get(key, key)
        seen.add(key)
        bx.plot(max(r['gate_use'], 1e-7), r['speedup'], marker=m, ls='none', ms=3.0, color=col,
                mfc=col if r['pass'] else 'white', mew=0.6, label=label, alpha=0.9)
    for r in rows:
        bx.plot(max(r['u'], 1e-7), r['wall'], marker='*', ms=6.5, color=n['ink'], ls='none')
    bx.plot([], [], marker='*', ms=6, color=n['ink'], ls='none', label=T('chosen', lang))
    bx.axvline(1.0, color='#B2182B', lw=0.7, ls='--')
    bx.axhline(1.0, color=n['zero'], lw=0.5)
    bx.set_xscale('log')
    bx.set_yscale('log')
    bx.set_xlim(5e-8, 20)
    bx.set_xlabel(T('gate_use', lang))
    bx.set_ylabel(T('speedup_wall', lang))
    bx.legend(loc='upper left', fontsize=5.2, ncol=2, handlelength=1.2)
    panel(ax, 'a')
    panel(bx, 'b')
    S.save(fig, 'sparsity_speedup', lang, {'auto_runs': rows,
                                           'note': 'wall = single-run ratio on a loaded machine; call ratios '
                                                   'count operator invocations (block calls), not columns'})


# ------------------------------------------------------------- figure 5
METRICS = [('energy', 'energy_ev_per_atom', 'energy_ev_per_atom'),
           ('force', 'force_max_ev_per_angstrom', 'force_ev_per_angstrom'),
           ('stress', 'stress_max_gpa', 'stress_gpa'),
           ('eig', 'eigenvalue_max_ev', 'eigenvalue_ev'),
           ('bands', 'bands_max_ev', 'eigenvalue_ev'),
           ('dos', 'dos_l1', 'dos_l1'),
           ('proj', 'projection_worst_diff', None)]


def fig_anchor(style, lang, D, S):
    import numpy as np
    c, n = style['colors'], style['neutral']
    A = D('anchor_vasp')
    gates = A['gates']
    tiers = [('default', 'o', c[1]), ('matched', 's', c[0]), ('auto', '^', c[2])]
    fig = new_fig(style, 183, 78)
    axes = fig.subplots(1, len(METRICS), sharey=True)
    y = {s: i for i, s in enumerate(SYSTEM_ORDER)}
    values = {}
    for ax, (key, col, gate_key) in zip(axes, METRICS):
        ax.axvspan(1, 1e3, color=n['fail_fill'], lw=0, zorder=0)
        ax.axvline(1, color='#B2182B', lw=0.7)
        for k, (tier, m, colr) in enumerate(tiers):
            xs, ys = [], []
            for r in A['scf']:
                if r['tier'] != tier or r.get(col) is None:
                    continue
                gate = r['projection_worst_gate'] if gate_key is None else gates[gate_key]
                if gate is None:
                    continue
                v = abs(r[col]) / gate
                xs.append(max(v, 1e-5))
                ys.append(y[r['system']] + (k - 1) * 0.22)
                values.setdefault(key, {}).setdefault(tier, {})[r['system']] = v
            ax.plot(xs, ys, marker=m, ls='none', ms=3.0, color=colr, mfc=colr if tier != 'auto' else 'white',
                    mew=0.7, label=T(f'tier_{tier}', lang))
        ax.set_xscale('log')
        ax.set_xlim(1e-5, 30)
        ax.set_xticks([1e-4, 1e-2, 1])
        ax.set_title(T(key, lang), fontsize=6.4)
        ax.grid(axis='y', color=n['guide'], lw=0.3, alpha=0.5)
    axes[0].set_yticks(range(len(SYSTEM_ORDER)), [name(s) for s in SYSTEM_ORDER])
    axes[0].invert_yaxis()
    axes[0].set_ylim(len(SYSTEM_ORDER) - 0.5, -0.5)
    fig.supxlabel(T('ratio_to_gate', lang), fontsize=6.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside upper center', ncol=3, fontsize=5.8)
    S.save(fig, 'accuracy_anchor', lang, {'deviation_over_gate': values, 'gates': gates,
                                          'projection_gate': 'per worst group: 5e-4 x bands in group + 5e-4',
                                          'display_floor': 1e-5})


# ------------------------------------------------------------- figure 6
def fig_codata(style, lang, D, S):
    import numpy as np
    c, n = style['colors'], style['neutral']
    B = D('codata_bias')
    A = D('anchor_vasp')
    rows = sorted(B['rows'], key=lambda r: r['atomic_reference_ev_per_atom'])
    x = np.array([r['atomic_reference_ev_per_atom'] for r in rows])
    yv = np.array([r['bias_mev_per_atom'] for r in rows])
    est = np.array([r['hartree_ratio_estimate_mev_per_atom'] for r in rows])
    fit = B['fit_bias_vs_atomic_reference']
    fig = new_fig(style, 183, 70)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1])
    ax, bx = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    ax.plot(x, yv, 'o', ms=3.6, color=c[1], label=T('measured', lang), zorder=3)
    ax.plot(x, est, 'D', ms=3.0, color=c[0], mfc='white', mew=0.7, label=T('estimate', lang), zorder=3)
    xx = np.array([0.0, x.max() * 1.05])
    ax.plot(xx, fit['intercept'] + fit['slope'] * xx, '-', lw=0.7, color=c[1], alpha=0.7,
            label=f"{T('fitline', lang)}: $y = {fit['slope']:.4f}x {fit['intercept']:+.2f}$, $R^2 = {fit['r2']:.3f}$")
    ax.axhline(1.0, color='#B2182B', lw=0.6, ls='--', label=T('gate_e', lang))
    # Al / Al(001) slab and Si / GaAs sit on top of each other at this scale: one label each
    labels = {'Al': 'Al, Al(001) slab', 'Al_slab': None, 'Si': 'Si, GaAs', 'GaAs': None}
    offsets = {'Al': (4, -8, 'left'), 'Si': (-4, 7, 'right'), 'GaAs': (6, -7, 'left'), 'MgO': (-5, 5, 'right'),
               'MoS2': (6, -6, 'left'), 'CO': (-5, 4, 'right'), 'LiF': (6, -5, 'left'), 'Fe': (6, -6, 'left'),
               'Ni': (-6, -2, 'right')}
    for r in rows:
        text = labels.get(r['system'], name(r['system']))
        if text is None:
            continue
        dx, dy, ha = offsets.get(r['system'], (5, 2, 'left'))
        ax.annotate(text, (r['atomic_reference_ev_per_atom'], r['bias_mev_per_atom']),
                    xytext=(dx, dy), textcoords='offset points', fontsize=5.0, ha=ha, va='center')
    ax.set_xlabel(T('eref', lang))
    ax.set_ylabel(T('bias', lang))
    ax.set_xlim(0, x.max() * 1.08)
    ax.set_ylim(-2.0, yv.max() * 1.15)
    ax.axhline(0, color=n['zero'], lw=0.4)
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.0), fontsize=5.2, ncol=2, borderaxespad=0.1)
    # b: |dE| per atom against VASP for default and matched tiers
    order = SYSTEM_ORDER
    yy = np.arange(len(order))
    for k, (tier, m, colr) in enumerate([('default', 'o', c[1]), ('matched', 's', c[0]), ('auto', '^', c[2])]):
        vals = [next(abs(r['energy_ev_per_atom']) for r in A['scf'] if r['system'] == s and r['tier'] == tier)
                for s in order]
        bx.plot(vals, yy + (k - 1) * 0.22, marker=m, ls='none', ms=3.2, color=colr,
                mfc=colr if tier != 'auto' else 'white', mew=0.7, label=T(f'tier_{tier}', lang))
    bx.axvline(1e-3, color='#B2182B', lw=0.7)
    bx.axvspan(1e-3, 1, color=n['fail_fill'], lw=0, zorder=0)
    bx.set_xscale('log')
    bx.set_xlim(1e-7, 0.05)
    bx.set_yticks(yy, [name(s) for s in order])
    bx.set_ylim(len(order) - 0.5, -0.5)
    bx.set_xlabel('$|\\Delta E|$ vs VASP (eV/atom)' if lang == 'en' else '与 VASP 的 $|\\Delta E|$（eV/atom）')
    bx.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=3, fontsize=5.2, borderaxespad=0.1)
    panel(ax, 'a')
    panel(bx, 'b')
    S.save(fig, 'accuracy_codata_bias', lang, {'fit': fit, 'rows': rows, 'gate_ev_per_atom': 1e-3})


# ------------------------------------------------------------- figure 7
def fig_electronic(style, lang, D, S):
    import numpy as np
    c, n = style['colors'], style['neutral']
    dos = D('dos_vasp')
    bands = D('bands_vasp')
    A = D('anchor_vasp')
    fig = new_fig(style, 183, 66)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 0.9, 1.0])
    ax, bx, cx = (fig.add_subplot(gs[i]) for i in range(3))
    groups = []
    for r in dos['rows']:
        key = (r['system'], r['channel'])
        if key not in groups:
            groups.append(key)
    for i, (s, ch) in enumerate(groups):
        rows = [r for r in dos['rows'] if r['system'] == s and r['channel'] == ch]
        pd = [r for r in rows if r['column'] != 'total' and r['integral_vasp'] >= 0.1]
        tot = next(r for r in rows if r['column'] == 'total')
        jitter = np.linspace(-0.18, 0.18, max(len(pd), 1))
        ax.plot([r['l1_normalised'] for r in pd], i + jitter[:len(pd)], 'o', ms=2.4, color=c[0], mfc='white',
                mew=0.6, label=('PDOS (integral ≥ 0.1)' if lang == 'en' else 'PDOS（积分 ≥ 0.1）') if i == 0 else None)
        ax.plot(tot['l1_normalised'], i, 'D', ms=3.6, color=c[1], label=('total DOS' if lang == 'en' else '总 DOS')
                if i == 0 else None)
    ax.axvline(1e-2, color='#B2182B', lw=0.7)
    ax.text(1e-2 * 0.85, len(groups) - 0.6, '1 %', color='#B2182B', fontsize=5.4, ha='right')
    ax.set_xscale('log')
    ax.set_xlim(1e-5, 2e-2)
    ax.set_yticks(range(len(groups)), [f'{name(s)}' + ('' if ch == 'total' else f' ({ch})') for s, ch in groups])
    ax.set_ylim(len(groups) - 0.5, -0.5)
    ax.set_xlabel(T('dos_col', lang))
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2, fontsize=5.2, borderaxespad=0.1)
    # b: band path MAE / max (meV), no alignment
    rows = bands['rows']
    x = np.arange(len(rows))
    bx.bar(x - 0.18, [r['mae_ev'] * 1e3 for r in rows], 0.36, color=c[0], label=T('mae', lang))
    bx.bar(x + 0.18, [r['max_abs_error_ev'] * 1e3 for r in rows], 0.36, color=c[1], label=T('max', lang))
    for xi, r in zip(x, rows):
        bx.text(xi + 0.18, r['max_abs_error_ev'] * 1e3 * 1.15, f"{r['max_abs_error_ev'] * 1e3:.2f}", ha='center',
                fontsize=5.0)
        bx.text(xi, 0.0045, f"N={r['samples']}", ha='center', fontsize=5.0, color=n['muted'])
    bx.axhline(5.0, color='#B2182B', lw=0.7)
    bx.text(len(rows) - 0.5, 5.0 * 0.78, '5 meV', color='#B2182B', fontsize=5.2, ha='right', va='top')
    bx.set_yscale('log')
    bx.set_ylim(0.003, 12)
    bx.set_xticks(x, [name(r['system']) for r in rows])
    bx.set_ylabel(T('band_err', lang))
    bx.legend(loc='upper left', fontsize=5.2, ncol=2)
    # c: relaxations, max position deviation vs relaxation-energy deviation
    for k, (tier, m, colr) in enumerate([('default', 'o', c[1]), ('matched', 's', c[0]), ('auto', '^', c[2])]):
        for j, case in enumerate(('Si_ionic', 'MgO_cell')):
            r = next(r for r in A['relax'] if r['case'] == case and r['tier'] == tier)
            metric = r['max_displacement_angstrom'] if case == 'Si_ionic' else r['lattice_metric_relative']
            cx.plot(max(metric, 1e-9), abs(r['relaxation_energy_diff_ev_per_atom']), marker=m, ls='none', ms=3.4,
                    color=colr, mfc=colr if j == 0 else 'white', mew=0.7,
                    label=T(f'tier_{tier}', lang) if j == 0 else None)
    cx.set_xscale('log')
    cx.set_yscale('log')
    cx.set_xlim(1e-6, 1e-3)
    cx.set_ylim(1e-9, 1e-4)
    cx.set_xlabel('Si: $\\max|\\Delta x|$ (Å); MgO: $\\Delta g/g$' if lang == 'en'
                  else 'Si：$\\max|\\Delta x|$（Å）；MgO：$\\Delta g/g$')
    cx.set_ylabel(T('relax_de', lang))
    cx.legend(loc='upper left', fontsize=5.0)
    cx.text(0.97, 0.05, ('filled: Si ionic, open: MgO cell' if lang == 'en' else '实心：Si 离子弛豫，空心：MgO 晶胞弛豫'),
            transform=cx.transAxes, ha='right', fontsize=5.0, color=n['muted'])
    for a, l in zip((ax, bx, cx), 'abc'):
        panel(a, l)
    S.save(fig, 'accuracy_electronic', lang, {'dos_groups': groups, 'bands': rows,
                                              'relax': A['relax'], 'gates': {'dos_l1': 0.01, 'bands_ev': 5e-3}})


# ------------------------------------------------------------- figure 8
def fig_performance(style, lang, D, S):
    import numpy as np
    c, n = style['colors'], style['neutral']
    P = D('performance')
    fig = new_fig(style, 183, 120)
    axs = fig.subplots(2, 2)
    ax, bx, cx, dx = axs.ravel()
    sc = P['runs']['scale_Si8']
    for label, key, colr, m in (('v2', 'v2', n['muted'], 'o'), ('v15', 'w1b', c[0], 's')):
        pts = sorted(((r['threads'], r['wall_seconds']) for r in sc if r['label'] == key))
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=m, ms=3.2, lw=0.8, color=colr, label=T(label, lang))
        for t, w in pts:
            if t in (1, 8):
                ax.annotate(f'{w:.1f}', (t, w), xytext=(3, 3), textcoords='offset points', fontsize=5, color=colr)
    later = [r for r in P['runs']['conv'] if r['input'] == 'Si8' and r['threads'] == 8]
    for r in later:
        ax.plot(8, r['wall_seconds'], marker='x', ms=3.4, color=c[2], ls='none')
    ax.plot([], [], 'x', color=c[2], label=('intermediate configurations (8 threads)' if lang == 'en' else '中间配置（8 线程）'))
    ax.set_xscale('log', base=2)
    ax.set_xticks([1, 2, 4, 8, 12, 18], ['1', '2', '4', '8', '12', '18'])
    ax.set_xlabel(T('threads', lang))
    ax.set_ylabel(T('seconds', lang))
    ax.set_ylim(0, 125)
    ax.legend(loc='upper right', fontsize=5.3)
    ax.set_title('Si$_8$, Γ, 25 SCF iterations' if lang == 'en' else 'Si$_8$，Γ 点，25 次 SCF 迭代', fontsize=6.2)
    # b: Si64 replay stages
    rep = P['replay']['runs']
    stages = sorted({r['run'].split('_')[0] for r in rep}, key=lambda s: (len(s), s))
    for backend, colr, m in (('cpu', c[0], 'o'), ('mps', c[1], 's')):
        pts = [(stages.index(r['run'].split('_')[0]), r['solver_seconds']) for r in rep if r['backend'] == backend]
        pts.sort()
        bx.plot([p[0] + 1 for p in pts], [p[1] for p in pts], marker=m, ms=3.2, lw=0.8, color=colr,
                label=T('cpu' if backend == 'cpu' else 'metal', lang))
        for i, w in pts:
            bx.annotate(f'{w:.1f}', (i + 1, w), xytext=(4, 3) if backend == 'cpu' else (-4, -7),
                        textcoords='offset points', fontsize=5.0, color=colr,
                        ha='left' if backend == 'cpu' else 'right')
    bx.set_yscale('log')
    bx.set_ylim(30, 500)
    bx.set_xticks(range(1, len(stages) + 1))
    bx.set_xlabel(T('replay_stage', lang))
    bx.set_ylabel(T('seconds', lang))
    bx.legend(loc='upper right', fontsize=5.3)
    bx.set_title('NPW 16663, 136 roots, residual ≤ 1e-12 Ha' if lang == 'en' else 'NPW 16663，136 个根，残差 ≤ 1e-12 Ha',
                 fontsize=6.2)
    # c: MPI layouts on Ni
    mp = [r for r in P['runs']['mpi'] if r['input'] == 'Ni']
    labels = [f"{r['ranks']}×{r['threads']}" for r in mp]
    cols = [c[0] if r['ranks'] * r['threads'] == 8 else c[3] for r in mp]
    cx.bar(range(len(mp)), [r['wall_seconds'] for r in mp], color=cols, width=0.62)
    for i, r in enumerate(mp):
        cx.text(i, r['wall_seconds'] + 1, f"{r['wall_seconds']:.1f}", ha='center', fontsize=5)
    cx.set_xticks(range(len(mp)), labels)
    cx.set_xlabel(T('ranks', lang))
    cx.set_ylabel(T('seconds', lang))
    cx.set_ylim(0, 62)
    cx.set_title('fcc Ni, spin, 64 k × 2, converged; bitwise-equal results' if lang == 'en'
                 else 'fcc Ni，自旋，64 k × 2，收敛；结果逐位相同', fontsize=6.2)
    from matplotlib.patches import Patch
    cx.legend(handles=[Patch(color=c[0], label='8 cores in total' if lang == 'en' else '共 8 核'),
                       Patch(color=c[3], label='18 cores in total' if lang == 'en' else '共 18 核')],
              loc='upper left', fontsize=5.3)
    # d: Metal vs CPU on the same converged Si8 input
    gpu = [r for r in P['runs']['gpu'] if r['label'].startswith('mps2')]
    names = ['CPU' if r['env'].endswith('cpu') else 'Metal' for r in gpu]
    dx.bar(range(len(gpu)), [r['wall_seconds'] for r in gpu], color=[c[0], c[1]][:len(gpu)], width=0.55)
    for i, r in enumerate(gpu):
        dx.text(i, r['wall_seconds'] + 0.4, f"{r['wall_seconds']:.1f} s\n|ΔE| {abs(r['dE_ev']):.1e} eV",
                ha='center', va='bottom', fontsize=5.0)
    dx.set_xticks(range(len(gpu)), names)
    dx.set_ylabel(T('seconds', lang))
    dx.set_ylim(0, 32)
    dx.set_title('Si$_8$ converged, 8 threads; ΔE vs baseline' if lang == 'en'
                 else 'Si$_8$ 收敛，8 线程；ΔE 相对基线', fontsize=6.2)
    for a, l in zip((ax, bx, cx, dx), 'abcd'):
        panel(a, l)
    S.save(fig, 'performance', lang, {
        'machine': P['machine'], 'scale_Si8': sc, 'replay': rep, 'mpi_Ni': mp, 'metal_vs_cpu': gpu,
        'replay_stage_order': stages})


FIGURES = [fig_concept, fig_tail, fig_components, fig_speedup, fig_anchor, fig_codata, fig_electronic,
           fig_performance]


def main(argv=None) -> int:
    a = parse_args(argv)
    if a.package_root and (a.package_root / 'sparse_dft').is_dir():
        sys.path.insert(0, str(a.package_root.resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data = Data(a.data)
    saver = None
    for lang in a.lang:
        style = figure_style(lang)
        saver = saver or Saver(a.out, data, style['source'])
        for draw in FIGURES:
            with plt.rc_context(style['rc']):
                draw(style, lang, data, saver)
    path = saver.write_manifest()
    bad = [(r['figure'], r['lang'], r['layout_issues']) for r in saver.records if r['layout_issues']]
    for rec in saver.records:
        print(f"{rec['figure']:24s} {rec['lang']:6s} {rec['size_mm']} issues={len(rec['layout_issues'])}")
    print(f'manifest: {path}')
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main())