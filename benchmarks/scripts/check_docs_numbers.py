#!/usr/bin/env python3
"""Check the documentation against benchmarks/data.

    python -I benchmarks/scripts/check_docs_numbers.py [--root .] [--release-root DIR ...] [--json REPORT]

Five checks, reported separately:

1. numbers: every number in the prose and tables of the English pages must
   round to a value in benchmarks/data, docs/assets/figures.provenance.json,
   docs/assets/electronic/atlas.provenance.json or docs/assets/gui/screenshots.json
   (after a unit scale of 1, 1e3, 1e-3 or 1e2), to a quantity derived from
   them by a formula in DERIVED, or to a documented constant in CONSTANTS.
   Code spans, code blocks, display equations, link targets, headings,
   section/table/figure/citation numbers, versions, dates and reference lists
   are skipped. Integers below 100 match almost any pool and are counted, not
   judged.
2. parity: each English page and its .zh-CN twin carry the same numbers, line by line.
3. links: relative links and #anchors resolve (in --root or any --release-root).
4. figures: every image exists; Chinese pages use the .zh-CN figure when one exists.
5. attribution: no name given with --forbid appears, and the author's name
   appears only between the license-block markers of the two READMEs.

Exit code 0 when checks 1-5 pass, 1 otherwise. Missing files that another
part of the release provides are listed but do not fail check 3 or 4 when
they are named in EXPECTED_ELSEWHERE.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAGES = ['README.md', 'docs/THEORY.md', 'docs/SPARSIFICATION.md', 'docs/ACCURACY.md',
         'docs/PERFORMANCE.md', 'docs/USER_GUIDE.md', 'benchmarks/README.md']
TWINS = {p: p[:-3] + '.zh-CN.md' for p in PAGES if p != 'benchmarks/README.md'}
SCALES = (1.0, 1e3, 1e-3, 1e2)

# Files written by other parts of the release build; a link to them is not an error here.
EXPECTED_ELSEWHERE = ('LICENSE', 'NOTICE.md', 'CITATION.cff', 'docs/LEGAL.md', 'docs/LEGAL.zh-CN.md',
                      'docs/PROVENANCE.md', 'docs/PROVENANCE.zh-CN.md', 'docs/assets/gui/',
                      'docs/assets/electronic/')

FORBIDDEN: tuple[str, ...] = ()   # names that must not appear; given with --forbid
AUTHOR = 'Yufeng Zhan'
BLOCK = re.compile(r'<!-- license-block:start -->.*?<!-- license-block:end -->', re.S)

# Documented constants and code defaults that are not results; each names where it is defined.
CONSTANTS = {
    1.8897261254578281: 'bohr per Å, codata (src/atomic_units.hpp)',
    27.211386245988: 'eV per Hartree, codata (src/atomic_units.hpp)',
    0.529177249: 'Å per bohr, vasp_legacy (src/atomic_units.hpp)',
    13.605826: 'Rydberg in eV, vasp_legacy (src/atomic_units.hpp)',
    2018: 'CODATA 2018, name of the constant set',
    300: 'cutoff floor in eV (automatic_cutoff, sparse_dft/settings.py)',
    1.15: 'cutoff margin, low', 1.30: 'cutoff margin, standard', 1.40: 'cutoff margin, high/production',
    3072: 'probe midpoints N_q (src/scf_core.cpp)',
    240: 'default --max-iterations (sparse_dft/cli.py)',
    0.2: 'default --mixing', 0.1: 'Kerker floor cap / Fermi–Dirac kT of the anchor metals',
    1.5: 'TPA scale factor (src/pw_core.cpp)', 0.15: 'default --max-displacement (Å)',
    0.02: 'default --max-strain and --force-tolerance', 0.05: 'default stress tolerance (GPa) and --dos-sigma',
    2.5e-7: 'balanced S1 budget (sparsity_rules.json)', 0.25: 'conservative level factor',
    1.7: 'aggressive level factor', 4.25e-7: 'aggressive S1 budget (sparsity_rules.json evidence)',
    0.3: 'vacuum-fraction threshold of S2.vacuum.v2', 5e-14: 'floor of the final eigen tolerance',
    0.125: '(4/16)^(3/2)', 600: 'figure resolution in dpi (benchmarks/plot_docs.py)',
    3.20: 'CMake minimum version', 3.11: 'Python minimum version', 3.4: 'Eigen version',
    1e-12: 'certificate / final eigen tolerance', 1e-10: 'L-BFGS curvature test',
    1e6: 'L-BFGS gamma clip', 1e-6: 'SCF thresholds, gamma clip, occupation test, operator floor',
    1e-8: 'relaxation density threshold, electron-count gate', 1e-4: 'S3 factor, Armijo constant, eigen tolerance cap',
    1e-5: 'S3 factor (metals), symmetry tolerance, figure floor', 1e-3: 'force gate, tail-budget cap',
    1e-2: 'S3 factor, symmetry tolerance, PDOS gate', 1e-7: 'figure floor, example budget', 1e3: 'inner-solve stop factor',
    1e-9: 'VASP EDIFF of the anchor runs', 5e-3: 'eigenvalue gate (eV)', 5e-4: 'projection gate',
    100: 'largest merge window tried (meV)', 177: 'k points of the GaAs PROCAR comparison',
}

MATH_TIMES = re.compile(r'(\d+(?:\.\d+)?)\s*\\times\s*10\^\{?([-−]?\d+)\}?')
MATH_POW10 = re.compile(r'(?<![\d.])10\^\{?([-−]?\d+)\}?')
NUMBER = re.compile(r'(?<![\w.§\\])(?<![A-Za-z]-)[-+−]?(?:\d{1,3}(?: \d{3})+|\d+)(?:\.\d+)?(?:e[-+−]?\d+)?(?![\d])')
SKIP_INLINE = [
    re.compile(r'`[^`]*`'),                                   # inline code
    re.compile(r'\]\([^)]*\)'),                               # link targets
    re.compile(r'<!--.*?-->'),                                # comments
    re.compile(r'\d{4}-\d{2}-\d{2}'),                         # dates
    re.compile(r'\d+\.\d+\.\d+'),                             # versions
    re.compile(r'(?:§|Table |Figure |Fig\. |表 |图 |stage |阶段 )\s*\d+(?:\.\d+)?'),
    re.compile(r'\[\d+(?:,\s*\d+)*\]'),                       # citation numbers
    re.compile(r'[_^]\{[^}]*\}|[_^]-?\d'),                    # sub- and superscripts
    re.compile(r'\\[td]?frac\{?\d+\}?\{?\d+\}?'),
    re.compile(r'(?<![\d.])1/(?=\d)'),                        # "1/x" written for "x times fewer"
]


def strip_blocks(text: str) -> list[str]:
    """Lines with code blocks and display equations blanked (line numbers kept)."""
    out, fence, display = [], False, False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith('```'):
            fence = not fence
            out.append('')
            continue
        if s.startswith('$$'):
            if not (s.endswith('$$') and len(s) > 2):
                display = not display
            out.append('')
            continue
        out.append('' if fence or display else line)
    return out


def normalise(line: str) -> str:
    line = MATH_TIMES.sub(lambda m: f'{m.group(1)}e{m.group(2)}', line)
    line = MATH_POW10.sub(lambda m: f'1e{m.group(1)}', line)
    for rx in SKIP_INLINE:
        line = rx.sub(' ', line)
    return re.sub(r'\\[A-Za-z]+', ' ', line)            # \le, \mathrm, ... after the patterns above


def numbers(line: str) -> list[str]:
    return [m.group(0) for m in NUMBER.finditer(normalise(line))]


def value(token: str) -> tuple[float, float]:
    """Value and half-unit of the last printed digit."""
    t = token.replace('−', '-').replace(' ', '').lstrip('+')
    mant, _, exp = t.partition('e')
    decimals = len(mant.split('.')[1]) if '.' in mant else 0
    scale = 10.0 ** int(exp) if exp else 1.0
    return abs(float(mant)) * scale, 0.5 * 10.0 ** (-decimals) * scale


# ------------------------------------------------------------------ value pool
def walk(obj, out: list[float]) -> None:
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, (int, float)):
        if math.isfinite(obj):
            out.append(abs(float(obj)))
    elif isinstance(obj, str):
        for m in re.finditer(r'[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?', obj):
            out.append(abs(float(m.group(0))))
    elif isinstance(obj, dict):
        for v in obj.values():
            walk(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            walk(v, out)


def derived(data: dict) -> dict[float, str]:
    """Quantities the documents compute from the data, with the formula used."""
    out: dict[float, str] = {}
    for r in data['codata_bias']['rows']:
        out[r['hartree_ratio_estimate_mev_per_atom'] / r['bias_mev_per_atom'] * 100] = \
            f"codata: Hartree-ratio estimate / measured bias, {r['system']} (%)"
    for r in data['bands_vasp']['rows']:
        g_s, g_v = r.get('gap_sparse_dft_ev'), r.get('gap_vasp_ev')
        if g_s is not None and g_v is not None:
            out[abs(g_s - g_v) * 1e3] = f"bands: |gap difference| {r['system']} (meV)"
    sweep = data['sparsity_sweep']
    for r in sweep['rows']:
        ref = (sweep['systems'].get(r['system']) or {}).get('reference_counters') or {}
        for key in ('operator_hamiltonian_calls', 'operator_overlap_calls', 'paw_block_projectors_calls'):
            if r.get(key) and ref.get(key):
                out[ref[key] / r[key]] = f"sweep: off/{r['variant']} {key} ratio, {r['system']}"
    mpi = {f"{r['ranks']}x{r['threads']}": r['wall_seconds']
           for r in data['performance']['runs']['mpi'] if r['input'] == 'Ni'}
    for k, v in mpi.items():
        out[abs(v / mpi['1x8'] - 1) * 100] = f'performance: Ni MPI {k} against 1x8 (%)'
    return out


# table first cells (en and zh) -> system names used in benchmarks/data
SYSTEM_NAMES = {
    'Si$_2$': 'Si2', 'Si$_8$': 'Si8', 'Si$_{64}$': 'Si64', 'MgO (O_h)': 'MgO_h', 'Ni$_4$ (Ni_pv)': 'Ni4pv',
    'Al(001) slab': 'Al_slab', 'Al(001) 表面板': 'Al_slab', 'MoS$_2$': 'MoS2', 'Si ionic': 'Si_ionic',
    'Si 离子': 'Si_ionic', 'MgO cell': 'MgO_cell', 'MgO 晶胞': 'MgO_cell',
    **{n: n for n in ('Si', 'GaAs', 'MgO', 'LiF', 'Al', 'Ni', 'Fe', 'CO')},
}


def system_pools(data: dict) -> dict[str, list[float]]:
    """Values recorded for each system, from every record that names it."""
    pools: dict[str, list[float]] = {}

    def visit(obj):
        if isinstance(obj, dict):
            for key in ('system', 'case', 'input'):
                name = obj.get(key)
                if isinstance(name, str):
                    walk(obj, pools.setdefault(name, []))
            for k, v in obj.items():
                if k in SYSTEM_NAMES.values() and isinstance(v, dict):
                    walk(v, pools.setdefault(k, []))
                visit(v)
        elif isinstance(obj, list):
            for v in obj:
                visit(v)
    for d in data.values():
        visit(d)
    return {k: sorted(set(v)) for k, v in pools.items()}


def check_rows(root: Path, pools: dict, extra: dict) -> dict:
    """Table rows whose first cell names a system: numbers must come from that system's records."""
    report = {'judged': 0, 'untraced': []}
    for page in [*PAGES, *TWINS.values()]:
        path = root / page
        if not path.exists():
            continue
        for n, line in enumerate(doc_lines(path), 1):
            if not line.startswith('|'):
                continue
            cells = [c.strip() for c in line.strip().strip('|').split('|')]
            name = SYSTEM_NAMES.get(cells[0])
            if name is None or name not in pools:
                continue
            for tok in numbers('|'.join(cells[1:])):
                v, half = value(tok)
                if float(v).is_integer() and v < 100 and 'e' not in tok:
                    continue
                report['judged'] += 1
                if traced(v, half, pools[name], extra) is None:
                    report['untraced'].append(f'{page}:{n}: {cells[0]} {tok}')
    return report


def chance_rate(pool: list[float], extra: dict, digits: int, trials: int = 2000) -> float:
    """Share of random numbers with this many significant digits that the pool would accept."""
    import random
    rng = random.Random(1)
    hits = 0
    for _ in range(trials):
        v, half = value(f'{rng.uniform(1, 10):.{digits - 1}f}e{rng.randint(-12, 3)}')
        hits += traced(v, half, pool, extra) is not None
    return hits / trials


def load_pool(root: Path) -> tuple[list[float], dict[float, str], dict]:
    data, pool = {}, []
    for path in sorted((root / 'benchmarks' / 'data').glob('*.json')):
        data[path.stem] = json.loads(path.read_text(encoding='utf-8'))
        walk(data[path.stem], pool)
    for path in sorted((root / 'benchmarks' / 'data').glob('*.csv')):
        with path.open(encoding='utf-8', newline='') as f:
            for row in csv.reader(f):
                walk([c for c in row], pool)
    prov = root / 'docs' / 'assets' / 'figures.provenance.json'
    if prov.exists():
        walk(json.loads(prov.read_text(encoding='utf-8')).get('figures'), pool)
    # atlas values and GUI screenshot geometry are recorded next to their images
    for rec in (root / 'docs' / 'assets' / 'electronic' / 'atlas.provenance.json',
                root / 'docs' / 'assets' / 'gui' / 'screenshots.json'):
        if rec.exists():
            walk(json.loads(rec.read_text(encoding='utf-8')), pool)
    return sorted(set(pool)), derived(data), data


def traced(v: float, half: float, pool: list[float], extra: dict[float, str]) -> str | None:
    tol = half * (1 + 1e-9) + 1e-300
    for scale in SCALES:
        target = v / scale
        t = tol / scale
        i = bisect.bisect_left(pool, target - t)
        if i < len(pool) and pool[i] <= target + t:
            return 'data' if scale == 1.0 else f'data x{scale:g}'
    for c, why in {**CONSTANTS, **extra}.items():
        if abs(abs(c) - v) <= tol:
            return why
    return None


def doc_lines(path: Path) -> list[str]:
    """Prose and table lines to check; the license block (written by the legal part of the build) is blanked."""
    text = BLOCK.sub(lambda m: '\n' * m.group(0).count('\n'), path.read_text(encoding='utf-8'))
    lines, refs = strip_blocks(text), False
    out = []
    for line in lines:
        s = line.strip()
        if s.startswith('#'):
            refs = s.lstrip('#').strip() in ('References', '参考文献')
            out.append('')
        elif refs or s.startswith('|---') or s.startswith('**English**') or s.startswith('[English]'):
            out.append('')
        else:
            out.append(line)
    return out


def check_numbers(root: Path, pool, extra) -> dict:
    report = {'judged': 0, 'small_integers': 0, 'untraced': []}
    for page in PAGES:
        path = root / page
        if not path.exists():
            continue
        for n, line in enumerate(doc_lines(path), 1):
            for tok in numbers(line):
                v, half = value(tok)
                if float(v).is_integer() and v < 100 and 'e' not in tok:
                    report['small_integers'] += 1
                    continue
                report['judged'] += 1
                if traced(v, half, pool, extra) is None:
                    report['untraced'].append(f'{page}:{n}: {tok}')
    return report


def check_parity(root: Path) -> list[str]:
    issues = []
    for en, zh in TWINS.items():
        pe, pz = root / en, root / zh
        if not (pe.exists() and pz.exists()):
            issues.append(f'missing twin: {en} / {zh}')
            continue
        le, lz = doc_lines(pe), doc_lines(pz)
        if len(le) != len(lz):
            issues.append(f'{zh}: {len(lz)} lines against {len(le)} in {en}; block alignment lost')
        for i, (a, b) in enumerate(zip(le, lz), 1):
            na, nb = sorted(numbers(a)), sorted(numbers(b))
            if na != nb:
                issues.append(f'{zh}:{i}: en {na} / zh {nb}')
    return issues


# ------------------------------------------------------------------ links and figures
LINK = re.compile(r'(!?)\[([^\]]*)\]\(([^)\s]+)\)')


def slug(heading: str) -> str:
    """GitHub-style anchor of a heading."""
    import unicodedata
    text = heading.strip().lower()
    kept = ''.join(c for c in text if c in ' -_' or unicodedata.category(c)[0] in 'LNM')
    return kept.replace(' ', '-')


def anchors(path: Path) -> set[str]:
    seen: dict[str, int] = {}
    out = set()
    for line in strip_blocks(path.read_text(encoding='utf-8')):
        m = re.match(r'#{1,6}\s+(.*)', line)
        if m:
            s = slug(m.group(1))
            k = seen.get(s, 0)
            out.add(s if k == 0 else f'{s}-{k}')
            seen[s] = k + 1
    return out


def resolve(target: str, page: Path, roots: list[Path]) -> Path | None:
    for root in roots:
        cand = (root / page.parent.relative_to(roots[0]) / target).resolve() if page.is_relative_to(roots[0]) \
            else (page.parent / target).resolve()
        if cand.exists():
            return cand
    return None


def check_links(root: Path, extra_roots: list[Path]) -> tuple[list[str], list[str]]:
    issues, elsewhere = [], []
    roots = [root, *extra_roots]
    pages = [root / p for p in PAGES] + [root / z for z in TWINS.values()]
    for page in pages:
        if not page.exists():
            continue
        for n, line in enumerate(page.read_text(encoding='utf-8').splitlines(), 1):
            for m in LINK.finditer(line):
                is_img, target = m.group(1), m.group(3)
                if re.match(r'[a-z]+://', target) or target.startswith('mailto:'):
                    continue
                file_part, _, frag = target.partition('#')
                where = f'{page.relative_to(root)}:{n}'
                dest = page if not file_part else resolve(file_part, page, roots)
                if dest is None:
                    rel = (page.parent / file_part).resolve()
                    try:
                        rel_s = str(rel.relative_to(root.resolve()))
                    except ValueError:
                        rel_s = file_part
                    if any(rel_s == e or rel_s.startswith(e) for e in EXPECTED_ELSEWHERE):
                        elsewhere.append(f'{where}: {target}')
                    else:
                        issues.append(f'{where}: missing {target}')
                    continue
                if frag and dest.suffix == '.md' and frag not in anchors(dest):
                    issues.append(f'{where}: no anchor #{frag} in {dest.name}')
                if is_img and page.name.endswith('.zh-CN.md') and not dest.name.endswith('.zh-CN.png'):
                    zh = dest.with_name(dest.stem + '.zh-CN' + dest.suffix)
                    if zh.exists():
                        issues.append(f'{where}: Chinese page uses {dest.name}; {zh.name} exists')
                if is_img and not page.name.endswith('.zh-CN.md') and '.zh-CN.' in dest.name:
                    issues.append(f'{where}: English page uses {dest.name}')
    return issues, elsewhere


def check_attribution(root: Path, forbidden: tuple[str, ...] = FORBIDDEN) -> list[str]:
    issues = []
    for page in [*PAGES, *TWINS.values()]:
        path = root / page
        if not path.exists():
            continue
        text = path.read_text(encoding='utf-8')
        for word in forbidden:
            if word.lower() in text.lower():
                issues.append(f'{page}: contains "{word}"')
        outside = BLOCK.sub('', text)
        if AUTHOR in outside:
            issues.append(f'{page}: author name outside the license block')
        if page in ('README.md', 'README.zh-CN.md') and len(BLOCK.findall(text)) != 1:
            issues.append(f'{page}: expected exactly one license block')
    return issues


# ------------------------------------------------------------------ headline claims
def _csv(root: Path, name: str) -> list[dict]:
    with (root / 'benchmarks' / 'data' / name).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def _f(x) -> float:
    return abs(float(x)) if x not in (None, '') else float('nan')


def claims(root: Path, data: dict) -> list[tuple[str, str, float]]:
    """(pages, printed token, value recomputed from data). Pages: en and zh twins of the same statement."""
    scf = _csv(root, 'anchor_vasp_scf.csv')
    matched = [r for r in scf if r['tier'] == 'matched']
    sw = data['sparsity_sweep']
    fitted = {r['system']: r for r in sw['fitted_auto']}
    auto_rows = {r['system']: r for r in sw['rows'] if r['variant'] == 'auto'}
    held = {**{k: v for k, v in auto_rows.items() if sw['systems'][k]['role'] == 'holdout'}, **fitted}

    def ratio(r, key):
        ref = sw['systems'][r['system']]['reference_counters'][key]
        return ref / r[key]
    h_ratios = [r.get('operator_hamiltonian_ratio') or ratio(r, 'operator_hamiltonian_calls') for r in held.values()]
    speed = [r['speedup'] for r in held.values()]
    perf = data['performance']['runs']
    si8 = {(r['label'], r['threads']): r for r in perf['scale_Si8']}
    rep = {r['run']: r for r in data['performance']['replay']['runs']}
    bands = {r['system']: r for r in data['bands_vasp']['rows']}
    dos_tot = {(r['system'], r['channel']): _f(r['l1_normalised']) for r in data['dos_vasp']['rows'] if r['column'] == 'total'}
    pdos = [_f(r['l1_normalised']) for r in data['dos_vasp']['rows']
            if r['column'] != 'total' and _f(r['integral_vasp']) >= 0.1]
    pdos_all = [_f(r['l1_normalised']) for r in data['dos_vasp']['rows'] if r['column'] != 'total']
    cb = data['codata_bias']
    class_e = [r for r in sw['rows'] if r['variant'].split('_')[0] in ('S3', 'S4', 'S5') or r['variant'] == 'E_bundle']
    readme, spars, acc, perfp = ('README.md',), ('docs/SPARSIFICATION.md',), ('docs/ACCURACY.md',), ('docs/PERFORMANCE.md',)
    return [
        (readme + acc, '1.05', max(_f(r['energy_ev_per_atom']) for r in matched) * 1e4),
        (readme + acc, '5.3', max(_f(r['force_max_ev_per_angstrom']) for r in matched) * 1e4),
        (readme + acc, '5.5', max(_f(r['stress_max_gpa']) for r in matched) * 1e3),
        (readme, '1.9 meV', max(_f(r['bands_max_ev']) for r in matched) * 1e3),
        (readme, '1.5', max(_f(r['dos_l1']) for r in matched) * 1e4),
        (acc, '1.2', max(_f(r['electron_relative']) for r in scf) * 1e12),
        (readme, '2.1', min(h_ratios)), (readme, '3.9', max(h_ratios)),
        (readme, '1.54', min(speed)), (readme, '8.13', max(speed)),
        (readme, '1778', sw['systems']['Si64']['reference_seconds']), (readme, '522', auto_rows['Si64']['seconds']),
        (spars, '1777.7', sw['systems']['Si64']['reference_seconds']), (spars, '521.7', auto_rows['Si64']['seconds']),
        (readme + perfp, '48.3', si8[('v2', 8)]['wall_seconds']), (readme + perfp, '23.5', si8[('w1b', 8)]['wall_seconds']),
        (readme + perfp, '2.8', max(_f(r['dE_ev']) for r in perf['scale_Si8']) * 1e12),
        (readme + perfp, '323.5', rep['rep_cpu']['solver_seconds']),
        (readme + perfp, '49.2', min(r['solver_seconds'] for r in rep.values() if r['backend'] == 'cpu')),
        (readme, '0.7', min(r['bias_mev_per_atom'] for r in cb['rows'])),
        (readme, '21', max(r['bias_mev_per_atom'] for r in cb['rows'])),
        (acc, '0.0207', cb['fit_bias_vs_atomic_reference']['slope']),
        (acc, '0.990', cb['fit_bias_vs_atomic_reference']['r2']),
        (acc, '49', min(r['hartree_ratio_estimate_mev_per_atom'] / r['bias_mev_per_atom'] for r in cb['rows']) * 100),
        (acc, '90', max(r['hartree_ratio_estimate_mev_per_atom'] / r['bias_mev_per_atom'] for r in cb['rows']) * 100),
        (acc, '0.46', bands['GaAs']['mae_ev'] * 1e3), (acc, '1.00', bands['GaAs']['max_abs_error_ev'] * 1e3),
        (acc, '885', bands['GaAs']['samples']), (acc, '1803', bands['Ni']['samples']), (acc, '963', bands['MoS2']['samples']),
        (acc, '0.027', bands['Ni']['mae_ev'] * 1e3), (acc, '0.068', bands['Ni']['max_abs_error_ev'] * 1e3),
        (acc, '0.077', bands['MoS2']['mae_ev'] * 1e3), (acc, '0.157', bands['MoS2']['max_abs_error_ev'] * 1e3),
        (acc, '0.4628', bands['GaAs']['gap_sparse_dft_ev']), (acc, '0.4626', bands['GaAs']['gap_vasp_ev']),
        (acc, '4.5', dos_tot[('Si', 'total')] * 1e5), (acc, '4.0', dos_tot[('GaAs', 'total')] * 1e5),
        (acc, '1.7', max(pdos) * 1e4), (acc, '7.5', max(pdos_all) * 1e3),
        (spars, '61', len(class_e)),
        (spars, '2.1', max(_f(r['energy_mev_per_atom']) for r in class_e) * 1e8),
        (spars, '1.4', max(_f(r['force_ev_per_angstrom']) for r in class_e) * 1e6),
        (spars, '2.6', max(_f(r['stress_gpa']) for r in class_e) * 1e6),
        (spars, '0.60', sw['holdout_failures'][0]['vacuum_fraction']),
        (spars, '1.98', sw['holdout_failures'][0]['force_ev_per_angstrom'] * 1e3),
        (spars, '133 219', auto_rows['Si64']['density_modes']), (spars, '1 066 147', sw['systems']['Si64']['reference_modes']),
        (spars, '250 047', auto_rows['Si64']['xc_points']), (spars, '16 387 064', sw['systems']['Si64']['reference_xc_points']),
    ]


def check_claims(root: Path, data: dict) -> tuple[int, list[str]]:
    issues, n = [], 0
    for pages, token, computed in claims(root, data):
        v, half = value(token.replace(' meV', ''))
        n += 1
        if not abs(v - computed) <= half * (1 + 1e-9):
            issues.append(f'{token}: data gives {computed:.6g}')
        for page in pages:
            for p in (page, TWINS.get(page)):
                if p and token not in (root / p).read_text(encoding='utf-8').replace('\\times10^', 'e'):
                    issues.append(f'{p}: printed token "{token}" not found')
    return n, issues


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--root', type=Path, default=HERE.parent.parent)
    ap.add_argument('--release-root', type=Path, action='append', default=[],
                    help='further trees in which link targets may exist (e.g. the assembled release)')
    ap.add_argument('--json', type=Path)
    ap.add_argument('--forbid', action='append', default=[], metavar='NAME',
                    help='a name that must not appear in the documentation (repeatable)')
    a = ap.parse_args(argv)
    root = a.root.resolve()
    pool, extra, data = load_pool(root)
    nums = check_numbers(root, pool, extra)
    pools = system_pools(data)
    rows = check_rows(root, pools, extra)
    sizes = sorted(len(v) for v in pools.values())
    median_pool = pools[min(pools, key=lambda k: abs(len(pools[k]) - sizes[len(sizes) // 2]))]
    chance = {'global_2_digits': chance_rate(pool, extra, 2), 'global_3_digits': chance_rate(pool, extra, 3),
              'row_2_digits_median_system': chance_rate(median_pool, extra, 2),
              'row_3_digits_median_system': chance_rate(median_pool, extra, 3)}
    parity = check_parity(root)
    links, elsewhere = check_links(root, [r.resolve() for r in a.release_root])
    attribution = check_attribution(root, tuple(a.forbid))
    n_claims, claim_issues = check_claims(root, data)
    report = {'numbers': nums, 'table_rows': rows, 'claims': {'checked': n_claims, 'issues': claim_issues}, 'chance_match_rate': chance, 'parity': parity, 'links': links, 'provided_elsewhere': elsewhere,
              'attribution': attribution}
    if a.json:
        a.json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f"numbers: {nums['judged']} judged, {len(nums['untraced'])} untraced, "
          f"{nums['small_integers']} small integers not judged")
    for item in nums['untraced']:
        print('  untraced', item)
    print(f"table rows: {rows['judged']} numbers checked against their own system, {len(rows['untraced'])} untraced")
    for item in rows['untraced']:
        print('  untraced', item)
    print(f'claims: {n_claims} headline numbers recomputed from data, {len(claim_issues)} issue(s)')
    for item in claim_issues:
        print('  ', item)
    print('chance match rate: ' + ', '.join(f'{k} {v:.0%}' for k, v in chance.items()))
    for name, items in (('parity', parity), ('links', links), ('attribution', attribution)):
        print(f'{name}: {len(items)} issue(s)')
        for item in items:
            print('  ', item)
    print(f'provided elsewhere: {len(elsewhere)} link(s) to files from other parts of the release')
    ok = not (nums['untraced'] or rows['untraced'] or claim_issues or parity or links or attribution)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
