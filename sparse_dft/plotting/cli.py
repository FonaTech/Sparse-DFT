"""``sparse-dft plot`` command line.

    sparse-dft plot list SOURCES...                     kinds available for the sources (JSON)
    sparse-dft plot auto SOURCES... --out DIR           every available kind into DIR
    sparse-dft plot KIND SOURCES... --out STEM          one figure (STEM.pdf, STEM.png, ...)
    sparse-dft plot band_suite SOURCES... --out DIR     bands, spin, element/orbital fat bands, PDOS, bands+PDOS,
                                                  and Sparse-DFT-vs-reference comparisons with --reference

Success prints a JSON summary on stdout and returns 0; errors print
``{"code", "message", "details"}`` on stderr and return 2 (like ``sparse_dft``).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from ..errors import InputError
from .figures import KINDS, MULTI_PANEL, available_kinds, make_figure, make_figures

LANG_CHOICES = ('en', 'zh_CN', 'zh_TW', 'ja')


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog='sparse-dft plot', description='Publication figures for Sparse-DFT outputs.')
    p.add_argument('kind', help="figure kind, 'auto' (all available), 'band_suite' (band toolkit) or 'list'; kinds: "
                   + ', '.join(KINDS))
    p.add_argument('sources', nargs='+', help='calculation directories or files')
    p.add_argument('--out', help="output stem (one kind) or directory ('auto'); default: sparse_dft_<kind> / sparse_dft_figures")
    p.add_argument('--formats', default='pdf,png', help='comma list of pdf,svg,png,tiff,eps (default pdf,png)')
    p.add_argument('--width', default='single', help='single (89 mm), onehalf (120 mm), double (183 mm) or mm')
    p.add_argument('--height', type=float, help='figure height in mm (default: chosen per kind)')
    p.add_argument('--lang', default='en', choices=LANG_CHOICES)
    p.add_argument('--palette', default='academic', choices=('academic', 'okabe_ito', 'macaron'))
    p.add_argument('--dpi', type=int, default=600, help='raster resolution (default 600)')
    p.add_argument('--panels', choices=('combined', 'split', 'both'), default='combined',
                   help='multi-panel kinds (electronic_atlas): one combined figure, one file per panel '
                        '(STEM_<panel>), or both')
    p.add_argument('--fontsize', type=float, default=7.0, help='base font size in pt (5-9, default 7)')
    g = p.add_argument_group('data options')
    g.add_argument('--emin', type=float, help='lower energy limit relative to the energy zero (eV)')
    g.add_argument('--emax', type=float, help='upper energy limit relative to the energy zero (eV)')
    g.add_argument('--fermi', type=float, help='Fermi energy (eV) when the data lacks one or to override it')
    g.add_argument('--align', choices=('fermi', 'vbm', 'none'), help='energy zero (default fermi)')
    g.add_argument('--sigma', type=float, help='Gaussian broadening for DOS from SCF eigenvalues (eV, default 0.1)')
    g.add_argument('--spin', choices=('up', 'down', 'both'), help='spin channel to show (default both)')
    g.add_argument('--reference', help='reference band source for bands_compare / dos_compare (SOURCES is the candidate)')
    g.add_argument('--dos-reference', help='band_suite: uniform-mesh vasprun.xml or DOS for dos_compare (default --reference)')
    g.add_argument('--structure', help='structure file for the reciprocal metric of a band path')
    g.add_argument('--top', type=int, help='timing: number of timers (default 15)')
    g.add_argument('--metric', help='sparsity_pareto work metric (default auto)')
    g.add_argument('--systems', help='sparsity_pareto: comma list of systems')
    g.add_argument('--group', choices=('l', 'lm', 'orbital', 'element', 'element+l', 'element+lm', 'atom', 'atom+l', 'atom+lm'),
                   help='fatbands/pdos/bands_pdos grouping (fatbands default l, pdos element+l, bands_pdos element or l)')
    g.add_argument('--mode', choices=('width', 'scatter', 'dominant', 'rgb'), help='fatbands drawing mode (default width)')
    g.add_argument('--atoms', help="atom subset: 1-based indices, ranges or elements, e.g. '1-4,7' or 'Ga'")
    g.add_argument('--orbitals', help="orbital subset: lm labels or l channels, e.g. 'd' or 'dxy,dyz,dxz'")
    g.add_argument('--channels', help="projection channel for fatbands (up, down, total, mx, my, mz; default all)")
    g.add_argument('--resolution', choices=('auto', 'lm', 'l'), help='projection resolution to load (default auto)')
    g.add_argument('--layout', choices=('side', 'overlay'), help='bands_spin layout (default side)')
    g.add_argument('--integrated', action='store_true', help='pdos: running integral (states) instead of DOS')
    g.add_argument('--stacked', action='store_true', help='pdos: stacked filled areas')
    g.add_argument('--bz-inset', action='store_true', help='bands/fatbands/bands_pdos: Brillouin-zone path inset (needs scipy)')
    g.add_argument('--gap-arrow', action='store_true', help='bands/fatbands: shaded gap and VBM->CBM arrow (bands_pdos always)')
    g.add_argument('--fermi-bands', action='store_true', help='bands/bands_pdos: highlight bands crossing the energy zero')
    g.add_argument('--tolerance', type=float, help='relax/forces tolerance line override')
    g.add_argument('--per-atom', action='store_true', help='energy_terms per atom')
    g.add_argument('--log', action='store_true', help='timing: logarithmic time axis')
    return p


def _options(a) -> dict:
    opts = {'emin': a.emin, 'emax': a.emax, 'fermi': a.fermi, 'align': a.align, 'sigma': a.sigma,
            'spin': a.spin, 'reference': a.reference, 'structure': a.structure, 'top': a.top,
            'metric': a.metric, 'group': a.group, 'tolerance': a.tolerance, 'mode': a.mode, 'atoms': a.atoms,
            'orbitals': a.orbitals, 'channel': a.channels, 'resolution': a.resolution, 'layout': a.layout}
    for flag in ('integrated', 'stacked', 'bz_inset', 'gap_arrow', 'fermi_bands'):
        if getattr(a, flag):
            opts[flag] = True
    if a.systems:
        opts['systems'] = [s.strip() for s in a.systems.split(',') if s.strip()]
    if a.per_atom:
        opts['per_atom'] = True
    if a.log:
        opts['log'] = True
    return {k: v for k, v in opts.items() if v is not None}


def _width(text):
    try:
        return float(text)
    except ValueError:
        return text


# Options a kind actually uses; others are ignored in 'auto' so one flag
# (e.g. --emin) does not break unrelated kinds.
KIND_OPTIONS = {
    'scf_convergence': set(), 'magnetization': set(), 'energy_terms': {'per_atom'}, 'forces': {'tolerance'},
    'stress': set(), 'timing': {'top', 'log'},
    'bands': {'emin', 'emax', 'fermi', 'align', 'spin', 'structure', 'bz_inset', 'gap_arrow', 'fermi_bands'},
    'bands_spin': {'emin', 'emax', 'fermi', 'align', 'structure', 'layout'},
    'bands_pdos': {'emin', 'emax', 'fermi', 'align', 'spin', 'structure', 'group', 'atoms', 'orbitals', 'bz_inset',
                   'fermi_bands'},
    'dos_compare': {'emin', 'emax', 'fermi', 'align', 'spin', 'reference'},
    'bands_dos': {'emin', 'emax', 'fermi', 'align', 'spin', 'sigma', 'structure'},
    'dos': {'emin', 'emax', 'fermi', 'align', 'spin', 'sigma'},
    'pdos': {'emin', 'emax', 'fermi', 'align', 'spin', 'sigma', 'group', 'atoms', 'orbitals', 'integrated', 'stacked'},
    'fatbands': {'emin', 'emax', 'fermi', 'align', 'group', 'structure', 'mode', 'atoms', 'orbitals', 'channel',
                 'resolution', 'bz_inset', 'gap_arrow'},
    'bands_compare': {'emin', 'emax', 'align', 'reference', 'structure'},
    'relax': {'tolerance'}, 'sparsity_pareto': {'metric', 'systems'}, 'scaling': set(),
    'electronic_atlas': {'emin', 'emax', 'reference', 'structure'},
}


def _render(kind, sources, a, opts, stem):
    from .export import export
    used = {k: v for k, v in opts.items() if k in KIND_OPTIONS.get(kind, set())}
    if kind in MULTI_PANEL:
        figs = make_figures(kind, sources, panels=a.panels, lang=a.lang, palette=a.palette,
                            base_fontsize=a.fontsize, **used)
        from .export import export_figures
        written = {name: [str(f) for f in files]
                   for name, files in export_figures(figs, stem, formats=a.formats, dpi=a.dpi).items()}
        meta = getattr(figs.get('combined') or next(iter(figs.values())), 'sparse_dft_meta', {}) or {}
        return {'kind': kind, 'panels': a.panels, 'files': written, 'values': meta.get('values', {})}
    fig = make_figure(kind, sources, width=_width(a.width), lang=a.lang, palette=a.palette,
                      base_fontsize=a.fontsize, **used)
    if a.height:
        fig.set_size_inches(fig.get_figwidth(), a.height / 25.4)
    files = export(fig, stem, formats=a.formats, dpi=a.dpi)
    return {'kind': kind, 'files': [str(f) for f in files],
            'values': (getattr(fig, 'sparse_dft_meta', {}) or {}).get('values', {})}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    try:
        a = parser.parse_args(argv)
    except SystemExit as exc:          # argparse prints its own usage message
        return 0 if exc.code == 0 else 2
    try:
        opts = _options(a)
        if a.kind == 'list':
            kinds = available_kinds(a.sources)
            print(json.dumps({'sources': a.sources, 'kinds': kinds,
                              'descriptions': {k: KINDS[k]['title'] for k in kinds}}, indent=2, ensure_ascii=False))
            return 0
        if a.kind == 'auto':
            out = Path(a.out or 'sparse_dft_figures')
            kinds = available_kinds(a.sources)
            if not kinds:
                raise InputError('no plottable outputs found in the given sources', details={'sources': a.sources})
            written, failed = [], []
            for kind in kinds:
                try:
                    written.append(_render(kind, a.sources, a, opts, out / kind))
                except InputError as exc:
                    failed.append({'kind': kind, **exc.as_dict()})
            print(json.dumps({'out': str(out), 'figures': written, 'skipped': failed}, indent=2,
                             ensure_ascii=False, default=str))
            return 0 if written else 2
        if a.kind == 'band_suite':
            from .band_toolkit import band_suite
            common = {k: v for k, v in opts.items() if k in ('emin', 'emax', 'fermi', 'align', 'structure', 'spin')}
            result = band_suite(a.sources, Path(a.out or 'sparse_dft_band_suite'), reference=a.reference,
                                dos_reference=a.dos_reference, formats=a.formats, dpi=a.dpi, width=_width(a.width),
                                lang=a.lang, palette=a.palette, base_fontsize=a.fontsize, **common)
            print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
            return 0 if result['figures'] else 2
        if a.kind not in KINDS:
            raise InputError(f'unknown figure kind {a.kind!r}',
                             details={'kinds': list(KINDS) + ['auto', 'band_suite', 'list']})
        result = _render(a.kind, a.sources, a, opts, Path(a.out or f'sparse_dft_{a.kind}'))
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return 0
    except Exception as exc:   # report every failure as machine-readable JSON, like sparse_dft.cli
        payload = exc.as_dict() if hasattr(exc, 'as_dict') else {'code': 'RUNTIME_ERROR', 'message': str(exc),
                                                                  'details': {'type': type(exc).__name__}}
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=str), file=sys.stderr)
        return 2


if __name__ == '__main__':      # pragma: no cover
    raise SystemExit(main())
