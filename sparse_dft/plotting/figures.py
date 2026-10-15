"""Figure registry: what can be plotted from a source, and how to make it."""
from __future__ import annotations

from pathlib import Path

from ..errors import InputError
from .detect import SourceInfo, scan
from .style import figure_style

KINDS: dict[str, dict] = {
    'scf_convergence': {'title': 'SCF convergence', 'inputs': ['*.iterations.jsonl', 'scf_result.json (tolerances)'],
                        'description': 'Density and one-centre residuals (log) per iteration with tolerance lines; '
                                       'eigensolver residual in a separate panel.'},
    'magnetization': {'title': 'Magnetic moment', 'inputs': ['*.iterations.jsonl (spin-polarised run)'],
                      'description': 'Output and mixed moment per SCF iteration and |Δm| between iterations.'},
    'energy_terms': {'title': 'Energy decomposition', 'inputs': ['scf_result.json'],
                     'description': 'Native total-energy ledger: waterfall when the terms are verified to sum to '
                                    'total_energy_ev within 1e-6 eV, else signed bars; diagnostic flag shown.'},
    'forces': {'title': 'Atomic forces', 'inputs': ['scf_result.json (hf_total_forces_ev_per_angstrom)'],
               'description': 'Per-atom |F| bars and Cartesian component heatmap (eV/Å).'},
    'stress': {'title': 'Stress tensor', 'inputs': ['scf_result.json (hf_total_stress_ev_per_angstrom3)'],
               'description': '3×3 stress heatmap in GPa on a diverging scale centred at zero.'},
    'timing': {'title': 'Timers', 'inputs': ['*.performance.json or scf_result.json sparsity_ledger'],
               'description': 'Top-N timers (seconds and calls); nested timers, not additive.'},
    'bands': {'title': 'Band structure', 'inputs': ['bands.json / bands_result.json / vasprun.xml / EIGENVAL (path)'],
              'description': 'Bands vs k-path, broken segments not joined, Γ labels, E − E_F, VBM/CBM and gap.'},
    'bands_dos': {'title': 'Bands and DOS', 'inputs': ['band path', 'DOSCAR / vasprun.xml <dos> / converged scf_result.json'],
                  'description': 'Band structure and total DOS sharing the energy axis.'},
    'dos': {'title': 'Density of states', 'inputs': ['DOSCAR / vasprun.xml <dos> / converged scf_result.json'],
            'description': 'Total DOS; spin-down mirrored to negative values.'},
    'pdos': {'title': 'Projected DOS', 'inputs': ['dos.json (sparse-dft dos) / DOSCAR with projected blocks / vasprun.xml <partial>'],
             'description': 'PDOS by element+l (default), l, lm, element+lm, atom+l ...; atom/orbital selection, '
                            'running integral and stacked areas; spin-down mirrored.'},
    'fatbands': {'title': 'Fat bands', 'inputs': ['band path', 'PROCAR / vasprun.xml <projected> / native lm weights'],
                 'description': 'Projected bands by l, lm, element, element+l/lm or atom subsets as width, scatter, '
                                'dominant-group colour or RGB blend.'},
    'bands_spin': {'title': 'Spin band structure', 'inputs': ['spin-polarised band path'],
                   'description': 'Majority and minority bands side by side (or overlaid) on one energy axis.'},
    'bands_pdos': {'title': 'Bands and PDOS', 'inputs': ['band path', 'projected DOS (dos.json / DOSCAR / vasprun.xml)'],
                   'description': 'Band structure with a stacked PDOS side panel, gap arrow and optional '
                                  'Brillouin-zone inset.'},
    'dos_compare': {'title': 'DOS comparison', 'inputs': ['dos.json (candidate)', 'reference DOS or VASP uniform-mesh vasprun.xml'],
                    'description': 'Reference vs candidate DOS on one Gaussian grid with a difference panel and '
                                   'normalised L1; VASP references are rebuilt from eigenvalues and lm weights.'},
    'bands_compare': {'title': 'Band comparison', 'inputs': ['two band sources (paths, or two SCF k-meshes)'],
                      'description': 'Reference/candidate overlay and ΔE vs energy panel.'},
    'electronic_atlas': {'title': 'Electronic-structure atlas',
                         'inputs': ['band path with lm projections', 'optional: reference band path (--reference), '
                                    'uniform k mesh (sparse-dft dos result or SCF k mesh), dos.json'],
                         'description': 'Seven panels: orbital-resolved bands with reference markers, PDOS, band-edge '
                                        'curvature and m*, k3 = 0 transition/Fermi map, per-band composition, '
                                        'engine residuals and projection parity; panels without inputs are '
                                        'omitted and the reason recorded. Split or combined export.'},
    'relax': {'title': 'Relaxation', 'inputs': ['relax_result.json / relax_checkpoint.json / cell_relax_checkpoint.json'],
              'description': 'Energy, max force (or stress) and step length per step with tolerance line.'},
    'sparsity_pareto': {'title': 'Sparsity Pareto', 'inputs': ['report_sparsity.py output or sweep summary.json'],
                        'description': 'Gate use (max over Table-1 gates, 1 = limit) vs counter work reduction.'},
    'scaling': {'title': 'Parallel scaling', 'inputs': ['sparse-dft-scaling-1 JSON (one or more)'],
                'description': 'Wall time, speed-up, efficiency and occupancy vs threads/ranks with ideal lines.'},
}


class _Pool:
    def __init__(self, sources):
        if isinstance(sources, (str, Path)):
            sources = [sources]
        self.sources = [Path(s) for s in sources]
        missing = [str(s) for s in self.sources if not s.exists()]
        if missing:
            raise InputError(f'source not found: {", ".join(missing)}')
        self.items: list[SourceInfo] = [info for s in self.sources for info in scan(s)]

    def all(self, role, pred=None):
        return [i for i in self.items if role in i.roles and (pred is None or pred(i))]

    def first(self, role, pred=None):
        found = self.all(role, pred)
        return found[0] if found else None


def _path_bands(pool, *, minimum=2):
    return pool.all('bands_file', lambda i: i.attrs.get('nkpoints', 0) >= minimum)


def _need(item, what, kind):
    if item is None:
        raise InputError(f'{kind}: no {what} found in the given sources')
    return item


def _resolve(kind, pool):
    """Return the inputs a kind needs, or raise InputError."""
    if kind not in KINDS:
        raise InputError(f'unknown figure kind {kind!r}; choose one of {sorted(KINDS)}')
    if kind == 'scf_convergence':
        return {'iterations': _need(pool.first('iterations'), 'SCF *.iterations.jsonl', kind).path}
    if kind == 'magnetization':
        return {'iterations': _need(pool.first('iterations', lambda i: i.attrs.get('spin')),
                                    'spin-polarised SCF history', kind).path}
    if kind in ('energy_terms', 'forces', 'stress'):
        attr = {'energy_terms': 'ledger', 'forces': 'forces', 'stress': 'stress'}[kind]
        return {'scf': _need(pool.first('scf_result', lambda i: i.attrs.get(attr)), f'SCF result with {attr}', kind).path}
    if kind == 'timing':
        perf = pool.first('performance')
        if perf:
            return {'timing': perf.path}
        return {'timing': _need(pool.first('scf_result', lambda i: i.attrs.get('counters')),
                                'performance counters', kind).path}
    if kind == 'bands':
        found = _path_bands(pool)
        return {'bands': _need(found[0] if found else None, 'band path (≥2 k-points)', kind).path}
    if kind == 'dos':
        return {'dos': _need(pool.first('dos'), 'DOS source', kind).path}
    if kind == 'pdos':
        return {'dos': _need(pool.first('pdos'), 'projected DOS', kind).path}
    if kind == 'bands_dos':
        found = _path_bands(pool)
        bands = _need(found[0] if found else None, 'band path (≥2 k-points)', kind)
        dos = pool.first('dos', lambda i: i.path != bands.path or 'bands_file' not in i.roles) or pool.first('dos')
        return {'bands': bands.path, 'dos': _need(dos, 'DOS source', kind).path}
    if kind == 'fatbands':
        found = _path_bands(pool)
        bands = _need(found[0] if found else None, 'band path (≥2 k-points)', kind)
        procar = pool.first('procar')
        if procar is None and not bands.attrs.get('projections'):
            raise InputError(f'{kind}: no PROCAR or native projector weights found')
        return {'bands': bands.path, 'procar': procar.path if procar else None}
    if kind == 'bands_spin':
        found = [i for i in _path_bands(pool) if i.attrs.get('spin')]
        return {'bands': _need(found[0] if found else None, 'spin-polarised band path', kind).path}
    if kind == 'bands_pdos':
        found = _path_bands(pool)
        bands = _need(found[0] if found else None, 'band path (≥2 k-points)', kind)
        return {'bands': bands.path, 'dos': _need(pool.first('pdos'), 'projected DOS', kind).path}
    if kind == 'dos_compare':
        native = pool.all('dos', lambda i: i.path.name == 'dos.json')
        others = [i for i in pool.all('dos') if i.path.name != 'dos.json'] + native[1:]
        if not native or not others:
            raise InputError(f'{kind}: needs a sparse-dft dos result and a reference DOS (--reference)')
        return {'reference': others[0].path, 'candidate': native[0].path}
    if kind == 'bands_compare':
        found = _distinct_runs([i.path for i in _path_bands(pool)])
        if len(found) < 2:
            found = [i.path for i in pool.all('scf_result', lambda i: i.attrs.get('kpoints'))]
            found = _distinct_runs(found)
        if len(found) < 2:
            raise InputError(f'{kind}: needs two band sources (reference first, candidate second)')
        return {'reference': found[0], 'candidate': found[1]}
    if kind == 'electronic_atlas':
        found = [i for i in _path_bands(pool) if i.attrs.get('projections')] or _path_bands(pool)
        bands = _need(found[0] if found else None, 'band path (≥2 k-points)', kind)
        others = [i for i in _path_bands(pool) if i.path.parent != bands.path.parent]
        mesh = pool.first('bands_mesh') or pool.first('scf_result', lambda i: i.attrs.get('kpoints') and
                                                      i.path.parent == bands.path.parent)
        dos = pool.first('pdos', lambda i: i.path.name == 'dos.json') or (
            pool.first('dos', lambda i: i.path == mesh.path) if mesh is not None else None)
        return {'bands': bands.path, 'reference': others[0].path if others else None,
                'mesh': mesh.path if mesh else None, 'dos': dos.path if dos else None}
    if kind == 'relax':
        return {'relax': _need(pool.first('relax'), 'relaxation result/checkpoint', kind).path}
    if kind == 'sparsity_pareto':
        return {'sweep': _need(pool.first('sweep'), 'sparsity sweep report/summary', kind).path}
    if kind == 'scaling':
        files = [i.path for i in pool.all('scaling')]
        if not files:
            raise InputError(f'{kind}: no sparse-dft-scaling-1 file found')
        return {'scaling': files}
    raise InputError(f'no resolver for {kind}')


def _distinct_runs(paths):
    """One SCF result per directory (scf_result.json preferred over scf.json duplicates)."""
    out, dirs = [], set()
    for p in paths:
        if p.parent in dirs:
            continue
        dirs.add(p.parent); out.append(p)
    return out


def available_kinds(path) -> list[str]:
    """Kinds that can be plotted from a directory, a file, or a list of them."""
    pool = _Pool(path)
    kinds = []
    for kind in KINDS:
        try:
            _resolve(kind, pool)
        except InputError:
            continue
        kinds.append(kind)
    return kinds


def resolve_inputs(kind, sources, *, reference=None) -> dict:
    pool = _Pool(sources)
    if kind == 'dos_compare' and reference is not None:
        ref_pool, native = _Pool(reference), pool.all('dos', lambda i: i.path.name == 'dos.json')
        ref = [i for i in ref_pool.all('bands_file') if i.path.suffix == '.xml'] or ref_pool.all('dos')
        if not native or not ref:
            raise InputError('dos_compare: the candidate needs a sparse-dft dos result and the reference a DOS or '
                             'a uniform-mesh vasprun.xml')
        return {'reference': ref[0].path, 'candidate': native[0].path}
    if kind == 'electronic_atlas' and reference is not None:
        found = _resolve(kind, pool)
        ref = [i for i in _path_bands(_Pool(reference))]
        if not ref:
            raise InputError('electronic_atlas: the reference has no band path')
        return {**found, 'reference': ref[0].path}
    if kind == 'bands_compare' and reference is not None:
        ref_pool = _Pool(reference)
        ref = _path_bands(ref_pool) or ref_pool.all('scf_result', lambda i: i.attrs.get('kpoints'))
        cand = _path_bands(pool) or pool.all('scf_result', lambda i: i.attrs.get('kpoints'))
        if not ref or not cand:
            raise InputError('bands_compare: reference and candidate must each provide band energies')
        return {'reference': ref[0].path, 'candidate': cand[0].path}
    return _resolve(kind, pool)


# --- loading and rendering ----------------------------------------------------
def _select_spin_bands(bands, spin):
    if spin in (None, 'both', 'all') or not bands.spin_channels:
        return bands
    if spin not in bands.spin_channels:
        raise InputError(f'spin channel {spin!r} not in {list(bands.spin_channels)}')
    import copy
    from ..bands import _infer_edges
    b = copy.copy(bands)
    rows = bands.spin_channels[spin]
    b.spin_channels = {spin: rows}
    b.eigenvalues = rows
    occ = {spin: bands.occupations[spin]} if bands.occupations and spin in bands.occupations else None
    b.occupations = occ
    b.vbm, b.cbm, b.gap_status = _infer_edges(rows, bands.fermi_energy, occ, {spin: rows} if occ else None)
    return b


def _select_spin_dos(dos, spin):
    if spin in (None, 'both', 'all') or not dos.spin_polarized:
        return dos
    import copy
    if spin not in ('up', 'down'):
        raise InputError("spin must be 'up', 'down' or 'both'")
    d = copy.copy(dos)
    from dataclasses import replace
    pick = lambda curves: [replace(c, spin=None, label=f'{c.label} ({spin})') for c in curves if c.spin == spin]
    d.total = pick(dos.total)
    d.pdos = pick(dos.pdos)
    d.spin_polarized = False
    d.metadata = {**dos.metadata, 'selected_spin': spin}
    return d


def _load(kind, inputs, opts):
    from ..analysis import electronic as E, relax as R, scf as S, sweep as W
    sigma, npoints = float(opts.get('sigma', 0.1)), int(opts.get('npoints', 2001))
    spin = opts.get('spin')
    if kind in ('scf_convergence', 'magnetization'):
        return S.load_scf_history(inputs['iterations'])
    if kind == 'energy_terms':
        return S.load_energy_terms(inputs['scf'])
    if kind == 'forces':
        return S.load_forces(inputs['scf'])
    if kind == 'stress':
        return S.load_stress(inputs['scf'])
    if kind == 'timing':
        return S.load_timing(inputs['timing'])
    if kind == 'bands':
        return _select_spin_bands(E.load_band_data(inputs['bands'], structure=opts.get('structure')), spin)
    if kind in ('dos', 'pdos'):
        return _select_spin_dos(E.load_dos(inputs['dos'], sigma=sigma, npoints=npoints), spin)
    if kind == 'bands_spin':
        return E.load_band_data(inputs['bands'], structure=opts.get('structure'))
    if kind == 'bands_pdos':
        bands = _select_spin_bands(E.load_band_data(inputs['bands'], structure=opts.get('structure')), spin)
        return bands, _select_spin_dos(E.load_dos(inputs['dos'], sigma=sigma, npoints=npoints), spin)
    if kind == 'dos_compare':
        cand = E.load_dos(inputs['candidate'])
        ref_path = Path(inputs['reference'])
        if ref_path.suffix == '.xml' and 'bands_file' in (scan(ref_path)[0].roles if scan(ref_path) else set()):
            vb = E.load_band_data(ref_path)
            if not E.is_mesh(vb) and not vb.kweights:
                raise InputError(f'{ref_path} is not a uniform-mesh calculation (no k weights)')
            proj = None
            try:
                proj = E.load_projections(vb, ref_path)
            except InputError:
                pass
            ref = E.dos_like(cand, vb, proj, source=ref_path)
        else:
            ref = E.load_dos(ref_path, sigma=cand.sigma_ev or sigma, npoints=npoints)
        return _select_spin_dos(ref, spin), _select_spin_dos(cand, spin)
    if kind == 'bands_dos':
        bands = _select_spin_bands(E.load_band_data(inputs['bands'], structure=opts.get('structure')), spin)
        return bands, _select_spin_dos(E.load_dos(inputs['dos'], sigma=sigma, npoints=npoints), spin)
    if kind == 'fatbands':
        bands = E.load_band_data(inputs['bands'], structure=opts.get('structure'))
        procar = inputs.get('procar')
        species = None
        if procar is not None:
            species = E._sibling_species(Path(procar), None)
        return bands, E.load_projections(bands, procar, species=species,
                                         resolution=opts.get('resolution', 'auto'))
    if kind == 'bands_compare':
        def band_source(path):
            info = scan(path)
            if info and 'scf_result' in info[0].roles and 'bands_file' not in info[0].roles:
                return E.bands_from_scf(path)
            return E.load_band_data(path, structure=opts.get('structure'))
        return band_source(inputs['reference']), band_source(inputs['candidate'])
    if kind == 'electronic_atlas':
        from .atlas import load_atlas
        return load_atlas(inputs['bands'], reference=inputs.get('reference'), mesh=inputs.get('mesh'),
                          dos=inputs.get('dos'), structure=opts.get('structure'))
    if kind == 'relax':
        return R.load_relax(inputs['relax'], tolerance=opts.get('tolerance'))
    if kind == 'sparsity_pareto':
        return W.load_sweep(inputs['sweep'])
    if kind == 'scaling':
        return W.load_scaling(inputs['scaling'])
    raise InputError(f'no loader for {kind}')


def _plotters():
    from . import atlas as AT, band_toolkit as BT, electronic_plots as EP, scf_plots as SP, workflow_plots as WP
    return {'scf_convergence': SP.plot_scf_convergence, 'magnetization': SP.plot_magnetization,
            'energy_terms': SP.plot_energy_terms, 'forces': SP.plot_forces, 'stress': SP.plot_stress,
            'timing': SP.plot_timing, 'bands': EP.plot_bands, 'bands_dos': EP.plot_bands_dos, 'dos': EP.plot_dos,
            'pdos': BT.plot_pdos_resolved, 'fatbands': BT.plot_fatbands, 'bands_compare': EP.plot_bands_compare,
            'bands_spin': BT.plot_bands_spin, 'bands_pdos': BT.plot_bands_pdos, 'dos_compare': BT.plot_dos_compare,
            'electronic_atlas': AT.plot_electronic_atlas, 'relax': WP.plot_relax, 'sparsity_pareto': WP.plot_sparsity_pareto, 'scaling': WP.plot_scaling}


LOADER_ONLY_OPTIONS = ('sigma', 'npoints', 'spin', 'structure', 'resolution')


MULTI_PANEL = {'electronic_atlas'}        # kinds that can be split into one figure per panel


def make_figures(kind, sources, *, panels='combined', width=None, lang='en', palette='academic', **options) -> dict:
    """{name: Figure} for ``kind``: 'combined' -> {'combined': fig}; 'split' -> one per panel; 'both'.

    Single-panel kinds always return {'combined': fig}.  Every figure carries its
    own provenance values (``fig.sparse_dft_meta``).
    """
    if kind not in MULTI_PANEL:
        return {'combined': make_figure(kind, sources, width=width or 'single', lang=lang, palette=palette,
                                        **options)}
    from ._util import attach_meta
    from .atlas import analyse_atlas, render_atlas
    opts = {k: v for k, v in options.items() if v is not None}
    base_fontsize = float(opts.pop('base_fontsize', 7.0))
    reference = opts.pop('reference', None)
    inputs = resolve_inputs(kind, sources, reference=reference)
    data = _load(kind, inputs, opts)
    with figure_style('atlas', lang=lang, palette=palette, base_fontsize=base_fontsize, frame=True):
        ctx = analyse_atlas(data, emin=opts.get('emin'), emax=opts.get('emax'))
        figs = render_atlas(ctx, panels)
    params = {**{k: (str(v) if isinstance(v, Path) else v) for k, v in opts.items()}, 'width': 'atlas',
              'lang': lang, 'palette': palette, 'panels': panels, 'reference': str(reference) if reference else None}
    for fig in figs.values():
        attach_meta(fig, kind, [p for p in inputs.values() if p], parameters=params)
    return figs


def make_figure(kind, sources, *, width='single', lang='en', palette='academic', **options):
    """Load ``sources`` (directories or files) and draw ``kind`` as a styled Figure.

    Options (all optional): emin, emax (eV, relative to the energy zero), fermi (eV),
    align ('fermi'|'vbm'|'none'), sigma (eV, SCF Gaussian DOS; default 0.1), spin
    ('up'|'down'|'both'), reference (band source for bands_compare), top (timing),
    metric/systems (sparsity_pareto), group/channel (fatbands), tolerance
    (relax/forces), per_atom/mode (energy_terms), base_fontsize (pt).
    """
    from ._util import attach_meta
    if kind in MULTI_PANEL:
        return make_figures(kind, sources, panels='combined', lang=lang, palette=palette, **options)['combined']
    opts = {k: v for k, v in options.items() if v is not None}
    base_fontsize = float(opts.pop('base_fontsize', 7.0))
    reference = opts.pop('reference', None)
    inputs = resolve_inputs(kind, sources, reference=reference)
    data = _load(kind, inputs, opts)
    plot_opts = {k: v for k, v in opts.items() if k not in LOADER_ONLY_OPTIONS}
    with figure_style(width, lang=lang, palette=palette, base_fontsize=base_fontsize):
        fig = _plotters()[kind](data, **plot_opts)
    recorded = {k: (str(v) if isinstance(v, Path) else v) for k, v in opts.items()}
    attach_meta(fig, kind, [p for v in inputs.values() for p in (v if isinstance(v, list) else [v]) if p],
                parameters={**recorded, 'width': width, 'lang': lang, 'palette': palette,
                            'base_fontsize': base_fontsize, 'reference': str(reference) if reference else None})
    return fig
