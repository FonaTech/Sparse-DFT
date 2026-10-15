"""One-command scientific band/DOS analysis and deterministic plotting.

The module has no mandatory plotting dependency.  When matplotlib is present
it emits publication PNG/SVG figures; otherwise it emits a standards-compliant
SVG with the same data.  Every artifact contains SHA-256 provenance and the
exact parser settings used to create it.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import math
from ..bands import load_bands, parse_doscar, parse_procar, density_of_states_from_scf, _distances
from ..errors import InputError


def _sha(path):
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def _formula(structure):
    if not structure:
        return None
    try:
        from ..structure import read_structure
        s = read_structure(structure) if isinstance(structure, (str, Path)) else structure
        counts = {}
        for x in s.species: counts[x] = counts.get(x, 0) + 1
        return ''.join(k + (str(v) if v != 1 else '') for k, v in sorted(counts.items()))
    except Exception:
        return None


def _band_metrics(bands):
    return {'vbm_ev': bands.vbm, 'cbm_ev': bands.cbm, 'gap_ev': bands.band_gap,
            'gap_status': bands.gap_status,
            'is_metal': bands.gap_status in ('metal', 'metal_or_numerical')}


def _normalize_projections(bands, projection):
    """Adapt raw native and PROCAR weights to one checked tensor contract."""
    if projection is None:
        return None
    orbitals = projection.get('projection_orbitals') or []
    if not orbitals:
        raise InputError('projection output lacks orbital labels')
    nions = projection.get('nions') or len(projection.get('projection_atom_symbols') or [])
    if not nions or projection.get('nkpoints', bands.nkpoints) != bands.nkpoints or projection.get('nbands', bands.nbands) != bands.nbands:
        raise InputError('projection dimensions differ from band output')
    weights = projection.get('projection_weights')
    if isinstance(weights, list):
        expected = bands.nkpoints * bands.nbands * nions * len(orbitals)
        if len(weights) != expected:
            raise InputError(f'native projection has {len(weights)} values; expected {expected}')
        iterator = iter(weights)
        weights = {'total': [[[[next(iterator) for _ in orbitals] for _ in range(nions)]
                              for _ in range(bands.nbands)] for _ in range(bands.nkpoints)]}
    if not isinstance(weights, dict) or not weights:
        raise InputError('projection output lacks weights')
    for name, channel in weights.items():
        if len(channel) != bands.nkpoints or any(len(row) != bands.nbands for row in channel):
            raise InputError(f'projection channel {name} has incorrect band dimensions')
        for row in channel:
            for atoms in row:
                if len(atoms) != nions or any(len(atom) != len(orbitals) for atom in atoms):
                    raise InputError(f'projection channel {name} has incorrect atom/orbital dimensions')
                if any(not math.isfinite(float(v)) for atom in atoms for v in atom):
                    raise InputError('projection contains nonfinite weights')
                if name not in ('mx', 'my', 'mz') and any(float(v)<-1e-10 for atom in atoms for v in atom):
                    raise InputError('charge/orbital projection contains negative weights')
        if projection.get('projection_kpoints'):
            points = projection['projection_kpoints'].get(name)
            if points is None or len(points)!=bands.nkpoints or any(
                max(abs(float(a)-float(c)) for a,c in zip(p,q))>1e-6
                for p,q in zip(points,bands.kpoints)):
                raise InputError('PROCAR kpoints do not match band output')
        if projection.get('projection_energies_ev'):
            energies = projection['projection_energies_ev'].get(name)
            target = (bands.spin_channels or {}).get(name, bands.eigenvalues)
            if name in ('up','down') and len(weights)>1 and name not in (bands.spin_channels or {}):
                raise InputError('spin-resolved PROCAR requires spin-resolved band energies')
            if energies is None or any(abs(a-c)>5e-4 for r,s in zip(energies,target) for a,c in zip(r,s)):
                raise InputError('PROCAR energies do not match band output')
    return {**projection, 'nions': nions, 'nkpoints': bands.nkpoints, 'nbands': bands.nbands,
            'projection_orbitals': orbitals, 'projection_weights': weights}


def _svg_escape(x):
    return str(x).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def _infer_path_labels(bands):
    """Retain explicit crystallographic labels; only Gamma is universal."""
    labels = list(bands.labels or [None] * bands.nkpoints)
    if any(label for label in labels):
        return labels
    # Only Gamma is independent of reciprocal-cell convention. Fractional
    # (.25,0,0) is not universally X (in particular not for primitive fcc).
    known = {(0.0, 0.0, 0.0): r'$\Gamma$'}
    for i, point in enumerate(bands.kpoints):
        key = tuple(round(float(x), 8) for x in point)
        if key in known:
            labels[i] = known[key]
    # Keep only genuine path vertices: an endpoint repeated across adjacent
    # line segments receives a single label at the first occurrence.
    for i in range(1, len(labels)):
        if labels[i] == labels[i - 1]:
            labels[i] = None
    return labels


def _write_fallback_svg(path, bands, dos, title, channels):
    """Minimal vector renderer used when matplotlib is unavailable."""
    W, H, left, right, top, bottom = 1200, 700, 80, 80, 55, 65
    gap = 40 if dos else 0
    bw = (W - left - right - gap) * (0.68 if dos else 1.0)
    dw = (W - left - right - gap) - bw if dos else 0
    x0, y0 = left, top; ph = H - top - bottom
    all_e = [e for ch in channels.values() for row in ch for e in row]
    if dos: all_e += dos['energies_ev']
    ef = bands.fermi_energy or 0.0
    emin, emax = min(all_e + [ef]), max(all_e + [ef])
    pad = max(0.5, 0.05 * (emax - emin)); emin -= pad; emax += pad
    def y(e): return y0 + ph * (emax - e) / (emax - emin)
    def bx(i): return x0 + bw * i / max(1, len(bands.distances) - 1)
    xmax = bands.distances[-1] if bands.distances else 1.0
    def bx2(d): return x0 + bw * d / max(xmax, 1e-12)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
           '<rect width="100%" height="100%" fill="white"/>',
           f'<text x="{W/2:.1f}" y="28" text-anchor="middle" font-family="sans-serif" font-size="18">{_svg_escape(title)}</text>']
    out += [f'<rect x="{x0}" y="{y0}" width="{bw}" height="{ph}" fill="none" stroke="#333"/>']
    if dos: out += [f'<rect x="{x0+bw+gap}" y="{y0}" width="{dw}" height="{ph}" fill="none" stroke="#333"/>']
    colors = {'up':'#1565c0', 'down':'#d32f2f', 'total':'#333333'}
    for name, ev in channels.items():
        color = colors.get(name, '#008577')
        for b in range(len(ev[0])):
            pts = ' '.join(f'{bx2(bands.distances[i]):.2f},{y(ev[i][b]):.2f}' for i in range(len(ev)))
            out.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1"/>')
    out += [f'<line x1="{x0}" x2="{x0+bw}" y1="{y(ef):.2f}" y2="{y(ef):.2f}" stroke="#777" stroke-dasharray="5,4"/>',
            f'<text x="{x0-8}" y="{y(ef)+4:.2f}" text-anchor="end" font-family="sans-serif" font-size="12">E<tspan baseline-shift="sub">F</tspan></text>']
    if dos:
        de = dos['energies_ev']; tot = dos['total']
        if 'total_up' in tot and 'total_down' in tot:
            # Spin DOSCAR: up to the right of the centre line, down mirrored to the left.
            curves = [(tot['total_up'], 1.0, '#0072B2'), (tot['total_down'], -1.0, '#D55E00')]
            dmax = max(max(tot['total_up'], default=0.0), max(tot['total_down'], default=0.0)) or 1.0
            centre = x0 + bw + gap + dw / 2; half = dw / 2
        else:
            curves = [(tot.get('total', []), 1.0, '#333333')]
            dmax = max(curves[0][0], default=0.0) or 1.0
            centre = x0 + bw + gap; half = dw
        for values, sign, color in curves:
            pts = ' '.join(f'{centre + sign*half*v/max(dmax,1e-12):.2f},{y(e):.2f}' for e,v in zip(de,values))
            out.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.2"/>')
    out += [f'<text x="{x0+bw/2:.1f}" y="{H-18}" text-anchor="middle" font-family="sans-serif">k-path</text>',
            f'<text x="18" y="{y0+ph/2:.1f}" transform="rotate(-90 18 {y0+ph/2:.1f})" text-anchor="middle" font-family="sans-serif">Energy (eV)</text>']
    if dos: out.append(f'<text x="{x0+bw+gap+dw/2:.1f}" y="{H-18}" text-anchor="middle" font-family="sans-serif">DOS</text>')
    out.append('</svg>')
    Path(path).write_text('\n'.join(out), encoding='utf-8')


def _write_fallback_pdos_svg(path, dos, elements, fermi):
    """Dependency-free PDOS: one panel per atom, one polyline per projected column."""
    colors = ['#0072B2', '#D55E00', '#009E73', '#E69F00', '#CC79A7', '#56B4E9']
    pdos = dos['pdos']; labels = (elements or [])[:len(pdos)] or [f'atom-{i+1}' for i in range(len(pdos))]
    W, H = 900, max(260, 180 * len(pdos))
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}"><rect width="100%" height="100%" fill="white"/>']
    for i, block in enumerate(pdos):
        xs = [x - fermi for x in block['energies_ev']]; cols = block['columns']
        xlo, xhi = min(xs), max(xs); ybase = 20 + i * 180
        ymax = max((v for c in cols.values() for v in c), default=0.0) or 1.0
        for j, (name, ys) in enumerate(cols.items()):
            pts = ' '.join(f'{60 + 780*(x-xlo)/max(xhi-xlo,1e-12):.2f},{ybase+140-110*v/ymax:.2f}' for x, v in zip(xs, ys))
            out.append(f'<polyline points="{pts}" fill="none" stroke="{colors[j % len(colors)]}"><title>{_svg_escape(name)}</title></polyline>')
        label = labels[i] if i < len(labels) else f'atom-{i+1}'
        out.append(f'<text x="10" y="{ybase+20}" font-family="sans-serif">{_svg_escape(label)}</text>')
    out.append('</svg>')
    Path(path).write_text('\n'.join(out), encoding='utf-8')


def _report_fermi(report, dos):
    """Energy zero for the report figures; refuses to invent 0 eV."""
    value = report['bands'].get('fermi_energy_ev')
    if value is None and dos:
        value = dos.get('fermi_energy_ev')
    if value is None:
        raise InputError(
            f"Fermi energy is unknown for {report['bands']['source']}; band/DOS plots are drawn as E - E_F. "
            'Pass fermi_energy=<eV> to analyze_bands (CLI: sparse-dft bands-analysis --fermi <eV>), '
            'or render=False for the numeric report only.',
            details={'source': report['bands']['source'], 'option': 'fermi_energy', 'cli_flag': '--fermi'})
    return float(value)


def _report_formats(svg_only=False):
    return ('svg',) if svg_only else ('png', 'pdf', 'svg')


def render_scientific_plots(report: dict, output_dir: str | Path, *, basename='sparse_dft_bands',
                            emin=None, emax=None) -> dict:
    """Render the band/DOS/PDOS/projection figures of an :func:`analyze_bands` report.

    Uses :mod:`sparse_dft.plotting` (journal style, no pyplot, no global rcParams);
    falls back to a dependency-free SVG when matplotlib is not installed.
    """
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    b = load_bands(report['bands']['source'])
    b.fermi_energy = report['bands']['fermi_energy_ev']
    b.distances = report['bands'].get('distances', b.distances)
    dos = report.get('dos')
    channels = b.spin_channels or {'total': b.eigenvalues}
    title = report.get('formula') or 'Sparse-DFT band structure'
    artifacts = []
    fermi = _report_fermi(report, dos)
    try:
        import matplotlib  # noqa: F401  (availability probe only; never matplotlib.use)
    except ImportError:
        b.fermi_energy = fermi
        svg = out / f'{basename}.svg'; _write_fallback_svg(svg, b, dos, title, channels); artifacts.append(str(svg))
        if dos and dos.get('pdos'):
            psvg = out / f'{basename}_pdos.svg'; _write_fallback_pdos_svg(psvg, dos, report.get('elements'), fermi)
            artifacts.append(str(psvg))
        return {'artifacts': artifacts, 'renderer': 'stdlib-svg', 'pdos_panels': len(dos.get('pdos', [])) if dos else 0}
    from ..plotting import export, figure_style
    from ..plotting.electronic_plots import plot_bands, plot_bands_dos, plot_fatbands, plot_pdos
    from .electronic import ProjectionSet, dos_from_payload
    elements = report.get('elements')
    dos_data = dos_from_payload(dos, elements) if dos else None
    inputs = [report['bands']['source']] + ([dos.get('source')] if dos else [])
    def save(fig, stem, extra_inputs=()):
        files = export(fig, out / stem, formats=_report_formats(), dpi=320,
                       provenance={'inputs': inputs + list(extra_inputs),
                                   'parameters': {'report_schema': report.get('schema_version'),
                                                  'formula': report.get('formula')}})
        artifacts.extend(str(f) for f in files if not str(f).endswith('.provenance.json'))
    with figure_style('double' if dos else 'single', lang='en'):
        if dos_data is not None:
            fig = plot_bands_dos((b, dos_data), emin=emin, emax=emax, fermi=fermi)
        else:
            fig = plot_bands(b, emin=emin, emax=emax, fermi=fermi)
        save(fig, basename)
    # Element/l projected DOS: one panel per element, one curve per l.
    if dos_data is not None and dos_data.pdos:
        with figure_style('single', lang='en'):
            save(plot_pdos(dos_data, emin=emin, emax=emax, fermi=fermi), f'{basename}_pdos')
    projection = report.get('projections')
    if projection:
        # The report already holds the validated tensor (_normalize_projections).
        weights = projection['projection_weights']
        species = projection.get('atom_elements')
        proj = ProjectionSet(source=str(projection.get('source') or report['bands']['source']),
                             channels=weights, orbitals=list(projection['projection_orbitals']),
                             nions=int(projection['nions']),
                             species=list(species) if species and len(species) == int(projection['nions']) else None,
                             signed=any(c in ('mx', 'my', 'mz') for c in weights),
                             weight_definition=projection.get('weight_definition'),
                             spin_projection_available=bool(projection.get('spin_projection_available')))
        windows = [('overall', emin, emax),
                   ('local', emin if emin is not None else -2.0, emax if emax is not None else 2.0)]
        for suffix, lower, upper in windows:
            with figure_style('single', lang='en'):
                fig = plot_fatbands((b, proj), emin=lower, emax=upper, fermi=fermi)
                save(fig, f'{basename}_projection_{suffix}', [proj.source])
    return {'artifacts': artifacts, 'renderer': 'matplotlib', 'pdos_panels': len(dos.get('pdos', [])) if dos else 0}


def analyze_bands(bands_path: str | Path, *, dos_path: str | Path | None = None,
                  dos_scf_path: str | Path | None = None,
                  procar_path: str | Path | None = None, structure=None, fermi_energy: float | None = None,
                  gaussian_sigma: float = 0.05, output_dir: str | Path | None = None,
                  render: bool = True, emin=None, emax=None) -> dict:
    bands = load_bands(bands_path)
    bands_path = Path(bands.source)
    if procar_path is None and bands_path.name in ('EIGENVAL','vasprun.xml'):
        sibling = bands_path.with_name('PROCAR')
        if sibling.is_file():procar_path=sibling
    # Sparse-DFT JSON deliberately stores fractional k-points.  When the source
    # structure is available, use its reciprocal metric for the plotting axis
    # so a path is not stretched according to arbitrary fractional axes.
    if structure:
        try:
            from ..structure import read_structure
            structure_obj = read_structure(structure) if isinstance(structure, (str, Path)) else structure
            reciprocal = structure_obj.reciprocal_lattice
            if reciprocal:
                bands.distances = _distances(bands.kpoints, reciprocal)
        except Exception:
            pass
    if fermi_energy is not None: bands.fermi_energy = float(fermi_energy)
    if dos_path and dos_scf_path:
        raise InputError('choose DOSCAR or a weighted SCF result for DOS')
    dos = (parse_doscar(dos_path) if dos_path else
           density_of_states_from_scf(dos_scf_path, sigma=gaussian_sigma) if dos_scf_path else None)
    procar = parse_procar(procar_path) if procar_path else None
    projection = _normalize_projections(bands, bands.projections) if bands.projections else None
    if procar:
        projection = _normalize_projections(bands, procar)
    elements = None
    if structure:
        try:
            from ..structure import read_structure
            s = read_structure(structure) if isinstance(structure, (str, Path)) else structure
            elements = s.species
        except Exception:
            elements = None
    report = {'schema_version': 'sparse-dft-band-analysis-1', 'formula': _formula(structure), 'elements': elements,
              'bands': {'source': str(bands_path), 'sha256': _sha(bands_path), 'nkpoints': bands.nkpoints,
                        'distances': bands.distances,
                        'nbands': bands.nbands, 'spin_channels': list((bands.spin_channels or {}).keys()),
                        'fermi_energy_ev': bands.fermi_energy, 'metrics': _band_metrics(bands)},
              'dos': dos, 'projections': projection,
              'provenance': {'bands_sha256': _sha(bands_path), 'dos_sha256': _sha(dos_path) if dos_path else None,
                                         'procar_sha256': _sha(procar_path) if procar_path else None,
                                         'structure_sha256': _sha(structure) if isinstance(structure, (str, Path)) else None,
                                         'dos_scf_sha256': _sha(dos_scf_path) if dos_scf_path else None,
                                         'parser': 'sparse_dft.analysis/1', 'raw_dos': bool(dos_path),
                                         'projection_source': str(procar_path or bands_path) if projection else None},
              'warnings': []}
    if dos_scf_path:
        report['warnings'].append('Weighted Gaussian DOS uses the executed SCF k-points; mesh and broadening convergence remain unverified')
    elif not dos_path:
        report['warnings'].append('DOS omitted: provide DOSCAR or weighted SCF eigenvalues; a symmetry path is insufficient')
    if projection and elements and len(elements) != projection['nions']:
        report['warnings'].append(f'projection atom count {projection["nions"]} differs from structure atom count {len(elements)}; atom labels omitted')
    elif projection:
        report['projections']['atom_elements'] = elements
    if projection and 'raw_sum_abs_Bpsi_squared' in projection.get('weight_definition',''):
        report['warnings'].append('Sparse-DFT raw PAW projector intensities are basis dependent; they are not VASP LORBIT sphere occupations or normalized probabilities')
    if not projection or not projection.get('spin_projection_available'):
        report['warnings'].append('No physical spin projection in this input; no synthetic spin channel is generated')
    if output_dir:
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        (out / 'analysis.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        if render:
            report['plots'] = render_scientific_plots(report, out, emin=emin, emax=emax)
            # Always deliver an ordinary local band view as well as the full view.
            report['plots']['artifacts'] += render_scientific_plots(
                {**report,'projections':None},out,basename='sparse_dft_bands_local',
                emin=emin if emin is not None else -2.,emax=emax if emax is not None else 2.)['artifacts']
            (out / 'analysis.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return report
