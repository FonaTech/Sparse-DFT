from __future__ import annotations
import argparse, json, sys
from .structure import read_structure
from .kpoints import recommend_kpoints, interpolate_band_path
from .paw import Paw54JsonArchive, PawSelectionPolicy
from .workflow import prepare_job, progress_snapshot
from .progress import init_progress, doctor, update_task
from .bands import load_bands, compare_bands
from .analysis import analyze_bands
from .mp_data import MaterialsProjectClient, write_snapshot, write_band_snapshot
from .scf import run_scf, run_bands, run_dos
from .spin_model import parse_cli_hubbard
from .relax import ionic_relax, cell_relax
from .backend import backend_info
from .settings import DEFAULT_SCF_MAX_ITERATIONS, DEFAULT_SCF_MIXING
from .checkpoint import canonical_projector_interpolation

def _cli_parallel(a):
    from .parallel import coerce_layout
    return coerce_layout({'ranks': a.mpi_ranks, 'kgroups': a.kgroups, 'bandgroups': a.bandgroups,
                          'threads_per_rank': a.threads_per_rank, 'deterministic': a.deterministic})


def _cli_sparsity(a):
    from .sparsity import coerce_profile
    from .errors import InputError
    components = {k: getattr(a, k) for k in ('density_cutoff_factor', 'xc_padding', 'eigen_search_factor',
                  'block_ortho', 'paw_compensation', 'projector_space', 'projector_radius_factor', 'search_precision')}
    components = {k: (bool(v) if k == 'block_ortho' and v is not None else v) for k, v in components.items() if v is not None}
    auto = {'level': a.sparsity_level, 'tail_budget': a.sparsity_tail_budget, 'candidates': a.sparsity_candidates,
            'min_factor': a.sparsity_min_factor, 'max_factor': a.sparsity_max_factor,
            'metric': a.sparsity_metric, 'rules': a.sparsity_rules}
    auto = {k: v for k, v in auto.items() if v is not None}
    if auto and a.sparsity != 'auto':
        raise InputError('--sparsity-level/-tail-budget/-candidates/-min-factor/-max-factor/-metric/-rules need --sparsity auto')
    if a.sparsity == 'off' and components:
        return coerce_profile({'mode': 'custom', **components})
    return coerce_profile({'mode': a.sparsity, **components, **({'auto_controls': auto} if auto else {})})


def _cli_solver(a):
    from .sparsity import SOLVER_CONTROLS, validate_solver_controls
    return validate_solver_controls({k: getattr(a, k) for k in SOLVER_CONTROLS})


# Subcommands owned by their own modules (argument parsing included), so the
# plotting and GUI-worker layers evolve without editing this dispatcher.
_DELEGATED = {'plot': ('sparse_dft.plotting.cli', 'publication figures for every supported calculation'),
              'worker': ('sparse_dft.runner', 'run one JSON job description in this process (GUI/batch worker)')}


def build_parser():
    """The complete argparse tree; the GUI introspects it to build its forms."""
    p=argparse.ArgumentParser(prog='sparse-dft'); sub=p.add_subparsers(dest='cmd',required=True)
    for name, (_module, text) in _DELEGATED.items():
        sub.add_parser(name, help=text, add_help=False)
    q=sub.add_parser('inspect'); q.add_argument('structure')
    q=sub.add_parser('suggest', help='classify a structure and print recommended options per task (JSON); nothing is run')
    q.add_argument('structure'); q.add_argument('--task', choices=['scf','bands','dos','relax'], default='scf')
    q.add_argument('--paw-root', help='PAW archive; enables cutoff and electron-count based suggestions')
    q.add_argument('--accuracy', choices=['low','standard','high','production'], default='standard')
    q.add_argument('--band-path-out', help='with --task bands: write the seekpath band path JSON here for --path')
    q.add_argument('--argv', action='store_true', help='print only the suggested sparse_dft command line')
    q=sub.add_parser('recommend-kpoints'); q.add_argument('structure'); q.add_argument('--task',default='scf'); q.add_argument('--accuracy',default='standard'); q.add_argument('--spacing',type=float,help='target reciprocal spacing in 1/A; validated against the supported range')
    q=sub.add_parser('make-band-path', help='interpolate a high-symmetry fractional path with explicit VASP-compatible samples'); q.add_argument('vertices', nargs='?', help='JSON file containing an array of fractional vertices (omit with --auto)'); q.add_argument('--points-per-segment', type=int, default=81); q.add_argument('--no-duplicate-vertices', action='store_true'); q.add_argument('--out', required=True)
    q.add_argument('--auto', metavar='STRUCTURE', help='standard labelled path from spglib+seekpath (Hinuma 2017) for this structure, in its own reciprocal basis')
    q.add_argument('--symprec', type=float, help='symmetry tolerance in Angstrom (default: ladder 1e-5, 1e-3, 1e-2; the first giving the highest symmetry up to 1e-2 is reported)')
    q.add_argument('--spacing', type=float, default=0.025, help='--auto: k-point spacing along the path in 1/Angstrom (2 pi included)')
    q.add_argument('--periodic-axes', choices=['auto', 'all'], default='auto', help='--auto: drop segments along vacuum axes of slabs, wires and molecules (auto) or keep the 3D path (all)')
    q.add_argument('--export-primitive', metavar='POSCAR', help='--auto: also write the spglib standard primitive cell (use it when the input is a supercell)')
    q=sub.add_parser('paw-inventory'); q.add_argument('--root',required=True)
    q=sub.add_parser('paw-convert', help='build the indexed PAW archive from your own licensed POTCAR set (directory or tar archive)')
    q.add_argument('source', help='potpaw directory or .tar/.tar.gz/.tgz/.tar.xz with <dataset>/POTCAR'); q.add_argument('--out', required=True, help='new or empty output directory')
    q.add_argument('--overwrite', action='store_true', help='allow a non-empty output directory')
    q=sub.add_parser('paw-resolve'); q.add_argument('--root',required=True); q.add_argument('--elements',nargs='+',required=True); q.add_argument('--family',default='standard')
    q=sub.add_parser('prepare'); q.add_argument('structure'); q.add_argument('--paw-root',required=True); q.add_argument('--out',default='job'); q.add_argument('--task',default='scf'); q.add_argument('--accuracy',default='standard')
    sub.add_parser('progress')
    sub.add_parser('doctor')
    sub.add_parser('backend-info')
    q=sub.add_parser('validate'); q.add_argument('--all',action='store_true')
    q=sub.add_parser('mp-fetch'); q.add_argument('--out',default='data/mp_100'); q.add_argument('--count',type=int,default=100); q.add_argument('--api-key'); q.add_argument('--api-root',default='https://api.materialsproject.org'); q.add_argument('--timeout',type=float,default=None,help='network timeout in seconds; omitted means no hard timeout')
    q=sub.add_parser('mp-fetch-bands', help='fetch raw MP eigenvalue paths; fails on summary-only API responses'); q.add_argument('--ids', nargs='+', required=True); q.add_argument('--out',default='data/mp_bands'); q.add_argument('--api-key'); q.add_argument('--api-root',default='https://api.materialsproject.org'); q.add_argument('--endpoint',default='/materials/bandstructure/'); q.add_argument('--timeout',type=float,default=None,help='network timeout in seconds; omitted means no hard timeout')
    q=sub.add_parser('bands-compare'); q.add_argument('reference'); q.add_argument('candidate'); q.add_argument('--align',choices=['auto','vbm','fermi','none'],default='auto',help='energy zero: auto = sampled VBM when both sources are gapped with equal occupied band counts, otherwise each SCF E_F; none = raw energies'); q.add_argument('--max-energy-ev',type=float,default=10.0); q.add_argument('--out')
    q=sub.add_parser('bands-analysis', help='parse bands/DOS and produce a provenance-tracked scientific report and plots'); q.add_argument('bands'); q.add_argument('--doscar'); q.add_argument('--dos-scf',help='converged SCF JSON with executed weighted k-points'); q.add_argument('--procar'); q.add_argument('--structure'); q.add_argument('--fermi',type=float); q.add_argument('--sigma',type=float,default=0.05); q.add_argument('--emin',type=float,help='plot lower limit relative to E_F (eV)'); q.add_argument('--emax',type=float,help='plot upper limit relative to E_F (eV)'); q.add_argument('--out',default='band_analysis'); q.add_argument('--no-render',action='store_true')
    q=sub.add_parser('scf'); q.add_argument('structure'); q.add_argument('--paw-root',required=True); q.add_argument('--out',default='job'); q.add_argument('--cutoff',type=float,default=None,help='cutoff in eV; omitted selects 1.30*max(PAW ENMAX), rounded up'); q.add_argument('--accuracy',choices=['low','standard','high','production'],default='standard'); q.add_argument('--kpoint',nargs=3,type=float,default=(0.0,0.0,0.0),metavar=('KX','KY','KZ')); q.add_argument('--max-iterations',type=int,default=240); q.add_argument('--density-tolerance',type=float,default=1e-6); q.add_argument('--mixing',type=float,default=DEFAULT_SCF_MIXING); q.add_argument('--bands',type=int,default=0)
    q=sub.add_parser('bands'); q.add_argument('structure'); q.add_argument('--paw-root',required=True); q.add_argument('--out',default='job'); q.add_argument('--path',required=True,help='JSON file containing a list of fractional k-points'); q.add_argument('--cutoff',type=float,default=None,help='cutoff in eV; omitted selects 1.30*max(PAW ENMAX), rounded up'); q.add_argument('--accuracy',choices=['low','standard','high','production'],default='standard'); q.add_argument('--kpoint',nargs=3,type=float,default=(0.0,0.0,0.0),metavar=('KX','KY','KZ'),help='SCF k-point used before NSCF path'); q.add_argument('--max-iterations',type=int,default=240); q.add_argument('--density-tolerance',type=float,default=1e-6); q.add_argument('--mixing',type=float,default=DEFAULT_SCF_MIXING); q.add_argument('--bands',type=int,default=0)
    q=sub.add_parser('dos', help='NSCF on a full uniform mesh, then Gaussian total, spin and lm-resolved PDOS (dos.json)'); q.add_argument('structure'); q.add_argument('--paw-root',required=True); q.add_argument('--out',default='dos')
    q.add_argument('--dos-grid',nargs=3,type=int,required=True,metavar=('NX','NY','NZ'),help='full uniform NSCF mesh for the DOS (no symmetry reduction); typically denser than --scf-grid')
    q.add_argument('--dos-mesh-mode',choices=['gamma','monkhorst-pack'],default='gamma',help='centring of the DOS mesh')
    q.add_argument('--dos-sigma',type=float,default=0.05,help='Gaussian broadening in eV; converge it together with --dos-grid')
    q.add_argument('--dos-step',type=float,default=0.005,help='energy grid step in eV; must not exceed sigma/3')
    q.add_argument('--dos-emin',type=float,help='lower edge of the energy grid, absolute eV (default lowest band - 1 eV)')
    q.add_argument('--dos-emax',type=float,help='upper edge, absolute eV (default where the highest solved band starts + 3 sigma; above it the DOS is incomplete)')
    q.add_argument('--cutoff',type=float,default=None,help='cutoff in eV; omitted selects 1.30*max(PAW ENMAX), rounded up'); q.add_argument('--accuracy',choices=['low','standard','high','production'],default='standard'); q.add_argument('--kpoint',nargs=3,type=float,default=(0.0,0.0,0.0),metavar=('KX','KY','KZ'),help='SCF k-point used before the NSCF mesh'); q.add_argument('--max-iterations',type=int,default=240); q.add_argument('--density-tolerance',type=float,default=1e-6); q.add_argument('--mixing',type=float,default=DEFAULT_SCF_MIXING); q.add_argument('--bands',type=int,default=0,help='bands solved on the DOS mesh; 0 = automatic (raise it to extend the DOS above E_F)')
    q=sub.add_parser('relax'); q.add_argument('structure'); q.add_argument('--paw-root',required=True); q.add_argument('--out',default='relax'); q.add_argument('--cell',action='store_true'); q.add_argument('--max-steps',type=int,default=10); q.add_argument('--cutoff',type=float,default=None,help='cutoff in eV; omitted selects it from PAW ENMAX'); q.add_argument('--accuracy',choices=['low','standard','high','production'],default='standard'); q.add_argument('--max-iterations',type=int,default=240); q.add_argument('--mixing',type=float,default=DEFAULT_SCF_MIXING); q.add_argument('--max-workers',type=int,default=1,help='independent derivative SCF workers'); q.add_argument('--max-displacement',type=float,default=0.15); q.add_argument('--max-strain',type=float,default=0.02)
    sub.choices['relax'].add_argument('--derivative-method', choices=['analytic','finite_difference'], default='analytic')
    sub.choices['relax'].add_argument('--force-tolerance', type=float, default=.02, help='maximum allowed atom force norm, eV/A')
    sub.choices['relax'].add_argument('--stress-tolerance', type=float, default=.00031207545, help='maximum stress component, eV/A^3; default 0.05 GPa')
    sub.choices['relax'].add_argument('--density-tolerance', type=float, default=1e-8)
    for command in ('scf', 'bands', 'dos', 'relax'):
        sub.choices[command].add_argument('--onecenter-density-tolerance', type=float, default=1e-6,
                                          help='maximum per-atom normalized Frobenius residual of the one-centre occupation matrix')
    for name in ('scf', 'bands', 'dos', 'relax'):
        parser = sub.choices[name]
        parser.add_argument('--spin-channels', type=int, choices=[1, 2], default=None, help='1 nonmagnetic; 2 collinear; inherit frozen model for bands')
        parser.add_argument('--initial-moments', nargs='+', type=float, help='one initial magnetic moment per atom, muB')
        parser.add_argument('--hubbard', action='append', metavar='ELEMENT:L:UEFF_EV', help='Dudarev model, e.g. Fe:2:4; may repeat for different elements')
        parser.add_argument('--scf-grid', nargs=3, type=int, metavar=('NX','NY','NZ'), help='full uniform SCF mesh; all points contribute to density')
        parser.add_argument('--mesh-mode', choices=['gamma','monkhorst-pack'], default='gamma')
        parser.add_argument('--smearing-ev', type=float, default=0.0, help='Fermi-Dirac kBT in eV; zero uses global zero-temperature occupations')
        parser.add_argument('--density-cutoff', type=float, default=None, help='PAW compensation/core Fourier cutoff in eV; default 16*wavefunction cutoff, converge separately')
        parser.add_argument('--atomic-units', choices=['codata','vasp_legacy'], default='codata', help='consistent length/energy constants; use vasp_legacy for matched VASP references')
        # vasp_cubic/vasp_spline remain accepted aliases; type= maps them before choices are checked.
        parser.add_argument('--projector-interpolation', type=canonical_projector_interpolation,
                            choices=['cubic','spline'], default=None,
                            help='PAW reciprocal projector interpolation; default cubic, or inherit frozen SCF for NSCF; use spline to match VASP NLSPLINE')
    for name in ('scf', 'bands', 'dos', 'relax'):
        parser = sub.choices[name]
        g = parser.add_argument_group('sparsification (each component has its own switch; see sparse_dft/sparsity.py)')
        g.add_argument('--sparsity', choices=['off', 'custom', 'auto'], default='off',
                       help='off: exact discretization (16xENCUT density, padding 2) with the certified solver defaults; custom: only the components given below; auto: probe-driven choice, explicit components override' + ('; relax resolves auto once on the starting geometry and keeps it for every step' if name == 'relax' else ''))
        g.add_argument('--density-cutoff-factor', type=float, help='S1 density support as a multiple of ENCUT, in [4, 24]; for more use --density-cutoff (eV)')
        g.add_argument('--xc-padding', type=int, choices=[1, 2, 3, 4], help='S2 XC quadrature padding per axis (code default 2)')
        g.add_argument('--eigen-search-factor', type=float, help='S3 inexact inner solve: eigen tolerance = factor * outer residual, in (0, 0.1]')
        g.add_argument('--block-ortho', type=int, choices=[0, 1], help='S4 block CGS2 orthogonalization of appended vectors')
        g.add_argument('--paw-compensation', choices=['full', 'blocks', 'off'], help='S5 scope of compensated PAW accumulation')
        g.add_argument('--projector-space', choices=['reciprocal', 'real-search'], help='S6 real-space projectors during inexact search only (FP64 reciprocal certificate unchanged); slower than reciprocal on every swept cell up to 64 atoms')
        g.add_argument('--projector-radius-factor', type=float, help='S6 real-space projector sphere as a multiple of the PAW radius, in [1, 2]')
        g.add_argument('--search-precision', choices=['fp64', 'fp32'], help='S7 working precision of inexact search (fp32 has no kernel in this build)')
        a = parser.add_argument_group('sparsity auto controls (only with --sparsity auto; S1 density support)')
        a.add_argument('--sparsity-level', choices=['conservative', 'balanced', 'aggressive'],
                       help='S1 tail budget: balanced = fitted rule (2.5e-7), conservative = 1/4 of it, aggressive = largest measured pass (4.25e-7)')
        a.add_argument('--sparsity-tail-budget', type=float, help='absolute S1 tail budget in (0, 1e-3]; replaces --sparsity-level')
        a.add_argument('--sparsity-candidates', type=float, nargs='+', metavar='F', help='S1 factors to try, in [4, 24); the probe measures each')
        a.add_argument('--sparsity-min-factor', type=float, help='smallest S1 factor auto may choose, in [4, 24]')
        a.add_argument('--sparsity-max-factor', type=float, help='largest S1 factor auto may choose (also caps the fallback), in [4, 24]')
        a.add_argument('--sparsity-metric', choices=['weighted_compensation_l2', 'shape_hartree'], help='probe tail metric; a non-default metric needs --sparsity-tail-budget')
        a.add_argument('--sparsity-rules', help='alternative sparsity rules JSON (schema sparse-dft-sparsity-rules-1)')
        s = parser.add_argument_group('solver controls (certificate-protected: change the route, not the converged result)')
        s.add_argument('--davidson-subspace-factor', type=int, help='Davidson search capacity as a multiple of bands, in [4, 16] (default 8)')
        s.add_argument('--search-operator-floor', type=float, help='below this eigen tolerance S6/S7 search operators are bypassed, in (0, 1e-2] (default 1e-6)')
        s.add_argument('--ritz-solver', choices=['auto', 'qr', 'divide_conquer', 'subset'], help='projected Ritz eigensolver (default auto)')
        s.add_argument('--incremental-ritz', type=int, choices=[0, 1], help='update the projected matrix incrementally (native default)')
        s.add_argument('--warm-guard-vectors', type=int, help='warm-start guard vectors, in [1, min(NPW, 2*bands)] (default 2*bands)')
        s.add_argument('--fast-density-grid', type=int, choices=[0, 1], help='round density FFT axes up to fast lengths (every mode kept)')
        s.add_argument('--scf-acceptance-margin', type=float, help='scale of the SCF acceptance thresholds, in (0, 1] (default 1)')
    for name in ('scf', 'bands', 'dos'):
        g = sub.choices[name].add_argument_group('process parallelism (needs a SDFT_ENABLE_MPI=ON build for ranks > 1)')
        g.add_argument('--mpi-ranks', type=int, default=1, help='MPI ranks; launcher from SDFT_MPI_LAUNCHER (default "mpiexec -n {ranks}")')
        g.add_argument('--kgroups', type=int, help='k-point groups; default ranks / bandgroups')
        g.add_argument('--bandgroups', type=int, default=1, help='ranks sharing one k point\'s Davidson block')
        g.add_argument('--threads-per-rank', type=int, help='OpenMP threads per rank; default OMP_NUM_THREADS (or cores) / ranks')
        g.add_argument('--deterministic', action='store_true', help='historical summation order and solver defaults (SDFT_DETERMINISTIC=1); slower, bitwise reproducible against earlier releases')
    for name in ('bands', 'dos'):
        sub.choices[name].add_argument('--scf-checkpoint', help='reuse a verified converged PAW scf.json; skip SCF recomputation')
    return p


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in _DELEGATED:
        import importlib
        return importlib.import_module(_DELEGATED[argv[0]][0]).main(argv[1:])
    a=build_parser().parse_args(argv)
    try:
        if a.cmd=='inspect': print(json.dumps(read_structure(a.structure).as_dict(),indent=2,default=str))
        elif a.cmd=='suggest':
            from .presets import suggest_file, suggested_argv
            payload = suggest_file(a.structure, a.task, a.paw_root, accuracy=a.accuracy, band_path_out=a.band_path_out)
            if a.argv:
                print(' '.join(suggested_argv(payload, paw_root=a.paw_root)))
            else:
                print(json.dumps(payload, indent=1, ensure_ascii=False, default=str))
        elif a.cmd=='recommend-kpoints': print(json.dumps(recommend_kpoints(read_structure(a.structure),a.task,a.accuracy,spacing_inverse_angstrom=a.spacing).as_dict(),indent=2))
        elif a.cmd=='make-band-path' and a.auto:
            from pathlib import Path
            from .symmetry import auto_band_path
            if a.vertices: raise ValueError('give either a vertices file or --auto STRUCTURE, not both')
            payload = auto_band_path(read_structure(a.auto), symprec=a.symprec, spacing=a.spacing,
                                     restrict_periodic=a.periodic_axes == 'auto', export_primitive=a.export_primitive)
            Path(a.out).write_text(json.dumps(payload, indent=1) + '\n', encoding='utf-8')
            print(json.dumps({'status':'PASS', 'kpoints':len(payload['kpoints']), 'segments':payload['segments'],
                              'spacegroup':payload['spacegroup_international'], 'symprec':payload['symprec'],
                              'warnings':payload['warnings'], 'output':str(a.out),
                              'primitive':payload.get('primitive_poscar')}, indent=2, ensure_ascii=False))
        elif a.cmd=='make-band-path':
            from pathlib import Path
            if not a.vertices: raise ValueError('make-band-path needs a vertices file or --auto STRUCTURE')
            vertices = json.loads(Path(a.vertices).read_text(encoding='utf-8'))
            path = interpolate_band_path(vertices, a.points_per_segment, include_duplicate_vertices=not a.no_duplicate_vertices)
            Path(a.out).write_text(json.dumps(path, indent=2) + '\n', encoding='utf-8')
            print(json.dumps({'status':'PASS', 'vertices':len(vertices), 'points_per_segment':a.points_per_segment, 'kpoints':len(path), 'output':str(a.out)}, indent=2))
        elif a.cmd=='paw-inventory': print(json.dumps(Paw54JsonArchive(a.root).inventory(),indent=2))
        elif a.cmd=='paw-convert':
            from .paw_convert import convert
            print(json.dumps(convert(a.source, a.out, overwrite=a.overwrite), indent=2))
        elif a.cmd=='paw-resolve':
            ar=Paw54JsonArchive(a.root); _,r=PawSelectionPolicy(a.family).resolve(ar,a.elements); print(json.dumps(r,indent=2))
        elif a.cmd=='prepare': print(json.dumps(prepare_job(a.structure,a.task,a.paw_root,a.out,a.accuracy),indent=2,default=str))
        elif a.cmd=='progress': print(json.dumps(progress_snapshot(),indent=2))
        elif a.cmd=='doctor': print(json.dumps(doctor(),indent=2))
        elif a.cmd=='backend-info': print(json.dumps(backend_info(),indent=2))
        elif a.cmd=='validate':
            import subprocess
            init_progress(); tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],text=True,capture_output=True)
            acceptance_path=__import__('pathlib').Path('workflow/final_acceptance.json')
            acceptance=json.loads(acceptance_path.read_text(encoding='utf-8')) if acceptance_path.exists() else {'status':'MISSING'}
            report={'status':'PASS' if tests.returncode==0 and acceptance.get('status')=='DONE' else 'FAIL','unit_tests_returncode':tests.returncode,'stdout':tests.stdout,'stderr':tests.stderr,'workflow':'workflow/progress.json','production_acceptance':acceptance.get('status'),'required':acceptance.get('required',{})}
            print(json.dumps(report,indent=2)); return 0 if report['status']=='PASS' else 3
        elif a.cmd=='mp-fetch':
            client=MaterialsProjectClient(a.api_key,a.api_root,timeout=a.timeout); url,rows=client.summaries(a.count); manifest=write_snapshot(a.out,rows,url,requested_count=a.count); print(json.dumps(manifest,indent=2)); return 0
        elif a.cmd=='mp-fetch-bands':
            client=MaterialsProjectClient(a.api_key,a.api_root,timeout=a.timeout); url,rows=client.bandstructures(a.ids,a.endpoint); manifest=write_band_snapshot(a.out,rows,url,requested_ids=a.ids); print(json.dumps(manifest,indent=2)); return 0
        elif a.cmd=='bands-compare':
            result=compare_bands(load_bands(a.reference),load_bands(a.candidate),align=a.align,max_energy_ev=a.max_energy_ev)
            text=json.dumps(result,indent=2)
            if a.out: from pathlib import Path; Path(a.out).write_text(text+'\n',encoding='utf-8')
            print(text); return 0
        elif a.cmd=='bands-analysis':
            result=analyze_bands(a.bands,dos_path=a.doscar,dos_scf_path=a.dos_scf,procar_path=a.procar,structure=a.structure,fermi_energy=a.fermi,gaussian_sigma=a.sigma,output_dir=a.out,render=not a.no_render,emin=a.emin,emax=a.emax)
            print(json.dumps(result,indent=2,default=str)); return 0
        elif a.cmd=='scf':
            result=run_scf(a.structure,a.paw_root,a.out,cutoff_ev=a.cutoff,accuracy=a.accuracy,kpoint=tuple(a.kpoint),max_iterations=a.max_iterations,density_tolerance=a.density_tolerance,onecenter_density_tolerance=a.onecenter_density_tolerance,mixing=a.mixing,bands=a.bands,
                scf_grid=a.scf_grid,mesh_mode=a.mesh_mode,smearing_ev=a.smearing_ev,atomic_units=a.atomic_units,density_cutoff_ev=a.density_cutoff,projector_interpolation=a.projector_interpolation,
                spin_channels=(a.spin_channels if a.cmd=='bands' else (a.spin_channels or 1)), initial_moments=a.initial_moments, hubbard=parse_cli_hubbard(a.hubbard), sparsity=_cli_sparsity(a), parallel=_cli_parallel(a), solver=_cli_solver(a))
            print(json.dumps(result,indent=2,default=str)); return 0 if result.get('converged') else 3
        elif a.cmd=='bands':
            path_payload=json.loads(__import__('pathlib').Path(a.path).read_text(encoding='utf-8'))
            path=path_payload['kpoints'] if isinstance(path_payload,dict) else path_payload
            labels=path_payload.get('labels') if isinstance(path_payload,dict) else None
            if labels is None and isinstance(path_payload,dict) and isinstance(path_payload.get('vertices'),dict):
                labels=[None]*len(path)
                for index,label in path_payload['vertices'].items():
                    index=int(index)
                    if not 0<=index<len(path): raise ValueError('band-path vertex index is out of range')
                    labels[index]=label
            result=run_bands(a.structure,a.paw_root,a.out,[tuple(float(x) for x in kp[:3]) for kp in path],cutoff_ev=a.cutoff,accuracy=a.accuracy,scf_kpoint=tuple(a.kpoint),max_iterations=a.max_iterations,density_tolerance=a.density_tolerance,onecenter_density_tolerance=a.onecenter_density_tolerance,mixing=a.mixing,bands=a.bands,
                scf_grid=a.scf_grid,mesh_mode=a.mesh_mode,smearing_ev=a.smearing_ev,atomic_units=a.atomic_units,density_cutoff_ev=a.density_cutoff,scf_checkpoint=a.scf_checkpoint,path_labels=labels,projector_interpolation=a.projector_interpolation,
                spin_channels=(a.spin_channels if a.cmd=='bands' else (a.spin_channels or 1)), initial_moments=a.initial_moments, hubbard=parse_cli_hubbard(a.hubbard), sparsity=_cli_sparsity(a), parallel=_cli_parallel(a), solver=_cli_solver(a))
            print(json.dumps(result,indent=2,default=str)); return 0 if result.get('metadata',{}).get('scf_converged') else 3
        elif a.cmd=='dos':
            result=run_dos(a.structure,a.paw_root,a.out,a.dos_grid,dos_mesh_mode=a.dos_mesh_mode,sigma_ev=a.dos_sigma,energy_step_ev=a.dos_step,emin_ev=a.dos_emin,emax_ev=a.dos_emax,
                cutoff_ev=a.cutoff,accuracy=a.accuracy,scf_kpoint=tuple(a.kpoint),max_iterations=a.max_iterations,density_tolerance=a.density_tolerance,onecenter_density_tolerance=a.onecenter_density_tolerance,mixing=a.mixing,bands=a.bands,
                scf_grid=a.scf_grid,mesh_mode=a.mesh_mode,smearing_ev=a.smearing_ev,atomic_units=a.atomic_units,density_cutoff_ev=a.density_cutoff,scf_checkpoint=a.scf_checkpoint,projector_interpolation=a.projector_interpolation,
                spin_channels=a.spin_channels, initial_moments=a.initial_moments, hubbard=parse_cli_hubbard(a.hubbard), sparsity=_cli_sparsity(a), parallel=_cli_parallel(a), solver=_cli_solver(a))
            meta=result['metadata']
            print(json.dumps({'dos_json':str(__import__('pathlib').Path(a.out)/'dos.json'),'fermi_energy_ev':result['fermi_energy_ev'],'nkpoints':meta['nkpoints'],'grid':meta['grid'],
                              'sigma_ev':meta['sigma_ev'],'energy_window_ev':[result['emin_ev'],result['emax_ev']],'complete_below_ev':meta['complete_below_ev'],
                              'pdos_atoms':len(result['pdos']),'pdos_orbitals':meta['pdos_orbitals'],'integrated_electrons':meta['integrated_electrons'],'integrated_to':meta['integrated_to']},indent=2)); return 0
        elif a.cmd=='relax':
            fn=cell_relax if a.cell else ionic_relax
            controls={'max_strain':a.max_strain,'stress_tolerance':a.stress_tolerance} if a.cell else {'max_displacement':a.max_displacement,'force_tolerance':a.force_tolerance}
            result=fn(a.structure,a.paw_root,a.out,max_steps=a.max_steps,cutoff_ev=a.cutoff,accuracy=a.accuracy,max_iterations=a.max_iterations,mixing=a.mixing,max_workers=a.max_workers,
                derivative_method=a.derivative_method,density_tolerance=a.density_tolerance,
                onecenter_density_tolerance=a.onecenter_density_tolerance,scf_grid=a.scf_grid,
                mesh_mode=a.mesh_mode,smearing_ev=a.smearing_ev,atomic_units=a.atomic_units,density_cutoff_ev=a.density_cutoff,projector_interpolation=a.projector_interpolation,spin_channels=a.spin_channels or 1,initial_moments=a.initial_moments,hubbard=parse_cli_hubbard(a.hubbard),
                **({'sparsity':sparsity} if (sparsity:=_cli_sparsity(a)).mode!='off' else {}),**({'solver':solver} if (solver:=_cli_solver(a)) else {}),**controls)
            print(json.dumps(result,indent=2,default=str)); return 0 if result.get('status')=='CONVERGED' else 3
    except Exception as e:
        payload=e.as_dict() if hasattr(e,'as_dict') else {'code':'RUNTIME_ERROR','message':str(e)}
        print(json.dumps(payload,indent=2),file=sys.stderr); return 2
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
