"""Versioned Agent-facing API dispatcher."""
from __future__ import annotations
import uuid, json
from .errors import SparseDFTError, UnsupportedError
from .structure import read_structure
from .kpoints import recommend_kpoints
from .paw import Paw54JsonArchive, PawSelectionPolicy
from .workflow import prepare_job
from .bands import load_bands, compare_bands
from .analysis import analyze_bands
from .mp_data import MaterialsProjectClient, write_snapshot, write_band_snapshot
from .scf import run_scf, run_bands, run_dos
from .relax import ionic_relax, cell_relax
from .backend import backend_info

API_VERSION='0.1'


def _spin_controls(params, *, inherit=False):
    # NSCF inherits an omitted model from its verified checkpoint.  SCF and
    # relaxation default to one channel, but explicit values must reach the
    # common model validator unchanged rather than silently being discarded.
    return {'spin_channels': params.get('spin_channels', None if inherit else 1),
            'initial_moments': params.get('initial_moments'),
            'hubbard': params.get('hubbard')}


def _response(status, data=None, warnings=None, errors=None, provenance=None, request_id=None):
    return {'api_version':API_VERSION,'request_id':request_id or str(uuid.uuid4()),'status':status,'data':data or {},'warnings':warnings or [],'errors':errors or [],'provenance':provenance or {}}

def dispatch(method, params):
    req=str(uuid.uuid4())
    try:
        if method=='structure.parse':
            s=read_structure(params['path']); return _response('PASS',s.as_dict(),s.warnings,request_id=req)
        if method=='structure.validate':
            s=read_structure(params['path']); return _response('PASS',{'natoms':s.natoms,'volume':s.volume,'hash':s.content_hash()},s.warnings,request_id=req)
        if method=='paw.inventory':
            a=Paw54JsonArchive(params['root']); return _response('PASS',a.inventory(),request_id=req,provenance={'root':str(a.root),'hashes':a.local_hashes()})
        if method=='paw.resolve':
            a=Paw54JsonArchive(params['root']); _, report=PawSelectionPolicy(params.get('family','standard')).resolve(a,params['elements']); return _response('PASS',{'selections':report},request_id=req,provenance={'root':str(a.root)})
        if method=='paw.restore':
            a=Paw54JsonArchive(params['root']); p=a.restore(params['record_id'],params.get('kind','POTCAR'),params['destination']); return _response('PASS',{'path':str(p)},request_id=req,provenance={'root':str(a.root)})
        if method=='paw.parse':
            a=Paw54JsonArchive(params['root']); ds=a.restore_dataset(params['record_id'],params['destination']); numeric=ds.numeric_data
            summary={'main_grid_size':numeric.main_grid_size,'radial_points':numeric.radial.npoints,'projector_channels':len(numeric.nonlocal_channels),'projector_count':sum(c.nprojectors for c in numeric.nonlocal_channels),'binary_cache':str(__import__('pathlib').Path(params['destination'])/'numeric.bin')}
            return _response('PASS',{'metadata':ds.metadata,'header':ds.parsed_header,'numeric_summary':summary},request_id=req)
        if method=='kpoints.recommend':
            s=read_structure(params['structure']); return _response('PASS',recommend_kpoints(s,params.get('task','scf'),params.get('accuracy','standard')).as_dict(),s.warnings,request_id=req)
        if method=='backend.info':
            return _response('PASS', backend_info(), request_id=req)
        if method=='bands.compare':
            result=compare_bands(load_bands(params['reference']),load_bands(params['candidate']),align=params.get('align','auto'),max_energy_ev=float(params.get('max_energy_ev',10.0)))
            return _response('PASS',result,request_id=req)
        if method=='bands.analyze':
            result=analyze_bands(params['bands'],dos_path=params.get('doscar'),dos_scf_path=params.get('dos_scf'),procar_path=params.get('procar'),structure=params.get('structure'),fermi_energy=params.get('fermi_energy_ev'),gaussian_sigma=float(params.get('sigma_ev',0.05)),output_dir=params.get('output','band_analysis'),render=bool(params.get('render',True)))
            return _response('PASS',result,request_id=req,provenance=result.get('provenance'))
        if method=='mp.fetch_structures':
            client=MaterialsProjectClient(params.get('api_key'),params.get('api_root','https://api.materialsproject.org'),timeout=params.get('timeout'))
            url,rows=client.summaries(int(params.get('count',100)),params.get('criteria')); manifest=write_snapshot(params['output'],rows,url,requested_count=int(params.get('count',100)))
            return _response('PASS',manifest,request_id=req,provenance={'source_url':url})
        if method=='mp.fetch_bands':
            ids=[str(x) for x in params.get('material_ids', [])]
            client=MaterialsProjectClient(params.get('api_key'),params.get('api_root','https://api.materialsproject.org'),timeout=params.get('timeout'))
            url,rows=client.bandstructures(ids,params.get('endpoint','/materials/bandstructure/'))
            manifest=write_band_snapshot(params['output'],rows,url,requested_ids=ids)
            return _response('PASS',manifest,request_id=req,provenance={'source_url':url,'raw_eigenvalues':True})
        if method=='job.prepare':
            return prepare_job(params['structure'],params.get('task','scf'),params['paw_root'],params.get('output','job'),params.get('accuracy','standard'))
        if method=='job.run':
            result=run_scf(params['structure'],params['paw_root'],params.get('output','job'),cutoff_ev=params.get('cutoff_ev'),accuracy=params.get('accuracy','standard'),kpoint=tuple(params.get('kpoint',(0.0,0.0,0.0))),max_iterations=int(params.get('max_iterations',240)),density_tolerance=float(params.get('density_tolerance',1e-6)),onecenter_density_tolerance=float(params.get('onecenter_density_tolerance',1e-6)),mixing=float(params.get('mixing',0.2)),bands=int(params.get('bands',0)),scf_grid=params.get('scf_grid'),scf_kpoints=params.get('scf_kpoints'),scf_weights=params.get('scf_weights'),mesh_mode=params.get('mesh_mode','gamma'),smearing_ev=float(params.get('smearing_ev',0.0)),atomic_units=params.get('atomic_units','codata'),density_cutoff_ev=params.get('density_cutoff_ev'),projector_interpolation=params.get('projector_interpolation'),sparsity=params.get('sparsity'),parallel=params.get('parallel'),**_spin_controls(params))
            return _response('PASS' if result.get('converged') else 'FAILED',result,request_id=req)
        if method=='bands.run':
            result=run_bands(params['structure'],params['paw_root'],params.get('output','job'),[tuple(float(x) for x in kp[:3]) for kp in params['kpoints']],cutoff_ev=params.get('cutoff_ev'),accuracy=params.get('accuracy','standard'),scf_kpoint=tuple(params.get('kpoint',(0.0,0.0,0.0))),max_iterations=int(params.get('max_iterations',240)),density_tolerance=float(params.get('density_tolerance',1e-6)),onecenter_density_tolerance=float(params.get('onecenter_density_tolerance',1e-6)),mixing=float(params.get('mixing',0.2)),bands=int(params.get('bands',0)),scf_grid=params.get('scf_grid'),scf_kpoints=params.get('scf_kpoints'),scf_weights=params.get('scf_weights'),mesh_mode=params.get('mesh_mode','gamma'),smearing_ev=float(params.get('smearing_ev',0.0)),atomic_units=params.get('atomic_units','codata'),density_cutoff_ev=params.get('density_cutoff_ev'),scf_checkpoint=params.get('scf_checkpoint'),path_labels=params.get('labels'),projector_interpolation=params.get('projector_interpolation'),sparsity=params.get('sparsity'),parallel=params.get('parallel'),**_spin_controls(params,inherit=True))
            return _response('PASS' if result.get('metadata',{}).get('scf_converged') else 'FAILED',result,request_id=req)
        if method=='dos.run':
            result=run_dos(params['structure'],params['paw_root'],params.get('output','dos'),params['dos_grid'],dos_mesh_mode=params.get('dos_mesh_mode','gamma'),sigma_ev=float(params.get('sigma_ev',0.05)),energy_step_ev=float(params.get('energy_step_ev',0.005)),emin_ev=params.get('emin_ev'),emax_ev=params.get('emax_ev'),
                cutoff_ev=params.get('cutoff_ev'),accuracy=params.get('accuracy','standard'),scf_kpoint=tuple(params.get('kpoint',(0.0,0.0,0.0))),max_iterations=int(params.get('max_iterations',240)),density_tolerance=float(params.get('density_tolerance',1e-6)),onecenter_density_tolerance=float(params.get('onecenter_density_tolerance',1e-6)),mixing=float(params.get('mixing',0.2)),bands=int(params.get('bands',0)),scf_grid=params.get('scf_grid'),mesh_mode=params.get('mesh_mode','gamma'),smearing_ev=float(params.get('smearing_ev',0.0)),atomic_units=params.get('atomic_units','codata'),density_cutoff_ev=params.get('density_cutoff_ev'),scf_checkpoint=params.get('scf_checkpoint'),projector_interpolation=params.get('projector_interpolation'),sparsity=params.get('sparsity'),parallel=params.get('parallel'),**_spin_controls(params,inherit=True))
            return _response('PASS',result,request_id=req)
        if method in ('relax.ionic','relax.cell'):
            fn=ionic_relax if method.endswith('ionic') else cell_relax
            controls={'max_strain':float(params.get('max_strain',0.02)),'stress_tolerance':float(params.get('stress_tolerance',.00031207545))} if method.endswith('cell') else {'max_displacement':float(params.get('max_displacement',0.15)),'force_tolerance':float(params.get('force_tolerance',.02))}
            result=fn(params['structure'],params['paw_root'],params.get('output','relax'),max_steps=int(params.get('max_steps',10)),cutoff_ev=params.get('cutoff_ev'),accuracy=params.get('accuracy','standard'),max_iterations=int(params.get('max_iterations',240)),mixing=float(params.get('mixing',0.2)),max_workers=int(params.get('max_workers',1)),
                derivative_method=params.get('derivative_method','analytic'),density_tolerance=float(params.get('density_tolerance',1e-8)),
                onecenter_density_tolerance=float(params.get('onecenter_density_tolerance',1e-8)),
                scf_grid=params.get('scf_grid'),scf_kpoints=params.get('scf_kpoints'),scf_weights=params.get('scf_weights'),
                mesh_mode=params.get('mesh_mode','gamma'),smearing_ev=float(params.get('smearing_ev',0.0)),atomic_units=params.get('atomic_units','codata'),density_cutoff_ev=params.get('density_cutoff_ev'),projector_interpolation=params.get('projector_interpolation'),bands=int(params.get('bands',0)),**_spin_controls(params),**controls)
            return _response('PASS' if result.get('status')=='CONVERGED' else 'FAILED',result,request_id=req)
        if method in ('results.summarize','results.export'):
            raise UnsupportedError(f'{method} requires a result artifact produced by job.run or bands.run')
        if method=='job.status': return _response('PASS',{'job_id':params.get('job_id'),'status':'UNKNOWN_UNTIL_PREPARED'},request_id=req)
        raise SparseDFTError(f'unknown API method: {method}',details={'method':method})
    except Exception as e:
        return _response('FAILED',errors=[e.as_dict() if isinstance(e,SparseDFTError) else {'code':'RUNTIME_ERROR','message':str(e)}],request_id=req)

def main_jsonrpc():
    import sys
    for line in sys.stdin:
        if not line.strip(): continue
        req=json.loads(line); print(json.dumps(dispatch(req['method'],req.get('params',{})),default=str),flush=True)
