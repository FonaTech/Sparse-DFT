from __future__ import annotations
from pathlib import Path
import json, hashlib, datetime
from .structure import read_structure
from .kpoints import recommend_kpoints
from .paw import Paw54JsonArchive, PawSelectionPolicy
from .progress import log_event
from .settings import recommended_cutoff

def _now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def _write_json(path,obj): Path(path).parent.mkdir(parents=True,exist_ok=True); Path(path).write_text(json.dumps(obj,indent=2,sort_keys=True,default=str)+"\n", encoding='utf-8')

def prepare_job(structure, task='scf', paw_root=None, output='job', accuracy='standard', kpoints=None, paw_family='standard'):
    out=Path(output); out.mkdir(parents=True,exist_ok=True)
    s=read_structure(structure) if isinstance(structure,(str,Path)) else structure
    s=s.normalized(); kp=kpoints or recommend_kpoints(s,task=task,accuracy=accuracy)
    if paw_root is None: raise ValueError('paw_root is required for a real PAW job')
    archive=Paw54JsonArchive(paw_root); choices,report=PawSelectionPolicy(paw_family).resolve(archive,s.species)
    paw_dir=out/'paw_cache'; paw_dir.mkdir(exist_ok=True)
    datasets={}
    for el,c in choices.items(): datasets[el]=archive.restore_dataset(c.record_id,paw_dir/el)
    resolved={'task':task,'accuracy':accuracy,'structure':s.as_dict(),'structure_hash':s.content_hash(),'kpoints':kp.as_dict(),'recommended_cutoff_ev':recommended_cutoff(choices),'scf_max_iterations':240,'nscf_path_min_points':2,'paw':report,'status':'PREPARED','created_at':_now()}
    manifest={'api_version':'0.1','job_id':hashlib.sha256((s.content_hash()+json.dumps(report,sort_keys=True)).encode()).hexdigest()[:16],'input_hash':s.content_hash(),'status':'PREPARED','created_at':_now(),'archive_hashes':archive.local_hashes()}
    _write_json(out/'resolved_input.json',resolved); _write_json(out/'paw_selection.json',report); _write_json(out/'manifest.json',manifest)
    log_event({'event':'job_prepared','job_id':manifest['job_id'],'directory':str(out),'status':'PREPARED','input_hash':manifest['input_hash']})
    return {'api_version':'0.1','request_id':manifest['job_id'],'status':'PREPARED','data':{'directory':str(out),'manifest':manifest},'warnings':s.warnings,'errors':[],'provenance':{'paw_archive':str(archive.root)}}

def progress_snapshot(root='workflow'):
    p=Path(root)/'progress.json'
    if not p.exists(): return {'status':'PENDING','tasks':[]}
    return json.loads(p.read_text(encoding='utf-8'))
