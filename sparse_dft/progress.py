"""Persistent execution control plane for unattended Sparse-DFT runs."""
from __future__ import annotations
from pathlib import Path
import datetime, json, platform, subprocess, hashlib
from .backend import backend_info

TASKS = [
    ("P0", "environment_audit"), ("P1", "paw_inventory"), ("P2", "paw_restore_and_parse"),
    ("P3", "structure_and_schema"), ("P4", "kpoints_and_cutoff"), ("P5", "plane_wave_scf"),
    ("P6", "energy_force_stress"), ("P7", "relax_and_restart"), ("P8", "bands_nscf"),
    ("P9", "agent_api_cli"), ("P10", "parallel_ci_release"),
]

def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()

def init_progress(root="workflow"):
    r=Path(root); r.mkdir(parents=True,exist_ok=True)
    p=r/'progress.json'
    if p.exists(): return json.loads(p.read_text(encoding='utf-8'))
    doc={"schema_version":"0.1","started_at":now(),"updated_at":now(),"status":"RUNNING",
         "tasks":[{"task_id":i,"name":name,"status":"PENDING","attempts":0,"commands":[],"tests":[],"artifacts":[],"blockers":[],"next_action":name} for i,name in TASKS],
         "current_task":"P0","completion_percent":0.0}
    p.write_text(json.dumps(doc,indent=2)+"\n", encoding='utf-8'); (r/'execution_log.jsonl').touch(); (r/'decision_log.md').write_text('# Sparse-DFT decision log\n\n', encoding='utf-8'); return doc

def update_task(task_id, status, *, tests=None, commands=None, artifacts=None, blockers=None, next_action=None, root="workflow"):
    doc=init_progress(root)
    for t in doc['tasks']:
        if t['task_id']==task_id:
            t['status']=status; t['attempts']+=1 if status in ('RUNNING','FAILED') else 0; t['updated_at']=now()
            if tests is not None: t['tests']=tests
            if commands is not None: t['commands']=commands
            if artifacts is not None: t['artifacts']=artifacts
            if blockers is not None: t['blockers']=blockers
            if next_action is not None: t['next_action']=next_action
            break
    passed=sum(t['status'] in ('DONE','PASSED') for t in doc['tasks']); doc['completion_percent']=round(100*passed/len(doc['tasks']),1); doc['updated_at']=now()
    if all(t['status']=='DONE' for t in doc['tasks']): doc['status']='DONE'
    Path(root,'progress.json').write_text(json.dumps(doc,indent=2)+"\n", encoding='utf-8')
    return doc

def log_event(event, root="workflow"):
    event=dict(event); event.setdefault('timestamp',now())
    with Path(root,'execution_log.jsonl').open('a', encoding='utf-8') as f: f.write(json.dumps(event,sort_keys=True,default=str)+"\n")

def doctor(root="workflow"):
    checks={"python":platform.python_version(),"platform":platform.platform(),"machine":platform.machine()}
    for cmd in [('c++','--version'),('cmake','--version')]:
        try: checks[cmd[0]]=subprocess.check_output(cmd,stderr=subprocess.STDOUT,text=True,timeout=10, encoding='utf-8', errors='replace').splitlines()[0]
        except Exception as e: checks[cmd[0]]=f"ERROR: {e}"
    checks['cwd']=str(Path.cwd()); checks['status']= 'PASS' if not any(str(v).startswith('ERROR') for v in checks.values()) else 'FAIL'
    checks['backends'] = backend_info()
    Path(root).mkdir(parents=True,exist_ok=True); Path(root,'doctor.json').write_text(json.dumps(checks,indent=2)+"\n", encoding='utf-8'); return checks
