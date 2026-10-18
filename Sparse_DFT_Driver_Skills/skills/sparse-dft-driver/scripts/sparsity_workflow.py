#!/usr/bin/env python3
"""Prepare real matched SCFs, recover a frozen profile and compare measured results."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from source_catalog import sha256

COMPONENTS=('density_cutoff_factor','xc_padding','eigen_search_factor','block_ortho','paw_compensation','projector_space','projector_radius_factor','search_precision')


def load(path): return json.loads(Path(path).read_text())


def save_new(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(data,f,indent=2,allow_nan=False);f.write('\n')


def frozen_profile(result):
    d=load(result)
    if d.get('converged') is not True:raise ValueError('A converged SCF result is required')
    p=d.get('provenance',{}).get('sparsity')
    if not isinstance(p,dict):raise ValueError('Use enriched scf_result.json containing provenance.sparsity')
    # rule_ids/estimates describe auto history and must not trigger auto again.
    profile={'mode':'custom',**{k:p[k] for k in COMPONENTS if k != 'density_cutoff_factor' and p.get(k) is not None}}
    return {'source':str(Path(result).resolve()),'source_sha256':sha256(result),'sparsity':profile,
            'cutoff_ev':d['cutoff_ev'],'density_cutoff_ev':d['density_cutoff_ev'],
            'atomic_units':d['atomic_units'],'projector_interpolation':d['projector_interpolation'],
            'spin_channels':d['spin_channels'],'note':'Pass sparsity to run_bands/run_dos with the raw sibling scf.json checkpoint; omit S1 factor because NSCF inherits density support from the checkpoint; preserve source spin/U and SCF sampling.'}


def prepare(structure,settings,variants,project):
    structure=Path(structure).resolve();settings=load(settings);variants=load(variants);project=Path(project).resolve()
    if not structure.is_file():raise ValueError('Real structure file required')
    if not {'paw_root','cutoff_ev'} <= settings.keys():raise ValueError('settings requires actual paw_root and explicit cutoff_ev')
    if not any(k in settings for k in ('scf_grid','scf_kpoints')):raise ValueError('Execute an explicit SCF grid or weighted list')
    if any(k in settings for k in ('structure','output','sparsity')):raise ValueError('Structure/output/profile belong to the workflow, not common settings')
    if not isinstance(variants,dict) or variants.get('off')!={'mode':'off'}:raise ValueError('variants needs off: {mode: off} as its matched baseline')
    steps=[];points=[]
    for name,profile in variants.items():
        if not name or not name.replace('-','').replace('_','').isalnum():raise ValueError('Use simple unique variant names')
        out=project/'runs'/name
        steps.append({'id':name,'mode':'python','call':'sparse_dft.scf:run_scf','params':{'args':[str(structure),settings['paw_root'],str(out)],'kwargs':{**{k:v for k,v in settings.items() if k!='paw_root'},'sparsity':profile}},'outputs':[str(out/'scf_result.json')]})
        points.append({'id':name,'profile':profile,'result':str(out/'scf_result.json')})
    save_new(project/'sparsity-recipe.json',{'schema':'sparse-dft-recipe-1','environment':{'SDFT_ENABLE_PAW_CLOSURE_DIAGNOSTIC':'1'},'steps':steps})
    save_new(project/'sparsity-points.json',{'schema':'sparse-dft-sparsity-points-1','structure':str(structure),'structure_sha256':sha256(structure),'points':points})
    return {'prepared':len(steps),'recipe':str(project/'sparsity-recipe.json'),'calculation_executed':False}


def flatten(x):
    if isinstance(x,list):
        for v in x:yield from flatten(v)
    else:
        v=float(x)
        if not math.isfinite(v):raise ValueError('Nonfinite observable')
        yield v


def deviation(a,b):
    a=list(flatten(a));b=list(flatten(b))
    if not a or len(a)!=len(b):raise ValueError('Observable dimensions mismatch')
    return max(abs(x-y) for x,y in zip(a,b))


def compare(points,out,energy_gate=1.0,force_gate=1e-3,stress_gate=0.01):
    for value in (energy_gate,force_gate,stress_gate):
        if not math.isfinite(value) or value<=0:raise ValueError('Gates must be finite and positive')
    spec=load(points);entries={p['id']:p for p in spec['points']}
    if 'off' not in entries:raise ValueError('Missing off baseline')
    results={name:load(e['result']) for name,e in entries.items()};base=results['off']
    if base.get('provenance',{}).get('sparsity',{}).get('mode')!='off':raise ValueError('Baseline is not off')
    if float(base['density_cutoff_ev'])!=16*float(base['cutoff_ev']):raise ValueError('Baseline density support is not 16x')
    rows=[]
    for name,d in results.items():
        for k in ('structure_hash','cutoff_ev','atomic_units','projector_interpolation','spin_channels','electron_count','hubbard_specs'):
            if k not in d or d[k]!=base[k]:raise ValueError(f'{name}: unmatched or missing {k}')
        def sampling(obj):
            return [(k['fractional'], k['weight'], k.get('spin_channel',0), k.get('plane_wave_index_sha256')) for k in obj['scf_kpoints']]
        if sampling(d) != sampling(base):raise ValueError(f'{name}: unmatched executed sampling/basis')
        for key in ('archive_hashes','scf_grid','scf_kpoints_fractional','scf_weights','smearing_ev','hubbard','initial_moments','onecenter_density_tolerance'):
            if d.get('provenance',{}).get(key)!=base.get('provenance',{}).get(key):raise ValueError(f'{name}: unmatched {key}')
        if d.get('converged') is not True:raise ValueError(f'{name}: unconverged result')
        residual=float(d['eigensolver_max_residual_hartree'])
        if not math.isfinite(residual) or residual>1e-12:raise ValueError(f'{name}: failed original eigen residual certificate')
        n=len(d['hf_total_forces_ev_per_angstrom'])
        if n<1:raise ValueError('Atom normalization unavailable')
        de=abs(float(d['total_energy_ev'])-float(base['total_energy_ev']))*1000/n
        df=deviation(d['hf_total_forces_ev_per_angstrom'],base['hf_total_forces_ev_per_angstrom'])
        ds=deviation(d['hf_total_stress_ev_per_angstrom3'],base['hf_total_stress_ev_per_angstrom3'])*160.2176634
        if not math.isfinite(de):raise ValueError('Nonfinite energy')
        ledger=d.get('sparsity_ledger',{});c=ledger.get('counters',{})
        rows.append({'id':name,'energy_mev_per_atom':de,'force_max_ev_per_angstrom':df,'stress_max_gpa':ds,'eigen_residual_hartree':residual,'passed':de<=energy_gate and df<=force_gate and ds<=stress_gate,'sparsity_ledger':ledger,'raw_validity':{'total_energy_valid':d.get('total_energy_valid'),'force_provenance':d.get('force_provenance'),'stress_provenance':d.get('stress_provenance')},'source':entries[name]['result'],'sha256':sha256(entries[name]['result'])})
    report={'schema':'sparse-dft-matched-sparsity-comparison-1','result_method':'synthetic' if any('synthetic' in str(d.get('fixture','')).lower() or d.get('result_method')=='synthetic' for d in results.values()) else 'simulated','gates':{'energy_mev_per_atom':energy_gate,'force_ev_per_angstrom':force_gate,'stress_gpa':stress_gate},'rows':rows,'passed':all(x['passed'] for x in rows),'scope':'Matched sampled SCFs only; not an all-material accuracy certificate. Timing estimates are not measured speedups.'}
    save_new(out,report);return report


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    q=sub.add_parser('prepare');q.add_argument('--structure',required=True);q.add_argument('--settings',required=True);q.add_argument('--variants',required=True);q.add_argument('--project',required=True)
    q=sub.add_parser('frozen-profile');q.add_argument('--result',required=True);q.add_argument('--out',required=True)
    q=sub.add_parser('compare');q.add_argument('--points',required=True);q.add_argument('--out',required=True);q.add_argument('--energy-gate-mev-atom',type=float,default=1);q.add_argument('--force-gate',type=float,default=1e-3);q.add_argument('--stress-gate-gpa',type=float,default=.01)
    a=p.parse_args()
    try:
        if a.mode=='prepare':result=prepare(a.structure,a.settings,a.variants,a.project)
        elif a.mode=='frozen-profile':result=frozen_profile(a.result);save_new(a.out,result)
        else:result=compare(a.points,a.out,a.energy_gate_mev_atom,a.force_gate,a.stress_gate_gpa)
        print(json.dumps(result,indent=2,allow_nan=False));return 0 if result.get('passed',True) else 3
    except (OSError,ValueError,KeyError,TypeError) as exc:print(json.dumps({'status':'FAILED','message':str(exc)}));return 2


if __name__=='__main__':raise SystemExit(main())
