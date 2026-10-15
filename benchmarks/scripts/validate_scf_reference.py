"""Matched spin-unpolarized PAW SCF energy/eigenvalue/force/stress comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sparse_dft.bands import parse_vasprun_bands, parse_eigenval

def reference_occupations(bands,directory):
    """Prefer six-decimal EIGENVAL occupations over four-decimal XML values.

    Verify both files describe the same final spectrum and mesh. This is a
    source-precision fix, not a relaxation of the occupation acceptance gate.
    """
    path=directory/'EIGENVAL'
    if not path.exists():return next(iter(bands.occupations.values())),None
    eigenval=parse_eigenval(path)
    if eigenval.nkpoints!=bands.nkpoints or eigenval.nbands!=bands.nbands:
        raise ValueError('EIGENVAL dimensions disagree with final vasprun spectrum')
    original=next(iter(eigenval.occupations.values()))
    used=set();rows=[]
    for kp,energies in zip(bands.kpoints,bands.eigenvalues):
        hits=[i for i,point in enumerate(eigenval.kpoints) if i not in used and all(
            abs((a-b+.5)%1-.5)<1e-7 for a,b in zip(kp,point))]
        if len(hits)!=1:raise ValueError('EIGENVAL mesh disagrees with final vasprun spectrum')
        index=hits[0];used.add(index)
        # XML eigenvalues are also printed to four decimal places by VASP;
        # EIGENVAL uses six. Their maximum rounding discrepancy is 5.05e-5
        # eV. This file-consistency bound is separate from the physical gate.
        if max(abs(a-b) for a,b in zip(energies,eigenval.eigenvalues[index]))>5.1e-5:
            raise ValueError('EIGENVAL energies disagree with final vasprun spectrum')
        rows.append(original[index])
    return rows,path

def reference_convergence(root,calculation,directory):
    steps=calculation.findall('scstep')
    field=root.find(".//i[@name='NELM']")
    limit=int(field.text) if field is not None else None
    outcar=directory/'OUTCAR'
    stopped=outcar.exists() and 'aborting loop because EDIFF is reached' in outcar.read_text(errors='replace')
    # VASP often exits zero after exhausting NELM. An XML result and process
    # success alone are not convergence evidence. Missing OUTCAR permits only
    # the documented iteration-limit inference, never a NELM-exhausted run.
    below_limit=bool(steps) and limit is not None and len(steps)<limit
    return {'converged':bool(below_limit and (stopped or not outcar.exists())),
            'electronic_iterations':len(steps),'nelm':limit,'ediff_stop_observed':stopped,
            'evidence':'OUTCAR_EDIFF_stop' if outcar.exists() else 'XML_iteration_limit_inference'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',required=True);p.add_argument('--input',required=True)
    p.add_argument('--reference',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();candidate=Path(args.candidate);reference=Path(args.reference);inputfile=Path(args.input)
    d=json.loads(candidate.read_text());r=ET.parse(reference).getroot();calc=r.findall('calculation')[-1]
    bands=parse_vasprun_bands(reference)
    values=lambda name:[[float(x) for x in row.text.split()] for row in calc.findall(f"varray[@name='{name}']/v")]
    ref_force=values('forces');ref_stress=[[-v/1602.176634 for v in row] for row in values('stress')]
    ref_energy=float(calc.find("energy/i[@name='e_fr_energy']").text)
    reference_scf=reference_convergence(r,calc,reference.parent)
    native=d['scf_kpoints']
    # SCF meshes may be reordered and translated by integer reciprocal
    # vectors. Match exact periodic coordinates; never use this rule to
    # deform or interpolate a plotted high-symmetry path.
    mapping=[];used=set()
    for point in native:
        hits=[i for i,k in enumerate(bands.kpoints) if i not in used and all(
            abs((a-b+.5)%1-.5)<1e-7 for a,b in zip(point['fractional'],k))]
        if len(hits)!=1: mapping=[];break
        mapping.append(hits[0]);used.add(hits[0])
    matched_rows=[bands.eigenvalues[i] for i in mapping] if len(mapping)==len(native) else bands.eigenvalues
    raw=[a-b for point,ev in zip(native,matched_rows) for a,b in zip(point['eigenvalues_ev'],ev)]
    maximum=lambda a,b:max(abs(x-y) for row,rr in zip(a,b) for x,y in zip(row,rr))
    inputlines=inputfile.read_text().splitlines()
    cutoff=float(next(line.split()[1] for line in inputlines if line.startswith('cutoff_ev ')))
    ref_cutoff=float(r.find(".//i[@name='ENCUT']").text)
    ispin=int(r.find(".//i[@name='ISPIN']").text)
    reference_spline=r.find(".//i[@name='NLSPLINE']")
    reference_spline=reference_spline is not None and reference_spline.text.strip().lower() in ('t','true','1')
    bohr=.529177249 if d['atomic_units']=='vasp_legacy' else .529177210903
    ref_structure=r.find("structure[@name='finalpos']")
    ref_lattice=[[float(x) for x in row.text.split()] for row in ref_structure.findall("crystal/varray[@name='basis']/v")]
    native_lattice=[[v*bohr for v in row] for row in d['onecenter_lattice_bohr']]
    ref_positions=[[float(x) for x in row.text.split()] for row in ref_structure.findall("varray[@name='positions']/v")]
    native_positions=[state['fractional_position'] for state in d['onecenter_states']]
    files=[];seen=set()
    for line in inputlines:
        if not line.startswith('atom '):continue
        fields=line.split();symbol=fields[1]
        if symbol not in seen:
            files.append(Path(' '.join(fields[6:])).parent/'POTCAR');seen.add(symbol)
    paw_match=b''.join(path.read_bytes() for path in files)==(reference.parent/'POTCAR').read_bytes()
    conditions={'converged_complete_paw':d['converged'] and d['onecenter_closure_diagnostic'],
        'reference_scf_converged':reference_scf['converged'],
        'same_projector_interpolation':d.get('projector_interpolation','cubic').removeprefix('vasp_') ==
            ('spline' if reference_spline else 'cubic'),
        'same_paw_bytes':paw_match,'same_cutoff':abs(cutoff-ref_cutoff)<1e-8,'spin_unpolarized':ispin==1,
        'same_kpoints':len(native)==len(bands.kpoints)==len(mapping),
        'same_band_counts':len(native)==len(bands.eigenvalues) and all(len(point['eigenvalues_ev'])==len(row) for point,row in zip(native,matched_rows)),
        'same_lattice':maximum(native_lattice,ref_lattice)<1e-6,
        'same_fractional_atoms':len(native_positions)==len(ref_positions) and all(abs((x-y+.5)%1-.5)<1e-6 for row,rr in zip(native_positions,ref_positions) for x,y in zip(row,rr))}
    ref_weights=[bands.kweights[i] for i in mapping] if bands.kweights and len(mapping)==len(native) else bands.kweights
    conditions['same_weights']=ref_weights is not None and len(ref_weights)==len(native) and max(abs(point['weight']-w/sum(ref_weights)) for point,w in zip(native,ref_weights))<1e-7
    original_occ,occupation_file=reference_occupations(bands,reference.parent)
    ref_occ=[original_occ[i] for i in mapping] if len(mapping)==len(native) else original_occ
    occ_scale=2. if max(v for row in ref_occ for v in row)<=1.000001 else 1.
    occ_error=maximum([point['occupations'] for point in native],[[occ_scale*v for v in row] for row in ref_occ])
    conditions['equivalent_occupations']=occ_error<1e-5
    metrics={'total_energy_difference_ev':d['total_energy_ev']-ref_energy,
             'eigenvalue_mae_ev':sum(abs(x) for x in raw)/len(raw),'eigenvalue_maximum_ev':max(abs(x) for x in raw),
             'force_maximum_ev_per_angstrom':maximum(d['hf_total_forces_ev_per_angstrom'],ref_force),
             'stress_maximum_gpa':160.2176634*maximum(d['hf_total_stress_ev_per_angstrom3'],ref_stress),
             'occupation_maximum':occ_error}
    limits={'total_energy_ev':.002,'eigenvalue_maximum_ev':.005,'force_ev_per_angstrom':.002,'stress_gpa':.05}
    checks={'energy':abs(metrics['total_energy_difference_ev'])<limits['total_energy_ev'],
            'eigenvalues':metrics['eigenvalue_maximum_ev']<limits['eigenvalue_maximum_ev'],
            'forces':metrics['force_maximum_ev_per_angstrom']<limits['force_ev_per_angstrom'],
            'stress':metrics['stress_maximum_gpa']<limits['stress_gpa']}
    report={'status':'MATCHED_CASE_PASS' if all(conditions.values()) and all(checks.values()) else 'NOT_PASS',
        'metrics':metrics,'conditions':conditions,'checks':checks,'thresholds':limits,'reference_mesh_row_mapping':mapping,
        'reference_energy_ev':ref_energy,'reference_forces_ev_per_angstrom':ref_force,
        'reference_stress_ev_per_angstrom3':ref_stress,'candidate_iterations':d['iterations'],
        'reference_scf':reference_scf,
        'reference_occupation_source':str(occupation_file or reference),
        'scope':'matched case only; finite differences, cutoff/k-mesh convergence and material-wide claims require separate evidence',
        'provenance':{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (candidate,reference,inputfile,*files,*([occupation_file] if occupation_file else []))}}
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    return 0 if report['status']=='MATCHED_CASE_PASS' else 3

if __name__=='__main__':raise SystemExit(main())
