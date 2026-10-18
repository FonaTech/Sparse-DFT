#!/usr/bin/env python3
"""DFT_Skills schema 1.1 scaffolding and Academic_Writing_Skills v2 evidence."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
from pathlib import Path

from source_catalog import sha256

PLAN_SECTIONS = ['Research Decision And Scope', 'Control Snapshot', 'Scientific Rationale And Scale Decision', 'Evidence, Models, And Assumptions', 'Stage Roadmap', 'Stage Gates And Deliverables', 'Resource-Aware Execution Order', 'Risks, Negative Results, And Escalation', 'Research Spine Synchronization', 'Definition Of Done']
SNAPSHOT = [('objective', 'Objective'), ('active_claim_id', 'Active claim'), ('current_gate', 'Current gate'), ('next_action', 'Next action'), ('stop_rule', 'Stop rule')]


def write_new(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    with path.open('x') as file:
        file.write(text)


def csv_new(path, headers, rows=()):
    with Path(path).open('x', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader(); writer.writerows(rows)


def scaffold(root, system, claim, observable, acceptance):
    root = Path(root).expanduser().resolve()
    workflow = root / 'workflow'
    authority = ['experiment_manifest.json', 'integrated_research_plan.md', 'research_spine.md', 'research_contract.md', 'decision_brief.md', 'decision_log.md', 'assumptions.md', 'knowledge_sources.md', 'claim_matrix.csv', 'branch_register.csv', 'data_lineage.csv', 'next_action_queue.csv', 'handoff_register.csv', 'validation_plan.md', 'risk_register.md']
    if workflow.exists() and any((workflow / name).exists() for name in authority):
        raise ValueError('An existing workflow is authoritative. Use handoff to produce a sidecar and merge a reviewed stage; do not replace its manifest.')
    workflow.mkdir(parents=True, exist_ok=True)
    spine = {'objective': claim, 'active_claim_id': 'C01', 'current_gate': 'P0: verify Sparse-DFT inputs and validity domain', 'next_action': 'Resolve structure, local PAW, executable and observable acceptance controls', 'stop_rule': 'Stop when C01 is answered within the declared task-specific numerical bounds and existing validation basis', 'non_goals': ['Sparse-DFT-unimplemented physics and unvalidated downstream labels']}
    stage = {'id': 'S01', 'kind': 'dft', 'name': f'Sparse-DFT calculation for {system}', 'engine': 'Sparse-DFT', 'claim_ids': ['C01'], 'depends_on': [], 'inputs': ['provenance-tracked periodic structure', 'local PAW PBE.54 JSON archive', 'verified Sparse-DFT executable'], 'outputs': ['runs/sparse_dft/S01/scf_result.json', 'analysis/sparse_dft/S01/handoff.json'], 'controls': ['wavefunction cutoff and independent density cutoff', 'executed weighted SCF mesh', 'density, onsite and eigensolver tolerances', 'units, projector interpolation, PAW identities'], 'validation': ['finite converged results and full PAW closure', 'observable-specific numerical controls', 'project-provided completed Sparse-DFT accuracy validation', 'preserve false or unknown energy/derivative/production validity'], 'status': 'planned'}
    manifest = {'schema_version': '1.1', 'project': {'name': root.name, 'system': system, 'root': str(root)}, 'user_preferences': {'route': 'Sparse-DFT single-scale DFT', 'depth': 'design', 'deliverables': [observable], 'constraints': ['Accept project existing accuracy validation; require actual job convergence'], 'accepted_alternatives': []}, 'research_spine': spine, 'scale_decision': {'required_scales': ['electronic'], 'deferred_scales': ['finite-temperature-atomistic', 'atomistic-surrogate', 'extended-atomistic', 'continuum'], 'rationale': 'The requested observable is first tested at the electronic scale; extra scales require a scientific justification.', 'escalation_trigger': 'The observable or validity domain cannot be resolved with the implemented periodic PBE with supported collinear/Dudarev models model.'}, 'planning_artifact': {'path': 'workflow/integrated_research_plan.md', 'status': 'draft', 'required_sections': PLAN_SECTIONS, 'synchronizes': ['research_spine', 'claims', 'stages', 'handoffs', 'next_action_queue'], 'last_reviewed': ''}, 'claims': [{'id': 'C01', 'text': claim, 'observable': observable, 'pass_condition': acceptance}], 'stages': [stage], 'handoffs': [], 'resources': {'software': ['Sparse-DFT source and compatible C++ executable'], 'hardware': {}, 'licenses': ['User-provided locally licensed PAW dataset when applicable']}, 'metadata': {'created_by': 'sparse-dft-driver/research_bridge.py', 'created_at': dt.datetime.now(dt.timezone.utc).isoformat(), 'notes': 'Design scaffold only; no calculation has run. Reconcile project-specific methods, controls and thresholds before marking current.'}}
    stage['pass_condition'] = acceptance
    snapshot = '\n'.join(f'- {label}: {spine[key]}' for key, label in SNAPSHOT)
    bodies = [f'C01 tests whether {claim}\nSystem: {system}. Observable: {observable}. Acceptance: {acceptance}.', snapshot, 'Use the implemented periodic PAW PBE with explicit spin/U model lane only if it resolves C01. Review competing states and PAW choices. A second scale is deferred until the declared escalation trigger is met.', 'Structure, PAW files, binaries, hardware and convergence controls are unresolved preflight inputs. These omissions block calculation launch. Documentation reports implementation capability, not target-system accuracy. Literature attributions need independently verified DOI and source pages.', f'P0 verifies inputs and compatibility. S01 computes {observable} for C01, then completes requested task controls using the project existing accuracy basis. Analysis consumes the immutable S01 artifacts and retained validity flags.', f'S01 produces results in eV, Å and eV/Å³ as applicable. Acceptance: {acceptance}. Reject unconverged results or unsupported physics. Native validity fields remain raw source data; existing external accuracy validation is recorded separately.', 'Complete file and binary preflight, then a bounded pilot, then the requested parameter controls and result analysis. Determine memory and thread limits from the actual environment; do not launch an unbudgeted campaign.', 'Missing PAW or executable blocks execution. Nonconvergence yields a failed run with retained diagnostics. An out-of-domain model yields an explicit alternative-engine proposal, not a silent approximation. A negative or inconclusive C01 answer is a valid outcome.', 'The manifest, research_spine.md, claim_matrix.csv, branch register, decision log and next_action_queue.csv must agree before launch and after each gate. Handoffs carry hashes, locators, units, bounds and uncertainty.', f'Done means C01 has a traceable verdict using {observable}, meeting {acceptance}, or an explicit documented inconclusive/unsupported conclusion. No production certification follows from this draft scaffold.']
    plan = '# Integrated Sparse-DFT Research Plan\n\n- Plan status: draft\n\n' + '\n\n'.join(f'## {heading}\n\n{body}' for heading, body in zip(PLAN_SECTIONS, bodies)) + '\n'
    write_new(workflow / 'experiment_manifest.json', manifest)
    write_new(workflow / 'integrated_research_plan.md', plan)
    write_new(workflow / 'research_spine.md', '# Research Spine\n\n' + snapshot + '\n')
    for name, body in {'research_contract.md': f'C01: {claim}\nObservable: {observable}\nAcceptance: {acceptance}\n', 'decision_brief.md': 'Route: Sparse-DFT DFT. Depth: design. Resolve data, validity domain and resources before launch.\n', 'decision_log.md': 'Initial design: no new calculation has run; reuse the project completed Sparse-DFT accuracy validation.\n', 'assumptions.md': 'Open: structure identity, PAW archive, native executables, target-state physics, method-sensitive convergence and project accuracy-validation basis. Each must be resolved or excluded with a reason before launch.\n', 'knowledge_sources.md': 'Use supplied data, audited Sparse-DFT source, then verified scientific literature. No literature claims have been imported by this scaffold.\n', 'validation_plan.md': 'Resolve observable-specific acceptance thresholds, use the project completed accuracy validation, select the requested cutoff/density/k-mesh controls, then check actual convergence and outputs.\n', 'risk_register.md': 'Model-domain mismatch and missing task-specific numerical controls can prevent a scientific conclusion despite SCF convergence.\n'}.items():
        write_new(workflow / name, body)
    csv_new(workflow / 'claim_matrix.csv', ['claim_id', 'claim', 'observable', 'pass_condition', 'status', 'evidence'], [{'claim_id': 'C01', 'claim': claim, 'observable': observable, 'pass_condition': acceptance, 'status': 'planned', 'evidence': ''}])
    csv_new(workflow / 'branch_register.csv', ['branch_id', 'claim_id', 'purpose', 'status', 'kill_criterion', 'expected_information_gain'])
    csv_new(workflow / 'data_lineage.csv', ['artifact', 'sha256', 'parent', 'transform', 'stage_id'])
    csv_new(workflow / 'next_action_queue.csv', ['claim_id', 'priority', 'action', 'status'], [{'claim_id': 'C01', 'priority': 1, 'action': spine['next_action'], 'status': 'planned'}])
    csv_new(workflow / 'handoff_register.csv', ['handoff_id', 'from_stage', 'to_stage', 'artifact', 'mapping', 'units', 'validity', 'acceptance', 'quantity_type', 'schema', 'basis', 'voigt_order', 'sign_convention', 'state_variables', 'averaging_rule', 'uncertainty', 'status'])
    return {'status': 'SCAFFOLDED', 'calculation_executed': False, 'manifest': str(workflow / 'experiment_manifest.json')}


def finite(value):
    if isinstance(value, bool): return True
    if isinstance(value, (float, int)): return math.isfinite(value)
    if isinstance(value, dict): return all(finite(x) for x in value.values())
    if isinstance(value, list): return all(finite(x) for x in value)
    return True


def handoff(source, output, stage_id, entity, result_method, validation_basis='not-provided', validation_record=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    obj = json.loads(source.read_text())
    if not isinstance(obj, dict): raise ValueError('Result must be a JSON object')
    if output.exists(): raise ValueError('Use a new handoff directory; existing evidence is not overwritten')
    data = obj.get('data', obj) if isinstance(obj.get('data', obj), dict) else obj
    metadata = data.get('metadata', {})
    convergence = data.get('converged', metadata.get('scf_converged'))
    linked_source = None
    if convergence is None and metadata.get('sampling') == 'uniform_mesh' and data.get('source'):
        linked = Path(data['source']); linked = linked if linked.is_absolute() else source.parent / linked
        if not linked.is_file() or sha256(linked) != data.get('source_sha256'):
            convergence = False
        else:
            parent = json.loads(linked.read_text())
            convergence = parent.get('metadata', {}).get('scf_converged')
            linked_source = {'path': str(linked.resolve()), 'sha256': sha256(linked), 'role': 'verified DOS NSCF source'}
    if data.get('status') == 'CONVERGED': convergence = True
    if data.get('status') in ('SCF_FAILED', 'FAILED', 'MAX_STEPS', 'LINE_SEARCH_FAILED'): convergence = False
    values_finite = finite(data)
    finite_converged = convergence is True and values_finite and obj.get('status') not in ('FAILED', 'FAIL')
    state = 'converged' if finite_converged else 'failed' if convergence is False or not values_finite or obj.get('status') in ('FAILED', 'FAIL') else 'planned'
    flags = {key: data.get(key, metadata.get(key)) for key in ('total_energy_valid', 'onecenter_closure_diagnostic', 'production_validated')}
    flags.update(force_validated=data.get('force_provenance', {}).get('validated'), stress_validated=data.get('stress_provenance', {}).get('validated'))
    prefix = '/data' if data is not obj else ''
    provenance = data.get('provenance', {})
    scalar_fields = {'total_energy_ev': 'eV', 'internal_energy_ev': 'eV', 'entropy_energy_ev': 'eV', 'fermi_energy_ev': 'eV', 'density_residual': '', 'onecenter_density_residual': '', 'electron_count': 'electrons', 'integrated_electron_count': 'electrons', 'compensation_electron_count': 'electrons', 'cutoff_ev': 'eV', 'density_cutoff_ev': 'eV', 'eigensolver_max_residual_hartree': 'Ha', 'magnetic_moment_muB': 'muB', 'hubbard_energy_ev': 'eV', 'hubbard_potential_expectation_ev': 'eV', 'density_reciprocal_modes': 'modes'}
    records = []
    digest = sha256(source)
    for field, unit in scalar_fields.items():
        value = data.get(field)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            scope = 'per simulation cell; no atom normalization applied' if field in ('total_energy_ev', 'internal_energy_ev', 'entropy_energy_ev') else 'exact Sparse-DFT output field; definitions and intended-domain validation require review'
            if field == 'total_energy_ev': scope += '; finite-smearing F=E-TS; equals internal E when entropy term is zero'
            elif field == 'internal_energy_ev': scope += '; internal E before entropy term'
            elif field == 'entropy_energy_ev': scope += '; entropy contribution -TS added to internal E'
            records.append({'key': f'sparse_dft.{stage_id}.{field}', 'source_kind': 'raw_data', 'value': str(value), 'unit': unit, 'metric': field, 'entity': entity, 'conditions': provenance, 'accounting_scope': scope, 'result_method': result_method, 'source_role': 'original', 'access_status': 'raw-files', 'verification': 'pending', 'numerical_status': {'converged': convergence, 'finite': values_finite, **flags}, 'provenance': {'artifacts': [{'path': str(source), 'sha256': digest}], 'locator': f'JSON Pointer {prefix}/{field}', 'method': 'Exact Sparse-DFT JSON field extraction; no new calculation and no physical-certification claim', 'environment': {'engine': 'Sparse-DFT', 'source_provenance': provenance}}})
    report = {'schema': 'sparse-dft-research-handoff-1', 'source': {'path': str(source), 'sha256': digest}, 'stage_id': stage_id, 'state': state, 'evidence_result_method': result_method, 'numerical_convergence': convergence, 'finite': values_finite, 'validity': flags, 'scientific_verdict': 'REVIEW_REQUIRED', 'production_validated': False, 'units': {'energy': 'eV/cell unless output explicitly states otherwise', 'force': 'eV/angstrom', 'stress': 'eV/angstrom^3'}, 'stress_convention': 'positive tension; reverse VASP OUTCAR compression sign before comparison', 'source_conditions': provenance, 'uncertainty': 'not estimated by this exporter', 'next_action': 'Review convergence, full PAW closure, observable-specific scans and independent reference; preserve restrictions in every downstream claim.'}
    stage = {'id': stage_id, 'kind': 'dft', 'name': f'Sparse-DFT result for {entity}', 'engine': 'Sparse-DFT', 'depends_on': [], 'inputs': [str(source)], 'outputs': [str(output / 'handoff.json'), str(output / 'evidence.json')], 'controls': ['Source provenance; no parameter inferred by exporter'], 'validation': ['Numerical and physical gates are separate; see handoff.json'], 'status': state, 'execution_metadata': {'source_sha256': digest, 'sdft_validity': flags, 'driver_scientific_verdict': 'REVIEW_REQUIRED'}}
    external_validation = {'basis': validation_basis, 'scope': 'Project-provided existing Sparse-DFT accuracy validation; distinct from raw source flags', 'revalidated_by_exporter': False}
    if validation_basis == 'artifact-backed':
        if not validation_record or not Path(validation_record).is_file(): raise ValueError('artifact-backed requires an actual --validation-record file')
        external_validation['artifact'] = {'path': str(Path(validation_record).resolve()), 'sha256': sha256(validation_record)}
    report['sparsity_ledger'] = data.get('sparsity_ledger')
    if linked_source: report['linked_source'] = linked_source
    report['external_accuracy_validation'] = external_validation
    stage['execution_metadata']['external_accuracy_validation'] = external_validation
    for record in records:
        record['external_accuracy_validation'] = external_validation
    report['next_action'] = 'Review this job\'s actual convergence, requested observables and evidence extraction; use the project\'s completed Sparse-DFT accuracy validation.'
    output.mkdir(parents=True)
    write_new(output / 'handoff.json', report)
    write_new(output / 'evidence.json', records)
    write_new(output / 'stage-proposal.json', stage)
    write_new(output / 'methods.md', f'# Sparse-DFT methods handoff\n\nSource: {source}\nSHA-256: {digest}\nEntity: {entity}\nEvidence method: {result_method}\nStage: {stage_id}\nNumerical state: {state}\n\nThe source reports the following execution provenance (missing fields remain unknown):\n\n```json\n{json.dumps(provenance, ensure_ascii=False, indent=2)}\n```\n\nValidity restrictions:\n\n```json\n{json.dumps(flags, indent=2)}\n```\n\nCheck raw outputs, actual mesh/weights, PAW identity, binary identity and convergence scans before writing a scientific claim. Literature sources need separate Crossref/page verification. This export does not add citations or fabricate uncertainty.\n')
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='mode', required=True)
    q = sub.add_parser('init'); q.add_argument('--project', type=Path, required=True); q.add_argument('--system', required=True); q.add_argument('--claim', required=True); q.add_argument('--observable', required=True); q.add_argument('--acceptance', required=True)
    q = sub.add_parser('handoff'); q.add_argument('--result', type=Path, required=True); q.add_argument('--out', type=Path, required=True); q.add_argument('--stage-id', required=True); q.add_argument('--entity', required=True); q.add_argument('--result-method', choices=['simulated', 'synthetic'], default='simulated')
    q.add_argument('--validation-basis', choices=['not-provided', 'user-confirmed', 'artifact-backed'], default='not-provided', help='record existing external accuracy validation without changing raw result flags')
    q.add_argument('--validation-record', type=Path, help='actual existing validation artifact when basis=artifact-backed')
    a = p.parse_args(argv)
    try:
        result = scaffold(a.project, a.system, a.claim, a.observable, a.acceptance) if a.mode == 'init' else handoff(a.result, a.out, a.stage_id, a.entity, a.result_method, a.validation_basis, a.validation_record)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)); return 0
    except (OSError, ValueError, KeyError) as exc:
        print(json.dumps({'status': 'FAILED', 'message': str(exc)})); return 2


if __name__ == '__main__':
    raise SystemExit(main())
