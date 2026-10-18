# Interoperation with FonaTech skills

Verified contracts are the `main` snapshots downloaded on 2026-10-12 from [DFT_Skills](https://github.com/FonaTech/DFT_Skills) and [Academic_Writing_Skills](https://github.com/FonaTech/Academic_Writing_Skills). Distribution provenance records archive and contract-file hashes. No commit SHA is asserted without verification. Future upstream changes require checking the contract again.

## Ownership and one-way handoff

`dft-workflow-orchestrator` owns the research decision, model ladder, smallest required scale, stage DAG and research spine. `sparse-dft-driver` owns actual Sparse-DFT inputs, calls, diagnostic artifacts and result exports. Writing skills own prose/citations/figures/manuscript builds. Scientific planning should not bounce recursively between these owners.

Choose Sparse-DFT explicitly in the upstream plan and engine capability matrix for the implemented periodic PBE lane with explicit spin/U model. Retain VASP or other-engine comparison as a separately declared independent stage. Existing upstream execution helpers are VASP-specific and **are not** Sparse-DFT runners. The new driver does not modify the user's installed DFT_Skills or rewire its existing VASP path automatically.

## New standalone project

The bridge creates a DFT_Skills-compatible schema **1.1** design packet and detailed integrated plan without needing DFT_Skills installed:

```bash
python3 /absolute/sparse-dft-driver/scripts/research_bridge.py init \
  --project /actual/project --system "actual material and state" \
  --claim "scientific decision to test" --observable "decisive observable and units" \
  --acceptance "actual job convergence and requested observable gate; reuse existing accuracy validation"
```

It writes the upstream 16-file control plane in `workflow/`: integrated plan, manifest, research contract/brief/spine, branch/data/next-action registers, decision log, assumptions/sources/claim matrix, handoff register, validation plan and risk register. It is **draft/planned** and records unresolved preflight inputs. Fill methodology, thresholds, provenance and budgets; reconcile the exact `Control Snapshot` fields, then mark current before launch. Do not run with a generic default plan just because its structure validates.

When available, validate against the actual upstream scripts:

```bash
python3 /actual/DFT_Skills/skills/dft-workflow-orchestrator/scripts/validate_experiment_manifest.py \
  --manifest /actual/project/workflow/experiment_manifest.json \
  --handoff-register /actual/project/workflow/handoff_register.csv
```

Use `--strict` after project choices are resolved and the plan is current. These are structural checks. The driver has no upstream runtime dependency in standalone operation.

## Existing linked project

Use existing claim/stage IDs; never reinitialize an existing workflow. Before executing, read its plan, spine and manifest. Add/reconcile a Sparse-DFT `kind=dft` stage with engine `Sparse-DFT`, actual dependencies, inputs, outputs, controls, validations, resource bounds and failure behavior. This is an additive engine selection, not a change to the upstream schema's allowed kind values.

Export an actual result:

```bash
python3 /absolute/sparse-dft-driver/scripts/research_bridge.py handoff \
  --result /actual/project/runs/sparse_dft/S01/scf_result.json \
  --out /actual/project/analysis/sparse_dft/S01-v1 \
  --stage-id S01 --entity "actual material and physical state" --validation-basis user-confirmed
```

The sidecar contains `handoff.json`, `stage-proposal.json`, `evidence.json` and `methods.md`. It hashes the immutable input and does not overwrite another manifest or existing handoff. Merge the proposal's execution metadata/status into the existing S01 while preserving its claim IDs, dependencies, controls, purpose and validation gates. Never replace the whole existing stage with the generic proposal. Update claim matrix, data lineage, spine, branch register, decision log, next-action queue and integrated plan. Revalidate the manifest.

Allowed upstream statuses are `planned`, `ready`, `submitted`, `running`, `converged`, `validated`, `failed`, `blocked`, `skipped`, `interpreted`. Use the completed external accuracy validation declared by the user; bridge numerical `converged` alone does not decide the research claim. A file without convergence evidence remains planned/review-required. Failed/non-finite/unconverged data remain failed. Do not map `PREPARED` to scientific `validated`.

## Academic evidence v2

`evidence.json` is a list of original raw-data records for recognized actual scalar output fields. Each includes:

- `key`, `source_kind=raw_data`, exact source value, unit, metric and entity;
- execution `conditions`, `accounting_scope`, `result_method=simulated`, `source_role=original`, `access_status=raw-files`;
- `provenance.artifacts` with absolute path and SHA-256, JSON Pointer locator, extraction method and recorded source environment;
- native numerical/physical validity and `verification=pending`.

Original calculations do not get fabricated paper/DOI/page fields. Those fields are needed for literature evidence, with Crossref checks and actual source-page quotes in the literature skill. Source hashes establish identity only; review the underlying field, parameter provenance, convergence and analysis before promoting writing evidence to verified.

For artificial test artifacts, use `--result-method synthetic`; fixtures are never recast as a physical simulation. When copying original results into a different manuscript project, copy immutable raw artifacts and update artifact paths/check hashes so they remain available. Do not edit the result to improve its flags.

The exporter covers actual SCF scalar fields; use a project analysis script for band gaps, force tensors, stress tensors, fits and error statistics. Derived evidence must declare `source_kind=derived`, formula, source evidence keys, code/environment, hashes and exact locators. Preserve units, sign, averaging, uncertainty and validity. Record existing accuracy validation separately through `--validation-basis user-confirmed` (or an actual artifact-backed record); do not rerun completed software precision work. Do not replace vectors/tensors with an unexplained scalar.

Bind manuscript paragraphs/captions/tables to explicit evidence keys under the current writing kit's `claims`/`cell_claims`. Build EN/ZH blocks from one source so numbers, citations, conditions and units remain paired. If writing skills are absent, deliver the raw export and methods with pending review; computational calls remain fully available.

## Upstream engine extension text

For a maintained DFT_Skills copy, add this semantic route to its engine matrix/task router without replacing existing engines:

> When Sparse-DFT is requested or selected for implemented periodic PAW PBE (nonmagnetic or collinear, optional Dudarev U), invoke `sparse-dft-driver` for source/version preflight, execution and result handoff. Record `engine: Sparse-DFT` in a `kind: dft` stage. Retain the integrated plan and research-spine gates. Do not use VASP execution helpers for this stage. For unsupported Sparse-DFT physics or absent data, return an explicit capability gap and choose another declared engine only with a justified scientific route. Standalone sparse-dft-driver performs the same computational gates without requiring the orchestrator.

Install both bundles in the agent's discoverable skill directory or explicitly load both by path. Standard agent skill loaders do not guarantee that the existing upstream skill will discover an arbitrary newly installed adapter automatically; the project routing instruction or explicit invocation establishes that connection. See the distribution's `INTEGRATION_AGENTS.md` for a ready-to-use project routing snippet.
