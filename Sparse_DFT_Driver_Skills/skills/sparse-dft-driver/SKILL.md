---
name: sparse-dft-driver
description: >
  Drive Sparse-DFT through its real CLI, API, Python and native programs. Use for installation, PAW/POTCAR or PAW-UPF, SCF, density sparsification and spectrum probes, collinear spin and Dudarev U, frozen bands, DOS/PDOS, relaxation, plotting, parallel execution and reproducible result handoffs. Sparse-DFT 真实驱动、稀疏密度计算与验证。
---

# Sparse-DFT Driver

Operate the supplied **Sparse-DFT 1.0.0** software. Its Python namespace is `sparse_dft`, shell command `sparse-dft`, native binaries `sdft_*`, and runtime variables `SDFT_*`. Do not invent an old DFTX entry point or rename an incompatible binary. The skill is standalone; companion research/writing skills are optional.

## Resolve the real runtime

Use Python ≥3.11. Set actual paths, keep calculations outside this skill, and run the helper with that same Python:

```bash
export SDFT_SOURCE=/absolute/path/Sparse-DFT-1.0.0
export SDFT_SCF_EXE=/absolute/path/.install/bin/sdft_scf
export SDFT_PAW_ARCHIVE=/absolute/path/local-paw
python3 /absolute/path/sparse-dft-driver/scripts/sparse_dft_driver.py --source "$SDFT_SOURCE" doctor
```

`doctor` checks identity and files; `backend.info` probes compiled capabilities. Only run counters prove active GPU use. The release archive has no source manifest, binaries or PAW datasets. Build with `scripts/install.py`; preserve build logs. Do not inherit an archived declaration that every material is validated. Read upstream accuracy limitations and retain raw validity flags.

## Select the route

| Task | Read |
|---|---|
| Build, dependencies, CPU/Metal/CUDA/MPI, PAW selection | [runtime.md](references/runtime.md) |
| Exact commands, API keys, Python signatures, workers and native tools | [interfaces.md](references/interfaces.md), live `catalog` |
| Sparse-DFT S1–S6, probe, auto/custom/off, matched comparison | [sparsity.md](references/sparsity.md) |
| SCF, spin/U, bands, DOS/PDOS, symmetry, relaxation and plots | [workflows.md](references/workflows.md) |
| Convergence, restart, interpretation and evidence | [validation.md](references/validation.md) |
| Batch recipes and deterministic derived analyses | [recipe-runner.md](references/recipe-runner.md) |
| Research/writing handoff | [interop.md](references/interop.md) |
| Scientific design cases | [case-studies/INDEX.md](case-studies/INDEX.md) |

The executable helper forwards every real CLI argument. `api` rejects keys ignored by dispatch; `python` binds the live public signature, including controls omitted by API. Catalogue discovery is static reachability, distinct from tested functionality. Read the distribution validation report for actual tests and environmental gaps.

```bash
python3 "$DRIVER" --source "$SDFT_SOURCE" --cwd "$PROJECT" cli -- scf --help
python3 "$DRIVER" --source "$SDFT_SOURCE" --cwd "$PROJECT" api job.run --params /actual/scf.json
python3 "$DRIVER" --source "$SDFT_SOURCE" --cwd "$PROJECT" python sparse_dft.scf:preview_sparsity --params /actual/preview.json
```

Wrapper flags precede the mode. Software arguments follow `cli --`. JSON artifact paths resolve against PROJECT unless absolute. Tool names are source-relative catalogue keys such as `scripts/install.py`. Review a tool's code and dependencies before `tool --reviewed`; benchmark scripts can depend on omitted historical raw runs.

## Sparse-DFT specific invariants

- S1 density support and S2 XC padding change discretisation. The auto tail is a calibrated estimate, not a rigorous energy/force error bound. Compare actual matched off/auto results for the requested observable when assessing accuracy.
- S3–S6 change the search; accepted eigenpairs must pass the unchanged original-operator FP64 residual certificate. S6 `real-search` is implemented but no auto rule selects it. **S7 FP32 is unimplemented and must fail explicitly.**
- `off` uses density factor 16 and XC padding 2 when no explicit environment/absolute cutoff overrides exist. It still uses native block CGS2 and blocks compensation defaults. `--deterministic` selects historical summation/solver order; it is not synonymous with off.
- `preview_sparsity` runs the native spectrum probe and no SCF. SCF auto records `density_probe.json`, `sparsity_profile.json` and result `sparsity_ledger`. Retain choices, tail metric/budget, fallback, warnings, counters and residuals. A relaxation resolves auto once and freezes it across geometry steps.
- Bands/NSCF refuse `sparsity=auto`. Use `sparsity_workflow.py frozen-profile` to extract the **resolved custom** profile, then pass it with the source cutoff/units and checkpoint. Strict density, structure, PAW hash and model compatibility still apply.
- Spin channels 1/2 and collinear Dudarev U are implemented. Initial moments are per atom in μB; U requires two channels and `{l, ueff_ev}`. Preserve the frozen spin/U model. SOC, noncollinear magnetism, hybrid XC, AIMD, NEB and phonons have no implemented native workflow here.
- PAW UPF v2 scalar-relativistic PBE is an alternate data route. NC/USPP are refused. It is outside the supplied POTCAR accuracy comparison. DOS uses a uniform NSCF mesh; lm projections carry the upstream known GaAs/slab limitations. A path gap is not a full-Brillouin-zone gap.

## Execute and finish

Retain requested sampling and convergence tolerances. `prepare` and `suggest` do not run SCF. A zero process exit is not numerical convergence or a scientific accuracy certificate. Preserve `total_energy_valid`, force/stress and production flags; the upstream release explicitly leaves `total_energy_valid=false`.

New methods/inputs/binaries use new result directories. `--resume` permits reviewed compatible reuse while the software validates its checkpoints. Unique records under `workflow/sparse-dft-driver/runs` retain actual calls, identities, exit codes and logs. `monitor` reads completed JSONL records and result flags; a stored PID alone does not prove liveness. Do not overwrite another session's work.

`job.status` is an upstream placeholder; `results.summarize/export` are unsupported. Use `monitor` and `research_bridge.py handoff`. Every exported number must come from actual output or a reproducible derived formula; synthetic fixtures remain labelled synthetic. Finish with artifact paths, what was executed, passed checks and unresolved gaps.

In linked research mode preserve the orchestrator's claim/stage IDs, use `kind=dft, engine=Sparse-DFT`, and synchronize authority files. Never send Sparse-DFT inputs to VASP launchers. Writing skills receive traceable original-result evidence; literature DOI/page verification is a separate task.
