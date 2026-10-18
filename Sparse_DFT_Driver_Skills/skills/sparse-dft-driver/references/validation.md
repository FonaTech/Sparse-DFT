# Job completion, existing accuracy validation and recovery

Use identifiable existing project validation and the supplied release docs/ACCURACY.md; the old bundle’s archived validation declaration is not a new all-feature validation result. These instructions check whether the requested new job actually ran and delivered its outputs; they do not require repeating completed software accuracy benchmarks.

## Record distinct facts

| Fact | Evidence used here |
|---|---|
| Existing Sparse-DFT accuracy validation | User-confirmed or actual project validation record; reuse it |
| Source and runtime identity | Source/binary hashes, compiler/build information, actual backend |
| Process completion | Actual exit and logs; no reused stale output |
| Electronic convergence | Finite result, converged flag and requested smooth/onsite/eigen controls |
| Geometry completion | Actual accepted states and force/stress stopping conditions |
| Requested observable | Real raw fields or a saved, reproducible analysis formula |
| Research interpretation | Project claim, conditions and applicable model |

The supplied candidate stores some historical native validation fields as false. Preserve those raw fields for file identity and checkpoint compatibility. Record the actual external validation source separately with `research_bridge.py handoff --validation-basis artifact-backed --validation-record FILE`; use user-confirmed only when the current project provides that confirmation. Do not make old flags a requirement to rerun the author's tests or rewrite the original JSON to change them.

## Monitor actual artifacts

```bash
python3 "$DRIVER" monitor "$PROJECT/runs/sparse_dft/scf-001"
python3 "$DRIVER" monitor "$PROJECT/workflow/sparse-dft-driver/runs/ACTUAL_RUN_DIRECTORY"
```

Iteration JSONL is reported by complete parsed records, with the latest record. A partial final write is ignored and retained on disk. A run record with an exit reports process completion/failure; without exit it reports `UNKNOWN_PROCESS_LIVENESS`. Establish liveness from the actual host process/session or scheduler because a PID can be stale. The helper does not kill, schedule or start a second copy. Result files are marked `RESULT_PRESENT_REVIEW_REQUIRED`; inspect the actual requested result rather than inferring it from file presence.

## Return codes

- CLI SCF/bands: `0` when reported converged, `3` otherwise; `2` for caught input/runtime errors.
- CLI relax: `0` for `status=CONVERGED`, otherwise `3`; inspect optimizer checkpoint/result.
- Worker API: `FAILED/FAIL/UNSUPPORTED` map to `3`; pre-dispatch parameter errors map to `2`. `PREPARED` exits `0` but means input preparation.
- Native replay: `0` when its requested residual and normalization gates pass; `2` numerical failure, `1` input/runtime error. Preserve this tool's code semantics.
- Expert tools define their own codes. Read the requested output, not only a zero exit.

## Recover the affected stage

Keep failed JSON, process logs, inputs and settings. Missing binaries, archive records or source files require the actual dependency. For a new run that does not converge, examine smooth/onsite/eigen residuals, mixer, occupation and basis controls, then rerun the affected stage in a new directory. Keep scientific input changes visible and retain the previous result. This is run recovery, not a new global accuracy audit.

For analytic relaxation, `--resume` requires the same initial structure, constraints, PAW/archive, binaries and numerical controls. Sparse-DFT checks the compatible signature and last accepted-state hash. Increasing max steps is a supported continuation. New kernels, PAW or methods get new result directories.

Sparse-DFT stress is positive tension in eV/Å³; VASP OUTCAR uses positive compression. Fixed-index strain derivatives and current-cutoff basis changes are different operations; use the method appropriate to the requested response and record its basis policy. Symmetric shear changes two tensor entries, so preserve the implemented factor convention. Do not change a derivative definition while combining its data with another analysis.

## Optional diagnostics, only when requested or needed

Solver snapshots/replay are useful for an actual failed or analyzed H/S problem. They are not frozen SCF checkpoints. PWO1 and PWO2 represent different actual stored matrices; preserve their identity. Capture to a unique directory and replay the same operator. Retain the original requested tolerance while changing a stated iteration budget.

Backend/performance work uses actual per-run counters, device and timing. Record identical inputs, complete-run wall time and load conditions for a requested comparison. GPU availability is not actual dispatch, and nested stage timings do not sum into wall time. Reuse the user's completed precision basis; do not add an unsolicited new performance or precision campaign.

`sparse-dft validate` is an existing development command that checks tests plus an upstream final-acceptance file. It is available through the driver when explicitly requested. It is not a prerequisite for every case, and the minimal archive's missing historical acceptance file does not negate separately sourced validation.

## Report upstream gaps

The 1.0.0 release reports unmatched lm projection accuracy on GaAs and the Al slab; inspect docs/ACCURACY.md before quantitative PDOS claims. UPF is outside that comparison. S7 FP32 is unimplemented. GPU/MPI/MP credentials and historical benchmark data are separate runtime prerequisites. Never label an unrun route as passed.
