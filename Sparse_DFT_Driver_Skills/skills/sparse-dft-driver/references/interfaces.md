# Real interface contracts

Generate `catalog --out FILE` against the actual source. The bundled baseline enumerates 22 top-level CLI commands, 21 API branches, 377 public Python definitions (including analysis/plotting subpackages), 53 literal native targets and 11 source scripts. Discovery proves reachability only. Some public classes return runtime objects and need a project-local Python script rather than JSON serialization.

## CLI

CLI: inspect, suggest, recommend-kpoints, make-band-path, paw-inventory, paw-convert, paw-resolve, prepare, progress, doctor, backend-info, validate, mp-fetch, mp-fetch-bands, bands-compare, bands-analysis, scf, bands, dos, relax, plot and worker. `plot` and `worker` delegate to sparse_dft.plotting.cli and sparse_dft.runner; their own --help is authoritative. The wrapper preserves all arguments. Output flags are checked before execution; use unique result paths.

`suggest STRUCTURE --paw-root DIR --argv` prints recommended controls only. `make-band-path --auto STRUCTURE --out FILE` uses optional symmetry dependencies; explicitly supplied vertices remain available. `plot auto INPUT... --out DIR` draws from actual results. `worker JOB_JSON` runs the software's own supported job envelope; see runner.make_job and runner --help. It is distinct from the helper batch recipe format.

## API

Call sparse_dft.api.dispatch(method, params). It is a local dispatcher, not a registered MCP service. Its response has api_version, request_id, status, data, warnings, errors and provenance. API version remains 0.1. Job preparation may return PREPARED. The newline main_jsonrpc interface is not a full JSON-RPC 2.0 envelope.

The catalogue contains exact allowed/required keys, including parameters consumed by `_spin_controls`. Implemented routes: structure.parse/validate, paw.inventory/resolve/restore/parse, kpoints.recommend, backend.info, job.prepare/run, bands.run/compare/analyze, **dos.run**, relax.ionic/cell, mp.fetch_structures/bands. `job.status` is a placeholder mapped to unsupported; results.summarize/export raise unsupported.

job.run, bands.run and dos.run accept sparsity, parallel, spin_channels, initial_moments and hubbard. Nested profiles use the software schemas in sparsity.md. API relax passes spin/U but **does not forward sparsity, solver or parallel**. Use CLI relax or public Python for those controls. API job.run/bands.run do not forward family or solver; these keys are rejected instead of silently dropped. API dos.run does not expose arbitrary scf_kpoints/scf_weights. API bands.analyze omits CLI emin/emax. API kpoints.recommend omits spacing/metallic. Unknown API keys always fail before execution.

## Python

`python MODULE:PUBLIC_NAME --params FILE` accepts `{args: [...], kwargs: {...}}`, binds the live signature and serializes dataclasses, as_dict results and paths. Typed objects: `$structure`, `$bands`, `$paw_archive`, `$kpoint_recommendation`; use exact constructor fields for the latter. Sparsity and parallel accept ordinary nested dicts. Functions outside the top-level package are discoverable, including plotting/analysis. For sessions, instance methods, array objects and complex return types use a project-local script and recorded project-python recipe.

Examples:

```json
{"args":["/actual/POSCAR","/actual/paw","preview"],"kwargs":{"cutoff_ev":400,"controls":{"level":"balanced"}}}
```

Call sparse_dft.scf:preview_sparsity (preview controls are auto controls; it reports all levels). run_scf/run_bands/run_dos expose solver and family. run_scf exposes basis_reference_lattice_bohr, geometry_guess and save_geometry_guess. analytic_relax forwards numerical kwargs to SCF. PreparedPawSession is for repeated geometries in one process; create/use it within a project-local Python script. Do not serialize an opaque session as a successful calculation.

## Tools and native

`native --name sdft_solver_replay --bin-dir /actual/build -- SNAPSHOT OUTPUT` preserves native argv and codes. Targets must appear in the actual CMake catalogue. `tool --name scripts/install.py --reviewed -- --no-python --backend cpu` uses **relative source paths**, not obsolete tools/. Read [expert-tools.md](expert-tools.md) before historical benchmarks; omitted raw runs and data are genuine prerequisites.

The installed GUI can be launched by sparse-dft-gui or python -m sparse_dft_gui with its optional dependencies. Scientific functions remain callable without GUI automation. A headless test does not establish desktop interaction quality.
