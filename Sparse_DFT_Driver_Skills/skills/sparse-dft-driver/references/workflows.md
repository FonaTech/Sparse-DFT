# Operational recipes

These recipes are reproducible call patterns. Numerical settings are starting controls and require convergence studies for the actual material. Paths and environments must be real. All examples use a separate PROJECT and unique outputs.

## Structure → PAW → execution

1. Parse the supplied structure and inspect periodicity, lattice, species, Cartesian/fractional convention and selective dynamics. Ordinary XYZ can be parsed but has no lattice for periodic SCF. The simple CIF parser does not perform general symmetry expansion; use a fully expanded structure or the real symmetry API with its installed dependencies and record provenance.
2. Inspect local PAW inventory, selection family, record IDs, metadata ENMAX and archive hashes. `paw.restore/parse` produce local restored files/numeric caches; do not share licensed potential contents in the skill package.
3. Recommend sampling and record its rationale. `prepare` writes `resolved_input.json`, `paw_selection.json`, `manifest.json` and `paw_cache/`, but does not launch the recommendation as SCF.
4. Supply the mesh or explicit weighted points to the executable call. For equivalent references, match physical model, PAW, spin assumption, constant convention and projector interpolation.

```bash
python3 "$DRIVER" --source "$SDFT_SOURCE" --cwd "$PROJECT" cli -- \
  recommend-kpoints /actual/POSCAR --task scf --accuracy standard
python3 "$DRIVER" --source "$SDFT_SOURCE" --cwd "$PROJECT" cli -- \
  paw-resolve --root "$SDFT_PAW_ARCHIVE" --elements Si --family standard
python3 "$DRIVER" --source "$SDFT_SOURCE" --cwd "$PROJECT" \
  --env SDFT_ENABLE_PAW_CLOSURE_DIAGNOSTIC=1 --env OMP_NUM_THREADS=2 \
  --env VECLIB_MAXIMUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 cli -- \
  scf /actual/POSCAR --paw-root "$SDFT_PAW_ARCHIVE" --scf-grid 2 2 2 \
  --density-tolerance 1e-8 --onecenter-density-tolerance 1e-8 \
  --out runs/sparse_dft/scf-001
```

SCF output is `scf.json`, enriched `scf_result.json`, `scf_input.sdft`, `scf.process.json`, PAW caches and native sidecars such as `scf.json.iterations.jsonl` and `scf.json.performance.json`. Preserve the JSON result and the actual process exit even after failure. `internal_energy_ev` is E before entropy; `entropy_energy_ev` is −TS; `total_energy_ev` is F=E−TS at finite smearing and equals E when the entropy term is zero. There is no separate native `free_energy_ev` field.

## Numerical input contract

| Parameter | Candidate contract; not an accuracy claim |
|---|---|
| wavefunction `cutoff_ev` | finite 20–5000 eV; automatic `ceil_to_5(max(300, factor*max(ENMAX)))`, capped at 5000; factors 1.15/1.30/1.40/1.40 for low/standard/high/production; no ENMAX fallback 520 eV |
| density cutoff | optional independent support; explicit ≥4×wavefunction cutoff; C++ default 16×, converge separately; no inherited 5000 eV upper limit |
| iterations | integer 2–100000, SCF default 240; no convergence guarantee |
| smooth/onsite tolerances | finite in (0,1); API SCF/bands default 1e-6; API relax default 1e-8; CLI relax onsite default 1e-6, so set explicitly for matched calls |
| mixing | finite in (0,1], default 0.2 |
| SCF sampling | full `scf_grid` or explicit `scf_kpoints`+`scf_weights`; mutually exclusive; positive finite weights normalized by implementation; API/Python only for arbitrary weighted lists |
| mesh | `gamma` or `monkhorst-pack`; ≤4096 SCF points; default single Gamma when absent |
| path | explicit finite fractional points, 2–32768; labeled point array must match executed point count |
| smearing | Fermi–Dirac kBT in eV, finite nonnegative; zero uses global zero-temperature occupations; not Gaussian sigma |
| units/interpolation | `codata`/`vasp_legacy`; `cubic`/`spline` (legacy aliases accepted); keep matched throughout freeze and comparison |

`recommended_cutoff()` in job preparation uses the standard 1.30 factor; `automatic_cutoff()` in actual SCF uses the requested accuracy. Read `scf_input.sdft` and result provenance to establish executed controls. The bundled generic `schemas/input.schema.json` does not match every live CLI/API accuracy/control value and is not an executable request format.

## Converged SCF → frozen NSCF → analysis

Generate the actual high-symmetry vertices for the actual lattice/convention. Do not label a generic FCC path as valid for every structure. `make-band-path` interpolates supplied vertices; `--auto STRUCTURE` discovers symmetry with spglib/seekpath. Review primitive-cell mapping and periodic axes for slabs. Input can be an array or, for `bands`, a mapping with `kpoints` and same-length `labels` or `vertices` index-to-label map.

```bash
python3 "$DRIVER" --source "$SDFT_SOURCE" --cwd "$PROJECT" cli -- \
  make-band-path /actual/vertices.json --points-per-segment 81 --out runs/path-001.json
python3 "$DRIVER" --source "$SDFT_SOURCE" --cwd "$PROJECT" \
  --env SDFT_ENABLE_PAW_CLOSURE_DIAGNOSTIC=1 cli -- \
  bands /actual/POSCAR --paw-root "$SDFT_PAW_ARCHIVE" --path runs/path-001.json \
  --scf-checkpoint runs/sparse_dft/scf-001/scf.json --cutoff ACTUAL_SOURCE_CUTOFF \
  --atomic-units ACTUAL_SOURCE_UNITS --bands ACTUAL_BAND_COUNT --out runs/sparse_dft/bands-001
```

`ACTUAL_*` tokens must be replaced with values read from the SCF result, not passed literally. The Python path first resolves cutoff before freezing; omitting it can choose a different ENMAX default. Density cutoff and reference lattice are inherited with strict provenance checks; explicit mismatches are refused. PAW cache paths must still be reachable and match stored hashes. Recomputing SCF via `bands` without a checkpoint is available but must remain in a separate output.

Analyze a real Sparse-DFT JSON, EIGENVAL or `vasprun.xml`. Use DOSCAR/PROCAR only when actually available. `--fermi` supplies an explicit documented Fermi energy if the parser needs it. `--dos-scf` consumes **converged weighted uniform-grid** SCF spectra; review actual grid coverage and mesh convergence, not just file availability.

```bash
python3 "$DRIVER" --source "$SDFT_SOURCE" --cwd "$PROJECT" cli -- \
  bands-analysis runs/sparse_dft/bands-001/bands.json \
  --dos-scf runs/sparse_dft/dense-scf-001/scf.json --sigma 0.05 --out analysis/bands-001
```

`analysis.json` records source hashes, method, rendering backend and DOS limitations. Matplotlib yields PNG/PDF/SVG; minimal offline fallback yields stdlib SVG. CLI display `--emin/--emax` and Python render options are reachable even when API omits them. Compare both raw (`align=none`) and documented Fermi/VBM-aligned metrics. Coordinate geometry, reciprocal basis, dimensions, bands/spin channels and physical settings must match; interpolation does not make different paths equivalent.

## Geometry and derivatives

`relax` is ionic at fixed cell; `relax --cell` is cell-only at fixed fractional positions. Analytic default uses converged full-closure SCF, C++ L-BFGS direction and real-energy Armijo acceptance. Finite differences are an independent, more expensive route. Use explicit density and onsite tolerances; force threshold is eV/Å, stress is eV/Å³. Default force tolerance 0.02 eV/Å and stress 0.00031207545 eV/Å³ are optimizer stopping criteria.

```bash
python3 "$DRIVER" --source "$SDFT_SOURCE" --cwd "$PROJECT" \
  --env SDFT_ENABLE_PAW_CLOSURE_DIAGNOSTIC=1 cli -- \
  relax /actual/POSCAR --paw-root "$SDFT_PAW_ARCHIVE" --scf-grid 2 2 2 \
  --density-tolerance 1e-8 --onecenter-density-tolerance 1e-8 \
  --max-steps 20 --force-tolerance 0.02 --out runs/ionic-001
```

Reissue the identical compatible command with wrapper `--resume` to continue accepted states. Increasing `max_steps` is allowed by the optimizer; changed binaries, PAW, initial structure, constraints or numerical parameters require a new result directory. Inspect `relax_result.json`, checkpoint, accepted/trial SCF hashes and line-search history. For joint equilibrium, perform justified alternating cell/ionic steps with both stopping gates; do not describe a single cell-only call as joint optimization.

## External data and remaining derived quantities

`mp-fetch` obtains summaries/structures; `mp-fetch-bands` demands raw eigenvalues and rejects summary-only responses. Sparse-DFT uses `MP_API_KEY` or `MAPI_KEY`. Keep raw snapshots/manifests and material IDs. Use bands-compare with actual candidates and raw reference eigenvalues. The old tools/compare_mp_band_batch.py is absent from this release.

Equation-of-state curves, adsorption/defect energies, elastic fits or other quantities that can be derived from implemented SCF/derivative calls require an explicit physical method, reference states, configuration set and fitting/provenance script. They are agent-orchestrated derived workflows, not dedicated native Sparse-DFT commands. Energy/derivative validity restrictions travel with them. Unsupported native physics cannot be manufactured by renaming a SCF task.

## Collinear spin and Dudarev U

CLI SCF/bands/dos/relax expose --spin-channels 1/2, --initial-moments (one μB value per atom) and repeatable --hubbard ELEMENT:L:UEFF_EV. API/Python use hubbard as `{Fe: {l: 2, ueff_ev: 4.0}}` or per-atom specs. Values here illustrate shape, not a recommended U. Nonzero moments and U require two channels. The code validates valence bounds, matching elements and l. Separate U/J, SOC and noncollinear spin are unsupported. Use explicit model choices; a suggest recommendation does not authorize applying U. Frozen NSCF inherits omitted spin/U and rejects incompatible explicit values.

## Independent DOS, PDOS and plotting

`dos STRUCTURE --paw-root DIR --scf-checkpoint SCF/scf.json --dos-grid NX NY NZ --out DOS` runs uniform NSCF then Gaussian DOS, spin DOS and lm-resolved PAW PDOS. Match source cutoff, atomic units, SCF grid, projector interpolation and resolved sparsity profile. Supply enough bands and inspect complete_below_ev and integrated electrons; energy window/broadening do not establish spectral completeness. CLI uses --dos-sigma, --dos-step, --dos-emin, --dos-emax; API uses sigma_ev, energy_step_ev, emin_ev, emax_ev. Preserve dos.json metadata and projection limitations reported in upstream ACCURACY.md.

`plot auto SCF BANDS DOS --out FIGURES` uses actual input hashes. Consult `plot --help` and each plot subcommand for SCF, forces, stress, relax, DOS, PDOS, band comparisons, BZ, sparsity and atlas modes. Optional plotting/symmetry dependencies are separate from numerical core requirements.

## Parallel and certified solver controls

CLI SCF/bands/dos: --mpi-ranks, --kgroups, --bandgroups, --threads-per-rank, --deterministic. API/Python parallel dict uses ranks, kgroups, bandgroups, threads_per_rank, deterministic. kgroups*bandgroups=ranks; MPI>1 requires the MPI build and real launcher (SDFT_MPI_LAUNCHER). No MPI flags are wired into CLI relax; use supported SCF kwargs through Python for a suitable workflow.

CLI SCF/bands/dos/relax exposes --davidson-subspace-factor, --search-operator-floor, --ritz-solver, --incremental-ritz, --warm-guard-vectors, --fast-density-grid and --scf-acceptance-margin. Public Python takes solver dict. API omits solver. These choose a route and preserve the original certificate; they do not permit relaxing the requested acceptance tolerances. Prepared PAW sessions, geometry guesses and fixed reference-lattice basis are Python controls; inspect their live signatures and checkpoint provenance.
