# Density sparsification and certificate protected search

Source authority: sparse_dft/sparsity.py, sparse_dft/sparsity_rules.json, sparse_dft/scf.py and docs/SPARSIFICATION.md in the supplied 1.0.0 archive. No downloaded or invented rules are needed.

| Component | Real control | Meaning |
|---|---|---|
| S1 | density_cutoff_factor in [4,24] | density Fourier support; absolute density_cutoff_ev is independent |
| S2 | xc_padding 1/2/3/4 | XC quadrature padding per axis |
| S3 | eigen_search_factor in (0,0.1] | inexact eigen search relative to outer residual |
| S4 | block_ortho bool | block two-pass CGS |
| S5 | paw_compensation full/blocks/off | compensated PAW summation scope |
| S6 | projector_space reciprocal/real-search, projector_radius_factor [1,2] | real-space projectors during search; final reciprocal certificate |
| S7 | search_precision fp64/fp32 | only fp64 implemented; fp32 raises UNSUPPORTED |

## JSON profile

```json
{"mode":"auto","auto_controls":{"level":"balanced","candidates":[4,5,6,8,10,12,16]}}
```

Levels: conservative, balanced and aggressive. Controls: level, tail_budget, candidates, min_factor, max_factor, metric, rules. Candidates are factors in [4,24), because the probe ends at 24. Nondefault shape_hartree metric requires explicit tail_budget. Missing candidate measurements fail. Min/max clip candidates and cap fallback; if none meets budget the result records fallback and warning. Review these fields instead of assuming success from mode=auto.

```json
{"mode":"custom","density_cutoff_factor":8,"xc_padding":2,"eigen_search_factor":0.0001,"block_ortho":true,"paw_compensation":"blocks","projector_space":"real-search","projector_radius_factor":1.5,"search_precision":"fp64"}
```

Explicit components override auto. Conflicting environment variables and explicit profiles fail. API/Python mode=off with components fails; CLI component flags with default off are converted to custom by its parser. Auto controls require mode=auto. Density factor and explicit absolute density cutoff must agree. Preserve final requested density and eigen thresholds.

## Preview and evidence

Call Python sparse_dft.scf:preview_sparsity with args [structure, paw_root, output], optional cutoff_ev, accuracy, scf_grid, mesh_mode, smearing_ev, spin_channels, initial_moments, atomic_units, family and controls. It executes the native spectrum probe once and writes density_probe.json plus sparsity_preview.json; it does **not** calculate a total energy. Each reported work ratio is an estimate of mode count scaling (f/16)^1.5. It is not measured wall time or a force error bound.

SCF auto writes density_probe.json, sparsity_profile.json and scf_result.json with provenance.sparsity and **sparsity_ledger**. The raw scf.json preserves native data for checkpoint export. The enriched result retains auto controls, rule IDs, tail budget/source, measured tail, features, fallback and performance counters. Bands/NSCF and DOS reject unresolved auto. Extract the resolved custom profile:

```bash
python3 "$SKILL/scripts/sparsity_workflow.py" frozen-profile \
  --result "$PROJECT/auto/scf_result.json" --out "$PROJECT/frozen-profile.json"
```

Use the returned sparsity dict in public Python run_bands/run_dos kwargs or API params; copy cutoff, units and source sampling, keep the raw sibling scf.json as checkpoint. The frozen custom profile must omit auto_controls **and density_cutoff_factor**; run_bands explicitly rejects S1 factor with a checkpoint, because density support is inherited. Relax auto resolves on the first geometry and freezes the components for all steps.

## Execute matched off/auto/component comparisons

`sparsity_workflow.py prepare --structure POSCAR --settings settings.json --variants variants.json --project NEW_PROJECT` creates real Python calls and declared result artifacts, without launching. Settings need paw_root, explicit cutoff_ev and explicit scf_grid or weighted k points; all common numerical/model controls belong there. Variant file maps run IDs to real profiles and must include `off: {mode: off}`. The bundled examples/sparsity-variants.json covers off, auto, each S1–S6 ; choose deterministic execution separately through parallel.deterministic. Read its values as example inputs, not converged settings.

Execute the generated sparsity-recipe.json with run_recipe.py. Then `sparsity_workflow.py compare --points PROJECT/sparsity-points.json --out NEW_REPORT.json` checks convergence, equal structure/model/cutoff/sampling, finite energy/forces/stress, the 1e-12 Ha eigen residual certificate, and explicit gates. Defaults are 1 meV/atom, 1e-3 eV/Å, 0.01 GPa; override for the actual question. It keeps raw validity and complete ledgers. Failure returns nonzero. It cannot establish accuracy outside the sampled configurations.

S1/S2 alter discretisation; the calibrated tail budget is not a mathematical error bound. S6 has no auto rule and no promised speedup. Off retains the native block/compensation solver defaults; use parallel.deterministic=true or CLI --deterministic for the historical ordering. Do not present deterministic/off as the same switch.
