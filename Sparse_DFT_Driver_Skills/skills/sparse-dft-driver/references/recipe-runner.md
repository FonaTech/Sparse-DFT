# Executable case recipes

The case library specifies actual entrypoints and a concrete task ladder. `run_recipe.py` executes a reviewed linear recipe with explicit dependencies, keeps per-call logs, stops at failure and hashes declared file outputs. It does not invent numerical controls, launch unknown physics or create scientific results from a scaffold.

```bash
python3 /absolute/sparse-dft-driver/scripts/run_recipe.py \
  --source /actual/Sparse-DFT-source --project /actual/project --recipe /actual/project/recipe.json
```

Example (replace paths and the example grid with the actual project inputs):

```json
{
  "schema": "sparse-dft-recipe-1",
  "accuracy_validation_basis": "user-confirmed existing Sparse-DFT accuracy validation",
  "environment": {
    "SDFT_SCF_EXE": "/actual/build/sdft_scf",
    "SDFT_ENABLE_PAW_CLOSURE_DIAGNOSTIC": "1",
    "OMP_NUM_THREADS": "2"
  },
  "steps": [
    {
      "id": "scf",
      "mode": "api",
      "method": "job.run",
      "params": {
        "structure": "/actual/project/structures/POSCAR",
        "paw_root": "/actual/potpaw_PBE.54-json",
        "output": "runs/scf-001",
        "scf_grid": [2, 2, 2]
      },
      "outputs": ["runs/scf-001/scf.json", "runs/scf-001/scf_result.json"]
    },
    {
      "id": "analysis",
      "depends_on": ["scf"],
      "mode": "project-python",
      "script": "analysis/extract_scf.py",
      "args": ["runs/scf-001/scf_result.json", "analysis/scf-table.csv"],
      "outputs": ["analysis/scf-table.csv"]
    }
  ]
}
```

`project-python` runs an existing project-local analysis/structure-generation script. Such scripts use actual Sparse-DFT functions or read raw outputs; the runner does not fabricate them. Scientific fitting, structure enumeration, endpoint arithmetic and plotting can be implemented here without companion skills. Keep code and formulas with the output provenance.

Other modes:

| Mode | Required step fields |
|---|---|
| `api` | `method`, `params` object (actual dispatch keys) |
| `python` | `call`, `params` with `args`/`kwargs` |
| `cli` | `args` array (Sparse-DFT argv, without shell syntax) |
| `native` | `name`, optional `bin_dir`, `args` |
| `tool` | `name`, `reviewed=true`, `args`; read source/default paths first |
| `project-python` | project-local `script`, `args` |

All steps need unique simple IDs. `depends_on` may refer only to earlier completed steps. Environment values are inherited then overridden per step. Output paths resolve against the project; declare real files, not directories. Parameters containing credentials are transient with restricted permissions; prefer supported MP credential environment variables.

To resume, supply the printed `state.json` through `--resume-state`. Successful steps are skipped only when declared files still match their saved hashes, and recipe/source/configured binary/project identity agrees. A completed step with no declared files cannot be safely skipped. Failed steps run only on an explicit resume; intentional compatible relaxation reuse additionally needs the step's `resume=true`. Changed method, binary or result files require a new state/results directory. The source backend remains responsible for SCF/geometry convergence; recipe completion means all declared calls and output-presence gates completed.

## Ready-to-run EOS generator and analysis

Case 19 has a stdlib-only structure/recipe generator and local quadratic estimator. Supply a JSON array of actual chosen volume factors and SCF settings with `paw_root` and numerical controls (omit `structure`/`output`, which are generated per point).

```bash
python3 /absolute/sparse-dft-driver/scripts/eos_workflow.py prepare \
  --source /actual/Sparse-DFT-source --structure /actual/POSCAR \
  --volume-factors /actual/volume-factors.json --settings /actual/scf-settings.json \
  --project /actual/eos-project
python3 /absolute/sparse-dft-driver/scripts/run_recipe.py \
  --source /actual/Sparse-DFT-source --project /actual/eos-project \
  --recipe /actual/eos-project/eos-recipe.json
python3 /absolute/sparse-dft-driver/scripts/eos_workflow.py analyze \
  --points /actual/eos-project/volume_points.json --project /actual/eos-project \
  --energy-field total_energy_ev --out /actual/eos-project/analysis/eos-v1
```

`prepare --relax-ions` produces per-volume ionic relaxation calls. Analysis follows each last accepted SCF artifact and its recorded SHA-256. `total_energy_ev` is F=E−TS at finite smearing; `internal_energy_ev` is E. Choose the same field for all points; no native `free_energy_ev` exists. The quadratic local fit reports its formula, interval, residual, V0 and B0; a nonpositive curvature or out-of-interval minimum is rejected. More elaborate EOS models remain project-script analyses with explicit formulas. No EOS results are produced by prepare alone.
