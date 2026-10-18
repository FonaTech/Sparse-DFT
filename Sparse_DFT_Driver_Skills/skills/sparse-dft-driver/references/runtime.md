# Runtime and source identity

Audited source: the user-supplied Sparse-DFT-1.0.0.tar.xz, identified by the distribution validation/source-provenance.json and the live catalogue hashes. No SOURCE_MANIFEST.json is shipped. Archive identity, interface identity, actual executable hash and scientific validity are separate facts.

Python ≥3.11, CMake ≥3.20, C++20 and FFTW3 are required. Eigen headers are bundled (CMake expands Eigen/src.zip). Optional extras in pyproject.toml are plots, symmetry, bz and gui; check that environment before installing. Pinned symmetry dependencies are spglib and seekpath; scipy is used by bz. GUI is PyQt6. The CLI core has no mandatory Python dependency.

```bash
python3 "$SDFT_SOURCE/scripts/install.py" --backend cpu --jobs 4
source "$SDFT_SOURCE/.install/env.sh"
python3 -m sparse_dft backend-info
```

The installer defaults to PyPI. `--no-python` builds/tests/installs native programs only and avoids Python downloads. `--no-plots`, `--prefix`, `--venv`, `--build-dir`, `--serial` and repeated `--cmake-arg=-DNAME=VALUE` are real flags. OpenMP requests require a successful runtime probe; do not silently downgrade. `--backend metal` builds Metal on macOS; runtime spelling is `SDFT_COMPUTE_BACKEND=mps`. CUDA requires its actual toolkit/device. MPI needs `--cmake-arg=-DSDFT_ENABLE_MPI=ON` and an actual launcher.

For source-only calls set SDFT_SCF_EXE explicitly to the built sdft_scf, and optionally SDFT_RELAX_STEP_EXE to sdft_relax_step. SDFT_OPENMP_PROBE selects the runtime probe. Source package resources search bundled bin/<platform>, build including configuration subdirs, and PATH; a separate build-cpu needs an explicit override or the installer's env.sh. The helper also resolves common local build/install directories and passes the chosen binary explicitly.

SDFT_SOURCE is helper configuration; SDFT_PAW_ARCHIVE selects the local archive in resource lookup. Ordinary CLI calculations still take --paw-root. SDFT_ENABLE_PAW_CLOSURE_DIAGNOSTIC=1 enables the full closure diagnostic. Bound OMP_NUM_THREADS; set VECLIB_MAXIMUM_THREADS=1 and OPENBLAS_NUM_THREADS=1 to avoid nested BLAS oversubscription.

PAW source routes:

- Existing indexed archive: paw-inventory / paw-resolve and --paw-root DIR.
- Local POTCAR directory or tar archive: `paw-convert SRC --out NEW_DIR`; retain exact data hashes. Do not bundle potentials in the skill.
- PAW UPF v2 file or directory: direct --paw-root. Only scalar-relativistic PBE PAW is accepted; NC/USPP and unsupported models fail. Inspect with sparse_dft.upf:read_upf_paw / sparse_dft.paw_sources:resolve_paw_sources. There is no UPF-specific Agent API method.

`SDFT_COMPUTE_BACKEND=cpu/mps/cuda` requests a backend. Actual dispatch comes from performance counters, not its spelling. Metal uses compensated FP32 pairs, not native IEEE FP64. The environment catalogue includes compile macros and format identifiers; inspect actual getenv use before setting a tuning variable. SDFT_GPU_FFT, SDFT_GPU_PBE and SDFT_GPU_PAW_CHAIN select implemented GPU paths subject to build and size thresholds. Native failures must remain failures.

The helper works with an explicit source checkout for complete catalog/tool/native discovery. A wheel alone lacks CMake/scripts needed for those routes; use the software CLI/Python directly or obtain the matching checkout. No scheduler, PAW download, MP credential or GPU availability is implied.
