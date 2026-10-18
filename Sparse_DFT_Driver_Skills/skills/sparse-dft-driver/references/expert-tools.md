# Source tools in Sparse-DFT 1.0.0

Use the full relative name in `tool --name ... --reviewed`. Review source and prerequisites first. Scripts under benchmarks may refer to historical workflow/ files not distributed in this release; this list is not proof they can run without that data.

- `scripts/build_element_table.py`: arguments `--out`
- `scripts/install.py`: arguments `--backend, --build-dir, --cmake-arg, --index-url, --jobs, --no-plots, --no-python, --prefix, --serial, --venv`
- `benchmarks/scripts/anchor_report.py`: arguments ``
- `benchmarks/scripts/check_docs_numbers.py`: arguments `--forbid, --json, --release-root, --root`
- `benchmarks/scripts/compare_vasp_dos.py`: arguments `--out, sparse_dft, vasp`
- `benchmarks/scripts/report_sparsity.py`: arguments `--out, sweeps`
- `benchmarks/scripts/sweep_sparsity.py`: arguments `--only, campaign, out`
- `benchmarks/scripts/validate_scf_reference.py`: arguments `--candidate, --input, --output, --reference`
- `benchmarks/scripts/vasp_anchor.py`: arguments `--out, --ranks, --systems, --threads, --tier, command`
- `packaging/make_manifest.py`: arguments `--check, --layout-only, --platform-tag, --write, bundle`
- `packaging/sparse_dft_gui_entry.py`: arguments ``

Native programs are the live catalogue’s native list; their C++ main defines the actual arguments and return codes. Do not assume all native programs accept --help.
