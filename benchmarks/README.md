[README](../README.md) · [Accuracy](../docs/ACCURACY.md) · [Sparsification](../docs/SPARSIFICATION.md) · [Performance](../docs/PERFORMANCE.md)

# Benchmarks

This directory holds the derived data behind every number and figure in the documentation, the scripts that produced it, and the scripts that run new benchmarks. 本目录的中文说明见文末。

## Provenance of the data

The runs were made on 2026-10-14 and 2026-10-15 (file dates) with pre-release builds of this code. The numbers were copied from those runs and not recomputed. Each file in `data/` lists the SHA-256 of every source file it was read from. Absolute paths are replaced by tags:

| tag | source |
|---|---|
| `<anchor_runs>`, `<anchor_engine_runs>` | VASP anchor workspace of `scripts/vasp_anchor.py`, and the Sparse-DFT runs inside it |
| `<sweep_runs>` | workspace of `scripts/sweep_sparsity.py`: fixed-component sweep, fitted auto runs, spectrum probes |
| `<review_sweeps>`, `<review_benchmarks>` | summaries of the sweep and of the timing runs |
| `<bench_runs>` | Si$_{64}$ eigensolver replay runs |
| `<dos_compare>` | output of `scripts/compare_vasp_dos.py` |
| `<band_suite>` | band-path comparison against VASP |
| `<package>` | `sparsity_measured.json`, the copy of the fitted runs shipped with the package |

The VASP inputs, outputs and PAW datasets are not included; they are licensed and cannot be redistributed.

## Files

| file | content | label |
|---|---|---|
| `data/anchor_vasp.json` | ten systems × three tiers against VASP 6.4.1: gates, conditions, every SCF check, band path, DOS, projections, relaxations | measured |
| `data/anchor_vasp_scf.csv`, `data/anchor_vasp_relax.csv` | flat tables of the same SCF and relaxation rows | measured |
| `data/codata_bias.json`, `.csv` | energy shift between the `codata` and `vasp_legacy` constants, with the linear fit | measured; Hartree-ratio column estimated |
| `data/sparsity_sweep.json`, `.csv` | fixed-component sweep against `--sparsity off`, references, fitted auto runs, the slab hold-out failure | measured |
| `data/sparsity_fitted_auto.csv` | fitted auto runs with operator-call ratios | measured |
| `data/sparsity_probe_tails.json`, `.csv` | spectrum-probe tails per system and factor, species multipole weights | computed |
| `data/performance.json` | OpenMP, MPI and Metal timings, and the Si$_{64}$ replay, with result deviations | measured |
| `data/performance_scaling_Si8.csv`, `data/performance_replay_Si64.csv` | flat tables of the Si$_8$ runs and the replay | measured |
| `data/dos_vasp.json`, `.csv` | total and projected DOS $L^1$ differences against VASP | measured |
| `data/bands_vasp.json` | band-path MAE, maximum differences and gaps against VASP | measured |

Every JSON file carries `schema_version: sparse-dft-release-data-1`, a `dataset` name, a `label` and `sources_sha256`. Numbers are copied without rounding; rounding happens only in the documents.

## Scripts

| script | purpose | needs |
|---|---|---|
| `scripts/vasp_anchor.py` | builds the matched VASP and Sparse-DFT anchor workspace | VASP licence, PAW datasets |
| `scripts/anchor_report.py` | compares the anchor runs against the gates | the anchor workspace |
| `scripts/compare_vasp_dos.py` | rebuilds the VASP DOS and compares it with `sparse-dft dos` | VASP outputs |
| `scripts/sweep_sparsity.py`, `scripts/report_sparsity.py` | runs and summarises a sparsity sweep | PAW datasets |
| `scripts/validate_scf_reference.py` | matched spin-unpolarised SCF comparison of energy, eigenvalues, forces and stress | a reference result |
| `scripts/check_docs_numbers.py` | checks the documentation against `data/`: numbers, en/zh parity, links, figures | `data/` and `docs/` only |
| `plot_docs.py` | draws the documentation figures from `data/` alone | `plots` extra |

## Regenerating

The figures need only `data/` and the Python package with the `plots` extra:

```bash
OMP_NUM_THREADS=2 python -I benchmarks/plot_docs.py --data benchmarks/data --out docs/assets --package-root .
python -I benchmarks/scripts/check_docs_numbers.py --root .
```

`plot_docs.py` writes each figure as PNG (600 dpi) and PDF, in English and Simplified Chinese (`<name>.png`, `<name>.zh-CN.png`), and records the values drawn, the input hashes and the result of the layout check in `docs/assets/figures.provenance.json`.

`data/` was extracted from the run workspaces named by the tags above. Those workspaces hold VASP inputs, outputs and PAW datasets, so they and the extraction step are not distributed. To check a number, follow `sources_sha256` in its file; to make new data, run the scripts in the table above with your own licence.

## 中文说明

本目录保存文档中每个数字和每张图背后的派生数据、生成它们的脚本，以及运行新基准的脚本。

- **数据来源。** 这些计算于 2026-10-14 和 2026-10-15（文件日期）使用本程序的预发布构建完成。数字取自这些计算，没有重新计算。`data/` 中每个文件都列出其所读取的每个源文件的 SHA-256，绝对路径以上表中的标签代替。
- **不包含的内容。** VASP 的输入、输出和 PAW 数据集均不包含；它们受许可证约束，不得再分发。
- **文件。** 各文件的内容与标注（实测、计算、估计）见上面的"Files"表。每个 JSON 文件都带有 `schema_version: sparse-dft-release-data-1`、`dataset` 名称、`label` 和 `sources_sha256`。数字原样复制，不做舍入；舍入只在文档中进行。
- **再生。** 作图只需要 `data/` 和带 `plots` 附加依赖的 Python 包，命令见"Regenerating"一节。`plot_docs.py` 以 PNG（600 dpi）和 PDF 输出每张图的英文版和简体中文版，并在 `docs/assets/figures.provenance.json` 中记录绘制的数值、输入哈希和版面检查结果。`data/` 由上表标签所指的计算工作区提取。这些工作区含 VASP 输入、输出和 PAW 数据集，因此工作区和提取步骤都不随包分发。要核对某个数字，请查该文件的 `sources_sha256`；要产生新数据，请用你自己的许可运行上表中的脚本。
