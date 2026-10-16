# Licence and legal boundaries

[简体中文](LEGAL.zh-CN.md)

This page explains what the Sparse-DFT licence allows, what the package contains, and where third-party terms take over. The binding texts are [`LICENSE`](../LICENSE) and [`NOTICE.md`](../NOTICE.md). This page summarises them and is not legal advice.

## 1. Sparse-DFT licence

Sparse-DFT is released under the Apache License, Version 2.0. The licence covers the original source code, documentation, figures and derived benchmark records in this package.

| You may | Under these conditions |
|---|---|
| use, study and modify the code, for any purpose including commercial use | keep the copyright, licence and NOTICE texts in every copy you distribute (§4(a), §4(c), §4(d)) |
| redistribute it, in source or binary form | mark the files you changed (§4(b)) |
| combine it with other software under other licences | follow the other licences for their parts |
| use any contributor's patent claims that the contributed code necessarily infringes (§3) | the patent licence ends for anyone who sues over patent infringement in the work (§3) |

The software comes without warranty and without liability (§7, §8). The licence grants no trademark rights (§6).

The copyright holder and citation details are in [`NOTICE.md`](../NOTICE.md) and [`CITATION.cff`](../CITATION.cff).

## 2. What this package contains

- Original Sparse-DFT code:
  - C++ in `src/`;
  - Python packages `sparse_dft` and `sparse_dft_gui`;
  - tests, scripts, schemas and documentation.
- Eigen 3.4.0 headers in `third_party/eigen3`, with `Eigen/src` packed unmodified into `Eigen/src.zip`. They are a byte-identical copy of upstream under MPL-2.0, and a few headers keep their own BSD-style or Apache-2.0 notices. The build defines `EIGEN_MPL2_ONLY`, so the compiler rejects Eigen modules that are LGPL-only.
- An element table compiled from published values (IUPAC atomic weights; Cordero et al. covalent radii).
- Small test fixtures:
  - output files written by VASP 6.4.1 for our own test inputs (`tests/fixtures/vasp/`). They hold eigenvalues, densities of states, forces, stresses and the input settings. The only POTCAR-derived content is the dataset title line, such as `PAW_PBE Si 05Jan2001`.
  - synthetic POTCAR-style header templates with invented numbers.

## 3. What this package does not contain

| Not included | What you need to know |
|---|---|
| VASP source code, executables, PAW datasets (POTCAR) | VASP is proprietary software of VASP Software GmbH. Sparse-DFT contains no VASP source code and no POTCAR data. To use VASP PAW datasets, convert your own licensed copy with `sparse-dft paw-convert` and keep the result private. Your VASP licence agreement governs those files. |
| Quantum ESPRESSO, pslibrary or UPF files | Sparse-DFT reads UPF files that you supply. Their own licences apply to them. |
| FFTW | Required at build time, under GPL-2.0-or-later or a commercial licence from MIT. |
| BLAS/LAPACK, OpenMP, MPI, Metal, CUDA | Linked from your system; their own terms apply. |
| Python packages (NumPy, Matplotlib, SciPy, spglib, SeeK-path, PyQt6) | Installed by you from their publishers. |

VASP, POTCAR files and Quantum ESPRESSO appear in two roles only:
- as file formats that Sparse-DFT reads or writes;
- as external programs used for numerical comparison in [`ACCURACY.md`](ACCURACY.md).

## 4. Distributing binaries

The source package ships no compiled binaries. If you build and distribute binaries or bundles, check the components you link.

- **FFTW (GPL-2.0-or-later).** A binary that links FFTW must be distributed under terms compatible with the GPL. Apache-2.0 code may be combined into a GPL-3.0 work but not into a GPL-2.0-only work. Distributing under GPL-3.0, or obtaining a commercial FFTW licence, satisfies this.
- **PyQt6 (GPL-3.0-only or commercial).** A bundle that includes the GUI together with PyQt6 must meet GPL-3.0 or use a commercial PyQt licence. The command-line program and the Python API do not import PyQt6.
- **Qt 6 (LGPL-3.0).** Users must be able to replace the Qt libraries in a GUI bundle.
- **Eigen (MPL-2.0).** If you change any Eigen header, those files stay under MPL-2.0 and their source must stay available.

## 5. PAW datasets and your data

Sparse-DFT ships no PAW datasets. It reads PAW datasets that you obtain from one of two sources. The terms of each source stay with the files.

| Source | How to obtain it | Terms you must follow |
|---|---|---|
| VASP PAW datasets (POTCAR) | Download them from the VASP Portal under your own VASP licence. Then convert them on your machine with `sparse-dft paw-convert`. The converter keeps each POTCAR text exactly as supplied, with its SHA-256, and adds a metadata index for fast lookup. | The POTCAR files are proprietary to VASP Software GmbH and stay under your VASP licence agreement. Do not share, upload or redistribute the archive, and do not bundle it with Sparse-DFT. Check that your agreement covers your intended use. Sparse-DFT cannot check this for you. |
| PAW datasets in UPF v2 format (`pseudo_type="PAW"`), e.g. generated with pslibrary and the Quantum ESPRESSO atomic code `ld1.x`, or downloaded from libraries that publish PAW UPF files | Download or generate them freely. Pass the file paths to Sparse-DFT. | Quantum ESPRESSO and pslibrary are distributed under the GNU GPL, version 2 or (at your option) any later version. Generated or downloaded UPF files have no separate licence statement in those projects. Each pseudopotential library sets its own terms and citation requests. Check the header of each file and the library's own page, and cite the dataset authors. |

Sparse-DFT reads UPF files as data and runs no Quantum ESPRESSO code. Using GPL-licensed tools to make a dataset does not put Sparse-DFT under the GPL.

- Converted archives, results and logs stay on your machine. Calculations and plotting make no network connections.
- Only the Materials Project commands, `sparse-dft mp-fetch` and `mp-fetch-bands`, connect to the network, whether started from the command line or from the GUI command page. They download structures and band data from the Materials Project API with your API key, and the Materials Project terms of use apply to that data. These commands send only the query parameters, the API key and a program user-agent string.

## 6. Independence and trademarks

Sparse-DFT is an independent implementation of published methods (see [`THEORY.md`](THEORY.md)). It is not affiliated with, sponsored by or endorsed by VASP Software GmbH, the Quantum ESPRESSO Foundation, Apple, NVIDIA or any other party named here. The product names of these parties are used only to identify file formats, hardware backends and reference programs. How the source was checked against VASP and Quantum ESPRESSO is described in [`PROVENANCE.md`](PROVENANCE.md).

## 7. Contributions and reports

Contributions are accepted under Apache-2.0 with a Developer Certificate of Origin sign-off (see [`CONTRIBUTING.md`](../CONTRIBUTING.md)). Do not submit code, comments or data copied from VASP, Quantum ESPRESSO or other software whose licence does not allow it. Report licence or provenance concerns as described in [`SECURITY.md`](../SECURITY.md).
