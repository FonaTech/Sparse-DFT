Sparse-DFT
Copyright 2026 Yufeng Zhan

This product is licensed under the Apache License, Version 2.0 (see LICENSE).
It includes software developed by third parties, listed below with their own
terms. Those terms continue to apply to those components.

# Copyright, licence and third-party boundaries / 版权、许可与第三方边界

## Sparse-DFT

Copyright 2026 Yufeng Zhan. This covers the original Sparse-DFT source code, documentation, figures and derived benchmark records. These components are released under the Apache License, Version 2.0.

This notice does not certify who owns material supplied by others, and it does not replace any dependency's licence. Apache-2.0 §3 grants a patent licence only for patent claims licensable by a Contributor.

Sparse-DFT 原创源代码、文档、图和派生基准记录的版权归 Yufeng Zhan 所有（Copyright 2026），以 Apache License 2.0 发布。

本声明不替外部材料确认权属，也不取代任何依赖的许可。Apache-2.0 第 3 条只授予贡献者可许可的专利权利要求。

## Included in this package / 本包包含的第三方组件

| Component / 组件 | Rights holder / 权利方 | Terms / 条款 | Use here / 用途 |
| --- | --- | --- | --- |
| Eigen 3.4.0 headers (`third_party/eigen3/Eigen`) | Eigen contributors | MPL-2.0 (`COPYING.MPL2`). A few headers keep their own BSD-style notices (Intel MKL/LAPACKE bindings, `Half.h` conversion routines by F. Giesen) or Apache-2.0 (`BFloat16.h`, TensorFlow Authors); `COPYING.BSD`, `COPYING.MINPACK` and `COPYING.README` are included. | Unmodified copy of upstream `Eigen/` (byte-identical, see `docs/PROVENANCE.md`). Sparse-DFT includes only `<Eigen/Dense>` and builds with `EIGEN_MPL2_ONLY`, which makes the compiler reject Eigen's LGPL-only modules. MPL-2.0 file-level copyleft applies to these files; changes to them must stay under MPL-2.0. |
| Element data (`sparse_dft/data/elements.json`) | Facts from IUPAC (atomic weights 2013, doi:10.1515/pac-2015-0305), Cordero et al. 2008 (covalent radii, doi:10.1039/B801115J) and webelements.com moments, as tabulated in ASE | Factual values; ASE (LGPL-2.1-or-later) was used once to generate the table and is not distributed or imported | Structure classification and default magnetic moments |

## Not included; installed or linked by the user / 不包含，由用户安装或链接

| Component / 组件 | Rights holder / 权利方 | Terms / 条款 | Boundary / 边界 |
| --- | --- | --- | --- |
| VASP source, executables, PAW datasets (POTCAR) | VASP Software GmbH and licensors | Proprietary; the user's licence agreement controls | **None included.** Sparse-DFT reads a PAW archive that each user builds from their own licensed POTCAR files. VASP is only an external reference for comparison. |
| Quantum ESPRESSO, pslibrary, UPF files | QE Foundation and contributors; pslibrary authors | GPL-2.0-or-later (QE and pslibrary); each pseudopotential library sets its own terms | **None included.** UPF files are read as user data; no QE code is run or linked. See `docs/LEGAL.md` §5. |
| FFTW 3 | M. Frigo, S. G. Johnson, MIT | GPL-2.0-or-later, or a commercial licence from MIT | Linked at build time. A binary that links GPL FFTW must be distributed under GPL-compatible terms. Apache-2.0 is compatible with GPL-3.0 but not with GPL-2.0-only. This package ships no binaries. |
| BLAS/LAPACK (Apple Accelerate, OpenBLAS, reference LAPACK) | Apple Inc.; OpenBLAS contributors; LAPACK authors | Apple SDK terms; BSD-3-Clause; BSD-style | Linked locally |
| OpenMP runtime (libomp / libgomp) | LLVM contributors; FSF | Apache-2.0 WITH LLVM-exception; GPL-3.0 WITH GCC-exception-3.1 | Linked locally |
| MPI (Open MPI, MPICH, MS-MPI) | respective contributors | BSD-3-Clause-style; MIT-style (MS-MPI) | Optional, linked locally |
| Apple Metal and Foundation frameworks | Apple Inc. | Apple developer and SDK agreements | Optional backend; calls only, no SDK files included |
| NVIDIA CUDA toolkit, cuBLAS, cuFFT | NVIDIA Corporation | NVIDIA CUDA EULA | Optional backend; no NVIDIA files included |
| CMake, C++ compiler | Kitware; LLVM/GCC contributors | BSD-3-Clause; Apache-2.0 WITH LLVM-exception / GPL | Build tools only |
| Python | Python Software Foundation | PSF-2.0 | Interpreter |
| NumPy 2.4 | NumPy developers | BSD-3-Clause (with 0BSD, MIT, Zlib, CC0 parts) | Optional (plots) |
| Matplotlib 3.11 | Matplotlib development team | Matplotlib licence (PSF-based) | Optional (plots) |
| SciPy 1.17.1 | SciPy developers | BSD-3-Clause | Optional (Brillouin-zone inset) |
| spglib 2.8.0 | spglib developers | BSD-3-Clause | Optional (symmetry) |
| SeeK-path 2.2.2 | SeeK-path authors | MIT | Optional (band paths) |
| PyQt6 6.11 | Riverbank Computing | GPL-3.0-only, or a Riverbank commercial licence | Optional GUI. Any distribution that combines Sparse-DFT with PyQt6 (for example a frozen GUI bundle) must meet GPL-3.0 or use a commercial PyQt licence. Apache-2.0 code may be combined into a GPL-3.0 work. |
| Qt 6 (PyQt6-Qt6 wheels) | The Qt Company | LGPL-3.0 (or commercial) | Optional GUI runtime |

Versions are the tested set. Licences were read from the installed package metadata and the upstream licence files on 2026-10-15.

表中版本为测试所用版本。许可信息于 2026-10-15 读自已安装包的元数据和上游许可文件。

## Trademarks / 商标

VASP is a trademark of VASP Software GmbH. Quantum ESPRESSO is a trademark of the Quantum ESPRESSO Foundation. Apple and Metal are trademarks of Apple Inc. NVIDIA and CUDA are trademarks of NVIDIA Corporation. These names only identify compatible formats and comparison programs.

Sparse-DFT is independent of these companies and foundations. No sponsorship, partnership or endorsement is claimed. This licence grants no trademark rights (Apache-2.0 §6).

VASP、Quantum ESPRESSO、Apple、Metal、NVIDIA、CUDA 均为各自权利人的商标，本项目只用这些名称标识兼容格式和对比程序。Sparse-DFT 独立于上述公司和基金会，不声称获得任何赞助、合作或背书。本许可不授予任何商标权（Apache-2.0 第 6 条）。
