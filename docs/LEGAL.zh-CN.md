# 许可与法律边界

[English](LEGAL.md)

本页说明三件事：Sparse-DFT 的许可允许做什么，本包包含哪些内容，以及从哪里起由第三方条款约束。有约束力的文本是 [`LICENSE`](../LICENSE) 和 [`NOTICE.md`](../NOTICE.md)。本页只是概要，不构成法律意见。

## 1. Sparse-DFT 的许可

Sparse-DFT 以 Apache License 2.0 发布。许可覆盖本包中的原创源代码、文档、图和派生基准记录。

| 你可以 | 条件 |
|---|---|
| 使用、研究和修改代码，用于任何目的，包括商业用途 | 分发的每一份副本都保留版权、许可和 NOTICE 文本（第 4(a)、4(c)、4(d) 条） |
| 以源代码或二进制形式再分发 | 标明你修改过的文件（第 4(b) 条） |
| 与其他许可的软件组合 | 其他部分遵循其各自的许可 |
| 使用贡献者对其贡献代码必然涉及的专利权利要求（第 3 条） | 就本作品提起专利侵权诉讼者，其专利许可终止（第 3 条） |

软件不附带任何担保，作者不承担责任（第 7、8 条）。本许可不授予商标权（第 6 条）。

版权人与引用信息见 [`NOTICE.md`](../NOTICE.md) 和 [`CITATION.cff`](../CITATION.cff)。

## 2. 本包包含的内容

- **Sparse-DFT 原创代码**：
  - `src/` 中的 C++ 代码；
  - Python 包 `sparse_dft` 和 `sparse_dft_gui`；
  - 测试、脚本、模式文件和文档。
- **Eigen 3.4.0 头文件**（`third_party/eigen3`，其中 `Eigen/src` 未经修改打包为 `Eigen/src.zip`）：与上游逐字节一致，许可为 MPL-2.0。少数头文件保留各自的 BSD 类或 Apache-2.0 声明。构建时定义 `EIGEN_MPL2_ONLY`，编译器会拒绝仅限 LGPL 的 Eigen 模块。
- **元素数据表**：整理自公开发表的数值，包括 IUPAC 原子量和 Cordero 等人的共价半径。
- **小型测试数据**：
  - `tests/fixtures/vasp/`：VASP 6.4.1 用我们自己的测试输入算出的输出文件，含本征值、态密度、力、应力和输入设置。其中来自 POTCAR 的内容只有数据集标题行，如 `PAW_PBE Si 05Jan2001`。
  - 合成的 POTCAR 风格表头模板，其中数值为虚构。

## 3. 本包不包含的内容

| 不包含 | 需要知道的事 |
|---|---|
| VASP 源代码、可执行文件、PAW 数据集（POTCAR） | VASP 是 VASP Software GmbH 的专有软件。Sparse-DFT 不含任何 VASP 源代码，也不含 POTCAR 数据。要使用 VASP PAW 数据集，请用 `sparse-dft paw-convert` 转换你自己持有许可的副本，转换结果仅限私用。这些文件受你的 VASP 许可协议约束。 |
| Quantum ESPRESSO、pslibrary 或 UPF 文件 | Sparse-DFT 读取你提供的 UPF 文件，这些文件适用其自身许可。 |
| FFTW | 构建时必需，许可为 GPL-2.0-or-later，或向 MIT 购买的商业许可。 |
| BLAS/LAPACK、OpenMP、MPI、Metal、CUDA | 从你的系统链接，适用各自条款。 |
| Python 包（NumPy、Matplotlib、SciPy、spglib、SeeK-path、PyQt6） | 由你从发布者处安装。 |

VASP、POTCAR 文件和 Quantum ESPRESSO 只在两种场合出现：
- 作为 Sparse-DFT 读写的文件格式；
- 作为 [`ACCURACY.zh-CN.md`](ACCURACY.zh-CN.md) 中数值对比所用的外部程序。

## 4. 分发二进制

源代码包不含编译好的二进制。如果你构建并分发二进制或打包程序，请检查所链接的组件：

- **FFTW（GPL-2.0-or-later）**：链接 FFTW 的二进制必须以与 GPL 兼容的条款分发。Apache-2.0 代码可以并入 GPL-3.0 作品，但不能并入 GPL-2.0-only 作品。以 GPL-3.0 分发，或取得 FFTW 商业许可，都可满足要求。
- **PyQt6（GPL-3.0-only 或商业许可）**：如果打包程序同时包含 GUI 和 PyQt6，必须满足 GPL-3.0，或使用 PyQt 商业许可。命令行程序和 Python API 不导入 PyQt6。
- **Qt 6（LGPL-3.0）**：GUI 打包程序必须允许用户替换其中的 Qt 库。
- **Eigen（MPL-2.0）**：如果你修改了任何 Eigen 头文件，这些文件仍受 MPL-2.0 约束，并须提供其源代码。

## 5. PAW 数据集与你的数据

Sparse-DFT 不附带任何 PAW 数据集，它读取你从以下两种来源获得的数据集。各来源的条款随文件本身适用。

| 来源 | 获取方式 | 须遵守的条款 |
|---|---|---|
| VASP PAW 数据集（POTCAR） | 凭你本人的 VASP 许可从 VASP Portal 下载，再在本机用 `sparse-dft paw-convert` 转换。转换工具原样保存每个 POTCAR 文本及其 SHA-256，另加元数据索引以便快速查找。 | POTCAR 文件是 VASP Software GmbH 的专有材料，仍受你的 VASP 许可协议约束。不要共享、上传或再分发归档，也不要与 Sparse-DFT 一起打包。请自行确认你的许可协议涵盖你的用途，Sparse-DFT 无法替你核实。 |
| UPF v2 格式的 PAW 数据集（`pseudo_type="PAW"`），例如用 pslibrary 和 Quantum ESPRESSO 原子程序 `ld1.x` 生成，或从公开发布 PAW UPF 文件的赝势库下载 | 可自由下载或生成，把文件路径交给 Sparse-DFT。 | Quantum ESPRESSO 和 pslibrary 以 GNU GPL 第 2 版或（由你选择）任何更新版本发布。这两个项目没有为生成或下载的 UPF 文件单独声明许可。各赝势库有自己的条款和引用要求，请查看每个文件的表头和该库的页面，并引用数据集作者。 |

Sparse-DFT 把 UPF 文件当作数据读取，不运行任何 Quantum ESPRESSO 代码。用 GPL 许可的工具制作数据集，不会使 Sparse-DFT 本身受 GPL 约束。

- 转换后的归档、结果和日志都留在你的机器上。计算和绘图不发起任何网络连接。
- 只有 Materials Project 相关命令 `sparse-dft mp-fetch` 和 `mp-fetch-bands` 会联网，无论从命令行还是从 GUI 命令页启动。它们用你的 API key 从 Materials Project API 下载结构和能带数据，这些数据适用 Materials Project 的使用条款。请求中只发送查询参数、API key 和程序的 user-agent 字符串。

## 6. 独立性与商标

Sparse-DFT 是对已发表方法的独立实现（见 [`THEORY.zh-CN.md`](THEORY.zh-CN.md)）。它与 VASP Software GmbH、Quantum ESPRESSO 基金会、Apple、NVIDIA 及本页提到的任何其他方均无隶属关系，也未获得其赞助或背书。这些名称只用来标识文件格式、硬件后端和参考程序。源代码如何与 VASP 和 Quantum ESPRESSO 比对，见 [`PROVENANCE.zh-CN.md`](PROVENANCE.zh-CN.md)。

## 7. 贡献与报告

贡献以 Apache-2.0 接受，每个提交须附 Developer Certificate of Origin 签署（见 [`CONTRIBUTING.md`](../CONTRIBUTING.md)）。请勿提交从 VASP、Quantum ESPRESSO 或其他许可不允许复制的软件中拷贝的代码、注释或数据。许可或来源方面的问题，请按 [`SECURITY.md`](../SECURITY.md) 所述报告。
