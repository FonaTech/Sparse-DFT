# Contributing / 贡献

Sparse-DFT accepts issues and pull requests.

**Licence of contributions.** Unless you state otherwise, a contribution you submit is licensed under the Apache License 2.0 (Section 5). Add a `Signed-off-by:` line to each commit (`git commit -s`). By doing so you confirm that you have the right to submit the work.


**Code you must not contribute:**
- VASP source code, or text derived from it;
- POTCAR files or other licensed pseudopotential data;
- code copied from GPL-licensed programs such as Quantum ESPRESSO;
- any material whose licence you cannot state.

Describe an algorithm in your own words and cite the paper or manual it comes from; do not translate code.

**Numerical changes.** A change that affects results must show the following on at least one system from `benchmarks/data`:
- `ctest` and `python -m unittest discover -s tests` pass;
- the deviation from the previous release.

The acceptance gates are in `docs/ACCURACY.md`.

---

**贡献的许可。** 除非另有声明，你提交的贡献按 Apache License 2.0 第 5 条授权。每个提交请加 `Signed-off-by:` 行（`git commit -s`），以确认你有权提交这份工作。


**不得提交的内容：**
- VASP 源代码或由其派生的文字；
- POTCAR 或其他受许可约束的赝势数据；
- 从 Quantum ESPRESSO 等 GPL 程序复制的代码；
- 无法说明许可来源的材料。

描述算法请用自己的话，并注明出处论文或手册；不要翻译代码。

**影响数值结果的改动**需要在 `benchmarks/data` 中至少一个体系上给出：
- `ctest` 和 `python -m unittest discover -s tests` 通过；
- 与上一版本的偏差。

验收门槛见 `docs/ACCURACY.md`。
