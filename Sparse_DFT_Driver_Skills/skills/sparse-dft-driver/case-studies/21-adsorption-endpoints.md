# 21 · 闭壳层吸附与解离端点能

类型：原生＋项目差分。这是可复用的驱动案例，未在此文档中填入任何虚构计算结果。复用有来源的项目精度记录；本次运行仍须核查输入、收敛与交付物。

## 目标与选择条件

比较现有 PBE 非自旋模型适用的吸附位点、分子/解离端点和几何变化。

## 输入与结构策略

- 真实非磁性 slab、终止面和覆盖度
- 闭壳层吸附物及一致参照态
- 候选位点、约束、周期盒和参数

从已定 slab 生成 clean、adsorbed 和 reference 三类结构。真空中的分子参照也用用户定义的三维周期盒；不同位点保持同覆盖度和相同能量参照。

## 可执行任务阶梯

| 阶段 | 要完成的操作 | 实际入口 | 输出与阶段完成条件 |
|---|---|---|---|
| S01 | 构建位点和参照集合 | project-python:structure enumeration | 标识明确的结构与计量表 |
| S02 | 固定下层/优化活动原子 | relax.ionic | 各实际端点几何 |
| S03 | 统一静态精修 | job.run | 所有参照与端点 E |
| S04 | 计算吸附/端点差并排序 | project-python:endpoint arithmetic | Eads 与端点能差/几何表 |

API 通过 `sparse_dft_driver.py api METHOD --params FILE` 调用；CLI 通过 `cli -- COMMAND ...` 调用；Python 通过 `python MODULE:FUNCTION --params FILE` 或项目内 Python 脚本调用；native/tool 使用对应 helper 模式。所有路径与数值控制来自真实项目。`project-python` 表示用已有输出做确定性计算或创建结构的项目脚本，并非新增 Sparse-DFT 原生功能。

## 控制变量与分析方法

Eads=E(slab+adsorbate)−E(clean slab)−E(reference adsorbate)，计量系数按真实反应给定。解离端点差是热力学排序，不由此输出不存在的 NEB 势垒或速率。 能量字段保持一致：internal_energy_ev 为 E，total_energy_ev 在有限展宽时为 F=E−TS。

- 同一覆盖度与计量基准
- slab/vacuum/constraints 一致
- 只使用本模型适用的实际闭壳层状态

## 失败与恢复

- 优化到另一 motif 时记录真实归属
- 参照计算缺失则补完整参照
- 需要未实现自旋/电势物理的输入不进入此闭壳层案例

保留失败输出与退出码。输入或二进制改变后使用新运行目录；兼容几何重启才复用目录。修复本次调用后从受影响阶段继续，不重复项目已完成的软件精度验证。

## 独立运行和 skills 联动

独立运行时由本 skill 完成上述全部 Sparse-DFT 操作与项目脚本分析。联动时保留 `dft-workflow-orchestrator` 的 claim/stage ID，将每个 Sparse-DFT 阶段记录为 `kind=dft, engine=Sparse-DFT`，分析阶段记录为 `kind=analysis`。交接端点表和归一化公式，对应 DFT_Skills 吸附/反应端点分支。

记录源文件 SHA-256、实际参数、单位、公式及 JSON/表格定位。项目已有精度验证作为外部验证依据保留；原始输出中的历史状态字段原样保存。需要论文时交接给 Academic_Writing_Skills，原始结果不编造 DOI 或页码。

## 完成定义

请求位点与端点的真实能量和几何排序完成，参照/计量完整。

## 实施入口

```bash
python3 /absolute/sparse-dft-driver/scripts/case_studies.py scaffold \
  --case 21 --project /actual/project --system "实际体系与状态"
```

生成详细案例说明、项目控制平面与 `workflow/experiment_matrix.csv`。补齐真实路径、作业参数与资源限制后按任务阶梯调用驱动。批量执行使用 [run_recipe.py](../scripts/run_recipe.py)，请求格式见 [recipe-runner.md](../references/recipe-runner.md)。
