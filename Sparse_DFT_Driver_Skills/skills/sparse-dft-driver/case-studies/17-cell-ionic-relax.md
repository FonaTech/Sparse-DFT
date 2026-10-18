# 17 · 晶胞优化与交替离子—晶胞收敛

类型：原生＋项目调度。这是可复用的驱动案例，未在此文档中填入任何虚构计算结果。复用有来源的项目精度记录；本次运行仍须核查输入、收敛与交付物。

## 目标与选择条件

完成晶胞优化，必要时交替离子与晶胞操作直到同时满足所选停止条件。

## 输入与结构策略

- 初始结构、目标周期条件和 PAW
- force/stress tolerance 与优化预算
- 应变 bounds、固定/释放离子策略

单次 relax.cell 保持分数坐标。联合平衡用明确交替循环：cell→ionic→static gate，每轮真实结构作为下一轮输入。

## 可执行任务阶梯

| 阶段 | 要完成的操作 | 实际入口 | 输出与阶段完成条件 |
|---|---|---|---|
| S01 | 晶胞解析优化 | relax.cell | cell checkpoint 与晶格 |
| S02 | 在新晶胞释放离子 | relax.ionic | 原子位置与力 |
| S03 | 计算共同最终状态 | job.run | 同一最终几何的 force/stress |
| S04 | 按双条件停止并导出 | project-python:alternating loop | 每轮结构、应力/力与停止记录 |

API 通过 `sparse_dft_driver.py api METHOD --params FILE` 调用；CLI 通过 `cli -- COMMAND ...` 调用；Python 通过 `python MODULE:FUNCTION --params FILE` 或项目内 Python 脚本调用；native/tool 使用对应 helper 模式。所有路径与数值控制来自真实项目。`project-python` 表示用已有输出做确定性计算或创建结构的项目脚本，并非新增 Sparse-DFT 原生功能。

## 控制变量与分析方法

应力单位 eV/Å³且拉伸为正。六维对称应变与实际晶格统一；记录 cell-only 与交替联合结果的差别。交替循环有最大轮数和无改善停止条件。

- stress 与 force 两个独立门槛
- 各轮独立目录与初始结构哈希
- cutoff/密度支持及 mesh 的一致策略

## 失败与恢复

- cell-only 结果仍有离子力时执行 ionic 阶段
- 几何状态不匹配时不组合两份残差
- 达到预算而未双收敛报告真实停止状态

保留失败输出与退出码。输入或二进制改变后使用新运行目录；兼容几何重启才复用目录。修复本次调用后从受影响阶段继续，不重复项目已完成的软件精度验证。

## 独立运行和 skills 联动

独立运行时由本 skill 完成上述全部 Sparse-DFT 操作与项目脚本分析。联动时保留 `dft-workflow-orchestrator` 的 claim/stage ID，将每个 Sparse-DFT 阶段记录为 `kind=dft, engine=Sparse-DFT`，分析阶段记录为 `kind=analysis`。交接最终平衡结构、晶胞与共同静态结果，用于 EOS/弹性/能带。

记录源文件 SHA-256、实际参数、单位、公式及 JSON/表格定位。项目已有精度验证作为外部验证依据保留；原始输出中的历史状态字段原样保存。需要论文时交接给 Academic_Writing_Skills，原始结果不编造 DOI 或页码。

## 完成定义

所请求 cell-only 或双条件交替优化完成，算法路线与实际停止条件清楚。

## 实施入口

```bash
python3 /absolute/sparse-dft-driver/scripts/case_studies.py scaffold \
  --case 17 --project /actual/project --system "实际体系与状态"
```

生成详细案例说明、项目控制平面与 `workflow/experiment_matrix.csv`。补齐真实路径、作业参数与资源限制后按任务阶梯调用驱动。批量执行使用 [run_recipe.py](../scripts/run_recipe.py)，请求格式见 [recipe-runner.md](../references/recipe-runner.md)。
